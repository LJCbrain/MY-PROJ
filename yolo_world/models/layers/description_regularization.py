"""Fixed description relations and class-prototype ranking for DIOR 3.2.

This module is opt-in. No detector scores, visual labels, or test images are
used to construct the teacher. The loss is a DescReg-inspired adaptation to
normalized, image-conditioned CLIP prototypes, not a full DescReg detector.
"""

import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def read_description_manifest(path, expected_classes=None):
    """Return class names, descriptions and the exact source-file checksum."""
    source = Path(path).read_bytes()
    manifest = json.loads(source.decode('utf-8-sig'))
    if manifest.get('schema_version') != 1:
        raise ValueError('Description manifest must have schema_version=1.')
    entries = manifest.get('classes', [])
    if not isinstance(entries, list) or not entries:
        raise ValueError('Description manifest needs a non-empty classes list.')
    names, descriptions = [], []
    for entry in entries:
        name, description = entry.get('name'), entry.get('description')
        if not isinstance(name, str) or not name.strip():
            raise ValueError('Each class needs a non-empty name.')
        if not isinstance(description, str) or not description.strip():
            raise ValueError('Each class needs a non-empty description.')
        names.append(name)
        descriptions.append(description)
    if len(set(names)) != len(names):
        raise ValueError('Duplicate class names in description manifest.')
    if expected_classes is not None:
        expected_classes = tuple(expected_classes)
        if len(set(expected_classes)) != len(expected_classes):
            raise ValueError('Duplicate class names in anchor vocabulary.')
        if set(names) != set(expected_classes):
            raise ValueError('Description classes do not match anchor classes.')
        lookup = dict(zip(names, descriptions))
        names = list(expected_classes)
        descriptions = [lookup[name] for name in names]
    return names, descriptions, hashlib.sha256(source).hexdigest()


def validate_similarity_matrix(similarity, num_classes):
    matrix = torch.as_tensor(similarity).detach().to(
        device='cpu', dtype=torch.float32).clone()
    if matrix.shape != (num_classes, num_classes):
        raise ValueError('Description similarity must have shape [C, C].')
    if not torch.isfinite(matrix).all():
        raise ValueError('Description similarity contains non-finite values.')
    if (matrix.abs() > 1.0001).any():
        raise ValueError('Description similarity must be raw cosine in [-1, 1].')
    if not torch.allclose(matrix, matrix.T, atol=1e-4, rtol=0):
        raise ValueError('Raw cosine similarity must be symmetric.')
    if not torch.allclose(matrix.diag(), torch.ones(num_classes),
                          atol=1e-4, rtol=0):
        raise ValueError('Raw cosine similarity must have a unit diagonal.')
    return matrix.clamp(-1, 1)


def similarity_from_embeddings(embeddings):
    embeddings = torch.as_tensor(embeddings).detach().float()
    if embeddings.ndim != 2 or not torch.isfinite(embeddings).all():
        raise ValueError('Description embeddings must be finite [C, D] values.')
    if (embeddings.norm(dim=-1) <= 1e-6).any():
        raise ValueError('Description embeddings cannot contain zero vectors.')
    normalized = F.normalize(embeddings, dim=-1)
    return validate_similarity_matrix(normalized @ normalized.T,
                                      len(embeddings))


def load_description_matrix(path, expected_classes, source_sha256=None):
    """Load a locally generated artifact and reorder BOTH axes by class name."""
    payload = torch.load(path, map_location='cpu')
    if not isinstance(payload, dict) or payload.get('schema_version') != 1:
        raise ValueError('Expected a schema_version=1 description matrix dict.')
    names = payload.get('class_names', [])
    if (not isinstance(names, (list, tuple)) or
            any(not isinstance(name, str) for name in names) or
            len(set(names)) != len(names) or
            set(names) != set(expected_classes)):
        raise ValueError('Matrix class names do not match anchor classes.')
    if (source_sha256 is not None and
            payload.get('source_sha256') != source_sha256):
        raise ValueError('Description matrix is stale: source checksum differs.')
    matrix = validate_similarity_matrix(payload.get('cosine_similarity'),
                                        len(names))
    indices = torch.tensor([names.index(name) for name in expected_classes])
    return matrix[indices][:, indices]


def description_triplet_loss(prompted_embeddings, similarity,
                             temperature=0.03, topk=3, margin_scale=1.0):
    """Mean adaptive-margin triplets, with a detached description teacher.

    Deterministically pair the top-k related neighbours with bottom-k unrelated
    neighbours. This avoids extra RNG consumption changing the 2.17 training
    pipeline. Self-pairs, overlapping groups and equal-relation pairs are
    excluded. Similar classes are prototype neighbours, NOT extra GT labels.
    """
    if prompted_embeddings.ndim != 3:
        raise ValueError('Prompted embeddings must have shape [B, C, D].')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive.')
    if not isinstance(topk, int) or topk <= 0:
        raise ValueError('topk must be a positive integer.')
    if not math.isfinite(margin_scale) or margin_scale < 0:
        raise ValueError('margin_scale must be finite and non-negative.')
    classes = prompted_embeddings.shape[1]
    if tuple(similarity.shape) != (classes, classes):
        raise ValueError('Teacher and prompted class counts do not match.')
    if classes < 3:
        return prompted_embeddings.float().sum() * 0
    with torch.autocast(device_type=prompted_embeddings.device.type,
                        enabled=False):
        teacher = similarity.detach().to(prompted_embeddings.device).float()
        diagonal = torch.eye(classes, device=teacher.device, dtype=torch.bool)
        relations = F.softmax(
            (teacher / temperature).masked_fill(diagonal, -torch.inf), dim=-1)
        neighbours = min(topk, (classes - 1) // 2)
        high = relations.masked_fill(diagonal, -torch.inf).topk(
            neighbours, dim=-1).indices
        low = relations.masked_fill(diagonal, torch.inf).topk(
            neighbours, dim=-1, largest=False).indices
        anchors = torch.arange(classes, device=teacher.device)[:, None, None]
        anchors = anchors.expand(-1, neighbours, neighbours).reshape(-1)
        high = high[:, :, None].expand(-1, -1, neighbours).reshape(-1)
        low = low[:, None, :].expand(-1, neighbours, -1).reshape(-1)
        gaps = relations[anchors, high] - relations[anchors, low]
        valid = (high != low) & (gaps > 1e-8)
        if not valid.any():
            return prompted_embeddings.float().sum() * 0
        anchors, high, low = anchors[valid], high[valid], low[valid]
        vectors = F.normalize(prompted_embeddings.float(), dim=-1)
        close = torch.linalg.vector_norm(
            vectors[:, anchors] - vectors[:, high], dim=-1)
        far = torch.linalg.vector_norm(
            vectors[:, anchors] - vectors[:, low], dim=-1)
        return (close - far + margin_scale * gaps[valid]).clamp_min(0).mean()
