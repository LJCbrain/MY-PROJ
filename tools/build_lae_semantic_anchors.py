"""Build the fixed, diverse 96-text semantic-anchor table for LAE-1M.

The selected prompts are deterministic.  Each source receives its configured
quota, duplicate normalized prompts are excluded globally, and prompts within
one source are chosen by greedy farthest-point sampling over frozen local CLIP
text embeddings.  The script writes both the exact JSON list consumed by the
training config and the matching embedding table checked by the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModelWithProjection


SEED = 20260714
MODEL_NAME = 'pretrained_models/clip-vit-large-patch14-336'
OUTPUT_DIR = Path('data/LAE-1M/semantic_anchors')


@dataclass(frozen=True)
class AnchorSource:
    name: str
    text_path: str
    quota: int


SOURCES: Sequence[AnchorSource] = (
    AnchorSource('AID', 'LAE-COD/AID/AID_categories.json', 20),
    AnchorSource('EMS', 'LAE-COD/EMS/EMS_categories.json', 21),
    AnchorSource('NWPU-RESISC45',
                 'LAE-COD/NWPU-RESISC45/NWPU-RESISC45_categories.json', 22),
    AnchorSource('SLM', 'LAE-COD/SLM/SLM_categories.json', 4),
    AnchorSource('DIOR', 'LAE-FOD/DIOR/DIOR_categories.json', 4),
    AnchorSource('DOTAv2', 'LAE-FOD/DOTAv2/DOTAv2_categories.json', 4),
    AnchorSource('FAIR1M', 'LAE-FOD/FAIR1M/FAIR1M_categories.json', 4),
    AnchorSource('HRSC2016', 'LAE-FOD/HRSC2016/HRSC2016_categories.json', 1),
    AnchorSource('NWPU-VHR-10',
                 'LAE-FOD/NWPU VHR-10/NWPU VHR-10_categories.json', 4),
    AnchorSource('Power-Plant',
                 'LAE-FOD/Power-Plant/Power-Plant_categories.json', 4),
    AnchorSource('RSOD', 'LAE-FOD/RSOD/RSOD_categories.json', 4),
    AnchorSource('xView', 'LAE-FOD/xview/xview_categories.json', 4),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='data/LAE-1M')
    parser.add_argument('--model-name', default=MODEL_NAME)
    parser.add_argument('--output-dir', default=str(OUTPUT_DIR))
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument(
        '--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text).strip()
                  .casefold())


def read_texts(path: Path) -> List[str]:
    with path.open('r', encoding='utf-8') as file:
        entries = json.load(file)
    if not isinstance(entries, list):
        raise TypeError(f'Expected a JSON list: {path}')

    texts = []
    for index, entry in enumerate(entries):
        if isinstance(entry, str):
            text = entry
        elif (isinstance(entry, list) and len(entry) == 1
              and isinstance(entry[0], str)):
            text = entry[0]
        else:
            raise TypeError(
                f'Invalid class-text entry {index} in {path}: {entry!r}')
        if not text:
            raise ValueError(f'Empty class-text entry {index} in {path}.')
        texts.append(text)
    return texts


def source_candidates(data_root: Path, source: AnchorSource) -> List[str]:
    """Return source-local normalized-unique text prompts in stable order."""
    seen = set()
    candidates = []
    for text in read_texts(data_root / source.text_path):
        key = normalize(text)
        if key not in seen:
            seen.add(key)
            candidates.append(text)
    if len(candidates) < source.quota:
        raise ValueError(
            f'{source.name} has only {len(candidates)} unique candidate texts '
            f'for quota {source.quota}.')
    return candidates


def encode_texts(texts: Sequence[str], tokenizer, model, device: torch.device,
                 batch_size: int) -> torch.Tensor:
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start:start + batch_size])
        inputs = tokenizer(text=batch, return_tensors='pt', padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            embeddings = model(**inputs).text_embeds
            embeddings = F.normalize(embeddings, dim=-1)
        vectors.append(embeddings.cpu())
    return torch.cat(vectors, dim=0)


def stable_start_index(source_name: str, texts: Sequence[str], seed: int) -> int:
    scores = []
    for index, text in enumerate(texts):
        payload = f'{seed}\0{source_name}\0{normalize(text)}'.encode('utf-8')
        scores.append((hashlib.sha256(payload).digest(), index))
    return min(scores)[1]


def farthest_point_indices(embeddings: torch.Tensor, source_name: str,
                           texts: Sequence[str], count: int,
                           seed: int) -> List[int]:
    """Greedily maximize minimum cosine distance with a deterministic start."""
    if count > len(texts):
        raise ValueError(f'Cannot select {count} prompts from {len(texts)}.')
    first_index = stable_start_index(source_name, texts, seed)
    selected = [first_index]
    selected_mask = torch.zeros(len(texts), dtype=torch.bool)
    selected_mask[first_index] = True
    max_similarity = embeddings @ embeddings[first_index]

    while len(selected) < count:
        candidate_scores = max_similarity.masked_fill(selected_mask, float('inf'))
        next_index = int(candidate_scores.argmin().item())
        selected.append(next_index)
        selected_mask[next_index] = True
        max_similarity = torch.maximum(max_similarity,
                                       embeddings @ embeddings[next_index])
    return selected


def unique_candidates(candidates: Iterable[str], selected_keys: set) -> List[str]:
    return [text for text in candidates if normalize(text) not in selected_keys]


def main() -> None:
    args = parse_args()
    if sum(source.quota for source in SOURCES) != 96:
        raise AssertionError('The LAE anchor quotas must total 96.')
    if args.batch_size <= 0:
        raise ValueError('--batch-size must be positive.')

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, local_files_only=True)
    model = CLIPTextModelWithProjection.from_pretrained(
        args.model_name, local_files_only=True).to(device).eval()

    candidates_by_source = {
        source.name: source_candidates(data_root, source) for source in SOURCES
    }
    # Reserve prompts for small vocabularies first so, for example, HRSC's only
    # "ship" anchor cannot be consumed by an earlier source.
    selection_order = sorted(
        SOURCES, key=lambda source: (len(candidates_by_source[source.name]),
                                     source.name))
    selected_keys = set()
    selected_by_source = {}

    for source in selection_order:
        available = unique_candidates(candidates_by_source[source.name],
                                      selected_keys)
        if len(available) < source.quota:
            raise ValueError(
                f'After global de-duplication, {source.name} has '
                f'{len(available)} candidates for quota {source.quota}.')
        embeddings = encode_texts(available, tokenizer, model, device,
                                  args.batch_size)
        indices = farthest_point_indices(embeddings, source.name, available,
                                         source.quota, args.seed)
        chosen = [available[index] for index in indices]
        selected_by_source[source.name] = chosen
        selected_keys.update(normalize(text) for text in chosen)

    selected_texts = [
        text for source in SOURCES for text in selected_by_source[source.name]
    ]
    if len(selected_texts) != 96 or len({normalize(text) for text in selected_texts}) != 96:
        raise AssertionError('Anchor selection must produce 96 normalized-unique texts.')

    anchor_embeddings = encode_texts(selected_texts, tokenizer, model, device,
                                     args.batch_size)
    if anchor_embeddings.shape[0] != 96:
        raise AssertionError('Anchor embedding count does not match anchor texts.')

    output_dir.mkdir(parents=True, exist_ok=True)
    anchor_path = output_dir / 'lae_1m_anchor_96.json'
    manifest_path = output_dir / 'lae_1m_anchor_96_manifest.json'
    embedding_path = output_dir / 'lae_1m_anchor_96_clip_vit_large.pt'
    with anchor_path.open('w', encoding='utf-8') as file:
        json.dump(selected_texts, file, ensure_ascii=False, indent=2)
        file.write('\n')
    with manifest_path.open('w', encoding='utf-8') as file:
        json.dump(
            dict(
                seed=args.seed,
                model_name=args.model_name,
                normalization='NFKC + strip + whitespace collapse + casefold',
                quotas={source.name: source.quota for source in SOURCES},
                source_order=[source.name for source in SOURCES],
                selected_by_source=selected_by_source,
                texts=selected_texts),
            file,
            ensure_ascii=False,
            indent=2)
        file.write('\n')
    torch.save(
        dict(
            model_name=args.model_name,
            texts=selected_texts,
            embeddings=anchor_embeddings), embedding_path)

    print(f'Wrote {anchor_path}')
    print(f'Wrote {manifest_path}')
    print(f'Wrote {embedding_path}')


if __name__ == '__main__':
    main()
