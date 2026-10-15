"""Encode the frozen DIOR 3.2 aerial-description teacher, without any images."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from mmengine.config import Config
from transformers import (AutoTokenizer, CLIPTextConfig,
                          CLIPTextModelWithProjection)

from yolo_world.models.layers.description_regularization import (
    read_description_manifest, similarity_from_embeddings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--descriptions',
                        default='configs/pretrain/dior_v3_2/aerial_descriptions.json')
    parser.add_argument('--baseline', default='configs/pretrain/DIORv2.17best.py')
    parser.add_argument('--model-name',
                        default='pretrained_models/clip-vit-large-patch14-336')
    parser.add_argument('--output',
                        default='assets/dior_v3_2_description_matrix_v2.pt')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--num-threads', type=int, default=4)
    parser.add_argument('--force', action='store_true',
                        help='Explicitly permit replacing the output artifact.')
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_threads <= 0:
        parser.error('batch-size and num-threads must be positive.')
    output = Path(args.output)
    if output.exists() and not args.force:
        parser.error('Output exists; use --force only to deliberately rebuild it.')
    torch.set_num_threads(args.num_threads)
    baseline = Config.fromfile(args.baseline)
    expected = baseline.model.backbone.text_model.prompt_bank_cfg.semantic_anchor_texts
    names, descriptions, checksum = read_description_manifest(args.descriptions,
                                                             expected)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    config = CLIPTextConfig.from_pretrained(args.model_name, local_files_only=True)
    model = CLIPTextModelWithProjection.from_pretrained(
        args.model_name, config=config, local_files_only=True).to(args.device).eval()
    model.requires_grad_(False)
    batches = []
    for start in range(0, len(descriptions), args.batch_size):
        inputs = tokenizer(text=descriptions[start:start + args.batch_size],
                           return_tensors='pt', padding=True, truncation=False)
        if inputs['input_ids'].shape[1] > config.max_position_embeddings:
            parser.error('A description exceeds CLIP context length; shorten it.')
        with torch.no_grad():
            vectors = model(**inputs.to(args.device)).text_embeds
        batches.append(vectors.cpu())
    matrix = similarity_from_embeddings(torch.cat(batches))
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(schema_version=1, class_names=names, descriptions=descriptions,
                    encoder_model=args.model_name, source_sha256=checksum,
                    cosine_similarity=matrix), output)
    print('Saved {}: {} x {} raw cosine teacher'.format(output, *matrix.shape))
    neighbours = matrix.clone()
    neighbours.fill_diagonal_(-torch.inf)
    for index, name in enumerate(names):
        values, indices = neighbours[index].topk(min(3, len(names) - 1))
        entries = ['{} ({:.3f})'.format(names[int(j)], float(s))
                   for j, s in zip(indices, values)]
        print('{}: {}'.format(name, ', '.join(entries)))


if __name__ == '__main__':
    main()
