"""Create a frozen CLIP text embedding table for semantic anchoring."""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from transformers import (AutoTokenizer, CLIPTextConfig,
                          CLIPTextModelWithProjection)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-name',
        default='pretrained_models/clip-vit-large-patch14-336')
    parser.add_argument(
        '--class-text-path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument(
        '--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def load_texts(path):
    with open(path, 'r', encoding='utf-8') as file:
        entries = json.load(file)
    texts = []
    for entry in entries:
        if isinstance(entry, list):
            if len(entry) != 1 or not isinstance(entry[0], str):
                raise ValueError('Each class-text entry must contain one string.')
            entry = entry[0]
        if not isinstance(entry, str):
            raise TypeError('Class texts must be strings.')
        texts.append(entry)
    return texts


def main():
    args = parse_args()
    texts = load_texts(args.class_text_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = CLIPTextConfig.from_pretrained(args.model_name)
    model = CLIPTextModelWithProjection.from_pretrained(
        args.model_name, config=config).to(args.device).eval()
    inputs = tokenizer(text=texts, return_tensors='pt', padding=True)
    inputs = inputs.to(args.device)
    with torch.no_grad():
        embeddings = model(**inputs).text_embeds
        embeddings = F.normalize(embeddings, dim=-1).cpu()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(
        dict(
            model_name=args.model_name,
            texts=texts,
            embeddings=embeddings), args.output)


if __name__ == '__main__':
    main()
