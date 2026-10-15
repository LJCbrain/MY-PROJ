"""Short CAVP-Full validation on two legal DIOR training images.

This is a smoke/fit check, not an AP experiment or a full training run.
Use the normal project environment; no extension or optimizer shims are used.
"""

import argparse
import contextlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--image-size', type=int, default=160)
    parser.add_argument('--fit-steps', type=int, default=40)
    parser.add_argument('--output-dir', default='.codex_tmp/cavp_validation')
    args = parser.parse_args()

    from mmengine.config import Config
    from mmengine.logging import MMLogger
    from mmengine.optim import build_optim_wrapper
    from mmengine.runner import load_checkpoint
    from mmdet.structures import DetDataSample
    from mmdet.structures.bbox import bbox_overlaps
    from mmyolo.registry import MODELS
    from mmyolo.utils import register_all_modules
    import yolo_world  # noqa: F401

    register_all_modules()
    torch.manual_seed(17)
    torch.set_num_threads(4)
    cfg = Config.fromfile(str(ROOT / 'configs/pretrain/DIORv2.11.py'))
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    logger = MMLogger.get_instance('cavp_validation',
                                   log_file=str(output / 'run.log'))
    for handler in logger.handlers:
        if not isinstance(handler, logging.FileHandler):
            handler.setLevel(logging.WARNING)
    payload = json.loads((ROOT / 'data/DIOR/Annotations2/train_base_16cls_image_disjoint_yoloworld.json').read_text())
    class_names = list(cfg.DIOR_BASE_CLASSES)
    categories = {cat['id']: cat['name'] for cat in payload['categories']}
    by_image = {}
    for ann in payload['annotations']:
        by_image.setdefault(ann['image_id'], []).append(ann)
    # Keep all GT, but select scenes whose boxes still contain feature-grid
    # centers at this deliberately small smoke-test resolution.
    chosen = [im for im in payload['images']
              if 1 <= len(by_image.get(im['id'], [])) <= 10 and
              all(min(ann['bbox'][2] * args.image_size / im['width'],
                      ann['bbox'][3] * args.image_size / im['height']) >= 16
                  for ann in by_image[im['id']])][:2]
    assert len(chosen) == 2
    tensors, targets, metas = [], [], []
    for index, info in enumerate(chosen):
        image = Image.open(ROOT / 'data/DIOR/JPEGImages' / info['file_name']).convert('RGB')
        width, height = image.size
        resized = image.resize((args.image_size, args.image_size), Image.Resampling.BILINEAR)
        tensors.append(torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1).float() / 255)
        sx, sy = args.image_size / width, args.image_size / height
        for ann in by_image[info['id']]:
            x, y, w, h = ann['bbox']
            targets.append([index, class_names.index(categories[ann['category_id']]),
                            x * sx, y * sy, (x + w) * sx, (y + h) * sy])
        metas.append(dict(img_id=info['id'], img_shape=(args.image_size, args.image_size),
                          ori_shape=(height, width), scale_factor=(sx, sy),
                          pad_param=(0, 0, 0, 0)))
    inputs = torch.stack(tensors).to(args.device)
    batch = dict(bboxes_labels=torch.tensor(targets, device=args.device),
                 img_metas=metas, texts=[class_names] * 2)
    print('Building complete v2.11 with local CLIP and pretrained YOLO weights.', flush=True)
    with (output / 'checkpoint_load.log').open('w', encoding='utf-8') as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            model = MODELS.build(cfg.model)
            model.init_weights()
            load_checkpoint(model, cfg.load_from, map_location='cpu', strict=False)
    model.to(args.device).train()
    optim_cfg = cfg.optim_wrapper.copy()
    if not args.device.startswith('cuda'):
        optim_cfg.type = 'OptimWrapper'
        optim_cfg.pop('loss_scale', None)
    with (output / 'optimizer_build.log').open('w', encoding='utf-8') as log:
        with contextlib.redirect_stdout(log):
            wrapper = build_optim_wrapper(model, optim_cfg)
    main_weight = model.bbox_head.head_module.reg_preds[0][-1].weight
    cavp_weight = model.cavp_head.head_module.reg_preds[0][-1].weight
    before_main, before_cavp = main_weight.detach().clone(), cavp_weight.detach().clone()
    with wrapper.optim_context(model):
        losses = model.loss(inputs, batch)
        total = sum(value for key, value in losses.items() if 'loss' in key)
    assert torch.isfinite(total)
    if args.device.startswith('cuda'):
        # Remove first-step overflow ambiguity from the smoke check. The
        # regular training config still uses dynamic loss scaling unchanged.
        wrapper.loss_scaler = torch.cuda.amp.GradScaler(init_scale=128.)
    wrapper.update_params(total)
    assert not torch.equal(main_weight, before_main), 'Main optimizer step was skipped.'
    assert not torch.equal(cavp_weight, before_cavp), 'CAVP optimizer step was skipped.'
    initial = {key: float(value.detach()) for key, value in losses.items() if 'loss' in key}
    print('Joint training step:', json.dumps(initial), flush=True)

    # A bounded fit diagnostic isolates the new head on fixed real features.
    # This does not freeze the main path in the actual training configuration.
    model.eval()
    with torch.no_grad():
        raw = model.backbone.forward_image(inputs)
    model.cavp_neck.train()
    model.cavp_head.train()
    aux_optimizer = torch.optim.AdamW(
        list(model.cavp_neck.parameters()) + list(model.cavp_head.parameters()),
        lr=cfg.base_lr, weight_decay=cfg.weight_decay)
    history = []
    for step in range(args.fit_steps):
        aux_optimizer.zero_grad(set_to_none=True)
        loss = sum(model.cavp_head.loss(model.cavp_neck(raw), batch).values())
        assert torch.isfinite(loss)
        loss.backward()
        aux_optimizer.step()
        history.append(float(loss.detach()))
        if step == 0 or (step + 1) % 10 == 0:
            print(f'CAVP fit {step + 1}/{args.fit_steps}: {history[-1]:.5f}', flush=True)
    if history:
        assert history[-1] < history[0], 'CAVP fit loss did not improve.'
    model.eval()
    inference = [DetDataSample(metainfo=dict(meta, texts=list(cfg.DIOR_FULL_CLASSES)))
                 for meta in metas]
    with torch.no_grad():
        predicted = model.predict(inputs, inference)
        object_logits, distances = model.cavp_head(model.cavp_neck(raw))
        proposals = model.cavp_head.predict_proposals(object_logits, distances, metas)
    recalls = []
    for i, proposal in enumerate(proposals):
        gt = batch['bboxes_labels'][batch['bboxes_labels'][:, 0] == i, 2:]
        best = bbox_overlaps(gt, proposal.bboxes).max(1).values
        recalls.extend((best >= 0.5).float().tolist())
    result = dict(
        config='DIORv2.11', device=args.device, image_size=args.image_size,
        image_ids=[im['id'] for im in chosen], joint_losses=initial,
        cavp_fit_losses=history, fit_gt_coverage_iou50=float(np.mean(recalls)),
        detections=[len(x.pred_instances) for x in predicted],
        fusion_stats=[x.cavp_stats for x in predicted],
        total_parameters=sum(p.numel() for p in model.parameters()),
        cavp_parameters=sum(p.numel() for module in (model.cavp_neck, model.cavp_head)
                            for p in module.parameters()),
        note='Two training images only; not validation AP or Novel accuracy.')
    if args.device.startswith('cuda'):
        result['peak_gpu_memory_bytes'] = torch.cuda.max_memory_allocated()
    (output / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('RESULT', json.dumps({k: v for k, v in result.items() if k != 'cavp_fit_losses'}), flush=True)


if __name__ == '__main__':
    main()
