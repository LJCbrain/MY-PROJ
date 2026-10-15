"""Compare two semantic readers on identical epoch44 CAVP proposals; no training."""
import argparse
import copy
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

import torch
from mmengine.config import Config
from mmengine.model import revert_sync_batchnorm
from mmengine.runner import Runner, load_checkpoint
from mmyolo.registry import MODELS

from tools.val.test_dior_cavp_semantics import label_boxes  # registers metrics
from yolo_world.models.dense_heads.cavp_head import CAVPHead, map_cavp_boxes
from yolo_world.models.detectors.yolo_world_cavp import (
    YOLOWorldCAVPDetector, read_cavp_logits, fuse_cavp_detections)

VARIANTS = ('B_current_reader', 'C_old_reader')


@MODELS.register_module()
class DIORCrossReaderDetector(YOLOWorldCAVPDetector):
    def __init__(self, *args, reader_config, reader_checkpoint, **kwargs):
        super().__init__(*args, **kwargs)
        self.reader_config = reader_config
        self.reader_checkpoint = reader_checkpoint
        # Load lazily AFTER Runner has loaded the CAVP checkpoint. This keeps
        # the two state_dict namespaces independent and avoids missing keys.
        object.__setattr__(self, '_old_reader', None)

    def _reader(self, device):
        if self._old_reader is None:
            reader = MODELS.build(copy.deepcopy(self.reader_config))
            reader = revert_sync_batchnorm(reader)
            # Checkpoints saved by this project's EMAHook put deployment EMA
            # weights in state_dict. Never extract ema_state_dict manually.
            load_checkpoint(reader, self.reader_checkpoint, map_location='cpu', strict=True)
            reader.to(device)
            reader.eval()
            reader.requires_grad_(False)
            object.__setattr__(self, '_old_reader', reader)
        return self._old_reader

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        if not rescale:
            raise ValueError('This comparison requires original-image coordinates.')
        reader = self._reader(batch_inputs.device)
        features, text, masks, visual = self._extract_dual_features(batch_inputs, batch_data_samples)
        self.bbox_head.num_classes = text.shape[-2]
        current_cls, current_reg = self.bbox_head(features, text, masks)
        old_features, old_text, old_masks = reader.extract_feat(batch_inputs, batch_data_samples)
        reader.bbox_head.num_classes = old_text.shape[-2]
        old_cls, old_reg = reader.bbox_head(old_features, old_text, old_masks)
        if any(a.shape != b.shape for a, b in zip(current_cls, old_cls)):
            raise ValueError('Reader grids or class counts differ.')
        if (masks is None) != (old_masks is None) or (
                masks is not None and not torch.equal(masks, old_masks)):
            raise ValueError('Reader class masks differ.')
        metas = [s.metainfo for s in batch_data_samples]
        # A, B and C all use these exact old-model main detections.
        main = reader.bbox_head.predict_by_feat(old_cls, old_reg,
            batch_img_metas=metas, rescale=True)
        settings = dict(pre_nms_topk=3000, nms_iou_threshold=.7, max_per_img=300)
        old_props = CAVPHead.predict_proposals(reader.bbox_head,
            [s.float().max(1, keepdim=True).values for s in old_cls], old_reg, metas, **settings)
        current_props = CAVPHead.predict_proposals(self.bbox_head,
            [s.float().max(1, keepdim=True).values for s in current_cls], current_reg, metas, **settings)
        obj, distances = self.cavp_head(visual)
        proposals = self.cavp_head.predict_proposals(obj, distances, metas, **settings)
        cfg = self.cavp_cfg
        def array(value):
            return value.detach().cpu().numpy()
        for i, (sample, base, op, np_, cp, meta) in enumerate(zip(
                batch_data_samples, main, old_props, current_props, proposals, metas)):
            boxes = map_cavp_boxes(cp.bboxes, meta)
            outputs = {}
            diag = dict(main=array(map_cavp_boxes(op.bboxes, meta)),
                current_main=array(map_cavp_boxes(np_.bboxes, meta)), cavp=array(boxes),
                variants={}, gt_reference={})
            for variant, scores, valid_text in (
                    (VARIANTS[0], current_cls, masks), (VARIANTS[1], old_cls, old_masks)):
                mask = None if valid_text is None else (
                    valid_text[i] if valid_text.ndim > 1 else valid_text)
                logits = read_cavp_logits(scores, cp, i, self.cavp_head.featmap_strides,
                    text_mask=mask, mode='source_center', source_weight=.5)
                probs = (logits / cfg['score_temperature'] + cfg['score_bias']).sigmoid()
                rescue = label_boxes(boxes, probs, cfg['semantic_score_thr'])
                final, _ = fuse_cavp_detections(base, rescue,
                    cfg['duplicate_iou'], cfg['max_per_img'])
                outputs[variant] = final.to_dict()
                diag['variants'][variant] = dict(probabilities=array(probs),
                    rescue_boxes=array(rescue.bboxes), rescue_labels=array(rescue.labels),
                    final_boxes=array(final.bboxes), final_labels=array(final.labels))
            sample.semantic_outputs = outputs
            sample.semantic_diagnostic = diag
        reader._check_pred_label_range(main, reader.bbox_head.num_classes)
        return self.add_pred_to_datasample(batch_data_samples, main)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/pretrain/DIORv2.11.py')
    parser.add_argument('--old-config', default='configs/pretrain/DIORv2.10.py')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--old-checkpoint', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--limit', type=int, default=0, help='Smoke test only; no AP. 0 = full validation.')
    args = parser.parse_args()
    os.chdir(ROOT)
    for name in ('checkpoint', 'old_checkpoint'):
        path = Path(getattr(args, name)).resolve()
        if not path.is_file():
            parser.error(f'Missing checkpoint: {path}')
        setattr(args, name, str(path))
    if args.limit < 0:
        parser.error('--limit must be nonnegative.')
    cfg, old = Config.fromfile(args.config), Config.fromfile(args.old_config)
    if cfg.model.type != 'YOLOWorldCAVPDetector' or old.model.type != 'YOLOWorldDetector':
        parser.error('Expected a 2.11 CAVP config and a 2.10 main-model config.')
    if cfg.val_dataloader.dataset != old.val_dataloader.dataset:
        parser.error('Validation data, class text order and transforms must match.')
    if cfg.model.data_preprocessor != old.model.data_preprocessor:
        parser.error('Reader preprocessing must match.')
    if tuple(cfg.model.cavp_head.head_module.featmap_strides) != tuple(old.model.bbox_head.head_module.featmap_strides):
        parser.error('Reader strides must match.')
    coco = copy.deepcopy(cfg.val_evaluator[0])
    old_coco = old.val_evaluator[0] if isinstance(old.val_evaluator, (list, tuple)) else old.val_evaluator
    if coco != old_coco:
        parser.error('COCO evaluation settings must match.')
    ann_file = Path(coco.ann_file).resolve()
    classes = tuple(cfg.val_dataloader.dataset.dataset.metainfo.classes)
    dataset_cfg = cfg.val_dataloader.dataset.dataset
    if (Path(dataset_cfg.data_root) / dataset_cfg.ann_file).resolve() != ann_file:
        parser.error('Dataset and evaluator annotations must match.')
    if len(classes) != 20 or set(classes) != set(coco.base_classes) | set(coco.novel_classes):
        parser.error('Expected the complete DIOR 20-class vocabulary.')
    cfg.model.update(type='DIORCrossReaderDetector', reader_config=copy.deepcopy(old.model),
        reader_checkpoint=args.old_checkpoint)
    cfg.model.cavp_cfg.update(enabled=True, fusion_enabled=False,
        semantic_readout='source_center', source_weight=.5, semantic_margin=0.)
    cfg.load_from, cfg.resume, cfg.launcher = args.checkpoint, False, 'none'
    cfg.work_dir = str(Path(args.work_dir).resolve())
    cfg.test_dataloader = copy.deepcopy(cfg.val_dataloader)
    if args.limit:
        cfg.test_dataloader.dataset.dataset.indices = args.limit
        cfg.test_dataloader.num_workers = 0
        cfg.test_dataloader.persistent_workers = False
    cfg.test_evaluator = []
    groups = (('A_old_main', 'main'),) + tuple((v, v) for v in VARIANTS)
    if not args.limit:
        for prefix, variant in groups:
            metric = copy.deepcopy(coco)
            metric.update(type='DIORSemanticVariantAP', variant=variant, prefix=prefix)
            cfg.test_evaluator.append(metric)
    cfg.test_evaluator.append(dict(type='DIORSemanticStages', classes=classes, variants=VARIANTS,
        output_dir=cfg.work_dir, threshold=cfg.model.cavp_cfg.semantic_score_thr,
        ann_file=str(ann_file), base_classes=coco.base_classes, novel_classes=coco.novel_classes))
    runner = Runner.from_cfg(cfg)
    destination = Path(cfg.work_dir)
    protocol = dict(current_checkpoint=args.checkpoint, old_checkpoint=args.old_checkpoint,
        current_config=str(Path(args.config).resolve()), old_config=str(Path(args.old_config).resolve()),
        checkpoint_loading='Deployment state_dict for both checkpoints; no manual ema_state_dict extraction.',
        partial_run=bool(args.limit), classes=classes,
        annotation_sha256=hashlib.sha256(ann_file.read_bytes()).hexdigest(),
        main='Same 2.10 detections for A/B/C', proposals='Same 2.11 CAVP top300 for B/C',
        readout='0.5 source + 0.5 same-level center; top1 label',
        semantic_settings=dict(cfg.model.cavp_cfg),
        cavp_only_novel='CAVP-covered but old main top300 proposals missed, IoU0.5',
        cavp_only_vs_current_main_novel='CAVP-covered but current main top300 proposals missed (previous 194-GT group)',
        gt_reference='Not run here; GT is used only by diagnostic metrics, never prediction.')
    (destination / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    metrics = runner.test()
    (destination / 'metrics.json').write_text(json.dumps({k:float(v) for k,v in metrics.items()}, indent=2), encoding='utf-8')
    if not args.limit:
        columns = ('bbox_mAP', 'bbox_mAP_50', 'bbox_mAP_50_base', 'bbox_mAP_50_novel', 'bbox_mAP_50_HM')
        with (destination / 'ap_comparison.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=['variant'] + list(columns))
            writer.writeheader()
            for prefix, _ in groups:
                writer.writerow(dict(variant=prefix, **{k:metrics[f'{prefix}/{k}'] for k in columns}))


if __name__ == '__main__':
    main()
