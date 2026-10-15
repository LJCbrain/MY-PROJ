"""One-forward S0/S1/S2/S3 semantic readout ablation on a CAVP checkpoint.

No training. All variants share the same main detections and CAVP top-300 boxes.
GT boxes are used only for diagnostic reference readouts, never detections.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.structures import InstanceData
from mmyolo.registry import MODELS, METRICS

import yolo_world  # noqa: F401
from yolo_world.evaluation.dior_coco_metric import DIORCocoMetric
from yolo_world.evaluation.proposal_recall_metric import PriorProposalRecallMetric
from yolo_world.models.dense_heads.cavp_head import CAVPHead, map_cavp_boxes
from yolo_world.models.detectors.yolo_world_cavp import (
    YOLOWorldCAVPDetector, read_cavp_logits, fuse_cavp_detections)
from tools.val.test_dior_cavp_diagnosis import novel_stage

VARIANTS = ('S0', 'S1', 'S2', 'S3', 'S0_multilabel')


def _sample_field(sample, key):
    """Read a field from either a runner dict or a DetDataSample."""
    if isinstance(sample, dict):
        return sample[key]
    return getattr(sample, key)


def sample_box_logits(cls_scores, boxes, image_index, strides, grid_size=3, mask=None):
    """Per-class max of bilinear samples over every level and fixed box grid.

    S3 uses 3x3 bin centers: fractions 1/6, 1/2, 5/6 on each axis. This is a
    high-recall readout probe; max pooling can increase false positives.
    """
    classes = cls_scores[0].shape[1]
    if not len(boxes):
        return boxes.new_empty((0, classes), dtype=torch.float32)
    fractions = (torch.arange(grid_size, device=boxes.device).float() + .5) / grid_size
    yy, xx = torch.meshgrid(fractions, fractions, indexing='ij')
    fractions = torch.stack((xx.flatten(), yy.flatten()), -1)
    positions = boxes[:, None, :2] + (boxes[:, None, 2:] - boxes[:, None, :2]) * fractions
    levels = []
    for feature, stride in zip(cls_scores, strides):
        height, width = feature.shape[-2:]
        grid = 2 * positions / positions.new_tensor([stride * width, stride * height]) - 1
        values = F.grid_sample(feature[image_index:image_index+1].float(), grid[None],
                               align_corners=False, padding_mode='border')[0]
        levels.append(values.amax(-1).T)
    result = torch.stack(levels).amax(0)
    if mask is not None:
        result = result.masked_fill(~mask.to(device=result.device, dtype=torch.bool)[None], -torch.inf)
    return result


def label_boxes(boxes, probabilities, threshold, multilabel=False):
    if multilabel:
        indices, labels = (torch.isfinite(probabilities) & (probabilities > threshold)).nonzero(as_tuple=True)
        scores = probabilities[indices, labels]
    else:
        scores, labels = probabilities.max(1)
        indices = (torch.isfinite(scores) & (scores > threshold)).nonzero(as_tuple=False).flatten()
        scores, labels = scores[indices], labels[indices]
    valid = ((boxes[indices, 2:] - boxes[indices, :2]) > 0).all(-1)
    return InstanceData(bboxes=boxes[indices][valid], scores=scores[valid], labels=labels[valid])


@MODELS.register_module()
class DIORCAVPSemanticDetector(YOLOWorldCAVPDetector):
    def __init__(self, *args, semantic_ann_file=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.semantic_gt = defaultdict(list)
        if semantic_ann_file:
            payload = json.loads(Path(semantic_ann_file).read_text(encoding='utf-8'))
            for ann in payload['annotations']:
                x, y, w, h = ann['bbox']
                self.semantic_gt[int(ann['image_id'])].append([x, y, x+w, y+h])

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        if not rescale:
            raise ValueError('Original-image coordinates are required.')
        features, text, masks, visual = self._extract_dual_features(batch_inputs, batch_data_samples)
        self.bbox_head.num_classes = text.shape[-2]
        cls_scores, bbox_preds = self.bbox_head(features, text, masks)
        metas = [sample.metainfo for sample in batch_data_samples]
        main = self.bbox_head.predict_by_feat(cls_scores, bbox_preds, batch_img_metas=metas, rescale=True)
        settings = dict(pre_nms_topk=3000, nms_iou_threshold=.7, max_per_img=300)
        main_props = CAVPHead.predict_proposals(self.bbox_head,
            [s.float().max(1, keepdim=True).values for s in cls_scores], bbox_preds, metas, **settings)
        obj, distances = self.cavp_head(visual)
        proposals = self.cavp_head.predict_proposals(obj, distances, metas, **settings)
        cfg = self.cavp_cfg
        def array(value):
            return value.detach().cpu().numpy()
        for i, (sample, original, mp, cp, meta) in enumerate(zip(batch_data_samples, main, main_props, proposals, metas)):
            mask = None if masks is None else (masks[i] if masks.ndim > 1 else masks)
            strides = self.cavp_head.featmap_strides
            source = read_cavp_logits(cls_scores, cp, i, strides, text_mask=mask, mode='source')
            center = read_cavp_logits(cls_scores, cp, i, strides, text_mask=mask,
                                      mode='source_center', source_weight=0.)
            logits = dict(S0=.5*source+.5*center, S1=source, S2=center,
                          S3=sample_box_logits(cls_scores, cp.bboxes, i, strides, mask=mask))
            logits['S0_multilabel'] = logits['S0']
            boxes = map_cavp_boxes(cp.bboxes, meta)
            sample.semantic_outputs = {}
            diag = dict(main=array(map_cavp_boxes(mp.bboxes, meta)), cavp=array(boxes), variants={})
            for name, values in logits.items():
                probs = (values / cfg['score_temperature'] + cfg['score_bias']).sigmoid()
                rescue = label_boxes(boxes, probs, cfg['semantic_score_thr'], name == 'S0_multilabel')
                final, _ = fuse_cavp_detections(original, rescue, cfg['duplicate_iou'], cfg['max_per_img'])
                sample.semantic_outputs[name] = final
                diag['variants'][name] = dict(probabilities=array(probs), rescue_boxes=array(rescue.bboxes),
                    rescue_labels=array(rescue.labels), final_boxes=array(final.bboxes), final_labels=array(final.labels))
            # GT reference is computed only after predictions have been constructed.
            # Annotation order matches the diagnostic metric's ordered GT lists.
            gt_boxes = boxes.new_tensor(self.semantic_gt[int(meta['img_id'])]).reshape(-1, 4)
            scale = gt_boxes.new_tensor(meta['scale_factor']).flatten()
            if scale.numel() == 2:
                scale = scale.repeat(2)
            pad = meta.get('pad_param', (0, 0, 0, 0))
            input_gt = gt_boxes * scale + gt_boxes.new_tensor([pad[2], pad[0], pad[2], pad[0]])
            diag['gt_reference'] = {}
            for size, name in ((1, 'center_all_levels'), (3, 'grid3_all_levels')):
                ref = sample_box_logits(cls_scores, input_gt, i, strides, size, mask)
                diag['gt_reference'][name] = array((ref / cfg['score_temperature'] + cfg['score_bias']).sigmoid())
            sample.semantic_diagnostic = diag
        self._check_pred_label_range(main, self.bbox_head.num_classes)
        return self.add_pred_to_datasample(batch_data_samples, main)


@METRICS.register_module()
class DIORSemanticVariantAP(DIORCocoMetric):
    def __init__(self, variant, **kwargs):
        super().__init__(**kwargs)
        self.variant = variant

    def process(self, data_batch, data_samples):
        selected = []
        for sample in data_samples:
            # Do not pass the large per-variant diagnostic fields to CocoMetric.
            sample_dict = sample if isinstance(sample, dict) else sample.to_dict()
            result = {k:v for k,v in sample_dict.items()
                      if k not in ('semantic_outputs', 'semantic_diagnostic')}
            if self.variant != 'main':
                result['pred_instances'] = _sample_field(sample, 'semantic_outputs')[self.variant]
            selected.append(result)
        super().process(data_batch, selected)


@METRICS.register_module()
class DIORSemanticStages(PriorProposalRecallMetric):
    default_prefix = 'semantic'

    def __init__(self, classes, output_dir, threshold, variants=None, **kwargs):
        super().__init__(**kwargs)
        self.classes = tuple(classes)
        self.output_dir = Path(output_dir)
        self.threshold = threshold
        self.variants = tuple(VARIANTS if variants is None else variants)

    def process(self, data_batch, data_samples):
        for sample in data_samples:
            img_id = int(_sample_field(sample, 'img_id'))
            diag = _sample_field(sample, 'semantic_diagnostic')
            rows, refs = [], []
            for index, (box, group, name) in enumerate(self._ground_truth.get(img_id, [])):
                c = self.classes.index(name)
                for variant in self.variants:
                    v = diag['variants'][variant]
                    row = novel_stage(box, c, dict(main=diag['main'], cavp=diag['cavp'], **v), self.threshold)
                    iou = self._iou(np.asarray(box), diag['cavp'])
                    covered = iou >= .5
                    probs = v['probabilities']
                    rank = 1 + (probs > probs[:, c:c+1]).sum(1)
                    best_rank = int(rank[covered].min()) if covered.any() else len(self.classes)+1
                    final_iou = self._iou(np.asarray(box), v['final_boxes'])
                    row.update(image_id=img_id, gt_index=index, gt_class=name, group=group,
                        variant=variant, best_gt_rank_on_covered=best_rank,
                        final_correct=int(((final_iou >= .5) & (v['final_labels'] == c)).any()))
                    if 'current_main' in diag:
                        row['cavp_only_vs_current_main'] = int(covered.any() and not (
                            self._iou(np.asarray(box), diag['current_main']) >= .5).any())
                    rows.append(row)
                for method, ref in diag['gt_reference'].items():
                    p = ref[index]
                    refs.append(dict(image_id=img_id, gt_index=index, gt_class=name, group=group,
                        method=method, correct_top1=int(p.argmax() == c),
                        gt_rank=int(1+(p > p[c]).sum()), gt_score=float(p[c]),
                        predicted_class=self.classes[int(p.argmax())]))
            self.results.append(dict(image_id=img_id, rows=rows, refs=refs))

    def compute_metrics(self, results):
        seen, rows, refs = set(), [], []
        for r in results:
            if r['image_id'] not in seen:
                seen.add(r['image_id'])
                rows.extend(r['rows'])
                refs.extend(r['refs'])
        summary = {}
        flags = ('cavp_covered', 'any_correct_top1', 'any_gt_score_above_threshold',
                 'any_correct_top1_above_threshold', 'rescue_correct', 'final_correct')
        for variant in self.variants:
            summary[variant] = {}
            scopes = ('base', 'novel', 'cavp_only_novel') + self.novel_classes
            if rows and 'cavp_only_vs_current_main' in rows[0]:
                scopes += ('cavp_only_vs_current_main_novel',)
            for scope in scopes:
                selected = [r for r in rows if r['variant'] == variant and (
                    (scope in ('base','novel') and r['group'] == scope) or
                    (scope == 'cavp_only_novel' and r['group'] == 'novel' and r['cavp_only']) or
                    (scope == 'cavp_only_vs_current_main_novel' and r['group'] == 'novel'
                     and r.get('cavp_only_vs_current_main', 0)) or
                    r['gt_class'] == scope)]
                summary[variant][scope] = dict(gt=len(selected), **{k:sum(r[k] for r in selected) for k in flags},
                    rank_top3=sum(r['best_gt_rank_on_covered'] <= 3 for r in selected),
                    rank_top5=sum(r['best_gt_rank_on_covered'] <= 5 for r in selected))
        reference = {}
        for method in ('center_all_levels', 'grid3_all_levels'):
            reference[method] = {}
            for group in ('base', 'novel'):
                selected = [r for r in refs if r['method'] == method and r['group'] == group]
                reference[method][group] = dict(gt=len(selected), correct_top1=sum(r['correct_top1'] for r in selected),
                    rank_top3=sum(r['gt_rank'] <= 3 for r in selected),
                    above_threshold=sum(r['gt_score'] > self.threshold for r in selected))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report = dict(images=len(seen), threshold=self.threshold, variants=summary,
                      gt_reference=reference if refs else {})
        (self.output_dir/'semantic_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        for filename, items in (('novel_semantic_details.csv', [r for r in rows if r['group']=='novel']),
                                ('gt_reference.csv', refs)):
            if items:
                with (self.output_dir/filename).open('w', newline='', encoding='utf-8-sig') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(items[0]))
                    writer.writeheader()
                    writer.writerows(items)
        return {f'{v}/cavp_only_rescue_correct': summary[v]['cavp_only_novel']['rescue_correct'] for v in self.variants}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/pretrain/DIORv2.11.py')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--work-dir', default='work_dirs/DIORv2.11/semantic_epoch44')
    parser.add_argument('--limit', type=int, default=0, help='Smoke test only, no AP; 0 = full validation.')
    args = parser.parse_args()
    os.chdir(ROOT)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        parser.error(f'Checkpoint not found: {checkpoint}')
    if args.limit < 0:
        parser.error('--limit must be nonnegative.')
    cfg = Config.fromfile(args.config)
    if cfg.model.type != 'YOLOWorldCAVPDetector':
        parser.error('Use a CAVP model config.')
    coco = copy.deepcopy(cfg.val_evaluator[0])
    ann_file = Path(coco.ann_file).resolve()
    cfg.test_dataloader = copy.deepcopy(cfg.val_dataloader)
    dataset = cfg.test_dataloader.dataset
    classes = tuple(dataset.dataset.metainfo.classes)
    data_ann = (Path(dataset.dataset.data_root)/dataset.dataset.ann_file).resolve()
    if data_ann != ann_file or len(classes) != 20 or set(classes) != set(coco.base_classes) | set(coco.novel_classes):
        parser.error('Requires matching full DIOR 20-class validation annotations.')
    # All reference annotations must be among the 20 evaluated categories so
    # model and metric retain exactly the same per-image GT order.
    payload = json.loads(ann_file.read_text(encoding='utf-8'))
    if {cat['name'] for cat in payload['categories']} != set(classes):
        parser.error('Annotation category vocabulary mismatch.')
    cfg.model.type = 'DIORCAVPSemanticDetector'
    cfg.model.semantic_ann_file = str(ann_file)
    cfg.model.cavp_cfg.update(enabled=True, fusion_enabled=False,
        semantic_readout='source_center', source_weight=.5, semantic_margin=0.)
    cfg.load_from, cfg.resume, cfg.launcher = str(checkpoint), False, 'none'
    cfg.work_dir = str(Path(args.work_dir).resolve())
    if args.limit:
        dataset.dataset.indices = args.limit
        cfg.test_dataloader.num_workers = 0
        cfg.test_dataloader.persistent_workers = False
    cfg.test_evaluator = []
    if not args.limit:
        for variant in ('main',) + VARIANTS:
            evaluator = copy.deepcopy(coco)
            evaluator.update(type='DIORSemanticVariantAP', variant=variant, prefix=variant)
            cfg.test_evaluator.append(evaluator)
    cfg.test_evaluator.append(dict(type='DIORSemanticStages', classes=classes,
        output_dir=cfg.work_dir, threshold=cfg.model.cavp_cfg.semantic_score_thr,
        ann_file=str(ann_file), base_classes=coco.base_classes, novel_classes=coco.novel_classes))
    runner = Runner.from_cfg(cfg)
    protocol = dict(checkpoint=str(checkpoint), config=str(Path(args.config).resolve()),
        checkpoint_loading='Standard Runner load_from; resume=False; no manual ema_state_dict extraction.',
        limit=args.limit, partial_run=bool(args.limit), classes=classes,
        annotation_sha256=hashlib.sha256(ann_file.read_bytes()).hexdigest(),
        variants=dict(S0='0.5 source + 0.5 same-level center', S1='source only',
            S2='same-level center only', S3='per-class max over 3x3 bin centers on all three levels',
            S0_multilabel='S0 logits, all classes above the same threshold'),
        shared_candidates=dict(pre_nms_topk=3000, class_agnostic_nms=.7, max_per_img=300),
        shared_fusion='Existing same-class duplicate removal then global top300, unchanged across variants.',
        semantic_settings=dict(cfg.model.cavp_cfg),
        gt_reference='Diagnostic only; GT boxes never enter prediction or fusion.')
    destination = Path(cfg.work_dir)
    (destination/'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    metrics = runner.test()
    (destination/'metrics.json').write_text(json.dumps({k:float(v) for k,v in metrics.items()}, indent=2), encoding='utf-8')
    if not args.limit:
        columns = ('bbox_mAP', 'bbox_mAP_50', 'bbox_mAP_50_base', 'bbox_mAP_50_novel', 'bbox_mAP_50_HM')
        with (destination/'ap_comparison.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=['variant'] + list(columns))
            writer.writeheader()
            for variant in ('main',) + VARIANTS:
                writer.writerow(dict(variant=variant, **{k:metrics[f'{variant}/{k}'] for k in columns}))


if __name__ == '__main__':
    main()
