"""Same-checkpoint DIOR proposal complementarity and semantic diagnostics.

Uses the standard Runner checkpoint/EMA loading path. Main AP/HM is evaluated
with fusion disabled. This file adds no model parameters and performs no training.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

import numpy as np
import torch
from mmcv.ops import nms
from mmengine.config import Config
from mmengine.dist import is_main_process
from mmengine.runner import Runner
from mmyolo.registry import METRICS, MODELS

import yolo_world  # noqa: F401
from yolo_world.evaluation.proposal_recall_metric import PriorProposalRecallMetric
from yolo_world.models.dense_heads.cavp_head import CAVPHead, map_cavp_boxes
from yolo_world.models.detectors.yolo_world_cavp import (
    YOLOWorldCAVPDetector, read_cavp_logits)


def rank_union(main_boxes, cavp_boxes, budget=300):
    """Alternate ranks, main first on ties, then class-agnostic NMS at 0.7.

    Objectness and classification probabilities are not compared. This fixed
    ranking uses no GT and is a diagnostic policy, not an optimal union oracle.
    Input boxes must be clipped input-space boxes, as in CAVP postprocessing.
    """
    boxes = torch.cat((main_boxes, cavp_boxes))
    if not len(boxes):
        return boxes
    order = []
    for rank in range(max(len(main_boxes), len(cavp_boxes))):
        if rank < len(main_boxes):
            order.append(rank)
        if rank < len(cavp_boxes):
            order.append(len(main_boxes) + rank)
    ordered = boxes[torch.tensor(order, device=boxes.device)]
    scores = torch.arange(len(ordered), 0, -1, device=boxes.device,
                          dtype=torch.float32)
    _, keep = nms(ordered.float(), scores, 0.7)
    return ordered[keep[:budget]]


@MODELS.register_module()
class DIORCAVPDiagnosticDetector(YOLOWorldCAVPDetector):
    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        if not rescale:
            raise ValueError('Diagnostics require original-image coordinates.')
        features, text, masks, visual = self._extract_dual_features(
            batch_inputs, batch_data_samples)
        self.bbox_head.num_classes = text.shape[-2]
        cls_scores, bbox_preds = self.bbox_head(features, text, masks)
        metas = [sample.metainfo for sample in batch_data_samples]
        main_results = self.bbox_head.predict_by_feat(
            cls_scores, bbox_preds, batch_img_metas=metas, rescale=True)
        settings = dict(pre_nms_topk=3000, nms_iou_threshold=0.7,
                        max_per_img=300)
        main_scores = [s.float().max(dim=1, keepdim=True).values for s in cls_scores]
        main_props = CAVPHead.predict_proposals(
            self.bbox_head, main_scores, bbox_preds, metas, **settings)
        object_logits, cavp_boxes = self.cavp_head(visual)
        cavp_props = self.cavp_head.predict_proposals(
            object_logits, cavp_boxes, metas, **settings)
        cfg = self.cavp_cfg
        for i, (sample, mp, cp, meta) in enumerate(zip(
                batch_data_samples, main_props, cavp_props, metas)):
            mask = None if masks is None else (masks[i] if masks.ndim > 1 else masks)
            logits = read_cavp_logits(
                cls_scores, cp, i, self.cavp_head.featmap_strides,
                text_mask=mask, mode=cfg['semantic_readout'],
                source_weight=cfg['source_weight'])
            probs = (logits / cfg['score_temperature'] + cfg['score_bias']).sigmoid()
            rescue = self._label_proposals(cls_scores, cp, i, masks, meta, True)
            union = rank_union(mp.bboxes, cp.bboxes)
            def array(tensor):
                return tensor.detach().float().cpu().numpy()
            sample.cavp_diagnostic = dict(
                main=array(map_cavp_boxes(mp.bboxes, meta)),
                cavp=array(map_cavp_boxes(cp.bboxes, meta)),
                union=array(map_cavp_boxes(union, meta)),
                probabilities=array(probs),
                rescue_boxes=array(rescue.bboxes),
                rescue_labels=rescue.labels.detach().cpu().numpy())
        self._check_pred_label_range(main_results, self.bbox_head.num_classes)
        return self.add_pred_to_datasample(batch_data_samples, main_results)


def novel_stage(gt_box, class_index, diag, threshold):
    """One row per GT; best-IoU label plus any-overlapping-proposal success."""
    iou = PriorProposalRecallMetric._iou(np.asarray(gt_box), diag['cavp'])
    main_iou = PriorProposalRecallMetric._iou(np.asarray(gt_box), diag['main'])
    covered = iou >= 0.5
    main_covered = bool((main_iou >= 0.5).any())
    probs = diag['probabilities']
    labels = probs.argmax(1) if len(probs) else np.empty(0, dtype=int)
    correct = covered & (labels == class_index)
    above = covered & (probs[:, class_index] > threshold)
    rescue_iou = PriorProposalRecallMetric._iou(np.asarray(gt_box), diag['rescue_boxes'])
    rescue_hit = bool(((rescue_iou >= .5) & (diag['rescue_labels'] == class_index)).any())
    best = int(iou.argmax()) if len(iou) else None
    return dict(
        main_covered=int(main_covered), cavp_covered=int(covered.any()),
        cavp_only=int(covered.any() and not main_covered),
        main_only=int(main_covered and not covered.any()),
        any_correct_top1=int(correct.any()),
        any_gt_score_above_threshold=int(above.any()),
        any_correct_top1_above_threshold=int((correct & above).any()),
        rescue_correct=int(rescue_hit),
        best_iou=float(iou[best]) if best is not None else 0.,
        best_label=int(labels[best]) if best is not None else -1,
        best_top1_score=float(probs[best].max()) if best is not None else 0.,
        best_gt_score=float(probs[best, class_index]) if best is not None else 0.,
        max_gt_score_on_covered=float(probs[covered, class_index].max()) if covered.any() else 0.)


@METRICS.register_module()
class DIORCAVPDiagnosticMetric(PriorProposalRecallMetric):
    default_prefix = 'diagnostic'

    def __init__(self, classes, output_dir, semantic_threshold, **kwargs):
        super().__init__(**kwargs)
        self.classes = tuple(classes)
        self.output_dir = Path(output_dir)
        self.semantic_threshold = semantic_threshold

    def process(self, data_batch, data_samples):
        for sample in data_samples:
            image_id = int(sample['img_id'])
            diag = sample['cavp_diagnostic']
            counts = Counter()
            rows = []
            for index, (box, group, name) in enumerate(self._ground_truth.get(image_id, [])):
                groups = ('all', group) + ((name,) if group == 'novel' else ())
                for g in groups:
                    counts['gt/' + g] += 1
                for branch in ('main', 'cavp', 'union'):
                    overlaps = self._iou(np.asarray(box), diag[branch])
                    for k in (100, 300):
                        best = float(overlaps[:k].max()) if len(overlaps[:k]) else 0.
                        for t in (.3, .5):
                            if best >= t:
                                for g in groups:
                                    counts[f'{branch}/recall@{k}_iou{int(t*100)}_{g}'] += 1
                # Unlimited union of two top-300 lists: explicitly a 600-box
                # coverage reference, never reported as Recall@300.
                if any((self._iou(np.asarray(box), diag[b]) >= .5).any()
                       for b in ('main', 'cavp')):
                    for g in groups:
                        counts['union600_coverage_iou50/' + g] += 1
                if group == 'novel':
                    row = novel_stage(box, self.classes.index(name), diag,
                                      self.semantic_threshold)
                    best_label = row.pop('best_label')
                    row.update(image_id=image_id, gt_index=index, gt_class=name,
                               best_pred_class=self.classes[best_label] if best_label >= 0 else '')
                    rows.append(row)
            self.results.append(dict(image_id=image_id, counts=dict(counts), rows=rows))

    def compute_metrics(self, results):
        counts = Counter()
        rows = []
        seen = set()
        for result in results:
            if result['image_id'] in seen:
                continue
            seen.add(result['image_id'])
            counts.update(result['counts'])
            rows.extend(result['rows'])
        metrics = {}
        groups = ('all', 'base', 'novel') + self.novel_classes
        for group in groups:
            denom = counts['gt/' + group]
            metrics['gt/' + group] = denom
            for branch in ('main', 'cavp', 'union'):
                for k in (100, 300):
                    for t in (30, 50):
                        key = f'{branch}/recall@{k}_iou{t}_{group}'
                        metrics[key] = counts[key] / max(denom, 1)
            key = 'union600_coverage_iou50/' + group
            metrics[key] = counts[key] / max(denom, 1)
        stages = {}
        flags = ('main_covered', 'cavp_covered', 'cavp_only', 'main_only',
                 'any_correct_top1', 'any_gt_score_above_threshold',
                 'any_correct_top1_above_threshold', 'rescue_correct')
        for name in ('novel',) + self.novel_classes:
            selected = [row for row in rows if name == 'novel' or row['gt_class'] == name]
            extra = [row for row in selected if row['cavp_only']]
            stages[name] = dict(gt=len(selected), **{f: sum(r[f] for r in selected) for f in flags})
            stages[name]['cavp_only_correct_top1'] = sum(r['any_correct_top1'] for r in extra)
            stages[name]['cavp_only_rescue_correct'] = sum(r['rescue_correct'] for r in extra)
        metrics['cavp_only_novel_count'] = stages['novel']['cavp_only']
        metrics['cavp_only_rescue_correct_novel_count'] = stages['novel']['cavp_only_rescue_correct']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report = dict(images=len(seen), metrics=metrics, stages=stages,
                      semantic_threshold=self.semantic_threshold)
        (self.output_dir/'diagnosis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        if rows:
            with (self.output_dir/'novel_gt_diagnosis.csv').open('w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/pretrain/DIORv2.11.py')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--work-dir', default='work_dirs/DIORv2.11/diagnosis_epoch44')
    parser.add_argument('--limit', type=int, default=0, help='Smoke test only: first N images; 0 evaluates all.')
    args = parser.parse_args()
    os.chdir(ROOT)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        parser.error(f'Checkpoint does not exist: {checkpoint}')
    cfg = Config.fromfile(args.config)
    if cfg.model.type != 'YOLOWorldCAVPDetector':
        parser.error('Expected a CAVP config, e.g. DIORv2.11.py.')
    cfg.model.type = 'DIORCAVPDiagnosticDetector'
    cfg.model.cavp_cfg.update(enabled=True, fusion_enabled=False)
    cfg.load_from = str(checkpoint)
    cfg.resume = False
    cfg.launcher = 'none'
    cfg.work_dir = str(Path(args.work_dir).resolve())
    cfg.test_dataloader = copy.deepcopy(cfg.val_dataloader)
    coco = copy.deepcopy(cfg.val_evaluator[0])
    dataset = cfg.test_dataloader.dataset
    classes = tuple(dataset.dataset.metainfo.classes)
    ann_file = Path(coco.ann_file).resolve()
    data_ann = (Path(dataset.dataset.data_root) / dataset.dataset.ann_file).resolve()
    if data_ann != ann_file or set(classes) != set(coco.base_classes) | set(coco.novel_classes) or len(classes) != 20:
        parser.error('Requires matching full DIOR 20-class validation data and annotations.')
    if args.limit < 0:
        parser.error('--limit must be nonnegative.')
    if args.limit:
        dataset.dataset.indices = args.limit
        cfg.test_dataloader.num_workers = 0
        cfg.test_dataloader.persistent_workers = False
    # A subset run must not report AP against the full annotation denominator.
    cfg.test_evaluator = ([] if args.limit else [coco]) + [dict(
        type='DIORCAVPDiagnosticMetric', classes=classes, output_dir=cfg.work_dir,
        ann_file=str(ann_file), base_classes=coco.base_classes, novel_classes=coco.novel_classes,
        semantic_threshold=cfg.model.cavp_cfg.semantic_score_thr)]
    runner = Runner.from_cfg(cfg)
    protocol = dict(checkpoint=str(checkpoint), config=str(Path(args.config).resolve()),
                    checkpoint_loading='Standard Runner load_from, resume=False; no manual ema_state_dict extraction.',
                    ann_file=str(ann_file), annotation_sha256=hashlib.sha256(ann_file.read_bytes()).hexdigest(),
                    limit=args.limit, partial_run=bool(args.limit),
                    candidate_settings=dict(pre_nms_topk=3000, nms_iou=.7, max_per_img=300),
                    union='Alternate main/CAVP ranks, main first; cross-branch class-agnostic NMS 0.7; top 300. No GT selection.',
                    recall='Per-GT max IoU; class agnostic; GT-instance weighted, matching existing CAVP metric.',
                    semantics=dict(cfg.model.cavp_cfg))
    if is_main_process():
        (Path(cfg.work_dir)/'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    metrics = runner.test()
    if is_main_process():
        (Path(cfg.work_dir)/'metrics.json').write_text(json.dumps({k: float(v) for k, v in metrics.items()}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
