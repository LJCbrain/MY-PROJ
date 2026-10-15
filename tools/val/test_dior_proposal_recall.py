"""Evaluate a non-CAVP DIOR checkpoint with the CAVP proposal protocol.

The original detector's AP/HM output is preserved. Two candidate rankings
are measured separately, using the original regression head in both cases:
main_proposal uses the maximum classification score over the full vocabulary;
regional_proposal uses the existing Regional reliability map. Neither is
CAVP objectness. Both share CAVP's decoding, clipping, filtering and NMS code.
"""

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.dist import is_main_process
from mmengine.evaluator import DumpResults
from mmengine.runner import Runner
from mmyolo.registry import METRICS, MODELS

import yolo_world  # noqa: F401
from yolo_world.evaluation.proposal_recall_metric import PriorProposalRecallMetric
from yolo_world.models.dense_heads.cavp_head import CAVPHead, map_cavp_boxes
from yolo_world.models.detectors.yolo_world import YOLOWorldDetector


PROPOSAL_SETTINGS = dict(
    pre_nms_topk=3000, nms_iou_threshold=0.7, max_per_img=300)


@MODELS.register_module()
class DIORProposalEvalDetector(YOLOWorldDetector):
    """Evaluation-only adapter with exactly the original parameter layout."""

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        features, texts, masks = self.extract_feat(batch_inputs, batch_data_samples)
        self.bbox_head.num_classes = texts.shape[-2]
        cls_scores, bbox_preds = self.bbox_head(features, texts, masks)
        metas = [sample.metainfo for sample in batch_data_samples]
        original = self.bbox_head.predict_by_feat(
            cls_scores, bbox_preds, batch_img_metas=metas, rescale=rescale)

        # Do not use already thresholded/multilabel/NMSed final detections:
        # every dense location supplies one original regressed candidate.
        main_scores = [score.float().max(dim=1, keepdim=True).values
                       for score in cls_scores]
        main_proposals = CAVPHead.predict_proposals(
            self.bbox_head, main_scores, bbox_preds, metas, **PROPOSAL_SETTINGS)

        priors = self.neck.get_proposal_prior_maps()
        if not priors or len(priors) != len(bbox_preds):
            raise RuntimeError('Regional proposal maps are unavailable.')
        regional_scores = []
        for prior, boxes in zip(priors, bbox_preds):
            prior = prior.detach().float()
            if prior.shape[-2:] != boxes.shape[-2:]:
                prior = F.interpolate(prior, size=boxes.shape[-2:],
                                      mode='bilinear', align_corners=False)
            regional_scores.append(prior.clamp(0, 1))
        regional_proposals = CAVPHead.predict_proposals(
            self.bbox_head, regional_scores, bbox_preds, metas,
            scores_are_logits=False, **PROPOSAL_SETTINGS)

        for sample, main, regional, meta in zip(
                batch_data_samples, main_proposals, regional_proposals, metas):
            main.bboxes = map_cavp_boxes(main.bboxes, meta, rescale)
            regional.bboxes = map_cavp_boxes(regional.bboxes, meta, rescale)
            sample.main_proposal_instances = main
            sample.regional_proposal_instances = regional
        self._check_pred_label_range(original, self.bbox_head.num_classes)
        return self.add_pred_to_datasample(batch_data_samples, original)


@METRICS.register_module()
class DIORFieldProposalRecallMetric(PriorProposalRecallMetric):
    """Select an explicit candidate field; keep the CAVP metric unchanged."""

    def __init__(self, proposal_field, **kwargs):
        super().__init__(**kwargs)
        self.proposal_field = proposal_field

    def process(self, data_batch, data_samples):
        selected = []
        for sample in data_samples:
            if isinstance(sample, dict):
                image_id = (sample['img_id'] if 'img_id' in sample else
                            sample['metainfo']['img_id'])
                proposals = sample.get(self.proposal_field)
            else:
                image_id = sample.metainfo['img_id']
                proposals = getattr(sample, self.proposal_field, None)
            if proposals is None:
                raise RuntimeError(f'Missing proposal field: {self.proposal_field}')
            selected.append(dict(img_id=image_id, proposal_instances=proposals))
        super().process(data_batch, selected)


def build_eval_config(config_path, checkpoint, work_dir):
    cfg = Config.fromfile(str(config_path))
    if cfg.model.type != 'YOLOWorldDetector':
        raise ValueError('Use the original non-CAVP YOLOWorldDetector config.')
    cfg.model.type = 'DIORProposalEvalDetector'
    cfg.load_from = str(checkpoint)
    cfg.resume = False
    cfg.work_dir = str(work_dir)

    # Evaluate the same complete GZSD validation set used by the CAVP log.
    cfg.test_dataloader = copy.deepcopy(cfg.val_dataloader)
    coco = copy.deepcopy(cfg.val_evaluator)
    if not isinstance(coco, dict) or coco['type'] != 'DIORCocoMetric':
        raise ValueError('Expected the DIOR GZSD validation AP/HM evaluator.')
    dataset = cfg.test_dataloader.dataset
    inner = dataset.dataset
    data_annotation = (Path(inner.data_root) / inner.ann_file).resolve()
    metric_annotation = Path(coco['ann_file']).resolve()
    if data_annotation != metric_annotation:
        raise ValueError('Dataloader and evaluator must use the same annotations.')
    if (set(inner.metainfo.classes) !=
            set(coco['base_classes']) | set(coco['novel_classes']) or
            len(inner.metainfo.classes) != 20):
        raise ValueError('Use the full DIOR 20-class GZSD vocabulary.')
    cfg.test_evaluator = [coco]
    for prefix in ('main_proposal', 'regional_proposal'):
        cfg.test_evaluator.append(dict(
            type='DIORFieldProposalRecallMetric',
            proposal_field=f'{prefix}_instances', prefix=prefix,
            ann_file=coco['ann_file'], base_classes=coco['base_classes'],
            novel_classes=coco['novel_classes'], proposal_nums=(100, 300),
            iou_thresholds=(0.3, 0.5), report_classwise=True))
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/pretrain/DIORv2.10.py')
    parser.add_argument('--checkpoint', default=(
        'work_dirs/DIORv2.6/2.10/'
        'best_coco_bbox_mAP_50_HM_epoch_36.pth'))
    parser.add_argument('--work-dir',
                        default='work_dirs/DIORv2.10_recall_epoch36')
    parser.add_argument('--out', help='Optional .pkl prediction dump path.')
    parser.add_argument('--launcher', default='none',
                        choices=('none', 'pytorch', 'slurm', 'mpi'))
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    os.chdir(ROOT)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        parser.error(f'Checkpoint does not exist: {checkpoint}')
    if args.out and Path(args.out).suffix not in ('.pkl', '.pickle'):
        parser.error('--out must be a .pkl or .pickle path.')
    work_dir = Path(args.work_dir).resolve()
    cfg = build_eval_config(args.config, checkpoint, work_dir)
    cfg.launcher = args.launcher
    annotation = Path(cfg.test_evaluator[0]['ann_file'])
    payload = json.loads(annotation.read_text(encoding='utf-8'))
    protocol = dict(
        config=str(Path(args.config).resolve()), checkpoint=str(checkpoint),
        annotation=str(annotation.resolve()),
        annotation_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest(),
        images=len(payload['images']), annotations=len(payload['annotations']),
        vocabulary='DIOR GZSD full 20 classes',
        candidate_budget_per_image=[100, 300], iou_thresholds=[0.3, 0.5],
        matching='Per-GT maximum IoU; no class check; not one-to-one matching',
        averaging='GT-instance weighted within All/Base/Novel',
        candidate_postprocessing=PROPOSAL_SETTINGS,
        clipping='Input clipping before class-agnostic NMS; original-space after',
        proposal_score_threshold=None,
        ranking=dict(main_proposal='max over dense classification logits',
                     regional_proposal='Regional reliability probability'),
        shared_implementation={str(p.relative_to(ROOT)):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                ROOT / 'yolo_world/models/dense_heads/cavp_head.py',
                ROOT / 'yolo_world/evaluation/proposal_recall_metric.py')})
    runner = Runner.from_cfg(cfg)
    if args.out:
        out = Path(args.out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        runner.test_evaluator.metrics.append(DumpResults(out_file_path=str(out)))
    if is_main_process():
        (work_dir / 'proposal_protocol.json').write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
        runner.logger.info('Candidate recall uses the exact CAVP postprocessing '
                           'and GT metric. Rankings: main=max class score; '
                           'regional=Regional prior. Original AP/HM unchanged.')
    results = runner.test()
    if is_main_process():
        destination = work_dir / 'metrics.json'
        destination.write_text(json.dumps(
            {key: float(value) for key, value in results.items()}, indent=2),
            encoding='utf-8')
        runner.logger.info('Saved metrics to %s', destination)


if __name__ == '__main__':
    main()
