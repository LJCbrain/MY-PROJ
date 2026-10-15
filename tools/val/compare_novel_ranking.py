import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CLASSES = [
    'airplane',
    'airport',
    'baseballfield',
    'basketballcourt',
    'bridge',
    'chimney',
    'dam',
    'Expressway-Service-area',
    'Expressway-toll-station',
    'golffield',
    'groundtrackfield',
    'harbor',
    'overpass',
    'ship',
    'stadium',
    'storagetank',
    'tenniscourt',
    'trainstation',
    'vehicle',
    'windmill',
]

NOVEL = [
    'airport',
    'basketballcourt',
    'groundtrackfield',
    'windmill',
]

BASE = [x for x in CLASSES if x not in NOVEL]


ALIASES = {
    'gt_class': [
        'gt_class_name', 'gt_class', 'class_name',
        'gt_name', 'category_name', 'gt_label', 'label'
    ],
    'rank': [
        'true_class_rank', 'gt_class_rank', 'class_rank',
        'semantic_rank', 'text_rank', 'rank'
    ],
    'pred_class': [
        'pred_class_name', 'pred_class', 'top1_class',
        'best_class', 'pred_label', 'top1_label'
    ],
    'true_score': [
        'true_class_score', 'gt_class_score',
        'true_score', 'novel_score', 'gt_score'
    ],
    'best_base_score': [
        'best_base_score', 'max_base_score',
        'base_max_score', 'top_base_score'
    ],
    'iou': [
        'matched_iou', 'best_iou', 'proposal_iou',
        'match_iou', 'iou'
    ],
}


def find_column(df, names):
    # 先精确匹配
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]

    # 再模糊匹配
    for col in df.columns:
        lc = col.lower()
        for name in names:
            if name.lower() in lc:
                return col
    return None


def normalize_class(x):
    if pd.isna(x):
        return None

    # 数字标签
    try:
        if isinstance(x, (int, np.integer)):
            idx = int(x)
            if 0 <= idx < len(CLASSES):
                return CLASSES[idx]

        sx = str(x).strip()

        if sx.isdigit():
            idx = int(sx)
            if 0 <= idx < len(CLASSES):
                return CLASSES[idx]
    except Exception:
        pass

    sx = str(x).strip()

    # 大小写归一
    mapping = {c.lower(): c for c in CLASSES}
    return mapping.get(sx.lower(), sx)


def analyze(csv_path):
    df = pd.read_csv(csv_path)

    cols = {
        k: find_column(df, v)
        for k, v in ALIASES.items()
    }

    print(f'\nFILE: {csv_path}')
    print('Columns:')
    print(list(df.columns))
    print('\nDetected:')
    print(cols)

    if cols['gt_class'] is None:
        raise RuntimeError(
            '找不到 GT 类别列。请把上面打印的 CSV 列名发给我。'
        )

    df['_gt_class'] = df[cols['gt_class']].apply(normalize_class)
    df = df[df['_gt_class'].isin(NOVEL)].copy()

    # 如果有 IoU 列，只保留已经定位成功的 GT
    if cols['iou'] is not None:
        df['_iou'] = pd.to_numeric(
            df[cols['iou']], errors='coerce')
        matched = df[df['_iou'] >= 0.5].copy()
    else:
        matched = df.copy()

    out = {
        'num_novel_gt_rows': int(len(df)),
        'num_matched_rows': int(len(matched)),
        'matched_fraction': (
            float(len(matched) / len(df))
            if len(df) else 0.0
        ),
        'columns': cols,
        'overall': {},
        'per_class': {},
    }

    rank_col = cols['rank']

    if rank_col is None:
        raise RuntimeError(
            'CSV 中没有检测到类别 rank 列。'
            '请把 gt_diagnostics.csv 的表头发给我。'
        )

    matched['_rank'] = pd.to_numeric(
        matched[rank_col], errors='coerce'
    )

    def rank_stats(x):
        x = x.dropna()

        if len(x) == 0:
            return {}

        return {
            'n': int(len(x)),
            'top1': float((x <= 1).mean()),
            'top3': float((x <= 3).mean()),
            'top5': float((x <= 5).mean()),
            'top10': float((x <= 10).mean()),
            'mean_rank': float(x.mean()),
            'median_rank': float(x.median()),
        }

    out['overall'].update(rank_stats(matched['_rank']))

    for cls in NOVEL:
        sub = matched[matched['_gt_class'] == cls]
        out['per_class'][cls] = rank_stats(sub['_rank'])

    # true Novel score
    if cols['true_score'] is not None:
        matched['_true_score'] = pd.to_numeric(
            matched[cols['true_score']], errors='coerce')

        out['overall']['true_score_mean'] = float(
            matched['_true_score'].mean()
        )

        for cls in NOVEL:
            sub = matched[matched['_gt_class'] == cls]
            out['per_class'][cls]['true_score_mean'] = float(
                sub['_true_score'].mean()
            )

    # best Base score + margin
    if (
        cols['true_score'] is not None
        and cols['best_base_score'] is not None
    ):
        matched['_best_base_score'] = pd.to_numeric(
            matched[cols['best_base_score']],
            errors='coerce'
        )

        matched['_margin'] = (
            matched['_true_score']
            - matched['_best_base_score']
        )

        out['overall']['best_base_score_mean'] = float(
            matched['_best_base_score'].mean()
        )
        out['overall']['novel_minus_base_margin'] = float(
            matched['_margin'].mean()
        )

        for cls in NOVEL:
            sub = matched[matched['_gt_class'] == cls]

            out['per_class'][cls]['best_base_score_mean'] = float(
                sub['_best_base_score'].mean()
            )
            out['per_class'][cls]['novel_minus_base_margin'] = float(
                sub['_margin'].mean()
            )

    # Novel -> Base confusion
    confusion = None

    if cols['pred_class'] is not None:
        matched['_pred_class'] = matched[
            cols['pred_class']
        ].apply(normalize_class)

        confusion = pd.crosstab(
            matched['_gt_class'],
            matched['_pred_class'],
            normalize='index'
        )

        # 只保留 Base 列
        base_cols = [
            c for c in confusion.columns
            if c in BASE
        ]

        if base_cols:
            confusion = confusion[base_cols]

    return out, confusion


def flatten_summary(name, d):
    rows = []

    overall = d['overall'].copy()
    overall['group'] = 'ALL_NOVEL'
    overall['checkpoint'] = name
    rows.append(overall)

    for cls, vals in d['per_class'].items():
        r = vals.copy()
        r['group'] = cls
        r['checkpoint'] = name
        rows.append(r)

    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--best-csv', required=True)
    parser.add_argument('--final-csv', required=True)
    parser.add_argument('--output-dir', required=True)

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best, best_conf = analyze(args.best_csv)
    final, final_conf = analyze(args.final_csv)

    with open(out_dir / 'best_summary.json', 'w') as f:
        json.dump(best, f, indent=2)

    with open(out_dir / 'final_summary.json', 'w') as f:
        json.dump(final, f, indent=2)

    rows = []
    rows += flatten_summary('best', best)
    rows += flatten_summary('final', final)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(
        out_dir / 'ranking_comparison.csv',
        index=False
    )

    if best_conf is not None:
        best_conf.to_csv(
            out_dir / 'confusion_best.csv'
        )

    if final_conf is not None:
        final_conf.to_csv(
            out_dir / 'confusion_final.csv'
        )

    print('\n===== RANKING COMPARISON =====')
    print(summary_df.to_string(index=False))

    print(
        f'\nSaved to: {out_dir}'
    )


if __name__ == '__main__':
    main()