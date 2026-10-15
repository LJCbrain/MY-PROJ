"""Compare three matched seeds, reading full-validation diagnostic summaries."""
import argparse
import json
from pathlib import Path
import statistics


def load_runs(entries):
    runs = {}
    for entry in entries:
        seed, path = entry.split('=', 1)
        if seed in runs:
            raise ValueError('Duplicate seed: ' + seed)
        report = json.loads(Path(path).read_text(encoding='utf-8'))
        if not report.get('full_validation', False):
            raise ValueError('Only complete validation reports can be compared: ' + path)
        # Accept metrics with or without the evaluator prefix.
        runs[seed] = {k.split('bbox_', 1)[-1]: float(v)
                      for k, v in report['metrics'].items() if 'bbox_' in k}
    if len(runs) != 3:
        raise ValueError('Exactly three distinct seeds are required.')
    return runs


def compare(control, candidate):
    if set(control) != set(candidate):
        raise ValueError('Control and candidate must use the same seeds.')
    seeds = sorted(control)
    result = dict(seeds=seeds, units='percentage points', metrics={})
    keys = set.intersection(*(set(run) for run in [*control.values(), *candidate.values()]))
    for key in sorted(keys):
        a, b = ([100 * runs[s][key] for s in seeds] for runs in (control, candidate))
        delta = [y - x for x, y in zip(a, b)]
        result['metrics'][key] = dict(
            control_mean=statistics.mean(a), control_std=statistics.stdev(a),
            candidate_mean=statistics.mean(b), candidate_std=statistics.stdev(b),
            paired_delta_mean=statistics.mean(delta), paired_delta_std=statistics.stdev(delta),
            paired_deltas=delta)
    diff = lambda k: result['metrics'][k]['paired_delta_mean']
    result['passes_default_promotion'] = (
        diff('mAP_50_novel') >= .5 - 1e-10
        and diff('mAP_50_HM') > 0 and diff('mAP_50_base') >= -1.0 - 1e-10)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', nargs=3, required=True, metavar='SEED=SUMMARY_JSON')
    parser.add_argument('--candidate', nargs=3, required=True, metavar='SEED=SUMMARY_JSON')
    parser.add_argument('--output', default='work_dirs/ovc_seed_comparison.json')
    args = parser.parse_args()
    result = compare(load_runs(args.control), load_runs(args.candidate))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
