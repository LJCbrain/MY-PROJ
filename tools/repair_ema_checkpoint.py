#!/usr/bin/env python
"""Repair non-floating EMA buffers in an MMEngine checkpoint.

MMEngine's EMAHook saves averaged deployment weights in ``state_dict`` and
the online training model in ``ema_state_dict``. Some YOLO EMA versions do
not update integer or boolean buffers after initialization. This tool keeps
all averaged floating-point weights unchanged and copies only matching
non-floating buffers from the online model.
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

import torch


def load_checkpoint(path: Path) -> MutableMapping[str, Any]:
    """Load a trusted training checkpoint on CPU."""
    try:
        return torch.load(str(path), map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location='cpu')


def _online_key_for(deployment_key: str,
                    online_state: Mapping[str, Any]) -> Optional[str]:
    candidates = (deployment_key, 'module.' + deployment_key)
    if deployment_key.startswith('module.'):
        candidates += (deployment_key[len('module.'):], )
    for candidate in candidates:
        if candidate in online_state:
            return candidate
    return None


def repair_ema_nonfloating_buffers(
        checkpoint: MutableMapping[str, Any],
        strict: bool = True) -> Dict[str, Any]:
    """Synchronize deployment non-floating buffers from the online model."""
    if 'state_dict' not in checkpoint:
        raise KeyError('Checkpoint has no state_dict.')
    if 'ema_state_dict' not in checkpoint:
        raise KeyError('Checkpoint has no ema_state_dict.')

    deployment_state = checkpoint['state_dict']
    online_state = checkpoint['ema_state_dict']
    copied: List[Tuple[str, str]] = []
    changed: List[Tuple[str, str]] = []
    missing: List[str] = []
    incompatible: List[str] = []

    for deployment_key, deployment_value in deployment_state.items():
        if (not isinstance(deployment_value, torch.Tensor)
                or deployment_value.dtype.is_floating_point
                or deployment_value.is_complex()):
            continue

        online_key = _online_key_for(deployment_key, online_state)
        if online_key is None:
            missing.append(deployment_key)
            continue
        online_value = online_state[online_key]
        if (not isinstance(online_value, torch.Tensor)
                or online_value.shape != deployment_value.shape
                or online_value.dtype != deployment_value.dtype):
            incompatible.append(deployment_key)
            continue

        if not torch.equal(deployment_value, online_value):
            changed.append((deployment_key, online_key))
        deployment_state[deployment_key] = online_value.detach().clone()
        copied.append((deployment_key, online_key))

    if strict and (missing or incompatible):
        details = []
        if missing:
            details.append('missing=' + ', '.join(missing))
        if incompatible:
            details.append('incompatible=' + ', '.join(incompatible))
        raise RuntimeError('Cannot repair every non-floating buffer: ' +
                           '; '.join(details))

    return dict(
        copied=copied,
        changed=changed,
        missing=missing,
        incompatible=incompatible,
    )


def verify_saved_checkpoint(path: Path,
                            expected: Mapping[str, Any]) -> int:
    """Reload a saved checkpoint and verify its deployment state exactly."""
    saved = load_checkpoint(path)
    expected_state = expected['state_dict']
    saved_state = saved.get('state_dict', {})
    if expected_state.keys() != saved_state.keys():
        raise RuntimeError('Saved state_dict keys differ from repaired state.')

    checked = 0
    for key, expected_value in expected_state.items():
        saved_value = saved_state[key]
        if isinstance(expected_value, torch.Tensor):
            if (not isinstance(saved_value, torch.Tensor)
                    or saved_value.dtype != expected_value.dtype
                    or saved_value.shape != expected_value.shape
                    or not torch.equal(saved_value, expected_value)):
                raise RuntimeError(
                    'Saved tensor does not match repaired state: ' + key)
            checked += 1
    return checked


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(input_path.stem +
                                '_ema_nonfloating_fixed' +
                                input_path.suffix)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('-o', '--output', type=Path)
    parser.add_argument(
        '--force',
        action='store_true',
        help='Replace an existing output file. The input is never overwritten.')
    parser.add_argument(
        '--skip-verify',
        action='store_true',
        help='Skip reloading and checking the saved checkpoint.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.checkpoint.resolve()
    output_path = (args.output or default_output_path(input_path)).resolve()
    if input_path == output_path:
        raise ValueError('The output path must differ from the input path.')
    if not input_path.is_file():
        raise FileNotFoundError(str(input_path))
    if output_path.exists() and not args.force:
        raise FileExistsError(
            '{} already exists; use --force to replace it.'.format(
                output_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(input_path)
    report = repair_ema_nonfloating_buffers(checkpoint)

    temporary_path = output_path.with_name(output_path.name + '.tmp')
    if temporary_path.exists():
        raise FileExistsError(
            'Temporary output already exists: {}'.format(temporary_path))
    try:
        torch.save(checkpoint, str(temporary_path))
        os.replace(str(temporary_path), str(output_path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    checked = 0
    if not args.skip_verify:
        checked = verify_saved_checkpoint(output_path, checkpoint)

    print('Input: {}'.format(input_path))
    print('Output: {}'.format(output_path))
    print('Copied non-floating buffers: {}'.format(len(report['copied'])))
    print('Buffers whose values changed: {}'.format(len(report['changed'])))
    print('Verified deployment tensors: {}'.format(checked))
    for deployment_key, online_key in report['changed']:
        if deployment_key.endswith(('_gamma_step', '_caqr_beta_step')):
            value = checkpoint['state_dict'][deployment_key].item()
            print('Critical buffer: {} <- {} = {}'.format(
                deployment_key, online_key, value))


if __name__ == '__main__':
    main()
