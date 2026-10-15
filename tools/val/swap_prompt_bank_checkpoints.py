import argparse
import copy
from pathlib import Path

import torch


PROMPT_MARKER = (
    'backbone.text_model.prompt_bank.'
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--best',
        required=True
    )

    parser.add_argument(
        '--final',
        required=True
    )

    parser.add_argument(
        '--output-dir',
        required=True
    )

    return parser.parse_args()


def get_state_containers(ckpt):
    """
    返回 checkpoint 中所有可能真正存参数的 dict。

    支持：
    - state_dict
    - ema_state_dict
    - 直接就是 tensor state dict
    """
    result = {}

    for key in [
        'state_dict',
        'ema_state_dict'
    ]:
        if (
            key in ckpt
            and isinstance(ckpt[key], dict)
        ):
            result[key] = ckpt[key]

    # 某些 checkpoint 本身就是 state_dict
    if not result:
        tensor_count = sum(
            torch.is_tensor(v)
            for v in ckpt.values()
        )

        if tensor_count > 0:
            result['__root__'] = ckpt

    if not result:
        raise RuntimeError(
            '没有找到 state_dict / ema_state_dict。'
        )

    return result


def canonical_prompt_key(key):
    pos = key.find(PROMPT_MARKER)

    if pos < 0:
        return None

    return key[pos:]


def make_prompt_map(state):
    out = {}

    for k, v in state.items():
        ck = canonical_prompt_key(k)

        if ck is not None:
            out[ck] = (
                k,
                v
            )

    return out


def swap_one_container(
    dst_state,
    src_state,
):
    dst_map = make_prompt_map(dst_state)
    src_map = make_prompt_map(src_state)

    common = sorted(
        set(dst_map)
        & set(src_map)
    )

    if not common:
        raise RuntimeError(
            '没有找到共同的 Prompt Bank 参数。\n'
            f'需要包含: {PROMPT_MARKER}'
        )

    changed = []

    for canonical in common:

        dst_key, dst_val = dst_map[
            canonical
        ]

        src_key, src_val = src_map[
            canonical
        ]

        if dst_val.shape != src_val.shape:
            raise RuntimeError(
                f'shape 不一致:\n'
                f'{dst_key}: {dst_val.shape}\n'
                f'{src_key}: {src_val.shape}'
            )

        dst_state[dst_key] = (
            src_val.detach()
            .clone()
            .to(
                device=dst_val.device,
                dtype=dst_val.dtype
            )
        )

        changed.append(
            canonical
        )

    return changed


def create_hybrid(
    destination_ckpt,
    source_ckpt,
):
    out = copy.deepcopy(
        destination_ckpt
    )

    dst_containers = get_state_containers(
        out
    )

    src_containers = get_state_containers(
        source_ckpt
    )

    report = {}

    for name, dst_state in dst_containers.items():

        # 同名优先
        if name in src_containers:
            src_state = src_containers[
                name
            ]

        # 找不到同名时退到 state_dict
        elif 'state_dict' in src_containers:
            src_state = src_containers[
                'state_dict'
            ]

        # 最后任取一个
        else:
            src_state = next(
                iter(
                    src_containers.values()
                )
            )

        changed = swap_one_container(
            dst_state,
            src_state
        )

        report[name] = len(changed)

    return out, report


def main():
    args = parse_args()

    out_dir = Path(
        args.output_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print('Loading best...')
    best = torch.load(
        args.best,
        map_location='cpu'
    )

    print('Loading final...')
    final = torch.load(
        args.final,
        map_location='cpu'
    )

    # Final visual/main params
    # + Best Prompt Bank
    final_best_pb, report1 = create_hybrid(
        final,
        best
    )

    path1 = (
        out_dir
        / 'final_with_best_prompt_bank.pth'
    )

    torch.save(
        final_best_pb,
        path1
    )

    # Best visual/main params
    # + Final Prompt Bank
    best_final_pb, report2 = create_hybrid(
        best,
        final
    )

    path2 = (
        out_dir
        / 'best_with_final_prompt_bank.pth'
    )

    torch.save(
        best_final_pb,
        path2
    )

    print('\n===== DONE =====')

    print(
        'final + best PB:',
        path1
    )
    print(
        'replaced:',
        report1
    )

    print(
        '\nbest + final PB:',
        path2
    )
    print(
        'replaced:',
        report2
    )


if __name__ == '__main__':
    main()