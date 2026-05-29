import os
import random
from pathlib import Path

import mmcv
import numpy as np
from mmengine.fileio import load
from mmengine.structures import InstanceData

from mmdet.structures import DetDataSample
from mmdet.visualization import DetLocalVisualizer


def to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(x, np.ndarray):
        return x
    return None


def main():
    pkl_path = Path(r"run/result.pkl")
    out_dir = Path(r"run/vis_50")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load(str(pkl_path))
    n = min(50, len(results))
    random.seed(0)
    idxs = random.sample(range(len(results)), k=n)

    visualizer = DetLocalVisualizer()
    visualizer.dataset_meta = {}  # 不依赖固定 classes，名称用 texts 动态生成

    for j, i in enumerate(idxs):
        r = results[i]
        img_path = r.get("img_path", None)
        if not img_path or not os.path.exists(img_path):
            continue

        img = mmcv.imread(img_path, channel_order="rgb")

        pi = r.get("pred_instances", {})
        bboxes = to_numpy(pi.get("bboxes", None))
        scores = to_numpy(pi.get("scores", None))
        labels = to_numpy(pi.get("labels", None))

        texts = r.get("texts", None)
        if isinstance(texts, (list, tuple)) and all(isinstance(t, str) for t in texts):
            class_names = list(texts)
        else:
            class_names = None

        ds = DetDataSample()
        inst = InstanceData()

        if bboxes is not None and scores is not None and labels is not None and bboxes.shape[0] > 0:
            inst.bboxes = bboxes
            inst.scores = scores
            inst.labels = labels.astype(np.int64)
            ds.pred_instances = inst

            if class_names is not None:
                # 用 texts 当作类别名，保证 label id 能映射到字符串
                visualizer.dataset_meta = {"classes": tuple(class_names)}
        else:
            # 没有预测就保存原图（用于确认流程跑通）
            ds.pred_instances = InstanceData()

        save_name = out_dir / f"{j:03d}_idx{i:05d}.jpg"
        visualizer.add_datasample(
            name=save_name.stem,
            image=img,
            data_sample=ds,
            draw_gt=False,
            draw_pred=True,
            show=False,
            out_file=str(save_name),
            pred_score_thr=0.001,
        )

    print(f"saved_to: {out_dir}")


if __name__ == "__main__":
    main()
