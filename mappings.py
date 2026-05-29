import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ====== 你需要维护/确认的输入：各数据集 categories 文件路径 ======
# key: dataset_name（会用于创建 data/LAE-1M/<dataset_name>/dataset_class_to_global.json）
# value: categories.json 的相对路径（相对项目根目录）
SUB_DATASETS: Dict[str, str] = {
    # LAE-COD
    "LAE-COD/AID": r"data/LAE-1M/LAE-COD/AID/AID_categories.json",
    "LAE-COD/EMS": r"data/LAE-1M/LAE-COD/EMS/EMS_categories.json",
    "LAE-COD/NWPU-RESISC45": r"data/LAE-1M/LAE-COD/NWPU-RESISC45/NWPU-RESISC45_categories.json",
    "LAE-COD/SLM": r"data/LAE-1M/LAE-COD/SLM/SLM_categories.json",
    # LAE-FOD
    "LAE-FOD/DIOR": r"data/LAE-1M/LAE-FOD/DIOR/DIOR_categories.json",
    "LAE-FOD/DOTAv2": r"data/LAE-1M/LAE-FOD/DOTAv2/DOTAv2_categories.json",
    "LAE-FOD/FAIR1M": r"data/LAE-1M/LAE-FOD/FAIR1M/FAIR1M_categories.json",
    "LAE-FOD/HRSC2016": r"data/LAE-1M/LAE-FOD/HRSC2016/HRSC2016_categories.json",
    "LAE-FOD/NWPU VHR-10": r"data/LAE-1M/LAE-FOD/NWPU VHR-10/NWPU VHR-10_categories.json",
    "LAE-FOD/Power-Plant": r"data/LAE-1M/LAE-FOD/Power-Plant/Power-Plant_categories.json",
    "LAE-FOD/RSOD": r"data/LAE-1M/LAE-FOD/RSOD/RSOD_categories.json",
    "LAE-FOD/xview": r"data/LAE-1M/LAE-FOD/xview/xview_categories.json",
}

# 输出根目录
OUT_ROOT = Path(r"data/LAE-1M/texts")
OUT_GLOBAL_NAMES = OUT_ROOT / "global_class_names.json"
OUT_FINAL_NAMES = OUT_ROOT / "final_class_names.json"
OUT_GLOBAL_TO_FINAL = OUT_ROOT / "global_to_final_map.json"
OUT_REPORT = OUT_ROOT / "class_mapping_report.json"

# 每个数据集映射文件名（生成到 data/LAE-1M/<dataset_name>/ 下）
OUT_PER_DATASET_MAP_NAME = "dataset_class_to_global.json"

# 可选：如果你已有人写的硬规则文件，可放到这里（dict: global->final）
# 若不存在则不会报错，脚本会输出一个“恒等映射”为主的版本。
OPTIONAL_RULES_PATHS = [
    Path(r"global_to_final_rules.py"),  # 允许你在根目录维护一个 python 规则文件
    Path(r"data/LAE-1M/global_to_final_rules.json"),  # 或者 json 规则文件
]


_norm_space = re.compile(r"\s+")
_norm_keep_alnum = re.compile(r"[^0-9a-z]+")


def normalize_name(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("&", " and ")
    s = s.replace("/", " ")
    s = s.replace("-", " ")
    s = s.replace("_", " ")
    s = _norm_space.sub(" ", s)
    return s


def normalize_match_key(s: str) -> str:
    s = normalize_name(s)
    s = _norm_keep_alnum.sub("", s)
    return s


def _iter_strings(obj: Any) -> Iterable[str]:
    """尽可能从任意 JSON 结构里提取“看起来像类别名”的字符串。"""
    if obj is None:
        return
    if isinstance(obj, str):
        t = obj.strip()
        if t:
            yield t
        return
    if isinstance(obj, list):
        for it in obj:
            yield from _iter_strings(it)
        return
    if isinstance(obj, dict):
        # 常见字段优先
        for k in ("name", "class", "label", "category", "category_name", "class_name", "classname", "title"):
            v = obj.get(k, None)
            if isinstance(v, str) and v.strip():
                yield v.strip()

        # 常见容器字段
        for k in ("classes", "class_names", "classnames", "categories", "category_names", "labels", "names"):
            v = obj.get(k, None)
            if v is not None:
                yield from _iter_strings(v)

        # 兜底：遍历所有 value
        for v in obj.values():
            yield from _iter_strings(v)
        return
    return


def load_class_names_any_format(p: Path) -> List[str]:
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    names: List[str] = []
    for s in _iter_strings(obj):
        t = s.strip()
        if not t:
            continue
        # 过滤掉明显不是类别名的内容：太短、纯数字
        if len(t) <= 1:
            continue
        if t.isdigit():
            continue
        names.append(t)

    # 去重（保留首次出现的写法）
    seen = set()
    out: List[str] = []
    for n in names:
        k = n.strip()
        if k not in seen:
            seen.add(k)
            out.append(n)
    return out


def load_optional_global_to_final_rules(project_root: Path) -> Dict[str, str]:
    """读取可选硬规则：仅支持 json（避免 exec/import 风险）。"""
    for rp in OPTIONAL_RULES_PATHS:
        p = (project_root / rp).resolve()
        if not p.exists():
            continue
        if p.suffix.lower() != ".json":
            continue

        with p.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, dict):
            continue

        rules: Dict[str, str] = {}
        for k, v in raw.items():
            if isinstance(k, str) and isinstance(v, str):
                kn = normalize_name(k)
                vn = normalize_name(v)
                if kn and vn:
                    rules[kn] = vn
        return rules

    return {}


@dataclass
class PerDatasetReport:
    dataset: str
    categories_path: str
    extracted_total: int
    extracted_unique: int
    unmapped_raw_count: int


def main() -> None:
    project_root = Path(os.getcwd()).resolve()

    # 1) 读取各数据集原始类名
    dataset_raw_names: Dict[str, List[str]] = {}
    all_raw: List[Tuple[str, str]] = []  # (dataset, raw_name)

    for dataset_name, rel_path in SUB_DATASETS.items():
        p = (project_root / rel_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"找不到 categories 文件: {p}")

        names = load_class_names_any_format(p)
        dataset_raw_names[dataset_name] = names
        for n in names:
            all_raw.append((dataset_name, n))

    # 2) 构建 global canonical：强归一化 key 去重，canonical 用温和归一化后的名字
    canonical_by_key: Dict[str, str] = {}
    for _, raw_name in all_raw:
        rk = normalize_match_key(raw_name)
        if not rk:
            continue
        if rk not in canonical_by_key:
            canonical_by_key[rk] = normalize_name(raw_name)

    global_canonicals = sorted(set(canonical_by_key.values()))

    # 3) 写 global_class_names.json (List[List[str]])
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with OUT_GLOBAL_NAMES.open("w", encoding="utf-8") as f:
        json.dump([[c] for c in global_canonicals], f, ensure_ascii=False, indent=2)

    # 4) 生成每个数据集的 dataset_class_to_global.json
    per_report: List[PerDatasetReport] = []
    for dataset_name, names in dataset_raw_names.items():
        per_map: Dict[str, str] = {}
        unmapped_raw = 0

        for raw_name in names:
            rk = normalize_match_key(raw_name)
            if not rk or rk not in canonical_by_key:
                unmapped_raw += 1
                continue
            global_name = canonical_by_key[rk]
            # key 用“原始写法”和“normalize 后写法”都写入，提高召回（不影响原标签）
            per_map[raw_name] = global_name
            per_map[normalize_name(raw_name)] = global_name

        out_dir = (project_root / "data" / "LAE-1M" / dataset_name).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / OUT_PER_DATASET_MAP_NAME
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(dict(sorted(per_map.items())), f, ensure_ascii=False, indent=2)

        per_report.append(
            PerDatasetReport(
                dataset=dataset_name,
                categories_path=str((project_root / SUB_DATASETS[dataset_name]).resolve()),
                extracted_total=len(names),
                extracted_unique=len(set(names)),
                unmapped_raw_count=unmapped_raw,
            )
        )

    # 5) 阶段2：global -> final
    # 5.1 先默认恒等映射（每个 global 指向自己）
    global_to_final: Dict[str, str] = {c: c for c in global_canonicals}

    # 5.2 应用硬规则（如果存在）
    rules = load_optional_global_to_final_rules(project_root)
    # 只在 key 和 value 都能落在 global 集合时应用；否则忽略（避免写错）
    global_set = set(global_canonicals)
    applied_rules = 0
    skipped_rules = 0
    for g, f in rules.items():
        if g in global_set and f in global_set:
            if global_to_final[g] != f:
                global_to_final[g] = f
            applied_rules += 1
        else:
            skipped_rules += 1

    # 5.3 写 global_to_final_map.json
    with OUT_GLOBAL_TO_FINAL.open("w", encoding="utf-8") as f:
        json.dump(dict(sorted(global_to_final.items())), f, ensure_ascii=False, indent=2)

    # 5.4 写 final_class_names.json（final 类集合）
    final_set = sorted(set(global_to_final.values()))
    with OUT_FINAL_NAMES.open("w", encoding="utf-8") as f:
        json.dump([[c] for c in final_set], f, ensure_ascii=False, indent=2)

    # 6) 报告
    report = {
        "outputs": {
            "global_class_names": str(OUT_GLOBAL_NAMES),
            "final_class_names": str(OUT_FINAL_NAMES),
            "global_to_final_map": str(OUT_GLOBAL_TO_FINAL),
            "per_dataset_map_filename": OUT_PER_DATASET_MAP_NAME,
        },
        "stats": {
            "num_sub_datasets": len(SUB_DATASETS),
            "num_global_classes": len(global_canonicals),
            "num_final_classes": len(final_set),
            "hard_rules_loaded": len(rules),
            "hard_rules_applied": applied_rules,
            "hard_rules_skipped": skipped_rules,
        },
        "per_dataset": [r.__dict__ for r in per_report],
    }
    with OUT_REPORT.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"OK: 写出 `global`: {OUT_GLOBAL_NAMES}  (n={len(global_canonicals)})")
    print(f"OK: 写出 `final`:  {OUT_FINAL_NAMES}  (n={len(final_set)})")
    print(f"OK: 写出 `g2f`:    {OUT_GLOBAL_TO_FINAL}  (rules_loaded={len(rules)}, applied={applied_rules}, skipped={skipped_rules})")
    print(f"OK: 写出 report:   {OUT_REPORT}")
    print("OK: 各数据集 `dataset_class_to_global.json` 已写入对应 `data/LAE-1M/<dataset_name>/` 目录")


if __name__ == "__main__":
    main()
