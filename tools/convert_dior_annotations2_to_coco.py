"""Convert the split DIOR Pascal VOC annotations in Annotations2 to COCO JSON."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def read_classes(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def convert_split(
    split: str,
    xml_dir: Path,
    image_dir: Path,
    classes: list[str],
    output_path: Path,
) -> tuple[int, int]:
    class_to_id = {name: idx for idx, name in enumerate(classes)}
    images: list[dict] = []
    annotations: list[dict] = []

    for image_id, xml_path in enumerate(sorted(xml_dir.glob("*.xml"))):
        root = ET.parse(xml_path).getroot()
        filename = root.findtext("filename") or f"{xml_path.stem}.jpg"
        size = root.find("size")
        width = int(size.findtext("width", "800")) if size is not None else 800
        height = int(size.findtext("height", "800")) if size is not None else 800
        image_path = image_dir / filename
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image for {split}: {image_path}")

        images.append({
            "file_name": filename,
            "id": image_id,
            "width": width,
            "height": height,
        })

        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            if name not in class_to_id:
                raise ValueError(f"Unknown DIOR class {name!r} in {xml_path}")
            box = obj.find("bndbox")
            if box is None:
                raise ValueError(f"Missing bndbox in {xml_path}")
            xmin = float(box.findtext("xmin", "0"))
            ymin = float(box.findtext("ymin", "0"))
            xmax = float(box.findtext("xmax", "0"))
            ymax = float(box.findtext("ymax", "0"))
            bbox_width = max(0.0, xmax - xmin)
            bbox_height = max(0.0, ymax - ymin)
            annotations.append({
                "id": len(annotations),
                "image_id": image_id,
                "category_id": class_to_id[name],
                "bbox": [xmin, ymin, bbox_width, bbox_height],
                "area": bbox_width * bbox_height,
                "iscrowd": 0,
                "segmentation": [],
            })

    payload = {
        "info": {"description": f"DIOR {split} converted from Annotations2 XML"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": idx, "name": name, "supercategory": "object"} for idx, name in enumerate(classes)],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(images), len(annotations)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete-xml", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1] / "data" / "DIOR"
    annotation_root = root / "Annotations2"
    image_root = root / "images2"
    classes = read_classes(root / "classes.txt")
    results = {}
    for split in ("train", "val", "test"):
        results[split] = convert_split(
            split,
            annotation_root / split,
            image_root / split,
            classes,
            annotation_root / f"{split}.json",
        )

    if args.delete_xml:
        for split in ("train", "val", "test"):
            for xml_path in (annotation_root / split).glob("*.xml"):
                xml_path.unlink()

    for split, (image_count, annotation_count) in results.items():
        print(f"{split}: images={image_count}, annotations={annotation_count}")


if __name__ == "__main__":
    main()
