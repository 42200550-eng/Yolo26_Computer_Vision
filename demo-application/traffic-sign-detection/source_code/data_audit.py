import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import yaml


def load_data_yaml(data_yaml_path: Path) -> dict:
    with data_yaml_path.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    root = Path(data_cfg.get("path", "."))
    if not root.is_absolute():
        root = (data_yaml_path.parent / root).resolve()

    data_cfg["_root"] = root
    return data_cfg


def resolve_dir(root: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    return p if p.is_absolute() else (root / p).resolve()


def yolo_line_ok(parts: list[str]) -> tuple[bool, str]:
    if len(parts) != 5:
        return False, "invalid_columns"
    try:
        cls_id = int(parts[0])
        x, y, w, h = [float(v) for v in parts[1:]]
    except ValueError:
        return False, "parse_error"

    if cls_id < 0:
        return False, "negative_class"
    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
        return False, "bbox_out_of_range"
    if x - w / 2 < 0 or y - h / 2 < 0 or x + w / 2 > 1 or y + h / 2 > 1:
        return False, "bbox_outside_image"
    return True, "ok"


def collect_label_stats(label_dir: Path, num_classes: int) -> dict:
    class_counter = Counter()
    bbox_areas = []
    errors = Counter()
    image_to_classes = defaultdict(set)
    total_boxes = 0

    txt_files = sorted(label_dir.rglob("*.txt"))
    for txt in txt_files:
        try:
            lines = txt.read_text(encoding="utf-8").strip().splitlines()
        except UnicodeDecodeError:
            errors["unicode_error"] += 1
            continue

        if not lines:
            errors["empty_label_file"] += 1
            continue

        image_id = txt.stem
        for ln in lines:
            parts = ln.strip().split()
            ok, reason = yolo_line_ok(parts)
            if not ok:
                errors[reason] += 1
                continue

            cls_id = int(parts[0])
            if cls_id >= num_classes:
                errors["class_out_of_range"] += 1
                continue

            _, _, w, h = [float(v) for v in parts[1:]]
            bbox_areas.append(w * h)
            class_counter[cls_id] += 1
            image_to_classes[image_id].add(cls_id)
            total_boxes += 1

    return {
        "files": len(txt_files),
        "total_boxes": total_boxes,
        "class_counter": dict(class_counter),
        "bbox_areas": bbox_areas,
        "errors": dict(errors),
        "image_to_classes": {k: sorted(list(v)) for k, v in image_to_classes.items()},
    }


def plot_distributions(class_counter: dict, bbox_areas: list[float], names: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    cls_ids = sorted(names.keys())
    labels = [names[c] for c in cls_ids]
    values = [class_counter.get(c, 0) for c in cls_ids]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values)
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), str(v), ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Instances")
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "class_distribution.png", dpi=150)
    plt.close()

    if bbox_areas:
        plt.figure(figsize=(8, 5))
        plt.hist(bbox_areas, bins=30)
        plt.xlabel("Normalized BBox Area")
        plt.ylabel("Count")
        plt.title("BBox Area Distribution")
        plt.tight_layout()
        plt.savefig(out_dir / "bbox_area_distribution.png", dpi=150)
        plt.close()


def validate_image_pairs(root: Path, img_dir_rel: str, lbl_dir_rel: str) -> dict:
    img_dir = resolve_dir(root, img_dir_rel)
    lbl_dir = resolve_dir(root, lbl_dir_rel)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = {p.stem: p for p in img_dir.rglob("*") if p.suffix.lower() in image_exts}
    labels = {p.stem: p for p in lbl_dir.rglob("*.txt")}

    missing_labels = sorted([k for k in images if k not in labels])
    missing_images = sorted([k for k in labels if k not in images])

    unreadable_images = []
    for stem, p in images.items():
        img = cv2.imread(str(p))
        if img is None:
            unreadable_images.append(stem)

    return {
        "images": len(images),
        "labels": len(labels),
        "missing_labels": len(missing_labels),
        "missing_images": len(missing_images),
        "unreadable_images": len(unreadable_images),
        "missing_labels_examples": missing_labels[:20],
        "missing_images_examples": missing_images[:20],
        "unreadable_images_examples": unreadable_images[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit YOLO dataset quality and class balance")
    parser.add_argument("--data", default="data.yaml", help="Path to dataset yaml")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Split to audit")
    parser.add_argument("--labels", default="labels/{split}", help="Label directory pattern")
    parser.add_argument("--images", default=None, help="Image directory path or pattern")
    parser.add_argument("--output", default="data_audit_report.json", help="Output json report")
    parser.add_argument("--plot-dir", default="audit_plots", help="Output plot directory")
    args = parser.parse_args()

    data_yaml_path = Path(args.data).resolve()
    data_cfg = load_data_yaml(data_yaml_path)
    root = data_cfg["_root"]

    names_raw = data_cfg.get("names", {})
    if isinstance(names_raw, list):
        names = {i: n for i, n in enumerate(names_raw)}
    else:
        names = {int(k): v for k, v in names_raw.items()}

    num_classes = int(data_cfg.get("nc", len(names)))
    label_rel = args.labels.format(split=args.split)
    label_dir = resolve_dir(root, label_rel)

    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    stats = collect_label_stats(label_dir, num_classes)
    cls_counts = {i: stats["class_counter"].get(i, 0) for i in range(num_classes)}

    nonzero = [v for v in cls_counts.values() if v > 0]
    imbalance_ratio = float(max(nonzero) / max(1, min(nonzero))) if nonzero else 0.0

    img_rel = args.images if args.images else data_cfg.get(args.split, f"images/{args.split}")
    pair_report = validate_image_pairs(root, img_rel, label_rel)

    report = {
        "data_yaml": str(data_yaml_path),
        "root": str(root),
        "split": args.split,
        "num_classes": num_classes,
        "class_names": names,
        "class_counts": cls_counts,
        "imbalance_ratio": imbalance_ratio,
        "label_files": stats["files"],
        "total_boxes": stats["total_boxes"],
        "label_errors": stats["errors"],
        "pair_report": pair_report,
    }

    out_path = Path(args.output).resolve()
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_dir = Path(args.plot_dir).resolve()
    plot_distributions(cls_counts, stats["bbox_areas"], names, plot_dir)

    print("[INFO] Data audit completed")
    print(f"[INFO] Report: {out_path}")
    print(f"[INFO] Plots: {plot_dir}")


if __name__ == "__main__":
    main()
