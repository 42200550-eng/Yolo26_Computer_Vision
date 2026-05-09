import argparse
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg.get("path", "."))
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    cfg["_root"] = root
    return cfg


def resolve(root: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    return p if p.is_absolute() else (root / p).resolve()


def parse_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    out = []
    for ln in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = ln.split()
        if len(parts) != 5:
            continue
        out.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return out


def find_image_for_label(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def augment_image(img: np.ndarray) -> np.ndarray:
    if random.random() < 0.5:
        alpha = random.uniform(0.85, 1.15)
        beta = random.randint(-20, 20)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    if random.random() < 0.35:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + random.uniform(-7, 7)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.85, 1.2), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(0.85, 1.2), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return img


def collect_stats(label_dir: Path) -> tuple[Counter, dict[str, set[int]]]:
    cls_counter = Counter()
    image_to_classes = defaultdict(set)

    for p in sorted(label_dir.glob("*.txt")):
        labels = parse_labels(p)
        for cls_id, *_ in labels:
            cls_counter[cls_id] += 1
            image_to_classes[p.stem].add(cls_id)

    return cls_counter, image_to_classes


def list_stems(label_dir: Path) -> list[str]:
    return sorted([p.stem for p in label_dir.glob("*.txt")])


def stratified_split_stems(
    stems: list[str],
    label_dir: Path,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[set[str], set[str], set[str]]:
    random.seed(seed)
    by_class = defaultdict(list)

    for stem in stems:
        labels = parse_labels(label_dir / f"{stem}.txt")
        if not labels:
            by_class[-1].append(stem)
            continue
        primary_cls = labels[0][0]
        by_class[primary_cls].append(stem)

    train_set: set[str] = set()
    val_set: set[str] = set()
    test_set: set[str] = set()

    for _, items in by_class.items():
        random.shuffle(items)
        n = len(items)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))

        test_items = items[:n_test]
        val_items = items[n_test:n_test + n_val]
        train_items = items[n_test + n_val:]

        test_set.update(test_items)
        val_set.update(val_items)
        train_set.update(train_items)

    # Keep non-empty training split by moving a few stems back if needed.
    if not train_set and (val_set or test_set):
        pull_from = val_set if val_set else test_set
        train_set.add(next(iter(pull_from)))
        pull_from.remove(next(iter(train_set)))

    return train_set, val_set, test_set


def copy_stems(src_img: Path, src_lbl: Path, dst_img: Path, dst_lbl: Path, stems: set[str]) -> int:
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)
    copied = 0

    for stem in sorted(stems):
        src_im = find_image_for_label(src_img, stem)
        src_lb = src_lbl / f"{stem}.txt"
        if src_im is None or not src_lb.exists():
            continue
        shutil.copy2(src_im, dst_img / src_im.name)
        shutil.copy2(src_lb, dst_lbl / src_lb.name)
        copied += 1

    return copied


def copy_base_dataset(src_img: Path, src_lbl: Path, dst_img: Path, dst_lbl: Path) -> None:
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    for p in src_img.glob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            shutil.copy2(p, dst_img / p.name)

    for p in src_lbl.glob("*.txt"):
        shutil.copy2(p, dst_lbl / p.name)


def oversample_minority(
    src_img: Path,
    src_lbl: Path,
    dst_img: Path,
    dst_lbl: Path,
    cls_counter: Counter,
    image_to_classes: dict[str, set[int]],
    min_instances: int,
    seed: int,
    num_classes: int,
    prefer_single: bool,
) -> Counter:
    random.seed(seed)
    current = Counter(cls_counter)
    by_class = defaultdict(list)
    by_class_single = defaultdict(list)

    for stem, classes in image_to_classes.items():
        for c in classes:
            by_class[c].append(stem)
        if len(classes) == 1:
            only_cls = next(iter(classes))
            by_class_single[only_cls].append(stem)

    for cls_id in range(num_classes):
        count = current.get(cls_id, 0)
        if count >= min_instances:
            continue

        candidates = by_class.get(cls_id, [])
        if prefer_single and by_class_single.get(cls_id):
            candidates = by_class_single[cls_id]
        if not candidates:
            continue

        need = min_instances - count
        for i in range(need):
            stem = random.choice(candidates)
            src_im = find_image_for_label(src_img, stem)
            src_lb = src_lbl / f"{stem}.txt"
            if src_im is None or not src_lb.exists():
                continue

            img = cv2.imread(str(src_im))
            if img is None:
                continue

            img_aug = augment_image(img)
            new_stem = f"{stem}_os_{cls_id}_{i:04d}"
            out_img = dst_img / f"{new_stem}{src_im.suffix.lower()}"
            out_lbl = dst_lbl / f"{new_stem}.txt"
            cv2.imwrite(str(out_img), img_aug)
            shutil.copy2(src_lb, out_lbl)

            labels = parse_labels(src_lb)
            for c, *_ in labels:
                if c == cls_id:
                    current[c] += 1

    return current


def main() -> None:
    parser = argparse.ArgumentParser(description="Balance YOLO dataset by oversampling minority classes")
    parser.add_argument("--data", default="data.yaml", help="Dataset YAML path")
    parser.add_argument("--split", default="train", choices=["train", "val", "all"], help="Split to balance")
    parser.add_argument("--labels", default="labels/{split}", help="Label directory pattern")
    parser.add_argument("--output-root", default="datasets_v2", help="Balanced dataset root")
    parser.add_argument("--min-instances", type=int, default=300, help="Target minimum instances per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Val ratio when creating splits from train")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio when creating splits from train")
    parser.add_argument("--ensure-splits", action="store_true", help="Ensure val/test exist in output dataset")
    parser.add_argument(
        "--prefer-single",
        dest="prefer_single",
        action="store_true",
        help="Prefer single-class images when oversampling",
    )
    parser.add_argument(
        "--no-prefer-single",
        dest="prefer_single",
        action="store_false",
        help="Allow multi-class images when oversampling",
    )
    parser.set_defaults(prefer_single=True)
    args = parser.parse_args()

    cfg = load_yaml(Path(args.data).resolve())
    root = cfg["_root"]

    out_root = Path(args.output_root).resolve()
    source_train_img = resolve(root, cfg.get("train", "images/train"))
    source_train_lbl = resolve(root, args.labels.format(split="train"))

    if not source_train_img.exists() or not source_train_lbl.exists():
        raise FileNotFoundError("Source train image or label directory not found")

    src_val_img = resolve(root, cfg.get("val", "images/val"))
    src_val_lbl = resolve(root, args.labels.format(split="val"))
    src_test_img = resolve(root, cfg.get("test", "images/test"))
    src_test_lbl = resolve(root, args.labels.format(split="test"))

    use_existing_val = src_val_img.exists() and src_val_lbl.exists()
    use_existing_test = src_test_img.exists() and src_test_lbl.exists()

    if args.split in {"train", "all"}:
        if use_existing_val and use_existing_test:
            copy_base_dataset(
                source_train_img,
                source_train_lbl,
                out_root / "images" / "train",
                out_root / "labels" / "train",
            )
            copy_base_dataset(src_val_img, src_val_lbl, out_root / "images" / "val", out_root / "labels" / "val")
            copy_base_dataset(src_test_img, src_test_lbl, out_root / "images" / "test", out_root / "labels" / "test")
        else:
            stems = list_stems(source_train_lbl)
            train_stems, val_stems, test_stems = stratified_split_stems(
                stems=stems,
                label_dir=source_train_lbl,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
            )
            copy_stems(source_train_img, source_train_lbl, out_root / "images" / "train", out_root / "labels" / "train", train_stems)
            copy_stems(source_train_img, source_train_lbl, out_root / "images" / "val", out_root / "labels" / "val", val_stems)
            copy_stems(source_train_img, source_train_lbl, out_root / "images" / "test", out_root / "labels" / "test", test_stems)

        train_lbl_out = out_root / "labels" / "train"
        train_img_out = out_root / "images" / "train"
        cls_counter, image_to_classes = collect_stats(train_lbl_out)
        updated_counter = oversample_minority(
            src_img=train_img_out,
            src_lbl=train_lbl_out,
            dst_img=train_img_out,
            dst_lbl=train_lbl_out,
            cls_counter=cls_counter,
            image_to_classes=image_to_classes,
            min_instances=args.min_instances,
            seed=args.seed,
            num_classes=int(cfg.get("nc", 6)),
            prefer_single=args.prefer_single,
        )
    else:
        src_img = resolve(root, cfg.get("val", "images/val"))
        src_lbl = resolve(root, args.labels.format(split="val"))
        dst_img = out_root / "images" / "val"
        dst_lbl = out_root / "labels" / "val"
        if not src_img.exists() or not src_lbl.exists():
            raise FileNotFoundError("Source val image or label directory not found")
        copy_base_dataset(src_img, src_lbl, dst_img, dst_lbl)
        updated_counter, _ = collect_stats(dst_lbl)

    print("[INFO] Data balancing completed")
    print(f"[INFO] Output root: {out_root}")
    print("[INFO] Class stats after balancing:")
    for c in sorted(updated_counter):
        print(f"  class {c}: {updated_counter[c]}")


if __name__ == "__main__":
    main()
