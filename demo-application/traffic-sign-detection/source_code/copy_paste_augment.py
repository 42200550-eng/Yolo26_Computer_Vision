import argparse
import random
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    labels = []
    for ln in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = ln.split()
        if len(parts) != 5:
            continue
        labels.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return labels


def find_image(stem: str, image_dir: Path) -> Path | None:
    for ext in IMAGE_EXTS:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(val, hi))


def yolo_to_xyxy(label, w: int, h: int) -> tuple[int, int, int, int]:
    _, xc, yc, bw, bh = label
    x1 = int((xc - bw / 2) * w)
    y1 = int((yc - bh / 2) * h)
    x2 = int((xc + bw / 2) * w)
    y2 = int((yc + bh / 2) * h)
    return x1, y1, x2, y2


def xyxy_to_yolo(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[float, float, float, float]:
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    xc = (x1 + x2) / 2 / w
    yc = (y1 + y2) / 2 / h
    return xc, yc, bw, bh


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy-paste augmentation for minority classes")
    parser.add_argument("--images", default="datasets_v2/images/train", help="Input images directory")
    parser.add_argument("--labels", default="datasets_v2/labels/train", help="Input labels directory")
    parser.add_argument("--output-images", default="datasets_v2/images/train", help="Output images directory")
    parser.add_argument("--output-labels", default="datasets_v2/labels/train", help="Output labels directory")
    parser.add_argument("--class-ids", nargs="*", type=int, default=[4], help="Target class ids")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of synthetic images to create")
    parser.add_argument("--min-scale", type=float, default=0.6, help="Min scale of pasted box")
    parser.add_argument("--max-scale", type=float, default=1.2, help="Max scale of pasted box")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    image_dir = Path(args.images).resolve()
    label_dir = Path(args.labels).resolve()
    out_img = Path(args.output_images).resolve()
    out_lbl = Path(args.output_labels).resolve()
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    stems = [p.stem for p in label_dir.glob("*.txt")]
    if not stems:
        raise FileNotFoundError(f"No labels found at {label_dir}")

    candidates = []
    for stem in stems:
        labels = parse_labels(label_dir / f"{stem}.txt")
        if any(cls_id in args.class_ids for cls_id, *_ in labels):
            candidates.append(stem)

    if not candidates:
        raise FileNotFoundError("No candidate labels contain target class ids")

    backgrounds = []
    for stem in stems:
        img_path = find_image(stem, image_dir)
        if img_path:
            backgrounds.append(img_path)

    if not backgrounds:
        raise FileNotFoundError(f"No images found at {image_dir}")

    created = 0
    for i in range(args.num_samples):
        src_stem = random.choice(candidates)
        src_img_path = find_image(src_stem, image_dir)
        src_lbl_path = label_dir / f"{src_stem}.txt"
        if not src_img_path.exists() or not src_lbl_path.exists():
            continue

        src_img = cv2.imread(str(src_img_path))
        if src_img is None:
            continue

        src_h, src_w = src_img.shape[:2]
        src_labels = parse_labels(src_lbl_path)
        target_labels = [lbl for lbl in src_labels if lbl[0] in args.class_ids]
        if not target_labels:
            continue

        label = random.choice(target_labels)
        x1, y1, x2, y2 = yolo_to_xyxy(label, src_w, src_h)
        x1, y1 = clamp(x1, 0, src_w - 2), clamp(y1, 0, src_h - 2)
        x2, y2 = clamp(x2, x1 + 1, src_w - 1), clamp(y2, y1 + 1, src_h - 1)
        patch = src_img[y1:y2, x1:x2]

        bg_path = random.choice(backgrounds)
        bg_img = cv2.imread(str(bg_path))
        if bg_img is None:
            continue

        bg_h, bg_w = bg_img.shape[:2]
        scale = random.uniform(args.min_scale, args.max_scale)
        new_w = max(2, int(patch.shape[1] * scale))
        new_h = max(2, int(patch.shape[0] * scale))
        patch = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if new_w >= bg_w or new_h >= bg_h:
            continue

        px = random.randint(0, bg_w - new_w)
        py = random.randint(0, bg_h - new_h)
        bg_img[py:py + new_h, px:px + new_w] = patch

        cls_id = label[0]
        xc, yc, bw, bh = xyxy_to_yolo(px, py, px + new_w, py + new_h, bg_w, bg_h)

        out_stem = f"{bg_path.stem}_cp_{cls_id}_{i:04d}"
        out_img_path = out_img / f"{out_stem}{bg_path.suffix.lower()}"
        out_lbl_path = out_lbl / f"{out_stem}.txt"

        cv2.imwrite(str(out_img_path), bg_img)
        out_lbl_path.write_text(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")
        created += 1

    print(f"[INFO] Created {created} synthetic images")


if __name__ == "__main__":
    main()
