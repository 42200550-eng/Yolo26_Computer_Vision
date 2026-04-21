import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


DEFAULT_TEXTS = [
    "no entry sign",
    "no parking sign",
    "no parking no stopping sign",
    "no left turn sign",
    "slow down sign",
    "keep right sign",
]


def yolo_txt_line(cls_id: int, x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> str:
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    xc = x1 + bw / 2
    yc = y1 + bh / 2
    return f"{cls_id} {xc / w:.6f} {yc / h:.6f} {bw / w:.6f} {bh / h:.6f}"


def map_class_id(pred_cls: int, num_classes: int, class_map: dict[int, int]) -> int | None:
    if class_map:
        if pred_cls not in class_map:
            return None
        mapped = class_map[pred_cls]
        return mapped if 0 <= mapped < num_classes else None
    return max(0, min(pred_cls, num_classes - 1))


def optional_review(img, boxes_xyxy, classes, confs, class_names, use_gui: bool) -> bool:
    if not use_gui:
        return True

    canvas = img.copy()
    for (x1, y1, x2, y2), cls_id, conf in zip(boxes_xyxy, classes, confs):
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(canvas, p1, p2, (0, 255, 255), 2)
        text = f"{class_names.get(cls_id, str(cls_id))} {conf:.2f}"
        cv2.putText(canvas, text, (p1[0], max(18, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    cv2.imshow("review: a=accept r=reject q=quit", canvas)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("a"), ord("A")):
            return True
        if key in (ord("r"), ord("R")):
            return False
        if key in (ord("q"), ord("Q"), 27):
            raise KeyboardInterrupt("Review interrupted by user")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pseudo-label traffic signs using YOLOE-26 and filter by confidence")
    parser.add_argument("--input-dir", default="external_data/images", help="Directory of unlabeled input images")
    parser.add_argument("--out-images", default="datasets_v2/images/train", help="Output image directory")
    parser.add_argument("--out-labels", default="datasets_v2/labels/train", help="Output label directory")
    parser.add_argument("--model", default="yoloe-26s-seg.pt", help="YOLOE model path/name")
    parser.add_argument("--conf", type=float, default=0.70, help="Confidence threshold")
    parser.add_argument("--texts", nargs="*", default=DEFAULT_TEXTS, help="Text prompts for open-vocabulary detection")
    parser.add_argument("--num-classes", type=int, default=6, help="Project class count")
    parser.add_argument("--class-map-json", default="", help="JSON mapping predicted cls id to project cls id")
    parser.add_argument("--gui-review", action="store_true", help="Enable image-by-image review window")
    parser.add_argument("--report", default="smart_augment_report.json", help="Output report file")
    args = parser.parse_args()

    in_dir = Path(args.input_dir).resolve()
    out_images = Path(args.out_images).resolve()
    out_labels = Path(args.out_labels).resolve()
    report_path = Path(args.report).resolve()

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    class_map: dict[int, int] = {}
    if args.class_map_json:
        raw_map = json.loads(Path(args.class_map_json).read_text(encoding="utf-8"))
        class_map = {int(k): int(v) for k, v in raw_map.items()}

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        image_paths.extend(in_dir.rglob(ext))
    image_paths = sorted(image_paths)

    if not image_paths:
        raise FileNotFoundError(f"No input images found at {in_dir}")

    model = YOLO(args.model)

    accepted = 0
    rejected = 0
    empty = 0
    total_boxes = 0

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        try:
            results = model.predict(source=str(img_path), conf=args.conf, verbose=False, texts=args.texts)
        except TypeError:
            # Fallback if installed ultralytics version does not expose texts argument.
            results = model.predict(source=str(img_path), conf=args.conf, verbose=False)

        if not results:
            empty += 1
            continue

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            empty += 1
            continue

        boxes_xyxy = r.boxes.xyxy.cpu().numpy().tolist()
        confs = r.boxes.conf.cpu().numpy().tolist()
        pred_cls = r.boxes.cls.cpu().numpy().tolist()

        classes = []
        kept_boxes = []
        kept_confs = []
        for box, conf, c in zip(boxes_xyxy, confs, pred_cls):
            mapped = map_class_id(int(c), args.num_classes, class_map)
            if mapped is None:
                continue
            kept_boxes.append(box)
            kept_confs.append(conf)
            classes.append(mapped)

        boxes_xyxy = kept_boxes
        confs = kept_confs

        if not boxes_xyxy:
            empty += 1
            continue

        try:
            keep = optional_review(
                img=img,
                boxes_xyxy=boxes_xyxy,
                classes=classes,
                confs=confs,
                class_names={i: str(i) for i in range(args.num_classes)},
                use_gui=args.gui_review,
            )
        except KeyboardInterrupt:
            break

        if not keep:
            rejected += 1
            continue

        dst_img = out_images / img_path.name
        dst_lbl = out_labels / f"{img_path.stem}.txt"

        cv2.imwrite(str(dst_img), img)
        yolo_lines = []
        for (x1, y1, x2, y2), cls_id in zip(boxes_xyxy, classes):
            yolo_lines.append(yolo_txt_line(cls_id, x1, y1, x2, y2, w, h))
        dst_lbl.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

        accepted += 1
        total_boxes += len(yolo_lines)

    if args.gui_review:
        cv2.destroyAllWindows()

    report = {
        "input_dir": str(in_dir),
        "output_images": str(out_images),
        "output_labels": str(out_labels),
        "model": args.model,
        "confidence_threshold": args.conf,
        "class_map": class_map,
        "images_total": len(image_paths),
        "images_accepted": accepted,
        "images_rejected": rejected,
        "images_empty": empty,
        "boxes_total": total_boxes,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[INFO] Smart augmentation pseudo-label step completed")
    print(f"[INFO] Report: {report_path}")


if __name__ == "__main__":
    main()
