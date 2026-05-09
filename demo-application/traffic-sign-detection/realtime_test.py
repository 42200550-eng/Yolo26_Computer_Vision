import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def draw_detections(frame, boxes: list[tuple[int, int, int, int]], labels: list[str], confs: list[float]) -> None:
    for (x1, y1, x2, y2), label, conf in zip(boxes, labels, confs):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def open_writer(output_path: str, fps: float, size: tuple[int, int]) -> cv2.VideoWriter | None:
    if not output_path:
        return None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, size)


def filter_detections(
    result,
    conf_by_class: dict[int, float],
    min_box: int,
) -> dict[int, tuple[tuple[int, int, int, int], float]]:
    if result.boxes is None:
        return {}

    best_by_class: dict[int, tuple[tuple[int, int, int, int], float]] = {}
    for xyxy, cls_id, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
        cls_int = int(cls_id)
        conf_val = float(conf)
        if conf_val < conf_by_class.get(cls_int, 0.0):
            continue

        x1, y1, x2, y2 = [int(v) for v in xyxy.tolist()]
        if (x2 - x1) < min_box or (y2 - y1) < min_box:
            continue

        prev = best_by_class.get(cls_int)
        if prev is None or conf_val > prev[1]:
            best_by_class[cls_int] = ((x1, y1, x2, y2), conf_val)

    return best_by_class


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime YOLO26 test")
    parser.add_argument("--weights", default="runs/detect/yolo26s_traffic_final/weights/best.pt", help="Weights path")
    parser.add_argument("--input", default="0", help="Camera index or video path")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "dshow", "msmf"],
        help="Video backend override for Windows cameras",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--go-slowdown-conf", type=float, default=0.6, help="Confidence threshold for Go_slowdown")
    parser.add_argument("--min-box", type=int, default=20, help="Minimum bbox size in pixels")
    parser.add_argument("--temporal", type=int, default=3, help="Min consecutive frames before showing a class")
    parser.add_argument("--track", action="store_true", help="Use tracker for smoother results")
    parser.add_argument("--tracker", default="botsort.yaml", help="Tracker config for track mode")
    parser.add_argument("--imgsz", type=int, default=800, help="Inference image size")
    parser.add_argument("--output", default="", help="Optional output video path")
    args = parser.parse_args()

    source = int(args.input) if args.input.isdigit() else args.input
    if args.backend == "dshow":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    elif args.backend == "msmf":
        cap = cv2.VideoCapture(source, cv2.CAP_MSMF)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.input}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = open_writer(args.output, fps, (width, height))

    model = YOLO(args.weights)
    names = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}

    go_slowdown_id = None
    for cls_id, name in names.items():
        if str(name).lower() == "go_slowdown":
            go_slowdown_id = int(cls_id)
            break
    if go_slowdown_id is None and 4 in names:
        go_slowdown_id = 4

    conf_by_class = {cls_id: args.conf for cls_id in names}
    if go_slowdown_id is not None:
        conf_by_class[go_slowdown_id] = args.go_slowdown_conf

    class_state = {cls_id: {"count": 0, "box": None, "conf": 0.0} for cls_id in names}

    frame_idx = 0
    start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.track:
            result = model.track(source=frame, conf=args.conf, imgsz=args.imgsz, verbose=False, persist=True, tracker=args.tracker)[0]
        else:
            result = model.predict(source=frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]

        best_by_class = filter_detections(result, conf_by_class, args.min_box)

        for cls_id in class_state:
            if cls_id in best_by_class:
                box, conf = best_by_class[cls_id]
                class_state[cls_id]["count"] = min(class_state[cls_id]["count"] + 1, args.temporal)
                class_state[cls_id]["box"] = box
                class_state[cls_id]["conf"] = conf
            else:
                class_state[cls_id]["count"] = 0

        boxes = []
        labels = []
        confs = []
        for cls_id, state in class_state.items():
            if state["count"] >= args.temporal and state["box"] is not None:
                boxes.append(state["box"])
                labels.append(names.get(cls_id, str(cls_id)))
                confs.append(state["conf"])

        draw_detections(frame, boxes, labels, confs)

        elapsed = time.perf_counter() - start
        fps_now = (frame_idx + 1) / elapsed if elapsed > 0 else 0.0
        cv2.putText(frame, f"FPS: {fps_now:.2f}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)

        cv2.imshow("YOLO26 Realtime", frame)
        if writer:
            writer.write(frame)

        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
