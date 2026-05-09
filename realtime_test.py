"""
YOLO26 Realtime Traffic Sign Detection — Standalone Test Script
Runs inference on a video file, saves output with bounding boxes and FPS overlay.
No external utils dependency required.
"""

import argparse
import time
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

# ── Colour palette (one per class, BGR) ────────────────────────────
COLORS = [
    (0, 0, 255),      # 0 No_entry       — Red
    (255, 165, 0),     # 1 No_parking     — Orange
    (0, 255, 255),     # 2 No_parking_stop— Yellow
    (255, 0, 255),     # 3 No_turn_left   — Magenta
    (0, 200, 200),     # 4 Go_slowdown    — Gold
    (0, 255, 0),       # 5 Keep_going     — Green
]


def draw_detections(frame, result, model):
    """Draw bounding boxes, class names, and confidence on frame."""
    if result.boxes is None or len(result.boxes) == 0:
        return 0

    count = 0
    for xyxy, cls_id, conf in zip(
        result.boxes.xyxy, result.boxes.cls, result.boxes.conf
    ):
        x1, y1, x2, y2 = map(int, xyxy.tolist())
        cid = int(cls_id)
        cls_name = model.names[cid]
        color = COLORS[cid % len(COLORS)]

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        label = f"{cls_name} {float(conf):.0%}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        count += 1

    return count


def draw_hud(frame, fps, frame_idx, total_frames, det_count):
    """Draw FPS, progress, and detection count overlay."""
    h, w = frame.shape[:2]

    # Semi-transparent bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)

    # Detections count
    cv2.putText(frame, f"Det: {det_count}", (160, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Progress
    if total_frames > 0:
        pct = frame_idx / total_frames * 100
        cv2.putText(frame, f"{pct:.1f}%", (w - 100, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Progress bar at bottom
        bar_y = h - 6
        bar_w = int(w * frame_idx / total_frames)
        cv2.rectangle(frame, (0, bar_y), (bar_w, h), (0, 220, 255), -1)


def run_inference(weights, input_video, output_video, conf_thres, show_window):
    """Main inference loop."""

    # ── Load model ──────────────────────────────────────────────
    print(f"[INFO] Loading model: {weights}")
    model = YOLO(weights)
    print(f"[INFO] Classes: {model.names}")

    # ── Open video ──────────────────────────────────────────────
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_video}")
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video: {w}x{h} @ {fps_src:.1f}fps, {total} frames")

    # ── Output writer ──────────────────────────────────────────
    Path(output_video).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps_src, (w, h))

    # ── Warm-up (5 dummy frames) ───────────────────────────────
    import numpy as np
    dummy = np.zeros((h, w, 3), dtype=np.uint8)
    for _ in range(5):
        model.predict(source=dummy, conf=conf_thres, verbose=False)
    print("[INFO] Warm-up done. Starting inference...")

    # ── Main loop ──────────────────────────────────────────────
    frame_idx = 0
    t_start = time.perf_counter()
    inference_times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        result = model.predict(source=frame, conf=conf_thres, verbose=False)[0]
        t1 = time.perf_counter()
        inference_times.append(t1 - t0)

        det_count = draw_detections(frame, result, model)

        elapsed = time.perf_counter() - t_start
        fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0.0
        draw_hud(frame, fps, frame_idx + 1, total, det_count)

        out.write(frame)
        frame_idx += 1

        # Show window if requested
        if show_window:
            cv2.imshow("YOLO26 Realtime", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                print("\n[INFO] User quit")
                break

        # Console progress
        if frame_idx % 10 == 0 or frame_idx == total:
            pct = frame_idx / total * 100 if total > 0 else 0
            avg_inf = sum(inference_times[-50:]) / len(inference_times[-50:]) * 1000
            print(f"\r[INFO] Frame {frame_idx}/{total} ({pct:.1f}%) | "
                  f"FPS: {fps:.1f} | Inference: {avg_inf:.1f}ms | "
                  f"Det: {det_count}", end="", flush=True)

    cap.release()
    out.release()
    if show_window:
        cv2.destroyAllWindows()

    # ── Summary ────────────────────────────────────────────────
    total_time = time.perf_counter() - t_start
    avg_inf = sum(inference_times) / len(inference_times) * 1000 if inference_times else 0
    p50 = sorted(inference_times)[len(inference_times) // 2] * 1000 if inference_times else 0
    p99 = sorted(inference_times)[int(len(inference_times) * 0.99)] * 1000 if inference_times else 0

    print(f"\n\n{'='*60}")
    print(f"  YOLO26 Inference Summary")
    print(f"{'='*60}")
    print(f"  Model         : {Path(weights).name}")
    print(f"  Video          : {Path(input_video).name}")
    print(f"  Resolution     : {w}x{h}")
    print(f"  Total frames   : {frame_idx}")
    print(f"  Total time     : {total_time:.1f}s")
    print(f"  Average FPS    : {frame_idx / total_time:.1f}")
    print(f"  Inference avg  : {avg_inf:.1f}ms")
    print(f"  Inference P50  : {p50:.1f}ms")
    print(f"  Inference P99  : {p99:.1f}ms")
    print(f"  Confidence     : {conf_thres}")
    print(f"  Output saved   : {output_video}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO26 Realtime Traffic Sign Detection"
    )
    parser.add_argument(
        "--weights",
        default=r"runs\detect\yolo26s_traffic_v1\weights\best.pt",
        help="Model weights path"
    )
    parser.add_argument(
        "--input", required=True,
        help="Input video path"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output video path (default: input_yolo26.mp4)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Show live window (press Q to quit)"
    )
    args = parser.parse_args()

    if args.output is None:
        stem = Path(args.input).stem
        args.output = str(Path(args.input).parent / f"{stem}_yolo26.mp4")

    run_inference(args.weights, args.input, args.output, args.conf, args.show)


if __name__ == "__main__":
    main()
