import argparse
import time

import cv2
from ultralytics import YOLO

from utils import VideoProcessor


def draw_box(frame, box, cls_name: str, score: float) -> None:
    x, y, w, h = box
    frame_h = frame.shape[0]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, cls_name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    conf_y = min(frame_h - 8, y + h + 16)
    cv2.putText(frame, f"{score:.2f}", (x, conf_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


def video_process(self, model: YOLO, output_path: str, conf_thres: float) -> None:
    self.open_save_video(output_path)
    frame_idx = 0
    start_time = time.perf_counter()

    while True:
        ret, frame = self.cap.read()
        if not ret:
            break

        # YOLO26 forward pass already includes end-to-end post-processing.
        result = model.predict(source=frame, conf=conf_thres, verbose=False)[0]

        if result.boxes is not None and len(result.boxes) > 0:
            for xyxy, cls_id, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                x1, y1, x2, y2 = xyxy.tolist()
                box = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                cls_name = model.names[int(cls_id)]
                draw_box(frame, box, cls_name, float(conf))

        elapsed = time.perf_counter() - start_time
        fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0.0
        cv2.putText(frame, f"FPS: {fps:.2f}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)

        self.out.write(frame)
        frame_idx += 1

        if self.total_frames:
            percent = (frame_idx / self.total_frames) * 100
            percent_int = min(int(percent / 2), 50)
            bar = "=" * percent_int
            print(f"\r[INFO] Processing: {bar}{percent:5.1f}% ({frame_idx}/{self.total_frames}) FPS={fps:.2f}", end="", flush=True)
        else:
            print(f"\r[INFO] Processed {frame_idx} frames FPS={fps:.2f}", end="", flush=True)

    self.close_save_video()
    print()


VideoProcessor.video_process_w_yolo = video_process


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO26 inference on video")
    parser.add_argument("--weights", default="runs/detect/yolo26s_traffic_final/weights/best.pt", help="Model weights path")
    parser.add_argument("--input", default="video/train_video.mp4", help="Input video path")
    parser.add_argument("--output", default="video/output_video.mp4", help="Output video path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    model = YOLO(args.weights)
    processor = VideoProcessor(args.input)
    processor.video_process_w_yolo(model, args.output, args.conf)
    print("[INFO] Video processing completed")


if __name__ == "__main__":
    main()