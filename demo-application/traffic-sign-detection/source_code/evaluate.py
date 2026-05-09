import argparse
import json
import time
from pathlib import Path

import yaml
from ultralytics import YOLO

try:
    import torch
except ImportError:  # Optional dependency for CUDA sync
    torch = None


def safe_metric_value(results, attr: str) -> float | None:
    obj = getattr(results, "box", None)
    if obj is None:
        return None
    return float(getattr(obj, attr)) if hasattr(obj, attr) else None


def load_class_names(data_yaml: str) -> dict[int, str]:
    with Path(data_yaml).resolve().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names_raw = data.get("names", {})
    if isinstance(names_raw, list):
        return {i: n for i, n in enumerate(names_raw)}
    return {int(k): v for k, v in names_raw.items()}


def per_class_metrics(results, class_names: dict[int, str]) -> list[dict]:
    box = getattr(results, "box", None)
    if box is None:
        return []

    items = []
    class_ids = sorted(class_names.keys())

    if hasattr(box, "class_result"):
        for i in class_ids:
            try:
                p, r, ap50, ap = box.class_result(i)
                items.append(
                    {
                        "class_id": i,
                        "class_name": class_names[i],
                        "precision": float(p),
                        "recall": float(r),
                        "ap50": float(ap50),
                        "ap50_95": float(ap),
                    }
                )
            except Exception:
                continue
        if items:
            return items

    maps = getattr(box, "maps", None)
    if maps is not None:
        maps_list = maps.tolist() if hasattr(maps, "tolist") else list(maps)
        for i in class_ids:
            ap = maps_list[i] if i < len(maps_list) else None
            items.append(
                {
                    "class_id": i,
                    "class_name": class_names[i],
                    "precision": None,
                    "recall": None,
                    "ap50": None,
                    "ap50_95": float(ap) if ap is not None else None,
                }
            )
    return items


def evaluate_model(model_path: str, data_yaml: str, imgsz: int, device: str, class_names: dict[int, str]) -> dict:
    model = YOLO(model_path)
    results = model.val(data=data_yaml, imgsz=imgsz, device=device, verbose=False, plots=True)
    return {
        "model": model_path,
        "save_dir": str(getattr(results, "save_dir", "")),
        "map50": safe_metric_value(results, "map50"),
        "map50_95": safe_metric_value(results, "map"),
        "mp": safe_metric_value(results, "mp"),
        "mr": safe_metric_value(results, "mr"),
        "per_class": per_class_metrics(results, class_names),
    }


def benchmark_speed(
    model_path: str,
    image_dir: Path,
    imgsz: int,
    device: str,
    max_images: int = 200,
    warmup: int = 10,
) -> dict:
    model = YOLO(model_path)
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        image_paths.extend(image_dir.rglob(ext))
    image_paths = sorted(image_paths)[:max_images]

    if not image_paths:
        return {"images": 0, "fps": 0.0, "avg_ms": 0.0}

    warmup_paths = image_paths[: min(len(image_paths), warmup)]
    for p in warmup_paths:
        model.predict(source=str(p), imgsz=imgsz, device=device, verbose=False)
    _sync_device(device)

    measured_paths = image_paths[len(warmup_paths):] or image_paths

    _sync_device(device)
    start = time.perf_counter()
    for p in measured_paths:
        model.predict(source=str(p), imgsz=imgsz, device=device, verbose=False)
    _sync_device(device)
    elapsed = time.perf_counter() - start

    fps = len(measured_paths) / elapsed if elapsed > 0 else 0.0
    avg_ms = (elapsed / len(measured_paths)) * 1000
    return {
        "images_total": len(image_paths),
        "images_warmup": len(warmup_paths),
        "images_measured": len(measured_paths),
        "fps": fps,
        "avg_ms": avg_ms,
    }


def _sync_device(device: str) -> None:
    if torch is None:
        return
    device_lower = str(device).lower()
    if "cpu" in device_lower:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def print_comparison_table(baseline: dict, candidate: dict) -> None:
    print(f"\n{'Class':<20} {'Baseline AP50':>14} {'Candidate AP50':>15} {'Delta':>8}")
    print("-" * 60)
    for b, c in zip(baseline.get("per_class", []), candidate.get("per_class", [])):
        b_ap = b.get("ap50", 0.0) or 0.0
        c_ap = c.get("ap50", 0.0) or 0.0
        delta = c_ap - b_ap
        arrow = "^" if delta > 0 else "v" if delta < 0 else "="
        print(f"{b['class_name']:<20} {b_ap:>13.1%} {c_ap:>14.1%} {delta:>+7.1%} {arrow}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and compare YOLO models")
    parser.add_argument("--baseline", default="runs/detect/traffic_detect_run3/weights/best.pt", help="Baseline model path")
    parser.add_argument("--candidate", default="runs/detect/yolo26s_traffic_final/weights/best.pt", help="Candidate model path")
    parser.add_argument("--data", default="data_v2.yaml", help="Dataset yaml")
    parser.add_argument("--imgsz", type=int, default=800, help="Validation image size")
    parser.add_argument("--device", default="0", help="Device for evaluation")
    parser.add_argument("--benchmark-dir", default="datasets_v2/images/test", help="Directory of benchmark images")
    parser.add_argument("--report", default="comparison_report.json", help="Output report path")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup images before speed benchmark")
    args = parser.parse_args()

    class_names = load_class_names(args.data)

    baseline_metrics = evaluate_model(args.baseline, args.data, args.imgsz, args.device, class_names)
    candidate_metrics = evaluate_model(args.candidate, args.data, args.imgsz, args.device, class_names)

    benchmark_dir = Path(args.benchmark_dir).resolve()
    baseline_speed = benchmark_speed(args.baseline, benchmark_dir, args.imgsz, args.device, warmup=args.warmup)
    candidate_speed = benchmark_speed(args.candidate, benchmark_dir, args.imgsz, args.device, warmup=args.warmup)

    report = {
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "speed_baseline": baseline_speed,
        "speed_candidate": candidate_speed,
    }

    report_path = Path(args.report).resolve()
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[INFO] Evaluation completed")
    print(f"[INFO] Report: {report_path}")
    print_comparison_table(baseline_metrics, candidate_metrics)


if __name__ == "__main__":
    main()
