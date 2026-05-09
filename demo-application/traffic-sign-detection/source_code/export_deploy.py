import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def export_model(model_path: str, fmt: str, imgsz: int, device: str) -> str:
    model = YOLO(model_path)
    kwargs = {"format": fmt, "imgsz": imgsz, "device": device}
    if fmt == "onnx":
        kwargs["simplify"] = True
    if fmt == "engine":
        kwargs["half"] = True
    exported = model.export(**kwargs)
    return str(exported)


def validate_export(exported_path: str, data_yaml: str, imgsz: int, device: str) -> dict:
    model = YOLO(exported_path)
    results = model.val(data=data_yaml, imgsz=imgsz, device=device, verbose=False)
    box = getattr(results, "box", None)
    return {
        "map50": float(getattr(box, "map50")) if box and hasattr(box, "map50") else None,
        "map50_95": float(getattr(box, "map")) if box and hasattr(box, "map") else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO model for deployment and validate")
    parser.add_argument("--model", default="runs/detect/yolo26s_traffic_final/weights/best.pt", help="Input model path")
    parser.add_argument("--data", default="data_v2.yaml", help="Dataset yaml for validation")
    parser.add_argument("--imgsz", type=int, default=800, help="Image size")
    parser.add_argument("--device", default="0", help="Device")
    parser.add_argument("--formats", nargs="*", default=["onnx", "engine", "openvino"], help="Export formats")
    parser.add_argument("--report", default="export_report.json", help="Output report file")
    args = parser.parse_args()

    report = {"source_model": args.model, "exports": []}

    for fmt in args.formats:
        item = {"format": fmt, "status": "ok"}
        try:
            exported = export_model(args.model, fmt, args.imgsz, args.device)
            item["path"] = exported
            item["metrics"] = validate_export(exported, args.data, args.imgsz, args.device)
        except Exception as e:
            item["status"] = "error"
            item["error"] = str(e)
        report["exports"].append(item)

    report_path = Path(args.report).resolve()
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[INFO] Export completed")
    print(f"[INFO] Report: {report_path}")


if __name__ == "__main__":
    main()
