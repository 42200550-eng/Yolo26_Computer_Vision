import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def auto_tune(weights_path: str, data_yaml: str, imgsz: int, device: str, epochs: int, iterations: int) -> None:
    model = YOLO(weights_path)
    model.tune(
        data=data_yaml,
        epochs=epochs,
        iterations=iterations,
        imgsz=imgsz,
        device=device,
        plots=True,
    )


def train_final(model_path: str, data_yaml: str, cfg_path: str, run_name: str, epochs: int, imgsz: int, device: str) -> None:
    model = YOLO(model_path)
    cfg_file = Path(cfg_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Hyperparameter file not found: {cfg_file}")

    with cfg_file.open("r", encoding="utf-8") as f:
        raw_hyps = yaml.safe_load(f) or {}

    scalar_hyps = {k: v for k, v in raw_hyps.items() if isinstance(v, (int, float, bool, str))}

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        name=run_name,
        patience=40,
        **scalar_hyps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter evolution and final training for YOLO26")
    parser.add_argument("--mode", choices=["tune", "final"], required=True, help="Run tune or final training")
    parser.add_argument("--weights", default="runs/detect/yolo26s_traffic_v1/weights/best.pt", help="Weights for tune mode")
    parser.add_argument("--model", default="yolo26s.pt", help="Model for final training")
    parser.add_argument("--data", default="data_v2.yaml", help="Dataset yaml")
    parser.add_argument("--cfg", default="runs/detect/tune/best_hyperparameters.yaml", help="Best hyperparameter cfg")
    parser.add_argument("--name", default="yolo26s_traffic_final", help="Final run name")
    parser.add_argument("--imgsz", type=int, default=800, help="Image size")
    parser.add_argument("--device", default="0", help="Device")
    parser.add_argument("--tune-epochs", type=int, default=50, help="Epochs per tune iteration")
    parser.add_argument("--iterations", type=int, default=100, help="Tune iterations")
    parser.add_argument("--final-epochs", type=int, default=200, help="Final training epochs")
    args = parser.parse_args()

    if args.mode == "tune":
        auto_tune(
            weights_path=args.weights,
            data_yaml=args.data,
            imgsz=args.imgsz,
            device=args.device,
            epochs=args.tune_epochs,
            iterations=args.iterations,
        )
    else:
        train_final(
            model_path=args.model,
            data_yaml=args.data,
            cfg_path=args.cfg,
            run_name=args.name,
            epochs=args.final_epochs,
            imgsz=args.imgsz,
            device=args.device,
        )


if __name__ == "__main__":
    main()
