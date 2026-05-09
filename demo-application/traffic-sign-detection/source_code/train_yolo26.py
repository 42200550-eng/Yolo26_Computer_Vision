import argparse
from ultralytics import YOLO


def train_yolo26_optimized(
    model_path: str,
    data_yaml: str,
    name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    resume: str,
) -> None:
    model = YOLO(model_path)
    model.train(
        data=data_yaml,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        device=device,
        resume=resume or False,
        workers=4,
        patience=30,
        optimizer="auto",
        lr0=0.01,
        lrf=0.001,
        cos_lr=True,
        warmup_epochs=5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,
        degrees=15,
        scale=0.5,
        translate=0.2,
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.0,
        erasing=0.3,
        close_mosaic=15,
        save_period=10,
        plots=True,
        name=name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO26 for traffic sign detection")
    parser.add_argument("--model", default="yolo26s.pt", help="Pretrained model")
    parser.add_argument("--data", default="data_v2.yaml", help="Dataset yaml")
    parser.add_argument("--name", default="yolo26s_traffic_v1", help="Run name")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=800, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="0", help="Device id or cpu")
    parser.add_argument("--resume", default="", help="Resume from last checkpoint path")
    args = parser.parse_args()

    train_yolo26_optimized(
        model_path=args.model,
        data_yaml=args.data,
        name=args.name,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
