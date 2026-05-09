# Final Report - YOLO26 Traffic Sign Detection

Date: 2026-05-09

## 1) Project overview

This project builds a traffic sign detection pipeline using Ultralytics YOLO26. The focus is on a data-first workflow: audit labels, balance class distribution, train baseline, tune hyperparameters, evaluate, and export for deployment.

## 2) Architecture

YOLO26 uses a CSP-style backbone, PAN/FPN neck, and dual head (one-to-many for training, one-to-one for inference). This design targets fast, NMS-free inference while preserving accuracy on small objects.

## 3) Data pipeline

Pipeline stages:

1. Audit labels and stats
2. Balance dataset and ensure train/val/test splits
3. (Optional) External data import and pseudo-labeling
4. Train baseline
5. Tune and train final
6. Evaluate and export

Key scripts:

- source_code/data_audit.py
- source_code/data_balance.py
- source_code/download_external_data.py
- source_code/smart_augment.py
- source_code/train_yolo26.py
- source_code/tune_yolo26.py
- source_code/evaluate.py
- source_code/export_deploy.py

## 4) Training configuration

Baseline (recommended profile):

- Model: yolo26s.pt
- Epochs: 150
- Image size: 800
- Batch: 16
- Device: GPU 0

## 5) Results summary

From demo-application/traffic-sign-detection/comparison_report.json:

| Model | mAP50 | mAP50-95 | Precision | Recall |
| --- | --- | --- | --- | --- |
| Baseline (yolo26s_traffic_v1) | 0.9290 | 0.8704 | 0.8222 | 0.9446 |
| Final (yolo26s_traffic_final) | 0.9026 | 0.8472 | 0.8525 | 0.9555 |

Per-class performance is stored in the same report for detailed inspection.

## 6) Speed benchmark (GPU)

Benchmark settings:

- Warmup: 10 images
- Measured: 190 images
- Image size: 800

Results:

| Model | FPS | Avg ms | Images |
| --- | --- | --- | --- |
| Baseline (yolo26s_traffic_v1) | 27.1 | 36.89 | 190 |
| Final (yolo26s_traffic_final) | 23.4 | 42.73 | 190 |

## 7) Comparison and notes

- Baseline has higher mAP on this run, while the final model improves recall slightly.
- Speed is lower for the final model in this evaluation snapshot.
- For small and rare classes, per-class metrics should guide further tuning and data augmentation.

## 8) Limitations

- Dataset imbalance remains a risk for rare classes.
- Results depend on dataset split and training randomness.
- External data import and pseudo-labeling require manual verification for label quality.

## 9) Reproducibility

Run the full pipeline from demo-application/traffic-sign-detection:

```powershell
python source_code/data_balance.py --data data.yaml --split all --ensure-splits --output-root datasets_v2 --min-instances 300
python source_code/train_yolo26.py --model yolo26s.pt --data data_v2.yaml --name yolo26s_traffic_v1 --epochs 150 --imgsz 800 --batch 16 --device 0
python source_code/tune_yolo26.py --mode tune --weights runs/detect/yolo26s_traffic_v1/weights/best.pt --data data_v2.yaml --imgsz 800 --device 0 --tune-epochs 50 --iterations 100
python source_code/tune_yolo26.py --mode final --model yolo26s.pt --data data_v2.yaml --cfg runs/detect/tune/best_hyperparameters.yaml --name yolo26s_traffic_final --imgsz 800 --device 0 --final-epochs 200
python source_code/evaluate.py --baseline runs/detect/yolo26s_traffic_v1/weights/best.pt --candidate runs/detect/yolo26s_traffic_final/weights/best.pt --data data_v2.yaml --imgsz 800 --device 0 --warmup 10
```

## 10) Deliverables

- Code: demo-application/traffic-sign-detection/source_code
- Weights: runs/detect/*/weights
- Reports: demo-application/traffic-sign-detection/comparison_report.json
- README: README.md
