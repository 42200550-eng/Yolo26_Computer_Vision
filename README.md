# Yolo26 Computer Vision

Traffic sign detection using Ultralytics YOLO26 with a data-first pipeline, evaluation tooling, and deployment exports.

## Highlights

- YOLO26 training and tuning scripts.
- Data audit, balancing, and optional pseudo-labeling utilities.
- Evaluation with per-class metrics and speed benchmarking.
- Demo app assets and trained weights.

## Repository layout

- demo-application/traffic-sign-detection: End-to-end pipeline for traffic sign detection.
- datasets_rf: Roboflow datasets (train/val/test).
- runs: Training and evaluation outputs.
- analysis_report.md, implementation_plan.md: Planning and analysis notes.

## Environment setup

Create a Python environment (3.10+ recommended) and install dependencies:

```powershell
pip install numpy matplotlib opencv-python ultralytics torch torchvision pyyaml
```

If you use an NVIDIA GPU, ensure your CUDA driver matches your PyTorch build.

## Quickstart (from traffic-sign-detection)

```powershell
cd demo-application/traffic-sign-detection

# 1) Build a balanced dataset with train/val/test splits
python source_code/data_balance.py --data data.yaml --split all --ensure-splits --output-root datasets_v2 --min-instances 300

# 2) Train baseline
python source_code/train_yolo26.py --model yolo26s.pt --data data_v2.yaml --name yolo26s_traffic_v1 --epochs 150 --imgsz 800 --batch 16 --device 0

# 3) Tune and train final (optional)
python source_code/tune_yolo26.py --mode tune --weights runs/detect/yolo26s_traffic_v1/weights/best.pt --data data_v2.yaml --imgsz 800 --device 0 --tune-epochs 50 --iterations 100
python source_code/tune_yolo26.py --mode final --model yolo26s.pt --data data_v2.yaml --cfg runs/detect/tune/best_hyperparameters.yaml --name yolo26s_traffic_final --imgsz 800 --device 0 --final-epochs 200

# 4) Evaluate baseline vs final
python source_code/evaluate.py --baseline runs/detect/yolo26s_traffic_v1/weights/best.pt --candidate runs/detect/yolo26s_traffic_final/weights/best.pt --data data_v2.yaml --imgsz 800 --device 0 --warmup 10
```

Notes:

- data_v2.yaml uses relative paths, so run commands inside demo-application/traffic-sign-detection.
- Reduce batch or image size if you hit CUDA OOM.

## Results snapshot (from comparison_report.json)

Baseline (yolo26s_traffic_v1):

- mAP50: 0.9290
- mAP50-95: 0.8704
- FPS (GPU, warmup=10, 190 imgs): 27.1

Candidate (yolo26s_traffic_final):

- mAP50: 0.9026
- mAP50-95: 0.8472
- FPS (GPU, warmup=10, 190 imgs): 23.4

See demo-application/traffic-sign-detection/comparison_report.json for full per-class metrics.