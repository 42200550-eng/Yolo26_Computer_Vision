# Traffic Sign Detection with YOLO26

Video sample:
https://youtu.be/tyLNNST96oc

## 1) Muc tieu

Muc tieu la nang cap pipeline sang YOLO26 theo huong Data First, sau do train baseline, tune, va evaluate.

## 2) Yeu cau moi truong

Chay lenh cai package:

pip install numpy matplotlib opencv-python ultralytics torch torchvision pyyaml

Neu ban dung GPU NVIDIA, hay dam bao da cai driver CUDA phu hop voi ban torch.

## 3) Cau truc script chinh

- source_code/data_audit.py: Audit label va thong ke class.
- source_code/data_balance.py: Tao bo du lieu datasets_v2 va dam bao split train val test.
- source_code/download_external_data.py: Download external data va convert labels ve YOLO format.
- source_code/smart_augment.py: Pseudo-label bang YOLOE va merge vao train split.
- source_code/train_yolo26.py: Train baseline YOLO26.
- source_code/tune_yolo26.py: Tune va train final voi hyperparameters toi uu.
- source_code/evaluate.py: Compare baseline va candidate, co per-class metrics va warmup benchmark.
- source_code/export_deploy.py: Export deployment formats.
- source_code/run_pipeline.py: Runner chay theo phase.

## 4) Chuan bi external data

File mau de cau hinh nguon external data:

- sources.template.json

Ban sao chep file nay thanh sources.json, sua URL that va class_map cho dung voi bo du lieu cua ban.

## 5) Chuoi lenh khuyen nghi de chay

Buoc 1. Audit va tao datasets_v2 co split day du:

python source_code/data_balance.py --data data.yaml --split all --ensure-splits --output-root datasets_v2 --min-instances 300

Buoc 2. Neu co external data, import vao train split:

python source_code/download_external_data.py --sources-json sources.json --dataset-out datasets_v2

Buoc 3. Neu muon pseudo-label them:

python source_code/smart_augment.py --input-dir external_data/images --out-images datasets_v2/images/train --out-labels datasets_v2/labels/train --conf 0.7

Buoc 4. Train baseline YOLO26:

python source_code/train_yolo26.py --model yolo26s.pt --data data_v2.yaml --name yolo26s_traffic_v1 --epochs 150 --imgsz 800 --batch 16 --device 0

Buoc 5. Tune va train final (tuy chon):

python source_code/tune_yolo26.py --mode tune --weights runs/detect/yolo26s_traffic_v1/weights/best.pt --data data_v2.yaml --imgsz 800 --device 0 --tune-epochs 50 --iterations 100

python source_code/tune_yolo26.py --mode final --model yolo26s.pt --data data_v2.yaml --cfg runs/detect/tune/best_hyperparameters.yaml --name yolo26s_traffic_final --imgsz 800 --device 0 --final-epochs 200

Buoc 6. Evaluate:

python source_code/evaluate.py --baseline runs/detect/traffic_detect_run3/weights/best.pt --candidate runs/detect/yolo26s_traffic_final/weights/best.pt --data data_v2.yaml --imgsz 800 --device 0 --warmup 10

## 6) Dung runner de chay nhanh

Xem toan bo lenh se chay (khong execute):

python source_code/run_pipeline.py --phase all --dry-run

Chay phase 1 den phase 3:

python source_code/run_pipeline.py --phase phase1
python source_code/run_pipeline.py --phase phase2 --sources-json sources.json
python source_code/run_pipeline.py --phase phase3 --device 0

## 7) Phuong an training de chon

### Phuong an A - Nhanh de co baseline (GPU tam trung)

- Model: yolo26s.pt
- Epochs: 100
- Image size: 768
- Batch: 16
- Device: 0

Lenh:

python source_code/train_yolo26.py --model yolo26s.pt --data data_v2.yaml --name yolo26s_quick --epochs 100 --imgsz 768 --batch 16 --device 0

### Phuong an B - Can bang giua accuracy va thoi gian (khuyen nghi)

- Model: yolo26s.pt
- Epochs: 150
- Image size: 800
- Batch: 16
- Device: 0

Lenh:

python source_code/train_yolo26.py --model yolo26s.pt --data data_v2.yaml --name yolo26s_traffic_v1 --epochs 150 --imgsz 800 --batch 16 --device 0

### Phuong an C - Toi uu chat luong cao

- Chay Phuong an B
- Tune 50 x 60 hoac 50 x 100 iterations tuy theo GPU
- Train final 200 epochs

Lenh tune vua phai:

python source_code/tune_yolo26.py --mode tune --weights runs/detect/yolo26s_traffic_v1/weights/best.pt --data data_v2.yaml --imgsz 800 --device 0 --tune-epochs 50 --iterations 60

Lenh final:

python source_code/tune_yolo26.py --mode final --model yolo26s.pt --data data_v2.yaml --cfg runs/detect/tune/best_hyperparameters.yaml --name yolo26s_traffic_final --imgsz 800 --device 0 --final-epochs 200

## 8) Kiem tra output sau train

- Baseline weights: runs/detect/yolo26s_traffic_v1/weights/best.pt
- Final weights: runs/detect/yolo26s_traffic_final/weights/best.pt
- Evaluate report: comparison_report.json

## 9) Note quan trong

- data_v2.yaml dang dung path tuong doi datasets_v2, hay chay lenh tai thu muc traffic-sign-detection.
- Neu gap CUDA OOM, giam batch truoc, sau do giam imgsz.
- Neu class minority van yeu, tang min-instances trong data_balance va bo sung external data co label chat luong.