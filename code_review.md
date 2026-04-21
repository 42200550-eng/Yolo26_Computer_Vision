# 🔍 Đánh Giá Kỹ Thuật — Code Triển Khai YOLO26 Pipeline

---

## Tổng Quan Triển Khai

| Hạng mục | Kết quả |
|:---|:---|
| Files mới tạo | **8/8** ✅ |
| Files modified | **2/2** ✅ (`data.yaml`, `test.py`) |
| Config mới | **1/1** ✅ (`data_v2.yaml`) |
| Training đã chạy | ❌ Chưa (chỉ có `traffic_detect_run3` cũ) |
| Pipeline sẵn sàng chạy? | ⚠️ **Chưa** — có bugs cần fix |

---

## Đánh Giá Chi Tiết Từng File

### ✅ Tốt — [data_audit.py](file:///d:/Download/ComputerVision/My-Deep-Learning/demo-application/traffic-sign-detection/source_code/data_audit.py) (215 lines)

**Điểm mạnh:**
- Logic validate YOLO label rất chặt chẽ — check bbox bounds, overflow, negative class
- `yolo_line_ok()` kiểm tra cả `bbox_outside_image` (x - w/2 < 0) — chi tiết và chính xác
- Output cả JSON report + matplotlib plots
- Hỗ trợ cả `names` dạng list và dict

**Vấn đề nhỏ:**
- `validate_image_pairs()` gọi `cv2.imread()` cho mỗi ảnh để check readable → rất chậm trên dataset lớn. Nên dùng `os.path.getsize()` hoặc chỉ check header bytes

**Đánh giá: 9/10** 🟢

---

### ⚠️ Cần fix — [data_balance.py](file:///d:/Download/ComputerVision/My-Deep-Learning/demo-application/traffic-sign-detection/source_code/data_balance.py) (193 lines)

**Điểm mạnh:**
- `augment_image()` chỉ dùng color augmentation (không geometric) — đúng thiết kế vì fliplr=0.0
- Oversampling logic đúng: chọn random candidate → augment → save với stem mới

> [!CAUTION]
> **BUG #1 — Thiếu train/val/test split**
> Script chỉ oversample training set, nhưng `data_v2.yaml` tham chiếu `images/val` và `images/test` → **những thư mục này không bao giờ được tạo**. Pipeline sẽ crash khi train YOLO26 vì không tìm thấy validation set.

> [!WARNING]  
> **BUG #2 — Oversampled images bump ALL class counts**
> Khi oversample 1 ảnh cho class minority (ví dụ No_parking), nhưng ảnh đó cũng chứa class khác (No_entry), thì `current[c] += 1` tăng cả class majority → metrics after balancing bị inflate.

**Đánh giá: 6/10** 🟡

---

### ✅ Tốt — [smart_augment.py](file:///d:/Download/ComputerVision/My-Deep-Learning/demo-application/traffic-sign-detection/source_code/smart_augment.py) (168 lines)

**Điểm mạnh:**
- GUI review với accept/reject/quit — UI đơn giản nhưng đủ dùng
- Defensive `TypeError` fallback cho YOLOE API
- JSON report cho traceability
- Confidence threshold 0.7 — hợp lý cho pseudo labels

> [!WARNING]
> **BUG #3 — Class mapping fragile**
> `map_class_by_prompt()` ánh xạ `prompt_idx → cls_id` theo thứ tự danh sách `DEFAULT_TEXTS`. Nếu YOLOE-26 trả về class ID theo vocabulary riêng (không theo thứ tự prompt), mapping sẽ sai hoàn toàn. Cần verify bằng cách print `pred_cls` values trước khi trust.

**Đánh giá: 7.5/10** 🟢

---

### ⚠️ Cần fix — [download_external_data.py](file:///d:/Download/ComputerVision/My-Deep-Learning/demo-application/traffic-sign-detection/source_code/download_external_data.py) (102 lines)

> [!CAUTION]
> **BUG #4 — URL rỗng + thiếu class mapping**
> - `DEFAULT_SOURCES[0]["url"]` là string rỗng `""` → script skip tất cả sources
> - Không có logic map GTSDB classes → project's 6 classes
> - `copy_flat_images()` chỉ copy ảnh, **không tạo labels** → ảnh sẽ vô dụng cho training

**Đánh giá: 4/10** 🔴

---

### ✅ Tốt — [train_yolo26.py](file:///d:/Download/ComputerVision/My-Deep-Learning/demo-application/traffic-sign-detection/source_code/train_yolo26.py) (71 lines)

**Điểm mạnh:**
- Tất cả hyperparameters đúng theo plan: `fliplr=0.0`, `copy_paste=0.3`, `close_mosaic=15`
- CLI interface sạch với defaults hợp lý
- `cos_lr=True` + `warmup_epochs=5` — best practice

**Vấn đề nhỏ:**
- Thiếu `amp=True` (mixed precision) — tăng tốc training ~30%
- Thiếu `deterministic=True` — cần cho reproducibility
- `lrf=0.001` đúng nhưng nên comment giải thích: "final_lr = lr0 × lrf = 1e-5"

**Đánh giá: 8.5/10** 🟢

---

### ⚠️ Cần fix — [tune_yolo26.py](file:///d:/Download/ComputerVision/My-Deep-Learning/demo-application/traffic-sign-detection/source_code/tune_yolo26.py) (68 lines)

> [!WARNING]
> **BUG #5 — Sai parameter name trong `train_final()`**
> ```python
> model.train(cfg=cfg_path, ...)  # ❌ 'cfg' là cho model config, không phải hyperparameters
> ```
> Ultralytics dùng **trực tiếp đọc yaml rồi unpack** hoặc override từng param. `cfg` parameter load **model architecture YAML** (yolo26.yaml), không phải training hyperparameters. Cần fix thành đọc yaml + unpack kwargs.

**Đánh giá: 6.5/10** 🟡

---

### ⚠️ Cần fix — [evaluate.py](file:///d:/Download/ComputerVision/My-Deep-Learning/demo-application/traffic-sign-detection/source_code/evaluate.py) (83 lines)

> [!CAUTION]
> **BUG #6 — Thiếu per-class metrics**
> Plan yêu cầu so sánh per-class (đặc biệt No_parking recall), nhưng `evaluate_model()` chỉ trích xuất **aggregate** metrics (map50, mp, mr). Thiếu hoàn toàn:
> - Per-class AP, precision, recall
> - Confusion matrix comparison
> - Small/medium/large object breakdown

> [!WARNING]
> **BUG #7 — Benchmark không warm-up**
> `benchmark_speed()` tính FPS bao gồm cả first inference (model compilation, CUDA warmup) → kết quả bị skew thấp. Cần chạy 5-10 warmup iterations trước khi đo.

**Đánh giá: 5.5/10** 🟡

---

### ✅ Tốt — [export_deploy.py](file:///d:/Download/ComputerVision/My-Deep-Learning/demo-application/traffic-sign-detection/source_code/export_deploy.py) (56 lines)

**Điểm mạnh:**
- Try/except cho từng format — graceful error handling
- Validate ngay sau export — đảm bảo quality không drop

**Vấn đề:**
- Thiếu `half=True` cho TensorRT export (FP16 nhanh gấp đôi)
- Thiếu `simplify=True` cho ONNX (giảm graph nodes 20-30%)
- `engine` format sẽ fail nếu không có NVIDIA GPU + TensorRT SDK

**Đánh giá: 7/10** 🟢

---

### ✅ Tốt — [test.py](file:///d:/Download/ComputerVision/My-Deep-Learning/demo-application/traffic-sign-detection/source_code/test.py) (74 lines)

**Điểm mạnh:**
- FPS overlay trên video — trực quan khi demo
- Argparse cho weights/input/output/conf — flexible
- Comment rõ "YOLO26 forward pass already includes end-to-end post-processing"

**Vấn đề nhỏ:**
- `fps = frame_idx / elapsed` — frame 0 → `0/elapsed = 0` luôn, nên dùng `(frame_idx + 1)`
- Confidence text bị đè dưới box (`y + h + 16`) — nếu box ở đáy frame sẽ bị cắt

**Đánh giá: 8/10** 🟢

---

### ✅ Tốt — [data_v2.yaml](file:///d:/Download/ComputerVision/My-Deep-Learning/demo-application/traffic-sign-detection/data_v2.yaml)

Config chuẩn format, có test split. Tuy nhiên:
- `path: datasets_v2` là relative → cần chạy từ đúng working directory
- Phụ thuộc `data_balance.py` tạo đúng cấu trúc thư mục

**Đánh giá: 8/10** 🟢

---

## Tổng Hợp Bugs & Critical Gaps

### 🔴 Bugs Phải Fix (chặn pipeline)

| # | File | Bug | Severity |
|:--|:-----|:----|:---------|
| 1 | `data_balance.py` | **Thiếu val/test split** — YOLO26 training crash | 🔴 Critical |
| 2 | `data_balance.py` | Oversampled images inflate majority class counts | 🟡 Medium |
| 3 | `smart_augment.py` | Class mapping phụ thuộc prompt order, fragile | 🟡 Medium |
| 4 | `download_external_data.py` | URL rỗng + thiếu labels + thiếu class mapping | 🔴 Critical |
| 5 | `tune_yolo26.py` | `cfg=` sai parameter — load model config thay vì hyps | 🔴 Critical |
| 6 | `evaluate.py` | Thiếu per-class metrics (core requirement) | 🔴 Critical |
| 7 | `evaluate.py` | Benchmark không warm-up → FPS sai | 🟡 Medium |

### 🟡 Critical Gaps (thiếu hoàn toàn)

| # | Thiếu gì | Tại sao cần |
|:--|:---------|:------------|
| G1 | **Script stratified split** (train/val/test) | `data_v2.yaml` reference val+test nhưng không ai tạo |
| G2 | **Pipeline runner script** | 8 scripts riêng lẻ, không có cách chạy sequential tự động |
| G3 | Per-class recall trong evaluate | Mục tiêu chính: No_parking recall 25%→70% |
| G4 | Confusion matrix side-by-side | Plan yêu cầu nhưng evaluate.py không implement |
| G5 | README/documentation cho pipeline mới | 8 scripts mới không có hướng dẫn sử dụng |

---

## Score Card

```
data_audit.py         ██████████░  9.0/10  🟢
data_balance.py       ██████░░░░░  6.0/10  🟡 (2 bugs)
smart_augment.py      ████████░░░  7.5/10  🟢
download_external.py  ████░░░░░░░  4.0/10  🔴 (placeholder)
train_yolo26.py       █████████░░  8.5/10  🟢
tune_yolo26.py        ███████░░░░  6.5/10  🟡 (1 bug)
evaluate.py           ██████░░░░░  5.5/10  🟡 (2 bugs)
export_deploy.py      ███████░░░░  7.0/10  🟢
test.py               ████████░░░  8.0/10  🟢
data_v2.yaml          ████████░░░  8.0/10  🟢
────────────────────────────────────────────
OVERALL               ███████░░░░  7.0/10  🟡 Needs fixes
```

> [!IMPORTANT]
> **Verdict**: Code structure rất tốt (argparse, type hints, error handling), nhưng có **4 critical bugs** chặn pipeline chạy thực tế. Cần 1 round fix trước khi execute Phase 1-5.
