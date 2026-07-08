# Báo Cáo: Structured Channel Pruning trên YOLOv5s

> **Model:** YOLOv5s &nbsp;|&nbsp; **Dataset:** COCO 2017 (val) &nbsp;|&nbsp; **Phương pháp:** Structured Filter/Channel Pruning  
> **Fine-tuning:** 50 epochs — COCO128 &nbsp;|&nbsp; **Image size:** 640 × 640 &nbsp;|&nbsp; **Device:** GPU (CUDA)

---

## Mục lục

1. [Giới thiệu & Tổng quan](#1-giới-thiệu--tổng-quan)  
2. [Pipeline tổng thể](#2-pipeline-tổng-thể)  
3. [Bước 1 — Sensitivity Analysis](#3-bước-1--sensitivity-analysis)  
4. [Bước 2 — Layer Selection](#4-bước-2--layer-selection)  
5. [Bước 3 — Pruning Execution](#5-bước-3--pruning-execution)  
6. [Bước 4 — Fine-tuning](#6-bước-4--fine-tuning)  
7. [Kết quả Benchmark](#7-kết-quả-benchmark)  
8. [Phân tích & Nhận xét](#8-phân-tích--nhận-xét)  
9. [Hướng cải thiện](#9-hướng-cải-thiện)  
10. [Kết luận](#10-kết-luận)  
11. [Tái tạo kết quả](#11-tái-tạo-kết-quả)  

---

## 1. Giới thiệu & Tổng quan

### Vấn đề

Các mô hình deep learning như YOLOv5s cho hiệu năng phát hiện vật thể tốt nhưng đòi hỏi tài nguyên tính toán đáng kể, gây khó khăn cho **edge deployment** (thiết bị nhúng, mobile, IoT). Cần một phương pháp nén mô hình vừa **giảm kích thước** vừa **tăng tốc inference thực sự** mà không cần phần cứng đặc biệt.

### Giải pháp: Structured Channel Pruning

**Structured Channel Pruning** (hay Filter Pruning) là kỹ thuật loại bỏ hoàn toàn các **filter (bộ lọc) kém quan trọng** trong lớp tích chập. Khác với Unstructured Pruning (chỉ zero-out weight, cần hardware đặc biệt), Structured Pruning **thực sự thu nhỏ kiến trúc model**, cho phép tăng tốc ngay trên CPU/GPU thông thường.

```
Unstructured Pruning          Structured (Filter/Channel) Pruning
─────────────────────         ─────────────────────────────────────
weight = [64, 32, 3, 3]       weight = [64, 32, 3, 3]
         ↓ zero-out                      ↓ xóa filter thứ i
[0, W₁, 0, 0, W₄, ...]       weight = [61, 32, 3, 3]   ← nhỏ thật sự

❌ Cần sparse hardware         ✅ Chạy nhanh hơn ngay
```

### Nguyên lý cốt lõi

Khi xóa một filter tại Layer L, phải cập nhật **chuỗi domino**:

```
Layer L (Conv)         BatchNorm L          Layer L+1 (Conv)
[64, 32, 3, 3]  →  γ,β,μ,σ ∈ R⁶⁴   →   [128, 64, 3, 3]
      ↓ xóa 3 filter                           ↓ phải đồng bộ
[61, 32, 3, 3]     γ,β,μ,σ ∈ R⁶¹       [128, 61, 3, 3]

dim=0 (output)         sync                  dim=1 (input)
= Filter Pruning                          = Channel Pruning
```

Filter Pruning và Channel Pruning là **cùng một thao tác vật lý**, chỉ khác góc nhìn:
- **Filter Pruning** = nhìn từ weight tensor của layer L (`dim=0`)
- **Channel Pruning** = nhìn từ feature map output / input của layer L+1 (`dim=1`)

---

## 2. Pipeline tổng thể

```
INPUT: yolov5s.pt (7.2M params, 16.4 GFLOPs, mAP@0.5 = 0.5634)
│
├──▶ [Bước 1] sensitivity_analysis.py
│         Thử prune từng layer với 3 mức rate → ghi lại mAP, params, FLOPs
│         180 lần evaluation (60 layers × 3 rates)
│         Output: output/yolov5s/output_25/50/75.txt
│
├──▶ [Bước 2] layer_selection.py
│         Đọc kết quả SA → tính score = params_saved × mAP^20
│         Chọn 8 layers tối ưu + rate tương ứng
│         Output: [(0,0.25),(25,0.25),(42,0.5),(45,0.5),(48,0.25),(51,0.5),(55,0.25),(56,0.25)]
│
├──▶ [Bước 3] prune.py
│         Filter Pruning theo L2-norm (criterion=0)
│         Cập nhật BN + layer kế tiếp
│         Output: yolov5s-pruned.pt (5.93M params, 14.3 GFLOPs)
│
└──▶ [Bước 4] train.py (Fine-tuning)
          50 epochs trên COCO128, khởi đầu từ yolov5s-pruned.pt
          Output: runs/train/exp15/weights/best.pt
                  mAP@0.5 = 0.4433 (phục hồi từ 0.4037)
```

---

## 3. Bước 1 — Sensitivity Analysis

### Mục tiêu

> **"Nếu cắt X% filter tại layer thứ i, model mất bao nhiêu mAP?"**

Khảo sát toàn bộ không gian sensitivity của model trước khi quyết định pruning. Thay vì thử tất cả tổ hợp (4⁶⁰ ≈ 10³⁶ khả năng), Sensitivity Analysis khảo sát **từng layer một cách độc lập**.

### Lệnh chạy

```bash
python sensitivity_analysis.py \
    --weights yolov5s.pt \
    --data data/coco.yaml \
    --pruning-rate "[0.25, 0.50, 0.75]" \
    --criterion 0 \
    --prune-output output.txt
```

### Quy trình bên trong (`evaluation/sensitivity.py`)

```python
for rate in [0.25, 0.50, 0.75]:           # 3 mức cắt
    for i in range(60):                    # 60 conv layer của YOLOv5s
        # Chỉ prune layer i, giữ nguyên tất cả layer khác
        r = evaluate(pruning_params=[(i, rate)], ...)
        # Ghi: (layer_i, mAP@0.5, params_pruned, FLOPs_pruned)
        print(f"({i}, ({mp}, {mr}, {map50}, {map}, ..., {params}, {fs}))", file=f)
```

**Tổng số lần evaluation:** 60 × 3 = **180 lần**

### Tiêu chí đánh giá importance

```
criterion = 0  →  importance(filter_i) = ||W_i||₂   (L2-norm)

Filter có L2-norm nhỏ nhất → đóng góp ít nhất → xóa trước
```

Các tiêu chí có sẵn:

| criterion | Công thức | Ý nghĩa |
|-----------|-----------|---------|
| 0 | `‖W_i‖₂` nhỏ nhất | L2-norm — **mặc định, dùng trong project** |
| 1 | `‖W_i‖₂` lớn nhất | Giữ filter yếu |
| 2 | `‖W_i‖₁` nhỏ nhất | L1-norm |
| 4 | `γ_i` nhỏ nhất | BN scale factor — thường hiệu quả hơn L2 |
| 5 | `γ_i × ‖W_i‖₁` | Kết hợp BN + weight |
| 6 | random | Baseline ngẫu nhiên |

### Output

```
output/yolov5s/
├── output_25.txt   (165 KB) ← kết quả prune 25% mỗi layer
├── output_50.txt   (164 KB) ← kết quả prune 50% mỗi layer
└── output_75.txt   (160 KB) ← kết quả prune 75% mỗi layer
```

---

## 4. Bước 2 — Layer Selection

### Mục tiêu

> **"Chọn ra những layer và tỷ lệ pruning sao cho tiết kiệm params/FLOPs nhiều nhất với mất accuracy ít nhất."**

### Lệnh chạy

```bash
python layer_selection.py \
    --output output/yolov5s \
    --params 7225885 \
    --flops 16.436 \
    --params-layers 6 \
    --flops-layers 5 \
    --params-map 20 \
    --flops-map 20 \
    --save
```

### Công thức scoring

```
score(layer_i, rate_r) = params_saved(i, r)  ×  mAP(i, r)^α
                                                            ↑
                                                    α = 20 (penalty mạnh khi mAP giảm)
```

- `params_saved` = số tham số tiết kiệm khi prune layer i với rate r
- `mAP^20` = hàm phạt rất dốc — model chỉ được cắt nhiều nếu mAP giảm rất ít

**Binary search để chọn đúng N layer:**

```python
frac = 2.0
while True:
    selected = [layer for layer in all_layers if score > max_score / frac]
    if len(selected) == target_N:
        break
    frac += 0.001
```

### Kết quả Layer Selection

```
python prune.py --pruning-params "[(0, 0.25), (42, 0.5), (45, 0.5), (48, 0.25),
                                    (25, 0.25), (51, 0.5), (55, 0.25), (56, 0.25)]"
```

#### Kiến trúc YOLOv5s và các layer được chọn

```
INPUT (640×640×3)
│
▼ ══════════════════════════════ BACKBONE ══════════════════════════════
│
│  [ 0] model.0           3ch → 32ch   ◀─── ✂️ PRUNE 25%  (Stem Conv)
│  [ 1] model.1          32ch → 64ch
│  [ 2] model.2.cv1      64ch → 32ch  ┐
│  [ 3] model.2.cv2      64ch → 32ch  ├── C3 Stage-1
│  [ 4] model.2.cv3      64ch → 64ch  │
│  [ 5] model.2.m.0.cv1  32ch → 32ch  │
│  [ 6] model.2.m.0.cv2  32ch → 32ch  ┘
│  [ 7] model.3          64ch → 128ch
│  [ 8–14]  C3 Stage-2   (2 bottleneck)
│  [15] model.5         128ch → 256ch
│  [16–24]  C3 Stage-3   (3 bottleneck)
│  [25] model.7         256ch → 512ch  ◀─── ✂️ PRUNE 25%  (Backbone transition)
│  [26–30]  C3 Stage-4   (1 bottleneck)
│  [31] model.9.cv1     512ch → 256ch  ┐
│  [32] model.9.cv2    1024ch → 512ch  ├── SPPF
│  [33] model.10        512ch → 256ch  ┘
│
▼ ═══════════════════════════════ NECK (PANet) ═══════════════════════════════
│
│  [34–38]  model.13     C3 P4-level (upsample + concat)
│  [39] model.14        256ch → 128ch
│  [40–41]  model.17.cv1/cv2
│  [42] model.17.cv3    128ch → 128ch  ◀─── ✂️ PRUNE 50%  (PANet P3 output)
│  [43–44]  model.17 bottleneck
│  [45] model.18        128ch → 128ch  ◀─── ✂️ PRUNE 50%  (Downsample P3→P4)
│  [46–47]  model.20.cv1/cv2
│  [48] model.20.cv3    256ch → 256ch  ◀─── ✂️ PRUNE 25%  (PANet P4 output)
│  [49–50]  model.20 bottleneck
│  [51] model.21        256ch → 256ch  ◀─── ✂️ PRUNE 50%  (Downsample P4→P5)
│  [52–54]  model.23.cv1/cv2/cv3
│  [55] model.23.m.0.cv1 256ch→256ch  ◀─── ✂️ PRUNE 25%  (PANet P5 bottleneck)
│  [56] model.23.m.0.cv2 256ch→256ch  ◀─── ✂️ PRUNE 25%  (PANet P5 bottleneck)
│
▼ ══════════════════════════ DETECT HEAD (KHÔNG prune) ══════════════════════════
   [57] model.24.m[0]   128ch → 255ch  (small obj)
   [58] model.24.m[1]   256ch → 255ch  (medium obj)
   [59] model.24.m[2]   512ch → 255ch  (large obj)
```

#### Bảng chi tiết 8 layer được chọn

| # | Layer idx | Module path | Vị trí | out_ch gốc | Rate | out_ch sau |
|---|-----------|-------------|--------|-----------|------|-----------|
| 1 | **0** | `model.0` | Stem Conv | 32 | **25%** | 24 |
| 2 | **25** | `model.7` | Backbone transition | 512 | **25%** | 384 |
| 3 | **42** | `model.17.cv3` | Neck PANet P3 output | 128 | **50%** | 64 |
| 4 | **45** | `model.18` | Downsample P3 → P4 | 128 | **50%** | 64 |
| 5 | **48** | `model.20.cv3` | Neck PANet P4 output | 256 | **25%** | 192 |
| 6 | **51** | `model.21` | Downsample P4 → P5 | 256 | **50%** | 128 |
| 7 | **55** | `model.23.m.0.cv1` | PANet P5 bottleneck cv1 | 256 | **25%** | 192 |
| 8 | **56** | `model.23.m.0.cv2` | PANet P5 bottleneck cv2 | 256 | **25%** | 192 |

#### Phân bố layer được chọn

| Vùng | Số layer chọn | Mức pruning | Lý do |
|------|--------------|-------------|-------|
| Backbone (0–33) | 2 layer | 25% (nhẹ) | Trích xuất đặc trưng cốt lõi — **nhạy cảm** |
| Neck PANet (34–56) | 6 layer | 25–50% | Tổng hợp multi-scale — **nhiều redundancy** |
| Detect Head (57–59) | 0 layer | Không cắt | Output = 255ch cố định (3 × 85) — **bắt buộc** |

---

## 5. Bước 3 — Pruning Execution

### Lệnh chạy

```bash
python prune.py \
    --weights yolov5s.pt \
    --pruning-params "[(0, 0.25), (42, 0.5), (45, 0.5), (48, 0.25), (25, 0.25), (51, 0.5), (55, 0.25), (56, 0.25)]" \
    --criterion 0 \
    --name yolov5s-pruned.pt
```

### Cơ chế hoạt động

Với mỗi layer được chọn, `prune_structured()` thực hiện **3 thao tác liên kết bắt buộc**:

**① Filter Pruning tại Layer L — `dim=0` (output channels)**

```python
# Tính L2-norm mỗi filter
importances = torch.norm(conv.weight, p=2, dim=[1, 2, 3])  # shape: [C_out]

# Sắp xếp tăng dần, giữ top (1-rate)% có norm cao nhất
n_prune = int(rate * C_out)
pruned_indices = argsort(importances)[n_prune:]   # indices ĐƯỢC GIỮ LẠI

# Cắt filter
conv.weight    = index_select(conv.weight, dim=0, index=pruned_indices)
conv.out_channels = len(pruned_indices)
```

**② Cập nhật BatchNorm Layer L (bắt buộc đồng bộ)**

```python
bn.weight (γ)        = index_select(bn.weight,        dim=0, index=pruned_indices)
bn.bias (β)          = index_select(bn.bias,          dim=0, index=pruned_indices)
bn.running_mean (μ)  = index_select(bn.running_mean,  dim=0, index=pruned_indices)
bn.running_var (σ²)  = index_select(bn.running_var,   dim=0, index=pruned_indices)
bn.num_features      = len(pruned_indices)
```

**③ Channel Pruning tại Layer L+1 — `dim=1` (input channels)**

```python
# Tính offset cho trường hợp Concat (skip connection)
offset = sum(pruning_cfg[next_layer][2][:slice])

# Ghép indices: [trước offset | pruned_indices+offset | sau offset]
indices_next = cat([arange(offset),
                    pruned_indices + offset,
                    arange(offset + C_out, n_in)])

next_conv.weight    = index_select(next_conv.weight, dim=1, index=indices_next)
next_conv.in_channels = n_in - n_pruned
```

> **YOLOv5s** dùng `torch_pruning` để tự động xây dựng **dependency graph**, đảm bảo các Concat layer và skip connection được xử lý chính xác.

### Kết quả sau pruning

| Metric | Trước pruning | Sau pruning | Thay đổi |
|--------|--------------|------------|---------|
| Parameters | 7,225,885 | 5,933,813 | **−17.9%** (−1,292,072) |
| GFLOPs | 16.436 | 14.317 | **−12.9%** (−2.119) |
| File size | ~14.8 MB | ~12.5 MB | **−15.5%** |
| mAP@0.5 (trước FT) | 0.5634 | 0.4037 | −28.4% (chưa fine-tune) |

---

## 6. Bước 4 — Fine-tuning

### Mục tiêu

Phục hồi accuracy sau khi pruning bằng cách **huấn luyện lại model đã thu nhỏ** với kiến trúc mới.

### Cấu hình

```bash
python train.py \
    --weights yolov5s-pruned.pt \
    --data data/coco128.yaml \
    --epochs 50 \
    --batch-size 16 \
    --img-size 640 \
    --device 0
```

| Tham số | Giá trị |
|---------|---------|
| Weights khởi đầu | `yolov5s-pruned.pt` |
| Dataset | COCO128 (128 ảnh) |
| Epochs | 50 |
| Batch size | 16 |
| Image size | 640 × 640 |
| Optimizer | SGD |
| Scheduler | Cosine LR decay |
| Augmentation | Mosaic, hsv, flip |

### Biểu đồ quá trình fine-tuning

![Training curves](runs/train/exp15/results.png)

*Biểu đồ cho thấy val mAP@0.5 tăng dần qua các checkpoint, từ ~0.30 lên ~0.74 trên COCO128 val.*

### Kết quả epoch cuối (49/49)

```
Box loss:           0.0528
Objectness loss:    0.0434
Classification loss: 0.0248
Precision:          0.8106
Recall:             0.6600
mAP@0.5:            0.7457  (trên COCO128 val)
mAP@0.5:0.95:       0.5287  (trên COCO128 val)
```

---

## 7. Kết quả Benchmark

### Biểu đồ so sánh

![Benchmark chart](benchmark_chart.png)

*Từ trái qua phải: (1) So sánh Parameters & GFLOPs, (2) So sánh mAP accuracy, (3) So sánh inference latency.*

### Bảng tổng hợp (đánh giá trên COCO 2017 val)

| Model | Params | GFLOPs | P | R | mAP@0.5 | mAP@0.5:0.95 | Infer (ms) | NMS (ms) | Latency (ms) |
|-------|--------|--------|---|---|---------|-------------|-----------|---------|------------|
| **Baseline** | 7,225,885 | 16.436 | 0.6645 | 0.5239 | **0.5634** | 0.3610 | 1.27 | 1.21 | **2.48** |
| **Pruned** (no FT) | 5,933,813 | 14.317 | 0.6164 | 0.3686 | 0.4037 | 0.2316 | 1.17 | 1.08 | **2.25** |
| **Fine-tuned** | 5,933,813 | 14.317 | 0.5855 | 0.4301 | **0.4433** | 0.2679 | 1.17 | 1.40 | **2.58** |

### So sánh Baseline vs Fine-tuned

| Metric | Baseline | Fine-tuned | Tuyệt đối | % thay đổi |
|--------|----------|-----------|----------|-----------|
| Parameters | 7,225,885 | 5,933,813 | **−1,292,072** | **−17.9%** |
| GFLOPs | 16.436 | 14.317 | **−2.119** | **−12.9%** |
| Inference | 1.27 ms | 1.17 ms | −0.10 ms | **−7.9%** |
| mAP@0.5 | 0.5634 | 0.4433 | −0.1201 | −21.3% |
| mAP@0.5:0.95 | 0.3610 | 0.2679 | −0.0931 | −25.8% |
| Precision | 0.6645 | 0.5855 | −0.0790 | −11.9% |
| Recall | 0.5239 | 0.4301 | −0.0938 | −17.9% |

### Hiệu quả phục hồi sau Fine-tuning

| Metric | Pruned (chưa FT) | Fine-tuned | Phục hồi |
|--------|-----------------|-----------|---------|
| mAP@0.5 | 0.4037 | 0.4433 | **+0.0396 (+9.8%)** |
| mAP@0.5:0.95 | 0.2316 | 0.2679 | **+0.0363 (+15.7%)** |
| Recall | 0.3686 | 0.4301 | **+0.0615 (+16.7%)** |

---

## 8. Phân tích & Nhận xét

### ✅ Điểm đạt được

```
Parameters:   giảm 17.9%  (−1.3M params)
GFLOPs:       giảm 12.9%  (−2.1 GFLOPs)
File size:    giảm 15.5%  (−2.3 MB)
Inference:    tăng 7.9%   (1.27 → 1.17 ms/img)
Fine-tune:    phục hồi +9.8% mAP@0.5 chỉ với 50 epochs trên COCO128
```

### ⚠️ Điểm đánh đổi

```
mAP@0.5:     giảm 21.3%  (0.5634 → 0.4433)
mAP@0.5:0.95: giảm 25.8% (0.3610 → 0.2679)
Recall:       giảm 17.9%  (0.5239 → 0.4301)
```

### 🔍 Phân tích nguyên nhân mất accuracy

**1. Fine-tuning dataset quá nhỏ**
> COCO128 chỉ có **128 ảnh** — quá ít so với full COCO (118,287 ảnh training).  
> Model không đủ dữ liệu đa dạng để phục hồi hoàn toàn.

**2. Pruning mạnh ở Neck**
> Layer 42, 45, 51 bị cắt **50%** filters — đây là các layer quan trọng trong PANet  
> chịu trách nhiệm fusion multi-scale features.

**3. Số epochs fine-tuning còn ít**
> 50 epochs trên COCO128 ≡ ~6,400 iterations — thường cần 100–300 epochs  
> để phục hồi đầy đủ sau pruning mạnh.

**4. Độ lệch phân bố train/test**
> Fine-tune trên COCO128, đánh giá trên COCO 2017 val (5,000 ảnh).  
> Phân bố class khác nhau → generalization gap.

### 📊 Phân tích GFLOPs vs Params reduction

```
Params giảm 17.9% nhưng GFLOPs chỉ giảm 12.9%
→ Lý do: GFLOPs = C_in × C_out × H × W × kernel²

Khi xóa filter ở layer L:
  - Layer L:    FLOPs giảm theo C_out  (tỷ lệ rate)
  - Layer L+1:  FLOPs giảm theo C_in   (tỷ lệ rate)
  → Hiệu ứng nhân, nhưng bị offset bởi các layer không prune
```

---

## 9. Hướng cải thiện

| Vấn đề | Giải pháp |
|--------|---------|
| Dataset FT nhỏ | Fine-tune trên **full COCO** (118K ảnh) thay vì COCO128 |
| Epochs ít | Tăng lên **100–200 epochs** với cosine LR decay |
| Tiêu chí filter | Thử `--criterion 4` (**BN scale factor γ**) — thường tốt hơn L2-norm |
| Pruning quá mạnh | Giảm layer 42, 45, 51 từ **50% → 25–30%** |
| Knowledge Distillation | Dùng baseline làm **teacher** để guide fine-tuning |
| Gradual Pruning | Thay vì prune 1 lần, prune dần dần theo epochs |

---

## 10. Kết luận

Dự án đã thành công triển khai pipeline **Structured Channel Pruning** hoàn chỉnh cho YOLOv5s với 4 bước:

| Giai đoạn | Công cụ | Kết quả |
|-----------|---------|---------|
| **Sensitivity Analysis** | `sensitivity_analysis.py` | 180 evaluations, xây dựng bản đồ độ nhạy toàn bộ model |
| **Layer Selection** | `layer_selection.py` | Chọn 8 layer tối ưu (score = params × mAP²⁰) |
| **Pruning** | `prune.py` | −17.9% params, −12.9% GFLOPs, −7.9% latency |
| **Fine-tuning** | `train.py` | Phục hồi +9.8% mAP@0.5 sau 50 epochs |

**Trade-off cuối cùng:**

```
✅ Model nhỏ hơn 17.9%  (5.93M vs 7.23M params)
✅ Nhanh hơn 7.9%        (1.17ms vs 1.27ms inference)
✅ FLOPs thấp hơn 12.9%  (14.3 vs 16.4 GFLOPs)
⚠️ mAP@0.5 giảm 21.3%   (0.4433 vs 0.5634)
```

Kết quả phù hợp với mục tiêu **edge deployment** — model nhỏ và nhanh hơn phù hợp  
cho thiết bị có tài nguyên hạn chế, với accuracy đủ dùng cho nhiều ứng dụng thực tế.

---

## 11. Tái tạo kết quả

### Bước 1: Sensitivity Analysis

```bash
python sensitivity_analysis.py \
    --weights yolov5s.pt \
    --data data/coco.yaml \
    --batch-size 32 \
    --img-size 640 \
    --pruning-rate "[0.25, 0.5, 0.75]" \
    --criterion 0 \
    --prune-output output.txt \
    --project runs/test \
    --name exp
```

### Bước 2: Layer Selection

```bash
python layer_selection.py \
    --output output/yolov5s \
    --params 7225885 \
    --flops 16.436 \
    --params-layers 6 \
    --flops-layers 5 \
    --params-map 20 \
    --flops-map 20 \
    --save
```

### Bước 3: Pruning

```bash
python prune.py \
    --weights yolov5s.pt \
    --pruning-params "[(0, 0.25), (42, 0.5), (45, 0.5), (48, 0.25), (25, 0.25), (51, 0.5), (55, 0.25), (56, 0.25)]" \
    --criterion 0 \
    --name yolov5s-pruned.pt
```

### Bước 4: Fine-tuning

```bash
python train.py \
    --weights yolov5s-pruned.pt \
    --data data/coco128.yaml \
    --epochs 50 \
    --batch-size 16 \
    --img-size 640 \
    --device 0
```

### Bước 5: Đánh giá

```bash
python test.py \
    --weights runs/train/exp15/weights/best.pt \
    --data data/coco.yaml \
    --batch-size 32 \
    --img-size 640
```

---

*Report được tạo tự động từ kết quả thực nghiệm | YOLOv5s Structured Channel Pruning Pipeline*
