# Model Pruning & Sensitivity Analysis Framework (YOLOv5s & DETR)

A robust, SOLID-compliant structured pruning and sensitivity analysis framework for deep learning architectures, specifically supporting **YOLOv5s** and **DETR (DEtection TRansformer)** families.

---

## 🌟 Key Features

* **SOLID Architecture Design**: Bypasses hardcoding by using dynamic Model Adapters (`ModelAdapter`, `DETRPrunerAdapter`, `YOLOv5PrunerAdapter`). Highly extensible to support future models such as RT-DETR or YOLOv8.
* **Support for Transformer Architectures (DETR)**:
  * Dynamic CNN Backbone scaling supporting **ResNet-18**, **ResNet-34**, **ResNet-50**, and **ResNet-101**.
  * Custom `FrozenBatchNormPruner` registered in `torch_pruning` to avoid tensor shape mismatch crashes.
  * Attention-protection to guard the `input_proj` projection layer during pruning.
* **Isolated Dataloading**: Dynamic namespace isolation of `datasets.coco` prevents shadowing conflicts with the Hugging Face `datasets` library from `site-packages`.
* **Hardware-Accelerated Evaluations**: Fully supports CPU and GPU (CUDA) execution with customizable batch evaluation caps to prevent system swapping and Out-of-Memory (OOM) errors.

---

## 📦 Requirements & Environment Setup

Run the scripts in a Python environment with the following dependencies installed (e.g., `env_cv` conda environment):

```bash
# Activate environment (replace with your env name)
conda activate env_cv

# Install core and analysis packages
pip install torch-pruning pycocotools tqdm pandas openpyxl matplotlib
```

---

## 🚀 Step-by-Step Pruning Pipeline

Below is the step-by-step workflow for Sensitivity Analysis, Layer Selection, Evaluation, and Fine-tuning.

### 1. Sensitivity Analysis
To evaluate the impact of pruning individual layers, run the sensitivity analysis script. This applies structured pruning with `--modification prune-structured` at specified pruning rates (e.g. `[0.25, 0.5, 0.75]`), writing results to a specified output file.

#### 📊 YOLOv5s Sensitivity Analysis:
```bash
python sensitivity_analysis.py --data data/coco.yaml --img-size 640 --batch-size 32 --conf-thres 0.001 --iou-thres 0.65 --device 0 --weights yolov5s.pt --name yolov5s_640_sensitivity --modification prune-structured --prune-output yolov5s_sensitivity.txt --pruning-rate "[0.25, 0.5, 0.75]"
```

#### 📊 DETR ResNet-18 Sensitivity Analysis:
Create a placeholder file (or load your own checkpoint) to automatically initialize DETR with a ResNet-18 backbone:
```bash
# Create placeholder weights (if needed)
touch detr_resnet18.pt

# Run sensitivity analysis
python sensitivity_analysis.py --weights detr_resnet18.pt --data coco --img-size 800 --batch-size 2 --device 0 --name detr_resnet18_sensitivity --modification prune-structured --pruning-rate "[0.25, 0.5, 0.75]" --prune-output detr_resnet18_sensitivity.txt
```

#### 📊 RT-DETR-R18 (Pretrained) Sensitivity Analysis:
RT-DETR models are loaded with official COCO pretrained weights directly from Baidu's PyTorch Hub repository. They require exactly `640x640` input image size (which is automatically configured by our adapter):
```bash
python sensitivity_analysis.py --weights rtdetr_r18vd --data coco --img-size 640 --batch-size 2 --device 0 --name rtdetr_r18vd_sensitivity --modification prune-structured --pruning-rate "[0.25, 0.5, 0.75]" --prune-output rtdetr_sensitivity.txt
```

#### 📂 Output Format
The resulting output file contains a list of tuples for each layer and pruning rate combination:
```python
(layer_name, metrics, timing, num_params, flops)
```
* **`layer_name`**: Name of the pruned module (e.g. `backbone.0.body.conv1`).
* **`metrics`**: A tuple containing:
  * Mean Recall value
  * Mean Precision value
  * Mean Average Precision at IoU 0.50 (mAP@.50)
  * Mean Average Precision from IoU 0.50 to 0.95 (mAP@.50:.95)
  * List of mAP values for all classes
* **`timing`**: A tuple containing:
  * Time taken for inference
  * Time taken for Non-Maximum Suppression (NMS)
  * Combined time for inference and NMS
  * Height of the image
  * Width of the image
  * Batch size used
* **`num_params`**: Number of parameters in the model.
* **`flops`**: GFLOPs of the model.

---

### 2. Selecting Layers and Pruning Rates
Once the sensitivity analysis is complete, select the pruning parameters automatically by running `layer_selection.py`. The command prints the target pruning parameters to the terminal logs and saves them.

#### 📊 YOLOv5s Layer Selection:
```bash
python layer_selection.py --output output/yolov5s --params 7225885 --flops 16.436 --params-layers 6 --flops-layers 5
```

#### 📊 DETR ResNet-18 Layer Selection:
*(Note: Since GFLOPs calculation is bypassed for DETR and defaults to `0.0`, we set `--flops 0.0` and `--flops-layers 0`)*:
```bash
python layer_selection.py --output output/detr_resnet18 --params 28813704 --flops 0.0 --params-layers 3 --flops-layers 0
```

#### 📊 RT-DETR-R18 Layer Selection:
```bash
python layer_selection.py --output output/rtdetr_r18vd --params 21955472 --flops 0.0 --params-layers 3 --flops-layers 0
```

| Option | Description |
| :--- | :--- |
| `--output` | Output directory of sensitivity analysis |
| `--params` | Number of parameters of the original model (obtained during baseline test) |
| `--flops` | Number of GFLOPs of the original model (obtained during baseline test) |
| `--params-layers` | Number of layers to be pruned for parameter constraints |
| `--flops-layers` | Number of layers to be pruned for FLOPs constraints |

---

### 3. Testing with the Pruning Parameters
Test the model's accuracy under the selected pruning parameters to determine if fine-tuning is required:

#### YOLOv5s Evaluation:
```bash
python test.py --data data/coco128.yaml --img-size 640 --batch-size 32 --conf-thres 0.001 --iou-thres 0.65 --device 0 --weights yolov5s.pt --name yolov5s_val --modification prune-structured --pruning-params "[(0, 0.25), (42, 0.5), (45, 0.5), (48, 0.25)]"
```

#### DETR ResNet-18 Evaluation:
```bash
python test.py --data coco --img-size 800 --batch-size 2 --conf-thres 0.001 --iou-thres 0.65 --device 0 --weights detr_resnet18.pt --name detr_resnet18_val --modification prune-structured --pruning-params "[(2, 0.25), (10, 0.5)]"
```

#### RT-DETR-R18 Evaluation:
```bash
python test.py --data coco --img-size 640 --batch-size 2 --conf-thres 0.001 --iou-thres 0.65 --device 0 --weights rtdetr_r18vd --name rtdetr_r18vd_val --modification prune-structured --pruning-params "[('model.backbone.res_layers.0.blocks.0.branch2a.conv', 0.25)]"
```

---

### 4. Fine-Tuning with the Pruning Parameters
Apply structured pruning and fine-tune simultaneously to restore any accuracy loss from pruning:

#### YOLOv5s Fine-tuning:
```bash
python train.py --device 0 --batch-size 32 --data data/coco128.yaml --img-size 640 --weights yolov5s.pt --name yolov5s_fine_tuned --pruning-params "[(0, 0.25), (42, 0.5), (45, 0.5), (48, 0.25)]" --criterion 0
```

---

## 📂 Project Architecture

```
Prune/
├── datasets/                 # DETR coco dataset builders and transforms
│   ├── __init__.py           # Converts to a regular package to bypass Hugging Face shadowing
│   ├── coco.py               # Custom COCO dataset builder for DETR
│   └── transforms.py         # Transforms and augmentations for DETR
├── evaluation/               # Evaluation, adapters, and custom pruners
│   ├── core.py               # Core evaluate loop (returns mAP, metrics, params, and FLOPs)
│   ├── dataset.py            # Dataloader builders with isolated imports
│   ├── model_adapter.py      # Unified Model Adapter with FrozenBatchNormPruner registration
│   └── sensitivity.py        # Sensitivity Analysis loop
├── sensitivity_analysis.py   # CLI entry point for running sensitivity analysis
├── prune.py                  # CLI entry point for performing structured pruning
├── layer_selection.py        # Automatic layer selection using genetic/constrained search
├── test.py                   # Script for testing model on validation set
├── train.py                  # Script for model training and fine-tuning
└── README.md                 # Project documentation (this file)
```
