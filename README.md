# Model Pruning & Sensitivity Analysis Framework (YOLOv5 & RT-DETR)

A robust, SOLID-compliant structured/unstructured pruning and sensitivity analysis framework for deep learning architectures, supporting **YOLOv5** and **RT-DETR (Real-Time DEtection TRansformer)** families.

---

## 🌟 Key Features

* **SOLID Architecture Design**: Bypasses hardcoding by using dynamic Model Adapters (`ModelAdapter`, `DETRPrunerAdapter`, `YOLOv5PrunerAdapter`). Highly extensible to support future models.
* **Hugging Face RT-DETR Integration**:
  * Native support for `RTDetrForObjectDetection` and `RTDetrImageProcessor` from Hugging Face `transformers`.
  * Dynamic channel pruning of RT-DETR ResNet/HGNet backbones, encoders, and decoders using `torch_pruning`.
  * Auto-exclusion/protection for input projection layers, bbox predictors, classifier embed heads, and residual additions.
* **Isolated Dataloading**: Dynamic namespace isolation of `datasets.coco` and `dataset_coco_rtdetr` prevents shadowing conflicts with the Hugging Face `datasets` library from `site-packages`.
* **Hardware-Accelerated Evaluations**: Fully supports CPU and GPU (CUDA) execution with customizable batch evaluation caps to prevent system swapping and Out-of-Memory (OOM) errors.

---

## 📦 Requirements & Environment Setup

Run the scripts in a Python environment with the following dependencies installed (e.g., `env_cv` conda environment):

```bash
# Activate environment (replace with your env name)
conda activate env_cv

# Install core and analysis packages
pip install torch-pruning pycocotools tqdm pandas openpyxl matplotlib transformers
```

---

## 🚀 Step-by-Step Pruning Pipeline

Below is the step-by-step workflow for Sensitivity Analysis, Layer Selection, Evaluation, and Fine-tuning.

### 1. Sensitivity Analysis
To evaluate the impact of pruning individual layers, run the sensitivity analysis. This applies structured pruning layer-by-layer at specified pruning rates (e.g., `0.3`), writing results to a specified output file.

#### 📊 YOLOv5s Sensitivity Analysis:
```bash
python sensitivity_analysis.py --data data/coco.yaml --img-size 640 --batch-size 32 --conf-thres 0.001 --iou-thres 0.65 --device 0 --weights yolov5s.pt --name yolov5s_640_sensitivity --modification prune-structured --prune-output yolov5s_sensitivity.txt --pruning-rate "[0.05, 0.10, 0.20, 0.30, 0.50 , 0.75]"
```

#### 📊 RT-DETR-R18 Sensitivity Analysis:
RT-DETR sensitivity analysis evaluates all 56 prunable convolutional layers in the backbone, encoder, and decoder.
```bash
python test_rtdetr.py --data data/coco_1000.yaml --weights PekingU/rtdetr_r18vd --batch-size 32 --device 0 --ann-file ./coco/annotations/instances_val2017.json --img-dir ./coco/images/val2017 --name rtdetr_r18vd_sensitivity --task pruning_sensitivity_analysis --pruning-rate "[0.05, 0.10, 0.20, 0.25, 0.50, 0.75]" --modification prune-structured
```

#### 📂 Output Format
The resulting output file contains a list of tuples for each layer and pruning rate combination:
```python
(layer_name, metrics, timing, num_params, flops)
```
* **`layer_name`**: Index of the pruned module (or name/string depending on model adapter).
* **`metrics`**: A tuple containing:
  * Mean Recall value (0.0 placeholder for RT-DETR)
  * Mean Precision value (0.0 placeholder for RT-DETR)
  * Mean Average Precision at IoU 0.50 (mAP@.50)
  * Mean Average Precision from IoU 0.50 to 0.95 (mAP@.50:.95)
  * List of mAP values for all classes
* **`timing`**: A tuple containing timing details and shapes.
* **`num_params`**: Number of parameters in the model.
* **`flops`**: GFLOPs of the model.

---

### 2. Selecting Layers and Pruning Rates
Once the sensitivity analysis is complete, select the pruning parameters automatically by running `layer_selection.py`. The command prints the target pruning parameters to the terminal logs and saves them.

#### 📊 YOLOv5s Layer Selection:
```bash
python layer_selection.py --output output/yolov5s --params 7225885 --flops 16.436 --params-layers 6 --flops-layers 5
```

#### 📊 RT-DETR-R18 Layer Selection:
Use the baseline parameters and GFLOPs (obtained from running baseline evaluation or printed at start of sensitivity) to select target layers:
```bash
python layer_selection.py --output output/rtdetr_r18vd --params 20161384 --flops 60.287 --params-layers 6 --flops-layers 5
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

#### RT-DETR-R18 Evaluation:
Evaluate the RT-DETR model with specific layer pruning parameters:
```bash
# Structured Pruning Evaluation
python test_rtdetr.py --data data/coco_5.yaml --weights PekingU/rtdetr_r18vd --batch-size 8 --device 0 --ann-file ./coco/annotations/instances_val2017.json --img-dir ./coco/images/val2017 --task baseline --modification prune-structured --pruning-params "[(12, 0.3), (13, 0.3)]"

# Unstructured Pruning Evaluation
python test_rtdetr.py --data data/coco_5.yaml --weights PekingU/rtdetr_r18vd --batch-size 8 --device 0 --ann-file ./coco/annotations/instances_val2017.json --img-dir ./coco/images/val2017 --task baseline --modification prune-unstructured --pruning-rate 0.3
```

---

### 4. Performing Model Pruning and Exporting Weights
Export the pruned model weights once you select or test specific layer indices:

#### RT-DETR-R18 Model Pruning:
```bash
python prune_rtdetr.py --weights PekingU/rtdetr_r18vd --output-path pruned_rtdetr.pt --pruning-params "[(12, 0.3), (13, 0.3)]" --criterion 0 --modification prune-structured
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
├── dataset_coco_rtdetr.py    # Custom COCO dataset wrapper for RT-DETR
├── prune_rtdetr.py           # CLI script to prune RT-DETR models structured/unstructured
├── test_rtdetr.py            # CLI script to evaluate and run sensitivity analysis for RT-DETR
└── README.md                 # Project documentation (this file)
```
