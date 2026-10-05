# LeNet-5 classification support

The pruning core is model-agnostic. Detector-specific behavior is isolated in the YOLO/RT-DETR adapters, COCO evaluator and detector callbacks. LeNet-5 support uses [LeNet5](../prune_framework/models/lenet5.py), [LeNet5Adapter](../prune_framework/plugins/adapters/lenet5.py), local-only classification data loaders and [classification callbacks](../experiments/classification.py).

`dataset` is an optional config section for callbacks:

```yaml
dataset:
  name: emnist # mnist, fashionmnist, or emnist
  emnist_split: balanced # byclass, bymerge, balanced, letters, digits, mnist
  root: ./data/emnist
  batch_size: 64
  num_workers: 0
  train_limit: 512
  val_limit: 256
```

`custom_47labels` reads a flat local dataset laid out as `root/images/<name>.jpg` and
`root/labels/<name>.txt`; each text file contains one integer in `0..46`. It converts
images to grayscale, applies the regular classification transform, and creates a
deterministic stratified train/validation split using `validation_fraction` (default `0.2`).

The loader always sets `download=False`. A missing cache raises a clear error; it never downloads a dataset implicitly. `train_limit` and `val_limit` select deterministic subsets for smoke runs. EMNIST defaults to `balanced` (47 classes); set `model.num_classes` to the matching split count: `byclass: 62`, `bymerge: 47`, `balanced: 47`, `letters: 26`, `digits: 10`, or `mnist: 10`. The loader converts EMNIST `letters` labels from 1–26 to 0–25 for `CrossEntropyLoss`.

`LeNet5_Numbers&Characters_FP32.onnx` is a separate 32x32 classic-LeNet topology with 47 output classes. Set `model.name: lenet5_emnist_onnx`, point `model.weights` at the ONNX file, use `model.input_shape: [1, 1, 32, 32]`, and set `dataset.image_padding: 2`. The adapter copies the verified ONNX initializers into the matching PyTorch model so sensitivity, pruning and recovery remain PyTorch operations.

| Component | Status | Notes |
|---|---|---|
| LeNet-5 model | Validated | Grayscale `1×28×28` input and configurable class logits. |
| Adapter | Validated | Conv/Linear targets; final classifier is protected. |
| Evaluator | Validated | `CrossEntropyLoss`, accuracy, loss and sample count; eval/no-grad. |
| Calibration | Validated synthetic | SNIP signed/absolute mean, GraSP, Taylor and SynFlow use scalar classification loss/input. |
| Recovery | Validated synthetic | Optimizer is constructed after pruning and masks are enforced. |
| Checkpoint | Validated synthetic, unstructured | LeNet adapter restores persistent mask parametrizations before state-dict load. Structured architecture serialization needs a topology-aware format. |
| Conv → Pool → Flatten → Linear | Validated synthetic | torch-pruning propagated structured Conv changes through flatten to Linear. |
| MNIST/FashionMNIST/EMNIST E2E | Not run | No local dataset cache was present. |

| Technique | Applicable to standard LeNet-5 | Unit smoke | MNIST E2E | Notes |
|---|---|---|---|---|
| Magnitude, SNIP, GraSP, SynFlow, LAMP | Yes | Yes | Not run | Elementwise masks on approved Conv/Linear weights. |
| Channel, Filter, Taylor | Yes | Yes | Not run | Filter only roots Conv; structured propagation was tested. |
| Block sparse | Yes, `[1, 1]` | Yes | Not run | Larger blocks must divide every approved matrix. |
| N:M 2:4 | No | Rejection tested | Not run | First Conv has flattened width 25; tail groups are deliberately unsupported. |
| L1/L0 BN regularization | No | Adapter capability tested | Not run | Standard LeNet-5 has no adapter-declared BatchNorm scale targets. |
| Layer/structural block | No | Adapter capability tested | Not run | No removable architecture block contract is declared. |
| Attention head | No | Adapter capability tested | Not run | LeNet has no attention modules. |

Use [lenet5_emnist_balanced_plan_smoke.yaml](../configs/lenet5_emnist_balanced_plan_smoke.yaml) with `--build-plan-only` for a no-dataset EMNIST plan. [lenet5_emnist_balanced_recovery_smoke.yaml](../configs/lenet5_emnist_balanced_recovery_smoke.yaml) runs prune → evaluate → recovery after EMNIST Balanced is manually cached under `./data/emnist`. [lenet5_emnist_balanced_sensitivity_full.yaml](../configs/lenet5_emnist_balanced_sensitivity_full.yaml) loads the checked-in 47-class ONNX model, then runs baseline evaluation, layer-wise sensitivity, selection, structured pruning and recovery. [lenet5_mnist_plan_smoke.yaml](../configs/lenet5_mnist_plan_smoke.yaml) remains available for the ten-class MNIST setup. [lenet5_block_sparse_smoke.yaml](../configs/lenet5_block_sparse_smoke.yaml) demonstrates the universally divisible block setting.
