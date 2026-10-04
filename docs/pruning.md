# Pruning Framework

## Overview

Framework tổ chức pruning detector qua adapter, importance criterion, calibration, sensitivity, policy và executable plan. Đây là tài liệu chính cho code hiện tại; số liệu trong [REPORT](../REPORT.md) thuộc pipeline lịch sử.

| Khái niệm | Vai trò |
|---|---|
| Criterion | Tính importance scores từ trọng số, gradient hoặc trạng thái đã học. |
| Strategy / pruner | Cơ chế thay đổi model: mask weight, bỏ channel hoặc bỏ block. |
| Sensitivity | Đo tác động của từng candidate ratio bằng temporary pruning và evaluation. |
| Policy | Chọn layer/group và tỷ lệ cần prune. |
| Plan | Các mutation cụ thể: module, operation, indices và dependencies. |
| Dependency | Ràng buộc giữa producer, normalization, consumer và các nhánh graph. |
| Recovery | Huấn luyện sau pruning qua callback. |
| Deployment target | Điều kiện chấp nhận model dựa trên metric/cost đo được. |

## Status levels

- **Implemented**: code/contract tồn tại.
- **Validated**: có unit, tiny topology, mock callback hoặc synthetic integration tests; không đồng nghĩa detector experiment.
- **Experimentally validated**: đã chạy detector và dataset thật end-to-end bằng unified flow hiện tại.

| Capability | Implemented | Validated | Experimentally validated |
|---|---|---|---|
| Unified orchestration, baseline snapshot, build-plan-only | Có | Unit/tiny/mock | Chưa end-to-end |
| Calibration → sensitivity → policy → plan | Có | Contract/tiny/mock | Chưa full detector sweep |
| Structured dependency pruning | Có | Tiny Conv-BN, consumer, residual, concat, protected head, remapping | Chưa YOLOv5 full topology end-to-end |
| Unstructured và depth mechanisms | Có, tùy adapter/criterion | Unit/mock | Chưa end-to-end trong flow hiện tại |
| Evaluation normalization và target checker | Có | Unit, synthetic evaluator integration | Chưa xác nhận toàn bộ pruning experiment |
| YOLOv5 recovery | Fine-tune callback sau prune | Unit + tiny pipeline (mask/structured/checkpoint) | Chưa detector + dataset thật |
| Attention-head | Masked Q/K/V + output-projection slices | Tiny MHA + RT-DETR protocol fixture | Chưa detector thật; không physical compression |
| N:M sparsity | Magnitude mask, N non-zero trong mỗi group M | Linear/Conv, tie, optimizer, checkpoint | Chưa detector thật |
| Block sparse | Magnitude mask rectangular blocks | Linear/Conv, tie, optimizer | Chưa detector thật |
| Pruned checkpoint reload, export, hardware measurements | Có các đường xử lý tương ứng | Không chứng nhận bằng bảng này | Chưa xác nhận round-trip end-to-end |

Các mức Validated mô tả phạm vi test đã có trong repository. Lần cập nhật documentation này không chạy lại suite và không tạo kết quả experiment.

## Current flow and execution boundaries

```text
Config validation + callback resolution
→ Model + plugin compatibility
→ Baseline snapshot
→ Normalized baseline evaluation / complexity on clone
→ Optional pre-pruning regularization
→ Module compatibility + dependency preflight
→ Calibration
→ Sensitivity Analysis (optional)
→ Sensitivity-aware policy (or configured ratio/layer_params)
→ Build pruning plan → Validate plan
→ [BUILD-PLAN-ONLY STOP]
→ Apply plan → Architecture validation
→ Recovery (optional)
→ Normalized evaluation / complexity
→ Optional PyTorch latency measurement
→ Deployment target check
→ Finalize / ONNX export / checkpoint / result artifacts
```

Đây là thứ tự trong [unified.py](../prune_framework/pipelines/unified.py). Latency hiện được đo **trước** target check để cung cấp metric cho checker. Không có stage benchmark ONNX/TensorRT đã triển khai trên thiết bị đích ở cuối flow.

**Dry-run** parse/validate config và mô tả orchestration qua `describe_flow`; không khởi tạo pipeline, import callback người dùng, load model/data hay ghi experiment artifacts. Nó không chứng nhận checkpoint, callback, dependency graph hoặc dữ liệu thực sự chạy được.

**Build-plan-only** khởi tạo pipeline, có thể load model, evaluate baseline, calibration/backward, chạy sensitivity và validate graph trên shadow models. Nó lưu plan rồi dừng trước final apply, recovery, latency benchmark và export. Temporary pruning trên clone vẫn xảy ra; complexity/FLOPs có thể vẫn được tính. Không phải zero-compute. Pre-pruning regularization training bị từ chối ở chế độ này; cần checkpoint đã chuẩn bị. Adapter preparation vẫn có thể chuẩn hóa model (ví dụ FrozenBN của RT-DETR).

`ModelSnapshot` deepcopy model; mỗi restore tạo bản sao độc lập gồm architecture và weights. Đây không phải baseline checkpoint đã serialize hoặc cơ chế đóng băng `requires_grad`.

## Importance Criterion != Pruning Strategy

L1, L2, magnitude, Random, Taylor, SNIP, GraSP, SynFlow, LAMP, BN scale và L0 gate là criteria. Unstructured, structured channel và depth/block là mechanisms.

```text
Criterion → Importance Scores → Sensitivity / Selection
          → Policy → Pruning Plan → Pruner
```

SNIP tính điểm cho Conv2d/Linear không có nghĩa SNIP là structured pruning. Criterion có thể được registry nhận diện nhưng không tương thích với strategy hoặc target mà adapter expose.

## Sensitivity Analysis

[SensitivityAnalyzer](../prune_framework/modules/analysis/sensitivity.py) tham gia trực tiếp unified pipeline:

```text
Calibrated baseline
→ restore independent baseline clone
→ probe one layer/group at one candidate ratio
→ temporary prune → evaluate metric drop
→ discard probe; restore baseline for next candidate
→ SensitivityResult → selector → SelectionResult / layer-wise policy
→ final plan
```

Calibration chạy trước probes. Kết quả calibration đã detach được dùng chung cho probes và final planning; không tính lại gradient trên model đã bị prune tích lũy. Baseline sensitivity được evaluate trên bản restore; mỗi probe bắt đầu từ baseline riêng.

`SensitivityResult` chứa các `LayerSensitivityProfile`; mỗi `SensitivityPoint` lưu `rate`, `score`, `metric_drop`, `relative_drop`, `is_valid`, `error_message`. Probe lỗi được đánh dấu invalid. Không có profiles hoặc tất cả invalid làm unified pipeline fail. Không đo params/FLOPs/latency riêng cho từng probe.

Selector `sensitivity` lấy **tỷ lệ đã thử lớn nhất** không vượt `pruning.amount` và có relative drop không vượt `max_allowed_relative_drop`; không có điểm phù hợp thì chọn 0. Vì vậy policy có thể khác nhau giữa các layer. Đây không phải solver ngân sách toàn model và không đảm bảo tổng accuracy drop sau khi prune đồng thời.

Policy trả về thật sự đi vào `engine.build_plan`; với structured pruning, danh sách layer ratios được ưu tiên hơn yêu cầu global scalar. Ví dụ khái niệm: layer A → 0.10, layer B → 0.25. Đây không phải số liệu experiment.

Giới hạn hiện tại:

- Không chạy layer-wise sensitivity cho depth hoặc global unstructured selection, kể cả criteria có `global_selection=True`.
- L0 forced indices không được override policy từ sensitivity; cấu hình xung đột bị reject.
- Calibration payload chung không tự đảm bảo evaluator dùng cùng subset/augmentation/RNG cho mọi probe; callback phải kiểm soát điều đó.
- Chi phí xấp xỉ số target × số candidate rates lần probe/evaluation, cộng graph building và model copies. Chưa chạy full detector sweep bằng flow hiện tại.

## Calibration

[CalibrationContext](../prune_framework/modules/calibration/context.py) chứa `batches`, `loss_fn`, `seed`, `sample_count` tùy chọn và `device` tùy chọn. `batches` nhận iterable, thường lấy từ dataloader; validation materialize thành list không rỗng. Context không có field `num_batches`: số batch được giới hạn qua `pruning.calibration_batches` khi callback chuẩn bị dữ liệu, hoặc xác định từ danh sách batches.

Callback `pruning.calibration_callback` trả context một lần. `loss_fn(candidate, batch)` cung cấp loss cho runner; không yêu cầu callback tự gọi backward. Calibration phải đi trước sensitivity vì importance cần nhất quán khi so sánh probes và tạo final plan.

- Taylor/SNIP: task data và first-order gradients.
- GraSP: task data và higher-order/Hessian-vector calibration.
- SynFlow: data-free input tổng hợp và backward riêng; không nhận task-data callback.
- L1/L2/magnitude/Random/LAMP và deterministic BN/gate scoring: không cần calibration gradient.

Runners khôi phục gradients, mode, requires-grad, buffers và RNG mà chúng quản lý sau calibration. Chúng không thay thế trách nhiệm kiểm soát randomness trong callback tùy chỉnh.

`experiments.yolov5:calibrate_taylor` nhận local data YAML hoặc dataloader, lấy số batch cấu hình, chuẩn hóa ảnh và dùng YOLO `ComputeLoss`. Checkpoint phải có loss hyperparameters `model.hyp`. Đường fallback calibration YOLO dùng ảnh ngẫu nhiên/empty labels không đại diện dataset; không dùng nó làm bằng chứng experimental.

## Compatibility matrix

Bảng dựa trên [criterion metadata](../prune_framework/contracts/criterion.py) và [criteria implementations](../prune_framework/plugins/criteria). “Có” nghĩa code chấp nhận ở mức criterion/mechanism; adapter và dependency validation vẫn giới hạn target thực tế.

| Criterion | Unstructured | Structured | Depth | Needs data | Needs gradient | Conv2d | Linear | Notes |
|---|---|---|---|---|---|---|---|---|
| L1 / L2 / magnitude | Có | Có | Có* | Không | Không | Có | Có | Unstructured dùng absolute weight scores |
| Random | Có | Có | Có* | Không | Không | Có | Có | Random importance |
| Taylor | Không | Có | Không | Có | Có | Có | Hạn chế** | `taylor`, `taylor_first_order` |
| SNIP | Có | Không | Không | Có | Có | Có | Có | Intrinsic global selection |
| GraSP | Có | Không | Không | Có | Có, higher-order | Có | Có | Intrinsic global selection |
| SynFlow | Có | Không | Không | Không, synthetic input | Có | Có | Có | Intrinsic global selection |
| LAMP | Có | Không | Không | Không | Không | Có | Có | Intrinsic global selection |
| BN scale / BN-L1 | Không | Có | Không | Không để score | Không để score | Conv-BN | Không | Cần BN ownership |
| Hard-concrete / L0 gate | Không | Có | Không | Không để score | Không để score | Conv-BN + gate | Không | Học gate cần training/data/gradient |

\* Depth dùng block targets và scoring do adapter cung cấp; không phải áp dụng tùy ý criterion lên mọi layer. Khả năng chọn block phụ thuộc adapter.

\** Taylor có score implementation cho Conv/Linear nhưng unified calibration targets hiện chỉ là Conv output channels; không quảng bá unified Taylor Linear pruning.

Fail-fast: SNIP/GraSP/SynFlow/LAMP với structured/depth; Taylor với unstructured; BN criteria trên Linear hoặc Conv không có BN ownership; layer-wise policy với mechanism không nhận policy; global unstructured với layer-wise policy; forced indices override policy. Callback calibration trên criterion không tiêu thụ task-data calibration cũng bị reject.

Registry có granularities `weight`, `channel`, `filter`, `block`, `layer`, `head`. `attention_head` dùng masked-head semantics qua adapter; nó không thay đổi `embed_dim` hoặc hứa hẹn structural compression. `nm` và `block_sparse` là mechanisms riêng, không phải global unstructured threshold.

## Structured pruning and dependency safety

Structured pruning thay đổi architecture thật, ví dụ khái niệm `Conv2d.out_channels: 64 → 56`. Dependency graph cần cập nhật BN và input channels của consumer:

```text
producer Conv → BN → consumer Conv
             ↘ residual / concat branches
```

Pruner dùng `torch_pruning`, kiểm tra từng dependency group rồi mô phỏng cả sequence và forward trên shadow copy. Khi apply, graph được rebuild cho từng action; indices được remap từ baseline coordinates qua những channels đã bị dependency action trước đó xóa.

Tiny tests bao phủ Conv-BN, consumer Conv, residual Add, concat, protected output/head, indirect protection, stale indices/remapping và graph rebuild. YOLO bảo vệ module Detect/Segment và các tên/ancestor liên quan detect/segment/anchor; không loại mọi `m.*` một cách rộng.

Đây là **Validated trên tiny topology**. YOLOv5 full-topology end-to-end structured pruning **chưa Experimentally validated**. Validation không phải chứng minh an toàn cho mọi graph hoặc transactional rollback cho mutation tùy ý.

Unstructured pruning tạo weight sparsity nhưng giữ dense tensor shapes; không đảm bảo dense runtime nhanh hơn. Structured/depth giảm architecture cũng không đảm bảo speedup trên mọi phần cứng.

## Policy and plan

`SensitivityResult` là đo lường; `SelectionResult` là policy layer index/name/ratio. `PruningPlan` chứa executable groups: primary target, exact indices, operation, dependencies, validation status/error và metadata. Policy nói **what should be pruned**; plan nói **exact executable mutations**.

Engine tách `build_plan` khỏi `execute(..., plan=...)`. Validation kiểm tra ratio/index hợp lệ, duplicated layer/action, protected target/dependency, unsupported action, stale shape/index và sequence consistency. Plan rỗng khi yêu cầu prune dương bị reject trừ `allow_noop: true`; zero-action do rounding có thể được phép trong probe. Đây là kiểm tra theo từng pruner, không phải một generic validator chứng minh mọi topology.

Artifacts có thể gồm `config.resolved.json`, `baseline.json`, `calibration*.json`, `sensitivity.json`, `selection.json`, `pruning_plan.json`; sensitivity/selection chỉ xuất khi stage tương ứng chạy. `target_check.json` và `result.json` thuộc execution sau planning. Run thất bại có thể để lại artifact directory; directory tồn tại không chứng minh thành công.

## Deployment targets

[DeploymentTargets / TargetChecker](../prune_framework/contracts/deployment.py) độc lập với tỷ lệ yêu cầu `pruning.amount`.

| Field | Điều kiện đo |
|---|---|
| `parameter_reduction` | `1 - current.params / baseline.params` đạt ngưỡng tối thiểu |
| `flops_reduction` | Tương tự với FLOPs; thiếu measurement thì unavailable |
| `max_map50_drop` | `baseline.map50 - current.map50` không vượt ngưỡng |
| `max_map50_95_drop` | Absolute drop của `map50_95`, không phải relative drop |
| `max_latency_ms` | `current.total_ms` không vượt ngưỡng |
| `max_memory_mb` | Explicit `current.memory_mb`; không dùng model storage size thay peak memory |

Giá trị phải finite, không âm; reduction/drop trong [0, 1]. Metric thiếu hoặc không finite đi vào `unavailable`; vi phạm đi vào `violations`; cả hai đều làm `reached=False`. Unified pipeline ghi kết quả rồi raise trước export/checkpoint khi target không đạt. Không có iterative target search: `pruning.iterative_steps` phải bằng 1.

Latency phụ thuộc measurement được đưa vào checker; PyTorch benchmark có thể thay `total_ms` của evaluator. Phải ghi rõ measurement protocol trong experiment tương lai, không coi hai phép đo là tương đương deployed inference.

## Evaluation

[EvaluationResult và normalizers](../prune_framework/contracts/evaluation.py) chuẩn hóa callback về `metrics`, `num_samples`, `metadata`. `normalize_evaluation` nhận scalar, nonempty numeric metric dict hoặc `EvaluationResult`; reject boolean, nonfinite và kiểu không hợp lệ. Alias `map` / `map50_95` phải nhất quán.

YOLO [test.test](../test.py) trả `(results, maps, times)`:

```text
results = (precision, recall, map50, map50_95, box_loss, obj_loss, cls_loss)
maps    = per-class AP array
times   = (inference_ms, nms_ms, total_ms, image_height, image_width, batch_size)
```

Normalizer kiểm tra tuple lengths 3/7/6, finite metrics, nonnegative timing và finite one-dimensional maps. Per-class AP được giữ trong metadata; class không có observations có thể mang aggregate AP fallback. Timing là milliseconds per image. Đây là AP theo YOLO utilities của repository, không phải tuyên bố official COCOeval.

Callback `experiments.yolov5:evaluate` gọi evaluator và normalizer. Khi truyền dataloader, sample count lấy từ dataset length; khi callback tự dựng loader, `num_samples` hiện mặc định 0, không phải bằng chứng đã evaluate 0 ảnh. Target checker dùng các normalized metrics.

## YOLOv5 Integration

**Implemented:** adapter checkpoint loading (ưu tiên EMA nếu có), evaluator normalization, calibration callback, structured dependency integration và protected head logic.

**Validated:** synthetic evaluator integration, tiny topology và callback/orchestration contracts. File test mang tên integration không tự chứng minh full COCO evaluation.

**Chưa Experimentally validated trong unified flow:** full structured pruning + recovery, full sensitivity sweep, structured checkpoint reload round-trip, export sau pruning và full hardware benchmark. Không dùng smoke test hoặc báo cáo cũ để xác nhận các capability này. Alias YOLOv7 không phải một implementation độc lập đã kiểm chứng; không có ResNet adapter hiện hành.

## RT-DETR Integration

Adapter có Conv weight/output-channel targets và xử lý chuẩn bị model; Linear projections không được expose như general channel-pruning targets. `RTDetrSelfAttention` với protocol Q/K/V/O rõ ràng được expose riêng cho masked-head pruning; adapter không đoán attention từ Linear bất kỳ. Có block-removal path trên ModuleList encoder/decoder với sửa count/link, nhưng phạm vi kiểm chứng là unit/mock, chưa full detector experiment.

Adapter không triển khai channel-sparsity regularization hoặc integrated gradient loss đầy đủ. Gradient criteria cần callback phù hợp và vẫn phải qua compatibility checks. Recovery chưa tích hợp thành training flow thực cho RT-DETR. `from_pretrained` có thể cần network nếu dùng model ID thay local directory.

## CLI

Chạy từ repository root trong môi trường đã có dependencies của project:

```bash
python main.py --config configs/research_unified.yaml --dry-run
```

Planning command dưới đây **có compute** và yêu cầu checkpoint/data/loss hyperparameters phù hợp. Documentation task không chạy command này:

```bash
python main.py --config configs/research_unified.yaml --build-plan-only
```

Các flags hiện có: `--list-models`, `--list-pruners`, `--list-criteria`, `--list-granularities`, `--list-selectors`; overrides `--model`, `--strategy`, `--criterion`, `--granularity`, `--weights`, `--amount`, `--output-path`. Khi truyền cả hai boundary flags, dry-run được ưu tiên. Một run recovery thật cần local train split và `recovery.kwargs.data`; test tiny dùng injected dataloader/loss thay vì detector dataset.

## Unified configuration

Source of truth cho fields là [config.py](../prune_framework/core/config.py); ví dụ thực tế: [research_unified.yaml](../configs/research_unified.yaml).

| Section thực | Vai trò |
|---|---|
| `model` | `name`, `weights`, `device`, `input_shape` |
| `dataset` | Optional classification callback settings: local MNIST/FashionMNIST cache, loader and smoke subset limits |
| `pruning` | Mechanism/criterion/granularity/amount, layer_params, calibration settings |
| `analysis` | Compatibility settings cho YAML cũ |
| `sensitivity` | enabled, candidate rates, selector, relative-drop threshold, metric direction |
| `evaluation` | enabled, callback, metric, kwargs |
| `targets` | Deployment constraints |
| `regularization` | Optional pre-pruning training settings; không tự áp vào post-prune recovery |
| `recovery` | enabled, epochs, callback, kwargs |
| `benchmark` | enabled, latency, flops, params, warmup, runs |
| `export` | enabled, onnx, output_path |
| `experiment` | Run name/output directory, seed, deterministic |
| `output_path` | Checkpoint destination, scalar top-level field |

Không có top-level `data`, `calibration`, `policy` hoặc `baseline`; unknown sections bị reject. Dataset đi qua `evaluation.kwargs` / `pruning.calibration_kwargs`. Policy được selector tạo hoặc cấu hình bằng `pruning.layer_params`.

Trích nguyên các section từ config mẫu (đây là phần của file, không phải một full experiment config mới):

```yaml
pruning:
  method: structured
  criterion: taylor
  structure: channel
  target_ratio: 0.30
  global: true
  min_channels: 8
  calibration_callback: experiments.yolov5:calibrate_taylor
  calibration_kwargs:
    data: data/coco50.yaml

sensitivity:
  enabled: true
  rates: [0.10, 0.20, 0.30, 0.40]
  selector: sensitivity
  max_allowed_relative_drop: 0.05

evaluation:
  enabled: true
  callback: experiments.yolov5:evaluate
  metric: map50_95
  kwargs:
    data: data/coco50.yaml

targets:
  parameter_reduction: 0.20
  max_map50_95_drop: 0.03
```

Aliases: `method → pruner`, `structure → granularity`, `target_ratio → amount`, `global → global_pruning`; conflicting aliases bị reject. Calibration còn có `calibration_batches`, `calibration_batch_size`, `calibration_seed`, `calibration_accumulate`. Config mẫu dùng local weights/data; recovery được tắt cho đến khi người dùng chỉ định train split trong `recovery.kwargs.data`, nên không phải ready-to-run detector experiment.

Benchmark defaults bật latency/params/FLOPs. `benchmark.enabled: false` riêng lẻ không tắt complexity flags; targets params/FLOPs cũng có thể yêu cầu đo complexity. Thiếu FLOPs dependency/measurement không được thay bằng số đo giả. `export.onnx: true` suy ra enabled nếu chưa đặt enabled rõ ràng.

Các mẫu khác: [YOLO structured](../configs/yolov5_structured.yaml), [YOLO unstructured](../configs/yolov5_unstructured.yaml), [RT-DETR](../configs/rtdetr_structured.yaml). File RT-DETR dù tên `structured` hiện cấu hình **depth/l2/layer**; không suy ra behavior từ filename.

## Recovery

`experiments.yolov5:recover` fine-tune đúng model đã nhận sau `apply(plan)`. Nó tạo optimizer sau pruning, hỗ trợ SGD/Adam và Step/Cosine scheduler, re-enforce persistent masks sau mỗi optimizer step, và trả `{model, metrics, metadata}`. Với local run, cần `recovery.kwargs.data` trỏ đến YAML có train split; callback dùng `ComputeLoss` của repository. `dataloader` và `loss_fn` chỉ là injection cho test/programmatic use.

Unit tests xác nhận recovery giữ unstructured zeros, dùng parameter graph structured sau prune, và pipeline tiny có thể prune → recover → checkpoint. Chưa có experiment YOLO dataset thật; build-plan-only vẫn dừng trước callback.

## Constrained sparsity and attention heads

`nm` dùng convention phổ biến **N non-zero trên mỗi M value**; sparsity weight được quyết định chính xác bởi `1 - N/M`, còn `amount` không dùng để chọn thêm weight. Với Linear grouping là `[out_features, in_features]`; Conv2d là `[out_channels, in_channels/groups × kH × kW]`. Mỗi group là đoạn contiguous theo input dimension, tail không chia hết `m` bị reject. P0 chỉ hỗ trợ magnitude/L1/L2 score, chọn `N` score lớn nhất theo stable tie-break và không revive zero từ mask cũ.

`block_sparse` mask block chữ nhật trong cùng matrix view. `block_shape: [rows, columns]` phải chia hết shape; score block là `mean(abs(weight))`, chọn block score thấp nhất với stable tie-break. Đây là sparse region bên trong weight tensor, khác với `structural_block`/DepthPruner xóa module/block kiến trúc.

`attention_head`/`head` chỉ hoạt động khi adapter khai báo target attention. RT-DETR adapter nhận exact `RTDetrSelfAttention` protocol, mask nhất quán Q/K/V output slices và output-projection input slices, gồm bias liên quan. Đây là **masked head**, checkpoint/optimizer-safe, nhưng không giảm tensor shape hay parameter count.

## Testing

[Test tree](../tests) bao gồm unit contracts, tiny structural topologies, mock callbacks và [synthetic YOLO evaluator integration](../tests/integration/test_yolo_evaluation.py). Các tests kiểm tra orchestration/calibration sharing, policy propagation, plan validation, normalization và target rejection.

Passing tests != full detector experiment validation. Không hard-code test count ở tài liệu chính; không chạy lại full suite trong lần chỉnh docs này. Xem engineering notes để biết phạm vi các lần kiểm tra trước.

## Experimental Status and Resource Constraints

Full detector sensitivity sweep có chi phí compute/memory cao do evaluation lặp lại, model snapshots và dependency graphs. Full structured pruning + recovery chưa được chạy lại với unified flow. Với resource hiện tại, project ưu tiên correctness của contracts/orchestration trước experimental validation.

Các bước tiếp theo trước khi công bố kết quả: hoàn thiện recovery, xác nhận local checkpoint/data/calibration loss, chạy experiment có protocol và seed/subset rõ ràng, kiểm tra pruned checkpoint reload/export round-trip, rồi đo trên thiết bị đích. Không có mAP, speedup hoặc benchmark mới được tạo trong cập nhật docs này.

## Historical Results

[REPORT](../REPORT.md), các legacy guides và artifacts cũ giữ lại để truy xuất lịch sử.

> These results were produced by an earlier pipeline and have not yet been revalidated against the current unified pruning flow.

Không dùng số liệu cũ hoặc evaluator smoke trên subset làm final benchmark hoặc bằng chứng full COCO/end-to-end validation hiện tại.

## Architecture Notes

- [Engineering audit trước sửa contracts](architecture/pruning-flow-audit.md): historical findings, không phải trạng thái hiện hành.
- [Implementation contracts](architecture/pruning-contracts.md): chi tiết implementation và phạm vi tiny/mock verification.
- [Documentation audit](documentation-audit.md): tài liệu được hợp nhất, sửa hoặc đánh dấu legacy.
