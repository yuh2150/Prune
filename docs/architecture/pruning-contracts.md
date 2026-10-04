# Unified pruning contracts — implementation after flow audit

> Engineering implementation notes; [Pruning Framework](../pruning.md) là entrypoint hiện hành. Verification bên dưới thuộc lần triển khai contracts trước đó.

Phạm vi: hoàn thiện orchestration/contracts bằng static inspection, mocks và tiny synthetic models. Không chạy YOLO/COCO evaluation, detector pruning, fine-tuning, hardware benchmark hoặc export model thật.

## Flow trước và sau

Trước: baseline → sensitivity thiếu calibration → selector → calibration → engine vừa build vừa apply → recovery/evaluation; không deployment checker, không build-only boundary.

Sau:

```text
config validation + callback resolution
→ model load + plugin-mode compatibility
→ deep baseline snapshot
→ normalized baseline evaluation / complexity (trên clone)
→ optional pre-pruning regularization
→ module compatibility + dependency preflight
→ configured calibration callback → CalibrationContext → detached score context
→ sensitivity probes trên các bản restore độc lập
→ SelectionResult: layer indices/names/ratios
→ PruningEngine.build_plan: exact indices + dependency validation trên shadow model
→ persist baseline/policy/validated plan
→ BUILD-PLAN-ONLY STOP
→ apply validated plan (remap overlap indices; rebuild dependency graph)
→ architecture validation
→ recovery callback
→ normalized final evaluation / complexity / optional latency
→ TargetChecker
→ chỉ finalize/export/save khi target đạt
```

Targets được parse/validate từ đầu, độc lập với pruning.amount. Selector hiện vẫn dùng sensitivity cap và relative-drop constraint; không phải target-budget optimizer. TargetChecker kiểm tra các deployment constraints sau evaluation. Chưa có iterative search; `iterative_steps != 1` báo lỗi rõ.

## Files và trách nhiệm

| File | Thay đổi |
|---|---|
| `main.py` | Thêm `--build-plan-only`, giữ dry-run không computation |
| `configs/research_unified.yaml` | Calibration/evaluation kwargs local-data, metric AP50:95, deployment targets, recovery tắt mặc định |
| `contracts/evaluation.py` | `normalize_evaluation`, strict `normalize_yolo_evaluation`; giữ dict/scalar callback và alias `map` |
| `experiments/__init__.py`, `experiments/yolov5.py` | Callback module thật: evaluate wrapper, representative calibration provider, pruning-aware YOLO recovery |
| `contracts/deployment.py` | `DeploymentTargets`, `TargetCheckResult`, pure `TargetChecker` |
| `core/config.py` | Parse targets, calibration kwargs, allow_noop; reject unknown sections và unsupported iterations |
| `modules/calibration/context.py` | `CalibrationContext`: fixed batches, loss_fn, seed, optional sample_count/device |
| `modules/model/snapshot.py` | `ModelSnapshot`: deepcopy kiến trúc + weights, mỗi restore trả clone độc lập |
| `modules/analysis/sensitivity.py` | Dùng typed target order giống pruner; baseline restore và calibration payload; reject unsupported policy |
| `core/engine.py` | Mode/module compatibility, `build_plan`, apply supplied validated plan, ownership checks |
| `contracts/pruner.py` | Layerwise capability, ratio/index validation, duplicate/empty plan checks |
| `contracts/criterion.py`, `plugins/criteria/{bn_criteria,taylor_criteria,snip_criteria,grasp_criteria,synflow_criteria,lamp_criteria,l0_gate_criteria}.py` | Capability metadata và lựa chọn Conv-BN wrapper đúng criterion |
| `plugins/pruners/{structured,unstructured,depth}.py` | Tôn trọng hoặc reject layerwise policy; action validation; structured sequence validation/remapping |
| `plugins/adapters/yolov5.py` | Protected head theo ancestor Detect/Segment, không loại nhầm toàn bộ `m.0/m.1/m.2` bên trong C3 |
| `core/results.py` | `PlanBuildResult`, tách kết quả planning khỏi experiment |
| `pipelines/unified.py` | Calibration trước sensitivity, build-only stop, normalization, targets/checker, dry-run order mới |
| `tests/unit/test_orchestration_contracts.py` | Regression tests cho các contract mới và tiny topology |
| `tests/unit/test_flow_audit.py`, `tests/unit/test_unified_pipeline.py` | Cập nhật assertions cho flow mới/metric alias, giữ test cũ |

`test.py` và `tests/integration/test_yolo_evaluation.py` từ task evaluator trước được giữ nguyên trong task này.

## Calibration → sensitivity → policy → plan

`pruning.calibration_callback` được load bằng module:function và invoke đúng một lần tại stage `calibration`, với `calibration_kwargs`. Callback phải trả `CalibrationContext`; invalid schema báo lỗi. Runners dùng batches/loss/seed từ context, không gọi adapter noise provider khi callback đã cung cấp dữ liệu.

Detached calibration result được đóng gói thành `gradient_calibration`, `higher_order_calibration` hoặc `synflow_calibration`. Cùng result object keyed theo tên target được dùng trong các probe độc lập và final build_plan. Không dùng gradient còn sót trong model sau training. SynFlow tiếp tục dùng input toàn 1; cấu hình batch/loss callback cho criterion không sử dụng nó bị reject, không âm thầm bỏ qua.

Nếu không cấu hình callback, integrated adapter calibration được giữ để tương thích API cũ. YOLO integrated default là synthetic/no-label; config nghiên cứu mẫu hiện dùng provider local-data explicit. Callback YOLO yêu cầu model.hyp phù hợp để tạo ComputeLoss; checkpoint thiếu sẽ báo lỗi rõ.

Sensitivity không mutate model cuối. Nó restore deep snapshot cho từng probe; temporary structural mutation chỉ xảy ra trên clone. Analyzer enumerate typed targets giống mechanism. Nếu mọi probe invalid, unified fail trước build plan. Policy all-zero trong khi amount>0 dẫn tới empty-plan error, trừ `pruning.allow_noop: true`.

## Compatibility / no silent fallback

- SNIP, GraSP, SynFlow, LAMP: unstructured weight scoring; structured/depth bị reject.
- Taylor: structured output-channel scoring; unstructured/depth bị reject.
- BN scale, BN×L1 và hard-concrete: structured Conv-BN path; Linear không có BN ownership bị reject.
- Global unstructured selection (config flag hoặc intrinsic global criterion) không hỗ trợ layerwise policy: fail fast. Không bỏ layer ratios để lấy global amount.
- Depth không hỗ trợ layerwise policy; explicit `block_names`/numeric amount vẫn là đường riêng.
- L0 forced indices không được override sensitivity/layerwise policy.
- L1/L2/Magnitude/Random nhận module Conv/Linear thực; chỉ criterion cần BN mới nhận wrapper.
- Negative/out-of-range ratios, invalid/duplicate layer indices, unsupported action, unknown/protected forced targets và empty executable plan đều bị reject.
- Plugin mới có thể khai báo supported_pruning_modes/supported_module_types; custom criterion cũ dùng convention base để giữ extensibility.

## Structured safety và semantics của indices

Plan indices thuộc hệ tọa độ channel của model tại thời điểm build. Trước mutation, validate từng group và chạy cả chuỗi trên deep-cloned adapter/model/plan. Sau mỗi action, ánh xạ các channel còn sống được cập nhật cho mọi Conv/Linear/BN output bị ảnh hưởng bởi dependency group. Action kế tiếp chuyển original indices sang current positions; channel đã bị xóa bởi action trước không bị xóa lần hai.

Graph được rebuild trước mỗi action. Protected output bị reject kể cả khi nó bị ảnh hưởng gián tiếp qua residual dependency; prune input của protected consumer vẫn hợp lệ. Model shapes khác thời điểm validation làm apply fail, yêu cầu rebuild plan. Model ownership cũng được kiểm tra.

Tiny topology tests bao phủ:

- Conv→BN cùng đổi width.
- Residual Add giữ hai nhánh cùng shape.
- Nhánh split→concat→consumer bảo toàn đúng offsets/weight slices.
- Detect-like output giữ nguyên channel count; dependency gián tiếp vào output được chặn.
- Hai action `[1,3]` rồi `[4,6]` trên hai root cùng residual group giữ đúng original channels `[0,2,5,7]`.
- Rebuild graph trước hai mutations; stale plan sau model mutation bị reject.
- Conv-BN wrapper chấm L1 được; BN criterion trên Linear bị reject.

Đây là chứng minh các pattern tối thiểu, không chứng nhận mọi custom op/topology YOLO. Shadow validation dùng thêm bộ nhớ do deepcopy; không thực hiện optimization lớn trong task này.

## Deployment targets

Các constraint optional: parameter_reduction, flops_reduction, max_map50_drop, max_map50_95_drop, max_latency_ms, max_memory_mb. mAP drop là chênh lệch tuyệt đối trên thang 0–1. Reduction được tính từ complexity baseline/current; khác với tỷ lệ mask hoặc channel selection.

Checker thuần logic trả reached, violations, unavailable. Metric bắt buộc bị thiếu không thể pass. Peak memory phải được đo bằng `memory_mb`, không dùng model_size_mb để thay thế. Latency dùng `total_ms` từ callback hoặc benchmark result nếu được bật. Unified lưu `target_check.json`; target không đạt/unavailable chặn export/checkpoint finalization và báo lỗi. Không tự điều chỉnh sparsity.

## Commands và boundary

Static audit, đã chạy:

```bash
conda run -n env_cv python main.py --config configs/research_unified.yaml --dry-run
```

Planning command cho lần sau (không chạy với model thật trong task này):

```bash
conda run -n env_cv python main.py --config configs/research_unified.yaml --build-plan-only
```

`--dry-run` không load model/data hoặc import callback user. Callback resolution được test riêng qua pipeline constructor. `--build-plan-only` có load model, evaluation/calibration và sensitivity trên clone, validate plan rồi dừng trước final apply/recovery/benchmark/export. Không được hiểu build-plan-only là zero-compute. Nó cũng không cho chạy pre-pruning regularization training; phải cung cấp prepared checkpoint.

## Remaining blockers trước experiment thật

1. `experiments.yolov5:recover` là pruning-aware callback đã có unit/tiny validation. Detector experiment vẫn cần local train split, `ComputeLoss` tương thích checkpoint và protocol đánh giá rõ ràng.
2. Kiểm tra local checkpoint có loss hyperparameters và dữ liệu local/subset phù hợp cho calibration/evaluation; task này không load detector thật để chứng nhận.
3. Nếu experiment cần export/reload, cần kiểm chứng round-trip checkpoint qua adapter (đặc biệt EMA khi recovery trả model instance mới), thay vì chỉ state-dict vào clone. Không chạy export trong task này.
4. Constraint latency/memory yêu cầu measurement thực phù hợp; khi chưa có checker trả unavailable, không coi target đạt.

Không còn silent bypass sensitivity cho các mechanism chưa hỗ trợ: những configuration đó được reject trước experiment. Iterative search không nằm trong scope và không được giả lập bằng iterative_steps.

## Verification

Lần kiểm tra implementation trước cập nhật documentation ghi nhận suite pass; không chạy lại suite trong task docs này. Bao gồm test CLI dispatch build-only, calibration callback/config/order/context identity, nonuniform ratios `.25/.5`, baseline/probe isolation, model weights/shapes không đổi sau plan-only, empty/all-invalid rejection, normalization, target gate, dependency/protection/remapping. Chỉ dùng tiny models, mock callbacks và synthetic evaluator fixtures; không có metric experiment mới.
