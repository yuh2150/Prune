# Audit orchestration pruning detector — 2026-09-30

> Đây là audit lịch sử trước khi sửa contracts. Các mô tả “hiện tại” và test results bên dưới thuộc thời điểm audit; xem [tài liệu hiện hành](../pruning.md). Xem [implementation sau audit](pruning-contracts.md) để biết flow và giới hạn hiện tại.

Phạm vi: static inspection, dry-run, unit/contract test và mock/tiny model offline. Không chạy pruning experiment, sensitivity sweep COCO, fine-tune detector, evaluation dataset thật, hardware benchmark hay export model thật trong task này. Không có metric experiment mới.

**Kết luận:** sensitivity là stage thật trong unified pipeline và có đường truyền sang policy. Tuy nhiên flow hiện tại **chưa sẵn sàng chạy experiment end-to-end đáng tin cậy**, đặc biệt với config Taylor mẫu. Dry-run mới chỉ mô tả flow thực tế và gap; không chứng nhận readiness.

## 1. Flow thực tế

Entrypoint: `main.main` → `run_pruning_pipeline` → `run_unified_pruning_pipeline` → `UnifiedPruningPipeline.__init__` → `run`.

```text
validate config + import evaluation/recovery callbacks
→ seed + tạo run directory/config.resolved.json
→ ModelLoader → adapter.load_model → chọn pruner/criterion/granularity
→ baseline complexity (params/flops flags)
→ baseline evaluation callback (nếu evaluation hoặc sensitivity bật)
→ pre-pruning regularization callback (tùy chọn)
→ sensitivity probes + selector + sensitivity.json/selection.json
→ structured dependency preflight
→ criterion-specific calibration
→ requested_pruning_plan
→ engine.execute: create_plan → validate_plan → apply_plan
→ forward/parameter/checkpoint-clone checks
→ recovery callback (tùy chọn)
→ final evaluation callback
→ final complexity
→ PyTorch dummy-input latency benchmark (tùy chọn)
→ ONNX export (tùy chọn)
→ checkpoint save
→ result.json
```

Sai thứ tự quan trọng: sensitivity hiện đứng **trước** dependency preflight và calibration. Probe vẫn tự xây graph thông qua structured pruner, nhưng không nhận calibration result. Chưa có baseline freeze/snapshot, deployment target, target check, policy-adjustment loop. Latency chỉ đo model cuối và đo trước ONNX export; chưa benchmark deployment runtime.

## 2. Bảng stage

“Test” dưới đây chỉ là unit/tiny/mock hoặc contract, không phải xác nhận detector thật. File viết theo đường dẫn tương đối repository.

| Stage | Existing | Connected | Tested | Gap / implementation |
|---|---|---|---|---|
| Environment / research setup | Một phần | Có seed/config/artifacts | Unit config/artifacts | `core/config.py`, `core/experiment.py`; chưa manifest Python đầy đủ/version/dataset manifest |
| Model preparation | Có | Có | Mock load + adapter tests | `modules/model/loader.py:ModelLoader.load`, `plugins/adapters/yolov5.py:load_model`; loader có thể download, dry-run không gọi |
| Freeze baseline | Chưa | Chưa | Không | Không clone/snapshot baseline trong unified; checkpoint object được giữ cùng model đang mutate |
| Baseline evaluation | Có contract | Callback | Mock | `UnifiedPruningPipeline._evaluate`; YOLO evaluator tuple cần wrapper sang dict; callback mẫu thiếu |
| Deployment targets | Chưa | Chưa | Không | Không có config/class target params/FLOPs/latency/memory/accuracy tổng thể |
| Strategy selection | Có | Registry → engine | Unit/contract | `core/registry.py`, `core/engine.py`; chưa kiểm tra mọi tổ hợp criterion/strategy |
| Criterion selection | Có | Registry → pruner | Unit criteria | Các giới hạn trong bảng criterion bên dưới |
| Dependency graph | Có | Preflight + validate/apply | Conv chain, Linear residual, tiny blocks | `PruningEngine.build_dependency_graph`, `StructuredPruner`; chưa đảm bảo full YOLO topology/protected edges |
| Calibration | Có runners | Đến final prune; chưa đến sensitivity | Unit gradient/HVP/SynFlow | `modules/calibration/`; callback config bị bỏ qua, YOLO default dùng noise |
| Sensitivity analysis | Có | Unified + standalone | Unit + mock isolation mới | `modules/analysis/sensitivity.py:SensitivityAnalyzer.analyze`; Conv-only enumeration, không context/calibration |
| Sensitivity → policy | Có | Có điều kiện | Mock nonuniform ratios mới + selector tests | Global unstructured và L0 forced indices có thể override kết quả |
| Final pruning plan | Có | create → validate → apply | Unit plan | `contracts/targets.py:PruningPlan/PruningGroup`; không target budget/exclusion policy đầy đủ |
| Apply pruning | Có | Engine gọi | Tiny unit tests | Không thực thi trên detector thật trong audit |
| Recovery | Có callback contract | Sau prune | Mock callback | Không optimizer/scheduler/dataloader implementation tích hợp trong unified; sample callback thiếu |
| Evaluation | Có | Final callback | Mock + synthetic evaluator suite | Chuẩn hóa key và wrapper YOLO còn thiếu |
| Target check | Chưa | Chưa | Không | Không so sánh baseline/current với deployment targets |
| Save/load | Có từng phần | Save cuối flow | State-dict → clone unit | Chưa round-trip qua adapter và file exported architecture độc lập |
| Export | ONNX + checkpoint | Có | Static trong audit | `modules/export/exporter.py`; TorchScript/TensorRT không có unified stage |
| Benchmark | Có PyTorch latency utility | Final only | Static inspection; chưa test latency riêng | `modules/evaluation/benchmark.py`; unified không truyền postprocess/NMS, không baseline latency |

## 3. Baseline và model preparation

`FrameworkConfig.model.name` chọn adapter qua `PluginRegistry`; `ModelLoader.load` gọi classmethod `adapter.load_model`. YOLO load checkpoint chọn `ema` nếu có, nếu không chọn `model`. RT-DETR dùng adapter riêng. Registry có YOLOv7 alias chung adapter YOLOv5, không phải implementation YOLOv7 độc lập; không có ResNet adapter dù CLI description cũ có nhắc.

`run()` dùng cùng instance cho baseline, regularization, pruning, recovery. Không snapshot kiến trúc/weights trước pruning, không khóa callback khỏi mutation, không lưu checkpoint baseline mới. “Freeze baseline” ở đây cần nghĩa snapshot bất biến để so sánh/khôi phục; không nên đơn thuần tắt requires_grad vì calibration/tracing cần gradient.

`_evaluate` nhận float, dict các giá trị số, hoặc `EvaluationResult`. Baseline/final metrics là `Dict[str,float]` trong `ExperimentResult` và `result.json`; baseline complexity được lưu riêng bằng `ComplexityResult`. Không có class `BaselineMetrics` thống nhất nhưng có representation tương đương dạng dict + dataclass cost.

- Precision, recall, AP50, AP50:95 có thể đi qua callback dict. Framework không bắt buộc đủ keys.
- `test.test` hiện trả `(results, maps, times)`; không thể dùng thẳng làm callback unified. Cần wrapper explicit chuyển `results[0:4]` thành `precision`, `recall`, `map50`, `map` (ở đây `map` = AP50:95).
- Params, trainable params, FLOPs/MACs nếu THOP hỗ trợ, sparsity, model size có trong `measure_complexity`. FLOPs không đo được trả `None`.
- `model_size_mb` là bytes parameters + buffers, không phải peak RAM/VRAM hay kích thước checkpoint thực tế.
- Không đo baseline latency dù bật benchmark; final latency nằm riêng trong `PruningResult.benchmark`.
- Complexity flags `params`/`flops` vẫn có hiệu lực dù `benchmark.enabled=false`.
- THOP/tracing và callback chưa có isolation tổng thể cho mode/buffers/RNG; cần kiểm soát trước khi coi baseline bất biến.

## 4. Deployment target và iteration

`pruning.amount` / alias `target_ratio` là tỷ lệ yêu cầu trên channel/weight/block, không phải tỷ lệ giảm tổng parameters. `SensitivitySelector.target_sparsity` thực chất là **cap theo từng layer**. `sensitivity.max_allowed_relative_drop` chỉ ràng buộc một probe, không bảo đảm accuracy sau ghép tất cả layer và recovery.

Chưa có deployment targets cho parameter reduction, FLOPs reduction, latency, peak memory, accuracy constraint tổng thể. Chưa có `TargetChecker`. Các section top-level không được `FrameworkConfig.from_dict` sử dụng (ví dụ `targets`, `policy`, `data`, `baseline`, `calibration`) hiện có thể bị bỏ qua im lặng. Không thêm các section đó vào YAML rồi cho rằng đã có hiệu lực.

`iterative_steps` được validate/forward nhưng không pruner nào dùng để lặp. Flow thực tế là one-shot; nhiều group mutation liên tiếp không tương đương iterative target-driven pruning. Đề xuất sau audit: `DeploymentTargets`, `TargetCheckResult` với trạng thái passed/failed/unavailable, normalize metrics/cost, và iteration controller có stop budget. Chưa implement trong task này.

## 5. Strategy / criterion

Registry tách `pruner`, `criterion`, `granularity`, `adapter`; criterion chấm importance, pruner lập selection/plan và mutation. Không coi SNIP là structured hay L1 là channel pruning.

Strategy có: unstructured element-wise masking; structured Conv output channel/Linear output feature; filter dùng cùng structured mechanism với cách làm tròn khác; depth/structural_block do adapter khai báo; `nm` và `block_sparse` là constrained mask mechanisms. `attention_head` là adapter-declared masked-head mutation, không physical compression.

| Criterion | Registry keys | Conv2d | Linear | Needs data | Needs gradient | Structured usable trong implementation hiện tại |
|---|---|---|---|---|---|---|
| Magnitude | `magnitude` | Có | Có | Không | Không | Có output score; chú ý wrapper YOLO bên dưới |
| L1 | `l1`, `l1_norm` | Có | Có | Không | Không | Có; wrapper YOLO là gap |
| L2 | `l2`, `l2_norm` | Có | Có | Không | Không | Có; wrapper YOLO là gap |
| Taylor | `taylor`, `taylor_first_order` | Có | Hàm score có | Task batches | Có bậc 1 | Pipeline khai báo target chỉ Conv output; chưa expose Linear Taylor |
| SNIP | `snip` | Có | Có | Task batches | Có bậc 1 | Chỉ weight scores; chưa có aggregate structured |
| GraSP | `grasp` | Có | Có | Task batches | Có HVP/bậc 2 | Chỉ unstructured global |
| SynFlow | `synflow` | Có | Có | Không dataset, input toàn 1 | Có backward | Chỉ unstructured global |
| LAMP | `lamp` | Có | Có | Không | Không | Element-wise, chưa aggregate channel |
| BN scale | `bn_scale`, `bn_gamma` | Qua BN/wrapper | Không | Không khi score | Không khi score | Có cho Conv-BN; học gamma là training riêng |
| BN × L1 | `bn_l1_combined` | Qua Conv-BN wrapper | Không | Không khi score | Không khi score | Có cho wrapper |
| Random | `random` | Có | Có | Không | Không | Có; cần seed ổn định |
| Hard-concrete gate | `l0_gate`, `hard_concrete` | Qua BN có gate | Không | Training để học gate | Training có gradient; score không | Có, forced indices từ controller |
| Movement | Chưa đăng ký | — | — | — | — | Chưa có |

Nguồn: `plugins/criteria/*.py`, `StructuredPruner._structural_targets/_scores`, `UnstructuredPruner._weight_scores/_uses_global_selection`.

Gap cụ thể: YOLO `get_importance_module` trả parent Conv-BN wrapper cho mọi criterion không `requires_gradients`. L1/L2/Magnitude/Random chỉ nhận `nn.Conv2d` hoặc `nn.Linear`, không nhận wrapper `models.common.Conv`. Vì vậy structured L1 trên YOLO wrapper có thể lỗi dù test Conv thuần pass. BN criteria thì cần wrapper này. Cần truyền module theo capability criterion, không áp wrapper vô điều kiện.

SNIP/GraSP/SynFlow khai báo target weight; khi ghép structured, giao với structural target rỗng có thể tạo **empty plan** thay vì báo tổ hợp unsupported. Taylor unstructured trả channel score, không khớp tensor weight shape. Cần compatibility validation, không thêm thuật toán mới để che gap.

## 6. Dependency safety

Có **explicit graph** dùng `torch_pruning.DependencyGraph`, cộng adapter rules chọn root. `STRUCTURAL_OPERATIONS` ánh xạ Conv output và Linear output sang mutation functions.

```text
criterion scores → indices → PruningGroup
→ trace dependency graph → get_pruning_group → check_pruning_group
→ lưu dependencies (consumer/operation/indices)
→ rebuild graph trước mỗi action → check lại → group.prune()
```

Conv→BN, Conv→Conv, Add/residual và Concat dựa trên graph của forward thực tế; không phải manual propagation trong framework. Depth C3/CSP dùng đường khác: adapter khai báo internal Bottleneck, validate trên deepcopy, xóa block và repair links.

**Chưa đủ bằng chứng tuyên bố “an toàn đầy đủ cho YOLOv5”:**

- Root protection dùng substring `detect`, `segment`, `anchor`, `m.0`, `m.1`, `m.2`. Các pattern `m.*` cũng loại nhầm nhiều Bottleneck bên trong C3. Không có kiểm tra explicit mọi edge dependency tới protected head output.
- Graph được rebuild sau mutation nhưng channel indices của các action vẫn được chốt trên model trước mutation. Các root cùng dependency group/residual có thể overlap; chưa có canonical group dedup/remapping cho action sau.
- Không có transactional rollback nếu lỗi ở action sau khi action trước đã mutate.
- Preflight engine không tự đặt eval hoặc snapshot buffers; graph trace có thể ảnh hưởng state khi model đang train.
- Post-check yêu cầu forward + parameter count, không đối chiếu output schema baseline; checkpoint-clone failure chỉ warning trong engine, không luôn chặn unified export.
- Test có Conv chain, Linear residual/reshape, tiny C3 removal; không chứng minh đầy đủ channel pruning trên C3/CSP→Concat→Detect detector thật.

Cần tiny graph fixtures bao phủ các kết nối trên và nhóm overlap/protected output trước experiment; không cần chạy detector thật để bổ sung các contract này.

## 7. Calibration

Các runner: `GradientCalibrationRunner` cho Taylor/SNIP, `HigherOrderCalibrationRunner` cho GraSP, `SynFlowCalibrationRunner` cho SynFlow. Kết quả chứa tensor detached, keyed theo target name; pruner lấy `context_for(target)` rồi `criterion.score(...)`. Không có `criterion.prepare()` riêng nhưng runner + context là abstraction tương đương.

Config có số batch, batch size, seed, accumulate và device model. Không có dataloader/split/subset/sample-count abstraction riêng trong config; adapter YOLO mặc định sinh `torch.randn(B,3,640,640)` + empty labels. Đây là calibration tổng hợp, không phải dữ liệu representative. `model.input_shape` chưa được YOLO dummy/calibration dùng, vẫn hard-code 640.

Runners khôi phục grad, requires_grad, module modes, buffers và Python/Torch RNG; SynFlow còn snapshot weights và Detect grids qua adapter. Đây là isolation của runner, không chứng minh replay dataset/evaluation subset cố định.

`pruning.calibration_callback` có field và xuất hiện trong config mẫu nhưng **không được run() resolve/invoke**. Calibration result chỉ truyền vào final `engine.execute`, không vào sensitivity. Việc chỉ đổi thứ tự hai stage chưa đủ; cần truyền context theo name cho từng clone và đảm bảo mapping target ổn định.

## 8. Sensitivity chi tiết

`UnifiedPruningPipeline.run` gọi analyzer khi `cfg.sensitivity.enabled`. Standalone `run_sensitivity_pipeline` cũng có, nhưng không phải đường duy nhất.

Analyzer:

1. Liệt kê `adapter.get_pruneable_modules()` (YOLO/RT-DETR hiện là Conv roots).
2. Gọi evaluator lấy baseline một lần nữa, dùng stage `sensitivity` trong unified; không reuse metric baseline stage trước đó.
3. Với mỗi `(index, rate)`, `copy.deepcopy(model)` và gọi engine với **chỉ** `{"pruning_params": [(index, rate)]}`, `verify_forward=False`.
4. Evaluate clone; tính absolute drop và relative drop; ghi `SensitivityPoint`.
5. Probe tiếp theo luôn clone từ model gốc, không từ probe trước. Mock test mới xác minh điều này.
6. Lỗi bị catch thành invalid point, error_message, infinite drop; không fail toàn run khi tất cả probe lỗi.

Các giới hạn:

- Không truyền calibration, min_channels từ config, global selection flags hay criterion preparation vào probe. Probe constraints có thể khác final plan.
- Structured Conv-only toy path có ý nghĩa; enumeration Conv không đồng bộ với target list gồm Linear; depth nhận index Conv nhưng cần block names/numeric amount. Chưa sensitivity theo canonical dependency group.
- SNIP/GraSP/SynFlow/LAMP có `global_selection=True`; unstructured probe có thể dùng global amount mặc định thay vì `(layer,rate)`, hoặc fail do thiếu calibration. Profile khi đó không phải layer sensitivity hợp lệ.
- Seed toàn run có, nhưng không reset RNG trước từng probe; random criterion/evaluator stochastic có thể dùng mẫu khác nhau. Cố định evaluation subset là trách nhiệm callback, chưa được enforce.
- Nếu regularization bật, sensitivity đo model sau regularization; baseline stage đầu lại là trước regularization. Hai reference không đồng nhất.
- Relative drop khi baseline score bằng 0 đang đặt 0, có thể bỏ qua degradation cho metric lower-is-better; greedy selector luôn tối đa score, không tôn trọng direction này.
- Chỉ có score/drop/relative_drop/valid/error; chưa params_removed/FLOPs_removed/latency từng probe.
- JSON đầy đủ được unified lưu thành `sensitivity.json`; `SensitivityResult.to_dict` là legacy score-only, bỏ invalid/error/metadata. `Infinity` ở lỗi là JSON extension của Python, không phải strict JSON interoperable.

## 9. Sensitivity → policy

Đường truyền tồn tại:

```text
SensitivityResult
→ LayerSelectorModule.select
→ SensitivitySelector.select_layers
→ SelectionResult (SelectedLayer: index/name/rate/expected_metric/drop)
→ list(selection) = [(index, rate), ...]
→ pruning_config["pruning_params"]
→ engine.execute → pruner.create_plan
```

Selector sensitivity chọn **rate lớn nhất đã probe**, không quá `pruning.amount`, với relative metric drop không quá threshold. Không rate nào đạt thì rate=0. Đây là policy theo layer, không phải solver tối ưu toàn model/parameter budget. Mock test mới tạo hai profile khác nhau, chứng minh ratios `.1` và `.3` đến final engine; dừng bằng sentinel **trước apply**.

Với structured, list ratios bỏ qua nhánh global top-k dù `global=true`: policy có ảnh hưởng. Với unstructured, `_uses_global_selection` xem flag hoặc criterion trước khi xử lý list; `_create_global_plan` dùng `amount`, bỏ layer ratios. Với L0 regularization, `forced_channel_indices` có ưu tiên cao hơn list sensitivity. Với depth, config `amount` có ưu tiên hơn list, không thực hiện policy Conv.

`PruningPlan` explicit có groups, primary targets, operation, indices, dependencies, validated/error và metadata. `requested_pruning_plan` chỉ là intent config; plan thực tế được lưu sau engine execution. Criterion được lưu trong `PruningResult`/config (structured plan metadata không ghi đầy đủ criterion). Excluded layers nằm trong adapter rules; deployment targets chưa nằm trong representation nào. Chưa có unified API build-only trả validated plan rồi stop; dry-run mới không pretend tạo plan/channel mask.

## 10. Recovery, export và benchmark

Regularization hiện là **pre-pruning training callback** với `stage="regularization"`, truyền adapter/controller. Sau prune, nếu recovery.enabled, cùng callback được gọi `stage="recovery"` nhưng không truyền regularizer ở lần này. Không được suy luận BN L1/L0 tự động tiếp tục trong recovery. Optimizer/scheduler/training/validation do callback tự sở hữu; không có implementation `experiments.yolov5:recover` trong repo.

`ModelExporter.export_checkpoint` sửa `ckpt["model"]` hoặc lưu HF folder hoặc state_dict. State_dict cần kiến trúc matching để reload. Nếu recovery trả instance mới trong checkpoint có EMA cũ, exporter không đồng bộ/xóa EMA; adapter ưu tiên EMA khi load nên có nguy cơ load nhầm model. Validator chỉ reload state_dict vào deepcopy kiến trúc hiện có, không chứng minh checkpoint→adapter round-trip độc lập.

ONNX có exporter chung và script `export.py` riêng; unified không save→reload→export, mà export trực tiếp instance trước checkpoint save. Chưa TorchScript/TensorRT unified stage. `TracedModel` utility/artifact traced sẵn không đồng nghĩa có deployment pipeline.

Benchmark unified dùng dummy input model PyTorch, không postprocess callback, nên NMS không được đo ở đường này. Standalone benchmarking có tham số postprocess. Không target gate trước export, không benchmark ONNX/TensorRT hoặc target hardware session.

## 11. Config/entrypoint sau audit

Config `configs/research_unified.yaml` là template; evaluation/recovery module `experiments.yolov5` chưa tồn tại, và calibration callback field chưa wired. Không xóa placeholders để làm config có vẻ chạy được.

Command an toàn **đã chạy**:

```bash
conda run -n env_cv python main.py --config configs/research_unified.yaml --dry-run
```

Dry-run validate typed config + registry names, in configured run order và cảnh báo. Không construct `UnifiedPruningPipeline`, không import callbacks, không load model/dataset, không seed/mutate RNG, không tạo run artifacts, không apply/recover/evaluate/export/benchmark. Không kiểm tra được checkpoint/data/callback runtime validity hoặc dependency compatibility. Không phải stop-after-policy computation mode.

Entrypoint cho experiment **sau khi giải quyết blockers**, chưa chạy:

```bash
conda run -n env_cv python main.py --config configs/research_unified.yaml
```

API tương đương: `run_unified_pruning_pipeline(config, evaluator=..., recovery=...)`. Callback evaluator phải trả dict/EvaluationResult, không trả nguyên tuple của `test.test`. Không có command hiện tại nào bảo đảm full desired flow chỉ bằng việc bỏ `--dry-run`.

## 12. Gaps và thứ tự xử lý

### BLOCKER trước experiment tương ứng

1. **Callbacks + contract:** implement/resolve evaluation/recovery callbacks; bridge tuple YOLO → dict; chọn metric key `map` rõ nghĩa AP50:95. Config mẫu hiện fail ở constructor trước model load.
2. **Calibration → sensitivity:** đưa dependency preflight/calibration lên trước probes; dùng calibration result cho probe và final; wire real-data callback/provider, không mặc định noise cho nghiên cứu representative.
3. **Policy semantics:** thống nhất targets by name/type giữa sensitivity/pruner; không bỏ layer ratios trong global unstructured, L0, depth; reject các combination chưa hỗ trợ.
4. **Silent invalid/empty policy:** chặn run khi mọi probe lỗi hoặc unsupported criterion tạo empty plan, phân biệt với no-op hợp lệ do accuracy constraint.
5. **Structured YOLO:** criterion-aware Conv/BN wrapper; kiểm tra overlap groups, protected head outputs, schema và index validity sau từng mutation. Graph existence chưa đủ chứng minh safety.
6. **Deployment-target flow yêu cầu:** chưa targets/checker/adjustment/finalization gate. Đây là thiếu architecture, không sửa giả bằng pruning.amount. Không cản thử nghiệm one-shot có kiểm soát nhưng cản tuyên bố đạt đầy đủ flow mục tiêu.

### IMPORTANT

- Baseline snapshot + normalized metrics/cost; cùng reference trước/sau regularization và cùng evaluation subset.
- Deterministic probe inputs/RNG, min_channels và candidate mapping nhất quán.
- Plan-only boundary để inspect/save validated policy trước apply.
- Checkpoint/EMA và architecture-aware adapter round-trip; chặn export nếu integrity không đạt.
- `iterative_steps` hiện no-op; reject hoặc document đến khi có controller.
- Config unknown top-level sections cần fail rõ; input_shape cần được adapter tôn trọng.
- Thiếu per-probe cost và baseline latency để trade-off deployment; memory size không phải peak memory.
- Không dùng mock/Conv-only test pass để kết luận full YOLO topology safety.

### OPTIONAL

- Sensitivity heatmap/Pareto visualization và CSV export.
- Dataset/run manifest hash, hardware/runtime metadata để so sánh thí nghiệm.
- TorchScript/TensorRT support sau khi save/load và target gate ổn định.

## 13. Flow mong muốn sau khi sửa blockers (design, chưa implement)

```text
config + callback/capability validation
→ model load → immutable baseline snapshot → normalized baseline metrics/cost
→ deployment targets
→ strategy/criterion/typed candidate targets
→ dependency preflight + criterion-specific representative calibration
→ isolated sensitivity probes (same baseline/data/constraints)
→ sensitivity-aware policy by target name/group
→ validate plan + save inspectable policy   [planning boundary]
→ apply → architecture validation → recovery → final evaluation/cost
→ target check (passed / failed / unavailable)
→ bounded policy adjustment nếu cần (khởi đầu lại từ baseline hoặc rõ incremental semantics)
→ finalize → checkpoint save/reload qua adapter → export → deployment-runtime benchmark
```

Không implement target solver/iterative architecture trong audit. Thay đổi runtime hiện tại chỉ là thêm static dry-run, không đổi algorithms, calibration, sensitivity selection hay recovery semantics.

## 14. Xác minh

File mới `tests/unit/test_flow_audit.py` có 5 test:

- CLI dry-run không construct pipeline, load model, import callback, execute pipeline hoặc ghi artifact.
- Disabled stages/global policy warning được mô tả đúng.
- Mỗi sensitivity probe nhận deepcopy độc lập cùng baseline; mock mutation không rò sang probe kế tiếp/model gốc.
- Real selector nhận profile mock và truyền nonuniform ratios đến final engine; sentinel chặn trước apply/export.
- Unified evaluator contract từ chối tuple, xác nhận cần wrapper explicit.

Các test này đều offline, nhanh; không chạy detector thật. Suite evaluator từ task trước chỉ dùng synthetic fixture khi chạy lại.

Kết quả: 5/5 test audit mới pass; toàn bộ **91/91 test pass** (13,700 giây). `git diff --check` pass. Không có test bị bỏ qua để làm suite xanh.

Lệnh đã chạy:

```bash
MPLCONFIGDIR=/tmp/prune-mpl /home/huy/miniconda3/envs/env_cv/bin/python -m unittest tests.unit.test_flow_audit
MPLCONFIGDIR=/tmp/prune-mpl /home/huy/miniconda3/envs/env_cv/bin/python -m unittest discover -s tests
MPLCONFIGDIR=/tmp/prune-mpl /home/huy/miniconda3/envs/env_cv/bin/python main.py --config configs/research_unified.yaml --dry-run
```

Artifact phát sinh từ unit test được chuyển sang `/tmp/prune-flow-test-artifacts`; không phải kết quả experiment. Các thay đổi evaluator (`test.py`, `tests/integration/test_yolo_evaluation.py`) thuộc task trước được giữ nguyên.

Thay đổi task audit: `main.py` thêm `--dry-run`; `UnifiedPruningPipeline.describe_flow` mô tả control flow hiện có; test audit mới; báo cáo này. Không thay `run()` hoặc pruning algorithms. Vì vậy những blockers nêu trên vẫn tồn tại và được báo rõ, không tuyên bố đã nối hoàn chỉnh toàn bộ desired flow.
