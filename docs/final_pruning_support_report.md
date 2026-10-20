# Báo cáo hỗ trợ 14 nhóm kỹ thuật pruning (20 recipe)

**Ngày:** 08/10/2026  
**Phạm vi:** framework pruning hiện tại, không dùng các benchmark legacy làm bằng chứng cho pipeline mới.

## 1. Kết luận tóm tắt

Framework đã có implementation cho **14 nhóm kỹ thuật pruning** qua **20 recipe canonical** (L1 và L0 tách riêng) và một pipeline thống nhất từ model/dataset đến artifact đã prune:

```text
model + dataset
→ scoring hoặc sensitivity theo semantics của kỹ thuật
→ chọn pruning policy
→ validate pruning plan
→ apply pruning
→ optional fine-tune / recovery
→ evaluate
→ save checkpoint
→ reload checkpoint + validate inference/evaluate
→ optional ONNX export + ONNX Runtime validation
```

Điểm quan trọng là pipeline không còn ép mọi kỹ thuật đi qua cùng một layer-wise sensitivity analysis. Structured pruning dùng sensitivity/structural policy; unstructured dùng score trên weight; N:M và block sparse dùng policy bị ràng buộc bởi pattern.

## 2. Ma trận kỹ thuật

| Nhóm | Kỹ thuật | Granularity / cơ chế | Trạng thái hiện tại |
|---|---|---|---|
| Unstructured | Magnitude | Weight magnitude, global mask | Pipeline integrated + test |
| Unstructured | SNIP | Weight, first-order saliency | E2E validated trên LeNet local |
| Unstructured | GraSP | Weight, higher-order saliency | E2E validated trên LeNet local |
| Unstructured | SynFlow | Weight, data-free flow score | E2E validated trên LeNet local |
| Unstructured | LAMP | Weight, global layer-adaptive magnitude | E2E validated trên LeNet local |
| Structured | L1 regularization | Channel sparsity qua BN scale | Pipeline integrated + test |
| Structured | L0 / Hard-Concrete | Learnable channel gate | Pipeline integrated + test |
| Structured | Channel pruning | Physical Conv/Linear output dimension | E2E validated trên LeNet local |
| Structured | Filter pruning | Physical Conv output filter | E2E validated trên LeNet local |
| Structured | Layer pruning | Adapter-approved depth/layer removal | Pipeline integrated + test |
| Structured | Block pruning | Adapter-approved residual/C3 block removal | Pipeline integrated + test |
| Structured | Taylor pruning | Gradient/Taylor score, channel policy | E2E validated trên LeNet local |
| Structured | Attention-head pruning | Mask whole attention head | Pipeline integrated + test; chưa physical compression |
| Semi-structured | N:M sparsity | Pattern N:M theo group M | E2E validated trên LeNet local |
| Semi-structured | Block sparse | Rectangular weight-block mask | E2E validated trên LeNet local |

`E2E validated` ở đây nghĩa là config validate, load model/dataset, score/sensitivity theo semantics, tạo/apply plan, forward, recovery 1 epoch, evaluate, save/reload/evaluate checkpoint, ONNX export và ONNX Runtime validation đều đã chạy thành công. `Pipeline integrated + test` không được diễn giải là đã có benchmark detector production.

## 3. Bằng chứng full pipeline đã chạy

Hai mươi config canonical trong `configs/final/` đều khai báo flow hoàn chỉnh, bao phủ 14 nhóm kỹ thuật; L1 và L0 dùng recipe riêng vì cơ chế regularization và selection khác nhau. Phân bổ hiện tại gồm 5 recipe unstructured trên LeNet, 14 recipe trên ResNet-18 CIFAR-10 và 1 recipe attention-head trên YOLOv5 C3TR. Mọi recipe dùng ba mức sensitivity `5%`, `10%`, `20%`. SNIP, GraSP và SynFlow không chạy recovery; các recipe còn lại recovery `5` epoch:

```text
sensitivity analysis → selection → apply pruning → optional fine-tune/recovery
→ evaluation → benchmark → checkpoint/ONNX validation
```

Các kết quả benchmark lịch sử dưới đây đến từ chín lần chạy LeNet với recovery một epoch. Bộ recipe canonical hiện tại đã tắt recovery cho SNIP, GraSP, SynFlow; đồng thời chuyển Channel, Filter, Taylor, N:M và Block Sparse sang ResNet-18 CIFAR-10 với recovery năm epoch. Vì vậy cần chạy lại trước khi dùng bảng cũ để so sánh kết quả mới. Các recipe ResNet-18 CIFAR-10 và YOLOv5 C3TR chỉ được ghi nhận E2E sau khi chạy với checkpoint/dataset local tương ứng:

Manifest benchmark và link đến từng raw `result.json`: [`artifacts/final/benchmark_final.md`](../artifacts/final/benchmark_final.md).

| Kỹ thuật | Accuracy baseline → final | Parameters trước → sau | Checkpoint reload / ONNX Runtime |
|---|---:|---:|---|
| SNIP | 0.84 → 0.88 | 64,851 → 64,851 | Pass / Pass |
| GraSP | 0.84 → 0.88 | 64,851 → 64,851 | Pass / Pass |
| SynFlow | 0.84 → 0.88 | 64,851 → 64,851 | Pass / Pass |
| LAMP | 0.84 → 0.88 | 64,851 → 64,851 | Pass / Pass |
| Channel | 0.84 → 0.83 | 64,851 → 51,017 | Pass / Pass |
| Filter | 0.84 → 0.88 | 64,851 → 56,180 | Pass / Pass |
| Taylor | 0.84 → 0.83 | 64,851 → 52,577 | Pass / Pass |
| N:M 2:4 | 0.84 → 0.81 | 64,851 → 64,851 | Pass / Pass |
| Block sparse | 0.84 → 0.875 | 64,851 → 64,851 | Pass / Pass |

Unstructured và semi-structured chỉ tạo mask, nên tensor shape và parameter count không giảm; không được dùng số này để claim speedup dense-runtime. Channel/filter/Taylor thay đổi topology thật, nên có physical parameter reduction.

## 4. Rút ra từ 14 nhóm kỹ thuật

1. **Không có một policy chung đúng cho mọi technique.** Sensitivity theo layer phù hợp với channel/filter/Taylor; global saliency phù hợp SNIP/GraSP/SynFlow/LAMP; N:M/block sparse cần validate pattern trước khi apply.
2. **Structured pruning khó hơn nhưng deploy rõ ràng hơn.** Dependency graph phải sửa các consumer liên quan, nhưng đổi lại model nhỏ hơn thật và có tiềm năng giảm latency ngay với dense runtime.
3. **Sparsity không đồng nghĩa speedup.** Unstructured và semi-structured chỉ nhanh khi inference runtime/kernel/phần cứng có sparse-pattern support phù hợp.
4. **Calibration và recovery quyết định chất lượng.** GraSP đạt 0.84 → 0.88 với policy canonical và recovery một epoch trên LeNet. Con số này là accuracy của một model/split cụ thể, không phải bằng chứng về speedup.
5. **Artifact validation là một phần của pruning.** Model structurally-pruned không thể giả định cùng topology với model gốc; checkpoint reload, forward/evaluate lại và ONNX Runtime validation cần nằm trong flow chuẩn.

## 5. Kết luận theo ba nhóm chính

### Unstructured pruning

Phù hợp khi mục tiêu chính là sparsity và duy trì architecture gốc. Các kỹ thuật SNIP, GraSP, SynFlow, LAMP đều tạo pruned weight artifact được. Tuy nhiên, zero weight không tự tạo tốc độ nhanh hơn trên CPU/GPU dense thông thường; cần sparse kernel hoặc accelerator hỗ trợ pattern. Mỗi policy vẫn cần được đánh giá lại trên checkpoint, dataset và sparsity mục tiêu khi triển khai production.

### Structured pruning

Phù hợp hơn với mục tiêu model compression và deployment latency vì channel/filter/layer/block có thể làm kiến trúc nhỏ đi thực sự. Đổi lại cần dependency-aware pruning, forward validation và recovery cẩn thận hơn. Attention-head hiện mới là **masking head**, chưa shrink `embed_dim`, nên chưa được claim là physical compression.

### Semi-structured pruning

N:M và block sparse tạo sparsity có cấu trúc, là điểm cân bằng giữa flexibility của unstructured và khả năng tăng tốc phần cứng. Framework đã validate dimension/pattern rõ ràng, nhưng lợi ích runtime vẫn phụ thuộc kernel và hardware hỗ trợ đúng N:M/block layout.

## 6. Issue và giới hạn hiện tại

Code có đủ stage sensitivity và recovery. Tuy nhiên, không phải mọi detector/config đều có đủ checkpoint, dataset, GPU/RAM và thời gian để chạy full pipeline trong môi trường hiện tại.

Vì vậy ở một số detector experiment, sensitivity và retrain được **tắt rõ trong config** để chạy plan/prune smoke hoặc không được chạy E2E. Các run này không được đánh dấu là full-pipeline validated. Đây là resource limitation, không nên diễn giải là lỗi của kỹ thuật hoặc kết quả benchmark cuối cùng.

Các hạn chế cụ thể:

- Full detector E2E cho L1/L0, layer/block và attention head chưa được xác nhận trên checkpoint/dataset production.
- Attention-head physical removal/compression chưa có.
- GraSP mới có bằng chứng E2E trên LeNet; vẫn cần lặp lại theo checkpoint, dataset và sparsity mục tiêu của production.

## 7. Đề xuất cần duyệt trước khi thực hiện

### Phương án A — Google Colab Free

Mục tiêu: chạy một số E2E detector representative có GPU miễn phí, ưu tiên Taylor/channel, N:M và attention head nếu có checkpoint tương thích.

- Upload/mount checkpoint và dataset subset đã chuẩn bị; không tự tải data không rõ license.
- Dùng split nhỏ, seed cố định, sensitivity rate giới hạn và recovery 1–5 epoch để proof-of-pipeline.
- Lưu `.pt`, ONNX, `result.json`, log môi trường và commit hash về Drive/repository artifact store.
- Rủi ro: quota/session timeout, GPU không ổn định, storage I/O chậm; kết quả chỉ là validation kỹ thuật, không thay thế benchmark production.

### Phương án B — máy Ubuntu (OSSA) của công ty

Mục tiêu: chạy experiment có thể tái lập và benchmark đáng tin cậy hơn.

- Cần duyệt quyền dùng GPU/CPU, disk quota, vị trí dataset/checkpoint và thời gian chạy.
- Dùng môi trường Conda/Docker được pin version; ghi GPU driver, CUDA, PyTorch, ONNX Runtime và commit hash.
- Chạy theo thứ tự: baseline → sensitivity → policy → prune → recovery → evaluate → export/reload → benchmark latency trên đúng máy đích.
- Ưu tiên full E2E detector cho L1/L0, layer/block, Taylor và attention head; sau đó so sánh latency giữa dense, sparse runtime và structured model.

## 8. Tiêu chí hoàn tất experiment production

Một technique chỉ nên được gọi là production-ready sau khi có checkpoint/dataset đích và đạt đầy đủ: baseline + post-prune + post-recovery metric, parameter/FLOPs/latency, checkpoint reload, ONNX Runtime inference, sparsity/physical reduction đúng semantics, và artifact/log có thể tái tạo.
