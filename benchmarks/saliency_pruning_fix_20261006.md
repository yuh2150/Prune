# Sửa GraSP và bổ sung iterative SynFlow — 06/10/2026

GraSP trước đây dùng score `-W * (H @ gradient)` rồi bỏ score thấp nhất.
Implementation của tác giả dùng top-k score cao nhất để chọn phần bỏ đi:
[GraSP.py, đoạn topk và keep_masks](https://github.com/alecwangcq/GraSP/blob/master/pruner/GraSP.py#L129-L142).
Framework nay giữ nguyên score có dấu và đảo chiều ranking riêng cho GraSP,
kể cả hai biến thể layer-normalized và conv1-capped; SNIP/SynFlow vẫn bỏ score thấp nhất.
Plan lưu `score_order` để truy vết.

Các benchmark GraSP accuracy 1% đã lưu là kết quả **trước khi sửa chiều ranking**.
Không ghi đè chúng hoặc coi chúng là bằng chứng về chất lượng implementation đã sửa.
Kết quả accuracy mới cần chạy lại trên cùng validation split.

SNIP hiện có `abs(W * mean(gradient))`, tương đương thứ tự ranking của
`abs(W * sum(gradient))` trong mã tham khảo khi chỉ khác một hệ số dương chung.
Đã sửa calibration đọc gradient trên original parameter khi module có persistent mask.
Framework tiếp tục giữ chính sách safe weight targets; không tự thêm bias hay classifier head
vào phạm vi pruning như đoạn tham khảo.

SynFlow hỗ trợ `iterative_steps > 1` cho unstructured pruning, tính lại score trên
network đã zero tạm qua từng lượt. Schedule cumulative là
`1 - (1 - amount) ** (step / steps)`, với final budget chính xác sau rounding.
Các mask đã bỏ không hồi sinh; quá trình build plan phục hồi original weights trong
`finally`, kể cả calibration lỗi. Mặc định một lượt được giữ để tương thích cấu hình cũ.
Config mới `configs/lenet5_custom_47labels_synflow_iterative.yaml` dùng 100 lượt.
Linearization gồm floating buffers; state/mode/gradient được khôi phục sau calibration.

Unit tests kiểm tra ranking GraSP có score phân biệt, calibration sau mask,
cardinality qua iterative SynFlow, rollback khi lỗi và tích hợp pipeline.
Dataset `Dataset_MNIST_47Labels_1000images` dùng cho artifacts cũ chưa có trong
workspace hiện tại, nên chưa có phép đo accuracy sau sửa trên cùng dữ liệu.

## Bổ sung từ GraSP reference

HVP nay tính hai lượt: cộng gradient detached ở lượt đầu, rồi cộng
`grad(dot(gradient_batch, gradient_accumulated))` ở lượt thứ hai. Mean thay
sum chỉ khác một hệ số dương chung với cùng số batch, không đổi ranking.
Pipeline GraSP dùng vector gồm Conv2d/Linear weights (kể cả protected head),
không gồm bias/BN như mã tham khảo. Runner generic vẫn hỗ trợ scope `all`.
RNG và buffer state được replay cho từng batch để hai lượt có cùng stochastic
objective; mỗi lượt chỉ giữ graph một batch.

`configs/lenet5_custom_47labels_grasp_reference.yaml` dùng logits temperature
200 và microbatch size 20. Chỉ calibration classification chia logits;
loss evaluation/recovery không đổi. Config GraSP cũ vẫn dùng temperature 1.
Không tự reinitialize classifier pretrained như đoạn reference: scoring với
classifier random là protocol khác, không đủ để khẳng định accuracy trên
checkpoint pretrained. Không clone/bỏ classifier head khỏi chính sách bảo vệ.
Normalization bằng một scalar dương chung không đổi thứ tự ranking nên không
bắt buộc để chọn exact budget. Khi tie, framework bỏ đúng số phần tử bằng
stable ranking thay vì threshold `<=` có thể giữ thêm weight. Amount 0 tạo
plan rỗng, không truy cập `threshold[-1]`.
