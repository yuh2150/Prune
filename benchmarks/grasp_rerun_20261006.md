# GraSP LeNet5 chạy lại — 06/10/2026

Nguồn: [lenet5_custom_47labels_grasp_reference_20261006T040300687515Z](../artifacts/lenet5_custom_47labels_grasp_reference_20261006T040300687515Z/result.json/result.json).

- Dataset: datasets/Dataset_MNIST_47Labels_1000images, validation fraction 0.2, seed 42.
- Model: LeNet5_Numbers&Characters_FP32.onnx, CPU, input 1×1×32×32.
- GraSP unstructured, ratio 30% trên 60630 safe weight targets; bỏ 18189 weights theo score cao nhất.
- Calibration: HVP hai lượt, weight scope, 4 batches chia thành 16 microbatches, temperature 200.
- Không reinitialize classifier; không fine-tuning.

| Chỉ số | Baseline | Sau pruning |
| --- | ---: | ---: |
| Accuracy | 84% | 4% |
| Loss | 0.409764 | 254.853731 |
| Dense parameters | 64851 | 64851 |
| Sparsity toàn model | 0% | 28.0474% |

Latency sau pruning: 0.662 ms/img; FPS lưu sẵn 1511.42. Không có baseline latency tương ứng; không kết luận speedup. FLOPs tắt do vấn đề profiler đã ghi nhận.

Run cũ ratio 30%, temperature 1, implementation chọn chiều score sai báo accuracy 1%; run mới là 4%. Có nhiều thay đổi calibration nên không quy chênh lệch 3 điểm % riêng cho một thay đổi. Run mới vẫn mất 80 điểm % so với baseline; chưa giữ được chất lượng pretrained. Forward verification pass và weights finite không chứng minh accuracy đạt yêu cầu. Target check không cấu hình giới hạn accuracy nên reached=true không phải đạt chất lượng.

Checkpoint: artifacts/lenet5_custom_47labels_grasp_reference.pt. Kết quả benchmark cũ được giữ nguyên.

## Giảm ratio xuống 10%

Nguồn: [lenet5_custom_47labels_grasp_reference_10_20261006T040553814313Z](../artifacts/lenet5_custom_47labels_grasp_reference_10_20261006T040553814313Z/result.json). Cùng checkpoint, dataset, seed, temperature 200 và không fine-tuning. Accuracy baseline 84.00% → sau prune 6.00%; loss 162.556510; sparsity toàn model 9.3491%. Checkpoint riêng: artifacts/lenet5_custom_47labels_grasp_reference_10.pt.
