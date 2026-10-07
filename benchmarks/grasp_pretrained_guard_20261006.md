# GraSP LeNet5 pretrained: magnitude guard — 06/10/2026

Config mặc định `configs/lenet5_custom_47labels_grasp.yaml` nay dùng hybrid
`grasp_magnitude_guarded`. Signed HVP score của GraSP vẫn được tính như trước;
ranking chỉ xét nhóm 1.5× ngân sách weights nhỏ nhất theo magnitude, sau
chuẩn hóa mean_abs từng lớp. GraSP gốc và config reference vẫn dùng làm đối chứng.

| Prune eligible | Baseline accuracy | GraSP gốc | Guarded hybrid | Conv1 hybrid | Conv2 hybrid |
|---|---:|---:|---:|---:|---:|
| 1% | 84% | — | 84% | 0% | 0.38% |
| 5% | 84% | — | 84% | 0% | 2.42% |
| 10% | 84% | 5% | 82.5% | 0.67% | 5.08% |
| 30% | 84% | 4% | 83.5% | 2% | 14.21% |

Ở 30%, hybrid bỏ đúng 18,189/60,630 eligible weights: Conv1=3/150,
Conv2=341/2400, Conv3=17,312/48,000, fc1=533/10,080. Sparsity toàn model
là 28.0474%; dense parameter count giữ nguyên 64,851. fc2/bias không prune.
Magnitude control đạt 83% ở 10%, 83.5% ở 30%; hybrid chưa tốt hơn magnitude.

Tất cả chạy từ checkpoint ONNX gốc, cùng dataset root `datasets/`, split
seed 42, validation 200 ảnh, calibration 4×64 ảnh train, temperature 200,
microbatch 20, scope weights, không recovery/reinitialize. Bộ thử ban đầu
khảo sát pool multiplier 1, 1.25, 1.5, 2 trên cùng validation; số liệu này
không phải đánh giá trên holdout độc lập. Multiplier triển khai là 1.5,
không chọn mức có accuracy cao nhất ở 30% (1.25). Khi ratio ≥ 2/3, pool
phủ toàn bộ weights nên không còn guard về magnitude.

Nguồn đầy đủ: [comparison.md](../artifacts/grasp_guarded_comparison_20261006/comparison.md),
[comparison.json](../artifacts/grasp_guarded_comparison_20261006/comparison.json),
[protocol.json](../artifacts/grasp_guarded_comparison_20261006/protocol.json).
Run config mặc định: [result.json](../artifacts/lenet5_custom_47labels_grasp_magnitude_guarded_20261006T092309048612Z/result.json).
Checkpoint: `artifacts/lenet5_custom_47labels_grasp_magnitude_guarded.pt`.

Chạy lại trong môi trường `env_cv`:

```bash
python main.py --config configs/lenet5_custom_47labels_grasp.yaml
python -m tools.benchmark_grasp_guarded
```

Tests GraSP kiểm tra candidate exclusion, signed ranking trong pool, exact
budget/ties ở ratio 0/nhỏ/lớn, plan-only giữ nguyên state, persistent masks,
finite values và không vượt pool khi kết hợp cap. Kiểm tra mở rộng có một
test N:M không khớp implementation hiện có: test yêu cầu lỗi khi width
không chia hết, trong khi N:M bỏ qua layer đó. Không thay code N:M.

Lượt kiểm tra cuối: 46 tests liên quan pass, gồm 21 tests GraSP và LeNet
build/apply/forward cho hybrid. Reload checkpoint đã lưu vẫn accuracy
83.5%, loss 0.427038 và 18,189 mask entries bằng 0.
