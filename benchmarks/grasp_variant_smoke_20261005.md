# GraSP variant smoke report — 2026-10-05

Plain `grasp` remains unchanged and is retained as the failed baseline. Both
variants use the same pretrained `lenet5_emnist_onnx`, EMNIST-47 split,
seed 42, no recovery, and all available training calibration batches (13).

| Variant | Requested global sparsity | Actual sparsity | conv1 | conv2 | conv3 | fc1 | Accuracy | Loss | Quality status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| grasp_layer_normalized | 1% | 0.934% | 0.00% | 0.08% | 1.02% | 1.12% | 0.600 | 1.608760 | SEVERE |
| grasp_layer_normalized | 3% | 2.805% | 0.00% | 2.33% | 3.04% | 3.00% | 0.160 | 6.956384 | FAILED |
| grasp_layer_normalized | 5% | 4.675% | 2.67% | 5.88% | 4.97% | 4.96% | 0.055 | 11.420682 | FAILED |
| grasp_layer_normalized | 10% | 9.349% | 22.67% | 15.75% | 9.71% | 9.80% | 0.020 | 16.845493 | FAILED |
| grasp_conv1_capped | 1% | 0.934% | 10.00% | 10.25% | 0.25% | 2.22% | 0.025 | 10.004558 | FAILED |
| grasp_conv1_capped | 3% | 2.805% | 10.00% | 19.42% | 1.56% | 5.82% | 0.020 | 14.361642 | FAILED |
| grasp_conv1_capped | 5% | 4.675% | 10.00% | 23.71% | 3.29% | 8.64% | 0.020 | 15.830271 | FAILED |
| grasp_conv1_capped | 10% | 9.349% | 10.00% | 32.54% | 8.13% | 13.53% | 0.010 | 20.848862 | FAILED |

For the normalized variant, the pruning plan records pre/post normalization
statistics in `metadata.score_statistics`. At 1%, raw mean-absolute scores
were `conv1=0.334993`, `conv2=0.037687`, `conv3=0.003609`, `fc1=0.011783`;
the normalized mean-absolute score is approximately 1.0 for every layer.

Conclusion: score-scale normalization reduces the first-layer concentration,
but neither variant is quality-preserving on this pretrained ONNX LeNet setup.
Do not extend either smoke variant to 30/50/70% without a further algorithmic
review or recovery protocol.
