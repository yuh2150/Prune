# Pruning and sensitivity concepts

Các khái niệm hiện hành được tập trung trong [Pruning Framework](../pruning.md).

- [Criterion vs strategy](../pruning.md#importance-criterion--pruning-strategy): Taylor/SNIP là importance criteria; channel/weight/depth là mechanisms.
- [Sensitivity](../pruning.md#sensitivity-analysis): calibration → independent probes → evaluation drops → selector → layer-wise policy.
- [Policy vs plan](../pruning.md#policy-and-plan): ratios khác executable indices/dependencies.
- [Compatibility](../pruning.md#compatibility-matrix): depth và global unstructured không nhận layer-wise sensitivity policy hiện tại.
- [Structured safety](../pruning.md#structured-pruning-and-dependency-safety): tiny topology validation, chưa full detector end-to-end.

Weight sparsity không đồng nghĩa giảm dense tensor shapes hoặc runtime. Giảm params/FLOPs không tự đảm bảo latency trên thiết bị đích.
