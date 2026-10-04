# Documentation audit

Phạm vi: đối chiếu Markdown hiện có với `main.py`, `test.py`, configs, framework contracts/pipeline/plugins và `experiments/yolov5.py`. Task chỉ sửa documentation; không chạy detector, sensitivity sweep, fine-tune, benchmark hoặc export. [Pruning Framework](pruning.md) là entrypoint hiện hành.

| Tài liệu | Kết quả audit trước chỉnh sửa | Xử lý |
|---|---|---|
| [README](../README.md) | Marketing/production claims, quickstart full run quá sớm | Viết ngắn, status levels, dry-run/build-only, link canonical |
| [Docs index](README.md) | Entry points trùng lặp, claims quá rộng | Index hiện hành và historical tách rõ |
| [Quickstart](getting-started/quickstart.md) | Environment/full-run examples chưa chứng nhận readiness | Chỉ inspection/planning, điều kiện compute/data |
| [Architecture overview](architecture/overview.md) | Taylor bị gọi strategy, flow/API cũ | Trỏ source và canonical flow |
| [Unified pipeline](architecture/unified-pipeline.md) | Calibration/order/callback hướng dẫn cũ | Calibration trước sensitivity, runner backward, target boundary |
| [API](api/framework-api.md) | Thiếu plan/build-only contracts | API index theo source, registry/decorator đúng |
| [Config reference](configuration/reference.md) | Calibration/recovery fields thiếu hoặc cũ | Sections/aliases thật, không invent data/policy section |
| [Concepts](concepts/pruning-and-sensitivity.md) | Nhầm criterion/strategy, sparsity/speedup | Tách khái niệm, bỏ guaranteed speedup |
| [Extension notes](development/extending-framework.md) | Example/signature cũ, fixed metric minh họa dễ hiểu nhầm | Contract-first notes, bỏ fake metric và recipe tải model |
| [Flow audit](architecture/pruning-flow-audit.md) | Hữu ích, nhưng findings thuộc trước fixes | Giữ historical engineering audit, thêm link hiện hành |
| [Contracts](architecture/pruning-contracts.md) | Implementation notes còn hữu ích | Giữ, chú thích verification thuộc lần implementation, bỏ test count dễ stale |
| [REPORT](../REPORT.md) | Số liệu và CLI pipeline cũ dễ bị coi là current | Giữ nguyên body/data, thêm historical warning |
| [PRUNING_GUIDE](PRUNING_GUIDE.md) | Legacy scripts, trùng hướng dẫn, claim speedup | Giữ body, đánh dấu legacy, chuyển hướng canonical |
| [TECHNICAL_DOCUMENTATION](TECHNICAL_DOCUMENTATION.md) | Legacy production/API/script claims, trùng nhiều nội dung | Giữ body, đánh dấu legacy |
| [pruning_workflows](pruning_workflows.md) | Depth/sensitivity/unstructured claims mâu thuẫn code | Giữ historical body, không dùng như guide hiện hành |
| Presentation/PDF và benchmark artifacts cũ | Không có provenance chứng minh unified flow hiện tại | Giữ nguyên; không coi là experimental validation hiện hành |

## Những claim được thay thế trong docs chính

Không quảng bá production-ready, arbitrary topology safety, ResNet adapter, full attention/Linear pruning cho RT-DETR, guaranteed speedup hoặc real YOLO recovery đã sẵn sàng. SNIP/GraSP/SynFlow/LAMP là unstructured criteria trong implementation hiện tại; Taylor là structured criterion với unified Conv calibration targets. Sensitivity không bắt buộc cho depth và không tương thích global-unstructured layer-wise policy.

Tests tiny/mock/synthetic chỉ được gọi Validated; full detector end-to-end chưa Experimentally validated. Historical metrics vẫn giữ nguyên và không được dùng để chứng minh flow mới. Legacy body có thể chứa path/script cũ không còn tồn tại; không sửa chúng thành kết quả hoặc recipe mới.

## Verification scope

Kiểm tra links/anchors trong tài liệu hiện hành và các link mới ở historical banners; đối chiếu CLI flags với `main.py`, symbols/config với source, snippet YAML với config thật; `git diff --check`. Không chạy commands planning hoặc experiment để kiểm chứng docs. Không cài dependencies hay chạy lại full test suite. Các historical body links không được chứng nhận là executable/current.
