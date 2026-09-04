# Model Pruning & Sensitivity Analysis Framework — Documentation Index

Welcome to the technical documentation for the **Model Pruning & Sensitivity Analysis Framework**. This framework is a production-grade, modular, plugin-based suite for model compression, sensitivity analysis, automated layer selection, and evaluation (targeting object detectors like YOLOv5, RT-DETR, and classification backbones like ResNet).

---

## 🧭 Developer Learning Path

```text
New Developer / User
         │
         ▼
[1. Getting Started](getting-started/quickstart.md)
  ├── Installation & Conda Setup
  ├── CLI Quickstart (main.py)
  └── Python Pipeline API Quickstart
         │
         ▼
[2. Architecture Overview](architecture/overview.md)
  ├── High-Level System Design
  ├── Composition & Plugin Pattern
  ├── Component Boundary Matrix
  └── End-to-End Data Flow
         │
         ▼
[3. Core Concepts](concepts/pruning-and-sensitivity.md)
  ├── Pruning Strategies (Structured, Unstructured, Depth, Taylor)
  ├── Sensitivity Analysis & Profiling
  ├── Automated Layer Selection Strategies
  └── Decoupled Evaluation Pipeline
         │
         ▼
[4. Framework API Reference](api/framework-api.md)
  ├── Core Registry & Configuration (prune_framework.core)
  ├── Domain Contracts & Models (prune_framework.contracts)
  ├── Analysis & Evaluation Modules (prune_framework.modules)
  └── Plugin Implementations (prune_framework.plugins)
         │
         ▼
[5. Developer & Extension Guide](development/extending-framework.md)
  ├── Adding a New Model Adapter
  ├── Adding a New Layer Selector
  ├── Adding a New Pruning Strategy
  └── Running Unit & Architecture Tests
         │
         ▼
[6. Configuration Reference](configuration/reference.md)
  ├── YAML Configuration Schema (FrameworkConfig)
  └── CLI Override Reference
```

---

## 📂 Documentation Directory Structure

```text
docs/
├── README.md                          # Main Documentation Sitemap & Learning Path
├── getting-started/
│   └── quickstart.md                  # Installation, Repository Map & Quickstart
├── architecture/
│   └── overview.md                    # System Architecture, Diagrams, Boundaries & Data Flow
├── concepts/
│   └── pruning-and-sensitivity.md     # Pruning, Sensitivity Analysis, Layer Selection, Evaluation
├── api/
│   └── framework-api.md               # Complete API Reference for Core, Contracts, Modules, Plugins
├── development/
│   └── extending-framework.md         # Developer Extension Guide (Models, Selectors, Pruners, Tests)
└── configuration/
    └── reference.md                   # YAML & CLI Configuration Reference
```

---

## 🔍 Key Framework Characteristics

* **Plugin Architecture**: Model adapters, pruners, importance criteria, granularities, and layer selectors are decoupled plugins registered via `@register_*` decorators.
* **Type-Safe Domain Models**: Rich, immutable dataclasses (`SensitivityResult`, `SelectionResult`, `EvaluationResult`, `Detection`, `PruningResult`) prevent raw dictionary coupling while supporting 100% backward-compatible tuple iteration.
* **Decoupled Evaluation Pipeline**: Clean separation between input pre-processing (`BasePreProcessor`), model execution, output post-processing (`BasePostProcessor`), and metric computation (`BaseEvaluator`).
* **Strict Architecture Boundaries**: Architectural tests enforce dependency isolation so `prune_framework` never illegally imports legacy root scripts or un-isolated utilities.
