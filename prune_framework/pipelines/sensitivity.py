import torch
from prune_framework.core.config import FrameworkConfig
from prune_framework.modules.model.loader import ModelLoader
from prune_framework.modules.analysis.sensitivity import SensitivityAnalyzer
from prune_framework.modules.analysis.layer_selection import LayerSelectorModule


def run_sensitivity_pipeline(config: FrameworkConfig, eval_fn, rates=[0.1, 0.2, 0.3, 0.4, 0.5]):
    device = torch.device(config.model.device if torch.cuda.is_available() else "cpu")
    model, _ = ModelLoader.load(config.model.name, config.model.weights, device)

    analyzer = SensitivityAnalyzer(
        model_name=config.model.name,
        pruner_name=config.pruning.pruner,
        criterion_name=config.pruning.criterion,
        granularity_name=config.pruning.granularity
    )

    results = analyzer.analyze(model, rates, eval_fn)

    if config.analysis.layer_selection:
        selector_module = LayerSelectorModule(selector_name="sensitivity")
        selection = selector_module.select(results, target_sparsity=config.pruning.amount)
        results.metadata["selection"] = selection

    return results

