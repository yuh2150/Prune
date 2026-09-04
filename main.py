import sys
import argparse
from prune_framework.core.registry import PluginRegistry
from prune_framework.core.config import FrameworkConfig
from prune_framework.pipelines.pruning import run_pruning_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Modular + Plugin-based Model Pruning Framework (YOLOv5, RT-DETR, ResNet)"
    )

    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")

    # Dynamic Listing Flags
    parser.add_argument("--list-models", action="store_true", help="List all registered model adapters")
    parser.add_argument("--list-pruners", action="store_true", help="List all registered pruner strategies")
    parser.add_argument("--list-criteria", action="store_true", help="List all registered importance criteria")
    parser.add_argument("--list-granularities", action="store_true", help="List all registered granularities")
    parser.add_argument("--list-selectors", action="store_true", help="List all registered layer selectors")

    # Overrides
    parser.add_argument("--model", type=str, default=None, help="Target model adapter architecture")
    parser.add_argument("--strategy", type=str, default=None, help="Pruning strategy plugin")
    parser.add_argument("--criterion", type=str, default=None, help="Importance evaluation metric plugin")
    parser.add_argument("--granularity", type=str, default=None, help="Granularity plugin")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights checkpoint")
    parser.add_argument("--amount", type=float, default=None, help="Global pruning rate amount")
    parser.add_argument("--output-path", type=str, default=None, help="Output checkpoint destination path")

    return parser.parse_args()


def main():
    args = parse_args()

    # Dynamic Listing Execution
    if args.list_models:
        print("Registered Model Adapters:", PluginRegistry.list_models())
        sys.exit(0)
    if args.list_pruners:
        print("Registered Pruners:", PluginRegistry.list_pruners())
        sys.exit(0)
    if args.list_criteria:
        print("Registered Criteria:", PluginRegistry.list_criteria())
        sys.exit(0)
    if args.list_granularities:
        print("Registered Granularities:", PluginRegistry.list_granularities())
        sys.exit(0)
    if args.list_selectors:
        print("Registered Selectors:", PluginRegistry.list_selectors())
        sys.exit(0)

    # Load Configuration
    if args.config:
        config = FrameworkConfig.from_yaml(args.config)
    else:
        config = FrameworkConfig()

    # Apply Command Line Overrides
    if args.model:
        config.model.name = args.model
    if args.weights:
        config.model.weights = args.weights
    if args.strategy:
        config.pruning.pruner = args.strategy
    if args.criterion:
        config.pruning.criterion = args.criterion
    if args.granularity:
        config.pruning.granularity = args.granularity
    if args.amount is not None:
        config.pruning.amount = args.amount
    if args.output_path:
        config.output_path = args.output_path

    # Run Pipeline
    run_pruning_pipeline(config)


if __name__ == "__main__":
    main()
