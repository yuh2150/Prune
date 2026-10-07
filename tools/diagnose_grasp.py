"""Run pretrained LeNet GraSP diagnostics without recovery or ranking heuristics."""
import json
from pathlib import Path

from prune_framework.core.config import FrameworkConfig
from prune_framework.pipelines.unified import UnifiedPruningPipeline


def main():
    output = Path('artifacts/grasp_diagnostic_20261006')
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for ratio in (.01, .05, .10):
        cfg = FrameworkConfig.from_yaml('configs/lenet5_custom_47labels_grasp_reference.yaml')
        cfg.pruning.amount = ratio
        cfg.experiment.name = f'lenet5_grasp_weighted_diagnostic_{int(ratio*100):02d}'
        cfg.output_path = f'artifacts/{cfg.experiment.name}.pt'
        cfg.validate()
        result = UnifiedPruningPipeline(cfg).run()
        path = max(Path('artifacts').glob(cfg.experiment.name + '_*/result.json'))
        saved = json.loads(path.read_text())
        calibration = saved['stages']['importance_calibration']
        plan = saved['pruning']['pruning_plan']
        counts = {g['primary']['name']: len(g['indices']) for g in plan['groups']}
        targets = []
        for name, statistics in calibration['target_statistics'].items():
            count = counts.get(name, 0)
            row = {'target_name': name, **statistics, 'pruned_count': count,
                   'pruned_fraction': count / statistics['numel']}
            targets.append(row)
            print(json.dumps(row), flush=True)
        total = plan['metadata']['total_elements']
        selected = sum(counts.values())
        record = {
            'result_path': str(path), 'ratio': ratio, 'baseline': saved['baseline_metrics'],
            'post_prune': saved['final_metrics'], 'requested_global_sparsity': ratio,
            'actual_global_sparsity_prunable': selected / total,
            'actual_global_sparsity_whole_model': saved['complexity_after']['sparsity_pct'] / 100,
            'total_prunable_elements': total, 'selected_elements': selected,
            'temperature': 200, 'parameter_scope': calibration['parameter_scope'],
            'fine_tuning': False, 'targets': targets,
        }
        records.append(record)
        (output/'diagnostics.json').write_text(json.dumps(records, indent=2))
        print(json.dumps({k: v for k, v in record.items() if k != 'targets'}), flush=True)
    lines = ['# GraSP weighted calibration diagnostic — 06/10/2026', '',
             'Pretrained checkpoint, temperature 200, 4 calibration batches, microbatch 20, seed 42, no fine-tuning. No layer normalization or caps.', '',
             '| Ratio | Baseline accuracy | Post-prune accuracy | Baseline loss | Post-prune loss | Actual sparsity (prunable) | Actual sparsity (whole model) |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for record in records:
        lines.append(f'| {record["ratio"]:.0%} | {record["baseline"]["accuracy"]:.2%} | {record["post_prune"]["accuracy"]:.2%} | {record["baseline"]["loss"]:.6f} | {record["post_prune"]["loss"]:.6f} | {record["actual_global_sparsity_prunable"]:.4%} | {record["actual_global_sparsity_whole_model"]:.4%} |')
    for record in records:
        lines += ['', f'## Ratio {record["ratio"]:.0%}', '', f'Source: {record["result_path"]}', '',
                  '| Target | Numel | Pruned | Fraction | Score min | Score max | Score mean | Score std | Gradient norm | HVP norm |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for t in record['targets']:
            lines.append(f'| {t["target_name"]} | {t["numel"]} | {t["pruned_count"]} | {t["pruned_fraction"]:.4%} | ' + ' | '.join(f'{t[key]:.8e}' for key in ['raw_score_min','raw_score_max','raw_score_mean','raw_score_std','first_order_grad_norm','hvp_norm']) + ' |')
    (output/'diagnostics.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
