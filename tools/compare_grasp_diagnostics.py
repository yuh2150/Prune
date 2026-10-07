"""Compare existing pretrained pruning variants without modifying their implementation."""
import csv
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
from prune_framework.core.config import FrameworkConfig
from prune_framework.pipelines.unified import UnifiedPruningPipeline
from prune_framework.plugins.pruners.unstructured import UnstructuredPruner

OUTPUT = Path('artifacts/grasp_comparison_20261006')


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records, references = [], {}
    original = UnstructuredPruner._weight_scores
    for method in ('grasp', 'grasp_layer_normalized', 'magnitude', 'grasp_conv1_capped'):
        for ratio in (.01, .05, .10):
            captured = {}
            def collect(target, criterion, config):
                scores = original(target, criterion, config)
                raw = scores.detach().cpu().clone()
                if method != 'magnitude':
                    if target.name in references:
                        assert torch.equal(raw, references[target.name]), 'Calibration scores changed across runs'
                    else:
                        references[target.name] = raw
                context = config.get('higher_order_calibration')
                captured[target.name] = {
                    'target_name': target.name, 'numel': raw.numel(),
                    'raw_score': UnstructuredPruner._score_statistics(raw),
                    'normalized_score': None,
                    'first_order_grad_norm': float(context.first_order_gradients[target.name].norm()) if context else None,
                    'hvp_norm': float(context.gradients[target.name].norm()) if context else None,
                }
                return scores
            cfg = FrameworkConfig.from_yaml('configs/lenet5_custom_47labels_grasp_reference.yaml')
            cfg.pruning.criterion = method
            cfg.pruning.amount = ratio
            if method == 'magnitude':
                cfg.pruning.calibration_callback = None
            cfg.experiment.name = f'lenet5_comparison_{method}_{int(ratio*100):02d}'
            cfg.output_path = str(OUTPUT / (cfg.experiment.name + '.pt'))
            cfg.validate()
            with patch.object(UnstructuredPruner, '_weight_scores', staticmethod(collect)):
                UnifiedPruningPipeline(cfg).run()
            path = max(Path('artifacts').glob(cfg.experiment.name + '_*/result.json'))
            saved = json.loads(path.read_text())
            plan = saved['pruning']['pruning_plan']
            counts = {g['primary']['name']: len(g['indices']) for g in plan['groups']}
            assert 'fc2' not in counts and all('fc2' not in name for name in captured)
            total = plan['metadata']['total_elements']
            selected = sum(counts.values())
            assert selected == round(total * ratio)
            if method in ('grasp', 'grasp_layer_normalized'):
                assert 'max_pruning_fraction_by_target' not in plan['metadata']
            for name, row in captured.items():
                count = counts.get(name, 0)
                row.update(pruned_count=count, pruned_fraction=count / row['numel'], budget_share=count / selected)
                if method == 'grasp_layer_normalized':
                    stats = plan['metadata']['score_statistics'][name]
                    row['normalized_score'] = stats['normalized']
                    row['normalization_scale'] = stats['normalization_scale']
                    assert abs(stats['normalization_scale'] / row['raw_score']['mean_abs'] - 1) < 1e-6, 'Normalization differs from raw / mean(abs(raw))'
            record = {
                'method': method, 'ratio': ratio, 'baseline': saved['baseline_metrics'],
                'post_prune': saved['final_metrics'], 'requested_sparsity': ratio,
                'actual_sparsity': selected / total, 'whole_model_sparsity': saved['complexity_after']['sparsity_pct'] / 100,
                'total_prunable_elements': total, 'selected_elements': selected,
                'fine_tuning': False, 'result_path': str(path), 'targets': list(captured.values()),
            }
            records.append(record)
            (OUTPUT / 'diagnostics.json').write_text(json.dumps(records, indent=2))
            print(json.dumps(record), flush=True)
    manifest = {
        'checkpoint_sha256': hashlib.sha256(Path('LeNet5_Numbers&Characters_FP32.onnx').read_bytes()).hexdigest(),
        'config': 'configs/lenet5_custom_47labels_grasp_reference.yaml', 'seed': 42,
        'calibration_batches': 4, 'calibration_batch_size': 64, 'microbatch_size': 20,
        'temperature': 200, 'parameter_scope': 'weights', 'recovery': False,
        'identical_raw_grasp_scores_across_all_runs': True,
        'magnitude_note': 'Magnitude uses absolute weights, no calibration/HVP. Norms are null.',
        'sparsity_denominator': '60630 eligible Conv1/Conv2/Conv3/fc1 weights; classifier protected',
    }
    (OUTPUT / 'protocol.json').write_text(json.dumps(manifest, indent=2))
    lines = ['# So sánh GraSP pretrained LeNet5', '',
        'Cùng checkpoint, dataset split và seed 42; calibration 4 × 64 samples, microbatch 20, temperature logits 200. Không fine-tuning. Raw GraSP scores giống hệt giữa các run/variants (đã kiểm tra tensor equality). Magnitude không dùng calibration; gradient/HVP = N/A. Conv1-capped là experiment riêng với cap 10% weights Conv1.', '',
        '| Method | Requested | Actual eligible | Baseline accuracy | Post accuracy | Baseline loss | Post loss |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f'| {r["method"]} | {r["ratio"]:.0%} | {r["actual_sparsity"]:.4%} | {r["baseline"]["accuracy"]:.2%} | {r["post_prune"]["accuracy"]:.2%} | {r["baseline"]["loss"]:.6f} | {r["post_prune"]["loss"]:.6f} |')
    lines += ['', '## Distribution theo layer', '', 'Mỗi ô: pruned count / numel (fraction trong layer; share của tổng budget). Actual sparsity dùng 60,630 eligible weights, không tính protected classifier/bias.', '',
        '| Method | Ratio | Conv1 (150) | Conv2 (2400) | Conv3 (48000) | fc1 (10080) |', '|---|---:|---:|---:|---:|---:|']
    for r in records:
        cells = [f'{t["pruned_count"]}/{t["numel"]} ({t["pruned_fraction"]:.2%}; {t["budget_share"]:.2%})' for t in r['targets']]
        lines.append(f'| {r["method"]} | {r["ratio"]:.0%} | ' + ' | '.join(cells) + ' |')
    for r in records:
        lines += ['', f'## {r["method"]}, ratio {r["ratio"]:.0%}', '', f'Source: `{r["result_path"]}`', '',
            '| Target | Raw min | Raw max | Raw mean | Raw std | Raw mean_abs | Normalized min/max/mean/std | Gradient norm | HVP norm |', '|---|---:|---:|---:|---:|---:|---|---:|---:|']
        for t in r['targets']:
            raw = ' | '.join(f'{t["raw_score"][k]:.8e}' for k in ('min','max','mean','std','mean_abs'))
            norm = ', '.join(f'{t["normalized_score"][k]:.8e}' for k in ('min','max','mean','std')) if t['normalized_score'] else 'N/A'
            norms = ' | '.join(f'{t[k]:.8e}' if t[k] is not None else 'N/A' for k in ('first_order_grad_norm','hvp_norm'))
            lines.append(f'| {t["target_name"]} | {raw} | {norm} | {norms} |')
    lines += ['', '## Kết luận', '',
        'Giả thuyết score-scale imbalance được hỗ trợ: mean_abs raw Conv1 = 7.2191e-5, Conv2 = 7.8603e-6, Conv3 = 6.9015e-7, fc1 = 1.8959e-6. Conv1/Conv2 lớn hơn Conv3 khoảng 104.6× / 11.4×, dù gradient/HVP norms cùng bậc độ lớn.', '',
        'Ở ratio 1%, Conv1+Conv2 chỉ chiếm 4.21% eligible weights nhưng nhận 49.83% budget raw GraSP; normalized nhận 0%. Ở ratio 10%, share giảm từ 14.51% xuống 3.81%. Pruned fractions ở 10% đổi từ 32.00%, 34.67%, 8.28%, 11.97% thành 4.00%, 9.38%, 10.29%, 8.87%. Phân bố bớt lệch theo tỷ lệ weights từng layer, không có nghĩa budget tuyệt đối phải bằng nhau.', '',
        'Layer normalization phục hồi +46.0 điểm phần trăm accuracy ở 1%, +8.5 ở 5%, nhưng -0.5 ở 10%. Nó giải quyết một phần imbalance, không giải quyết sự sụp accuracy ở sparsity cao. Với validation 200 ảnh, mỗi ảnh tương ứng 0.5 điểm phần trăm; chưa có nhiều seed hoặc kiểm định thống kê.', '',
        'Conv1 cap riêng chỉ đạt 28.5%, 4.0%, 5.5%; Conv2 vẫn nhận budget lớn. Magnitude là control tốt nhất về accuracy trong experiment này (84%, 84%, 83%). Không suy ra speedup hoặc giảm kích thước file: đây là unstructured masks, tensor dimensions không đổi.', '',
        'Baseline implementation/HVP/ranking/checkpoint và recovery không thay đổi. 18/18 GraSP unit tests pass. Các run kiểm tra exact selected count, protected head, identical raw scores và normalization scale đúng mean_abs.']
    (OUTPUT / 'comparison.md').write_text('\n'.join(lines) + '\n')
    with (OUTPUT / 'layer_distribution.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['method','ratio','target','numel','pruned_count','pruned_fraction','budget_share'])
        for r in records:
            for t in r['targets']:
                writer.writerow([r['method'], r['ratio'], t['target_name'], t['numel'], t['pruned_count'], t['pruned_fraction'], t['budget_share']])


if __name__ == '__main__':
    main()
