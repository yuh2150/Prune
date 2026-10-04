"""Static dry-run and mock-only orchestration checks; no detector experiment."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from main import main
from prune_framework.contracts.sensitivity_result import (
    SensitivityResult, LayerSensitivityProfile, SensitivityPoint,
)
from prune_framework.core.config import FrameworkConfig
from prune_framework.core.engine import PruningEngine
from prune_framework.modules.analysis.sensitivity import SensitivityAnalyzer
from prune_framework.pipelines.unified import UnifiedPruningPipeline


class StopAtPolicy(Exception):
    pass


class TestFlowAudit(unittest.TestCase):
    def test_cli_dry_run_never_constructs_pipeline_or_executes_work(self):
        config_path = Path(__file__).resolve().parents[2] / 'configs/research_unified.yaml'
        with patch('sys.argv', ['main.py', '--config', str(config_path), '--dry-run']), \
             patch.object(UnifiedPruningPipeline, '__init__', side_effect=AssertionError('pipeline constructed')), \
             patch('main.run_pruning_pipeline', side_effect=AssertionError('execution')), \
             patch('prune_framework.pipelines.unified.ModelLoader.load', side_effect=AssertionError('model load')), \
             patch.object(UnifiedPruningPipeline, '_load_callback', side_effect=AssertionError('callback import')), \
             patch('prune_framework.pipelines.unified.ExperimentArtifacts', side_effect=AssertionError('artifact write')):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                main()
        report = json.loads(out.getvalue())
        self.assertEqual(report['executed_stages'], [])
        self.assertEqual(report['criterion'], 'taylor')
        order = report['configured_run_order']
        self.assertLess(order.index('importance_calibration'), order.index('sensitivity_probes'))
        self.assertIn('BUILD_PLAN_ONLY_STOP', order)
        self.assertEqual(report['callbacks_not_imported']['calibration'], 'experiments.yolov5:calibrate_taylor')

    def test_dry_run_respects_disabled_stages_and_global_policy_warning(self):
        cfg = FrameworkConfig.from_dict({
            'pruning': {'pruner': 'unstructured', 'criterion': 'lamp', 'granularity': 'weight'},
            'sensitivity': {'enabled': True},
            'benchmark': {'enabled': False, 'params': False, 'flops': False},
        })
        with self.assertRaisesRegex(ValueError, 'does not support layer-wise'):
            UnifiedPruningPipeline.describe_flow(cfg)
        cfg.sensitivity.enabled = False
        report = UnifiedPruningPipeline.describe_flow(cfg)
        for disabled in ('baseline_complexity', 'dependency_preflight', 'importance_calibration',
                         'recovery_callback', 'pytorch_latency_benchmark', 'onnx_export'):
            self.assertNotIn(disabled, report['configured_run_order'])

    def test_sensitivity_probes_start_from_independent_baseline_copies(self):
        model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.Conv2d(4, 4, 1))
        original = {key: value.clone() for key, value in model.state_dict().items()}
        candidates = []
        def probe(engine, candidate, config, verify_forward):
            self.assertIsNot(candidate, model)
            for key, value in candidate.state_dict().items():
                torch.testing.assert_close(value, original[key])
            candidates.append(candidate)
            with torch.no_grad():
                next(candidate.parameters()).fill_(0)
        analyzer = SensitivityAnalyzer('yolov5', 'structured', 'l1', 'channel')
        with patch.object(PruningEngine, 'execute', autospec=True, side_effect=probe):
            result = analyzer.analyze(model, [.1, .2], lambda _: .8)
        self.assertEqual(len(candidates), 4)
        self.assertEqual(len({id(candidate) for candidate in candidates}), 4)
        self.assertTrue(all(point.is_valid for profile in result.profiles.values() for point in profile.points.values()))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, original[key])

    def test_sensitivity_selection_reaches_final_engine_as_nonuniform_ratios(self):
        profile = SensitivityResult(.8, {
            0: LayerSensitivityProfile(0, '0', {
                .1: SensitivityPoint(.1, .79, .01, .0125),
                .3: SensitivityPoint(.3, .6, .2, .25),
            }),
            1: LayerSensitivityProfile(1, '1', {
                .1: SensitivityPoint(.1, .8, 0, 0),
                .3: SensitivityPoint(.3, .79, .01, .0125),
            }),
        })
        model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.Conv2d(4, 4, 1))
        captured = []
        def stop(engine, candidate, config, verify_forward, plan=None):
            captured.append(config)
            raise StopAtPolicy()
        with tempfile.TemporaryDirectory() as directory:
            cfg = FrameworkConfig.from_dict({
                'model': {'device': 'cpu'},
                'pruning': {'amount': .3, 'global': True},
                'sensitivity': {'enabled': True, 'rates': [.1, .3]},
                'benchmark': {'enabled': False, 'params': False, 'flops': False},
                'experiment': {'output_dir': directory},
            })
            with patch('prune_framework.pipelines.unified.ModelLoader.load', return_value=(model, None)), \
                 patch.object(SensitivityAnalyzer, 'analyze', return_value=profile), \
                 patch.object(PruningEngine, 'build_dependency_graph'), \
                 patch.object(PruningEngine, 'execute', autospec=True, side_effect=stop), \
                 patch('prune_framework.pipelines.unified.ModelExporter.export_checkpoint', side_effect=AssertionError('export')):
                with self.assertRaises(StopAtPolicy):
                    UnifiedPruningPipeline(cfg, evaluator=lambda model: {'map': .8}).run()
            self.assertEqual(captured[0]['pruning_params'], [(0, .1), (1, .3)])
            selection_file = next(Path(directory).glob('*/selection.json'))
            self.assertEqual([row['rate'] for row in json.loads(selection_file.read_text())['layers']], [.1, .3])

    def test_evaluator_tuple_needs_explicit_callback_adapter(self):
        pipeline = UnifiedPruningPipeline(FrameworkConfig(), evaluator=lambda model: ((0.,) * 7, [], ()))
        with self.assertRaisesRegex(TypeError, 'float, metric dict, or EvaluationResult'):
            pipeline._evaluate(nn.Linear(2, 2), 'baseline')
