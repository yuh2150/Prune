"""Offline contracts: calibration/policy/plan boundary, targets and tiny graphs."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from experiments import yolov5
from prune_framework.contracts.deployment import DeploymentTargets, TargetChecker
from prune_framework.contracts.evaluation import normalize_evaluation, normalize_yolo_evaluation
from prune_framework.core.config import FrameworkConfig
from prune_framework.core.engine import PruningEngine
from prune_framework.core.results import ComplexityResult, PlanBuildResult
from prune_framework.modules.calibration.context import CalibrationContext
from prune_framework.modules.analysis.sensitivity import SensitivityAnalyzer
from prune_framework.pipelines.unified import UnifiedPruningPipeline
from prune_framework.plugins.adapters.yolov5 import YOLOv5Adapter
from prune_framework.plugins.pruners.structured import StructuredPruner


def calibration_callback(model, config, stage, **kwargs):
    if stage != 'calibration':
        raise AssertionError(stage)
    return CalibrationContext([torch.ones(1, 3, 8, 8)], lambda m, x: m(x).square().mean())


class TinyTopology(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Conv2d(3, 8, 1)
        self.bn = nn.BatchNorm2d(8)
        self.right = nn.Conv2d(3, 8, 1)
        self.split = nn.Conv2d(3, 4, 1)
        self.consumer = nn.Conv2d(12, 8, 1)
        self.detect = nn.Conv2d(8, 6, 1)

    def forward(self, x):
        added = self.bn(self.left(x)) + self.right(x)
        return self.detect(self.consumer(torch.cat([added, self.split(x)], dim=1)))


class TestEvaluationBridge(unittest.TestCase):
    def test_normalization_and_strict_schema(self):
        result = normalize_yolo_evaluation(((.8, .7, .6, .5, 1., 2., 3.), np.array([.5]), (1., 2., 3., 64, 64, 1)))
        self.assertEqual(result.map, .5)
        self.assertEqual(result.metrics['map50_95'], .5)
        self.assertEqual(result.metrics['total_ms'], 3.)
        for invalid in [None, ((), [], ()), ((0.,) * 6, np.array([]), (0.,) * 6)]:
            with self.subTest(invalid=invalid), self.assertRaises((ValueError, TypeError)):
                normalize_yolo_evaluation(invalid)
        for invalid in [{'map': float('nan')}, {'map': '0.5'}, {}, {'map': .3, 'map50_95': .4}]:
            with self.assertRaises(ValueError):
                normalize_evaluation(invalid)

    def test_callback_module_resolves_including_recovery(self):
        cfg = FrameworkConfig.from_yaml('configs/research_unified.yaml')
        pipeline = UnifiedPruningPipeline(cfg)
        self.assertIs(pipeline.evaluator, yolov5.evaluate)
        self.assertIs(pipeline.calibration_callback, yolov5.calibrate_taylor)
        self.assertIs(pipeline.recovery, yolov5.recover)
        output = ((.8, .7, .6, .5, 0., 0., 0.), np.array([.5]), (1., 2., 3., 64, 64, 1))
        with patch('test.test', return_value=output) as evaluator:
            result = yolov5.evaluate(nn.Linear(1, 1), data={'nc': 1})
        self.assertEqual(result.map50, .6)
        self.assertFalse(evaluator.call_args.kwargs['plots'])


class TestDeploymentTargets(unittest.TestCase):
    def test_target_check_gates_finalization_in_unified(self):
        model = nn.Sequential(nn.Conv2d(3, 8, 1))
        with tempfile.TemporaryDirectory() as directory:
            cfg = FrameworkConfig.from_dict({
                'targets': {'parameter_reduction': .99},
                'benchmark': {'enabled': False, 'params': False, 'flops': False},
                'experiment': {'output_dir': directory},
            })
            with patch('prune_framework.pipelines.unified.ModelLoader.load', return_value=(model, None)), \
                 patch.object(YOLOv5Adapter, 'get_dummy_input', return_value=torch.ones(1, 3, 8, 8)), \
                 patch('prune_framework.pipelines.unified.ModelExporter.export_checkpoint') as export:
                with self.assertRaisesRegex(Exception, 'Deployment targets not reached'):
                    UnifiedPruningPipeline(cfg).run()
                export.assert_not_called()
            report = json.loads(next(Path(directory).glob('*/target_check.json')).read_text())
            self.assertFalse(report['reached'])


    def test_parameter_map_constraints_and_missing_measurement(self):
        before, after = ComplexityResult(params=100, flops=200), ComplexityResult(params=70, flops=150)
        targets = DeploymentTargets(parameter_reduction=.2, max_map50_95_drop=.03)
        self.assertTrue(TargetChecker.check({'map': .5}, {'map50_95': .48}, targets, before, after).reached)
        failed = TargetChecker.check({'map': .5}, {'map': .4}, targets, before, ComplexityResult(params=90))
        self.assertEqual(len(failed.violations), 2)
        self.assertFalse(failed.reached)
        targets = DeploymentTargets(max_latency_ms=10, max_memory_mb=100, max_map50_drop=.1)
        missing = TargetChecker.check({}, {}, targets, before, after)
        self.assertFalse(missing.reached)
        self.assertEqual(len(missing.unavailable), 3)
        self.assertTrue(TargetChecker.check({'map50': .5}, {'map50': .45, 'total_ms': 8, 'memory_mb': 80}, targets).reached)
        self.assertTrue(TargetChecker.check({}, {}, DeploymentTargets(flops_reduction=.2), before, after).reached)

    def test_targets_are_not_pruning_amount(self):
        cfg = FrameworkConfig.from_dict({'targets': {'parameter_reduction': .2}, 'pruning': {'amount': .7}})
        self.assertEqual(cfg.targets.parameter_reduction, .2)
        self.assertEqual(cfg.pruning.amount, .7)
        with self.assertRaises(ValueError):
            DeploymentTargets(parameter_reduction=2).validate()


class TestPlanningBoundary(unittest.TestCase):
    def test_cli_build_plan_only_dispatch(self):
        import main
        from types import SimpleNamespace
        plan = SimpleNamespace(describe=lambda: {'groups': []})
        with patch('sys.argv', ['main.py', '--build-plan-only']), \
             patch('main.UnifiedPruningPipeline') as pipeline, \
             patch('main.run_pruning_pipeline', side_effect=AssertionError('full run')):
            pipeline.return_value.run.return_value = SimpleNamespace(plan=plan)
            main.main()
            pipeline.return_value.run.assert_called_once_with(build_plan_only=True)

    def test_all_invalid_sensitivity_cannot_reach_plan(self):
        from prune_framework.contracts.sensitivity_result import SensitivityResult, LayerSensitivityProfile, SensitivityPoint
        model = nn.Sequential(nn.Conv2d(3, 8, 1))
        profile = SensitivityResult(.8, {0: LayerSensitivityProfile(0, '0', {.1: SensitivityPoint(.1, 0, 1, 1, False, 'failed')})})
        with tempfile.TemporaryDirectory() as directory:
            cfg = FrameworkConfig.from_dict({'sensitivity': {'enabled': True},
                'benchmark': {'enabled': False, 'params': False, 'flops': False}, 'experiment': {'output_dir': directory}})
            with patch('prune_framework.pipelines.unified.ModelLoader.load', return_value=(model, None)), \
                 patch.object(PruningEngine, 'build_dependency_graph'), \
                 patch.object(SensitivityAnalyzer, 'analyze', return_value=profile), \
                 patch.object(PruningEngine, 'build_plan', side_effect=AssertionError('built invalid plan')):
                with self.assertRaisesRegex(Exception, 'no valid probes'):
                    UnifiedPruningPipeline(cfg, evaluator=lambda model: .8).run(build_plan_only=True)


    def test_callback_calibration_precedes_sensitivity_and_plan_only_preserves_model(self):
        model = nn.Sequential(nn.Conv2d(3, 8, 1), nn.ReLU(), nn.Conv2d(8, 8, 1))
        before = copy.deepcopy(model.state_dict())
        events, payloads, configs = [], [], []
        original_callback = calibration_callback
        analyze = SensitivityAnalyzer.analyze
        build = PruningEngine.build_plan
        def calibrated(*args, **kwargs):
            events.append('calibration')
            return original_callback(*args, **kwargs)
        def observed_analysis(analyzer, *args, **kwargs):
            events.append('sensitivity')
            self.assertIn('gradient_calibration', kwargs['calibration'])
            payloads.append(kwargs['calibration']['gradient_calibration'])
            return analyze(analyzer, *args, **kwargs)
        def observed_build(engine, candidate, config):
            configs.append(config)
            return build(engine, candidate, config)
        def evaluator(model, stage):
            # First layer has higher sensitivity than the second.
            width = model[0].out_channels
            return {'map': .8 if width == 8 else (.79 if width == 6 else .5)}
        with tempfile.TemporaryDirectory() as directory:
            cfg = FrameworkConfig.from_dict({
                'model': {'device': 'cpu'},
                'pruning': {'criterion': 'taylor', 'amount': .5,
                            'calibration_callback': 'tests.unit.test_orchestration_contracts:calibration_callback'},
                'sensitivity': {'enabled': True, 'rates': [.25, .5]},
                'benchmark': {'enabled': False, 'params': False, 'flops': False},
                'experiment': {'output_dir': directory},
                'output_path': str(Path(directory) / 'must-not-exist.pt'),
            })
            with patch('prune_framework.pipelines.unified.ModelLoader.load', return_value=(model, None)), \
                 patch.object(YOLOv5Adapter, 'get_dummy_input', return_value=torch.ones(1, 3, 8, 8)), \
                 patch('tests.unit.test_orchestration_contracts.calibration_callback', side_effect=calibrated) as callback, \
                 patch.object(SensitivityAnalyzer, 'analyze', autospec=True, side_effect=observed_analysis), \
                 patch.object(PruningEngine, 'build_plan', autospec=True, side_effect=observed_build), \
                 patch('prune_framework.pipelines.unified.ModelExporter.export_checkpoint', side_effect=AssertionError('export')), \
                 patch('prune_framework.pipelines.unified.LatencyBenchmark.benchmark', side_effect=AssertionError('benchmark')):
                # Explicit signature keeps the config callback invocation contract.
                callback.__signature__ = __import__('inspect').signature(original_callback)
                result = UnifiedPruningPipeline(cfg, evaluator=evaluator).run(build_plan_only=True)
            self.assertIsInstance(result, PlanBuildResult)
            self.assertEqual(events, ['calibration', 'sensitivity'])
            self.assertEqual(callback.call_count, 1)
            self.assertTrue(all(config['gradient_calibration'] is payloads[0] for config in configs))
            self.assertEqual(configs[-1]['pruning_params'], [(0, .25), (1, .5)])
            self.assertTrue(all(group.validated for group in result.plan.groups))
            self.assertFalse(Path(cfg.output_path).exists())
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, before[key])
            self.assertEqual(model[0].out_channels, 8)

    def test_invalid_combinations_and_policy_rejected_before_mutation(self):
        model = nn.Sequential(nn.Conv2d(3, 8, 1))
        for mode, criterion, config in [
            ('structured', 'snip', {'amount': .3}),
            ('depth', 'snip', {'amount': .3}),
            ('unstructured', 'l0_gate', {'amount': .3}),
            ('unstructured', 'lamp', {'pruning_params': [(0, .1)]}),
            ('depth', 'l1', {'pruning_params': [(0, .1)]}),
            ('structured', 'l1', {'pruning_params': [(99, .1)]}),
            ('structured', 'l1', {'pruning_params': [(0, -.1)]}),
            ('structured', 'l1', {'pruning_params': [(0, 1.)]}),
            ('structured', 'l1', {'pruning_params': [(0, .1), (0, .2)]}),
        ]:
            with self.subTest(mode=mode, criterion=criterion, config=config), self.assertRaises(ValueError):
                PruningEngine('yolov5', mode, criterion).build_plan(model, config)
        self.assertEqual(model[0].out_channels, 8)

    def test_empty_plan_requires_explicit_noop_and_layerwise_unstructured_works(self):
        model = nn.Sequential(nn.Conv2d(3, 8, 1), nn.Conv2d(8, 8, 1))
        engine = PruningEngine('yolov5', 'structured', 'l1')
        with self.assertRaisesRegex(ValueError, 'no executable actions'):
            engine.build_plan(model, {'amount': .2, 'pruning_params': [(0, 0.)]})
        empty = engine.build_plan(model, {'amount': .2, 'pruning_params': [(0, 0.)], 'allow_noop': True})
        self.assertTrue(empty.is_empty)
        plan = PruningEngine('yolov5', 'unstructured', 'magnitude', 'weight').build_plan(
            model, {'pruning_params': [(0, .25), (1, .5)]})
        self.assertEqual([len(g.indices) for g in plan.groups], [6, 32])


class TestTinyStructuredSafety(unittest.TestCase):
    def test_dependency_into_protected_output_is_rejected(self):
        class CoupledHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.branch = nn.Conv2d(3, 8, 1)
                self.detect = nn.Conv2d(3, 8, 1)
            def forward(self, x):
                return self.branch(x) + self.detect(x)
        model = CoupledHead()
        with patch.object(YOLOv5Adapter, 'get_dummy_input', return_value=torch.ones(1, 3, 8, 8)):
            with self.assertRaisesRegex(ValueError, 'Protected output'):
                PruningEngine('yolov5', 'structured', 'l1').build_plan(model, {'pruning_params': [(0, .25)]})
        self.assertEqual(model.branch.out_channels, 8)
        self.assertEqual(model.detect.out_channels, 8)

    def test_l1_receives_conv_bn_wrapper_correctly_and_bn_rejects_linear(self):
        from models.common import Conv
        model = nn.Sequential(Conv(3, 8, 1), Conv(8, 8, 1))
        with patch.object(YOLOv5Adapter, 'get_dummy_input', return_value=torch.ones(1, 3, 8, 8)):
            plan = PruningEngine('yolov5', 'structured', 'l1').build_plan(model, {'amount': .25})
            self.assertEqual(len(plan.groups), 2)
        with self.assertRaisesRegex(ValueError, 'BatchNorm2d'):
            PruningEngine('yolov5', 'structured', 'bn_scale').build_plan(nn.Sequential(nn.Linear(8, 8)), {'amount': .25})


    def test_conv_bn_residual_concat_head_and_rebuild(self):
        model = TinyTopology().eval()
        engine = PruningEngine('yolov5', 'structured', 'l1')
        original = model.consumer.weight.detach().clone()
        # left/right are coupled by Add. Second action refers to original indices.
        config = {'forced_channel_indices': {'left': [1, 3], 'right': [4, 6]}, 'amount': .5}
        with patch.object(YOLOv5Adapter, 'get_dummy_input', return_value=torch.ones(1, 3, 8, 8)):
            plan = engine.build_plan(model, config)
            self.assertEqual(model.left.out_channels, 8)
            build_graph = StructuredPruner._build_dependency_graph
            with patch.object(StructuredPruner, '_build_dependency_graph', side_effect=build_graph) as traced:
                engine.execute(model, config, plan=plan)
            self.assertEqual(traced.call_count, 2)
        self.assertEqual(model.left.out_channels, 4)
        self.assertEqual(model.right.out_channels, 4)
        self.assertEqual(model.bn.num_features, 4)
        self.assertEqual(model.consumer.in_channels, 8)
        self.assertEqual(model.detect.out_channels, 6)
        # Concat offsets: preserve left original [0,2,5,7], then split's [8..11].
        torch.testing.assert_close(model.consumer.weight, original[:, [0, 2, 5, 7, 8, 9, 10, 11]])
        self.assertEqual(model(torch.ones(1, 3, 8, 8)).shape, (1, 6, 8, 8))

    def test_protected_output_and_stale_plan_are_rejected(self):
        model = TinyTopology().eval()
        engine = PruningEngine('yolov5', 'structured', 'l1')
        with self.assertRaisesRegex(ValueError, 'protected'):
            engine.build_plan(model, {'forced_channel_indices': {'detect': [0, 1]}})
        with patch.object(YOLOv5Adapter, 'get_dummy_input', return_value=torch.ones(1, 3, 8, 8)):
            plan = engine.build_plan(model, {'forced_channel_indices': {'left': [1, 3]}})
            engine.execute(model, {}, plan=plan)
            with self.assertRaisesRegex(RuntimeError, 'shapes changed'):
                engine.execute(model, {}, plan=plan)
