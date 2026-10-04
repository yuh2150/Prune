"""Offline evaluator regression tests using actual NMS/AP and local image loading."""
import ast
import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import test as validation
from utils.metrics import fitness


class TinyDetector(torch.nn.Module):
    def __init__(self, rows=None):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer('rows', torch.tensor(rows if rows is not None else
                                                 [[32, 32, 20, 20, .99, .99, 0.0]], dtype=torch.float32))
        self.stride = torch.tensor([32])
        self.names = ['a', 'b']

    def forward(self, images, augment=False):
        assert not self.training
        assert not torch.is_grad_enabled()
        return self.rows[None].repeat(len(images), 1, 1), [self.anchor.expand(len(images), 1)]


def batch(targets=None):
    targets = [[0, 0, .5, .5, 20/64, 20/64]] if targets is None else targets
    return (torch.zeros(1, 3, 64, 64, dtype=torch.uint8),
            torch.tensor(targets, dtype=torch.float32).reshape(-1, 6),
            ['sample.jpg'], [((64, 64), ((1., 1.), (0., 0.)))])


class TestYoloEvaluation(unittest.TestCase):
    def run_eval(self, model=None, batches=None, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return validation.test({'nc': 2, 'names': ['a', 'b']},
                                   model=model if model is not None else TinyDetector(),
                                   dataloader=batches if batches is not None else [batch()],
                                   imgsz=64, batch_size=1, plots=False, **kwargs)

    def test_contract_nonzero_metrics_and_absent_class(self):
        results, maps, times = self.run_eval()
        self.assertIsInstance(results, tuple)
        self.assertEqual(len(results), 7)
        self.assertTrue(all(isinstance(x, float) for x in results))
        np.testing.assert_allclose(results[:2], [1, 1])
        self.assertGreater(results[2], .98)
        self.assertGreater(results[3], .98)
        self.assertEqual(results[4:], (0., 0., 0.))
        self.assertIsInstance(maps, np.ndarray)
        self.assertEqual(maps.shape, (2,))
        self.assertEqual(maps[1], results[3])
        self.assertEqual(len(times), 6)
        self.assertTrue(all(x >= 0 for x in times[:3]))
        self.assertEqual(times[3:], (64, 64, 1))

    def test_no_predictions_no_targets_and_wrong_class(self):
        cases = [(TinyDetector([[32, 32, 20, 20, 0., .99, 0]]), [batch()]),
                 (TinyDetector(), [batch([])]),
                 (TinyDetector([[32, 32, 20, 20, 0., .99, 0]]), [batch([])]),
                 (TinyDetector([[32, 32, 20, 20, .99, 0, .99]]), [batch()])]
        for model, batches in cases:
            with self.subTest(rows=model.rows):
                results, maps, _ = self.run_eval(model, batches)
                np.testing.assert_array_equal(results[:4], np.zeros(4))
                self.assertTrue(np.isfinite(maps).all())

    def test_missed_label_counts_in_recall_and_false_positive_in_precision(self):
        results, _, _ = self.run_eval(batches=[batch(), batch([[0, 0, .1, .1, .05, .05]])])
        self.assertAlmostEqual(results[1], .5)
        self.assertLess(results[0], 1.)
        self.assertLess(results[3], .6)

    def test_single_class_merges_model_and_label_classes(self):
        results, maps, _ = self.run_eval(TinyDetector([[32, 32, 20, 20, .99, 0, .99]]),
                                        [batch([[0, 1, .5, .5, 20/64, 20/64]])], single_cls=True)
        self.assertGreater(results[3], .98)
        self.assertEqual(maps.shape, (1,))

    def test_matching_is_one_to_one_and_threshold_specific(self):
        labels = torch.tensor([[0., 0, 0, 10, 10]])
        detections = torch.tensor([[0., 0, 10, 10, .9, 0], [0., 0, 10, 10, .8, 0]])
        correct = validation.process_batch(detections, labels, torch.tensor([.5, .95]))
        self.assertEqual(correct.sum(0).tolist(), [1, 1])
        detections = torch.tensor([[0., 0, 10, 8, .9, 0]])
        self.assertEqual(validation.process_batch(detections, labels, torch.tensor([.5, .95])).tolist(),
                         [[True, False]])

    def test_training_validation_call_and_loss_contract(self):
        # Execute the actual call expression in train.py with lightweight bindings.
        tree = ast.parse((Path(__file__).resolve().parents[2] / 'train.py').read_text())
        call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute) and node.func.attr == 'test'
                    and any(kw.arg == 'compute_loss' for kw in node.keywords))
        from types import SimpleNamespace
        model = TinyDetector().train()
        targets = batch()[1].clone()
        calls = []

        def loss_fn(raw, labels):
            torch.testing.assert_close(labels, targets)
            calls.append(True)
            return torch.tensor(9.), torch.tensor([1., 2., 3., 6.])

        with tempfile.TemporaryDirectory() as directory:
            bindings = dict(test=validation, data_dict={'nc': 2, 'names': ['a', 'b']},
                            batch_size=1, imgsz_test=64, ema=SimpleNamespace(ema=model),
                            opt=SimpleNamespace(single_cls=False, v5_metric=False),
                            testloader=[batch(), batch()], save_dir=Path(directory), nc=2,
                            final_epoch=False, plots=False, wandb_logger=None,
                            compute_loss=loss_fn, is_coco=False)
            with contextlib.redirect_stdout(io.StringIO()):
                results, maps, times = eval(compile(ast.Expression(call), 'train.py', 'eval'), bindings)
        self.assertEqual(results[4:], (1., 2., 3.))
        self.assertEqual(len(calls), 2)
        self.assertTrue(model.training)
        self.assertEqual(model.anchor.dtype, torch.float32)
        self.assertTrue(model.anchor.requires_grad)
        self.assertEqual(len('%10.4g' * 7 % results), 70)
        self.assertGreater(fitness(np.array(results).reshape(1, -1))[0], .98)
        self.assertEqual(((1 - maps) ** 2).shape, (2,))

    def test_restore_model_after_loss_error(self):
        model = TinyDetector().train()
        def fail(*args):
            raise RuntimeError('loss failed')
        with self.assertRaisesRegex(RuntimeError, 'loss failed'):
            self.run_eval(model, compute_loss=fail)
        self.assertTrue(model.training)
        self.assertEqual(model.anchor.dtype, torch.float32)

    def test_empty_loader_is_explicit_error(self):
        with self.assertRaisesRegex(ValueError, 'no images'):
            self.run_eval(batches=[])

    def test_cli_entrypoint_loads_local_checkpoint_and_dataset(self):
        # Importable fixture avoids pickle references to __main__ in the subprocess.
        from tests.integration.test_yolo_evaluation import TinyDetector as FixtureDetector
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'images').mkdir()
            (root / 'labels').mkdir()
            Image.new('RGB', (64, 64)).save(root / 'images/sample.jpg')
            (root / 'labels/sample.txt').write_text('0 0.5 0.5 0.3125 0.3125\n')
            (root / 'data.yaml').write_text(f'path: {directory}\nval: images\nnc: 2\nnames: [a, b]\n')
            # Validation rectangular padding produces a 96 x 96 tensor with 16px pad.
            torch.save({'model': FixtureDetector([[48, 48, 20, 20, .99, .99, 0]])}, root / 'model.pt')
            run = subprocess.run([sys.executable, 'test.py', '--weights', str(root / 'model.pt'),
                                  '--data', str(root / 'data.yaml'), '--img-size', '64', '--batch-size', '1',
                                  '--device', 'cpu', '--workers', '0', '--no-plots', '--verbose',
                                  '--project', str(root), '--name', 'evaluation', '--save-json'],
                                 cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn('mAP@0.5=0.99500', run.stdout)
            self.assertTrue((root / 'evaluation/predictions.json').is_file())
