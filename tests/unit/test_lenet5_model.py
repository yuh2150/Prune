"""Reference LeNet-5 architecture contract."""

import unittest

import torch

from prune_framework.models import LeNet5


class TestLeNet5(unittest.TestCase):
    def test_shape_and_eval_determinism(self):
        model = LeNet5().eval()
        images = torch.randn(3, 1, 28, 28)
        with torch.no_grad():
            first, second = model(images), model(images)
        self.assertEqual(tuple(first.shape), (3, 10))
        self.assertTrue(torch.equal(first, second))

    def test_supports_emnist_balanced_class_count(self):
        model = LeNet5(num_classes=47)
        self.assertEqual(tuple(model(torch.randn(2, 1, 28, 28)).shape), (2, 47))


if __name__ == "__main__":
    unittest.main()
