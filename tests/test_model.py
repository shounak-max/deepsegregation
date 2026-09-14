import unittest
import torch
import numpy as np

from deepsegregation.model import build_point_mlp, build_pointnet2_ssg


class ModelTests(unittest.TestCase):
    def test_point_mlp_forward(self):
        model = build_point_mlp(input_features=3, num_classes=3)
        x = torch.randn(50, 3)
        logits = model(x)
        self.assertEqual(logits.shape, (50, 3))

    def test_pointnet2_ssg_forward_and_backward(self):
        model = build_pointnet2_ssg(input_features=3, num_classes=3)
        model.train()
        # Test 2D shape (N, 3)
        x = torch.randn(300, 3, requires_grad=True)
        logits = model(x)
        self.assertEqual(logits.shape, (300, 3))

        loss = logits.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_pointnet2_ssg_batched(self):
        model = build_pointnet2_ssg(input_features=4, num_classes=3)
        model.eval()
        # Test 3D batched shape (B, N, 4)
        x = torch.randn(2, 200, 4)
        with torch.no_grad():
            logits = model(x)
        self.assertEqual(logits.shape, (2, 200, 3))


if __name__ == "__main__":
    unittest.main()
