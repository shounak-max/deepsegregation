import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from deepsegregation.clustering import dbscan, separate_instances
from deepsegregation.cylinder import fit_cylinder_ransac
from deepsegregation.dataset import iter_area_samples, remap_labels
from deepsegregation.metrics import segmentation_metrics
from deepsegregation.pointcloud import PointCloud, read_ply, write_ply
from deepsegregation.report import compliance, write_report
from deepsegregation.synthetic import generate_scene


class PipelineComponentTests(unittest.TestCase):
    def test_dbscan_separates_two_clusters_and_noise(self):
        first = np.zeros((40, 3)) + np.random.default_rng(2).normal(scale=.01, size=(40, 3))
        second = np.ones((40, 3)) + np.random.default_rng(3).normal(scale=.01, size=(40, 3))
        points = np.vstack((first, second, [[5, 5, 5]]))
        labels = dbscan(points, eps=.08, min_samples=5)
        self.assertEqual(set(labels[:80]), {0, 1})
        self.assertEqual(labels[-1], -1)

    def test_cylinder_fit_recovers_small_radius(self):
        rng = np.random.default_rng(4)
        x = rng.uniform(-.7, .7, 1000)
        theta = rng.uniform(0, 2*np.pi, 1000)
        points = np.column_stack((x, .03*np.cos(theta), .03*np.sin(theta)))
        fit = fit_cylinder_ransac(points, distance_threshold=.004, min_inliers=700, random_state=5)
        self.assertAlmostEqual(fit.radius, .03, delta=.004)
        self.assertGreaterEqual(fit.inlier_count, 700)
        self.assertGreaterEqual(fit.iterations, 1500)

    def test_ply_round_trip_and_area_adapter(self):
        cloud = PointCloud(np.array([[0., 1., 2.], [3., 4., 5.]]), labels=np.array([1, 2]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            area = root / "Area_1"
            write_ply(area / "scan.ply", cloud)
            loaded = read_ply(area / "scan.ply")
            self.assertTrue(np.allclose(loaded.points, cloud.points))
            sample = list(iter_area_samples(root))[0]
            self.assertEqual(sample.area, "Area_1")
            np.testing.assert_array_equal(remap_labels(sample.cloud.labels, {1: 0, 2: 2}), [0, 2])

    def test_report_and_metrics_are_serializable(self):
        result = compliance("p1", "straight", .101, .1, tolerance=.05)
        self.assertTrue(result.passed)
        metrics = segmentation_metrics(np.array([0, 1, 2, 2]), np.array([0, 1, 1, 2]))
        self.assertAlmostEqual(metrics["accuracy"], .75)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            write_report(path, [result])
            self.assertEqual(json.loads(path.read_text())["pipes"][0]["pipe_id"], "p1")

    def test_synthetic_scene_has_separable_semantic_instances(self):
        scan = generate_scene()
        instances = separate_instances(scan.cloud, scan.cloud.labels, eps=.05, min_samples=10, min_points=100)
        self.assertEqual(sorted(x.semantic_class for x in instances), [1, 2])

    def test_cylinder_fit_recovers_radius_with_heavy_outliers(self):
        rng = np.random.default_rng(42)
        theta = rng.uniform(0, 2*np.pi, 300)
        z = rng.uniform(-1, 1, 300)
        x = 0.05 * np.cos(theta)
        y = 0.05 * np.sin(theta)
        inliers = np.column_stack((x, y, z))
        outliers = rng.uniform(-0.5, 0.5, (245, 3))  # ~45% outliers
        points = np.vstack((inliers, outliers))

        fit = fit_cylinder_ransac(points, distance_threshold=0.005, min_inliers=100, random_state=42)
        self.assertAlmostEqual(fit.radius, 0.05, delta=0.005)
        self.assertGreaterEqual(fit.inlier_count, 280)
        self.assertGreater(fit.best_trial, 0)

    def test_area_samples_numeric_sort_order(self):
        cloud = PointCloud(np.array([[0., 1., 2.]]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ["Area_10", "Area_2", "Area_1"]:
                area = root / name
                area.mkdir()
                write_ply(area / "scan.ply", cloud)
            loaded_areas = [s.area for s in iter_area_samples(root)]
            self.assertEqual(loaded_areas, ["Area_1", "Area_2", "Area_10"])

    def test_voxel_downsample_preserves_colors(self):
        from deepsegregation.preprocessing import voxel_downsample
        points = np.array([[0.01, 0.01, 0.01], [0.02, 0.02, 0.02], [1.0, 1.0, 1.0]])
        colors = np.array([[255, 0, 0], [255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        cloud = PointCloud(points, colors=colors)
        downsampled = voxel_downsample(cloud, voxel_size=0.1)
        self.assertIsNotNone(downsampled.colors)
        self.assertEqual(len(downsampled.colors), len(downsampled.points))
        self.assertEqual(len(downsampled.points), 2)
        np.testing.assert_array_equal(downsampled.colors[0], [255, 0, 0])

    def test_inference_cpu_fallback(self):
        import torch
        import torch.nn as nn
        from deepsegregation.inference import predict_labels
        model = nn.Linear(3, 4)
        features = np.ones((5, 3), dtype=np.float32)
        preds = predict_labels(model, features, device="cuda")
        self.assertEqual(preds.shape, (5,))

    def test_estimate_normals_vectorized(self):
        from deepsegregation.preprocessing import estimate_normals
        rng = np.random.default_rng(123)
        # Generate points on a flat XY plane with z=0
        xy = rng.uniform(-1, 1, (300, 2))
        points = np.column_stack((xy, np.zeros(300)))
        cloud = PointCloud(points)
        computed = estimate_normals(cloud, neighbors=15)
        self.assertIsNotNone(computed.normals)
        self.assertEqual(computed.normals.shape, (300, 3))
        # Normals for flat plane z=0 should be approximately parallel to Z axis [0, 0, 1] or [0, 0, -1]
        z_abs = np.abs(computed.normals[:, 2])
        self.assertTrue(np.all(z_abs > 0.95))

    def test_cylinder_fit_45_degree_stable_axis(self):
        rng = np.random.default_rng(99)
        # Cylinder aligned along [1, 1, 0] / sqrt(2)
        t = rng.uniform(-1, 1, 500)
        theta = rng.uniform(0, 2*np.pi, 500)
        axis_dir = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
        u_dir = np.array([-1.0, 1.0, 0.0]) / np.sqrt(2)
        v_dir = np.array([0.0, 0.0, 1.0])
        r = 0.04
        points = (np.outer(t, axis_dir) +
                  np.outer(r * np.cos(theta), u_dir) +
                  np.outer(r * np.sin(theta), v_dir))
        fit = fit_cylinder_ransac(points, distance_threshold=0.005, random_state=42)
        self.assertAlmostEqual(fit.radius, 0.04, delta=0.005)
        # Axis should be deterministically oriented with positive dot product against global Y/X
        self.assertGreater(float(np.dot(fit.axis, [0, 1, 0])), 0.5)


if __name__ == "__main__":
    unittest.main()

