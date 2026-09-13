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


if __name__ == "__main__":
    unittest.main()
