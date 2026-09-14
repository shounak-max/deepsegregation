"""Tests for topological pipe-elbow fitting, directional DBSCAN, and baselines."""

import unittest
import numpy as np

from deepsegregation.cylinder import CylinderFit
from deepsegregation.clustering import dbscan, estimate_normals
from deepsegregation.topology import compute_line_intersection, fit_torus_topological
from deepsegregation.baselines import compare_elbow_fitting_methods


class TopologyAndBaselineTests(unittest.TestCase):

    def test_line_intersection_exact(self):
        # Line 1 along X at y=0, z=0
        p1 = np.array([-1.0, 0.0, 0.0])
        a1 = np.array([1.0, 0.0, 0.0])
        # Line 2 along Y at x=0, z=0
        p2 = np.array([0.0, -1.0, 0.0])
        a2 = np.array([0.0, 1.0, 0.0])

        midpoint, s, t = compute_line_intersection(p1, a1, p2, a2)
        np.testing.assert_allclose(midpoint, [0.0, 0.0, 0.0], atol=1e-6)

    def test_directional_dbscan_separates_touching_cylinders(self):
        # Create two touching parallel cylinders along Z
        # Cylinder 1 at x = 0, y = 0, radius = 0.05
        # Cylinder 2 at x = 0.10, y = 0, radius = 0.05 (touching at x = 0.05, y = 0)
        z = np.linspace(0, 0.5, 25)

        # Angles for cylinder 1
        theta1 = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        pts1 = []
        n1 = []
        for zi in z:
            for th in theta1:
                x = 0.05 * np.cos(th)
                y = 0.05 * np.sin(th)
                pts1.append([x, y, zi])
                n1.append([np.cos(th), np.sin(th), 0.0])

        # Angles for cylinder 2 (centered at x = 0.10)
        pts2 = []
        n2 = []
        for zi in z:
            for th in theta1:
                x = 0.10 + 0.05 * np.cos(th)
                y = 0.05 * np.sin(th)
                pts2.append([x, y, zi])
                n2.append([np.cos(th), np.sin(th), 0.0])

        all_pts = np.vstack([pts1, pts2])
        all_normals = np.vstack([n1, n2])

        # Standard Euclidean DBSCAN with eps = 0.04 merges them because touching points are at distance 0
        labels_euclidean = dbscan(all_pts, eps=0.04, min_samples=3)
        self.assertEqual(len(set(labels_euclidean[labels_euclidean >= 0])), 1, "Euclidean DBSCAN merges touching pipes into 1 cluster")

        # Verify that normal-consistent DBSCAN separates them into 2 clusters!
        labels_normal = dbscan(all_pts, eps=0.04, min_samples=3,
                               normals=all_normals, max_normal_angle_deg=50.0)

        clusters_normal = set(labels_normal[labels_normal >= 0])
        self.assertEqual(len(clusters_normal), 2, "Normal-consistent DBSCAN should separate touching cylinders into 2 clusters")

    def test_topological_torus_fitting(self):
        # Synthetic 90-degree bend:
        # Elbow center at (0, 0, 0), bend radius R = 0.30, minor radius r = 0.05
        # Connects (0.3, 0.0, 0.0) to (0.0, 0.3, 0.0) in the 1st quadrant.
        # Pipe A tangent at (0.3, 0.0, 0.0) runs along +Y (line x=0.3, z=0)
        # Pipe B tangent at (0.0, 0.3, 0.0) runs along -X (line y=0.3, z=0)
        # Tangent intersection point is (0.3, 0.3, 0.0)
        major_R = 0.30
        minor_r = 0.05

        cyl_a = CylinderFit(
            center=np.array([0.3, -0.2, 0.0]),
            axis=np.array([0.0, 1.0, 0.0]),
            radius=minor_r,
            inlier_mask=np.ones(10, dtype=bool),
            residuals=np.zeros(10),
            iterations=1,
        )
        cyl_b = CylinderFit(
            center=np.array([-0.2, 0.3, 0.0]),
            axis=np.array([1.0, 0.0, 0.0]),
            radius=minor_r,
            inlier_mask=np.ones(10, dtype=bool),
            residuals=np.zeros(10),
            iterations=1,
        )

        phi = np.linspace(0, np.pi / 2, 25)
        theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        elbow_pts = []
        for p in phi:
            for t in theta:
                x = (major_R + minor_r * np.cos(t)) * np.cos(p)
                y = (major_R + minor_r * np.cos(t)) * np.sin(p)
                z = minor_r * np.sin(t)
                elbow_pts.append([x, y, z])
        elbow_pts = np.array(elbow_pts)

        # Fit topological torus
        fit = fit_torus_topological(elbow_pts, cyl_a, cyl_b, distance_threshold=0.01)

        self.assertTrue(fit.c1_continuity_verified)
        self.assertAlmostEqual(fit.major_radius, major_R, delta=0.01)
        self.assertAlmostEqual(fit.minor_radius, minor_r, delta=0.005)
        self.assertAlmostEqual(fit.bend_angle_deg, 90.0, delta=1.0)
        self.assertGreater(fit.inlier_count / len(elbow_pts), 0.95)

    def test_compare_elbow_fitting_methods(self):
        major_R = 0.30
        minor_r = 0.05
        cyl_a = CylinderFit(
            center=np.array([0.3, -0.2, 0.0]),
            axis=np.array([0.0, 1.0, 0.0]),
            radius=minor_r,
            inlier_mask=np.ones(10, dtype=bool),
            residuals=np.zeros(10),
            iterations=1,
        )
        cyl_b = CylinderFit(
            center=np.array([-0.2, 0.3, 0.0]),
            axis=np.array([1.0, 0.0, 0.0]),
            radius=minor_r,
            inlier_mask=np.ones(10, dtype=bool),
            residuals=np.zeros(10),
            iterations=1,
        )

        phi = np.linspace(0, np.pi / 2, 20)
        theta = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        elbow_pts = []
        for p in phi:
            for t in theta:
                x = (major_R + minor_r * np.cos(t)) * np.cos(p)
                y = (major_R + minor_r * np.cos(t)) * np.sin(p)
                z = minor_r * np.sin(t)
                elbow_pts.append([x, y, z])
        elbow_pts = np.array(elbow_pts)

        res_u, res_t = compare_elbow_fitting_methods(
            elbow_pts, cyl_a, cyl_b,
            ground_truth_major_r=major_R,
            ground_truth_minor_r=minor_r,
        )

        self.assertTrue(res_t.success)
        self.assertLess(res_t.mean_radius_error_pct, 5.0)
        self.assertLess(res_t.bend_radius_error_pct, 5.0)
        self.assertEqual(res_t.c1_tangent_error_deg, 0.0)


if __name__ == "__main__":
    unittest.main()
