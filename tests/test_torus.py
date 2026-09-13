import unittest

import numpy as np

from deepsegregation.torus import fit_torus_ransac, torus_residuals


def make_elbow(
    *,
    major_radius=0.75,
    minor_radius=0.08,
    bend_angle=np.pi / 2,
    count=1800,
    noise=0.002,
    seed=7,
):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-bend_angle / 2.0, bend_angle / 2.0, count)
    tube_angle = rng.uniform(0.0, 2.0 * np.pi, count)
    points = np.column_stack(
        (
            (major_radius + minor_radius * np.cos(tube_angle)) * np.cos(theta),
            (major_radius + minor_radius * np.cos(tube_angle)) * np.sin(theta),
            minor_radius * np.sin(tube_angle),
        )
    )
    return points + rng.normal(scale=noise, size=points.shape)


class TorusFitTests(unittest.TestCase):
    def test_recovers_synthetic_elbow_with_outliers(self):
        points = make_elbow()
        rng = np.random.default_rng(12)
        outliers = rng.uniform(-1.5, 1.5, size=(220, 3))
        points = np.vstack((points, outliers))

        result = fit_torus_ransac(
            points,
            distance_threshold=0.015,
            max_trials=1600,
            min_inliers=1200,
            random_state=3,
        )

        self.assertGreater(result.inlier_count, 1450)
        self.assertAlmostEqual(result.major_radius, 0.75, delta=0.025)
        self.assertAlmostEqual(result.minor_radius, 0.08, delta=0.015)
        self.assertLess(result.rmse, 0.01)

    def test_accepts_known_axis_and_reports_surface_residuals(self):
        points = make_elbow(major_radius=1.2, minor_radius=0.12, bend_angle=np.pi / 4)
        result = fit_torus_ransac(
            points,
            axis=np.array([0.0, 0.0, 1.0]),
            distance_threshold=0.012,
            max_trials=500,
            min_inliers=1000,
            random_state=1,
        )

        residuals = torus_residuals(
            points, result.center, result.axis, result.major_radius, result.minor_radius
        )
        self.assertTrue(np.allclose(residuals, result.residuals, atol=1e-10))
        self.assertAlmostEqual(result.major_radius, 1.2, delta=0.04)
        self.assertAlmostEqual(result.minor_radius, 0.12, delta=0.02)

    def test_recovers_synthetic_elbow_with_heavy_45_percent_outliers_without_axis(self):
        rng = np.random.default_rng(42)
        count = 300
        bend = np.pi / 2
        theta = rng.uniform(-bend / 2, bend / 2, count)
        phi = rng.uniform(0, 2 * np.pi, count)
        R, r = 0.5, 0.05
        x = (R + r * np.cos(phi)) * np.cos(theta)
        y = (R + r * np.cos(phi)) * np.sin(theta)
        z = r * np.sin(phi)
        inliers = np.column_stack((x, y, z))
        outliers = rng.uniform(-1.0, 1.0, (245, 3))  # ~45% outliers
        points = np.vstack((inliers, outliers))

        result = fit_torus_ransac(
            points,
            distance_threshold=0.015,
            max_trials=1200,
            random_state=42,
        )

        self.assertGreaterEqual(result.inlier_count, 270)
        self.assertAlmostEqual(result.major_radius, 0.5, delta=0.08)
        self.assertAlmostEqual(result.minor_radius, 0.05, delta=0.015)


if __name__ == "__main__":
    unittest.main()
