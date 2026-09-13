"""Run the full non-GPU path on a deterministic synthetic scan."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepsegregation.pipeline import run_pipeline
from deepsegregation.synthetic import generate_scene


def main() -> int:
    scan = generate_scene(seed=11)
    result = run_pipeline(scan.cloud, scan.cloud.labels, dbscan_eps=.05,
                          dbscan_min_samples=10, fit_threshold=.015)
    print({"points": len(result.processed_cloud.points),
           "instances": len(result.instances), "fits": len(result.fits),
           "reports": len(result.compliance)})
    return 0 if result.fits else 1


if __name__ == "__main__":
    raise SystemExit(main())
