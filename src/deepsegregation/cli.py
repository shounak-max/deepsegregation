"""Command-line entry point for a labelled PLY smoke/evaluation run."""

from __future__ import annotations

import argparse

from .pipeline import run_pipeline
from .pointcloud import read_ply
from .report import write_report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Segmented pipe point-cloud geometry pipeline")
    parser.add_argument("input", help="labelled ASCII PLY input")
    parser.add_argument("--report", default="report.json")
    parser.add_argument("--voxel-size", type=float)
    parser.add_argument("--dbscan-eps", type=float, default=.05)
    parser.add_argument("--fit-threshold", type=float, default=.01)
    args = parser.parse_args(argv)
    result = run_pipeline(read_ply(args.input), voxel_size=args.voxel_size,
                          dbscan_eps=args.dbscan_eps, fit_threshold=args.fit_threshold)
    write_report(args.report, result.compliance, {"instances": len(result.instances)})
    print(f"processed {len(result.processed_cloud.points)} points; "
          f"found {len(result.instances)} instances; wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
