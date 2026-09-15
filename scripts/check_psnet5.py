"""Validate the staged raw PSNet5 area layout before preprocessing."""

from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deepsegregation.reference import inspect_psnet5_layout


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check raw PSNet5 staging")
    parser.add_argument("--data-root", default="data/PSNet/PSNet5")
    parser.add_argument("--manifest", help="Write the metadata manifest to this JSON path")
    args = parser.parse_args(argv)
    status = inspect_psnet5_layout(args.data_root)
    print(status)
    if args.manifest:
        manifest = {
            "root": str(status.root),
            "areas": list(status.areas),
            "missing_areas": list(status.missing_areas),
            "annotation_files": status.annotation_files,
            "class_file_counts": dict(status.class_file_counts),
            "area_file_counts": dict(status.area_file_counts),
            "total_annotation_bytes": status.total_annotation_bytes,
            "unexpected_class_prefixes": list(status.unexpected_class_prefixes),
            "ready_for_preprocessing": status.ready_for_preprocessing,
        }
        output = Path(args.manifest)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"wrote manifest: {output}")
    return 0 if status.ready_for_preprocessing else 1


if __name__ == "__main__":
    raise SystemExit(main())
