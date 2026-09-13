"""Print Phase-0 readiness for the vendored upstream reference."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deepsegregation.reference import inspect_reference


if __name__ == "__main__":
    status = inspect_reference(Path(__file__).resolve().parents[1] / "vendor" / "ResPointNet2")
    print(status)
    raise SystemExit(0 if status.ready_for_training else 1)
