"""Poll the authorized training host without persisting credentials."""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="10.0.24.7")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--user", default="saptarsi")
    parser.add_argument("--project", default="/home/saptarsi/deepsegregation")
    args = parser.parse_args(argv)
    try:
        import paramiko
    except ImportError as exc:
        raise SystemExit("install paramiko to use cluster_status.py") from exc
    password = os.environ.get("DEEPSEGREGATION_CLUSTER_PASSWORD")
    if password is None:
        password = getpass.getpass("Cluster password: ")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(args.host, port=args.port, username=args.user, password=password, timeout=15)
    command = (
        f"nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu "
        f"--format=csv,noheader; echo JOBS; pgrep -af 'scripts/remote_train.py' || true; "
        f"echo LOG; tail -20 {args.project}/train.log 2>/dev/null || true; "
        f"echo CHECKPOINTS; ls -lh {args.project}/checkpoints/respointnet2_psnet5 2>/dev/null || true"
    )
    _, output, errors = client.exec_command(command)
    print(output.read().decode(errors="replace"), end="")
    print(errors.read().decode(errors="replace"), end="")
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
