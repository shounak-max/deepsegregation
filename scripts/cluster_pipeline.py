"""CLI tool to manage the decoupled remote GPU execution pipeline.

Features:
- Start detached GPU training (user PC does NOT need to remain on)
- Inspect checkpoint evaluation metrics and target achievements
- Stop running remote jobs
- Fetch latest checkpoints, metrics, and logs to local artifacts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

try:
    import paramiko
except ImportError:
    paramiko = None


DEFAULT_HOST = "10.0.24.7"
DEFAULT_PORT = 2222
DEFAULT_USER = "saptarsi"
DEFAULT_PROJECT = "/home/saptarsi/deepsegregation"
DEFAULT_PYTHON = "/home/saptarsi/miniconda3/envs/deepseg_k80/bin/python"


def get_client(host=DEFAULT_HOST, port=DEFAULT_PORT, user=DEFAULT_USER, password=None) -> paramiko.SSHClient:
    if paramiko is None:
        sys.exit("Error: paramiko must be installed (pip install paramiko)")
    if password is None:
        password = os.environ.get("DEEPSEGREGATION_CLUSTER_PASSWORD")
    if password is None:
        import getpass
        password = getpass.getpass("Cluster password: ")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username=user, password=password, timeout=15)
    return client


def sync_project_files(client: paramiko.SSHClient, local_root: Path, remote_root: str):
    sftp = client.open_sftp()
    print(f"Syncing project files to {remote_root}...")
    
    # Ensure remote dirs exist
    client.exec_command(f"mkdir -p {remote_root}/src/deepsegregation {remote_root}/scripts {remote_root}/checkpoints {remote_root}/data/PSNet")

    def put_dir(local_dir: Path, remote_dir: str):
        for item in local_dir.iterdir():
            if item.is_file() and not item.name.endswith(('.pyc', '.git', '.tmp')):
                r_path = f"{remote_dir}/{item.name}"
                sftp.put(str(item), r_path)
            elif item.is_dir() and item.name not in ('__pycache__', '.git', 'artifacts', 'PSNet5'):
                r_dir = f"{remote_dir}/{item.name}"
                client.exec_command(f"mkdir -p {r_dir}")
                put_dir(item, r_dir)

    put_dir(local_root / "src" / "deepsegregation", f"{remote_root}/src/deepsegregation")
    sftp.put(str(local_root / "scripts" / "remote_train.py"), f"{remote_root}/scripts/remote_train.py")
    
    # Sync real dataset npz files if present locally
    for npz_name in ["real_train.npz", "real_val.npz"]:
        local_npz = local_root / "data" / "PSNet" / npz_name
        if local_npz.exists():
            remote_npz = f"{remote_root}/data/PSNet/{npz_name}"
            print(f"Syncing {npz_name} ({local_npz.stat().st_size / 1e6:.1f} MB)...")
            sftp.put(str(local_npz), remote_npz)

    sftp.close()
    print("Project files synchronized successfully.")


def cmd_start(args):
    client = get_client(args.host, args.port, args.user)
    local_root = Path(__file__).resolve().parents[1]
    sync_project_files(client, local_root, args.project)

    # Check if a training job is already active
    _, stdout, _ = client.exec_command(f"pgrep -af 'scripts/remote_train.py' || true")
    active_jobs = stdout.read().decode().strip()
    if active_jobs and not args.force:
        print("Warning: A training process is already running on the remote host:")
        print(active_jobs)
        print("Use --force to kill existing jobs before launching or run 'status' to monitor.")
        client.close()
        return 1

    if active_jobs and args.force:
        print("Stopping existing training job...")
        client.exec_command("pkill -f 'scripts/remote_train.py' || true")
        time.sleep(2)

    early_stop_flag = "--early-stop-on-targets" if args.early_stop_on_targets else ""
    model_flag = f"--model {args.model}"
    remap_flag = "--remap-3class" if args.remap_3class else ""
    cmd = (
        f"cd {args.project} && "
        f"nohup {DEFAULT_PYTHON} scripts/remote_train.py "
        f"--dataset {args.dataset} "
        f"--epochs {args.epochs} "
        f"--checkpoint-interval {args.checkpoint_interval} "
        f"--gpu-id {args.gpu_id} "
        f"--target-miou {args.target_miou} "
        f"--target-accuracy {args.target_accuracy} "
        f"--target-loss {args.target_loss} "
        f"{model_flag} "
        f"{remap_flag} "
        f"{early_stop_flag} "
        f"> train.log 2>&1 & echo $!"
    )
    _, stdout, stderr = client.exec_command(cmd)
    pid = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    client.close()

    if err:
        print(f"Error launching training: {err}")
        return 1

    print("\n" + "=" * 70)
    print(">> REMOTE TRAINING PIPELINE STARTED SUCCESSFULLY")
    print("=" * 70)
    print(f"Remote PID:             {pid}")
    print(f"GPU Allocated:          GPU {args.gpu_id} (Tesla K80)")
    print(f"Epochs:                 {args.epochs}")
    print(f"Checkpoint Interval:    Every {args.checkpoint_interval} epoch(s)")
    print(f"Targets Configured:     mIoU >= {args.target_miou} | Accuracy >= {args.target_accuracy} | Loss <= {args.target_loss}")
    print(f"Log Destination:        {args.project}/train.log")
    print("=" * 70)
    print("NOTE: The training pipeline is running completely detached via nohup.")
    print("You can safely close this terminal or shut down your local machine!")
    print("Whenever you return, run:")
    print("  python scripts/cluster_pipeline.py status")
    print("  python scripts/cluster_pipeline.py fetch")
    print("=" * 70)
    return 0


def cmd_status(args):
    client = get_client(args.host, args.port, args.user)
    
    # 1. GPU status
    _, stdout, _ = client.exec_command(
        f"nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu "
        f"--format=csv,noheader"
    )
    gpu_lines = stdout.read().decode().strip().splitlines()

    # 2. Process status
    _, stdout, _ = client.exec_command("pgrep -af 'scripts/remote_train.py' || true")
    proc_line = stdout.read().decode().strip()
    is_running = bool(proc_line)

    # 3. Read pipeline_status.json
    _, stdout, _ = client.exec_command(f"cat {args.project}/pipeline_status.json 2>/dev/null || true")
    status_content = stdout.read().decode().strip()

    # 4. Tail log
    _, stdout, _ = client.exec_command(f"tail -n 12 {args.project}/train.log 2>/dev/null || true")
    tail_log = stdout.read().decode().strip()

    client.close()

    print("\n" + "=" * 70)
    print("REMOTE GPU CLUSTER STATUS")
    print("=" * 70)
    print("GPU Hardware Status:")
    for line in gpu_lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            idx, name, mem_u, mem_t, util, temp = parts[:6]
            mark = "<- [TARGET GPU]" if idx == str(args.gpu_id) else ""
            print(f"  GPU {idx} ({name}): Util {util} | Mem {mem_u}/{mem_t} | Temp {temp} {mark}")
        else:
            print(f"  {line}")

    print("\nProcess Status:")
    if is_running:
        print(f"  RUNNING (PID info: {proc_line})")
    else:
        print("  INACTIVE (no active remote_train.py process)")

    if status_content:
        try:
            status = json.loads(status_content)
            print("\nPipeline Checkpoint State:")
            print(f"  Overall Status:       {status.get('status')}")
            print(f"  Device:               {status.get('device')} ({status.get('gpu_name')})")
            print(f"  Progress:             Epoch {status.get('current_epoch', 0)} / {status.get('epochs_total', 0)}")
            print(f"  Elapsed Time:         {status.get('elapsed_seconds', 0)}s")
            print(f"  Best Val Loss:        {status.get('best_val_loss')} (Epoch {status.get('best_epoch')})")

            tgt_eval = status.get("targets_evaluation")
            if tgt_eval:
                print("\nTarget Metrics Evaluation:")
                m_icon = "[OK]" if tgt_eval.get("miou_met") else "[PENDING]"
                a_icon = "[OK]" if tgt_eval.get("accuracy_met") else "[PENDING]"
                l_icon = "[OK]" if tgt_eval.get("loss_met") else "[PENDING]"
                print(f"  {m_icon} mIoU:     {tgt_eval.get('current_miou', 0):.4f} / Target: {tgt_eval.get('target_miou')}")
                print(f"  {a_icon} Accuracy: {tgt_eval.get('current_accuracy', 0):.4f} / Target: {tgt_eval.get('target_accuracy')}")
                print(f"  {l_icon} Val Loss: {tgt_eval.get('current_loss', 0):.4f} / Target: <= {tgt_eval.get('target_loss')}")
                all_met = status.get("all_targets_met", False)
                print(f"  Summary: {'*** ALL TARGETS MET! ***' if all_met else 'Targets in progress...'}")

            metrics = status.get("latest_metrics")
            if metrics and "per_class_iou" in metrics:
                ious = metrics["per_class_iou"]
                if len(ious) == 5:
                    classes = ["ibeam", "pipe", "pump", "rectangularbeam", "tank"]
                else:
                    classes = ["Background", "Straight Pipe", "Elbow"]
                class_str = ", ".join(f"{c}: {iou:.3f}" for c, iou in zip(classes, ious) if iou is not None)
                print(f"  Per-Class IoU:        {class_str}")
        except Exception as e:
            print(f"  (Could not parse pipeline_status.json: {e})")

    if tail_log:
        print("\nRecent Remote Logs (tail):")
        print("-" * 70)
        print(tail_log)
        print("-" * 70)
    print("=" * 70 + "\n")
    return 0


def cmd_stop(args):
    client = get_client(args.host, args.port, args.user)
    print("Stopping remote training process...")
    _, stdout, _ = client.exec_command("pkill -f 'scripts/remote_train.py' && echo 'Sent SIGTERM' || echo 'No process found'")
    msg = stdout.read().decode().strip()
    client.close()
    print(f"Result: {msg}")
    return 0


def cmd_fetch(args):
    client = get_client(args.host, args.port, args.user)
    sftp = client.open_sftp()
    local_dir = Path(args.dest_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    remote_files = [
        ("checkpoints/point_mlp_k80/best.pt", local_dir / "best.pt"),
        ("checkpoints/point_mlp_k80/eval_history.json", local_dir / "eval_history.json"),
        ("pipeline_status.json", local_dir / "pipeline_status.json"),
        ("train.log", local_dir / "train.log"),
    ]

    print(f"Fetching checkpoints and results to {local_dir}...")
    for remote_rel, local_path in remote_files:
        r_path = f"{args.project}/{remote_rel}"
        try:
            sftp.get(r_path, str(local_path))
            print(f"  Downloaded: {local_path.name} ({local_path.stat().st_size} bytes)")
        except IOError:
            print(f"  Not available yet: {remote_rel}")

    sftp.close()
    client.close()

    best_pt = local_dir / "best.pt"
    if best_pt.exists():
        sha = hashlib.sha256(best_pt.read_bytes()).hexdigest()
        print(f"\nSynced best checkpoint SHA-256: {sha}")
    print("Fetch complete.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Manage decoupled remote GPU training and checkpoint evaluation")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--project", default=DEFAULT_PROJECT)

    subparsers = parser.add_subparsers(dest="command", required=True)

    # start
    p_start = subparsers.add_parser("start", help="Start decoupled GPU training")
    p_start.add_argument("--dataset", choices=["real", "synthetic"], default="real", help="Dataset to train on (real PSNet5 or synthetic)")
    p_start.add_argument("--epochs", type=int, default=15)
    p_start.add_argument("--checkpoint-interval", type=int, default=1)
    p_start.add_argument("--gpu-id", type=int, default=0)
    p_start.add_argument("--target-miou", type=float, default=0.50)
    p_start.add_argument("--target-accuracy", type=float, default=0.75)
    p_start.add_argument("--target-loss", type=float, default=1.50)
    p_start.add_argument("--model", choices=["point_mlp", "pointnet2_ssg"], default="pointnet2_ssg")
    p_start.add_argument("--remap-3class", action="store_true", default=True, help="Remap to 3-class contract")
    p_start.add_argument("--early-stop-on-targets", action="store_true")
    p_start.add_argument("--force", action="store_true", help="Stop existing jobs before starting")

    # status
    p_status = subparsers.add_parser("status", help="Inspect training progress, checkpoints, and targets")
    p_status.add_argument("--gpu-id", type=int, default=0)

    # stop
    subparsers.add_parser("stop", help="Stop running remote training job")

    # fetch
    p_fetch = subparsers.add_parser("fetch", help="Download checkpoints and logs to local artifacts")
    p_fetch.add_argument("--dest-dir", default="artifacts/point_mlp_k80")

    args = parser.parse_args(argv)

    if args.command == "start":
        return cmd_start(args)
    elif args.command == "status":
        return cmd_status(args)
    elif args.command == "stop":
        return cmd_stop(args)
    elif args.command == "fetch":
        return cmd_fetch(args)


if __name__ == "__main__":
    raise SystemExit(main())
