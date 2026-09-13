"""Remote GPU training and checkpoint evaluation runner with target setting.

Designed to run decoupled on the GPU cluster independently of the local client.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from deepsegregation.metrics import segmentation_metrics
from deepsegregation.model import build_point_mlp
from deepsegregation.segmentation import boundary_class_balanced_loss
from deepsegregation.synthetic import generate_scene


_RUNNING = True


def _handle_signal(signum, frame):
    global _RUNNING
    print(f"\nReceived signal {signum}, initiating graceful shutdown...", flush=True)
    _RUNNING = False


def update_status(status_file: Path, data: dict):
    tmp_file = status_file.with_suffix(".tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp_file.replace(status_file)


def evaluate(model, val_x, val_y, num_classes=3):
    import torch
    model.eval()
    with torch.no_grad():
        val_logits = model(val_x)
        val_loss = boundary_class_balanced_loss(val_logits, val_y, beta=0.999, boundary_weight=2.0)
        predictions = val_logits.argmax(dim=1).cpu().numpy()
    metrics = segmentation_metrics(val_y.cpu().numpy(), predictions, num_classes)
    return float(val_loss), metrics, predictions


def check_targets(metrics: dict, val_loss: float, target_miou: float | None,
                  target_accuracy: float | None, target_loss: float | None) -> dict:
    miou_met = (metrics["mIoU"] >= target_miou) if target_miou is not None else True
    acc_met = (metrics["accuracy"] >= target_accuracy) if target_accuracy is not None else True
    loss_met = (val_loss <= target_loss) if target_loss is not None else True
    all_met = miou_met and acc_met and loss_met
    return {
        "target_miou": target_miou,
        "current_miou": metrics["mIoU"],
        "miou_met": miou_met,
        "target_accuracy": target_accuracy,
        "current_accuracy": metrics["accuracy"],
        "accuracy_met": acc_met,
        "target_loss": target_loss,
        "current_loss": val_loss,
        "loss_met": loss_met,
        "all_targets_met": all_met,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Remote GPU training with checkpoint evaluation and target setting")
    parser.add_argument("--dataset", choices=["synthetic", "real"], default="real", help="Dataset to train on (real PSNet5 or synthetic)")
    parser.add_argument("--real-train-path", default="data/PSNet/real_train.npz", help="Path to real_train.npz")
    parser.add_argument("--batch-size", type=int, default=16384, help="Points per training batch for real data")
    parser.add_argument("--steps-per-epoch", type=int, default=25, help="Gradient steps per epoch for real data")
    parser.add_argument("--epochs", type=int, default=20, help="Total training epochs")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--checkpoint-dir", default="checkpoints/point_mlp_k80", help="Directory to store checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=1, help="Evaluate and save checkpoint every N epochs")
    parser.add_argument("--gpu-id", type=int, default=1, help="CUDA GPU device index to use (e.g. 1)")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:N, or cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--target-miou", type=float, default=0.50, help="Target mIoU threshold")
    parser.add_argument("--target-accuracy", type=float, default=0.75, help="Target accuracy threshold")
    parser.add_argument("--target-loss", type=float, default=1.50, help="Target validation loss threshold")
    parser.add_argument("--early-stop-on-targets", action="store_true", help="Stop once all targets are met")
    parser.add_argument("--status-file", default="pipeline_status.json", help="Status tracking file")
    args = parser.parse_args(argv)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    import torch

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    status_file = Path(args.status_file)
    history_file = checkpoint_dir / "eval_history.json"

    # Select device
    if args.device == "auto":
        if torch.cuda.is_available():
            device_str = f"cuda:{args.gpu_id}" if args.gpu_id < torch.cuda.device_count() else "cuda:0"
        else:
            device_str = "cpu"
    else:
        device_str = args.device

    device = torch.device(device_str)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print("=" * 70, flush=True)
    print(f"DeepSegregation Remote Training Pipeline", flush=True)
    print(f"Device: {device} ({gpu_name})", flush=True)
    print(f"Epochs: {args.epochs} | Checkpoint interval: {args.checkpoint_interval}", flush=True)
    print(f"Targets: mIoU >= {args.target_miou} | Accuracy >= {args.target_accuracy} | Loss <= {args.target_loss}", flush=True)
    print(f"Process PID: {os.getpid()}", flush=True)
    print("=" * 70, flush=True)

    status_data = {
        "status": "INITIALIZING",
        "pid": os.getpid(),
        "start_time": time.time(),
        "device": str(device),
        "gpu_name": gpu_name,
        "epochs_total": args.epochs,
        "current_epoch": 0,
        "targets": {
            "target_miou": args.target_miou,
            "target_accuracy": args.target_accuracy,
            "target_loss": args.target_loss,
        },
        "best_val_loss": None,
        "best_epoch": None,
        "all_targets_met": False,
        "latest_metrics": None,
    }
    update_status(status_file, status_data)

    # Dataset setup
    if args.dataset == "real":
        print(f"Loading Real PSNet5 dataset from {args.real_train_path} and {args.real_val_path}...", flush=True)
        train_data = np.load(args.real_train_path)
        val_data = np.load(args.real_val_path)
        all_train_x = train_data["points"]
        all_train_y = train_data["labels"]
        all_val_x = val_data["points"]
        all_val_y = val_data["labels"]
        class_names = list(train_data["class_names"])
        num_classes = len(class_names)
        print(f"Real Train Points: {len(all_train_x):,d} | Real Val Points: {len(all_val_x):,d} | Classes: {class_names}", flush=True)

        # Pre-select validation sample (30,000 points) for quick checkpoint evaluation
        val_indices = np.random.choice(len(all_val_x), min(30000, len(all_val_x)), replace=False)
        val_x = torch.from_numpy(all_val_x[val_indices]).to(device)
        val_y = torch.from_numpy(all_val_y[val_indices]).to(device)
    else:
        num_classes = 3
        class_names = ["background", "straight", "elbow"]
        val_scene = generate_scene(args.seed + 1000)
        val_x = torch.from_numpy(val_scene.cloud.points.astype(np.float32)).to(device)
        val_y = torch.from_numpy(val_scene.cloud.labels.astype(np.int64)).to(device)

    model = build_point_mlp(input_features=3, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    best_epoch = None
    eval_history = []
    status_data["status"] = "RUNNING"
    status_data["dataset_type"] = f"real_psnet5 ({num_classes} classes)" if args.dataset == "real" else "synthetic"

    start_time = time.time()
    steps_per_epoch = args.steps_per_epoch if args.dataset == "real" else 1

    for epoch in range(1, args.epochs + 1):
        if not _RUNNING:
            status_data["status"] = "INTERRUPTED"
            break

        epoch_start = time.time()
        model.train()
        epoch_losses = []
        for _ in range(steps_per_epoch):
            if args.dataset == "real":
                batch_idx = np.random.choice(len(all_train_x), min(args.batch_size, len(all_train_x)), replace=False)
                train_x = torch.from_numpy(all_train_x[batch_idx]).to(device)
                train_y = torch.from_numpy(all_train_y[batch_idx]).to(device)
            else:
                train_scene = generate_scene(args.seed + epoch)
                train_x = torch.from_numpy(train_scene.cloud.points.astype(np.float32)).to(device)
                train_y = torch.from_numpy(train_scene.cloud.labels.astype(np.int64)).to(device)

            optimizer.zero_grad()
            logits = model(train_x)
            loss = boundary_class_balanced_loss(logits, train_y, beta=0.999, boundary_weight=2.0)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss_val = float(np.mean(epoch_losses))

        is_checkpoint = (epoch % args.checkpoint_interval == 0) or (epoch == args.epochs)
        if is_checkpoint:
            val_loss, metrics, _ = evaluate(model, val_x, val_y, num_classes=num_classes)
            target_status = check_targets(
                metrics, val_loss, args.target_miou, args.target_accuracy, args.target_loss
            )

            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
                best_epoch = epoch

            chk_info = {
                "epoch": epoch,
                "train_loss": train_loss_val,
                "val_loss": val_loss,
                "metrics": metrics,
                "targets_evaluation": target_status,
                "is_best": is_best,
                "duration_seconds": round(time.time() - epoch_start, 3),
            }
            eval_history.append(chk_info)

            # Save epoch checkpoint
            checkpoint_payload = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "metrics": metrics,
                "val_loss": val_loss,
                "train_loss": train_loss_val,
                "targets_evaluation": target_status,
                "model_name": "project_point_mlp_synthetic_baseline",
            }
            torch.save(checkpoint_payload, checkpoint_dir / f"epoch_{epoch:04d}.pt")
            if is_best:
                torch.save(checkpoint_payload, checkpoint_dir / "best.pt")

            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(eval_history, f, indent=2)

            target_badge = "[TARGETS ACHIEVED]" if target_status["all_targets_met"] else "[IN PROGRESS]"
            print(
                f"[Epoch {epoch:03d}/{args.epochs:03d}] {target_badge} "
                f"Train Loss: {train_loss_val:.4f} | Val Loss: {val_loss:.4f} | "
                f"mIoU: {metrics['mIoU']:.4f} (tgt: {args.target_miou}) | "
                f"Acc: {metrics['accuracy']:.4f} (tgt: {args.target_accuracy}) | "
                f"Best Val Loss: {best_loss:.4f} (Ep {best_epoch})",
                flush=True,
            )

            status_data.update({
                "current_epoch": epoch,
                "train_loss": train_loss_val,
                "val_loss": val_loss,
                "latest_metrics": metrics,
                "best_val_loss": best_loss,
                "best_epoch": best_epoch,
                "targets_evaluation": target_status,
                "all_targets_met": target_status["all_targets_met"],
                "last_updated": time.time(),
                "elapsed_seconds": round(time.time() - start_time, 2),
            })
            update_status(status_file, status_data)

            if args.early_stop_on_targets and target_status["all_targets_met"]:
                print(f"All target criteria reached at epoch {epoch}. Stopping early as requested.", flush=True)
                status_data["status"] = "TARGETS_ACHIEVED"
                break
        else:
            print(f"[Epoch {epoch:03d}/{args.epochs:03d}] Train Loss: {train_loss_val:.4f}", flush=True)
            status_data.update({
                "current_epoch": epoch,
                "train_loss": train_loss_val,
                "last_updated": time.time(),
                "elapsed_seconds": round(time.time() - start_time, 2),
            })
            update_status(status_file, status_data)

    if status_data["status"] == "RUNNING":
        status_data["status"] = "COMPLETED"

    status_data["completed_at"] = time.time()
    update_status(status_file, status_data)
    print("=" * 70, flush=True)
    print(f"Pipeline finished with status: {status_data['status']}", flush=True)
    print(f"Best checkpoint at epoch {best_epoch} with val_loss={best_loss:.4f}", flush=True)
    print("=" * 70, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
