"""Checkpointed synthetic 3-class GPU smoke trainer.

This deliberately trains the project-owned baseline model on generated data;
the PSNet5/ResPointNet++ training command remains in ``train_reference.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from deepsegregation.metrics import segmentation_metrics
from deepsegregation.model import build_point_mlp, build_pointnet2_ssg
from deepsegregation.segmentation import boundary_class_balanced_loss
from deepsegregation.synthetic import generate_elbow, generate_scene


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--checkpoint-dir", default="checkpoints/point_mlp")
    parser.add_argument("--model", choices=["point_mlp", "pointnet2_ssg"], default="pointnet2_ssg")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.epochs < 1:
        raise SystemExit("--epochs must be positive")
    import torch
    torch.manual_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if args.model == "pointnet2_ssg":
        model = build_pointnet2_ssg(input_features=3, num_classes=3).to(device)
    else:
        model = build_point_mlp(input_features=3, num_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    output = Path(args.checkpoint_dir)
    output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    validation = generate_scene(args.seed + 100)
    val_x = torch.from_numpy(validation.cloud.points.astype(np.float32)).to(device)
    val_y = torch.from_numpy(validation.cloud.labels).to(device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        scan = generate_scene(args.seed + epoch)
        x = torch.from_numpy(scan.cloud.points.astype(np.float32)).to(device)
        y = torch.from_numpy(scan.cloud.labels).to(device)
        logits = model(x)
        loss = boundary_class_balanced_loss(logits, y, beta=.999, boundary_weight=2.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = boundary_class_balanced_loss(val_logits, val_y, beta=.999, boundary_weight=2.0)
            predictions = val_logits.argmax(dim=1).cpu().numpy()
        metrics = segmentation_metrics(validation.cloud.labels, predictions, 3)
        checkpoint = {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "val_loss": float(val_loss), "metrics": metrics,
            "model_name": args.model,
        }
        torch.save(checkpoint, output / f"epoch_{epoch:04d}.pt")
        if float(val_loss) < best:
            best = float(val_loss)
            torch.save(checkpoint, output / "best.pt")
        print(f"epoch={epoch} train_loss={float(loss):.5f} val_loss={float(val_loss):.5f} "
              f"mIoU={metrics['mIoU']:.4f} accuracy={metrics['accuracy']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
