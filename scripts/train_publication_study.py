"""Train the preregistered PointNet++ study without opening Area_3 test labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from deepsegregation.metrics import segmentation_metrics
from deepsegregation.model import build_pointnet2_ssg
from deepsegregation.segmentation import boundary_class_balanced_loss, boundary_mask_from_labels
from deepsegregation.study import (audit_raw_class_counts, canonical_json_hash, load_manifest,
                                   sampled_block, validate_study)


def _load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class _Blocks:
    def __init__(self, root: Path, blocks, points_per_block: int, seed: int, augment: bool):
        self.root, self.blocks = root, list(blocks)
        self.points_per_block, self.seed, self.augment, self.epoch = points_per_block, seed, augment, 0

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, index):
        import torch
        x, y = sampled_block(self.root, self.blocks[index], points_per_block=self.points_per_block,
                             seed=self.seed + self.epoch * 1009)
        if self.augment:
            rng = np.random.default_rng(self.seed * 1000003 + self.epoch * 1009 + index)
            angle = rng.uniform(0, 2 * np.pi)
            rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]], dtype=np.float32)
            x = x @ rotation.T + rng.normal(0, .01, x.shape).astype(np.float32)
        boundary = boundary_mask_from_labels(x, y).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(boundary)


def _evaluate(model, loader, device, num_classes: int = 3) -> dict:
    import torch
    all_targets, all_predictions = [], []
    model.eval()
    with torch.no_grad():
        for x, y, _ in loader:
            logits = model(x.to(device))
            all_targets.append(y.numpy().reshape(-1))
            all_predictions.append(logits.argmax(dim=-1).cpu().numpy().reshape(-1))
    return segmentation_metrics(np.concatenate(all_targets), np.concatenate(all_predictions), num_classes)


def _train_seed(annotation_root: Path, train_blocks, val_blocks, config: dict, seed: int,
                output: Path, device: str, manifest_hash: str) -> dict:
    import torch
    from torch.utils.data import DataLoader
    _set_seed(seed)
    destination = output / f"seed_{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    train_set = _Blocks(annotation_root, train_blocks, config["points_per_block"], seed, True)
    val_set = _Blocks(annotation_root, val_blocks, config["points_per_block"], seed, False)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, generator=generator)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False)
    model = build_pointnet2_ssg(3, 3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=config["cosine_eta_min"])
    best, stale, history = -1.0, 0, []
    for epoch in range(1, config["epochs"] + 1):
        train_set.epoch = epoch
        model.train()
        losses = []
        for x, y, boundary in train_loader:
            x, y, boundary = x.to(device), y.to(device), boundary.to(device)
            logits = model(x)
            loss = boundary_class_balanced_loss(logits.reshape(-1, 3), y.reshape(-1), boundary.reshape(-1),
                                                beta=config["boundary_cb_beta"], boundary_weight=config["boundary_weight"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        scheduler.step()
        metrics = _evaluate(model, val_loader, device)
        item = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": metrics}
        history.append(item)
        if metrics["mIoU"] > best:
            best, stale = metrics["mIoU"], 0
            torch.save({"model_state": model.state_dict(), "model_name": "pointnet2_ssg", "seed": seed,
                        "epoch": epoch, "validation": metrics, "config_sha256": canonical_json_hash(config),
                        "manifest_sha256": manifest_hash}, destination / "best.pt")
        else:
            stale += 1
            if stale >= config["early_stopping_patience"]:
                break
    (destination / "development_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"seed": seed, "best_validation_miou": best, "epochs_completed": len(history),
            "checkpoint": str(destination / "best.pt")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, help="Versioned annotation package root")
    parser.add_argument("--raw-root", default="data/PSNet/PSNet5")
    parser.add_argument("--config", default="configs/pipe3_publication_v1.json")
    parser.add_argument("--output", default="artifacts/pipe3_publication_v1")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    annotation_root, output, config = Path(args.annotations), Path(args.output), _load_config(args.config)
    audit = validate_study(annotation_root, args.raw_root, splits={"train", "val"})
    manifest, blocks = load_manifest(annotation_root)
    train_blocks = [block for block in blocks if block.split == "train"]
    val_blocks = [block for block in blocks if block.split == "val"]
    import torch
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"config": config, "config_sha256": canonical_json_hash(config), "development_audit": audit,
                "raw_class_counts": audit_raw_class_counts(args.raw_root, {"Area_1", "Area_2", "Area_4"}),
                "manifest_sha256": canonical_json_hash(manifest), "device": device,
                "python": platform.python_version(), "test_area_status": "LOCKED_NOT_LOADED"}
    (output / "development_freeze.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    results = [_train_seed(annotation_root, train_blocks, val_blocks, config, seed, output, device,
                           metadata["manifest_sha256"]) for seed in config["seeds"]]
    (output / "development_summary.json").write_text(json.dumps({"runs": results, **metadata}, indent=2), encoding="utf-8")
    print(json.dumps({"runs": results, "test_area_status": metadata["test_area_status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
