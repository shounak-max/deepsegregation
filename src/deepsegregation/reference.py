"""Bring-up checks and command construction for the vendored ResPointNet++ repo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess


@dataclass(frozen=True)
class ReferenceRepoStatus:
    root: Path
    has_training_script: bool
    has_dataset_loader: bool
    has_custom_ops: bool
    has_checkpoint: bool

    @property
    def ready_for_training(self) -> bool:
        return self.has_training_script and self.has_dataset_loader and self.has_custom_ops


def inspect_reference(root: str | Path) -> ReferenceRepoStatus:
    root = Path(root)
    return ReferenceRepoStatus(
        root=root,
        has_training_script=(root / "function" / "train_psnet_dist.py").exists(),
        has_dataset_loader=(root / "datasets" / "PSNet5.py").exists(),
        has_custom_ops=(root / "ops").is_dir() and (root / "init.sh").exists(),
        has_checkpoint=any(root.glob("**/*.pth")) or any(root.glob("**/*.pt")),
    )


def training_command(reference_root: str | Path, *, data_root: str = "data",
                     config: str = "cfgs/psnet5/respointnet2_dp_fi_df_fc1_max.yaml",
                     num_gpus: int = 1, smoke: bool = False) -> list[str]:
    """Build the upstream training command without executing it."""
    if num_gpus < 1:
        raise ValueError("num_gpus must be >= 1")
    command = ["python", "-m", "torch.distributed.launch", "--master_port", "12346",
               "--nproc_per_node", str(num_gpus), "function/train_psnet_dist.py",
               "--dataset_name", "psnet5", "--data_root", data_root, "--cfg", config]
    if smoke:
        command.extend(["--epochs", "1"])
    return command


def run_bringup(reference_root: str | Path) -> ReferenceRepoStatus:
    """Validate the reference tree; CUDA compilation is deliberately explicit."""
    status = inspect_reference(reference_root)
    if not status.ready_for_training:
        raise RuntimeError(f"reference repo is incomplete: {status}")
    return status
