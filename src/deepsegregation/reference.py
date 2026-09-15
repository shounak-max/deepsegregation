"""Bring-up checks and command construction for the vendored ResPointNet++ repo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import Counter


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


@dataclass(frozen=True)
class PSNet5LayoutStatus:
    root: Path
    areas: tuple[str, ...]
    missing_areas: tuple[str, ...]
    annotation_files: int
    class_file_counts: tuple[tuple[str, int], ...] = ()
    area_file_counts: tuple[tuple[str, int], ...] = ()
    total_annotation_bytes: int = 0
    unexpected_class_prefixes: tuple[str, ...] = ()

    @property
    def ready_for_preprocessing(self) -> bool:
        return (
            not self.missing_areas
            and self.annotation_files > 0
            and not self.unexpected_class_prefixes
            and all(count > 0 for _, count in self.area_file_counts)
        )


def inspect_reference(root: str | Path) -> ReferenceRepoStatus:
    root = Path(root)
    return ReferenceRepoStatus(
        root=root,
        has_training_script=(root / "function" / "train_psnet_dist.py").exists(),
        has_dataset_loader=(root / "datasets" / "PSNet5.py").exists(),
        has_custom_ops=(root / "ops").is_dir() and (root / "init.sh").exists(),
        has_checkpoint=any(root.glob("**/*.pth")) or any(root.glob("**/*.pt")),
    )


def inspect_psnet5_layout(root: str | Path) -> PSNet5LayoutStatus:
    """Check the raw PSNet5 area layout without loading the dataset into memory."""
    root = Path(root)
    expected = ("Area_1", "Area_2", "Area_3", "Area_4")
    areas = tuple(area for area in expected if (root / area).is_dir())
    missing = tuple(area for area in expected if area not in areas)
    class_counts: Counter[str] = Counter()
    area_counts: Counter[str] = Counter()
    total_bytes = 0
    for area in areas:
        for path in (root / area).glob("Room_*/Annotations/*.txt"):
            if path.is_file():
                class_counts[path.name.split("_", 1)[0].lower()] += 1
                area_counts[area] += 1
                total_bytes += path.stat().st_size
    expected_classes = {"pipe", "pump", "tank", "ibeam", "rectangularbeam", "rbeam"}
    unexpected = tuple(sorted(set(class_counts) - expected_classes))
    return PSNet5LayoutStatus(
        root=root,
        areas=areas,
        missing_areas=missing,
        annotation_files=sum(class_counts.values()),
        class_file_counts=tuple(sorted(class_counts.items())),
        area_file_counts=tuple((area, area_counts[area]) for area in areas),
        total_annotation_bytes=total_bytes,
        unexpected_class_prefixes=unexpected,
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
