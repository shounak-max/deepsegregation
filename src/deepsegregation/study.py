"""Auditable data contracts for the publication three-class study.

This module intentionally refuses to infer elbow labels from PSNet5's five
semantic labels.  Human-reviewed labels are supplied in immutable block files
and validated before a training or test run can begin.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

import numpy as np


CLASS_NAMES = ("background", "straight_pipe", "elbow")
SPLIT_AREAS = {"train": {"Area_1", "Area_2"}, "val": {"Area_4"}, "test": {"Area_3"}}


class StudyValidationError(ValueError):
    """Raised when an annotation package cannot support a defensible result."""


@dataclass(frozen=True)
class StudyBlock:
    block_id: str
    split: str
    area: str
    source_path: str
    source_sha256: str
    point_file: str
    center: tuple[float, float, float]
    radius_m: float
    elbow_instance_ids: tuple[str, ...]
    primary_instance_id: str
    primary_annotator: str
    reviewer: str | None


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(payload).hexdigest()


def load_manifest(root: str | Path) -> tuple[dict, list[StudyBlock]]:
    root = Path(root)
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StudyValidationError(f"missing annotation manifest: {manifest_path}") from exc
    if manifest.get("schema_version") != 1 or not manifest.get("label_version"):
        raise StudyValidationError("manifest must declare schema_version=1 and label_version")
    blocks = []
    required = {"id", "split", "area", "source_path", "source_sha256", "point_file", "center",
                "radius_m", "elbow_instance_ids", "primary_instance_id", "primary_annotator"}
    for entry in manifest.get("blocks", []):
        missing = required - set(entry)
        if missing:
            raise StudyValidationError(f"block {entry.get('id', '<unknown>')} missing {sorted(missing)}")
        try:
            blocks.append(StudyBlock(
                block_id=str(entry["id"]), split=str(entry["split"]), area=str(entry["area"]),
                source_path=str(entry["source_path"]), source_sha256=str(entry["source_sha256"]),
                point_file=str(entry["point_file"]), center=tuple(float(v) for v in entry["center"]),
                radius_m=float(entry["radius_m"]),
                elbow_instance_ids=tuple(str(v) for v in entry["elbow_instance_ids"]),
                primary_instance_id=str(entry["primary_instance_id"]),
                primary_annotator=str(entry["primary_annotator"]), reviewer=entry.get("reviewer"),
            ))
        except (TypeError, ValueError) as exc:
            raise StudyValidationError(f"invalid block metadata: {entry.get('id', '<unknown>')}") from exc
    return manifest, blocks


def _load_labels(root: Path, block: StudyBlock) -> dict[str, np.ndarray]:
    path = root / block.point_file
    if not path.is_file():
        raise StudyValidationError(f"missing point file for {block.block_id}: {path}")
    with np.load(path) as values:
        required = {"points", "source_indices", "labels_primary", "labels_final"}
        missing = required - set(values.files)
        if missing:
            raise StudyValidationError(f"{block.block_id} missing point arrays {sorted(missing)}")
        output = {key: values[key].copy() for key in values.files}
    points = output["points"]
    size = len(points)
    if points.shape != (size, 3) or not np.isfinite(points).all():
        raise StudyValidationError(f"{block.block_id} has invalid points")
    if len(np.unique(output["source_indices"])) != size:
        raise StudyValidationError(f"{block.block_id} has duplicate source point indices")
    for key in ("labels_primary", "labels_final"):
        labels = output[key]
        if labels.shape != (size,) or not np.isin(labels, [0, 1, 2]).all():
            raise StudyValidationError(f"{block.block_id} has invalid {key}")
    if "labels_review" in output:
        review = output["labels_review"]
        if review.shape != (size,) or not np.isin(review, [0, 1, 2]).all():
            raise StudyValidationError(f"{block.block_id} has invalid labels_review")
    return output


def cohen_kappa(left: np.ndarray, right: np.ndarray, classes: int = 3) -> float:
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.size == 0:
        raise StudyValidationError("kappa requires equally sized non-empty label arrays")
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (left, right), 1)
    observed = np.trace(matrix) / matrix.sum()
    expected = (matrix.sum(axis=0) * matrix.sum(axis=1)).sum() / matrix.sum() ** 2
    return float((observed - expected) / (1.0 - expected)) if expected < 1.0 else 1.0


def validate_study(root: str | Path, raw_root: str | Path, *, min_train_elbows: int = 30,
                   min_val_elbows: int = 10, min_test_elbows: int = 20,
                   min_development_review_fraction: float = .2, min_kappa: float = .8,
                   splits: Iterable[str] | None = None) -> dict:
    """Validate annotations, provenance, quotas, split isolation, and review quality."""
    root, raw_root = Path(root), Path(raw_root)
    manifest, all_blocks = load_manifest(root)
    requested = set(splits) if splits is not None else set(SPLIT_AREAS)
    blocks = [block for block in all_blocks if block.split in requested]
    if not blocks:
        raise StudyValidationError("no annotation blocks for requested split")
    seen_ids, seen_points, seen_sources = set(), {}, {}
    elbow_ids = {split: set() for split in SPLIT_AREAS}
    review_primary, review_secondary = [], []
    reviewed = {split: 0 for split in SPLIT_AREAS}
    by_split = {split: 0 for split in SPLIT_AREAS}
    label_support = {split: np.zeros(3, dtype=np.int64) for split in SPLIT_AREAS}

    for block in blocks:
        if block.split not in SPLIT_AREAS or block.area not in SPLIT_AREAS[block.split]:
            raise StudyValidationError(f"{block.block_id} has invalid split/area assignment")
        if block.block_id in seen_ids or len(block.center) != 3 or block.radius_m != 2.0:
            raise StudyValidationError(f"duplicate block or invalid geometry: {block.block_id}")
        if not block.primary_instance_id or not block.primary_annotator:
            raise StudyValidationError(f"{block.block_id} lacks audit identity")
        seen_ids.add(block.block_id)
        by_split[block.split] += 1
        elbow_ids[block.split].update(block.elbow_instance_ids)
        source = (raw_root / block.source_path).resolve()
        if not source.is_file() or file_sha256(source) != block.source_sha256:
            raise StudyValidationError(f"source hash mismatch for {block.block_id}")
        source_key = str(source)
        previous_source_split = seen_sources.setdefault(source_key, block.split)
        if previous_source_split != block.split:
            raise StudyValidationError(f"source file leaks across splits: {block.source_path}")
        arrays = _load_labels(root, block)
        label_support[block.split] += np.bincount(arrays["labels_final"], minlength=3)
        for index in arrays["source_indices"].tolist():
            key = (source_key, int(index))
            previous = seen_points.setdefault(key, block.split)
            if previous != block.split:
                raise StudyValidationError(f"source point leaks across splits: {block.block_id}")
        if "labels_review" in arrays:
            if not block.reviewer or block.reviewer == block.primary_annotator:
                raise StudyValidationError(f"{block.block_id} has an invalid independent review")
            reviewed[block.split] += 1
            review_primary.append(arrays["labels_primary"])
            review_secondary.append(arrays["labels_review"])

    for split, minimum in {"train": min_train_elbows, "val": min_val_elbows, "test": min_test_elbows}.items():
        if split in requested and len(elbow_ids[split]) < minimum:
            raise StudyValidationError(f"{split} requires {minimum} elbow instances; found {len(elbow_ids[split])}")
        if split in requested and np.any(label_support[split] == 0):
            missing = [CLASS_NAMES[i] for i, count in enumerate(label_support[split]) if count == 0]
            raise StudyValidationError(f"{split} lacks point support for {missing}")
    if "test" in requested and reviewed["test"] != by_split["test"]:
        raise StudyValidationError("every test block requires independent review")
    development = by_split["train"] + by_split["val"]
    if development and (reviewed["train"] + reviewed["val"]) / development < min_development_review_fraction:
        raise StudyValidationError("fewer than 20% of development blocks have independent review")
    kappa = cohen_kappa(np.concatenate(review_primary), np.concatenate(review_secondary)) if review_primary else None
    if review_primary and kappa < min_kappa:
        raise StudyValidationError(f"annotation agreement kappa={kappa:.3f} is below {min_kappa:.3f}")
    return {
        "manifest_sha256": canonical_json_hash(manifest), "label_version": manifest["label_version"],
        "blocks": by_split, "elbow_instances": {key: len(value) for key, value in elbow_ids.items()},
        "label_support": {key: value.tolist() for key, value in label_support.items()},
        "reviewed_blocks": reviewed, "cohen_kappa": kappa,
    }


def sampled_block(root: str | Path, block: StudyBlock, *, points_per_block: int = 4096,
                  seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic, spatially coherent normalized block."""
    arrays = _load_labels(Path(root), block)
    count = len(arrays["points"])
    if count == 0:
        raise StudyValidationError(f"empty block: {block.block_id}")
    block_seed = int.from_bytes(sha256(f"{seed}:{block.block_id}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(block_seed)
    indices = rng.choice(count, points_per_block, replace=count < points_per_block)
    features = (arrays["points"][indices] - np.asarray(block.center)) / block.radius_m
    return features.astype(np.float32), arrays["labels_final"][indices].astype(np.int64)


def audit_raw_class_counts(raw_root: str | Path, areas: Iterable[str] = SPLIT_AREAS["train"] | SPLIT_AREAS["val"] | SPLIT_AREAS["test"]) -> dict:
    """Count raw PSNet annotation lines by class and area for result provenance."""
    raw_root = Path(raw_root)
    result = {}
    for area in sorted(areas):
        counts = {}
        for path in sorted((raw_root / area).glob("Room_*/Annotations/*.txt")):
            name = path.name.split("_")[0]
            with path.open("rb") as handle:
                counts[name] = counts.get(name, 0) + sum(1 for _ in handle)
        result[area] = counts
    return result
