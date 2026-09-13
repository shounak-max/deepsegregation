# Roadmap implementation status

This repository now contains the complete CPU-side architecture and interfaces
described by `Project_Roadmap.md`. The checklist below is deliberately explicit
about the remaining external operations.

| Phase | Status | Evidence |
|---|---|---|
| 0. Bring-up | Ready to execute | `vendor/ResPointNet2`, `reference.py`, `scripts/check_reference.py` |
| 1. Data pipeline | Implemented locally | `synthetic.py`, `pointcloud.py`, `preprocessing.py`, `dataset.py` |
| 2. Segmentation | Integration-ready | `SegmentationConfig`, `boundary_class_balanced_loss`, `inference.py`, `model.py` |
| 3. Separation/cylinder | Implemented and tested | `clustering.py`, `cylinder.py` |
| 4. Elbow torus RANSAC | Implemented and tested | `torus.py`, synthetic 45°/90° tests with clutter |
| 5. Integration/report/UI boundary | Implemented locally | `pipeline.py`, `report.py`, `ui.py`, `cli.py` |
| 6. Evaluation/ablation | Utilities implemented | `metrics.py`, `experiments.py` |
| 7. Manuscript/release | Project scaffolding ready | reproducible manifests and roadmap documentation |

## External gates

The following cannot be honestly completed from this repository alone:

- Downloading PSNet5 and verifying its checksum/content.
- Compiling the upstream CUDA operators on the campus GPU.
- Training a ResPointNet++ checkpoint and reporting measured PSNet5 metrics.
- Labeling the 15 real scans and validating against SHREC/PipeNet Lab access.
- Producing publication claims from measured experiments.

The project PointMLP synthetic baseline can be trained on a compatible CUDA
host with `scripts/train_point_mlp.py`; its checkpoints must not be described
as ResPointNet++ checkpoints.

## Cluster checkpoint

A 5-epoch GPU smoke run completed on the authorized K80 cluster using the
project PointMLP synthetic baseline. The checkpoint is available locally at
`artifacts/point_mlp_k80/best.pt`; its SHA-256 is
`f8d256c9dbf9fb92a68599d9567baee9ada189ac5861008ebfe04269b3a4a21`.
The final validation snapshot was accuracy `0.7465`, mIoU `0.3527`.
Use `python scripts/cluster_status.py` for subsequent status checks.

The code exposes these as explicit inputs and commands rather than silently
substituting synthetic numbers. Run `scripts/smoke_test.py` for the local
end-to-end check, then run the upstream `install-conda.sh`, `init.sh`, and
training command from the vendored reference on the authorized GPU host.
