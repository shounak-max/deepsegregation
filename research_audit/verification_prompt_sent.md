You are the senior reviewer for DeepSegregation. I re-ran local verification today (2026-09-14).

## Local execution (just ran on my machine)
- unittest: PASS (24 tests)
- smoke_test.py: PASS — {'points': 2300, 'instances': 2, 'fits': 2, 'reports': 2}
- run_academic_ablation.py: PASS

Test tail:
```
........................
----------------------------------------------------------------------
Ran 24 tests in 5.448s

OK

```

Ablation tail:
```
 Torus RANSAC  : Bend Err = 0.40% | RMSE = 0.0019m | Time = 0.693s
  Prior-Initialized RANSAC    : Bend Err = 0.40% | RMSE = 0.0019m | Time = 0.166s
  Topological Graph Torus     : Bend Err = 0.01% | RMSE = 0.0019m | Time = 0.003s
  RANSAC-only detected 5 primitives in 0.111s vs Hybrid Pipeline 3 primitives in 0.184s

--- Testing Noise = 5.0mm | Occlusion = 0% ---
  Unconstrained Torus RANSAC  : Bend Err = 0.06% | RMSE = 0.0046m | Time = 0.498s
  Prior-Initialized RANSAC    : Bend Err = 0.06% | RMSE = 0.0046m | Time = 0.137s
  Topological Graph Torus     : Bend Err = 0.08% | RMSE = 0.0047m | Time = 0.005s
  RANSAC-only detected 5 primitives in 0.106s vs Hybrid Pipeline 3 primitives in 0.188s

--- Testing Noise = 5.0mm | Occlusion = 30% ---
  Unconstrained Torus RANSAC  : Bend Err = 1.55% | RMSE = 0.0045m | Time = 0.711s
  Prior-Initialized RANSAC    : Bend Err = 1.55% | RMSE = 0.0045m | Time = 0.172s
  Topological Graph Torus     : Bend Err = 0.03% | RMSE = 0.0045m | Time = 0.005s
  RANSAC-only detected 5 primitives in 0.127s vs Hybrid Pipeline 3 primitives in 0.179s

Ablation results successfully saved to research_audit/ablation_results.json and research_audit/ablation_results.md

```

## Current ablation table (information-equal three-way)
# Empirical Baseline & Ablation Study Results

Three-way information-equal ablation: Unconstrained vs. Prior-Initialized (same axis info, no hard constraint) vs. Topological (same axis info + C1 constraint).

| Noise (mm) | Occlusion | Unconstrained Bend Err | Prior-Init Bend Err | Topological Bend Err | Unconstrained RMSE | Prior-Init RMSE | Topological RMSE | Speedup (vs Unconstrained) |
|---|---|---|---|---|---|---|---|---|
| 0.0 mm | 0% | 0.00% | 0.00% | **0.00%** | 0.000 mm | 0.000 mm | **0.000 mm** | **156.6x** |
| 0.0 mm | 30% | 0.00% | 0.00% | **0.00%** | 0.000 mm | 0.000 mm | **0.000 mm** | **183.4x** |
| 2.0 mm | 0% | 0.04% | 0.04% | **0.03%** | 1.904 mm | 1.904 mm | **1.915 mm** | **126.5x** |
| 2.0 mm | 30% | 0.40% | 0.40% | **0.01%** | 1.867 mm | 1.867 mm | **1.873 mm** | **224.6x** |
| 5.0 mm | 0% | 0.06% | 0.06% | **0.08%** | 4.628 mm | 4.628 mm | **4.662 mm** | **104.7x** |
| 5.0 mm | 30% | 1.55% | 1.55% | **0.03%** | 4.474 mm | 4.474 mm | **4.486 mm** | **146.4x** |


## IMPLEMENTATION_STATUS (repo truth)
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
| 6. Evaluation/ablation | Validated (24 tests pass) | `metrics.py`, `experiments.py`, `run_academic_ablation.py` |
| P0. Topological fitting | Implemented and tested | `topology.py`, `baselines.py`, `test_topology_and_baselines.py` |
| P1. Peer-review fixes | Implemented (commit 929f941) | See below |
| 7. Manuscript/release | Project scaffolding ready | reproducible manifests and roadmap documentation |

## P1 Peer-Review Fixes (commit 929f941)

Addresses all Claude Round-2 criticisms:

### Fixed: Circular Tangent Error Reporting
- `topology.py`: Replaced hardcoded `tangent_error_a_deg = 0.0` (circular/tautological)
  with `inlier_rmse` — the honest point-cloud RMSE measuring how well the fitted
  torus surface explains raw elbow observations.
- C1 tangent continuity is now documented as a **design guarantee** (algebraic property
  of the parameterization), not an empirical finding to be measured.

### Fixed: Shallow-Angle Singularity (Near-Parallel Cylinders)
- `topology.py`: Added `ElbowGeometryError` raised when `||a1×a2|| < sin(8°)`.
  Documents the known failure mode explicitly rather than silently dividing by near-zero.
- Tested in `test_shallow_angle_singularity_raises_elbow_geometry_error`.

### Fixed: Single-Cylinder Fallback
- `topology.py`: Added `fit_torus_topological_single_cylinder()` for occluded second stubs.
  Returns `c1_continuity_verified=False` and `bend_angle_deg=NaN` to signal degraded state.
- Tested in `test_single_cylinder_fallback`.

### Fixed: Information-Equal Ablation
- `baselines.py`: Three-way ablation — Unconstrained vs. Prior-Initialized (same axis
  info, no hard constraint) vs. Topological (same info + C1 hard constraint).
  Shows that **the constraint itself** (not just axis initialization) drives improvement.
  At 5mm noise + 30% occlusion: Unconstrained=1.55%, Prior-Initialized=1.55%, **Topological=0.03%**.

### Fixed: Normal-Consistency DBSCAN Threshold Justified
- `scripts/run_clustering_sweep.py`: Systematic sweep over 5 normal thresholds × 4 eps values.
- Results: 50° threshold is Pareto-optimal (separates touching cylinders without false-splitting
  single pipes with up to 40° weld-seam normal jitter).

## External gates

The following cannot be honestly completed from this repository alone:

- Downloading PSNet5 and verifying its checksum/content.
- Compiling the upstream CUDA operators on the campus GPU.
- Training a ResPointNet++ checkpoint and reporting measured PSNet5 metrics.
- Labeling the 15 real scans and validating against SHREC/PipeNet Lab access.
- Producing publication claims from measured experiments.
- **Elbow IoU on real PSNet5 data** — PSNet5 has no separate elbow annotations;
  requires either supplementary annotation or binary (pipe/background) framing.

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

## Test suite

24 tests, all passing: `$env:PYTHONPATH="src"; python -m pytest tests/ -v`


## Your task
1. Confirm which P0/P1 research gaps from the synthesis are **closed in code** vs still **external-only** (PSNet5 download, GPU training, real elbow labels).
2. State whether the topological graph torus + directional DBSCAN + baselines are sufficient to pivot the paper narrative away from "novel torus RANSAC".
3. Give a **publication readiness score** (0–10) and the **minimum remaining experiments** before a defensible manuscript.
4. Output a structured **Compliance & Gap Closure Report** with sections: Executive Summary, Verified Claims, Remaining Gaps, Recommended Evaluation Protocol (2-class vs 3-class on PSNet5), Manuscript Title/Contribution bullets.

Be adversarial but fair. Do not ask follow-up questions — give a complete report in one message.
