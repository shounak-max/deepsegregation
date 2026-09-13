# Deep Segregation

Deep Segregation is a hybrid point-cloud pipeline for separating industrial
pipe instances and estimating straight-pipe and elbow geometry.

The implementation now covers the CPU-side path across the roadmap: synthetic
scan generation, PSNet-shaped area discovery, preprocessing, three-class
segmentation contracts, Boundary-CB loss, DBSCAN instance separation,
straight-cylinder and torus RANSAC fitting, compliance reports, metrics,
ablation helpers, and a UI-neutral Qt/VTK view-model boundary. The vendored
ResPointNet++ tree is under `vendor/ResPointNet2` for Phase-0 GPU bring-up.

Run the current checks with:

```text
python -m unittest discover -s tests
```

Run an end-to-end synthetic smoke test with `python scripts/smoke_test.py`.
Generate the Phase-1 dataset with `python scripts/generate_synthetic_dataset.py`.
Inspect the Phase-0 reference with `python scripts/check_reference.py` and
print its training command with `python scripts/train_reference.py --smoke`.

GPU-dependent status: PSNet5 download, CUDA-op compilation, checkpoint
training, and real-scan validation still require the pinned upstream
environment and data access; use `scripts/check_reference.py` before starting
that operation. No checkpoint or real scans are fabricated by this project.
The downloaded `artifacts/point_mlp_k80/best.pt` is a real synthetic-baseline
checkpoint from the campus K80 cluster, not a ResPointNet++ checkpoint.
