# Deep Segregation

Deep Segregation is a focused point-cloud pipeline for industrial-scene pipe
segmentation, instance separation, and geometry estimation. The published
ResPointNet++ architecture is the intended GPU segmentation reference; the
current remote smoke runs use the local pure-PyTorch PointNet++ SSG only as a
bring-up baseline. The local package supplies dataset contracts, losses,
post-processing, and cylinder/torus fitting.

Run the checks with:

```text
python -m unittest discover -s tests
```

Run an end-to-end synthetic smoke test with `python scripts/smoke_test.py`.
Generate synthetic training data with `python scripts/generate_synthetic_dataset.py`.
Prepare the original PSNet5 areas with
`python scripts/prepare_real_dataset.py --data-root data/PSNet/PSNet5`.
Validate raw PSNet5 staging first with
`python scripts/check_psnet5.py --data-root data/PSNet/PSNet5`.
On the remote Linux host, use
`bash scripts/setup_psnet5_remote.sh`; set `DATA_URL` only when the official
archive is available as a direct download URL.
Inspect the upstream reference with `python scripts/check_reference.py`.
Use `python scripts/cluster_pipeline.py` to start, monitor, stop, and fetch
training jobs on the configured GPU host.

GPU-dependent status: upstream ResPointNet++ staging, CUDA-op compilation,
checkpoint training, and real-scan validation require the pinned upstream
environment and the authorized GPU host. Follow [STRATEGY.md](STRATEGY.md) for
the execution order and acceptance criteria. Checkpoints and generated
datasets are kept out of version control. The paper's reported 94% accuracy
and 87% mIoU are reference results for its full five-class ResPointNet++
experiment, not acceptance results for the current binary PointNet++ SSG
baseline. On the current GPU host, the upstream C++ subsampling operator
builds successfully, but the CUDA operator build is blocked because `nvcc`
and a CUDA toolkit are not installed; the PyTorch CUDA runtime alone cannot
compile the custom kernels. A Conda CUDA 11.3 `nvcc` compiler and runtime
headers were added under `/home`, but the upstream extension build still
requires the missing `cusparse.h` development header on this host.
