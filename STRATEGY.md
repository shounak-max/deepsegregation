# Industrial-scene segregation strategy

## Objective

Produce a reproducible industrial point-cloud model and a usable
downstream extractor:

1. **Semantic segmentation:** background, straight pipe, and elbow.
2. **Instance separation:** one cluster per physical pipe.
3. **Geometry:** cylinder radius/axis for straight pipes and torus radii for
   elbows.
4. **Operational output:** a JSON compliance report plus labelled point-cloud
   outputs.

The project must report held-out real-scene metrics. Synthetic results are
development checks only.

## Phase 1: bring up the reference GPU stack

Use the upstream [ResPointNet2](https://github.com/PointCloudYC/ResPointNet2)
repository on the authorized CUDA host. Match its documented environment:
Python 3.10, PyTorch 1.13.1 with CUDA 11.7, a driver supporting CUDA 11.7,
and a GPU with compute capability 6.0 or newer. Compile the custom operators
before training.

Stage the complete official PSNet5 distribution on storage available to the
GPU host, under `data/PSNet/PSNet5` (or pass its equivalent path to the
upstream tools). Do not replace it with the small derived `.npz` sample and do
not commit raw data or checkpoints. Verify that all four areas and the five
original classes are present. Run the upstream PSNet5 preprocessing against
the complete corpus and a one-epoch smoke run before adapting the model.

**Gate:** the reference loader, CUDA operators, forward pass, loss, and
checkpoint save/evaluation all work on the GPU host.

Local verification commands:

```text
python scripts/check_reference.py
python scripts/check_psnet5.py --data-root data/PSNet/PSNet5
```

These commands validate repository/data layout only. They do not prove that
the remote GPU can compile or execute CUDA code; that remains an external
assumption for this step.

## Phase 2: create the pipe task

Use the existing preparation script only as a derived-cache builder over the
original corpus; its default is now to read complete annotation files.
Preserve the original five-class labels for reproducibility,
then define the task mapping explicitly:

- background and non-pipe structural classes -> `background`
- `pipe` -> `straight_pipe` until elbow labels are available
- manually verified elbow annotations -> `elbow`

Do not claim elbow performance from PSNet5 alone: PSNet5 does not provide the
required straight-versus-elbow annotation. Add a small, audited elbow-labelled
industrial set and keep scenes, not random points, as the train/validation/test
split unit. Use Area_1/2 for training, Area_4 for validation, and Area_3 for
the final held-out test only after labels are fixed.

Normalize coordinates per scene, retain normals when available, and sample
fixed-size blocks with overlap so thin pipes are not discarded. Record the
mapping, point counts, and split manifest with every run.

**Gate:** every split has non-empty labels for each expected class, no scene
leaks across splits, and a visual label check passes.

## Phase 3: train segmentation on the K80

The paper's reported 94% accuracy and 87% mIoU apply to the full
five-class ResPointNet++ architecture and protocol. The current local
PointNet++ SSG implementation is only a bring-up baseline and must not be
compared to those numbers as if it were the paper model. Start with the
upstream ResPointNet++ five-class baseline. Then replace its
classifier with a three-class head and fine-tune on the pipe-labelled data.
Use Boundary-CB only after establishing ordinary cross-entropy as a baseline.
Track accuracy, per-class IoU/recall, macro mIoU, and boundary F1; elbow recall
is the primary risk metric.

The K80 is memory constrained. Prefer one GPU, conservative block sizes,
gradient accumulation, mixed precision only if supported by the pinned
software, and frequent checkpointing. Run a short smoke job first, then
several fixed-seed runs with early stopping selected on the validation scene.
Never treat a training loss or synthetic score as final evidence.

**Gate:** the selected checkpoint beats the baseline on held-out validation
and has acceptable elbow recall, with the exact command and environment
recorded.

## Phase 4: industrial segregation pipeline

Run inference block-by-block, merge overlapping predictions, and reject
low-confidence background points. Apply DBSCAN separately to straight-pipe and
elbow predictions. Fit cylinders to straight clusters and toruses to elbow
clusters. Reject fits using residual, inlier ratio, radius bounds, and
connectivity checks rather than forcing a geometry result.

Evaluate both semantic and geometric outputs: instance precision/recall,
cluster over/under-segmentation, radius MAE/MRE, elbow major/minor-radius
error, and processing time per scene. Keep a failure set containing occlusion,
clutter, touching pipes, sparse scans, and small diameters.

**Gate:** a complete held-out scene produces labels, instances, geometry, and a
report without manual intervention; failures are explicit and auditable.

## Phase 5: reinforcement-learning extension (only if needed)

Do not replace supervised segmentation with RL. Use RL as a constrained
post-processing policy only after the supervised pipeline is stable. The
agent may select DBSCAN and fitting parameters from scene statistics, while
the reward combines held-out segmentation/instance quality, geometric error,
runtime, and penalties for invalid fits. Keep the policy offline and evaluate
on unseen scenes; never use test labels for reward tuning.

## Required deliverables

- pinned upstream environment and reproducible GPU command;
- versioned split/mapping manifest;
- best three-class checkpoint with checksum;
- per-scene test metrics and failure analysis;
- labelled point-cloud and JSON outputs;
- one command for smoke testing and one command for remote status/fetch.
