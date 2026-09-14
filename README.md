# Deep Segregation

Deep Segregation is a hybrid point-cloud pipeline for separating industrial
pipe instances and estimating straight-pipe and elbow geometry.

The implementation now covers the CPU-side path across the roadmap: synthetic
scan generation, PSNet-shaped area discovery, preprocessing, three-class
segmentation contracts, Boundary-CB loss, DBSCAN instance separation,
straight-cylinder and torus RANSAC fitting, compliance reports, metrics,
ablation helpers, and a UI-neutral Qt/VTK view-model boundary. The vendored
ResPointNet++ tree is under `vendor/ResPointNet2` for Phase-0 GPU bring-up.

## Research-grade publication status

This repository is not yet publication-ready as a standalone "novel torus-fitting
RANSAC" paper. The strongest defensible interpretation of the work is as a
research system for graph-constrained industrial pipe network reconstruction,
where elbows are estimated from adjacent straight-pipe geometry rather than as
independent isolated primitive fits.

The reason is straightforward and now documented in the audit materials:

- isolated torus fitting is not a defensible novelty claim in the current
  literature;
- real PSNet5 evaluation does not support elbow claims because the dataset lacks
  separate elbow annotations;
- the method becomes substantially stronger when the torus fit is constrained by
  neighboring pipe axis continuity and topology.

The project therefore has a credible publication route if framed as:

- topological pipe-network reconstruction from 3D industrial point clouds;
- physically valid elbow estimation under continuity constraints;
- compliance-oriented radius and bend estimation for as-built pipe QA.

## Current validated baseline

Run the current checks with:

```text
python -m unittest discover -s tests
```

This repository currently validates successfully in the local environment with 24
passing tests. The codebase is therefore operational and reproducible in its
current form, but the manuscript framing must be tightened before it is
research-grade.

## Immediate publication checklist

1. Reframe the paper around graph-constrained pipe reconstruction rather than
   isolated torus novelty.
2. Remove or qualify all real elbow claims tied to PSNet5 unless separate elbow
   annotations are obtained or an explicit hybrid benchmark is constructed.
3. Add ablations comparing: RANSAC-only, unconstrained torus fitting,
   prior-initialized torus fitting, and topological constrained torus fitting.
4. Report synthetic and hybrid real/synthetic elbow benchmarks, not just a single
   synthetic claim.
5. State the limitations honestly: external GPU training, data access, and
   annotation gaps are part of the pipeline and should be documented as such.

## Local workflow

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

## Research-audit documents

See the review and publication-readiness materials in `research_audit/` for the
full cross-model analysis, ablation evidence, and the precise review-driven
rewriting plan.
