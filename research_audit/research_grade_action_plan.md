# Research-grade action plan for DeepSegregation

## Goal

Make the project publication-worthy by reframing the contribution around
**topological graph-constrained pipe reconstruction** rather than isolated
primitive novelty, while giving a truthful assessment of the remaining data and
methodological gaps.

## Consensus from the review loop

The repository is operational and the test suite passes, but the current framing
is not publication-ready in its present form.

The main pending items are:

1. Reframe the contribution around pipe-network topology, not isolated torus
   fitting.
2. Remove unsupported real-world elbow claims on PSNet5 unless elbow labels are
   added or a hybrid benchmark is built.
3. Add rigorous ablations for unconstrained, prior-initialized, and topological
   torus fitting.
4. Strengthen the evaluation with synthetic + hybrid real/synthetic elbow data.
5. Document GPU and data-access requirements honestly rather than treating them
   as minor implementation details.

## Immediate tasks

### P0 — Methodological fixes

- Replace the isolated torus claim with a connectivity-constrained geometric
  formulation.
- Derive torus axis from adjacent straight-pipe axes and enforce continuity
  constraints.
- Add a clear failure-mode guard for near-parallel axes and partial occlusion.
- Keep the unconstrained torus model as a baseline rather than as the main claim.

### P1 — Evaluation fixes

- Use synthetic 45°/90° elbow scans with known geometry as the primary benchmark.
- Build a hybrid real/synthetic benchmark for elbow estimation.
- Run a 3-way ablation matrix for method comparison with identical data splits.
- Present numbers for radius error, bend error, and failure rate, not only a
  single aggregate score.

### P2 — Paper framing

- Position the work as industrial pipe-network reconstruction and compliance QA.
- State clearly that PSNet5 is used for pipe/background segmentation and not for
  direct elbow evaluation without additional labels.
- Emphasize the physical validity of the model under continuity constraints.

## Current repo status

Validated evidence:

- `python -m unittest discover -s tests` passes locally.
- The implementation is credible as an engineering prototype.
- The critical problem is the paper framing and evaluation story, not the code
  execution itself.

## Recommended final paper direction

A defensible final direction is:

"Graph-Constrained Reconstruction of Industrial Pipe Networks from 3D Point
Clouds for As-Built Compliance and Elbow Geometry Estimation"

This framing matches the repository’s strengths, satisfies the review concerns,
and is much more likely to be accepted than a claim centered on isolated torus
RANSAC novelty.
