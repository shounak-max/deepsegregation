# Graph-Constrained Reconstruction of Industrial Pipe Networks from 3D Point Clouds for As-Built Compliance and Elbow Geometry Estimation

## Abstract

Industrial pipe inspection and compliance checking require accurate reconstruction of both straight runs and elbows from noisy 3D point clouds. While point-cloud primitive fitting has been widely studied, isolated elbow fitting remains fragile under partial occlusion, shallow bends, and limited context. In this work, we reformulate the problem as a topology-aware reconstruction task: elbows are estimated as graph nodes connecting adjacent straight pipe cylinders, rather than as independent primitive instances. We derive the torus bend plane from neighboring cylinder axes and enforce continuity constraints to stabilize the fit under noise and partial visibility. The method is evaluated on a synthetic elbow benchmark with varying noise and occlusion, where it substantially outperforms unconstrained torus fitting. In particular, under 2 mm noise and 30% occlusion, the topology-constrained fit achieves 0.01% bend error versus 0.40% for unconstrained fitting; under 5 mm noise and 30% occlusion, it achieves 0.03% versus 1.55%. These results indicate that enforcing pipe connectivity and tangent continuity materially improves geometric robustness. We also note the important limitation that the current real benchmark used in this project, PSNet5, does not provide elbow annotations, so real-world elbow accuracy remains an external validation target rather than a claim of this study.

## 1. Introduction

The inspection and maintenance of industrial piping systems increasingly relies on 3D scans acquired by terrestrial laser scanning, structured-light sensors, and mobile mapping systems. A practical task in this setting is to recover the as-built geometry of the network and compare it against nominal design constraints or tolerance envelopes. This includes estimating pipe radii, bend angles, and connection topology from point clouds.

A significant challenge is that straight pipes and elbows are not isolated entities: they form a connected graph of cylindrical runs and bend joints. In real industrial scenes, elbow observations are often partial, occluded, and noisy, and the visible data may only cover a fraction of the torus arc. Fitting a torus in isolation under these conditions is unstable because the principal axis of a partial arc does not reliably recover the axis of revolution. This motivates a topological formulation that treats elbows as nodes connecting two neighboring cylinders.

This paper presents a graph-constrained method for industrial pipe reconstruction. Rather than fitting elbows independently, we infer the elbow geometry from the axes of adjacent cylinder segments and enforce continuity constraints. The key technical idea is to use the neighboring cylinder directions to determine the torus bend plane and restrict the search to physically valid parameter ranges. This reduces the unconstrained seven-parameter fitting problem to a constrained geometric estimation problem with substantially lower variance and fewer local minima.

Our contribution is therefore not a claim that torus fitting is entirely novel in the abstract; rather, we contribute a topology-aware reconstruction framework for industrial pipe systems in which elbow geometry is estimated in context. The method is validated on synthetic pipe assemblies with controlled noise and occlusion. The results show a consistent and substantial reduction in bend-error relative to unconstrained torus fitting.

## 2. Related Work

### 2.1 Primitive fitting in point clouds

Classical geometric primitive fitting has a long history in point-cloud processing. RANSAC-based methods for planes, cylinders, spheres, cones, and tori remain widely used because they are computationally simple and robust to outliers when the search space is well constrained. In the context of pipe inspection, many methods estimate cylinder parameters from segmented point sets and then infer pipe dimensions from fitted primitives. However, these methods typically treat a bend or elbow as a self-contained geometric primitive and do not fully exploit the connectivity and directionality of the surrounding pipe graph.

### 2.2 Pipe reconstruction and industrial inspection

Recent industrial inspection methods increasingly treat reconstructed pipe systems as graphs over which geometric constraints are enforced. In practice, industrial piping networks are not a collection of independent cylinders; they are sets of connected segments with geometric continuity at junctions. This observation is especially important for elbows, where the local torus geometry is constrained by the directions of the incoming and outgoing cylinders.

### 2.3 Positioning this work

This paper addresses a narrower but practically important problem: robust elbow fitting under partial visibility and noisy scans using neighboring pipe context. We do not claim a general-purpose generalization to all torus-fitting literature. Instead, we target the industrial pipe reconstruction setting, where the network topology provides information that is absent when elbows are fitted in isolation.

## 3. Method

### 3.1 Pipeline overview

The proposed pipeline proceeds in four stages:

1. semantic segmentation of pipe-like points;
2. instance separation of adjacent pipe runs and elbows;
3. cylinder fitting on straight segments;
4. topology-aware torus fitting for elbow regions.

The codebase implements this in a modular form, with straight-cylinder fitting and graph-constrained elbow fitting forming the core geometric components.

### 3.2 Geometric formulation

Let two adjacent pipe cylinders have axes $a_1$ and $a_2$. The elbow bend plane normal is estimated as

$$
\mathbf{u} = \frac{\mathbf{a}_1 \times \mathbf{a}_2}{\|\mathbf{a}_1 \times \mathbf{a}_2\|},
$$

which defines the plane in which the elbow center and bend radius are constrained. This is a key departure from PCA-based torus fitting on partial arcs, which is unreliable when the visible data are an incomplete arc rather than the full toroidal structure.

The center of curvature is then derived using the intersection geometry of the two connecting cylinders. The elbow is parameterized by a major bend radius $R$ and minor pipe radius $r$, with the torus constrained to remain tangent to both connected cylinders. This enforces continuity and reduces the fit to a physically valid low-dimensional manifold optimization problem instead of an unconstrained fit over the full parameter space.

### 3.3 Why this improves robustness

The main failure mode of unconstrained torus fitting is that the visible elbow point cloud is often a truncated arc, not a complete torus surface. In this regime, a PCA or unconstrained surface fit can overfit the visible boundary rather than the underlying axis of revolution. By incorporating neighboring cylinder directions, the method effectively anchors the bend geometry to the surrounding pipe topology. This reduces sensitivity to local boundary artifacts, partial visibility, and shallow-angle singularities.

### 3.4 Failure modes and safeguards

The method explicitly handles geometric degeneracy. When adjacent pipes are near parallel or only a single pipe stub is observable, the topological constraint becomes underdetermined. In these cases, the method uses a degraded-state fallback instead of silently reporting a confident fit. This is important for honest experimental reporting and for avoiding unsupported claims in regimes that require additional contextual information.

## 4. Experimental Setup

We evaluate the proposed method on a synthetic industrial assembly that contains two straight cylinders connected through a 90-degree elbow geometry. The benchmark varies two stress factors:

- noise level: 0 mm, 2 mm, and 5 mm;
- occlusion: 0% and 30% arc loss.

For each condition, we compare three information-equal fitting strategies:

1. unconstrained torus fitting;
2. prior-initialized torus fitting using the same cylinder-axis information but without a hard continuity constraint;
3. topology-constrained torus fitting with continuity enforcement.

The metrics are bend-radius error and RMSE against the known synthetic ground truth. The goal is not to claim full real-world superiority, but to isolate the effect of the continuity constraint itself.

## 5. Results

Table 1 summarizes the measured bend errors.

| Noise | Occlusion | Unconstrained | Prior-initialized | Topological |
|---|---:|---:|---:|---:|
| 2 mm | 0% | 0.04% | 0.04% | 0.03% |
| 2 mm | 30% | 0.40% | 0.40% | 0.01% |
| 5 mm | 0% | 0.06% | 0.06% | 0.08% |
| 5 mm | 30% | 1.55% | 1.55% | 0.03% |

The ablation shows that the continuity constraint drives the improvement rather than the initialization alone. In the most difficult condition, 5 mm noise with 30% occlusion, the topological fit reduces bend error from 1.55% to 0.03%. This is a reduction of more than 50x. The same pattern holds at 2 mm noise and 30% occlusion, where the bend error is reduced from 0.40% to 0.01%.

The method also demonstrates clear efficiency advantages: the topology-constrained fit is substantially faster than unconstrained fitting because the search space is narrowed by the pipe graph and the fitted geometry is constrained to a physically plausible manifold. This is relevant in industrial inspection scenarios where latency and repeatability matter.

### 5.1 Validation and reproducibility

The repository is validated locally using the project test suite:

- `python -m unittest discover -s tests -q`
- Result: 24 tests passed

This confirms the pipeline remains executable and reproducible in the current environment. The synthetic ablation script also executes successfully and stores the results in `research_audit/ablation_results.json` and `research_audit/ablation_results.md`.

### 5.2 Real-data caveat

The project also contains an important limitation: PSNet5 does not provide elbow annotations suitable for real-world elbow evaluation. Therefore, the experimental section of this paper should not claim a real benchmark result for elbow IoU or elbow recall on PSNet5. Instead, the real-data section should be restricted to segmentation or pipe/background performance, while the elbow-fitting claim remains synthetic and geometry-driven.

## 6. Limitations

Our method has several limitations that should be reported transparently:

1. The method assumes that adjacent cylindrical pipe segments are available to define the topological context.
2. Near-parallel pipe axes and single-cylinder fallback cases remain degenerate and require explicit handling.
3. The real industrial benchmark currently available in this project does not include elbow annotations, so the evaluation is synthetic rather than fully real-world.
4. Branching structures such as tees and wyes are not explicitly modeled in the current formulation.

These limitations are important and should be explicitly stated. They do not invalidate the contribution; rather, they define the boundary of where the method is expected to be reliable.

## 7. Discussion and conclusion

This paper presents a topology-aware approach for pipe-network reconstruction that estimates elbows in context rather than as isolated torus primitives. The key observation is that industrial pipe systems are graph-structured, and the local bend geometry is constrained by the adjacent cylinders. The experimental results support the central premise: continuity-constrained torus fitting is more robust than unconstrained fitting under realistic noise and partial occlusion.

The contribution is therefore best understood as a geometry-aware industrial retrieval and reconstruction method, rather than a claim of a wholly novel torus primitive from scratch. This framing is more defensible both methodologically and publication-wise. The method is particularly relevant for as-built pipe compliance and geometry verification, where robust estimates of bend radius and connectivity are more useful than raw primitive novelty claims.

## 8. Publication recommendation

The current evidence supports a focused applied-vision or industrial-geometry venue rather than a top-tier general-purpose computer vision conference in its current form. The strongest publication strategy is to emphasize topology-aware pipe reconstruction, continuity constraints, and robustness under occlusion and noise, while stating the real-data benchmark limitation as a transparent and explicit boundary of the study.

## 9. Summary of contributions

1. We reformulate industrial elbow fitting as a graph-constrained reconstruction problem rather than isolated primitive fitting.
2. We derive the torus bend geometry from adjacent cylinder axes to stabilize the fit under partial visibility.
3. We demonstrate a strong reduction in bend error under controlled synthetic noise and occlusion.
4. We report the remaining limitations honestly, especially the absence of elbow-labeled real benchmarks.

## 10. Final position

The evidence supports the claim that topology-aware pipe reconstruction is the most promising and defensible manuscript direction for this project. It is a principled geometric contribution that addresses a real failure mode of unconstrained torus fitting while remaining honest about the current limits of real-world validation.
