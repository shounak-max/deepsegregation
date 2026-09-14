Here is your uncompromising, adversarial peer review. If this manuscript were submitted to *Automation in Construction*, *IEEE T-PAMI*, or *CVPR*, it would face immediate desk rejection or a unanimous strong reject from the panel.

Here is the unvarnished breakdown of why this submission is critically flawed and how you must pivot to salvage it.

### 1. Genuine Novelty: A Trivial Concatenation

Your claimed novelty—"Hybrid DL + Torus RANSAC"—is **not novel**. It is a trivial pipeline concatenation of a standard semantic segmentation network (PointNet++/PointMLP, circa 2017/2022) with a heuristic fitting algorithm.

- **Torus RANSAC is not new:** Schnabel et al. introduced torus fitting in their seminal 2007 paper, *Efficient RANSAC for Point-Cloud Shape Detection*. Claiming Torus RANSAC as a novel contribution in 2026 demonstrates a severe lack of literature review.
- **The Delta is Zero:** Stacking deep learning for coarse semantic segmentation followed by DBSCAN and standard geometric fitting is an archaic multi-stage pipeline. Errors compound at each stage. There is no joint learning, no topological constraint, and no theoretical advancement in robust statistics or differential geometry.

### 2. Closest Existing Research (2020–2026)

You are ignoring an entire generation of primitive-fitting and industrial reconstruction literature:

- **Supervised/Unsupervised Primitive Fitting:** *SPFN* (Li et al.), *ParSeNet* (Sharma et al.), and *HPNet* represent the modern standard for segmenting and fitting primitives natively via deep learning, rather than retrofitting RANSAC.
- **Differentiable RANSAC/Robust Estimators:** Works like *DSAC* (Brachmann et al.) or *Point-to-Symbol* architectures integrate robust fitting directly into the training loop.
- **Industrial Pipe Reconstruction:** Recent papers in *Automation in Construction* (e.g., works by Son, Lee, or Bosché’s group) have already moved past isolated primitive fitting to **topological graph reconstruction** of MEP (Mechanical, Electrical, Plumbing) systems.

### 3. Real Research Gaps (That You Missed)

The actual unsolved problems in 3D pipe reconstruction are:

1. **Topological Continuity:** Pipes are not isolated cylinders and tori; they form continuous graphs. The gap is reconstructing the *routing graph*, using cylinders as edges and tori as nodes, enforcing tangent continuity (G1/C1 continuity) between them.
1. **End-to-End Differentiable Fitting:** Bridging the gap between unstructured point clouds and parametric CAD (B-rep) via differentiable optimization, bypassing discrete clustering.
1. **Severe Occlusion & Insulation:** Handling pipes covered in insulation, obstructed by flanges/valves, or heavily occluded by dense rack structures where even local normal estimation fails.

### 4. Gaps Already Solved

- **Semantic Segmentation of Primitives:** PointNet++ and PointMLP solving Class 0/1/2 is a solved problem on synthetic data.
- **Isolated Cylinder Fitting:** Normal-constrained cylinder RANSAC is a solved commodity algorithm (available in PCL and Open3D). Adaptive iterations for small pipes is an engineering trick, not a research contribution.

### 5. Contradictions to Your Hypothesis & Mathematical Flaws

Your methodological claims contradict geometric reality:

- **Torus Axis via Robust PCA:** This is a **fatal mathematical flaw**. You claim to estimate the torus axis via PCA on candidate points. For a partial elbow (e.g., an occluded 45° bend), the principal components of the point distribution are heavily biased by the occlusion boundary and the arc length, *not* the mathematical axis of revolution. PCA will yield a severely tilted axis, causing your Levenberg-Marquardt refinement to diverge or settle in a local minimum.
- **Normals on Small Pipes (cm):** You claim normal-constrained RANSAC for small pipes. In real point clouds with typical sensor noise (mm), calculating local normals on a pipe with a 25mm radius results in catastrophic curvature aliasing. Your normals will be pure noise, rendering "normal-constrained" RANSAC worse than blind RANSAC.

### 6. Missing Methodology

- **Instance Clustering is Archaic:** Using KD-Tree DBSCAN on spatial coordinates after semantic segmentation is fundamentally inadequate. As you admitted, it merges touching parallel pipes. Modern pipelines solve this via **learned instance embeddings** (e.g., *PointGroup*, *SoftGroup*) or offset-vector voting (e.g., *VoteNet*). Relying on manual -tuning for DBSCAN in dense industrial racks is unacceptable for a top-tier venue.
- **Missing Topological Constraints:** You fit straight pipes and elbows independently. A real pipe rack requires the tangent of the elbow at its endpoints to perfectly align with the directional vector of the connecting straight cylinder. Your pipeline allows disjointed, physically impossible "floating" fittings.

### 7. Missing Datasets

- **Real-World Benchmarks:** Relying on PSNet5, where your target class (elbows) is virtually missing, is methodological suicide. You must evaluate on datasets like **PlantScan3D**, **CLSM**, or the **ABC Dataset** (for synthetic primitive ablation). If a dataset with elbows doesn't exist, *you must release one*. Scanning a local mechanical room with a FARO/Trimble scanner and manually annotating it is the bare minimum effort required.

### 8. Missing Experiments & Baselines

You claim a novel pipeline but test against nothing. You MUST benchmark against:

- **Geometric Baselines:** Efficient RANSAC (Schnabel), Hough Transform for cylinders.
- **Deep Learning Baselines:** SPFN (Supervised Primitive Fitting Network), ParSeNet, HPNet.
- **Instance Segmentation Baselines:** SoftGroup, PointGroup.
- **Ablation Studies:** What is the quantitative impact of your Boundary-CB loss? Where is the sim-to-real adaptation baseline (e.g., DANN or self-supervised domain adaptation)?

### 9. Top Reasons for Desk Rejection

If I am reviewing this, I am voting **Strong Reject** based on these fatal flaws:

1. **Catastrophic Empirical Failure on Real Data:** Your real-world Elbow IoU is literally **0.000**. You are proposing a paper centered on elbow geometry estimation, and your model cannot find a single elbow in real data. The paper ends right there.
1. **Flawed Torus Mathematics:** Using PCA to find the axis of a partially occluded torus is mathematically invalid and proves a lack of understanding of robust surface fitting.
1. **Unresolved Instance Merging:** Admitting that DBSCAN merges adjacent pipes—the exact primary challenge of industrial pipe reconstruction—without proposing a solution proves the pipeline is not fit for purpose.
1. **No Baselines:** Claiming performance metrics without comparing to a single state-of-the-art primitive fitting network (e.g., SPFN) violates basic scientific method.

### 10. How to Pivot: Stronger Contributions That Can Be Accepted

To salvage this into a high-impact paper, abandon the disjointed DBSCAN + RANSAC pipeline and implement the following:

- **Pivot 1: Graph-Constrained Joint Fitting (Topological RANSAC):** Do not fit elbows in isolation. Detect cylinders first. Formulate elbows as the **mathematical intersection and routing node** between two detected cylinders. If you know the axes of two intersecting straight pipes, the torus axis, major radius (), and minor radius () are heavily mathematically constrained. This solves your PCA flaw instantly and guarantees C1 continuity.
- **Pivot 2: Learned Vector Voting for Instance Segmentation:** Replace DBSCAN with a network head that predicts the offset vector from every point to the central axis of its parent pipe (similar to *VoteNet*). This forces touching pipes to cast votes to two distinct geometric centers, flawlessly solving the "adjacent pipe merge" problem.
- **Pivot 3: Solve the Sim-to-Real Gap:** Your Elbow IoU is 0 because your synthetic elbows lack the noise, clutter, and occlusion patterns of real data. Implement a Domain Adversarial Neural Network (DANN) layer, or apply contrastive learning to align the synthetic and real feature spaces. Release a small, high-quality, hand-annotated real-world test set (e.g., 5-10 dense scans of a mechanical room) to prove the sim-to-real gap is bridged.

**Summary:** Stop trying to chain obsolete algorithms together. Address the geometry mathematically, solve the instance clustering natively in the network, and evaluate against modern baselines. Until then, this is a clear reject.

Gemini is AI and can make mistakes.

Analyzing the Core Novelty
