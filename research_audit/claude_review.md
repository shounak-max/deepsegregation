## Reviewer #2 — Confidential Comments to Authors

 

**Recommendation: REJECT (major methodological and empirical gaps; not competitive at current maturity)**

 

---

 

### 1. Genuine Novelty — Weak

 

The pipeline is a **sequential concatenation of four well-established components**: (a) a learned semantic segmentation backbone (PointNet++/PointMLP — both pre-existing, off-the-shelf), (b) DBSCAN instance separation (textbook, unchanged), (c) RANSAC-based primitive fitting for cylinders (30+ years old, Fischler & Bolles 1981; refined extensively by Schnabel et al. 2007's Efficient RANSAC for exactly this class of shape), and (d) torus RANSAC for elbows.

 

The **only potentially novel piece** is the torus-fitting formulation with basin-consistency checks and the 3× adaptive iteration schedule for small-radius candidates. But:

 

 - Torus primitive detection via RANSAC is **not new** — Schnabel et al. (2007) explicitly include torus as one of the five canonical primitives in their efficient RANSAC framework, with minimal-sample estimation and refinement. Your torus RANSAC needs to be explicitly differentiated from theirs, and currently is not.
 - The residual formulation you give, (x′2+y′2−R)2+z′2−r2, is the standard implicit torus equation used in every CAD/reverse-engineering primitive-fitting paper since the 1990s (Chen & Wang 1994 revolve-surface fitting; PCL's `SampleConsensusModelCircle3D`/torus extensions).
 - "Basin-consistency checks to avoid local minima" is not defined precisely enough to assess as a contribution — is this multi-start LM with clustering of converged parameter estimates? If so, this is standard practice, not innovation.
 - Boundary-CB + Focal loss combination for class imbalance is an engineering choice, not a methodological contribution — both losses are used off-the-shelf and their combination has appeared in point cloud segmentation literature already (e.g., boundary-aware losses in ParametricNet, BAAF-Net-style boundary refinement).

 

**Where the real delta could be, but isn't demonstrated:** a genuinely novel contribution would be (i) a differentiable/learned torus-fitting head that back-propagates through geometric fitting (à la SPFN, ParSeNet, or Point2Cyl), replacing RANSAC entirely, or (ii) joint segmentation+fitting optimization rather than the current disjoint pipeline where segmentation errors are simply passed downstream uncorrected. Neither is present. **As submitted, this reads as a systems/engineering paper, not a methods paper**, and if positioned as the latter for CVPR/T-PAMI it will be rejected on novelty alone.

 

---

 

### 2. Closest Existing Research (2020–2026)

 

You are missing citation and comparison against a specific, well-known lineage:

 

 - **SPFN (Li et al., CVPR 2019)** — Supervised Primitive Fitting Network: differentiable, end-to-end primitive (plane/sphere/cylinder/cone) fitting directly from point clouds using a per-point type + parameter regression network with Hungarian matching. This is the direct predecessor your work should be benchmarked against.
 - **ParSeNet (Sharma et al., ECCV 2020)** — extends primitive fitting to include B-spline patches; explicitly handles the joint segmentation+fitting problem your pipeline decouples.
 - **Point2Cyl (Uy et al., CVPR 2022)** — decomposes point clouds into extrusion cylinders via differentiable fitting; directly relevant to pipe/cylinder recovery.
 - **Efficient RANSAC (Schnabel, Wahl, Klein, 2007)** — the torus/cylinder/sphere/plane/cone RANSAC baseline you must compare against and did not.
 - **PIE-NET (Wang et al. 2020)** and **HPNet (Yan et al. 2021)** — parametric surface/edge fitting from point clouds, relevant instance/primitive segmentation baselines.
 - **PSNet / PSNet5 origin paper** — you use their benchmark but do not appear to compare against their own reported segmentation numbers on the same split, which is a serious omission when your accuracy (73.25%, 0.316 mIoU) is a headline result.
 - **Industrial pipe-specific works**: Czerniawski & Leite (2020, *Automation in Construction*) on automated digital modeling of MEP systems from point clouds; Wang & Cho (2015) cylinder fitting for as-built piping; Kawashima et al. (2014) automated pipe reconstruction — these are the direct application-domain competitors in *Automation in Construction*, which is one of your target venues, and reviewers there **will** know this literature.
 - **Elbow/torus-specific**: Qiu et al. (2022) elbow parameter extraction from as-built scans; there is niche but real prior work specifically on pipe elbow reconstruction that must be discussed and, ideally, compared against.

 

If you submit to *Automation in Construction* without citing Czerniawski & Leite, Wang & Cho, or Kawashima, that alone signals inadequate literature review to a domain reviewer.

 

---

 

### 3. Real, Unsolved Research Gaps

 

These are legitimate:

 

 - **Joint learning of segmentation + geometric fitting under partial/occluded elbow observation** — no strong general solution exists for recovering full torus parameters from a small, non-uniformly sampled angular arc of an elbow (e.g., <90° of visible surface due to occlusion by racking/flanges). This is genuinely open.
 - **Uncertainty quantification / calibrated confidence for compliance decisions** — a PASS/FAIL threshold at ±5% with no propagated uncertainty from segmentation noise → clustering error → RANSAC fit variance is a real gap; nobody has a rigorous end-to-end uncertainty pipeline for this application.
 - **Sim-to-real domain adaptation specifically for thin, high-curvature structures (small-radius pipes and elbows)** under LiDAR/structured-light noise models — general sim-to-real point cloud DA exists, but the failure mode for *thin cylindrical* geometry (where noise magnitude can exceed pipe radius) is under-studied.
 - **Instance separation for touching/parallel pipe racks** — this is a real, acknowledged-by-you limitation with no strong general solution; anisotropic/oriented DBSCAN variants and normal-aware region growing exist but none robustly solve dense parallel pipe racks without manual tuning.

 

---

 

### 4. Gaps Already Considered Solved

 

 - **Cylinder fitting on clean/moderately noisy data** — solved to high precision by classic RANSAC + LM refinement; your <1.5% MRE on clean synthetic data is expected, not a contribution.
 - **Generic semantic segmentation architectures** (PointNet++, PointMLP, KPConv, Point Transformer) for indoor/industrial point clouds — architecture choice itself is not a contribution space anymore; the field has moved to comparing on hard benchmarks, not proposing new backbones.
 - **Basic outlier rejection / robust statistics for RANSAC** (MSAC, MLESAC, LM refinement with bounds) — standard, solved, well-documented.
 - **KD-tree accelerated DBSCAN** — a solved systems-engineering problem, not research.

 

---

 

### 5. Contradictions to Your Hypotheses

 

 - **Torus fitting is ill-conditioned for elbows with large bend radius R relative to pipe radius r** (near-straight approximations) and **degenerates entirely as bend angle → small values or R → ∞** (an elbow with very gentle curvature is numerically indistinguishable from a cylinder within noise tolerance). Your minimal-sample torus RANSAC will suffer severe variance in exactly the regime industrial 45°/22.5° long-radius elbows occupy. You report only 45°/90° elbows on clean synthetic data — this masks the conditioning problem rather than solving it.
 - **PCA-based axis estimation is well known to be unstable on partial/occluded arcs** (Besl & Jain 1988's original ICP-era literature already documents this): PCA principal axis is only reliable when the sampled surface spans a sufficient angular range around the tube; a partially occluded elbow (a realistic industrial scenario given flange clutter, insulation, hangers) will bias the PCA axis and propagate error into every downstream parameter (r, R).
 - **Small-radius RANSAC singularities**: For r < 5cm with typical terrestrial/handheld LiDAR noise (σ often 2–5mm), your signal-to-noise ratio in radius estimation approaches unity. The literature (Nurunnabi et al. 2017 on robust cylinder fitting) shows normal-based cylinder RANSAC becomes numerically unstable and multi-modal (multiple radius/axis hypotheses fit residuals similarly) exactly in this regime — your "3x adaptive iterations" is a band-aid, not a resolution, and you present no analysis of *why* it works or bounds on when it fails.
 - **Normal estimation on noisy point clouds** (needed for your "normal-constrained cylinder RANSAC") is itself unreliable at small scale — standard PCA-based normal estimation has bias that scales with curvature/radius ratio, meaning your normal constraints could be *actively hurting* small-pipe estimation rather than helping, and you present no ablation isolating this.
 - **DBSCAN's core theoretical property** — a single global eps cannot handle pipes of varying diameter and spacing in the same scene; this is a known, provable limitation (density-based clustering assumes roughly uniform density), not a fixable engineering detail. This contradicts your framing of it as a current implementation limitation ("unless eps is manually tuned") — it's a structural incompatibility between DBSCAN and heterogeneous industrial pipe racks.

 

---

 

### 6. Missing Methodology

 

 - No handling of **partial/occluded elbows** — the single most common real-world elbow scenario (blocked by flanges, valves, supports, insulation) is not addressed at all in the described pipeline.
 - No **flange/valve/fitting interference model** — these are geometrically similar to short cylinder/torus segments and will generate false positive primitive fits; no rejection criteria described.
 - **Orientation ambiguity**: for a torus fit to a partial arc, there can be a near-degenerate ambiguity between (R, r) pairs producing similar residuals (a classic torus-fitting pathology) — no discussion of how basin-consistency checks distinguish a true elbow from a degenerate alternative solution.
 - No **joint optimization or feedback loop** between segmentation confidence and geometric fit quality — errors are one-directional (seg → cluster → fit); a fit failure never informs re-segmentation.
 - No **multi-elbow / branching (tee/wye) joint handling** — real piping systems have tees, reducers, and branch fittings, none of which fit either cylinder or torus models; the pipeline appears to assume every non-straight segment is a simple elbow.
 - No formal **coordinate frame / registration** discussion — how is the point cloud aligned to a global frame for CAD comparison? Compliance QA requires this and it's absent.

 

---

 

### 7. Missing Datasets

 

 - **PSNet5 alone is insufficient** — it's a single benchmark with (by your own admission) essentially no annotated elbows, making your headline real-data claim untestable for your core novel contribution (torus fitting). This is close to a fatal flaw (see §9).
 - Missing: **S3DIS / ScanNet** — not pipe-specific, but standard for demonstrating segmentation generalization; absence weakens claims about backbone choice.
 - Missing: **real scanned plant/facility datasets** — e.g., academic-industrial partnerships have released refinery/plant scans (some construction-informatics groups at TUM, Georgia Tech, and NIST have released MEP/pipe scan datasets); if truly unavailable publicly, this should be stated as a limitation, and a **self-collected real elbow dataset** (even small, n=20–50 elbows with ground truth calipered measurements) would substantially strengthen the paper.
 - **SHREC pipe/tube challenges** and other CAD-reverse-engineering benchmarks (ABC dataset for CAD primitive fitting) are relevant for the geometric fitting sub-task in isolation and are conspicuously absent.

 

---

 

### 8. Missing Experiments & Baselines

 

Mandatory baselines currently absent, any one of which a competent reviewer will demand:

 

 1. **RANSAC-only, no DL segmentation** (run cylinder/torus RANSAC on raw or naively-thresholded point clouds) — to isolate what the learned segmentation actually buys you.
 1. **DL-only geometry regression** (direct radius/bend-radius regression from segmented points via MLP, no RANSAC) — to isolate what geometric fitting buys you over learned regression.
 1. **Efficient RANSAC (Schnabel et al.)** as the primitive-fitting baseline — directly comparable torus/cylinder detector.
 1. **SPFN or Point2Cyl** as a differentiable end-to-end primitive-fitting baseline — critical, since this is the modern SOTA family your method should be positioned against.
 1. **Learned instance segmentation** (SoftGroup, PointGroup, or similar) as a DBSCAN alternative — you assert DBSCAN's failure mode but never test whether a learned instance segmentation head resolves it.
 1. **Hough-transform-based cylinder/circle detection** — classical alternative, cheap to implement, expected by geometric-fitting reviewers.
 1. **PointNet++ vs PointMLP ablation with matched training budget** — currently unclear if the comparison controls for epochs/hardware; the K80 + 10-epoch result looks under-trained relative to typical PointNet++ literature (usually 100+ epochs), undermining the reported 0.316 mIoU as representative of the architecture's true capability.
 1. **Cross-dataset generalization** (train synthetic → test PSNet5 vs. train PSNet5 → test synthetic) is implied but not reported with numbers beyond a qualitative "substantial drop" — quantify this.

 

---

 

### 9. Top Reasons for REJECT

 

 1. **Elbow IoU = 0.000, Recall = 0.000 on the only real dataset used.** The paper's entire novel contribution (torus/elbow fitting) is validated *exclusively on synthetic data*, while the real-world evaluation shows the segmentation stage cannot detect elbows at all. This makes the claimed pipeline's real-world elbow reconstruction capability **entirely unvalidated** — a fatal disconnect between claimed contribution and empirical evidence.
 1. **10 epochs on a Tesla K80 is not a serious training budget** for PointNet++/PointMLP by 2024-2026 standards; 73.25% accuracy / 0.316 mIoU will read as an under-trained, non-competitive baseline to any reviewer familiar with S3DIS-class results (typical mIoU well above this even for older backbones on comparable indoor benchmarks).
 1. **No comparison to Schnabel et al.'s Efficient RANSAC, SPFN, ParSeNet, or Point2Cyl** — omitting the closest prior art as baselines is disqualifying for a novelty claim in this specific subfield.
 1. **The DL segmentation and geometric fitting stages are not jointly evaluated or ablated against each other** — no experiment isolates whether performance gains (where they exist) come from the DL stage, the RANSAC stage, or the loss design, making it impossible to attribute the pipeline's stated capabilities to any specific component.
 1. **Acknowledged unresolved DBSCAN clustering failure in exactly the scenario (dense industrial pipe racks) the paper claims to target** — the stated real-world use case (industrial compliance QA in cluttered plants) is precisely where the pipeline is documented to fail without manual intervention, undermining the practical deployability claim central to the paper's motivation.

 

---

 

### 10. Realistic Path to a Strong Paper

 

 - **Reframe as a systems/applications paper for *Automation in Construction*, not a methods paper for CVPR/T-PAMI.** The venue mismatch is part of the current weakness — a rigorous empirical study of an industrial pipeline with honest failure analysis is publishable at AinC if positioned correctly, but not as claiming DL/geometric-fitting novelty at a vision venue.
 - **Replace the disjoint RANSAC stage with a differentiable fitting head** (SPFN/Point2Cyl-style), enabling joint fine-tuning of segmentation + geometry and giving you an actual novelty claim over prior primitive-fitting work.
 - **Collect or partner for a small real annotated elbow dataset** — even 30–50 real elbows with calipered ground truth would let you validate the one claimed novel contribution on real data, closing the current fatal gap.
 - **Run the mandatory baselines** (§8), especially Efficient RANSAC and one learned instance-segmentation method (SoftGroup/PointGroup) — these are cheap to run and would substantially strengthen empirical credibility.
 - **Add rigorous ablations isolating conditioning failure modes**: quantify torus-fitting variance as a function of (a) angular coverage of the visible arc, (b) R/r ratio, (c) noise-to-radius ratio — this turns your "honest findings" limitations section into a real contribution (a characterization of when/why torus RANSAC fails), which is more valuable and more publishable than claiming it as a working novel method.
 - **Propagate uncertainty through the pipeline** to the PASS/FAIL compliance decision — a calibrated confidence-weighted compliance report (rather than a point estimate against a hard ±5% threshold) would be a genuinely useful, defensible contribution for the target application domain.
 - **Train segmentation properly** (100+ epochs, modern GPU, or explicitly justify the K80/10-epoch choice as a deliberate resource-constrained deployment scenario) — as currently framed it just looks under-resourced rather than intentional.

 

**Bottom line**: the engineering integration is competent, but the paper as described has no defensible novel methodological contribution validated on real data, omits the standard comparison baselines a reviewer in this subfield will expect, and its own reported results (Elbow IoU = 0) undercut its central claim. This needs another full research cycle — collect real elbow data, add a differentiable fitting baseline, and reframe the contribution around a rigorous failure-mode characterization — before resubmission.

just now[Claude is AI and can make mistakes. Please double-check responses.](https://support.anthropic.com/en/articles/8525154-claude-is-providing-incorrect-or-misleading-responses-what-s-going-on)
