# Execution Roadmap
## 3D Cylindrical Pipe Radius Estimation via Hybrid DL + RANSAC

This turns the proposal's 7-stage pipeline into a week-by-week build plan, anchored to the actual [ResPointNet2 repo](https://github.com/PointCloudYC/ResPointNet2), with a **Consensus checkpoint** in every phase so literature claims stay validated as the implementation evolves (not just at proposal-writing time).

Total horizon: **14 weeks (~3.5 months)**, matching the proposal's submission target.

---

## How to use Consensus throughout (not just once)

Treat Consensus as a standing validation tool, not a one-time literature review. Three recurring uses:

1. **Claim-checking before you write anything down.** Every number in your proposal (11.2% baseline, 30% small-pipe error, PipeSegNet's 2.73%) came from a paper. Before you cite a number in the final paper, re-run the query in Consensus and pull the number from the actual paper page, not from memory of this roadmap.
2. **Gap-checking before you build a component.** Before implementing torus-fitting RANSAC (your one genuinely novel module), search Consensus for "torus fitting RANSAC point cloud" and "pipe elbow radius estimation" to confirm no one has published this since your proposal was written — this is a 2026 project, the field moves fast.
3. **Metric-benchmarking after each experiment.** When you get a result (e.g., your MRE), search Consensus for the metric + domain to place your number against current literature, not just the 2022–2026 papers you already found.

Suggested standing queries to re-run at each phase (see phase sections below for when):
- `torus fitting RANSAC point cloud`
- `pipe elbow bend radius estimation`
- `PointNet++ industrial point cloud segmentation benchmark`
- `RANSAC cylinder fitting small radius accuracy`
- `point cloud instance segmentation cluttered industrial scene`

---

## Phase 0 — Environment & Repo Bring-Up (Week 1)

**Goal:** ResPointNet++ trains and evaluates on PSNet5 out of the box, before you touch your own data.

- Provision a CUDA-capable machine (repo is tested on RTX 3090/4090, CUDA 11.7, PyTorch 1.13.1, Python 3.10 — this is an older, pinned stack, not your latest CUDA toolkit).
- Run `install-conda.sh` (or `install-uv.sh`), then `init.sh` to compile the custom C++/CUDA ops (`grid_subsampling`, `pt_custom_ops`). This is the step most likely to fail — budget 2–3 days, and use `troubleshooting.md` / `yc/faq.md` in the repo directly.
- Download PSNet5 (80M-point, 5-class industrial dataset) and confirm `python datasets/PSNet5.py` preprocesses correctly.
- Run `train-psnet5.sh` for a short smoke-test (few epochs) to confirm the pipeline trains end-to-end and produces checkpoints under `log/`.
- **Note the gap:** the README's "Pre-trained Models" link is a placeholder (`TOADD`) — there is currently **no public pretrained checkpoint**. Your proposal's "pre-trained on PSNet5" assumption means *you* train that base model first; it is not a download-and-go asset. Budget GPU time for this.

**Deliverable:** a working ResPointNet++ training/eval loop on PSNet5, reproducing something close to the paper's 94% accuracy / 87% mIoU, so you have a trustworthy baseline before fine-tuning on pipes specifically.

---

## Phase 1 — Data Pipeline: Real + Synthetic Scans (Weeks 1–3, overlaps Phase 0)

**Goal:** produce the 200+ synthetic scans and get your 15 real pipe scans into a labeled, PSNet-compatible format.

- Set up Open3D virtual scanner + FreeCAD to generate synthetic CAD-based pipe scans (straight + 45°/90° elbow geometry, varying radius including small-diameter <5cm cases specifically, since that's your headline gap).
- Inject the three noise levels you promised (Gaussian noise, point dropout, outlier injection) at generation time — build this into the generator, not as an afterthought.
- Format synthetic + real scans to match `datasets/PSNet5.py`'s expected structure (`Area_N` folders) so you can reuse the existing data loader rather than writing a new one from scratch.
- Manually verify/clean labels on your 15 real scans (background / straight-cylinder / elbow, 3-class scheme — narrower than PSNet5's 5-class scheme, so you'll be remapping labels).

**Consensus checkpoint:** search `synthetic point cloud training industrial segmentation domain gap` — sim-to-real gap is a known failure mode; check whether recent work suggests specific augmentation strategies before you commit to your noise-injection design.

**Deliverable:** a combined real+synthetic dataset, PSNet-formatted, 3-class labeled, ready for fine-tuning.

---

## Phase 2 — Fine-Tune Segmentation Backbone (Weeks 3–6)

**Goal:** ResPointNet++ outputs per-point labels (background / straight pipe / elbow) on your pipe data.

- Adapt the model head: PSNet5 is 5-class (pipe, pump, tank, ibeam, rbeam); you need 3-class output. Check `models/heads/segmentation_head.py` and `models/build.py` for where the class count is configured.
- Implement Boundary-CB loss (not in the repo — it comes from the Industrial3D benchmark paper in your literature review). This is new code you write and integrate alongside the existing losses in `models/losses/`.
- Fine-tune from your Phase-0 PSNet5-trained checkpoint on your combined real+synthetic pipe dataset.
- Track: overall accuracy, mIoU, and **per-class recall for the elbow class specifically** — this is the rare class Boundary-CB loss is supposed to rescue, so measure it in isolation, not just averaged in.

**Consensus checkpoint:** search `class imbalance loss point cloud segmentation` to confirm Boundary-CB is still the strongest choice versus alternatives (focal loss variants, etc.) that may have appeared since your survey.

**Deliverable:** a fine-tuned checkpoint that segments background/pipe/elbow on held-out real scans, with a mIoU number you can defend.

---

## Phase 3 — Point Cloud Separation + Straight-Pipe RANSAC (Weeks 5–7, overlaps Phase 2)

**Goal:** stages 4–5 of your pipeline — go from labeled points to per-instance cylinder radius.

- Extract labeled points into `pipe_cylinder_points.ply` / `elbow_points.ply`, discard background.
- Implement DBSCAN clustering to separate multiple pipe instances in one scan (this is a new component — it's not in the ResPointNet2 repo, which is segmentation-only).
- Wire up the existing/validated C++/PCL RANSAC cylinder fitter on the straight-cylinder clusters, with the parameters you specified: 5mm distance threshold, 0.1 normal-distance weight, 3× iterations when estimated radius < 5cm.
- Build the Python↔C++ bridge via `.ply`/`.json` file I/O, since DL runs in PyTorch/Python and fitting runs in PCL/C++.

**Consensus checkpoint:** search `DBSCAN point cloud pipe instance separation` — confirm DBSCAN (vs. learned instance segmentation like SoftGroup/JSNet, which your own related-work doc flags as more robust for close/touching pipes) is still your best cost/complexity tradeoff for this stage. If cluttered multi-pipe scenes are a real test case, this is worth revisiting.

**Deliverable:** working straight-pipe radius output on multi-pipe scenes, with an MRE number against ground truth.

---

## Phase 4 — Torus-Fitting RANSAC for Elbows (Weeks 6–9)

This is your **flagged novel contribution** ("NOTE: This module requires implementation and validation") — treat it as the highest-risk item and start it early, in parallel with Phase 3, not after.

- Implement torus-fitting RANSAC: sample candidate (r, R) parameter pairs, score by inlier count within tolerance on the elbow point cluster, refine.
- Because there's no existing validated implementation to adapt (your own literature review confirms this — every paper either segments or fits, none combine + validate on elbows), plan for real dev/debug time and a synthetic-elbow test harness with known ground-truth (r, R) before testing on real scans.
- Validate against your synthetic elbow scans first (ground truth is exact), then real 45°/90° elbow scans.

**Consensus checkpoint — do this at the *start* of the phase, not the end:** re-run `torus fitting RANSAC point cloud` and `pipe elbow geometry reconstruction point cloud` on Consensus. If something has been published in the ~7 months since your proposal's survey window closed, you want to know before you spend three weeks building it, so you can either differentiate your approach or cite/build on theirs.

**Deliverable:** bend radius (R) and pipe cross-section radius (r) output for elbow segments, with an error percentage — this is the number that makes or breaks your novelty claim, so don't let it be a late addition.

---

## Phase 5 — Integration, Compliance Report, Qt UI (Weeks 9–11)

**Goal:** stage 7 — a usable end-to-end application, not just scripts.

- Wire stages 1–6 into one pipeline: scan in → segmentation → separation → straight/torus RANSAC → compliance report out.
- Compliance report generation: compare estimated radius/bend angle against nominal spec (isometric drawing or NPS table), compute `|estimated − nominal| / nominal × 100%`, PASS/FAIL at configurable tolerance (default ±5%).
- Update the Qt 5/6 + VTK interface to display point cloud, segmentation overlay, and the structured report (Pipe ID, GT radius, predicted radius, absolute/relative error, PASS/FAIL).
- Instrument end-to-end processing time per pipe against your <30s target.

**Deliverable:** the "working software system" deliverable from your proposal — a demoable application.

---

## Phase 6 — Evaluation, Ablation, Cross-Dataset Validation (Weeks 10–13, overlaps Phase 5)

**Goal:** produce every number your proposal promised.

- **Main results table:** MRE, small-pipe error (r<5cm), F1, mIoU, bend radius error, processing time — against the targets table in Section 6.1.
- **Ablation study:** RANSAC-only vs. DL-only vs. hybrid, run on the same test set, to isolate what the hybrid design actually buys you.
- **Cross-dataset validation:** SHREC 2022 and PipeNet Lab data, as promised — locate/request access to these early in this phase, not at the end, since external dataset access can be slow.
- **Sensor fusion ablation:** LiDAR vs. Kinect input, as described in Section 6.2.
- **Noise robustness check:** confirm MRE stays <5% up to 20% missing data, using your Phase-1 synthetic noise levels.

**Consensus checkpoint:** for each metric where you claim a target "based on comparable published results" (marked `*` in your proposal — PipeSegNet's 2.73%, Paper 1's F1=0.84), re-search Consensus to (a) confirm those are still the right benchmarks to compare against and (b) check whether newer 2026 work has since raised the bar, since your own table shows the field moving fast (Industrial3D benchmark, 2026).

**Deliverable:** the full experimental results package for the paper.

---

## Phase 7 — Manuscript + Dataset Release (Weeks 12–14, overlaps Phase 6)

**Goal:** submission-ready manuscript targeting *Automation in Construction* (Elsevier), plus public dataset release.

- Draft in IMRAD structure; your proposal's own literature table (Section 1.2) is already close to a related-work section skeleton — expand it with any new Consensus hits from Phases 4/6.
- Release the 200+ synthetic scans with ground-truth radius labels on GitHub/Zenodo, referencing this alongside the manuscript.
- **Final Consensus pass before submission:** re-run all five standing queries above one last time. A journal reviewer will do this search too; better you find a conflicting or superseding paper first.
- Have someone outside the project sanity-check that every number in the abstract traces to a specific table/figure in the results section (avoids the most common desk-reject reason: unsupported claims).

**Deliverable:** submitted manuscript + public dataset.

---

## Critical-path risks worth flagging now

| Risk | Why it matters | Mitigation |
|---|---|---|
| No pretrained ResPointNet++ checkpoint exists publicly | Your proposal assumes a pretrained backbone; you'll actually need to train it from scratch on PSNet5 first (Phase 0) | Budget real GPU time in week 1–2, not zero |
| Torus-fitting RANSAC has no reference implementation anywhere | It's your core novelty claim but also your biggest unknown | Start Phase 4 in parallel with Phase 3, not after; build a synthetic ground-truth harness first |
| DBSCAN instance separation may not hold up on touching/close pipes | Your own related-work review flags learned instance segmentation (SoftGroup etc.) as stronger for this case | Treat DBSCAN as a v1; keep a note in the paper's limitations if you don't have time to compare against a learned alternative |
| 14-week timeline is tight for training + a from-scratch geometric module + full evaluation + writing | Overlapping phases (as scheduled above) is the only way this fits | Start writing related-work/methods sections in Phase 4–5, not Phase 7 |

---

*This roadmap is a planning document derived from your uploaded proposal, the ResPointNet2 GitHub repository, and the attached Consensus literature summary. Re-verify GPU/dataset specifics against the live repo (`troubleshooting.md`, `yc/faq.md`) before committing to Phase 0 timelines, as tooling details can shift.*
