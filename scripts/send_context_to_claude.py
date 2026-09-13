"""Paste context message cleanly into Claude's ProseMirror editor and send."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._browseros import run_code

CONTEXT_MESSAGE = """Yes, exactly — that's the pipe-parameter-estimation / DeepSegregation point-cloud pipeline. Here is the full context and detailed progress update:

### 1. Architecture & Pipeline Components
- **DBSCAN Instance Separation** (`clustering.py`): separates individual pipe and elbow instances.
- **RANSAC Geometry Fitting** (`cylinder.py`, `torus.py`): fits cylinder geometries (straight pipes) and torus geometries (pipe elbows) with least-squares refinement.
- **Compliance Reporting** (`report.py`): validates dimensional tolerances against ASME/design specs.
- **ResPointNet++ Backbone** (`inference.py`): 5-class semantic segmentation head (`ibeam`, `pipe`, `pump`, `rectangularbeam`, `tank`).

---

### 2. Five Confirmed Bugs & Correctness Gaps Fixed (Commit 63a66a3)
1. 🔴 **Critical Cylinder RANSAC Refinement Discard Bug (`cylinder.py`)**:
   - **Root Cause**: In `fit_cylinder_ransac`, `mask` was initialized to all False (`np.zeros(len(values), dtype=bool)`), so `mask.any() == False` on pass 1. This caused the very first refinement iteration to run least-squares on all points including outliers, discarding the consensus model. On 45% outlier data, radius degraded from 0.0499 (301 inliers) to 0.2503 (5 inliers).
   - **Fix**: Seeded `mask = distances <= distance_threshold` from the RANSAC hypothesis before the refinement loop. After fix, accurately recovers r = 0.05019 with 301 inliers.
2. 🟡 **Lexical Directory Ordering (`dataset.py`)**:
   - Fixed `iter_area_samples` to sort `Area_N` numerically so `Area_10` does not sort before `Area_2`.
3. 🟡 **Dropped Point Cloud Colors (`preprocessing.py`)**:
   - In `voxel_downsample`, added voxel-aggregated RGB averaging so `cloud.colors` is preserved with correct dtypes.
4. 🟡 **Trial Budget vs Winning Trial (`cylinder.py`, `torus.py`)**:
   - Added `best_trial: int` to `CylinderFit` and `TorusFit` tracking the winning trial index, while keeping `iterations` for total budget.
5. 🟡 **Inference Device Fallback (`inference.py`)**:
   - Replaced hardcoded `device="cuda"` with `"cuda" if torch.cuda.is_available() else "cpu"` with CPU fallback.
6. ✅ All 11 unit tests passing (`python -m unittest discover tests`).

---

### 3. Remote GPU Training Run Completed (PointNet-4D-v4 on Real PSNet5)
- **Dataset**: Real PSNet5 industrial dataset (~80M points across Area 1-4; train on Area 1/2/4, validate on Area 3).
- **Environment**: Detached training on Tesla K80 GPU (`cuda:1`, PyTorch 1.10.2 + CUDA 11.3, scipy KDTree).
- **Configuration**: PointNet with 4D features (normalized relative-Z), spatial radius sampling ($R=15m$, 500k subsample), and class-balanced Focal Loss ($\gamma=2.0$).
- **Final Results (Full 200/200 Epochs, 42.6 min)**:
  - **Status**: COMPLETED
  - **mIoU**: **19.42%** (best at ep200)
  - **Accuracy**: **32.3%** (peak 35.4% at ep130)
  - **Validation Loss**: **0.9875** (Train loss: 0.0031)
  - **Per-Class IoU**: `pump`: 37.5%, `rectangularbeam`: 19.7%, `tank`: 16.6%, `pipe`: 15.0%, `ibeam`: 8.2% (all 5 classes actively learned).
- **Checkpoints**: All saved under `~/deepsegregation/checkpoints/pointnet_v4/` with `best.pt`.
"""


def main():
    print("Sending full context to Claude...")
    js_code = """
    const pageId = 5;
    const msg = """ + json.dumps(CONTEXT_MESSAGE) + """;

    // Use evaluate to paste text into contenteditable
    const pasteRes = await browser.evaluate(pageId, {
        code: `(() => {
            const el = document.querySelector('div[contenteditable="true"]');
            if (!el) return { ok: false, error: 'no contenteditable' };
            el.focus();
            
            // Try paste event with DataTransfer
            const dt = new DataTransfer();
            dt.setData('text/plain', ${JSON.stringify(msg)});
            const pasteEvent = new ClipboardEvent('paste', {
                clipboardData: dt,
                bubbles: true,
                cancelable: true
            });
            const dispatched = el.dispatchEvent(pasteEvent);
            
            // If text is still short, fallback to execCommand insertText
            if (el.innerText.length < 100) {
                document.execCommand('insertText', false, ${JSON.stringify(msg)});
            }
            
            return {
                ok: true,
                innerTextLen: el.innerText.length,
                sample: el.innerText.slice(0, 120)
            };
        })()`
    });

    await browser.wait(pageId, { value: 1500 });

    // Snapshot to find Send button
    let snap = await browser.observe(pageId).snapshot();
    let refs = Object.values(snap.refs || {});
    let sendBtn = refs.find(r => r.role === 'button' && (r.name || '').includes('Send message'));

    if (!sendBtn) {
        return { error: 'Send button not found', pasteRes, buttons: refs.filter(r => r.role === 'button') };
    }

    // Click Send
    await browser.input(pageId).click(sendBtn.ref);
    await browser.wait(pageId, { value: 4000 });

    // Read latest page state
    const afterSnap = await browser.observe(pageId).snapshot();
    return {
        ok: true,
        pasteRes,
        sent: true,
        textSlice: afterSnap.text.slice(afterSnap.text.indexOf('pipe-parameter-estimation'), afterSnap.text.indexOf('pipe-parameter-estimation') + 1000)
    };
    """
    res = run_code(js_code, timeout=45)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
