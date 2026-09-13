"""Script to ask Claude to verify the latest code fixes and analyze the PSNet5 data and training results."""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._browseros import run_code

PROMPT = """Here is an update on the codebase and the training results. Please review and provide a thorough analysis:

### 1. Codebase Verification: Torus Axis Fix Applied (Commit 0c9e958)
Following your exact stress-test diagnosis on `torus.py`:
- **Root cause addressed**: In `fit_torus_ransac`, raw PCA on the 45%-contaminated cloud or 70% subsets produced an axis skewed by ~24 degrees, causing the minimal-sample search and refinement to fail (9/10 failure rate).
- **Fix implemented**:
  1. We pre-filter the point cloud using `remove_statistical_outliers(cloud, neighbors=16, z_threshold=1.5)` before initial PCA. On your 45%-outlier test fixture, filtered PCA alignment jumped from 0.91 to **0.9988** (99.88% alignment with true axis).
  2. Increased trial budget to `trials_per_axis = max(200, max_trials // len(axes))` and default `max_trials = 1200`.
  3. Added `min_inlier_ratio: float = 0.3` parameter matching `cylinder.py`.
  4. Added regression test `test_recovers_synthetic_elbow_with_heavy_45_percent_outliers_without_axis`.
  5. All 14 unit tests pass cleanly.

Does everything in `torus.py` and the geometry fitting pipeline look solid and complete now?

---

### 2. Request for In-Depth Data & Results Analysis:
We want your analysis of the dataset characteristics and the 200-epoch training results obtained:

#### A. Dataset Characteristics (PSNet5 Real Industrial Cloud, ~80M points):
- **Spatial Coordinates**: Area_1, 3, 4 are centered locally (X, Y ~ 0-80m, Z ~ 0-25m). Area_2 uses UTM-like coordinates (X ~ 2717-2741m, Z ~ 78-107m). We resolved this by computing local relative-Z within each sampled sphere ($z_{rel} = (z - z_{min}) / \max(\Delta z, 1.0)$).
- **Class Distribution**: Heavy class imbalance across the ~80M points:
  - `pipe`: ~74.9% in Area_1
  - `pump`: ~1.2%
  - `ibeam`: ~4.8%
  - `tank`: ~8.5%
  - `rectangularbeam`: ~10.6%
- **Spatial Sampling**: Points are sampled via spatial spheres with radius $R=15\\text{m}$ (subsampled to 500k points for KDTree search) to guarantee batches contain multiple classes (avg 4.6 classes per 15m sphere vs only 1-2 classes with $R=2\\text{m}$).

#### B. Final Training Results (PointNet-4D-v4, Tesla K80 GPU, 200 Epochs, 42.6 min):
- **Train Loss**: 0.0031 (smooth convergence from 0.0436)
- **Val Loss (Area_3)**: 0.9875 (peaked around 1.14, settled under 1.0)
- **mIoU**: 19.42% (best reached at epoch 200)
- **Overall Accuracy**: 32.3% (peak 35.4% at ep130)
- **Per-Class IoU Breakdown**:
  - `pump`: **37.5%**
  - `rectangularbeam`: **19.7%**
  - `tank`: **16.6%**
  - `pipe`: **15.0%**
  - `ibeam`: **8.2%**
- **Loss Function**: Class-balanced Focal Loss ($\\gamma=2.0$, effective sample weights $\\beta=0.9999$).

#### Please provide your detailed analysis:
1. Is everything in the geometry fitting and dataset pipeline now robust and complete?
2. What are the key takeaways from the disparity between train loss (0.0031) and val loss (0.9875)?
3. Why did `pump` achieve 37.5% while `pipe` (majority class) only achieved 15.0% and `ibeam` 8.2%?
4. What concrete architectural improvements, data augmentations (e.g. random rotation, jitter), or sampling strategies should we deploy next to break through the 20% mIoU barrier toward the 50% target?
"""


def ask_claude():
    print("Delivering comprehensive analysis prompt to Claude in BrowserOS neo...")
    js_code = """
    const pages = await browser.pages.list();
    const claudePage = pages.find(p => p.url.includes('claude.ai'));
    if (!claudePage) return { error: 'Claude tab not found' };
    const pageId = claudePage.pageId;
    const msg = """ + json.dumps(PROMPT) + """;

    await browser.evaluate(pageId, {
        code: `(() => {
            const el = document.querySelector('div[contenteditable="true"]');
            if (!el) return { ok: false, error: 'no contenteditable' };
            el.focus();
            const dt = new DataTransfer();
            dt.setData('text/plain', ${JSON.stringify(msg)});
            const pasteEvent = new ClipboardEvent('paste', { clipboardData: dt, bubbles: true, cancelable: true });
            el.dispatchEvent(pasteEvent);
            if (el.innerText.length < 100) {
                document.execCommand('insertText', false, ${JSON.stringify(msg)});
            }
            return { ok: true, len: el.innerText.length };
        })()`
    });

    await browser.wait(pageId, { value: 2000 });

    let snap = await browser.observe(pageId).snapshot();
    let refs = Object.values(snap.refs || {});
    let sendBtn = refs.find(r => r.role === 'button' && (r.name || '').includes('Send message'));
    if (!sendBtn) {
        return { error: 'Send button not found', buttons: refs.filter(r => r.role === 'button') };
    }

    await browser.input(pageId).click(sendBtn.ref);
    console.log("Message sent, waiting for Claude's response...");

    // Wait for response to generate and complete
    await browser.wait(pageId, { value: 12000 });
    for (let i = 0; i < 8; i++) {
        const checkText = await browser.read(pageId);
        if (!checkText.includes('Claude is responding') && !checkText.includes('running')) {
            break;
        }
        await browser.wait(pageId, { value: 6000 });
    }

    const text = await browser.read(pageId);
    return { ok: true, textLength: text.length, fullText: text };
    """

    res = run_code(js_code, timeout=120)
    print("Sent prompt to Claude, result received.")

    critique_dir = Path("critiques")
    critique_dir.mkdir(exist_ok=True)
    out_file = critique_dir / "data_and_results_analysis.md"

    critique_text = ""
    if isinstance(res, dict):
        critique_text = res.get("fullText", str(res))
        import glob
        files = sorted(glob.glob(r'C:\Users\shoun\AppData\Local\BrowserClaw\Application\148.0.7988.97\.browseros\tool-output\*'), key=os.path.getmtime, reverse=True)
        if files:
            try:
                with open(files[0], 'r', encoding='utf-8', errors='ignore') as f:
                    raw = f.read()
                    if len(raw) > len(critique_text):
                        critique_text = raw
            except Exception:
                pass

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(critique_text)
    print(f"Analysis saved to {out_file}")
    return res


if __name__ == "__main__":
    ask_claude()

