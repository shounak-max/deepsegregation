"""Send resolution of critique and code of torus.py & dataset.py to Claude."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._browseros import run_code


def load_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    torus_code = load_file("src/deepsegregation/torus.py")
    dataset_code = load_file("src/deepsegregation/dataset.py")

    message = f"""I have acted on all your critique points and committed the fixes to main (commit f952d65):

### Actions Taken from Your Critique:
1. **`cylinder.py` Refinement Degeneracy Guard & Convergence**:
   - Added early convergence check (`abs(new_radius - radius) < 1e-6 and hypot(new_cx - cx, new_cy - cy) < 1e-6`).
   - Guarded against degenerate/empty masks (`int(mask.sum()) < 3` breaks immediately instead of fitting to all points).
   - Rollback guarantee: If the 4-pass refinement yields fewer inliers than the initial RANSAC consensus set, the function rolls back to the RANSAC hypothesis.
2. **`cylinder.py` Axis Orientation Tie-Breaking**:
   - Replaced fragile argmax component check with tolerance-based tie-breaking (`np.isclose(np.abs(axis), np.max(np.abs(axis)), atol=1e-2)`), ensuring stable deterministic orientation across 45-degree pipes.
3. **`cylinder.py` Inlier Ratio Configuration**:
   - Added `min_inlier_ratio: float = 0.3` to allow noisy clouds with up to 70% outliers to pass by default.
4. **`preprocessing.py` Vectorized `estimate_normals`**:
   - Replaced the $O(N)$ per-point Python loop and individual SVDs with batch covariance matrix multiplication (`(N, k, 3)`) and `np.linalg.eigh`, reducing runtime on 500k-point clouds from minutes to seconds.
5. **`preprocessing.py` Statistical Outliers Variance Floor**:
   - Added `min_std: float = 1e-6` floor in `remove_statistical_outliers`.
6. **Unit Tests**: All 13 unit tests passing.

---

### Files for Review (`torus.py` and `dataset.py`):
As requested, here are `torus.py` and `dataset.py`:

#### `src/deepsegregation/torus.py`:
```python
{torus_code}
```

#### `src/deepsegregation/dataset.py`:
```python
{dataset_code}
```

Please critique `torus.py` and `dataset.py`, and let me know if there are any remaining bugs, numerical pitfalls, or flaws to address!
"""

    print("Posting update to Claude in BrowserOS neo...")
    js_code = """
    const pages = await browser.pages.list();
    const claudePage = pages.find(p => p.url.includes('claude.ai'));
    if (!claudePage) return { error: 'Claude tab not found' };
    const pageId = claudePage.pageId;
    const msg = """ + json.dumps(message) + """;

    await browser.evaluate(pageId, {
        code: `(() => {
            const el = document.querySelector('div[contenteditable="true"]');
            if (!el) return { ok: false, error: 'no contenteditable' };
            el.focus();
            const dt = new DataTransfer();
            dt.setData('text/plain', ${JSON.stringify(msg)});
            const pasteEvent = new ClipboardEvent('paste', { clipboardData: dt, bubbles: true, cancelable: true });
            el.dispatchEvent(pasteEvent);
            if (el.innerText.length < 50) {
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
    await browser.wait(pageId, { value: 6000 });

    return { ok: true, sent: true };
    """

    res = run_code(js_code, timeout=45)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
