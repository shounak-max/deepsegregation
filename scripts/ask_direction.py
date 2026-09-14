"""Send latest commits and strategic direction inquiry to Claude in BrowserOS neo."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._browseros import run_code


def get_git_info():
    log = subprocess.check_output(
        ["git", "log", "-n", "3", "--stat", "--pretty=format:Commit: %h - %s (%ad)"],
        text=True,
    )
    diff = subprocess.check_output(["git", "show", "008c7a1", "--stat", "-p"], text=True)
    if len(diff) > 3000:
        diff = diff[:3000] + "\n... [diff truncated for brevity] ..."
    return log, diff


def build_prompt():
    log, diff = get_git_info()
    prompt = f"""Here is an update on the DeepSegregation project following your comprehensive critique, along with our latest commits, status audit, and a critical question on how to achieve our targets.

### 1. Actions Taken on Your Recommendations (Commit 008c7a1):
- **Torus Basin-Consistency Check**: As you diagnosed, minimal-sample circle projections and subsequent least-squares optimization could degenerate into a swapped-radius basin (`major < minor`). We implemented explicit basin-consistency guards (`major > minor`) across all three fitting phases in `torus.py` (the initial circle projection filter, the 2D circle fit, and the 6-DOF least-squares refinement). Any hypothesis where `major <= minor` is immediately rejected.
- **Unit & Regression Testing**: In `test_torus.py`, we added explicit assertions checking `result.major_radius > result.minor_radius` and axis alignment `> 0.95`. All 14 unit tests across the repository pass cleanly (`tests/test_pipeline_components.py` and `tests/test_torus.py`).

```diff
{diff}
```

Recent commits:
{log}

---

### 2. Comprehensive Status Audit: Where We Stand vs. Targets
We conducted a full audit against our proposal roadmap (`Project_Roadmap.md`):
- **CPU Geometry & System Pipeline (Stages 1, 3, 4, 5, 6)**: **Complete & Verified**. Preprocessing (vectorized normal estimation via `np.linalg.eigh`, voxel downsampling), DBSCAN instance separation, straight-pipe cylinder RANSAC (with tie-breaking, rollback guarantees, and degeneracy guards), elbow torus RANSAC, and compliance PASS/FAIL reporting all work end-to-end.
- **Deep Learning Segmentation (Stage 2)**: **NOT Achieved (Critical Bottleneck)**:
  - We trained on real PSNet5 (80M points, 5 classes) for 200 epochs on a remote Tesla K80 GPU using `PointNet-4D-v4` (with relative-Z normalization, R=15m sphere sampling, focal loss).
  - **Results**: Train Loss: `0.0031` (converged/memorized), Val Loss: `0.9875`, **mIoU: 19.42%** (Target: **50%**), **Accuracy: 32.3%** (Target: **75%**).
  - Per-class IoU: `pump`: 37.5%, `rectangularbeam`: 19.7%, `tank`: 16.6%, `pipe`: 15.0%, `ibeam`: 8.2%.
  - `pipeline_status.json` confirms: `"all_targets_met": false`.
- **Contract & Architecture Mismatch**:
  - The proposal called for a **3-class contract** (`[background, straight_pipe, elbow]`) with Boundary-CB loss and a fine-tuned **ResPointNet++** backbone.
  - Due to the difficulty of compiling upstream custom CUDA ops (`grid_subsampling`, `pt_custom_ops`) on the K80 host, we fell back to PointMLP/PointNet-4D, and ran on the 5 raw PSNet5 classes instead of our 3-class contract.
  - Because `PointNet-4D` lacks hierarchical local neighborhood aggregation (set abstraction / ball query / k-NN grouping), it cannot distinguish cylindrical pipe surfaces from flat I-beam flanges from raw coordinates alone, hitting a ~20% mIoU ceiling.

---

### 3. Request for Strategic Direction: How Do We Achieve the Targets?
We want your concrete recommendation on the path forward to break through this barrier and reach the proposal targets (50%+ mIoU, 75%+ accuracy, robust pipe segregation, <5% radius estimation error):

1. **Backbone Strategy**:
   - Compiling the pinned upstream ResPointNet2 CUDA ops on our campus K80 cluster is fragile. Should we:
     - **Option A**: Implement a pure PyTorch hierarchical architecture with Set Abstraction (PointNet++ style using PyTorch-native ball-query or k-NN grouping) that runs out-of-the-box on the K80 without custom CUDA C++ extensions?
     - **Option B**: Persist in getting the upstream ResPointNet2 CUDA ops compiled on the cluster?
     - **Option C**: Another lightweight neighborhood-aware architecture suitable for the K80?
2. **Class Scheme & Dataset Formulation**:
   - Should we immediately switch remote training to the target **3-class contract** (`background`, `straight_pipe`, `elbow`) by combining PSNet5 non-pipe classes as background + synthetic annotated pipes/elbows, rather than trying to solve the 5-class industrial benchmark first?
3. **Actionable Blueprint**:
   - Please provide the concrete architecture, loss function (Boundary-CB / focal), and data pipeline modifications we should implement right now to hit the targets.
"""
    return prompt


def send_to_claude():
    prompt = build_prompt()
    print("Preparing to send direction inquiry to Claude in BrowserOS neo...")

    js_code = f"""
    const pages = await browser.pages.list();
    let claudePage = pages.find(p => p.url.includes('claude.ai/chat/55e9e427-c9df-4a98-909e-fe95641853d4'));
    if (!claudePage) {{
        claudePage = pages.find(p => p.url.includes('claude.ai'));
    }}
    if (!claudePage) return {{ error: 'No Claude tab found' }};

    const pageId = claudePage.pageId;
    const msg = {json.dumps(prompt)};

    // Insert text into ProseMirror
    await browser.evaluate(pageId, {{
        code: `(() => {{
            const el = document.querySelector('div[contenteditable="true"]');
            if (!el) return {{ ok: false, error: 'no contenteditable' }};
            el.focus();
            const dt = new DataTransfer();
            dt.setData('text/plain', ${{JSON.stringify(msg)}});
            const pasteEvent = new ClipboardEvent('paste', {{ clipboardData: dt, bubbles: true, cancelable: true }});
            el.dispatchEvent(pasteEvent);
            if (el.innerText.length < 50) {{
                document.execCommand('insertText', false, ${{JSON.stringify(msg)}});
            }}
            return {{ ok: true, len: el.innerText.length }};
        }})()`
    }});

    await browser.wait(pageId, {{ value: 2000 }});

    // Find and click Send
    let snap = await browser.observe(pageId).snapshot();
    let refs = Object.values(snap.refs || {{}});
    let sendBtn = refs.find(r => r.role === 'button' && (r.name || '').includes('Send message'));
    if (!sendBtn) {{
        return {{ error: 'Send button not found', buttons: refs.filter(r => r.role === 'button') }};
    }}

    await browser.input(pageId).click(sendBtn.ref);
    console.log("Message sent to Claude! Waiting for response to start and complete...");

    // Wait for response to begin and finish
    await browser.wait(pageId, {{ value: 12000 }});
    for (let i = 0; i < 15; i++) {{
        const checkText = await browser.read(pageId);
        if (!checkText.includes('Claude is responding') && !checkText.includes('running')) {{
            break;
        }}
        await browser.wait(pageId, {{ value: 8000 }});
    }}

    const fullText = await browser.read(pageId);
    return {{ ok: true, textLength: fullText.length, fullText: fullText }};
    """

    res = run_code(js_code, timeout=180)
    print("Execution finished.")

    out_file = Path("critiques") / "claude_direction_response.md"
    critique_text = ""
    if isinstance(res, dict):
        critique_text = res.get("fullText", str(res))
        import glob
        files = sorted(
            glob.glob(r'C:\Users\shoun\AppData\Local\BrowserClaw\Application\148.0.7988.97\.browseros\tool-output\*'),
            key=os.path.getmtime,
            reverse=True
        )
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
    print(f"Claude direction saved to {out_file} (length: {len(critique_text)})")
    return res


if __name__ == "__main__":
    send_to_claude()
