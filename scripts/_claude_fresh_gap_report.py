"""Fresh BrowserOS tab: ask Claude for gap-closure verdict after P1 fixes."""
import json
import shutil
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._browseros import run_code

PROMPT = """DeepSegregation gap-closure check (2026-09-14). I just re-ran: 24/24 tests OK, smoke_test OK, ablation OK.

Key new evidence since your Round-2 review:
- Information-equal ablation: Prior-Init equals Unconstrained at 5mm+30% occl (1.55% bend err); Topological=0.03% — isolates the C1 constraint.
- topology.py: ElbowGeometryError for shallow angles; single-cylinder fallback; inlier_rmse not tautological tangent metric.
- Directional DBSCAN + clustering sweep (50 deg Pareto).

Still external: PSNet5 has no elbow GT (Elbow IoU=0 on real runs); full GPU training.

Give structured report: Executive Summary, Gaps Closed vs Open, Publication Readiness /10, Minimum Remaining Experiments, Evaluation Protocol (2-class vs 3-class), Manuscript bullets. Adversarial, complete in one reply."""

js = f"""
const pageId = await browser.pages.newPage('https://claude.ai/new', {{ background: false }});
await browser.wait(pageId, {{ value: 6000 }});
const msg = {json.dumps(PROMPT)};
await browser.evaluate(pageId, {{
  code: `(() => {{
    const el = document.querySelector('div[contenteditable="true"]');
    if (!el) return {{ ok: false }};
    el.focus();
    document.execCommand('insertText', false, ${{JSON.stringify(msg)}});
    return {{ ok: true, len: el.innerText.length }};
  }})()`
}});
await browser.wait(pageId, {{ value: 800 }});
let snap = await browser.observe(pageId).snapshot();
let sendBtn = Object.values(snap.refs||{{}}).find(r => r.role==='button' && (r.name||'').includes('Send message'));
if (!sendBtn) {{
  const ev = await browser.evaluate(pageId, {{
    code: `(() => {{
      const btn = Array.from(document.querySelectorAll('button')).find(b => (b.getAttribute('aria-label')||'').includes('Send message'));
      if (btn && !btn.disabled) {{ btn.click(); return {{ clicked: true }}; }}
      return {{ clicked: false, disabled: btn?.disabled }};
    }})()`
  }});
  return {{ pageId, sendBtn: null, ev }};
}}
await browser.input(pageId).click(sendBtn.ref);
return {{ pageId, sent: true }};
"""

res = run_code(js, timeout=90)
print("send:", json.dumps(res, indent=2)[:2000])
page_id = res.get("structuredContent", {}).get("value", {}).get("pageId")
if not page_id:
    sys.exit(1)

for i in range(30):
    time.sleep(12)
    poll = run_code(
        f"""
        const text = await browser.read({page_id});
        const responding = /Claude is responding|Stop response|thinking/i.test(text);
        const got = /Executive Summary|Publication Readiness|readiness/i.test(text) && text.includes('Claude responded');
        return {{ responding, got, len: text.length }};
        """,
        timeout=120,
    )
    v = poll.get("structuredContent", {}).get("value", {})
    print("poll", i, v)
    if v.get("got") and not v.get("responding"):
        break

tool_out = Path(r"C:\Users\shoun\AppData\Local\BrowserClaw\Application\148.0.7988.97\.browseros\tool-output")
files = sorted(tool_out.glob("read-*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
dest = ROOT / "research_audit" / "claude_gap_closure_report.md"
if files:
    shutil.copy(files[0], dest)
    print("saved", dest, dest.stat().st_size)
