"""Retry send on page 34 with condensed prompt."""
import json
import shutil
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._browseros import run_code

PROMPT = """Senior reviewer task — DeepSegregation gap closure (2026-09-14).

Verified locally today:
- 24/24 unittest OK; smoke_test 2300 pts, 2 instances, 2 fits, 2 compliance reports
- Ablation: at 5mm noise + 30% occlusion, unconstrained bend err 1.55% vs topological 0.03% (prior-init equals unconstrained)
- P0/P1 in repo: topology.py graph torus, directional DBSCAN, baselines three-way ablation, clustering sweep doc

Still external: PSNet5 elbow labels (IoU=0 for class 2), full ResPointNet++ training, 15 real scans.

Reply with ONE structured report:
1) P0/P1 gaps closed vs open
2) Is narrative pivot away from novel torus RANSAC sufficient?
3) Publication readiness 0-10 + minimum remaining experiments
4) Sections: Executive Summary, Verified Claims, Remaining Gaps, Evaluation Protocol (2-class vs 3-class PSNet5), Manuscript contribution bullets

Adversarial, no follow-up questions."""

page_id = 34

js = f"""
const pageId = {page_id};
await browser.evaluate(pageId, {{
  code: `(() => {{
    const el = document.querySelector('div[contenteditable="true"]');
    if (!el) return {{ ok: false }};
    el.focus();
    el.innerText = '';
    document.execCommand('insertText', false, {json.dumps(PROMPT)});
    return {{ ok: true, len: el.innerText.length }};
  }})()`
}});
await browser.wait(pageId, {{ value: 1000 }});
const clickRes = await browser.evaluate(pageId, {{
  code: `(() => {{
    const btns = Array.from(document.querySelectorAll('button'));
    const sendBtn = btns.find(b => (b.getAttribute('aria-label')||'').toLowerCase().includes('send message'));
    if (!sendBtn) return {{ ok: false, labels: btns.map(b=>b.getAttribute('aria-label')).filter(Boolean).slice(-8) }};
    sendBtn.click();
    return {{ ok: true, disabled: sendBtn.disabled }};
  }})()`
}});
return clickRes;
"""

print(run_code(js, timeout=45))
for i in range(25):
    time.sleep(12)
    poll = run_code(
        f"""
        const text = await browser.read({page_id});
        const responding = /Claude is responding|Stop response|thinking/i.test(text);
        const got = /Publication readiness|Executive Summary|Gap Closure|readiness score/i.test(text);
        return {{ responding, got, len: text.length }};
        """,
        timeout=90,
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
