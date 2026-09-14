"""Verify local code + ablation, send summary to Claude via BrowserOS, save verdict."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._browseros import run_code


def run_local_checks() -> dict:
    env = {**dict(__import__("os").environ), "PYTHONPATH": str(ROOT / "src")}
    tests = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    smoke = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "smoke_test.py")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    ablation = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_academic_ablation.py")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    return {
        "tests_ok": tests.returncode == 0,
        "tests_tail": (tests.stdout + tests.stderr)[-800:],
        "smoke_ok": smoke.returncode == 0,
        "smoke_out": smoke.stdout.strip(),
        "ablation_ok": ablation.returncode == 0,
        "ablation_tail": (ablation.stdout + ablation.stderr)[-1200:],
    }


def build_prompt(checks: dict) -> str:
    ablation_md = (ROOT / "research_audit" / "ablation_results.md").read_text(encoding="utf-8")
    status_md = (ROOT / "IMPLEMENTATION_STATUS.md").read_text(encoding="utf-8")
    return f"""You are the senior reviewer for DeepSegregation. I re-ran local verification today (2026-09-14).

## Local execution (just ran on my machine)
- unittest: {"PASS (24 tests)" if checks["tests_ok"] else "FAIL"}
- smoke_test.py: {"PASS" if checks["smoke_ok"] else "FAIL"} — {checks["smoke_out"]}
- run_academic_ablation.py: {"PASS" if checks["ablation_ok"] else "FAIL"}

Test tail:
```
{checks["tests_tail"]}
```

Ablation tail:
```
{checks["ablation_tail"]}
```

## Current ablation table (information-equal three-way)
{ablation_md}

## IMPLEMENTATION_STATUS (repo truth)
{status_md}

## Your task
1. Confirm which P0/P1 research gaps from the synthesis are **closed in code** vs still **external-only** (PSNet5 download, GPU training, real elbow labels).
2. State whether the topological graph torus + directional DBSCAN + baselines are sufficient to pivot the paper narrative away from "novel torus RANSAC".
3. Give a **publication readiness score** (0–10) and the **minimum remaining experiments** before a defensible manuscript.
4. Output a structured **Compliance & Gap Closure Report** with sections: Executive Summary, Verified Claims, Remaining Gaps, Recommended Evaluation Protocol (2-class vs 3-class on PSNet5), Manuscript Title/Contribution bullets.

Be adversarial but fair. Do not ask follow-up questions — give a complete report in one message.
"""


def send_and_poll(prompt: str) -> Path:
    print("Opening Claude chat and sending verification report...")
    js_code = f"""
    const pageId = await browser.pages.newPage('https://claude.ai/new', {{ background: false }});
    await browser.wait(pageId, {{ value: 5000 }});
    const msg = {json.dumps(prompt)};

    await browser.evaluate(pageId, {{
        code: `(() => {{
            const el = document.querySelector('div[contenteditable="true"]');
            if (!el) return {{ ok: false, error: 'no contenteditable' }};
            el.focus();
            document.execCommand('selectAll', false, null);
            document.execCommand('insertText', false, ${{JSON.stringify(msg)}});
            return {{ ok: true, len: el.innerText.length }};
        }})()`
    }});

    await browser.wait(pageId, {{ value: 1500 }});
    let snap = await browser.observe(pageId).snapshot();
    let refs = Object.values(snap.refs || {{}});
    let sendBtn = refs.find(r => r.role === 'button' && (r.name || '').includes('Send message'));
    if (!sendBtn) return {{ error: 'Send button not found', pageId }};
    await browser.input(pageId).click(sendBtn.ref);
    return {{ ok: true, pageId }};
    """
    res = run_code(js_code, timeout=60)
    val = res.get("structuredContent", {}).get("value", res)
    if val.get("error"):
        raise RuntimeError(json.dumps(val, indent=2))
    page_id = val["pageId"]
    print(f"Sent. Polling page {page_id}...")

    for i in range(1, 40):
        time.sleep(10)
        poll = run_code(
            f"""
            const text = await browser.read({page_id});
            const responding = /Claude is responding|thinking|Stop response/i.test(text);
            return {{ responding, len: text.length }};
            """,
            timeout=120,
        )
        pv = poll.get("structuredContent", {}).get("value", {})
        print(f"  poll {i}: responding={pv.get('responding')} len={pv.get('len')}")
        if not pv.get("responding") and (pv.get("len") or 0) > 2500 and i >= 3:
            break

    out_dir = Path(r"C:\\Users\\shoun\\AppData\\Local\\BrowserClaw\\Application")
    read_files = []
    if out_dir.exists():
        for app in out_dir.iterdir():
            tool_out = app / ".browseros" / "tool-output"
            if tool_out.is_dir():
                read_files.extend(tool_out.glob("read-*.md"))

    dest = ROOT / "research_audit" / "claude_gap_closure_report.md"
    if read_files:
        latest = max(read_files, key=lambda p: p.stat().st_mtime)
        shutil.copy(latest, dest)
    else:
        full = run_code(f"return await browser.read({page_id});", timeout=120)
        text = full.get("structuredContent", {}).get("value", "")
        dest.write_text(text, encoding="utf-8")

    print(f"Saved Claude report to {dest} ({dest.stat().st_size} bytes)")
    return dest


def main() -> int:
    checks = run_local_checks()
    if not all(checks[k] for k in ("tests_ok", "smoke_ok", "ablation_ok")):
        print("WARNING: some local checks failed; still sending to Claude for diagnosis.")
    prompt = build_prompt(checks)
    (ROOT / "research_audit" / "verification_prompt_sent.md").write_text(prompt, encoding="utf-8")
    send_and_poll(prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
