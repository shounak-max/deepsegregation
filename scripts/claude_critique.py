"""Automated critique runner that queries Claude via BrowserOS neo about recent commits."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._browseros import run_code


def get_git_summary():
    # Get last 2 commits and stat
    log = subprocess.check_output(
        ["git", "log", "-n", "3", "--stat", "--pretty=format:Commit: %h - %s (%ad)"],
        text=True,
    )
    # Get git diff for last commit
    diff = subprocess.check_output(["git", "show", "HEAD", "--stat", "-p"], text=True)
    if len(diff) > 4000:
        diff = diff[:4000] + "\n... [diff truncated for length] ..."
    return f"### Recent Git Commits:\n{log}\n\n### Latest Commit Diff:\n```diff\n{diff}\n```"


def ask_claude_for_critique():
    git_info = get_git_summary()
    prompt = f"""Here is what was committed to the repository:

{git_info}

Please critique these changes and codebase state thoroughly:
1. Are there any subtle bugs, edge cases, numerical instability, or boundary condition errors?
2. What are the flaws or correctness gaps in the geometry fitting, data pipeline, or model training?
3. How can the implementation and architecture be improved further?
"""

    print("Sending critique request to Claude in BrowserOS neo...")
    js_code = f"""
    // Find Claude tab
    const pages = await browser.pages.list();
    const claudePage = pages.find(p => p.url.includes('claude.ai'));
    if (!claudePage) {{
        return {{ error: 'Claude tab not found', pages }};
    }}
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

    await browser.wait(pageId, {{ value: 1500 }});

    // Find and click Send
    let snap = await browser.observe(pageId).snapshot();
    let refs = Object.values(snap.refs || {{}});
    let sendBtn = refs.find(r => r.role === 'button' && (r.name || '').includes('Send message'));
    if (!sendBtn) {{
        return {{ error: 'Send button not found', buttons: refs.filter(r => r.role === 'button') }};
    }}

    await browser.input(pageId).click(sendBtn.ref);
    await browser.wait(pageId, {{ value: 8000 }});

    // Read response
    const text = await browser.read(pageId);
    return {{ ok: true, textLength: text.length, tail: text.slice(-3000) }};
    """

    res = run_code(js_code, timeout=60)
    print("Claude critique response received.")
    return res


if __name__ == "__main__":
    res = ask_claude_for_critique()
    print(json.dumps(res, indent=2))
