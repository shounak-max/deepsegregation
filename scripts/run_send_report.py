"""Open new Claude page in session and send report."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._browseros import run_code
from scripts.send_pipeline_report import build_pipeline_report


def main():
    prompt = build_pipeline_report()
    print("Opening new page and sending report to Claude...")

    js_code = """
    const pageId = await browser.pages.newPage('https://claude.ai/chat/55e9e427-c9df-4a98-909e-fe95641853d4', { background: false });
    await browser.wait(pageId, { value: 6000 });

    const msg = """ + json.dumps(prompt) + """;
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
    if (!sendBtn) return { error: 'Send button not found', pageId, refs: refs.filter(r => r.role === 'button').map(r => r.name) };

    await browser.input(pageId).click(sendBtn.ref);
    return { ok: true, pageId, sent: true };
    """

    res = run_code(js_code, timeout=45)
    print("Result:", json.dumps(res, indent=2))
    
    data = res.get("structuredContent", {}).get("value", {})
    if data.get("sent"):
        page_id = data["pageId"]
        print(f"Report sent on page {page_id}! Polling for Claude response...")
        for i in range(1, 25):
            time.sleep(8)
            poll_res = run_code(f"""
            const text = await browser.read({page_id});
            const isResponding = text.includes('Claude is responding') || text.includes('thinking');
            return {{ isResponding, len: text.length }};
            """)
            poll_val = poll_res.get("structuredContent", {}).get("value", {})
            is_resp = poll_val.get("isResponding")
            length = poll_val.get("len", 0)
            print(f"Poll {i}: isResponding={is_resp}, length={length}")
            if not is_resp and i >= 2 and length > 1000:
                print("Claude finished generating response!")
                break
        
        # Save output
        save_res = run_code(f"return await browser.read({page_id});")
        full_text = save_res.get("structuredContent", {}).get("value", "")
        import glob
        files = sorted(
            glob.glob(r'C:\Users\shoun\AppData\Local\BrowserClaw\Application\148.0.7988.97\.browseros\tool-output\read-*.md'),
            key=lambda p: Path(p).stat().st_mtime,
            reverse=True
        )
        out_file = Path("critiques") / "claude_pipeline_feedback.md"
        if files:
            import shutil
            shutil.copy(files[0], out_file)
            print(f"Copied full feedback to {out_file} ({out_file.stat().st_size} bytes)")
        else:
            out_file.write_text(full_text, encoding="utf-8")
            print(f"Saved feedback to {out_file} ({len(full_text)} chars)")


if __name__ == "__main__":
    main()
