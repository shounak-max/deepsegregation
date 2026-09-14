import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._browseros import run_code


def main():
    page_id = 15
    print("Evaluating send click on page", page_id)
    js_code = f"""
    const result = await browser.evaluate({page_id}, {{
        code: `(() => {{
            const btns = Array.from(document.querySelectorAll('button'));
            const sendBtn = btns.find(b => (b.getAttribute('aria-label') || '').toLowerCase().includes('send') || (b.innerText || '').toLowerCase().includes('send'));
            if (sendBtn) {{
                sendBtn.focus();
                sendBtn.click();
                return {{ ok: true, found: 'sendBtn', text: sendBtn.innerText, aria: sendBtn.getAttribute('aria-label'), disabled: sendBtn.disabled }};
            }}
            return {{ ok: false, allBtns: btns.map(b => b.getAttribute('aria-label') || b.innerText).filter(Boolean).slice(-10) }};
        }})()`
    }});
    return result;
    """
    res = run_code(js_code)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
