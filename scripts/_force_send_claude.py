import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._browseros import run_code

js = """
const pageId = 34;
const snap = await browser.observe(pageId).snapshot();
const refs = Object.values(snap.refs || {});
const sendBtn = refs.find(r => r.role === 'button' && (r.name || '').includes('Send message'));
if (sendBtn) {
  await browser.input(pageId).click(sendBtn.ref);
  return { method: 'snapshot', sendBtn: sendBtn.name, disabled: sendBtn.disabled };
}
const evalRes = await browser.evaluate(pageId, {
  code: `(() => {
    const el = document.querySelector('div[contenteditable="true"]');
    if (el) {
      el.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', ctrlKey: true, bubbles: true }));
    }
    const btn = Array.from(document.querySelectorAll('button')).find(b =>
      (b.getAttribute('aria-label')||'').toLowerCase().includes('send'));
    if (btn) { btn.click(); return { ok: true, aria: btn.getAttribute('aria-label'), disabled: btn.disabled }; }
    return { ok: false };
  })()`
});
await browser.wait(pageId, { value: 3000 });
const snap2 = await browser.observe(pageId).snapshot();
const responding = snap2.text.includes('Claude is responding') || snap2.text.includes('Stop response');
return { evalRes, responding, textHead: snap2.text.slice(0, 400) };
"""
print(json.dumps(run_code(js, timeout=45), indent=2)[:6000])
