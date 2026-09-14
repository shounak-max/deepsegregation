import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._browseros import run_code


def main():
    page_id = 15
    print("Waiting for Claude to finish streaming response...")
    for i in range(1, 40):
        time.sleep(6)
        res = run_code(f"""
        const snap = await browser.observe({page_id}).snapshot();
        const refs = Object.values(snap.refs || {{}});
        const isStreaming = refs.some(r => (r.name || '').includes('Stop response')) || snap.text.includes('Currently streaming message');
        return {{ isStreaming, textLen: snap.text.length }};
        """)
        data = res.get("structuredContent", {}).get("value", {})
        is_stream = data.get("isStreaming")
        print(f"Check {i}: isStreaming={is_stream}")
        if not is_stream and i >= 2:
            print("Response finished streaming!")
            break

    # Read page markdown
    read_res = run_code(f"return await browser.read({page_id});")
    import glob, shutil
    files = sorted(
        glob.glob(r'C:\Users\shoun\AppData\Local\BrowserClaw\Application\148.0.7988.97\.browseros\tool-output\read-*.md'),
        key=lambda p: Path(p).stat().st_mtime,
        reverse=True
    )
    dest = Path("critiques/claude_pipeline_feedback.md")
    if files:
        shutil.copy(files[0], dest)
        print(f"Copied final markdown to {dest} ({dest.stat().st_size} bytes)")
    else:
        text = read_res.get("structuredContent", {}).get("value", "")
        dest.write_text(text, encoding="utf-8")
        print(f"Wrote to {dest}")


if __name__ == "__main__":
    main()
