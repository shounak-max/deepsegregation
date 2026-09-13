"""Wait for Claude's response to complete and save to critiques/data_and_results_analysis.md."""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._browseros import run_code

def fetch_analysis():
    print("Waiting for Claude to finish generating analysis...")
    for i in range(25):
        time.sleep(5)
        text = run_code("const pages = await browser.pages.list(); const p = pages.find(x => x.url.includes('claude.ai')); return await browser.read(p.pageId);", timeout=30)
        if isinstance(text, str):
            if "Claude is responding" not in text and "Running a command" not in text and "running" not in text[-500:].lower():
                print(f"Claude finished responding at loop {i}!")
                break
            else:
                print(f"Still running (loop {i}), len: {len(text)}...")
        else:
            print(f"Loop {i}: {text}")

    res = {"fullText": text if isinstance(text, str) else ""}
    out_file = Path("critiques/data_and_results_analysis.md")
    
    critique_text = res.get("fullText", "")
    import glob
    files = sorted(glob.glob(r'C:\Users\shoun\AppData\Local\BrowserClaw\Application\148.0.7988.97\.browseros\tool-output\*'), key=os.path.getmtime, reverse=True)
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
    print(f"Updated {out_file}")

if __name__ == "__main__":
    fetch_analysis()
