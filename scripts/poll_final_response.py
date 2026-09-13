import time
import sys
import shutil
import glob
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._browseros import run_code

def main():
    print("Polling Claude until generation completes...")
    for i in range(20):
        time.sleep(5)
        js = "const pages = await browser.pages.list(); const p = pages.find(x => x.url.includes('claude.ai')); return await browser.read(p.pageId);"
        t = run_code(js, timeout=30)
        if isinstance(t, str):
            if 'Claude is responding' not in t and 'Running a command' not in t and 'running' not in t[-500:].lower():
                print(f"Done generating at loop {i}!")
                files = sorted(glob.glob(r'C:\Users\shoun\AppData\Local\BrowserClaw\Application\148.0.7988.97\.browseros\tool-output\*'), key=os.path.getmtime, reverse=True)
                if files:
                    shutil.copy(files[0], r'critiques/data_and_results_analysis.md')
                    print(f"Copied {files[0]} to critiques/data_and_results_analysis.md")
                break
            else:
                print(f"Still generating (loop {i}), len: {len(t)}...")
        else:
            print(f"Loop {i}: {t}")

if __name__ == "__main__":
    main()
