#!/usr/bin/env python
import argparse, json, time
from pathlib import Path
from playwright.sync_api import sync_playwright

DOC = """
Usage:
  python scripts/playwright_harness.py --url http://localhost:3000 --steps steps.json --out out_dir

Steps JSON format:
{
  "before_wait_ms": 500,
  "after_wait_ms": 800,
  "viewport": {"width": 1280, "height": 800},
  "actions": [
    {"type": "click", "selector": "button#login"},
    {"type": "fill",  "selector": "input#email", "value": "alice@example.com"},
    {"type": "press", "key": "Enter"}
  ]
}
"""
def run(url: str, steps_path: Path, out_dir: Path, headless: bool = True):
    steps = json.loads(Path(steps_path).read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    before_ms = int(steps.get("before_wait_ms", 400))
    after_ms = int(steps.get("after_wait_ms", 600))
    viewport = steps.get("viewport", {"width": 1280, "height": 800})

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(viewport=viewport)
        page = context.new_page()

        page.goto(url)
        time.sleep(before_ms/1000.0)
        page.screenshot(path=str(out_dir / "before.png"))

        for i, act in enumerate(steps.get("actions", []), start=1):
            t = act.get("type")
            if t == "click":
                page.click(act["selector"])
            elif t == "fill":
                page.fill(act["selector"], act.get("value", ""))
            elif t == "press":
                page.keyboard.press(act["key"])
            elif t == "wait":
                time.sleep(float(act.get("seconds", 0.5)))
            else:
                print(f"[warn] unknown action type: {t}")
            time.sleep(0.2)

        time.sleep(after_ms/1000.0)
        page.screenshot(path=str(out_dir / "after.png"))

        context.close()
        browser.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Target URL to test")
    ap.add_argument("--steps", required=True, help="Path to steps.json")
    ap.add_argument("--out", default="runs/run1", help="Output directory for screenshots")
    ap.add_argument("--headed", action="store_true", help="Run in headed mode")
    args = ap.parse_args()
    run(args.url, Path(args.steps), Path(args.out), headless=not args.headed)
