#!/usr/bin/env python
import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright
import json

# This script expects that you have the axe-core Playwright integration.
# Install: pip install playwright pytest-playwright
# Also run: playwright install

AXE_JS_URL = "https://cdnjs.cloudflare.com/ajax/libs/axe-core/4.4.3/axe.min.js"

def run_axe(url: str, out_path: Path):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        # Inject axe-core
        page.add_script_tag(url=AXE_JS_URL)
        # Run axe in page context
        results = page.evaluate("async () => { return await axe.run(); }")
        out_path.write_text(json.dumps(results, indent=2))
        browser.close()
        print(f"Axe results written to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Target URL to audit")
    ap.add_argument("--out", default="axe_results.json", help="Output JSON file")
    args = ap.parse_args()
    run_axe(args.url, Path(args.out))
