#!/usr/bin/env python
import asyncio, sys
from playwright.async_api import async_playwright

async def run_patch_test(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url)
        # Example: check if login button is clickable
        try:
            await page.click("button#login")
            print("[PASS] Login button clickable")
        except Exception as e:
            print(f"[FAIL] Could not click login button: {e}")
        await browser.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: playwright_test.py <URL>")
        sys.exit(1)
    asyncio.run(run_patch_test(sys.argv[1]))
