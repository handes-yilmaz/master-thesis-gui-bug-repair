## Quickstart

1. Create env (conda recommended):
   ```bash
   conda create -n gui-bug-repair python=3.11 -y
   conda activate gui-bug-repair
   pip install -r requirements.txt
   ```
2. Open notebooks:
   ```bash
   jupyter lab
   ```
3. Use `configs/config.example.json` to set paths. Copy it to `configs/config.json` and edit.

## Folders

- `data/raw` — put the SWE-bench MM raw files here.
- `data/processed` — outputs from preprocessing and labeling.
- `notebooks` — analysis notebooks for each day’s tasks.
- `scripts` — CLI utilities used by notebooks.
- `docs` — keep paper PDFs, notes, and data schema here.
- `configs` — configuration JSON.

## Notes
- API calls are **stubbed**. Insert your keys where indicated if you plan to call an external LLM.
- OCR via Tesseract expects the binary installed on your system. If missing, OCR steps will be skipped.


## LLM Wiring
- Install requirements then set your API key in `.env` (copy `.env.example`).
- Example run:
  ```bash
  export $(grep -v '^#' .env | xargs)  # or use direnv
  python scripts/llm_prompt_test.py --bug_report "Button not clickable" --ocr_text "Login visible" --ui_events "tap login"
  ```

## Playwright Harness
1. `pip install -r requirements.txt`
2. `python -m playwright install`
3. Run:
   ```bash
   python scripts/playwright_harness.py --url http://localhost:3000 --steps scripts/steps.example.json --out runs/run1
   ```

## Labeling Spreadsheet
- Edit `data/processed/bug_labels.xlsx` with drop‑downs for `label`.
- CSV mirror remains at `data/processed/bug_labels.csv` for version control.


## Demo React App
A minimal React app with a seeded GUI bug is in `demo_app/`.  
To run:
```bash
cd demo_app
npm install
npm start
```

## Accessibility Audit
Run axe-core audit via Playwright:
```bash
python scripts/axe_audit.py --url http://localhost:3000 --out axe_results.json
```

## Visual Diff
Compare two screenshots:
```bash
python scripts/visual_diff.py runs/run1/before.png runs/run1/after.png --out diff.png
```


## Demo React App (with seeded bugs)
Located at `demo-app/`. It includes:
- Interaction bug (mobile button becomes unclickable)
- Low-contrast heading
- Missing `htmlFor` on label
- Decorative image without `alt`
- Incorrect `onClick` usage on small screens

### Run the app
```bash
cd demo-app
npm i
npm run dev
# open http://localhost:5173
```

## Accessibility Scan (axe-core via Playwright)
From `demo-app` in another terminal:
```bash
npm run test:axe
# exits with code 2 if violations are found; prints JSON
```

## Reproduce GUI bug and capture screenshots (Playwright steps)
```bash
npm run test:steps
# saves runs/run1/before.png and runs/run1/after.png
```

## Visual Diff
### Node:
```bash
npm run diff
# or provide custom paths:
node ../tests/visual-diff.mjs runs/run1/before.png runs/run1/after.png runs/run1/diff.png
```
### Python:
```bash
python scripts/visual_diff.py runs/run1/before.png runs/run1/after.png runs/run1/diff.png
```
