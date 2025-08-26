#!/usr/bin/env python
import os, argparse, json, glob
from pathlib import Path
from PIL import Image
import pytesseract
import pandas as pd

def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        return json.load(f)

def main(cfg_path, limit=None):
    cfg = load_config(cfg_path)
    processed = Path(cfg["paths"]["processed"])
    processed.mkdir(parents=True, exist_ok=True)

    screenshots = glob.glob(cfg["dataset"]["screenshots_glob"], recursive=True)
    if limit:
        screenshots = screenshots[:limit]

    rows = []
    for sp in screenshots:
        text = ""
        try:
            img = Image.open(sp)
            text = pytesseract.image_to_string(img)
        except Exception as e:
            text = f"[OCR skipped or failed: {e}]"
        rows.append({"screenshot_path": os.path.relpath(sp), "ocr_text": text})

    df = pd.DataFrame(rows)
    out_csv = processed / "screenshots_ocr.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(df)} rows.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    main(args.config, args.limit)
