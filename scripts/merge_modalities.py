#!/usr/bin/env python
import argparse, json
from pathlib import Path
import pandas as pd

def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        return json.load(f)

def main(cfg_path):
    cfg = load_config(cfg_path)
    processed = Path(cfg["paths"]["processed"])

    bugs = pd.read_csv(processed / "bug_reports_index.csv") if (processed / "bug_reports_index.csv").exists() else pd.DataFrame()
    ocr = pd.read_csv(processed / "screenshots_ocr.csv") if (processed / "screenshots_ocr.csv").exists() else pd.DataFrame()
    ui = pd.read_csv(processed / "ui_events.csv") if (processed / "ui_events.csv").exists() else pd.DataFrame()

    # Naive merge placeholders: in real dataset, link by bug_id or a shared key
    bugs["key"] = 1
    ocr["key"] = 1
    ui["key"] = 1

    merged = bugs.merge(ocr, on="key", how="left").merge(ui, on="key", how="left").drop(columns=["key"])
    out_path = processed / "merged_samples.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} with {len(merged)} rows (placeholder merge; adjust join keys for real dataset).")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.json")
    args = ap.parse_args()
    main(args.config)
