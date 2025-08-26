#!/usr/bin/env python
import os, json, glob, argparse
from pathlib import Path
import pandas as pd

def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        return json.load(f)

def main(cfg_path):
    cfg = load_config(cfg_path)
    raw = Path(cfg["paths"]["raw"])
    processed = Path(cfg["paths"]["processed"])
    processed.mkdir(parents=True, exist_ok=True)

    bug_glob = cfg["dataset"]["bug_reports_glob"]
    bug_files = glob.glob(bug_glob, recursive=True)

    rows = []
    for bf in bug_files:
        try:
            data = json.load(open(bf))
        except Exception:
            continue
        bug_id = data.get("id") or Path(bf).stem
        rows.append({
            "bug_id": bug_id,
            "bug_report_path": os.path.relpath(bf),
            "title": data.get("title"),
            "description": data.get("description") or data.get("body"),
            "project": data.get("repo") or data.get("project"),
        })

    df = pd.DataFrame(rows).drop_duplicates(subset=["bug_id"])
    out_csv = processed / "bug_reports_index.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(df)} rows.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.json", help="Path to config.json")
    args = ap.parse_args()
    main(args.config)
