#!/usr/bin/env python
from pathlib import Path
import argparse, json, pandas as pd

CATEGORIES = [
    "visual_layout",
    "visual_color_typography",
    "interaction_event",
    "state_transition",
    "accessibility",
    "other"
]

def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        return json.load(f)

def main(cfg_path, n=20):
    cfg = load_config(cfg_path)
    processed = Path(cfg["paths"]["processed"])
    processed.mkdir(parents=True, exist_ok=True)

    bugs_idx = processed / "bug_reports_index.csv"
    if not bugs_idx.exists():
        raise SystemExit("bug_reports_index.csv not found. Run parse_dataset.py first.")

    df = pd.read_csv(bugs_idx).head(n).copy()
    df["label"] = ""
    df["label_notes"] = ""
    out = processed / "bug_labels.csv"
    df.to_csv(out, index=False)
    print(f"Initialized {out} with {len(df)} rows. Categories: {', '.join(CATEGORIES)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.json")
    ap.add_argument("-n", type=int, default=20)
    args = ap.parse_args()
    main(args.config, args.n)
