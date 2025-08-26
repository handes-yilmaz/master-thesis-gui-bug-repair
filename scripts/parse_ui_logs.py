#!/usr/bin/env python
import os, json, glob, argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        return json.load(f)

def main(cfg_path, limit=None):
    cfg = load_config(cfg_path)
    processed = Path(cfg["paths"]["processed"])
    processed.mkdir(parents=True, exist_ok=True)

    ui_glob = cfg["dataset"]["ui_logs_glob"]
    files = glob.glob(ui_glob, recursive=True)
    if limit:
        files = files[:limit]

    rows = []
    for fp in tqdm(files, desc="Parsing UI logs"):
        try:
            with open(fp, "r") as f:
                for line in f:
                    try:
                        ev = json.loads(line.strip())
                    except Exception:
                        continue
                    rows.append({
                        "log_path": os.path.relpath(fp),
                        "ts": ev.get("timestamp"),
                        "type": ev.get("type") or ev.get("event"),
                        "target": ev.get("target"),
                        "value": ev.get("value"),
                    })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    out_csv = processed / "ui_events.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(df)} rows.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    main(args.config, args.limit)
