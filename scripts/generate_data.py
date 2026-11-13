"""Save synthetic candidates / jobs / activity / pairs to CSV (seeded)."""
import argparse
from pathlib import Path

import sys
sys.path.insert(0, "src")

from cam.data import generate_all

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-candidates", type=int, default=400)
    ap.add_argument("--n-jobs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="data")
    args = ap.parse_args()
    data = generate_all(args.n_candidates, args.n_jobs, args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in data.items():
        # skills / required_skills are lists — join them for a clean CSV.
        d = df.copy()
        for col in ("skills", "required_skills"):
            if col in d.columns:
                d[col] = d[col].apply(lambda v: ";".join(v))
        d.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}: {len(d)} rows -> {out / f'{name}.csv'}")
