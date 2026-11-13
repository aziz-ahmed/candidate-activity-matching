"""Run the full evaluation and print metrics.json to stdout."""
import argparse
import json
import sys

sys.path.insert(0, "src")

from cam.pipeline import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-candidates", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="outputs")
    args = ap.parse_args()
    print(json.dumps(run(args.n_candidates, 120, args.seed, args.output_dir), indent=2))
