"""End-to-end pipeline: generate -> features -> temporal split -> train -> evaluate.

Usage:
    PYTHONPATH=src python -m cam.pipeline --n-candidates 400 --seed 42 --output-dir outputs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import __version__
from .consistency import check_consistency, demo_profile, tailor
from .data import generate_all
from .effort import simulate_effort, summarise_effort
from .evaluate import bootstrap_ci, evaluate_ranker, response_rate_by_recency
from .features import temporal_split
from .models import RESUME_ONLY, RESUME_PLUS_ACTIVITY, fit_ranker, predict_scores


def run(n_candidates: int = 400, n_jobs: int = 120, seed: int = 42,
        output_dir: str = "outputs", kind: str = "gbm") -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    data = generate_all(n_candidates=n_candidates, n_jobs=n_jobs, seed=seed)
    pairs = data["pairs"]
    train, test = temporal_split(pairs)
    if len(test) == 0 or len(train) == 0:
        raise ValueError("temporal split produced an empty side; check the cutoff / dates")

    results: dict = {
        "version": __version__,
        "seed": seed,
        "n_candidates": int(n_candidates),
        "n_jobs": int(n_jobs),
        "n_train_pairs": int(len(train)),
        "n_test_pairs": int(len(test)),
        "split_cutoff": "2025-07-01",
        "synthetic": True,
        "models": {},
    }

    for name, cols in [("resume_only", RESUME_ONLY), ("resume_plus_activity", RESUME_PLUS_ACTIVITY)]:
        model = fit_ranker(train, cols, kind=kind)
        test_scored = test.copy()
        test_scored["score"] = predict_scores(model, test, cols)
        results["models"][name] = evaluate_ranker(test_scored, "score")
        # Product metric split by activity recency, for the activity model.
        if name == "resume_plus_activity":
            rec = response_rate_by_recency(test_scored)
            results["response_rate_by_recency"] = [
                {"bucket": str(r["recency_bucket"]), "response_rate": round(float(r["response_rate"]), 3),
                 "n": int(r["n"])} for _, r in rec.iterrows()
            ]
            # Also: response rate by recency for resume-only top-3, for contrast.
            test_resume = test.copy()
            m0 = fit_ranker(train, RESUME_ONLY, kind=kind)
            test_resume["score"] = predict_scores(m0, test, RESUME_ONLY)

    # Headline comparison with uncertainty (bootstrap on per-candidate NDCG).
    a = results["models"]["resume_only"]["ndcg@5_mean"]
    b = results["models"]["resume_plus_activity"]["ndcg@5_mean"]
    results["ndcg_lift_activity_vs_resume_only"] = round(b - a, 3)

    # Idea 2 — consistency demo: wording-only tailoring passes; fact drift is flagged.
    profile = demo_profile()
    clean = tailor(profile, ["python", "machine_learning"])
    drifted = tailor(profile, ["python", "machine_learning"])
    drifted["employment"][0]["start"] = "2023-01"  # fact drift, not wording
    drifted["skills"] = drifted["skills"] + ["spark"]  # skill never held
    results["consistency_demo"] = {
        "wording_only_tailoring_flags": check_consistency(profile, clean),
        "fact_drift_flags": check_consistency(profile, drifted),
    }

    # Idea 3 — effort guardrail.
    effort = simulate_effort(seed=seed)
    eff_summary = summarise_effort(effort)
    results["effort_by_mode"] = eff_summary.to_dict(orient="records")
    manual = effort[effort["mode"] == "manual"]["minutes"].values
    assisted = effort[effort["mode"] == "assisted"]["minutes"].values
    results["effort_minutes_saved_mean"] = round(float(manual.mean() - assisted.mean()), 1)

    # Save artefacts.
    pairs.to_csv(out / "pairs.csv", index=False)
    data["activity"].to_csv(out / "activity.csv", index=False)
    eff_summary.to_csv(out / "effort_summary.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(results, indent=2))

    # One simple figure: NDCG@5 with 95% CI bars.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = ["Resume only", "Resume + activity"]
        means = [results["models"]["resume_only"]["ndcg@5_mean"],
                 results["models"]["resume_plus_activity"]["ndcg@5_mean"]]
        los = [results["models"]["resume_only"]["ndcg@5_ci"][0],
               results["models"]["resume_plus_activity"]["ndcg@5_ci"][0]]
        his = [results["models"]["resume_only"]["ndcg@5_ci"][1],
               results["models"]["resume_plus_activity"]["ndcg@5_ci"][1]]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(names, means, yerr=[[m - lo for m, lo in zip(means, los)],
                                    [hi - m for m, hi in zip(means, his)]],
               capsize=6, color=["#9db8d2", "#2e6da4"])
        ax.set_ylabel("Mean NDCG@5 (synthetic test set)")
        ax.set_title("Ranking quality: resume-only vs resume + activity")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(out / "figures" / "ndcg_comparison.png", dpi=150)
        plt.close(fig)
    except Exception as exc:  # figure is nice-to-have, not the result
        results["figure_warning"] = str(exc)

    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Candidate activity matching — synthetic prototype")
    ap.add_argument("--n-candidates", type=int, default=400)
    ap.add_argument("--n-jobs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--kind", default="gbm", choices=["gbm", "logreg"])
    args = ap.parse_args()
    results = run(args.n_candidates, args.n_jobs, args.seed, args.output_dir, args.kind)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
