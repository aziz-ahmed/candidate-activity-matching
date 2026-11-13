"""Evaluation: NDCG, response rate, bias check — all with uncertainty.

Primary metric: mean per-candidate NDCG@5 on the temporal test set.
Secondary (the LandOffer product metric): application-to-meaningful-
response rate for the top-3 jobs each ranker recommends per candidate,
split by match source and by activity recency.
Guardrails: the same metrics split by synthetic visibility group, and
effort per application (from effort.py). Every headline number is
reported with a bootstrap 95% CI — never a single number alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def dcg_at_k(relevances, k: int = 5) -> float:
    rel = np.asarray(relevances[:k], dtype=float)
    discounts = np.log2(np.arange(2, len(rel) + 2))
    return float(np.sum((2.0**rel - 1.0) / discounts))


def ndcg_at_k(relevances, k: int = 5) -> float:
    ideal = dcg_at_k(sorted(relevances, reverse=True), k=k)
    if ideal == 0:
        return 0.0
    return dcg_at_k(relevances, k=k) / ideal


def per_candidate_ndcg(df: pd.DataFrame, score_col: str, k: int = 5) -> pd.DataFrame:
    """One NDCG@k per candidate, ranking that candidate's test jobs by score."""
    rows = []
    for cand, group in df.groupby("candidate_id"):
        ranked = group.sort_values(score_col, ascending=False)
        rows.append(
            {
                "candidate_id": cand,
                "group": group["group"].iloc[0],
                "ndcg": ndcg_at_k(ranked["relevance"].tolist(), k=k),
                "n_jobs": len(group),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(values, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05):
    """Bootstrap 95% CI for a mean. Returns (mean, lo, hi)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [
        float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(n_boot)
    ]
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.mean(values)), float(lo), float(hi)


def topk_response_rate(df: pd.DataFrame, score_col: str, k: int = 3) -> pd.DataFrame:
    """Top-k recommended pairs per candidate, with their response rate."""
    picked = []
    for _, group in df.groupby("candidate_id"):
        picked.append(group.sort_values(score_col, ascending=False).head(k))
    topk = pd.concat(picked) if picked else df.iloc[0:0]
    return topk


def response_rate_by_recency(df: pd.DataFrame) -> pd.DataFrame:
    """Response rate split by how recent the candidate's relevant activity was."""
    d = df.copy()
    d["recency_bucket"] = pd.cut(
        d["days_since_last_relevant"],
        bins=[-1, 30, 90, 180],
        labels=["<=30d", "31-90d", ">90d / none"],
    )
    out = (
        d.groupby("recency_bucket", observed=True)["meaningful_response"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "response_rate", "count": "n"})
        .reset_index()
    )
    return out


def evaluate_ranker(test: pd.DataFrame, score_col: str, k: int = 5, topk: int = 3) -> dict:
    per_cand = per_candidate_ndcg(test, score_col, k=k)
    ndcg_mean, ndcg_lo, ndcg_hi = bootstrap_ci(per_cand["ndcg"].values)
    top = topk_response_rate(test, score_col, k=topk)
    rr_mean, rr_lo, rr_hi = bootstrap_ci(top["meaningful_response"].astype(float).values)
    by_group = {}
    for grp, sub in per_cand.groupby("group"):
        m, lo, hi = bootstrap_ci(sub["ndcg"].values)
        by_group[str(grp)] = {"ndcg@5": round(m, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3), "n_candidates": int(len(sub))}
    return {
        "ndcg@5_mean": round(ndcg_mean, 3),
        "ndcg@5_ci": [round(ndcg_lo, 3), round(ndcg_hi, 3)],
        "top3_response_rate": round(rr_mean, 3),
        "top3_response_rate_ci": [round(rr_lo, 3), round(rr_hi, 3)],
        "n_test_pairs": int(len(test)),
        "n_test_candidates": int(test["candidate_id"].nunique()),
        "ndcg_by_group": by_group,
    }
