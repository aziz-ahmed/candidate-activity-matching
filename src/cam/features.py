"""Time-aware activity features — with a hard no-leakage rule.

Every feature is computed from activity events strictly before the
job's posted date (``as_of``). Events on or after ``as_of`` are never
used. The point is recency and relevance, not volume: decayed signals
with a 30-day half-life, not raw counts (raw counts are kept as a
column only so the evaluation can show why they mislead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RESUME_FEATURES = ["skill_overlap", "exp_match"]
ACTIVITY_FEATURES = [
    "decayed_relevant_30d",
    "decayed_all_30d",
    "activity_30d",
    "activity_90d",
    "activity_relevant_30d",
    "activity_relevant_90d",
    "days_since_last_relevant",
    "relevant_share",
]
# Deliberately misleading: raw volume, kept for comparison only.
MISLEADING_FEATURES = ["raw_total_activity"]
ALL_FEATURES = RESUME_FEATURES + ACTIVITY_FEATURES


def activity_features_for_pair(events: pd.DataFrame, required_skills: set, as_of: pd.Timestamp) -> dict:
    """Features for one candidate-job pair, using only events < as_of."""
    if events is None or len(events) == 0:
        return {
            "decayed_relevant_30d": 0.0,
            "decayed_all_30d": 0.0,
            "activity_30d": 0,
            "activity_90d": 0,
            "activity_relevant_30d": 0,
            "activity_relevant_90d": 0,
            "days_since_last_relevant": 180.0,
            "relevant_share": 0.0,
            "raw_total_activity": 0,
        }
    past = events[events["event_date"] < as_of].copy()
    if len(past) == 0:
        return {
            "decayed_relevant_30d": 0.0,
            "decayed_all_30d": 0.0,
            "activity_30d": 0,
            "activity_90d": 0,
            "activity_relevant_30d": 0,
            "activity_relevant_90d": 0,
            "days_since_last_relevant": 180.0,
            "relevant_share": 0.0,
            "raw_total_activity": 0,
        }
    days_ago = (as_of - past["event_date"]).dt.days.astype(float).values
    decay = np.exp(-days_ago / 30.0)  # 30-day half-life scale
    is_relevant = past["skill"].isin(required_skills).values
    quality = past["quality"].values.astype(float)
    relevant = past[is_relevant]
    if len(relevant):
        days_since = float((as_of - relevant["event_date"].max()).days)
    else:
        days_since = 180.0
    return {
        "decayed_relevant_30d": float(np.sum(decay[is_relevant] * quality[is_relevant])),
        "decayed_all_30d": float(np.sum(decay * quality)),
        "activity_30d": int(np.sum(days_ago <= 30)),
        "activity_90d": int(np.sum(days_ago <= 90)),
        "activity_relevant_30d": int(np.sum((days_ago <= 30) & is_relevant)),
        "activity_relevant_90d": int(np.sum((days_ago <= 90) & is_relevant)),
        "days_since_last_relevant": days_since,
        "relevant_share": float(np.mean(is_relevant)) if len(past) else 0.0,
        "raw_total_activity": int(len(past)),
    }


def temporal_split(pairs: pd.DataFrame, cutoff: str = "2025-07-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by job posted date, not at random.

    Train: jobs posted before the cutoff. Test: on/after it. Because
    every feature was built only from events before its job's posted
    date, no future activity can leak into training or test features.
    """
    cut = pd.Timestamp(cutoff)
    train = pairs[pairs["posted_date"] < cut].copy()
    test = pairs[pairs["posted_date"] >= cut].copy()
    return train, test
