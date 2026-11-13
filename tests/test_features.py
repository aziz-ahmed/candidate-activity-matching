import pandas as pd

from cam.features import activity_features_for_pair, temporal_split


def _events():
    return pd.DataFrame(
        [
            {"event_date": pd.Timestamp("2025-05-01"), "skill": "python", "quality": 1.0},
            {"event_date": pd.Timestamp("2025-06-15"), "skill": "sql", "quality": 0.5},
            # After the as-of date: must be ignored (no leakage).
            {"event_date": pd.Timestamp("2025-08-01"), "skill": "python", "quality": 1.0},
        ]
    )


def test_no_future_activity_leaks():
    feats = activity_features_for_pair(_events(), {"python"}, pd.Timestamp("2025-07-01"))
    assert feats["activity_90d"] == 2
    assert feats["raw_total_activity"] == 2
    # Only the pre-cutoff python event counts as relevant.
    assert feats["days_since_last_relevant"] == 61
    assert feats["decayed_relevant_30d"] > 0


def test_empty_activity_is_safe():
    feats = activity_features_for_pair(
        pd.DataFrame(columns=["event_date", "skill", "quality"]),
        {"python"}, pd.Timestamp("2025-07-01"),
    )
    assert feats["decayed_relevant_30d"] == 0.0
    assert feats["days_since_last_relevant"] == 180.0


def test_temporal_split_is_by_date_not_random():
    pairs = pd.DataFrame(
        {"posted_date": pd.to_datetime(["2025-03-01", "2025-08-01"]), "x": [1, 2]}
    )
    train, test = temporal_split(pairs)
    assert len(train) == 1 and len(test) == 1
    assert train["posted_date"].max() < test["posted_date"].min()
