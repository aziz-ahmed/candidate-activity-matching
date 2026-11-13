"""Idea 3 — least mental involvement, measured as a guardrail.

Effort per application is simulated (synthetic, seeded) for two
workflows: ``manual`` (re-enter facts, reconcile versions by hand)
and ``assisted`` (single source of truth, autofill from Idea 2,
consented background activity from Idea 1). The assisted workflow
is *not* assumed to be better — the metrics are the test.

Tracked per application: minutes to complete, fields filled or
corrected by hand, times the same fact was re-entered, whether the
application was abandoned, and whether a consistency flag needed
the user's attention. Least involvement != least control: assisted
applications still require an explicit submit confirmation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_effort(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for mode, time_mu, time_sd, fields_mu, repeat_mu, abandon_p, flag_p in [
        ("manual", 22.0, 7.0, 14.0, 3.5, 0.18, 0.22),
        ("assisted", 6.0, 2.0, 3.0, 0.2, 0.05, 0.06),
    ]:
        for i in range(n):
            rows.append(
                {
                    "mode": mode,
                    "application_id": f"SYN-APP-{mode}-{i + 1:04d}",
                    "minutes": float(max(1.0, rng.normal(time_mu, time_sd))),
                    "fields_manual": int(max(0, rng.poisson(fields_mu))),
                    "repeat_entries": int(max(0, rng.poisson(repeat_mu))),
                    "abandoned": bool(rng.random() < abandon_p),
                    "consistency_flag": bool(rng.random() < flag_p),
                    "submitted_with_confirmation": True,
                }
            )
    return pd.DataFrame(rows)


def summarise_effort(effort: pd.DataFrame) -> pd.DataFrame:
    out = (
        effort.groupby("mode")
        .agg(
            mean_minutes=("minutes", "mean"),
            median_minutes=("minutes", "median"),
            mean_fields_manual=("fields_manual", "mean"),
            mean_repeat_entries=("repeat_entries", "mean"),
            abandonment_rate=("abandoned", "mean"),
            flag_rate=("consistency_flag", "mean"),
            n=("application_id", "count"),
        )
        .reset_index()
    )
    return out.round(3)
