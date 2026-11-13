"""Synthetic data generation — seeded, reproducible, no real people.

Candidates, jobs, activity events and candidate-job outcomes are all
generated from scratch. Activity events are *synthetic* stand-ins for
GitHub-style commits, portfolio updates and LinkedIn-style posts:
they carry only a timestamp, a type and a skill tag. No scraping,
no API calls, no PII.

Visibility groups are synthetic cohorts used only for the bias check:
``high`` / ``mixed`` / ``low`` public activity. They are not protected
attributes and must not be read as such.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SKILLS = [
    "python", "sql", "machine_learning", "statistics", "nlp", "spark",
    "aws", "docker", "pytorch", "tensorflow", "visualization",
    "experimentation",
]
EVENT_TYPES = ["github_commit", "portfolio_update", "linkedin_post"]
GROUPS = ["high", "mixed", "low"]
GROUP_PROBS = [0.40, 0.35, 0.25]
# Mean activity events per 270-day window, by visibility group.
GROUP_ACTIVITY_RATE = {"high": 14.0, "mixed": 7.0, "low": 2.5}

START = pd.Timestamp("2025-01-01")
END = pd.Timestamp("2025-10-31")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_candidates(n_candidates: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_candidates):
        n_skills = int(rng.integers(4, 7))
        skills = list(rng.choice(SKILLS, size=n_skills, replace=False))
        rows.append(
            {
                "candidate_id": f"SYN-CAND-{i + 1:04d}",
                "group": str(rng.choice(GROUPS, p=GROUP_PROBS)),
                "years_exp": float(rng.integers(1, 9)),
                "skills": skills,
            }
        )
    return pd.DataFrame(rows)


def generate_jobs(n_jobs: int = 80, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    rows = []
    for j in range(n_jobs):
        n_req = int(rng.integers(3, 5))
        req = list(rng.choice(SKILLS, size=n_req, replace=False))
        posted = START + pd.Timedelta(days=int(rng.integers(31, 304)))
        rows.append(
            {
                "job_id": f"SYN-JOB-{j + 1:03d}",
                "title": str(rng.choice(["Data Scientist", "ML Engineer", "Analytics Engineer"])),
                "required_skills": req,
                "required_exp": float(rng.integers(1, 6)),
                "posted_date": posted,
            }
        )
    return pd.DataFrame(rows)


def generate_activity(candidates: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Synthetic activity events, one row per event."""
    rng = np.random.default_rng(seed + 2)
    rows = []
    eid = 0
    window_days = (END - START).days
    for _, c in candidates.iterrows():
        lam = GROUP_ACTIVITY_RATE[c["group"]]
        n_events = int(rng.poisson(lam))
        for _ in range(n_events):
            eid += 1
            # 70% of events are tagged with a skill the candidate actually has.
            if rng.random() < 0.70:
                skill = str(rng.choice(c["skills"]))
            else:
                skill = str(rng.choice(SKILLS))
            rows.append(
                {
                    "event_id": f"SYN-EVT-{eid:05d}",
                    "candidate_id": c["candidate_id"],
                    "event_date": START + pd.Timedelta(days=int(rng.integers(0, window_days))),
                    "event_type": str(rng.choice(EVENT_TYPES)),
                    "skill": skill,
                    "quality": float(rng.uniform(0.5, 1.0)),
                }
            )
    return pd.DataFrame(rows)


def generate_pairs(
    candidates: pd.DataFrame,
    jobs: pd.DataFrame,
    activity: pd.DataFrame,
    jobs_per_candidate: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Candidate-job pairs with a synthetic meaningful-response outcome.

    The outcome depends on skill overlap, experience match and *recent
    relevant* activity — the signal Idea 1 claims is predictive. Activity
    is only ever counted from events strictly before the job's posted
    date, so there is no leakage by construction; the features module
    enforces the same rule.
    """
    from .features import activity_features_for_pair  # local import: avoids a cycle

    rng = np.random.default_rng(seed + 3)
    rows = []
    act_by_cand = {cid: g for cid, g in activity.groupby("candidate_id")}
    for _, c in candidates.iterrows():
        sampled_jobs = jobs.sample(n=jobs_per_candidate, random_state=int(rng.integers(0, 10**6)))
        for _, job in sampled_jobs.iterrows():
            cand_events = act_by_cand.get(c["candidate_id"], pd.DataFrame(columns=activity.columns))
            feats = activity_features_for_pair(
                cand_events, set(job["required_skills"]), job["posted_date"]
            )
            skill_overlap = len(set(c["skills"]) & set(job["required_skills"])) / len(job["required_skills"])
            exp_match = min(1.0, c["years_exp"] / job["required_exp"])
            # Latent relevance: resume signal and recent relevant activity
            # both matter (0.35 / 0.20 / 0.45). The activity weight is a
            # design choice, documented here: it encodes the hypothesis
            # Idea 1 is testing — that recent relevant activity predicts
            # a meaningful response — so the evaluation can detect it.
            # If the weight were 0, no ranker could ever benefit.
            act_signal = 1.0 - np.exp(-feats["decayed_relevant_30d"])
            latent = 0.35 * skill_overlap + 0.20 * exp_match + 0.45 * act_signal
            p_response = float(_sigmoid(-3.0 + 6.0 * latent))
            meaningful_response = bool(rng.random() < p_response)
            # Graded relevance reflects the same latent relevance, not
            # skill overlap alone — otherwise NDCG could never reward
            # the activity signal, however predictive it was.
            relevance = 2 if meaningful_response else (1 if latent >= 0.55 else 0)
            rows.append(
                {
                    "candidate_id": c["candidate_id"],
                    "group": c["group"],
                    "job_id": job["job_id"],
                    "posted_date": job["posted_date"],
                    "skill_overlap": skill_overlap,
                    "exp_match": exp_match,
                    "latent_relevance": round(float(latent), 3),
                    "p_response_true": p_response,
                    "meaningful_response": meaningful_response,
                    "relevance": relevance,
                    **feats,
                }
            )
    return pd.DataFrame(rows)


def generate_all(n_candidates: int = 400, n_jobs: int = 120, seed: int = 42):
    candidates = generate_candidates(n_candidates, seed)
    jobs = generate_jobs(n_jobs, seed)
    activity = generate_activity(candidates, seed)
    pairs = generate_pairs(candidates, jobs, activity, seed=seed)
    return {"candidates": candidates, "jobs": jobs, "activity": activity, "pairs": pairs}
