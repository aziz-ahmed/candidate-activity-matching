"""Cross-resume relevance scoring — Idea 2, extended.

A candidate applies for different roles with different tailored
resumes, all stored on the same platform. This module scores the
RELEVANCE / agreement BETWEEN two resumes from the SAME candidate
for different roles, and calibrates a threshold on that score.

Design — why the weights are what they are
--------------------------------------------
The composite is a weighted sum of three explainable components,
all in [0, 1]:

* core_fact_agreement (weight 0.65) — the anchor. Employment
  history (exact title/employer/start/end tuples), education, and
  whether the skill SET is identical. A single changed date, title,
  employer, education, or skill set drops this component, because
  those are facts, not emphasis.
* skill_jaccard (weight 0.25) — Jaccard overlap of the two skill
  sets. A skill added or removed moves this, but only by the size
  of the change relative to the set.
* emphasis_top3 (weight 0.10) — overlap of the top-3 skills by
  listed order. Different roles legitimately emphasise different
  skills, so this is deliberately the smallest weight: a Data
  Scientist resume vs an ML Engineer resume from the same person
  can have almost no top-3 overlap and still score high, because
  the core facts agree.

That ordering is the whole point: core facts decide, emphasis
only nudges. A score below the calibrated threshold means "these
two resumes disagree on facts — review them", never "reject this
candidate". The threshold itself is learned from synthetic data
(see ``calibrate``), not chosen by hand, and it is platform- and
population-specific: it must be re-calibrated on real data.

No external embeddings or models are used: the score is computed
from the resume dicts produced by ``consistency.tailor``.
"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path

from .consistency import CanonicalProfile, Employment, check_consistency, tailor

# Composite weights: core facts anchor, skill overlap second,
# emphasis a mild nudge only.
W_CORE = 0.65
W_JACCARD = 0.25
W_EMPHASIS = 0.10

# Core-fact sub-weights: employment history dominates, education
# and skill-set identity are each a single fact field.
W_EMPLOYMENT = 0.60
W_EDUCATION = 0.20
W_SKILL_SET = 0.20

# Role skill lists used to generate different-role tailoring.
# They overlap on purpose (both roles use Python) but emphasise
# different skills — exactly the case the threshold must not flag.
DS_ROLE_SKILLS = ["python", "sql", "statistics", "visualization"]
MLE_ROLE_SKILLS = ["python", "pytorch", "machine_learning", "tensorflow"]
ANALYTICS_ROLE_SKILLS = ["sql", "statistics", "visualization", "experimentation"]

SKILL_POOL = [
    "python", "sql", "machine_learning", "statistics", "nlp", "spark",
    "aws", "docker", "pytorch", "tensorflow", "visualization",
    "experimentation",
]
EDUCATIONS = ["MS Data Science", "BS Computer Science", "MS Statistics"]


def _skill_set(resume: dict) -> set[str]:
    return {str(s).strip().lower() for s in resume.get("skills", [])}


def _employment_tuples(resume: dict) -> set[tuple]:
    return {
        (str(j.get("title")), str(j.get("employer")),
         str(j.get("start")), str(j.get("end")))
        for j in resume.get("employment", [])
    }


def _employment_agreement(a: dict, b: dict) -> float:
    ta, tb = _employment_tuples(a), _employment_tuples(b)
    if not ta and not tb:
        return 1.0
    denom = max(len(ta), len(tb))
    if denom == 0:
        return 1.0
    return len(ta & tb) / denom


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _emphasis_top3(a: dict, b: dict) -> float:
    sa = [str(s).strip().lower() for s in a.get("skills", [])]
    sb = [str(s).strip().lower() for s in b.get("skills", [])]
    k = min(3, len(sa), len(sb))
    if k == 0:
        return 1.0
    return len(set(sa[:k]) & set(sb[:k])) / k


def score_pair(resume_a: dict, resume_b: dict, canonical: CanonicalProfile | None = None) -> dict:
    """Score agreement between two resumes from the same candidate.

    Returns the composite score in [0, 1] plus every component,
    so a reviewer can see WHY a pair scored low. If ``canonical``
    is given, each resume is also checked against it with
    ``check_consistency`` and the flags are reported — that tells
    you *which* resume drifted, which a pair score alone cannot.
    """
    emp = _employment_agreement(resume_a, resume_b)
    edu = 1.0 if resume_a.get("education") == resume_b.get("education") else 0.0
    set_a, set_b = _skill_set(resume_a), _skill_set(resume_b)
    skill_exact = 1.0 if set_a == set_b else 0.0
    core = W_EMPLOYMENT * emp + W_EDUCATION * edu + W_SKILL_SET * skill_exact
    jaccard = _jaccard(set_a, set_b)
    emphasis = _emphasis_top3(resume_a, resume_b)
    composite = W_CORE * core + W_JACCARD * jaccard + W_EMPHASIS * emphasis
    result = {
        "score": round(float(composite), 4),
        "core_fact_agreement": round(float(core), 4),
        "employment_agreement": round(float(emp), 4),
        "education_match": round(float(edu), 4),
        "skill_set_exact": round(float(skill_exact), 4),
        "skill_jaccard": round(float(jaccard), 4),
        "emphasis_top3": round(float(emphasis), 4),
    }
    if canonical is not None:
        result["canonical_flags_a"] = check_consistency(canonical, resume_a)
        result["canonical_flags_b"] = check_consistency(canonical, resume_b)
    return result


def score_all_pairs(resumes: list[dict], canonical: CanonicalProfile | None = None) -> list[dict]:
    """Score every unordered pair in a list of resumes."""
    out = []
    for i in range(len(resumes)):
        for j in range(i + 1, len(resumes)):
            r = score_pair(resumes[i], resumes[j], canonical=canonical)
            r["pair"] = (i, j)
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Synthetic pair generation for threshold calibration
# ---------------------------------------------------------------------------

def _synthetic_profile(rng: random.Random, idx: int) -> CanonicalProfile:
    n_skills = rng.randint(5, 7)
    skills = rng.sample(SKILL_POOL, n_skills)
    # Ensure the role skills used below are mostly present, so
    # different-role tailoring is an emphasis change, not a
    # skill-set change: force python + sql into the pool.
    for s in ("python", "sql"):
        if s not in skills:
            skills[rng.randrange(len(skills))] = s
    return CanonicalProfile(
        candidate_id=f"SYN-CAND-{idx + 1:04d}",
        full_name="Synthetic Candidate",
        skills=skills,
        employment=[
            Employment(title="Data Scientist", employer="Synthetic Health Co",
                       start="2024-02", end="present"),
            Employment(title="Data Analyst", employer="Synthetic Health Co",
                       start="2022-08", end="2023-06"),
        ],
        education=rng.choice(EDUCATIONS),
    )


def _mutate_drift(resume: dict, rng: random.Random) -> dict:
    """Apply ONE fact drift to a copy of a resume. Returns (mutated, drift_type)."""
    mutated = copy.deepcopy(resume)
    drift = rng.choice(["date", "title", "employer", "education", "skill_added", "skill_removed"])
    if drift == "date":
        mutated["employment"][0]["start"] = "2023-01"
    elif drift == "title":
        mutated["employment"][0]["title"] = "Senior Data Scientist"
    elif drift == "employer":
        mutated["employment"][0]["employer"] = "Other Company"
    elif drift == "education":
        mutated["education"] = "MS Computer Science" if resume.get("education") != "MS Computer Science" else "PhD Statistics"
    elif drift == "skill_added":
        extra = next(s for s in SKILL_POOL if s not in _skill_set(resume))
        mutated["skills"] = list(mutated["skills"]) + [extra]
    elif drift == "skill_removed":
        mutated["skills"] = list(mutated["skills"])[:-1]
    return mutated, drift


def generate_calibration_pairs(n_per_class: int = 200, seed: int = 42) -> list[dict]:
    """Generate labelled synthetic resume pairs.

    Classes (label 0 = legitimate, 1 = fact drift):
    * ``legitimate`` — reorder/reword only, core facts identical.
    * ``different_role`` — DS skills emphasised vs ML skills
      emphasised, core facts identical. Label 0: this must NOT
      be flagged.
    * ``fact_drift`` — one fact changed in one resume (date,
      title, employer, education, skill added, skill removed).
      Label 1.
    """
    rng = random.Random(seed)
    pairs = []
    for cls in ("legitimate", "different_role", "fact_drift"):
        for i in range(n_per_class):
            profile = _synthetic_profile(rng, i)
            if cls == "legitimate":
                a = tailor(profile, DS_ROLE_SKILLS[:2])
                b = tailor(profile, DS_ROLE_SKILLS[1:3])
            elif cls == "different_role":
                a = tailor(profile, DS_ROLE_SKILLS)
                b = tailor(profile, MLE_ROLE_SKILLS)
            else:
                a = tailor(profile, DS_ROLE_SKILLS[:2])
                b = tailor(profile, DS_ROLE_SKILLS[1:3])
                b, drift = _mutate_drift(b, rng)
            result = score_pair(a, b, canonical=profile)
            result["class"] = cls
            result["label"] = 1 if cls == "fact_drift" else 0
            if cls == "fact_drift":
                result["drift_type"] = drift
            pairs.append(result)
    return pairs


def _distribution(scores: list[float]) -> dict:
    scores = sorted(scores)
    n = len(scores)
    def pct(p: float) -> float:
        return round(float(scores[min(n - 1, max(0, int(p * n)))]), 4)
    mean = sum(scores) / n if n else 0.0
    var = sum((s - mean) ** 2 for s in scores) / n if n else 0.0
    return {
        "n": n,
        "mean": round(float(mean), 4),
        "std": round(float(var ** 0.5), 4),
        "min": round(float(scores[0]), 4) if n else None,
        "p5": pct(0.05),
        "median": pct(0.50),
        "p95": pct(0.95),
        "max": round(float(scores[-1]), 4) if n else None,
    }


def calibrate_threshold(pairs: list[dict]) -> dict:
    """Choose a threshold separating legitimate pairs from fact drift.

    Predicted drift = score < threshold. The threshold maximises
    F1 over a 0.50–0.99 grid, breaking ties by Youden's J
    (sensitivity + specificity − 1). Returns the threshold, the
    classification performance at it, and score distributions.
    """
    legit = [p["score"] for p in pairs if p["label"] == 0]
    drift = [p["score"] for p in pairs if p["label"] == 1]
    best = None
    for t_int in range(50, 100):
        t = t_int / 100.0
        tp = sum(1 for s in drift if s < t)
        fn = len(drift) - tp
        fp = sum(1 for s in legit if s < t)
        tn = len(legit) - fp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        youden = recall + specificity - 1.0
        key = (round(f1, 6), round(youden, 6))
        if best is None or key > best["key"]:
            best = {"key": key, "threshold": t, "precision": precision, "recall": recall,
                    "f1": f1, "specificity": specificity, "youden_j": youden,
                    "tp": tp, "fp": fp, "tn": tn, "fn": fn}
    best.pop("key")
    return {
        "threshold": best["threshold"],
        "method": "max F1 over 0.50-0.99 grid (step 0.01), ties broken by Youden's J; predicted drift = score < threshold",
        "performance": {
            "precision": round(best["precision"], 4),
            "recall": round(best["recall"], 4),
            "f1": round(best["f1"], 4),
            "specificity": round(best["specificity"], 4),
            "false_positive_rate": round(1 - best["specificity"], 4),
            "tp": best["tp"], "fp": best["fp"], "tn": best["tn"], "fn": best["fn"],
        },
        "distributions": {
            cls: _distribution([p["score"] for p in pairs if p["class"] == cls])
            for cls in ("legitimate", "different_role", "fact_drift")
        },
        "distributions_by_drift_type": {
            dt: _distribution([p["score"] for p in pairs if p.get("drift_type") == dt])
            for dt in sorted({p.get("drift_type") for p in pairs if p.get("drift_type")})
        },
    }


def run_resume_relevance(n_per_class: int = 200, seed: int = 42,
                         output_path: str | Path | None = None) -> dict:
    """Generate calibration pairs, calibrate the threshold, optionally save JSON."""
    pairs = generate_calibration_pairs(n_per_class=n_per_class, seed=seed)
    result = calibrate_threshold(pairs)
    result["seed"] = seed
    result["n_per_class"] = n_per_class
    result["n_pairs"] = len(pairs)
    result["synthetic"] = True
    result["score_weights"] = {
        "core_fact_agreement": W_CORE,
        "skill_jaccard": W_JACCARD,
        "emphasis_top3": W_EMPHASIS,
    }
    result["interpretation"] = (
        "Score < threshold means the two resumes disagree on facts and should be "
        "reviewed by the candidate, not auto-rejected. Different-role pairs "
        "(DS vs ML Engineer emphasis) are legitimate and score with the "
        "legitimate class because core facts agree."
    )
    if output_path is not None:
        Path(output_path).write_text(json.dumps(result, indent=2))
    return result
