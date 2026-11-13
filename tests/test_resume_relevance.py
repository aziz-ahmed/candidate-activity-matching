"""Tests for cross-resume relevance scoring (Idea 2, extended)."""
import copy

from cam.consistency import demo_profile, tailor
from cam.resume_relevance import (
    generate_calibration_pairs,
    run_resume_relevance,
    score_all_pairs,
    score_pair,
)


def _calibration(n_per_class=50, seed=42):
    return run_resume_relevance(n_per_class=n_per_class, seed=seed)


def test_identical_resumes_score_one():
    profile = demo_profile()
    resume = tailor(profile, ["python", "sql"])
    result = score_pair(resume, resume)
    assert result["score"] == 1.0
    assert result["core_fact_agreement"] == 1.0
    assert result["skill_jaccard"] == 1.0
    assert result["emphasis_top3"] == 1.0


def test_components_are_bounded_and_composite_is_weighted_sum():
    profile = demo_profile()
    a = tailor(profile, ["python", "sql"])
    b = tailor(profile, ["pytorch", "machine_learning"])
    r = score_pair(a, b)
    for key in ("score", "core_fact_agreement", "skill_jaccard", "emphasis_top3",
                "employment_agreement", "education_match", "skill_set_exact"):
        assert 0.0 <= r[key] <= 1.0
    expected = 0.65 * r["core_fact_agreement"] + 0.25 * r["skill_jaccard"] + 0.10 * r["emphasis_top3"]
    assert abs(r["score"] - round(expected, 4)) < 1e-9


def test_legitimate_tailoring_scores_above_threshold():
    threshold = _calibration()["threshold"]
    profile = demo_profile()
    a = tailor(profile, ["python", "sql"])
    b = tailor(profile, ["sql", "statistics"])
    assert score_pair(a, b)["score"] >= threshold


def test_different_role_legitimate_pair_scores_above_threshold():
    # DS emphasis vs ML Engineer emphasis, core facts identical:
    # this must NOT be flagged, even though the top-3 skills differ.
    threshold = _calibration()["threshold"]
    profile = demo_profile()
    ds = tailor(profile, ["python", "sql", "statistics"])
    mle = tailor(profile, ["python", "pytorch", "machine_learning"])
    result = score_pair(ds, mle)
    assert result["core_fact_agreement"] == 1.0
    assert result["score"] >= threshold


def test_fact_drift_pairs_score_below_threshold():
    threshold = _calibration()["threshold"]
    profile = demo_profile()
    base = tailor(profile, ["python", "sql"])
    drifted = copy.deepcopy(base)
    drifted["employment"][0]["start"] = "2023-01"
    assert score_pair(base, drifted)["score"] < threshold
    drifted2 = copy.deepcopy(base)
    drifted2["skills"] = drifted2["skills"] + ["spark"]
    assert score_pair(base, drifted2)["score"] < threshold


def test_calibrated_threshold_separates_classes():
    result = _calibration(n_per_class=100)
    threshold = result["threshold"]
    # Threshold is learned, not hard-coded — but on this synthetic
    # data it must sit in the gap between drift and legitimate.
    assert 0.85 <= threshold <= 0.92
    assert result["performance"]["precision"] >= 0.95
    assert result["performance"]["recall"] >= 0.95
    assert result["performance"]["false_positive_rate"] <= 0.05
    dists = result["distributions"]
    assert dists["fact_drift"]["max"] < threshold
    assert dists["legitimate"]["min"] >= threshold
    assert dists["different_role"]["min"] >= threshold
    assert dists["different_role"]["mean"] < dists["legitimate"]["mean"]


def test_score_all_pairs_covers_every_pair():
    profile = demo_profile()
    resumes = [
        tailor(profile, ["python"]),
        tailor(profile, ["sql"]),
        tailor(profile, ["statistics"]),
    ]
    pairs = score_all_pairs(resumes)
    assert len(pairs) == 3  # C(3, 2)
    assert {(p["pair"]) for p in pairs} == {(0, 1), (0, 2), (1, 2)}
    assert all(0.0 <= p["score"] <= 1.0 for p in pairs)


def test_canonical_profile_identifies_which_resume_drifted():
    profile = demo_profile()
    clean = tailor(profile, ["python", "sql"])
    drifted = copy.deepcopy(clean)
    drifted["employment"][0]["start"] = "2023-01"
    result = score_pair(clean, drifted, canonical=profile)
    assert result["canonical_flags_a"] == []
    assert any(f["type"] == "date_changed" for f in result["canonical_flags_b"])


def test_generate_calibration_pairs_labels():
    pairs = generate_calibration_pairs(n_per_class=10, seed=42)
    assert len(pairs) == 30
    assert sum(p["label"] for p in pairs) == 10
    assert {p["class"] for p in pairs} == {"legitimate", "different_role", "fact_drift"}
