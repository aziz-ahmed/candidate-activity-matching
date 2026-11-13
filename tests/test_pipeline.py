from cam.evaluate import bootstrap_ci, ndcg_at_k
from cam.pipeline import run


def test_ndcg_is_one_for_perfect_ranking():
    assert ndcg_at_k([2, 1, 0]) == 1.0
    assert 0.0 <= ndcg_at_k([0, 0, 2]) <= 1.0


def test_bootstrap_ci_contains_mean():
    mean, lo, hi = bootstrap_ci([0.5, 0.6, 0.7, 0.4], n_boot=200, seed=1)
    assert lo <= mean <= hi


def test_pipeline_runs_small_and_reports_ci(tmp_path):
    results = run(n_candidates=120, n_jobs=30, seed=42, output_dir=str(tmp_path))
    assert results["synthetic"] is True
    for name in ("resume_only", "resume_plus_activity"):
        m = results["models"][name]
        assert 0.0 <= m["ndcg@5_mean"] <= 1.0
        assert m["ndcg@5_ci"][0] <= m["ndcg@5_mean"] <= m["ndcg@5_ci"][1]
        assert m["n_test_candidates"] > 0
    # Consistency demo: wording-only passes, fact drift is flagged.
    assert results["consistency_demo"]["wording_only_tailoring_flags"] == []
    assert len(results["consistency_demo"]["fact_drift_flags"]) >= 2
    assert results["effort_minutes_saved_mean"] > 0
    assert (tmp_path / "metrics.json").exists()
