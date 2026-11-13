import pandas as pd

from cam.data import generate_all


def test_generate_is_seeded_and_synthetic():
    a = generate_all(100, 20, seed=42)
    b = generate_all(100, 20, seed=42)
    assert a["candidates"].equals(b["candidates"])
    assert a["pairs"].equals(b["pairs"])
    assert a["candidates"]["candidate_id"].str.startswith("SYN-CAND-").all()
    assert (a["pairs"]["meaningful_response"].isin([True, False])).all()


def test_pairs_cover_both_sides_of_cutoff():
    data = generate_all(200, 40, seed=42)
    pairs = data["pairs"]
    assert (pairs["posted_date"] < "2025-07-01").any()
    assert (pairs["posted_date"] >= "2025-07-01").any()


def test_activity_events_are_before_end_and_have_skill_tags():
    data = generate_all(100, 20, seed=7)
    act = data["activity"]
    assert len(act) > 0
    assert act["event_date"].max() <= pd.Timestamp("2025-10-31")
    assert act["skill"].notna().all()
