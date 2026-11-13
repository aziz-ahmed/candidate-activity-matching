from cam.consistency import check_consistency, demo_profile, tailor


def test_wording_only_tailoring_passes():
    profile = demo_profile()
    tailored = tailor(profile, ["python", "machine_learning"])
    assert check_consistency(profile, tailored) == []


def test_skill_reorder_is_not_drift():
    profile = demo_profile()
    tailored = tailor(profile, ["statistics", "sql"])
    tailored["skills"] = list(reversed(tailored["skills"]))
    assert check_consistency(profile, tailored) == []


def test_date_change_is_flagged():
    profile = demo_profile()
    tailored = tailor(profile, ["python"])
    tailored["employment"][0]["start"] = "2023-01"
    flags = check_consistency(profile, tailored)
    assert any(f["type"] == "date_changed" for f in flags)


def test_title_change_is_flagged():
    profile = demo_profile()
    tailored = tailor(profile, ["python"])
    tailored["employment"][0]["title"] = "Senior Data Scientist"
    flags = check_consistency(profile, tailored)
    assert any(f["type"] == "title_or_employer_changed" for f in flags)


def test_added_skill_is_flagged():
    profile = demo_profile()
    tailored = tailor(profile, ["python"])
    tailored["skills"] = tailored["skills"] + ["spark"]
    flags = check_consistency(profile, tailored)
    assert any(f["type"] == "skill_added" for f in flags)
