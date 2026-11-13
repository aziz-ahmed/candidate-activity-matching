from cam.effort import simulate_effort, summarise_effort


def test_effort_modes_present_and_seeded():
    a = simulate_effort(n=100, seed=42)
    b = simulate_effort(n=100, seed=42)
    assert a.equals(b)
    assert set(a["mode"]) == {"manual", "assisted"}
    assert a["submitted_with_confirmation"].all()


def test_assisted_is_less_effort_than_manual():
    eff = simulate_effort(n=500, seed=42)
    summary = summarise_effort(eff)
    manual = summary[summary["mode"] == "manual"].iloc[0]
    assisted = summary[summary["mode"] == "assisted"].iloc[0]
    assert assisted["mean_minutes"] < manual["mean_minutes"]
    assert assisted["abandonment_rate"] < manual["abandonment_rate"]
    assert assisted["mean_repeat_entries"] < manual["mean_repeat_entries"]
