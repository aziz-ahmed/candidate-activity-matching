"""Idea 2 — one source of truth + a consistency check.

The canonical profile holds the core facts: employment history
(title, employer, start/end dates), skills and education. Those are
immutable. Tailoring may only *reorder* skills and *reword* summaries
— it must never change a date, a title or a skill.

``check_consistency`` flags FACT drift only. Rewording, reordering
and re-emphasis produce zero flags: tailoring is normal and expected.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Employment:
    title: str
    employer: str
    start: str  # YYYY-MM
    end: str    # YYYY-MM or "present"


@dataclass
class CanonicalProfile:
    candidate_id: str
    full_name: str
    skills: list[str]
    employment: list[Employment]
    education: str

    def skill_set(self) -> set[str]:
        return {s.strip().lower() for s in self.skills}


# Rewording map: same meaning, different words. Tailoring uses this
# and nothing else — no fact is ever rewritten through it.
REWORD = {
    "built": "developed",
    "developed": "built",
    "led": "drove",
    "improved": "strengthened",
}


def tailor(profile: CanonicalProfile, job_skills: list[str]) -> dict:
    """Tailor by emphasis only: job-relevant skills first, summary reworded.

    Core facts are copied verbatim from the canonical profile.
    """
    wanted = [s.lower() for s in job_skills]
    ordered = sorted(profile.skills, key=lambda s: (s.lower() not in wanted, s.lower()))
    summary = "Built models and analyses for healthcare data."
    for old, new in REWORD.items():
        summary = summary.replace(old, new)
        break  # one rewording only; the point is that wording may change
    return {
        "candidate_id": profile.candidate_id,
        "full_name": profile.full_name,
        "skills": ordered,
        "employment": [
            {"title": e.title, "employer": e.employer, "start": e.start, "end": e.end}
            for e in profile.employment
        ],
        "education": profile.education,
        "summary": summary,
    }


def check_consistency(profile: CanonicalProfile, tailored: dict) -> list[dict]:
    """Return one flag per fact mismatch. Wording/reordering: no flags."""
    flags: list[dict] = []
    tailored_skills = {s.strip().lower() for s in tailored.get("skills", [])}
    added = sorted(tailored_skills - profile.skill_set())
    removed = sorted(profile.skill_set() - tailored_skills)
    if added:
        flags.append({"type": "skill_added", "detail": added})
    if removed:
        flags.append({"type": "skill_removed", "detail": removed})

    canon_jobs = {(e.title, e.employer, e.start, e.end) for e in profile.employment}
    tailored_jobs = {
        (j.get("title"), j.get("employer"), j.get("start"), j.get("end"))
        for j in tailored.get("employment", [])
    }
    for job in sorted(tailored_jobs - canon_jobs):
        # Classify the drift so a reviewer sees *what* changed.
        title_match = any(c[0] == job[0] and c[1] == job[1] for c in canon_jobs)
        if not title_match:
            flags.append({"type": "title_or_employer_changed", "detail": list(job)})
        else:
            flags.append({"type": "date_changed", "detail": list(job)})
    if len(tailored_jobs) != len(canon_jobs):
        flags.append(
            {"type": "employment_count_changed",
             "detail": [len(canon_jobs), len(tailored_jobs)]}
        )
    if tailored.get("education") != profile.education:
        flags.append({"type": "education_changed", "detail": tailored.get("education")})
    return flags


def demo_profile() -> CanonicalProfile:
    return CanonicalProfile(
        candidate_id="SYN-CAND-0001",
        full_name="Synthetic Candidate",
        skills=["python", "sql", "machine_learning", "statistics"],
        employment=[
            Employment(title="Data Scientist", employer="Synthetic Health Co",
                       start="2024-02", end="present"),
            Employment(title="Data Analyst", employer="Synthetic Health Co",
                       start="2022-08", end="2023-06"),
        ],
        education="MS Data Science",
    )
