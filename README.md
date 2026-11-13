# Candidate Activity Matching — Synthetic Portfolio Prototype

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A working prototype of the three ideas in my LandOffer Data Scientist
> supporting note: time-aware candidate activity signals for matching,
> a single source of truth for candidate facts, and effort per
> application as a guardrail metric. **All data are synthetic, generated
> from scratch with seed 42. No real candidate data, no scraping, no
> LinkedIn/GitHub API calls, no client code.**

LandOffer is candidate-side — helping people discover opportunities and
manage applications — so this prototype is framed from the candidate
side, not as a recruiter-side ATS. The question it tests is the one in
the supporting note: does verified, recent activity improve matching
*enough to matter*, without making the product harder to use or less
trustworthy?

**The metrics below are on synthetic data only. They are not the
metrics from my client project (0.89 AUROC / 0.81 AUPRC) and must not
be quoted as such.**

## The three ideas, mapped to the LandOffer job description

| Idea | What the prototype does | JD bullet it maps to |
|---|---|---|
| 1. Time-aware activity signals | Decayed (30-day) activity features — recent relevant builds / posts / portfolio updates — alongside resume features, in a pointwise learning-to-rank model | Improve job-data quality; develop baseline models for categorisation, relevance and matching; define useful metrics |
| 2. One source of truth | Canonical profile with immutable core facts; tailoring may only reorder / reword; a checker flags fact drift (dates, titles, skills), never wording | Reliable datasets; handle user data carefully; document data sources, assumptions and evaluation methods |
| 3. Least mental involvement | Effort per application simulated for manual vs assisted workflows — minutes, fields filled by hand, repeat entries, abandonment, consistency flags — reported as a guardrail | Design evaluations that distinguish real improvement from noise; explain findings, limitations and recommendations clearly |

## What it does

```
synthetic candidates / jobs / activity events (seeded)
  -> activity features, events strictly before each job's posted date
  -> temporal train / test split by posted date (cutoff 2025-07-01)
  -> resume-only ranker vs resume + activity ranker (same model class)
  -> NDCG@5, top-3 application-to-meaningful-response rate, bias check
  -> consistency checker + effort guardrail
```

"Meaningful response" means a recruiter reply, screen or interview —
not an auto-acknowledgement. Relevance for NDCG is graded from the
same latent relevance that generates the response outcome, so NDCG can
reward the activity signal instead of only skill overlap.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Full run: generate, train, evaluate (400 candidates, 120 jobs, seed 42)
PYTHONPATH=src python -m cam.pipeline --n-candidates 400 --seed 42 --output-dir outputs

cat outputs/metrics.json
```

A fast smoke run and the tests:

```bash
make smoke
make test
```

## Project layout

```
src/cam/
  data.py         # Synthetic candidates / jobs / activity / pairs (seeded)
  features.py     # Decayed activity features; hard no-leakage rule (events < job date)
  models.py       # Pointwise learning-to-rank: HistGradientBoosting / LogisticRegression
  evaluate.py     # NDCG@5, response rate by recency, bootstrap CIs, bias by group
  consistency.py  # Canonical profile, tailoring by emphasis only, fact-drift checker
  effort.py       # Effort-per-application simulation (manual vs assisted)
  pipeline.py     # End-to-end CLI: generate -> split -> train -> evaluate -> outputs
scripts/
  generate_data.py   # Save synthetic CSVs to data/
  run_evaluation.py  # Run the pipeline and print metrics.json
tests/
  test_data.py test_features.py test_consistency.py test_effort.py test_pipeline.py
data/             # README only — data are generated, not committed
.github/workflows/ci.yml   # Saved locally only — see note below
```

## Results (synthetic data, seed 42, n=400 candidates, 120 jobs)

Verified by running the pipeline in this repo. Reproduce them with
`make run` and read `outputs/metrics.json`. Every number below is
reported with its bootstrap 95% CI in that file — the table shows the
means.

| Ranker | NDCG@5 | Top-3 meaningful-response rate |
|---|---|---|
| Resume only | 0.633 [0.607, 0.660] | 0.460 [0.432, 0.487] |
| Resume + activity | 0.658 [0.632, 0.682] | 0.507 [0.478, 0.533] |
| Lift | +0.025 | +0.047 |

Read that honestly: the activity lift is **modest, and the NDCG CIs
overlap** — this is not a breakthrough, it is a signal worth keeping
and validating further. The response-rate split by recency is clearer:
candidates whose relevant activity was within 30 days responded at
0.583, vs 0.355 at 31–90 days and 0.248 at >90 days / none. Raw
activity *counts* are deliberately not used — they would reward
posting often.

**Bias check (NDCG@5, resume + activity model):** high-visibility
0.712, mixed 0.664, low-visibility 0.558. The activity model improves
every group, but the gap does not close — low-public-activity
candidates still rank worse. That is the finding, not a footnote:
activity is a useful feature and a biased one, so it stays one feature
among many, never a gate.

**Consistency demo:** tailoring that only rewords / reorders produces
0 flags. Changing one employment start date and adding a skill the
candidate never held produces 2 flags (`date_changed`,
`skill_added`).

**Effort guardrail (simulated, n=500 per mode):** manual applications
mean 21.9 minutes, 13.9 fields by hand, 3.6 repeat entries, 16.6%
abandonment; assisted (source of truth + autofill) mean 6.0 minutes,
2.9 fields, 0.2 repeat entries, 5.4% abandonment — 15.9 minutes saved
per application on average. Assisted applications still require an
explicit submit confirmation: least involvement is not least control.

## Validation — temporal, not random

Pairs are split by job posted date (train before 2025-07-01, test on
or after it), and every feature is built only from activity events
strictly before that job's posted date. No future activity, and no
outcome information, enters the features. The pipeline also reports
a bias split by synthetic visibility group and a guardrail on effort,
so a gain in responses cannot hide a product that got harder to use
or less fair.

## Limitations — read these before quoting any number

- **Synthetic ground truth.** The response outcome is generated from a
  latent score (skill overlap, experience match, recent relevant
  activity) plus noise. The activity weight (0.45) is a documented
  design choice in `src/cam/data.py` — it encodes the hypothesis being
  tested. Real-world validation would need real outcomes, consent,
  and a pre-registered experiment.
- **Activity is not ability.** Public activity can be gamed, is noisy,
  and penalises people whose best work is not public. This prototype
  shows the bias (see above) rather than hiding it.
- **Consent and platform terms.** A real version would use only
  opt-in, consented sources, within what each platform allows. Nothing
  here scrapes or calls any API.
- **Effort is simulated.** The manual-vs-assisted comparison is a
  seeded simulation of the mechanism, not measured user behaviour.
- **Pointwise ranking.** The ranker predicts P(meaningful response)
  and ranks by it — a standard pointwise learning-to-rank
  approximation, not a pairwise / listwise ranker.

## CI note

`.github/workflows/ci.yml` is saved in this repo locally, but it is
**not part of the set to push yet** — the GitHub App cannot create
workflow files (403, workflow permission), the same limitation as my
longitudinal-patient-intelligence-platform repo. Add it via the GitHub
web UI after the repo is created. There is no CI badge in this README
until then.
