# Data — synthetic by design

No data files are committed here. Run `python scripts/generate_data.py`
(or the pipeline) to generate seeded synthetic data locally:

- `candidates.csv` — synthetic candidate IDs (`SYN-CAND-0001`, …), visibility
  group, years of experience, skills. No real names or PII.
- `jobs.csv` — synthetic jobs with required skills, required experience
  and a posted date.
- `activity.csv` — synthetic activity events (GitHub-style commit,
  portfolio update, LinkedIn-style post) with a timestamp, a skill tag
  and a quality weight. These are generated, not scraped.
- `pairs.csv` — candidate-job pairs with a synthetic meaningful-response
  outcome (recruiter reply / screen / interview, not an auto-acknowledgement).

Seed 42 reproduces the numbers in `outputs/metrics.json`.
