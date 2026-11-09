# Contributing

Small, reviewable pull requests are welcome.

- One change per PR; describe what and why, not just what.
- Numbers in docs must be reproducible: state the command, seed, and date.
- Do not commit secrets, datasets with unclear licensing, or large binaries.
- Run `pytest -q`, `python run_eval.py --offline`, and `python scripts/check_readme_drift.py` before opening a PR.
- If the offline table moves, replace it with the command output. Do not type scores, and do not add online LLM numbers that this repo did not run.
- Be honest about negatives: a well-documented failed experiment beats a hyped one.
