# rag-qa

[![CI](https://github.com/aidless/rag-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/aidless/rag-qa/actions/workflows/ci.yml)

The image above is the GitHub Actions status for [`.github/workflows/ci.yml`](.github/workflows/ci.yml). This repo publishes no PyPI package and no coverage or download badge.

## Problem

Contest-rules QA with forced citations: an answer quotes retrieved chunks as `[S#]`, or abstains with `EVIDENCE_MISSING`.
Offline smoke (`python run_eval.py --offline`) is a hash-trigram encoder plus an extractive answer; online generation (`python run_qa.py`, `python run_eval.py` without `--offline`) calls an OpenAI-compatible LLM and is a separate path.
Keep those paths apart. The offline smoke table in this file is the hash-encoder floor. This repo has no live embedding or LLM eval scores.

## STATUS

**STATUS: WORK-IN-PROGRESS — not A yet.**

Gaps that keep this at WIP:

- The labeled set is 16 questions (12 positive, 4 negative). Abstain thresholds were fit on that set.
- No live encoder run and no online LLM run is checked in. The offline table does not stand in for those scores.
- `answer_hit` is an extractive floor (top chunk, first 800 characters).
- The hash encoder fails cross-language queries; details are in FAILURE_MODES.
- Gold match is `section.endswith(heading)`. On this run that definition misses a merged Prizes chunk and also suffix-matches `Final Evaluation`. Measured effects are in FAILURE_MODES. The table follows the code.

## Offline path

No API key and no model download. CI on Ubuntu, Python 3.12, runs the same three commands after `pip install -r requirements.txt`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q
.venv/bin/python run_eval.py --offline
.venv/bin/python scripts/check_readme_drift.py
```

`scripts/check_readme_drift.py` exits non-zero when the table, the command stdout, or `results/eval-offline.jsonl` disagree. Replace the table with the command output.

Measured 2026-11-09 on this machine, after `python3 -m venv` into an empty directory: `pip install -r requirements.txt` 6.4s, pytest 0.7s, `run_eval.py --offline` 0.2s, drift check 0.2s (7.5s combined). The install pulls PyPI packages only. No embedding weights are downloaded.

`python scripts/build_corpus_from_evidence.py` rebuilds `corpus/official/` from a local evidence directory (`RAGQA_EVID_DIR`). That directory is not in this repo. The eval uses the checked-in transcripts.

## Online path

```bash
cp .env.example .env   # LLM_API_KEY; optional LLM_BASE_URL, LLM_MODEL, EMBED_MODEL
.venv/bin/python run_qa.py "How many turns does one game last?"
.venv/bin/python run_eval.py
```

Online answers stay out of the table and out of `results/`. `results/eval-offline.jsonl` is the `--offline` per-item file.

## Offline smoke results

Command: `python run_eval.py --offline`. Index: every file under `corpus/` (`official/` transcripts and `sample/` placeholders). Checked 2026-11-09 with Python 3.12.3, rank-bm25 0.2.2, numpy 2.5.4.

On this 16-question set, dropping `corpus/sample/` left per-item recall@5, answer_hit, neg_abstain, and the top chunk id unchanged for both chunkers. Sample files still change non-English queries; see FAILURE_MODES.

| strategy | encoder | recall@5 | answer_hit | neg_abstain |
|---|---|---|---|---|
| fixed | hash-trigram | 9/12 | 2/12 | 4/4 |
| structure | hash-trigram | 10/12 | 3/12 | 4/4 |

recall@5 does not use the generator. `answer_hit` is true only when the expected span sits in the top chunk's first 800 characters, so the gap (fixed 9/12 recall vs 2/12 answer_hit, structure 10/12 vs 3/12) is unscored generation headroom.

## FAILURE_MODES

1. **Hash encoder, cross-language.** BM25 (ASCII words plus CJK bigrams) and the hash-trigram encoder do not translate. For `每天可以提交几个智能体` with structure chunks, the top hit is `corpus/sample/rules.md` (the file header says those numbers are fictional), `has_evidence` is true, and the offline answer quotes the placeholder (`5 次`). The official English `Evaluation` section is outside the top 5. The coverage gate tokenizes `[a-z0-9]+` only, so a CJK-only query skips the 0.75 check once the BM25/cosine gate has passed.

2. **Threshold overfit to 16 items.** `has_evidence` rejects a query when the best BM25 among the first two fused hits is below 3 and the top cosine is at most 0.5. Otherwise it requires at least 0.75 of the query's ASCII content words to occur in the top chunk. A query with no ASCII content words skips that coverage check. The cutoffs were tuned on these 16 questions after two negative leaks (hash cosine noise, then English function words). `test_negatives_abstain_offline` locks the four negatives. Retune when the corpus or the encoder changes.

3. **Extractive `answer_hit` versus an LLM.** `answer_offline` returns `hits[0]["text"][:800]` plus ` [S1]`, or `EVIDENCE_MISSING`. It cannot read chunks 2–5 and it cannot synthesize. Fixed chunking recalls p11 (the `$50,000` total) inside the top 5, while the top chunk is `8. WINNER’S OBLIGATIONS`, so `answer_hit` is false. `answer_online` has no checked-in scores.

4. **Gold match is `section.endswith(heading)`.** Source match is a path suffix, so nested files still count. On this run, `Final Evaluation` also satisfies heading `Evaluation`, but every positive item with that suffix collision also had the real `Evaluation` leaf inside the same top 5, so the published recall rates do not gain a hit from it. The other direction drops a hit: structure p10 has a rank-4 chunk labeled `Prizes + … / How to Play` whose text contains `5,000`, and `endswith("Prizes")` is false, so recall counts a miss. Fixed p10 does not place that section in the top 5. The table uses this definition.

5. **Prize wording in a long block.** Structure p11 (`5. TOTAL PRIZES AVAILABLE: $50,000`) is outside the top 5. The top chunk is `8. WINNER’S OBLIGATIONS`. Fixed p11 reaches the gold section at rank 5 only, which is why recall passes and `answer_hit` does not. Splitting that block, or replacing the hash encoder, is future work.

Fixes already in the tree, so they stay visible:

| Symptom | Cause | What the code does now |
|---|---|---|
| A negative was answered, twice | Hash cosine noise, then English function words lifting every BM25 | ASCII content-word coverage gate; the four negatives are a regression test |
| A short section lost its heading | Blocks under 60 characters merge into the previous chunk | The label becomes `A + B` (item 4 is the remaining endswith miss) |
| A prose line `# PASS forever after.` became a heading | The heading regex trusted a leading `#` | `_is_heading` rejects tab-separated lines and titles ending in `.` `;` `,` |

## Architecture

```
corpus/ ─ ingest ─┬─ chunk.py: fixed (200 characters, overlap 40) or structure (by heading)
                  │    both use parse_sections, so each chunk carries a section label
                  └─ chunk fields: id, source, section, text
query ─ retrieve.py: BM25 (ASCII words + CJK bigrams) ⊕ cosine, RRF (k=60) → top 5
      ─ has_evidence() false → EVIDENCE_MISSING
      ─ generate.py: online LLM, citations [S#], or the offline extractive answer
```

`corpus/official/` is a timestamped transcript of the public Overview/Evaluation and Rules pages, captured 2026-09-20 in a logged-in browser. The source URL and time are in each file header. `scripts/build_corpus_from_evidence.py` adds headings and drops nav noise. `corpus/sample/` is a fictional placeholder and stays in the index.

## Not done

- [ ] Grow the set past these 16 questions, with a source line on each item
- [ ] Run a real embedding against this hash floor, retune `has_evidence`, and record the command plus the date
- [ ] Split sections of several thousand characters (Object Types, winner obligations)
- [ ] Score online answers under a rubric written down next to a separate table

Leave these open until the run that supports them is in the repo.
