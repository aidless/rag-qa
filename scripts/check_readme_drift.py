#!/usr/bin/env python3
"""Fail when the README offline table disagrees with a fresh offline eval.

Compares three views of the same run:

- the metric table in README.md
- stdout of ``python run_eval.py --offline``
- ``results/eval-offline.jsonl`` (committed per-item rows)

Also checks the honesty-bar anchors the README must keep while this repo
has no checked-in online LLM scores: a short problem statement, an explicit
WORK-IN-PROGRESS status, and the three failure modes (hash encoder, the
16-item threshold, extractive answer_hit).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
COMMITTED_JSONL = ROOT / "results" / "eval-offline.jsonl"
REQUIRED = ("strategy", "encoder", "recall@5", "answer_hit", "neg_abstain")
JSONL_KEYS = ("item", "strategy", "encoder", "recall@5", "answer_contains", "abstained", "top")
FORBIDDEN_BADGES = ("img.shields.io", "pypi.org", "pepy.tech", "badge/coverage", "badge/pypi")


class DriftError(Exception):
    """README, stdout, and the committed jsonl do not describe one run."""


def _cells(line: str) -> list[str]:
    return [c.strip().strip("`") for c in line.strip().strip("|").split("|")]


def _is_separator(line: str) -> bool:
    cells = _cells(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", c) for c in cells)


def strip_fences(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.S)


def parse_md_tables(text: str) -> list[list[dict[str, str]]]:
    lines = strip_fences(text).splitlines()
    tables: list[list[dict[str, str]]] = []
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("|") and i + 1 < len(lines) and _is_separator(lines[i + 1]):
            header = _cells(lines[i])
            i += 2
            rows: list[dict[str, str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = _cells(lines[i])
                if len(cells) != len(header):
                    raise DriftError(f"table row has {len(cells)} cells, header has {len(header)}: {lines[i]}")
                rows.append(dict(zip(header, cells)))
                i += 1
            tables.append(rows)
        else:
            i += 1
    return tables


def metric_table(text: str, label: str) -> dict[str, dict[str, str]]:
    found = [rows for rows in parse_md_tables(text) if rows and set(REQUIRED) <= set(rows[0])]
    if len(found) != 1:
        raise DriftError(f"{label} must contain exactly one offline metric table, found {len(found)}")
    out: dict[str, dict[str, str]] = {}
    for row in found[0]:
        strategy = row["strategy"]
        if strategy in out:
            raise DriftError(f"{label} repeats strategy {strategy}")
        out[strategy] = {col: row[col] for col in REQUIRED}
    if set(out) != {"fixed", "structure"}:
        raise DriftError(f"{label} strategies are {sorted(out)}; expected fixed and structure")
    for strategy, row in out.items():
        if row["encoder"] != "hash-trigram":
            raise DriftError(f"{label} {strategy} encoder is {row['encoder']!r}; offline table must say hash-trigram")
    return out


def assert_tables_match(left_rows: dict, right_rows: dict, left: str, right: str) -> None:
    if left_rows.keys() != right_rows.keys():
        raise DriftError(f"{left} strategies {sorted(left_rows)} vs {right} {sorted(right_rows)}")
    for strategy, right_row in right_rows.items():
        left_row = left_rows[strategy]
        for col in REQUIRED:
            if left_row[col] != right_row[col]:
                raise DriftError(
                    f"{left} disagrees with {right}: strategy={strategy} column={col} "
                    f"{left}={left_row[col]!r} {right}={right_row[col]!r}. "
                    "Re-run `python run_eval.py --offline` and replace the README table with that output."
                )


def _rate(values: list) -> str:
    kept = [v for v in values if v is not None]
    if not kept:
        return "-"
    return f"{sum(bool(v) for v in kept)}/{len(kept)}"


def aggregate_jsonl(path: Path) -> dict[str, dict[str, str]]:
    grouped: dict[str, list[dict]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            grouped.setdefault(row["strategy"], []).append(row)
    out: dict[str, dict[str, str]] = {}
    for strategy, rows in grouped.items():
        encoders = {r["encoder"] for r in rows}
        if len(encoders) != 1:
            raise DriftError(f"{path} strategy {strategy} mixes encoders {encoders}")
        out[strategy] = {
            "strategy": strategy,
            "encoder": encoders.pop(),
            "recall@5": _rate([r["recall@5"] for r in rows]),
            "answer_hit": _rate([r["answer_contains"] for r in rows]),
            "neg_abstain": _rate([r["abstained"] for r in rows]),
        }
    return out


def project_jsonl(path: Path) -> list[tuple]:
    projected = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        projected.append(tuple(row[key] for key in JSONL_KEYS))
    return sorted(projected)


def _section(text: str, heading: str) -> str:
    match = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", text, flags=re.M | re.S)
    if not match:
        raise DriftError(f"README is missing a ## {heading} section")
    return match.group(1)


def readme_contract(text: str) -> None:
    """Honesty-bar anchors. Numbers are checked separately against the eval."""
    if "STATUS: WORK-IN-PROGRESS" not in text or "not A yet" not in text:
        raise DriftError("README must say STATUS: WORK-IN-PROGRESS — not A yet, plus the gap list")
    for url in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):
        if "actions/workflows/ci.yml/badge.svg" not in url:
            raise DriftError(f"unexpected badge (only the ci.yml Actions badge is allowed): {url}")
    for needle in FORBIDDEN_BADGES:
        if needle in text.lower():
            raise DriftError(f"README contains a forbidden badge/publish marker: {needle}")

    problem = [line for line in _section(text, "Problem").splitlines() if line.strip()]
    if not 1 <= len(problem) <= 3:
        raise DriftError(f"## Problem must be 1 to 3 non-empty lines, found {len(problem)}")
    lowered = "\n".join(problem).lower()
    for needle in ("citation", "offline", "online"):
        if needle not in lowered:
            raise DriftError(f"## Problem must mention {needle!r} so offline and online stay separated")

    modes = _section(text, "FAILURE_MODES").lower()
    for needle in ("hash", "16", "extractive"):
        if needle not in modes:
            raise DriftError(f"## FAILURE_MODES must mention {needle!r}")


def run_offline_eval(out_path: Path) -> str:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "run_eval.py"), "--offline", "--out", str(out_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise DriftError(f"run_eval.py --offline failed ({proc.returncode}): {detail}")
    return proc.stdout


def check_repo(readme: Path = README, committed_jsonl: Path = COMMITTED_JSONL) -> None:
    text = readme.read_text(encoding="utf-8")
    readme_contract(text)
    documented = metric_table(text, "README")
    if not committed_jsonl.is_file():
        raise DriftError(f"missing committed offline detail: {committed_jsonl}")
    with tempfile.TemporaryDirectory() as tmp:
        fresh_path = Path(tmp) / "eval-offline.jsonl"
        stdout = run_offline_eval(fresh_path)
        live = metric_table(stdout, "run_eval.py --offline stdout")
        fresh_rates = aggregate_jsonl(fresh_path)
        assert_tables_match(documented, live, "README", "run_eval.py --offline")
        assert_tables_match(documented, fresh_rates, "README", "fresh jsonl")
        committed_rates = aggregate_jsonl(committed_jsonl)
        assert_tables_match(documented, committed_rates, "README", str(committed_jsonl.relative_to(ROOT)))
        fresh_rows = project_jsonl(fresh_path)
        committed_rows = project_jsonl(committed_jsonl)
        if fresh_rows != committed_rows:
            raise DriftError(
                f"{committed_jsonl.relative_to(ROOT)} per-item rows disagree with a fresh "
                "`python run_eval.py --offline` run. Regenerate the file with that command; "
                "do not hand-edit scores."
            )


def main() -> int:
    try:
        check_repo()
    except DriftError as exc:
        print(f"DRIFT: {exc}", file=sys.stderr)
        return 1
    print("OK: README offline table matches run_eval.py --offline and results/eval-offline.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
