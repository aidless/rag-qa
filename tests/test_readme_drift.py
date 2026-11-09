"""The README table and the honesty-bar anchors must match the offline eval."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_readme_drift import DriftError, assert_tables_match, check_repo, readme_contract  # noqa: E402


def _row(**overrides):
    base = {
        "strategy": "fixed",
        "encoder": "hash-trigram",
        "recall@5": "9/12",
        "answer_hit": "2/12",
        "neg_abstain": "4/4",
    }
    base.update(overrides)
    return base


def test_detector_rejects_a_disagreement():
    live = {
        "fixed": _row(),
        "structure": _row(strategy="structure", **{"recall@5": "10/12"}),
    }
    documented = {
        "fixed": _row(**{"recall@5": "12/12"}),
        "structure": _row(strategy="structure", **{"recall@5": "10/12"}),
    }
    with pytest.raises(DriftError, match="recall@5"):
        assert_tables_match(documented, live, "README", "eval")


def test_contract_rejects_a_long_problem_and_a_missing_status():
    text = (
        "# rag-qa\n\nSTATUS: WORK-IN-PROGRESS — not A yet\n\n"
        "## Problem\n\none\ntwo\nthree\nfour\n\n## FAILURE_MODES\n\nhash 16 extractive\n"
    )
    with pytest.raises(DriftError, match="1 to 3"):
        readme_contract(text)
    short = "# rag-qa\n\n## Problem\n\ncitation offline online\n\n## FAILURE_MODES\n\nhash 16 extractive\n"
    with pytest.raises(DriftError, match="WORK-IN-PROGRESS"):
        readme_contract(short)


def test_readme_matches_fresh_offline_eval():
    check_repo()
