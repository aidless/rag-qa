#!/usr/bin/env python3
"""Run the eval set over both chunking strategies, print a comparison table.

  python3 run_eval.py --offline          # zero-network: hash encoder + extractive
  python3 run_eval.py                    # real embeddings + LLM answers (.env)

Metrics are intentionally separable: retrieval (recall@5) does NOT depend on
the answer path, so a bad generator can't hide a bad retriever and vice versa.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ragqa import Retriever, build_chunks, answer_offline, answer_online  # noqa: E402


def load_env(path=ROOT / ".env"):
    if path.exists():
        for line in path.read_text().splitlines():
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def is_gold(hit, item):
    # corpus/ may nest files (sample/rules.md); match by suffix
    return hit["source"].endswith(item["source"]) and hit["section"].endswith(item["heading"])


def evaluate(strategy, items, offline):
    chunks = build_chunks(ROOT / "corpus", strategy)
    r = Retriever(chunks, offline=offline)
    rows, pos, neg = [], [i for i in items if i["type"] == "pos"], [i for i in items if i["type"] == "neg"]
    for item in items:
        hits = r.search(item["question"], 5)
        ev = r.has_evidence(hits)
        answer = answer_offline(item["question"], hits, ev) if offline else answer_online(item["question"], hits)
        rec5 = any(is_gold(h, item) for h in hits) if item["type"] == "pos" else None
        contains = (item["expect_contains"] in answer) if item["type"] == "pos" else None
        abstained = (answer.startswith("EVIDENCE_MISSING")) if item["type"] == "neg" else None
        rows.append({"item": item["id"], "strategy": strategy, "encoder": r.encoder,
                     "recall@5": rec5, "answer_contains": contains, "abstained": abstained,
                     "top": hits[0]["id"] if hits else None,
                     "top_bm25": hits[0]["bm25"] if hits else None})
    stats = {
        "strategy": strategy, "encoder": r.encoder,
        "recall@5": _rate(rows, "recall@5"),
        "answer_hit": _rate(rows, "answer_contains"),
        "neg_abstain": _rate(rows, "abstained"),
        "n_pos": len(pos), "n_neg": len(neg),
    }
    return stats, rows


def _rate(rows, key):
    vals = [r[key] for r in rows if r[key] is not None]
    return f"{sum(vals)}/{len(vals)}" if vals else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    load_env()
    items = [json.loads(l) for l in (ROOT / "evalset" / "qa.jsonl").read_text().splitlines() if l.strip()]

    all_rows, table = [], []
    for strategy in ("fixed", "structure"):
        stats, rows = evaluate(strategy, items, args.offline)
        all_rows += rows
        table.append(stats)

    print("\n| strategy | encoder | recall@5 | answer_hit | neg_abstain | n_pos | n_neg |")
    print("|---|---|---|---|---|---|---|")
    for s in table:
        print(f"| {s['strategy']} | {s['encoder']} | {s['recall@5']} | {s['answer_hit']} "
              f"| {s['neg_abstain']} | {s['n_pos']} | {s['n_neg']} |")
    if args.offline:
        print("NOTE: --offline = hash encoder + extractive answers; scores are a "
              "plumbing check and a lower bound, not the product's real quality.")

    out = Path(args.out or ROOT / "results" / f"eval-{'offline' if args.offline else 'online'}.jsonl")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"per-item detail -> {out}")


if __name__ == "__main__":
    main()
