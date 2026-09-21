#!/usr/bin/env python3
"""Single-question CLI.

  python3 run_qa.py "每天提交配额几次" --offline
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--strategy", default="structure", choices=["fixed", "structure"])
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    load_env()
    chunks = build_chunks(ROOT / "corpus", args.strategy)
    r = Retriever(chunks, offline=args.offline)
    hits = r.search(args.question, args.top_k)
    ev = r.has_evidence(hits, args.question)
    answer = (answer_offline(args.question, hits, ev) if args.offline
              else answer_online(args.question, hits))
    print(f"\n== 答案 ==\n{answer}\n")
    print("== 引用片段 ==")
    for i, h in enumerate(hits, 1):
        print(f"[S{i}] {h['source']} § {h['section']}  bm25={h['bm25']:.2f} cos={h['cos']:.3f}")
    print(json.dumps({"has_evidence": ev, "encoder": r.encoder}, ensure_ascii=False))


if __name__ == "__main__":
    main()
