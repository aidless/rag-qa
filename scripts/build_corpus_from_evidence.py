#!/usr/bin/env python3
"""Extract the corpus from timestamped official-page evidence (READ-ONLY inputs).

Sources live in the competition repo's browser evidence dir (captured
2026-09-20 via logged-in browser, plaintext renders of public Kaggle pages).
We slice + relabel; we never edit the substance of a rule sentence.
"""
import re
import sys
from pathlib import Path

EVID = Path("/home/liuzewen/Documents/Qoder/2026-09-18/fb67e4a9/repo"
            "/.agent-memory/kagg-150k/evidence/browser_20260920")
OUT = Path(__file__).resolve().parent.parent / "corpus" / "official"

NOISE = {"link", "keyboard_arrow_up", "keyboard_arrow_down", "chevron_right",
         "expand_more", "expand_less", "more_horiz", "smart_toy", "code",
         "emoji_events", "check_circle", "search", "update"}

MARKERS = ["Timeline", "Evaluation", "Ranking System", "Final Evaluation",
           "Prizes", "How to Play", "Overview", "Object Types"]


def clean(lines, start_marker=None):
    out, started = [], start_marker is None
    for raw in lines:
        ln = raw.strip()
        if not ln or ln in NOISE:
            continue
        if start_marker and not started:
            if ln == start_marker:
                started = True
                out.append(f"## {ln}")
            continue
        if ln in MARKERS and (not out or out[-1] != f"## {ln}"):
            out.append(f"## {ln}")
        else:
            out.append(ln)
    if not started:
        sys.exit(f"start marker {start_marker!r} not found")
    return out


def main():
    ev = (EVID / "overview_evaluation_plaintext_20260920T2211.txt").read_text().splitlines()
    overview = clean(ev, start_marker="Timeline")
    # stop before the repo/submission-mechanics and citation blocks
    try:
        k = next(i for i, l in enumerate(overview)
                 if l in ("Submissions", "## Submissions", "Citation", "## Citation",
                          "Submissions must be at most 100 MiB"))
        overview = overview[:k]
    except StopIteration:
        pass

    ru = (EVID / "rules_plaintext_20260920T2204.txt").read_text().splitlines()
    a = next(i for i, l in enumerate(ru) if l.strip() == "COMPETITION-SPECIFIC TERMS" or l.strip() == "1. COMPETITION-SPECIFIC TERMS")
    b = next(i for i, l in enumerate(ru) if l.strip().startswith("3. GENERAL COMPETITION RULES") and i > a)
    rules = [l.strip() for l in ru[a:b] if l.strip() and l.strip() not in NOISE]
    rules = [f"## {l}" if re.match(r"^\d+\. [A-Z]", l) or re.match(r"^\d+\. [A-Z]+-[A-Z ]+:", l) else l
             for l in rules]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "overview_20260920.md").write_text(
        "# Kaggriculture 官方 Overview/Evaluation 页（抄本）\n\n"
        "> 抄录自 https://www.kaggle.com/competitions/kaggriculture/overview （含 evaluation），\n"
        "> 取证时间 2026-09-20T22:11（UTC+8，登录态浏览器渲染，一手证据文件见本脚本 EVID 路径）。\n"
        "> 仅做分节与去导航噪声，条文语句未改动。\n\n" + "\n".join(overview) + "\n")
    (OUT / "rules_20260920.md").write_text(
        "# Kaggriculture 官方 Rules 页 · 赛事特定条款（抄本）\n\n"
        "> 抄录自 https://www.kaggle.com/competitions/kaggriculture/rules ，\n"
        "> 取证时间 2026-09-20T22:04（一手证据同上）。仅切片与加标题，条文未改动。\n\n"
        + "\n".join(rules) + "\n")
    print("wrote:", *[p.name for p in sorted(OUT.glob("*.md"))],
          f"\noverview lines={len(overview)} rules lines={len(rules)}")


if __name__ == "__main__":
    main()
