from ragqa.chunk import parse_sections, chunk_fixed, chunk_structure
from ragqa.retrieve import tokenize

TEXT = (
    "# 标题\n\n引言若干字。\n\n## 甲节\n\n甲节内容，讲了配额为 5 次这件事。" + "补" * 80 +
    "\n\n## 乙节\n\n乙节内容讲的是 120 回合上限。"
)


def test_parse_sections_covers_and_labels():
    blocks = parse_sections(TEXT)
    assert blocks[0]["end"] == len(TEXT) or blocks[-1]["end"] == len(TEXT)
    sections = [b["section"] for b in blocks]
    assert any(s.endswith("甲节") for s in sections)
    assert any(s.endswith("乙节") for s in sections)


def test_fixed_chunks_overlap_and_label():
    chunks = chunk_fixed(TEXT, size=60, overlap=20)
    assert len(chunks) > 2
    joined = "".join(c["text"] for c in chunks)
    assert "120" in joined and "配额" in joined
    assert all(c["section"] for c in chunks)


def test_structure_chunks_keep_heading():
    chunks = chunk_structure(TEXT)
    assert any("甲节" in c["section"] and "配额" in c["text"] for c in chunks)


def test_tokenize_mixed():
    toks = tokenize("每天 20:00 提交配额")
    assert "20" in toks and "00" in toks
    assert "提交" in toks and "交配" in toks and "配额" in toks


def _leaves(section):
    """Last title of each path, including both sides of a merged 'A + B' label."""
    return [part.split(" / ")[-1].strip() for part in section.split(" + ")]


def test_retriever_finds_gold_offline():
    from ragqa.ingest import build_chunks
    from ragqa.retrieve import Retriever
    from pathlib import Path
    chunks = build_chunks(Path(__file__).resolve().parent.parent / "corpus", "structure")
    r = Retriever(chunks, offline=True)
    hits = r.search("How many agents can a team submit per day?", 5)
    # Leaf equality: "Final Evaluation" must not satisfy heading "Evaluation".
    assert any(h["source"].endswith("overview_20260920.md") and "Evaluation" in _leaves(h["section"])
               for h in hits)


def test_offline_answer_is_extractive_prefix():
    from ragqa.generate import answer_offline
    hits = [{"source": "s", "section": "h", "text": "FACT " + ("x" * 1000)}]
    answered = answer_offline("q", hits, True)
    assert answered.endswith("[S1]")
    assert "FACT" in answered
    assert len(answered) < 900
    assert answer_offline("q", hits, False) == "EVIDENCE_MISSING"


def test_chinese_query_hits_sample_not_english_rule():
    """Hash/BM25 do not translate. The Chinese sample note outranks the English rule.

    Coverage uses ASCII tokens only, so this CJK query skips the 0.75 gate and
    has_evidence stays true. The quoted number comes from the fictional sample.
    """
    from pathlib import Path
    from ragqa.generate import answer_offline
    from ragqa.ingest import build_chunks
    from ragqa.retrieve import Retriever
    from run_eval import is_gold

    root = Path(__file__).resolve().parent.parent
    chunks = build_chunks(root / "corpus", "structure")
    r = Retriever(chunks, offline=True)
    question = "每天可以提交几个智能体"
    hits = r.search(question, 5)
    evidence = r.has_evidence(hits, question)
    answer = answer_offline(question, hits, evidence)
    assert hits[0]["source"].endswith("sample/rules.md")
    assert evidence is True
    assert "5 次" in answer
    assert "up to 5 agents" not in answer
    gold = {"source": "overview_20260920.md", "heading": "Evaluation"}
    assert not any(is_gold(h, gold) for h in hits)


def test_structure_p10_merged_prizes_chunk_is_not_gold_under_endswith():
    """Rank-4 Prizes text is in the top 5, but the merged label does not end with Prizes.

    recall@5 in the README uses this endswith definition. Update FAILURE_MODES
    and the table together if the matcher changes.
    """
    import json
    from pathlib import Path
    from ragqa.ingest import build_chunks
    from ragqa.retrieve import Retriever
    from run_eval import is_gold

    root = Path(__file__).resolve().parent.parent
    item = next(json.loads(line) for line in (root / "evalset" / "qa.jsonl").read_text().splitlines()
                if line.strip() and json.loads(line)["id"] == "p10")
    chunks = build_chunks(root / "corpus", "structure")
    hits = Retriever(chunks, offline=True).search(item["question"], 5)
    holding_prize = [h for h in hits if "Prizes" in h["section"] and item["expect_contains"] in h["text"]]
    assert holding_prize
    assert all(not is_gold(h, item) for h in hits)


def test_negatives_abstain_offline():
    """Regression: n02 leaked once via hash-encoder cosine noise."""
    from pathlib import Path
    from ragqa.ingest import build_chunks
    from ragqa.retrieve import Retriever
    chunks = build_chunks(Path(__file__).resolve().parent.parent / "corpus", "structure")
    r = Retriever(chunks, offline=True)
    for q in ("How much does it cost to register for the competition?",
              "What GPU model runs the match servers?",
              "In which city is the competition's data center located?",
              "What material is the winner's trophy made of?"):
        assert not r.has_evidence(r.search(q, 5), q), f"negative not abstained: {q}"
