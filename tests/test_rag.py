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


def test_retriever_finds_gold_offline(tmp_path):
    from ragqa.ingest import build_chunks
    from ragqa.retrieve import Retriever
    from pathlib import Path
    chunks = build_chunks(Path(__file__).resolve().parent.parent / "corpus", "structure")
    r = Retriever(chunks, offline=True)
    hits = r.search("每支队伍每天可以提交几次方案", 5)
    assert any(h["source"].endswith("rules.md") and h["section"].endswith("提交配额") for h in hits)


def test_negatives_abstain_offline():
    """Regression: n02 leaked once via hash-encoder cosine noise."""
    from pathlib import Path
    from ragqa.ingest import build_chunks
    from ragqa.retrieve import Retriever
    chunks = build_chunks(Path(__file__).resolve().parent.parent / "corpus", "structure")
    r = Retriever(chunks, offline=True)
    for q in ("本比赛的总奖金额度是多少？", "对局服务器的 GPU 显存规格是什么？",
              "报名费需要在哪个页面缴纳？", "主办方的联系电话是多少？"):
        assert not r.has_evidence(r.search(q, 5)), f"negative not abstained: {q}"
