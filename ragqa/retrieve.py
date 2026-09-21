"""Hybrid retrieval: BM25 over mixed tokens + dense cosine, fused by RRF."""
import re

import numpy as np
from rank_bm25 import BM25Okapi

from .embed import embed

WORD_RE = re.compile(r"[a-z0-9]+")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text):
    """ASCII words + CJK char bigrams — jieba-free, good enough for rules text."""
    toks = WORD_RE.findall(text.lower())
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        toks += [run] if len(run) == 1 else [run[i:i + 2] for i in range(len(run) - 1)]
    return toks


STOP = {"the", "of", "is", "to", "for", "in", "on", "a", "and", "or", "does",
        "do", "what", "when", "which", "who", "how", "much", "many", "at",
        "with", "by", "are", "was", "it", "its", "be", "any", "you", "your"}


class Retriever:
    def __init__(self, chunks, offline=False):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
        self.vecs, self.encoder = embed([c["text"] for c in chunks], offline=offline)

    def search(self, query, top_k=5):
        n = len(self.chunks)
        bq = self.bm25.get_scores(tokenize(query))
        qq, _ = embed([query], offline=self.encoder == "hash-trigram")
        cos = (self.vecs @ qq[0]).tolist()
        bm_rank = {i: r for r, i in enumerate(np.argsort(-bq))}
        cos_rank = {i: r for r, i in enumerate(np.argsort(-np.asarray(cos)))}
        k = 60.0
        fused = sorted(range(n), key=lambda i: -(1 / (k + bm_rank[i]) + 1 / (k + cos_rank[i])))
        hits = []
        for i in fused[:top_k]:
            c = self.chunks[i]
            hits.append({"id": c["id"], "source": c["source"], "section": c["section"],
                         "text": c["text"], "bm25": float(bq[i]), "cos": float(cos[i])})
        return hits

    def has_evidence(self, hits, query=""):
        """Abstain rule for negatives.

        Lexical BM25 alone failed on English: shared function-adjacent words
        ("competition", "cost") pushed ALL four negatives above the score
        floor. So we add a content-word coverage gate: fraction of non-stopword
        query tokens appearing (substring) in the top chunk must be >= 0.75.
        Both numbers are tuned on THIS 16-item set — recalibrate for a new
        corpus, and expect real embeddings to change the cos scale.
        """
        if not hits:
            return False
        if max(h["bm25"] for h in hits[:2]) < 3.0 and hits[0]["cos"] <= 0.5:
            return False
        words = [t for t in WORD_RE.findall(query.lower()) if t not in STOP]
        if not words:
            return True
        cov = sum(t in hits[0]["text"].lower() for t in words) / len(words)
        return cov >= 0.75
