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

    def has_evidence(self, hits):
        """Abstain rule for negatives.

        Thresholds calibrated on THIS small eval set (positives land at
        bm25>=5, negatives<=1.7) — treat as dev-set tuned and recalibrate for
        any new corpus/encoder, esp. real embeddings where the cos scale
        differs.
        """
        return (bool(hits) and (max(h["bm25"] for h in hits[:2]) >= 3.0
                                or hits[0]["cos"] > 0.5))
