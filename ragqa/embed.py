"""Embeddings: real API or offline hash fallback.

hash_embed is a deterministic char-trigram bag — it is deliberately weak.
It exists so the pipeline runs (and smoke tests pass) with zero network, and
every result row records which encoder produced it. Never compare arms across
encoders.
"""
import hashlib
import os

DIM = 256


def hash_embed(texts, dim=DIM):
    import numpy as np
    vecs = np.zeros((len(texts), dim), dtype="float32")
    for i, t in enumerate(texts):
        t = t.lower()
        grams = [t[j:j + 3] for j in range(max(1, len(t) - 2))]
        for g in grams:
            h = int.from_bytes(hashlib.md5(g.encode()).digest()[:8], "little")
            vecs[i, h % dim] += 1.0 if (h >> 63) & 1 else -1.0
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def api_embed(texts):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["LLM_API_KEY"],
                    base_url=os.environ.get("EMBED_BASE_URL") or os.environ.get("LLM_BASE_URL") or None)
    model = os.environ.get("EMBED_MODEL", "bge-large-zh-v1.5")
    import numpy as np
    out = []
    for i in range(0, len(texts), 32):  # batch to dodge per-request caps
        resp = client.embeddings.create(model=model, input=texts[i:i + 32])
        out.extend(d.embedding for d in resp.data)
    mat = np.asarray(out, dtype="float32")
    return mat / np.linalg.norm(mat, axis=1, keepdims=True)


def embed(texts, offline=False):
    if offline or not os.environ.get("LLM_API_KEY"):
        return hash_embed(texts), "hash-trigram"
    return api_embed(texts), os.environ.get("EMBED_MODEL", "bge-large-zh-v1.5")
