"""Load corpus/ into raw documents."""
from pathlib import Path

from .chunk import STRATEGIES


def load_documents(corpus_dir: Path):
    docs = []
    for p in sorted(Path(corpus_dir).rglob("*")):
        if p.suffix.lower() in (".md", ".txt"):
            docs.append({"source": str(p.relative_to(corpus_dir)), "text": p.read_text(encoding="utf-8")})
        elif p.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            text = "\n".join(page.extract_text() or "" for page in PdfReader(str(p)).pages)
            docs.append({"source": str(p.relative_to(corpus_dir)), "text": text})
    return docs


def build_chunks(corpus_dir: Path, strategy="fixed"):
    chunks = []
    for doc in load_documents(corpus_dir):
        for i, c in enumerate(STRATEGIES[strategy](doc["text"])):
            chunks.append({"id": f"{doc['source']}#{i}", "source": doc["source"],
                           "section": c["section"], "text": c["text"]})
    return chunks
