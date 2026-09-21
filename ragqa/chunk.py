"""Two chunking strategies over markdown/plain text.

parse_sections() is the shared ground truth: every character position maps to
the heading path in effect, so BOTH strategies carry a comparable `section`
label — otherwise recall@k against gold sections is meaningless for one arm.
"""
import re

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
MIN_BLOCK = 60  # chars; smaller blocks merge into the previous one


def parse_sections(text):
    """Return [{section, start, end}] covering the whole text."""
    blocks, stack, cur = [], [], None
    pos = 0
    for line in text.split("\n"):
        m = HEADING_RE.match(line)
        if m:
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            if cur:
                cur["end"] = pos
                blocks.append(cur)
            cur = {"section": " / ".join(t for _, t in stack), "start": pos}
        pos += len(line) + 1
    if cur:
        cur["end"] = len(text)
        blocks.append(cur)
    else:
        blocks = [{"section": "(top)", "start": 0, "end": len(text)}]
    return blocks


def _section_at(blocks, offset):
    for b in blocks:
        if b["start"] <= offset < b["end"]:
            return b["section"]
    return "(top)"


def chunk_structure(text):
    out = []
    for b in parse_sections(text):
        piece = {"section": b["section"], "text": text[b["start"]:b["end"]].strip()}
        if out and len(piece["text"]) < MIN_BLOCK:
            out[-1]["text"] += "\n\n" + piece["text"]
        else:
            out.append(piece)
    return [c for c in out if c["text"]]


def chunk_fixed(text, size=200, overlap=40):
    blocks = parse_sections(text)
    out, start = [], 0
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            out.append({"section": _section_at(blocks, start), "text": piece})
        if start + size >= len(text):
            break
        start += size - overlap
    return out


STRATEGIES = {"fixed": chunk_fixed, "structure": chunk_structure}
