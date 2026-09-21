"""Answer generation: online (LLM) or offline extractive, both cite-able."""
import os

SYSTEM = (
    "你是文档问答助手。只依据给定片段回答，答案中的每个事实句必须以 [S编号] 标注来源；"
    "片段不足以回答时，只输出 EVIDENCE_MISSING，不要猜测。"
)


def _context_block(hits):
    return "\n\n".join(f"[S{i + 1}] 来源: {h['source']} § {h['section']}\n{h['text']}"
                       for i, h in enumerate(hits))


def answer_online(question, hits):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["LLM_API_KEY"],
                    base_url=os.environ.get("LLM_BASE_URL") or None)
    resp = client.chat.completions.create(
        model=os.environ.get("LLM_MODEL", "deepseek-chat"),
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"问题：{question}\n\n片段：\n{_context_block(hits)}"}])
    return resp.choices[0].message.content.strip()


def answer_offline(question, hits, has_evidence):
    """Extractive stand-in: top chunk verbatim (capped) + citation.

    Lets the whole eval run without any API. It can only pass items whose
    fact lives in the top-retrieved chunk, so its scores read as a LOWER
    bound on the online path, never as a competitor to it.
    """
    if not has_evidence:
        return "EVIDENCE_MISSING"
    return hits[0]["text"][:160].strip() + " [S1]"
