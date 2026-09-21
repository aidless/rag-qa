# rag-qa —— 竞赛规则问答 + 引用溯源 + 可分离评测

小体量但完整的 RAG 管线：混合检索（BM25 ⊕ 向量，RRF 融合）、强制引用生成、
拒答判定，以及**把检索和生成拆开打分**的评测集（12 正样本 + 4 负样本）。

语料 `corpus/sample/` 是**虚构占位内容**，数值不可当真实规则引用；
正式版替换为官方页面抄录并标出处的文本即可，评测集跟着换。

## 快速开始

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q
.venv/bin/python run_eval.py --offline      # 零网络自检
cp .env.example .env                         # 填 OpenAI 兼容端点
.venv/bin/python run_qa.py "每天提交配额几次"  # 在线生成
.venv/bin/python run_eval.py                 # 在线评测
```

## 架构

```
corpus/ ─ ingest ─┬─ chunk.py 两种策略：fixed(200字,重叠40) | structure(按标题)
                  │      └── 两种策略共享同一 heading 映射 → gold 段可对齐
                  └─ 每 chunk 带 {id, source, section, text}
query ─ retrieve.py: BM25(ASCII词+CJK字二元) ⊕ cosine ──RRF──> top-5
      ─ has_evidence() 不过 → 直接 EVIDENCE_MISSING（拒答）
      ─ generate.py: 在线 LLM 强制 [S#] 引用 / 离线抽取式（同一引用格式）
```

## 离线冒烟结果（structure vs fixed，占位语料）

| strategy | encoder | recall@5 | answer_hit | neg_abstain |
|---|---|---|---|---|
| fixed | hash-trigram | 8/12 | 7/12 | 4/4 |
| structure | hash-trigram | 12/12 | 11/12 | 4/4 |

怎么读这张表：**检索指标（recall@5）与生成路径无关**，所以切分策略的差异
不被生成器噪声掩盖；answer_hit 在离线模式是抽取式下界。structure 全面占优
是预期内的（规则类文本段落自洽），但 **fixed 输在何处**（跨段边界切碎
"提交配额"首句）才是面试要讲的故事——上线前用真实语料重跑并回填此表。

## 评测纪律（写在代码里的约束）

- `--offline` 用 hash 三元组编码器 + 抽取式答案：只证明接线正确，结果行
  带 `encoder` 字段，跨编码器禁止比较。
- 拒答阈值（bm25≥3 或 cos>0.5）在本评测集上标定（正样本 bm25≥5、负≤1.7，
  分离清晰），换语料/换真编码器必须重标——`has_evidence` docstring 里写了。
- 负样本泄漏过一次：n02 靠哈希余弦噪声混过关（bm25=0 但 cos>旧阈值），
  修成按分数带标定并加了回归测试 `test_negatives_abstain_offline`。
- gold 判定 = source 后缀 + section 尾部标题匹配，两种切分共享 heading
  真值映射（`chunk.parse_sections`），否则 recall 对其中一种策略无意义。

## 失败模式手册（边跑边补）

| 症状 | 根因 | 修法 |
|---|---|---|
| recall@5=0 但 answer_hit 高 | source 路径含子目录，gold 精确匹配静默失败 | 后缀匹配 + 指标拆开看，矛盾即报警 |
| 负样本误答 | 弱编码器余弦噪声越过低阈值 | 分数带标定阈值 + 回归测试锁死 |
| 表格/代码块被切碎 | fixed 按字数滑窗 | 真语料里给结构化内容走 structure |

## 待办

- [ ] 替换 `corpus/sample/` 为官方规则文本（带抄录日期与 URL）
- [ ] 评测集扩到 30+10，每条标注抄录出处段落
- [ ] 跑真 embedding（bge）与 hash 的对照，确认阈值重标定后的表
- [ ] LLM-as-judge 忠实度评分接进 run_eval（判分规则写进 README）
