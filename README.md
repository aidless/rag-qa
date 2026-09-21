# rag-qa —— 竞赛规则问答 + 引用溯源 + 可分离评测

小体量但完整的 RAG 管线：混合检索（BM25 ⊕ 向量，RRF 融合）、强制引用生成、
拒答判定，以及**把检索和生成拆开打分**的评测集（12 正样本 + 4 负样本）。

语料 `corpus/official/` 是**官方页面的带时间戳抄本**（Overview/Evaluation 与
Rules 的赛事特定条款，2026-09-20 登录态浏览器一手取证，抄录日期与来源页写在
每个文件头部），由 `scripts/build_corpus_from_evidence.py` 切片生成——只加分节
标题、去导航噪声，条文语句未改动。

## 已知边界（会写进面试的实话）

- 评测问题是英文、语料是英文。中文查询跨语言检索需要真语义编码器
  （bge-m3 一类），hash 编码器与 BM25 都对跨语言无能为力——这是升级项不是缺陷掩盖。
- 拒答阈值（bm25≥3 + 内容词覆盖率≥0.75）标定在这 16 题上；语料或编码器一换必须重标。

## 快速开始

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q
.venv/bin/python scripts/build_corpus_from_evidence.py  # 可选：从一手证据重建语料
.venv/bin/python run_eval.py --offline      # 零网络自检
cp .env.example .env                         # 填 OpenAI 兼容端点
.venv/bin/python run_qa.py "How many turns does one game last?"  # 在线生成
.venv/bin/python run_eval.py                # 在线评测
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

## 离线冒烟结果（官方抄本语料）

| strategy | encoder | recall@5 | answer_hit | neg_abstain |
|---|---|---|---|---|
| fixed | hash-trigram | 9/12 | 2/12 | 4/4 |
| structure | hash-trigram | 10/12 | 3/12 | 4/4 |

怎么读这张表：**检索指标（recall@5）与生成路径无关**，切分策略差异不被生成器
噪声掩盖；answer_hit 是抽取式下界（只有 top1 命中且事实在前 800 字符内才可能
通过）——它的绝对值不重要，**与 recall 的落差本身**才是生成层待办存在的证据。
structure 输给 fixed 的题（p10/p11 奖金题）见失败模式表。
上线真编码器/真生成器后重跑并回填此表。

## 评测纪律（写在代码里的约束）

- `--offline` 用 hash 三元组编码器 + 抽取式答案：只证明接线正确，结果行
  带 `encoder` 字段，跨编码器禁止比较。
- 拒答判定 = BM25 分带（≥3）∧ 内容词覆盖率闸（≥0.75），在本 16 题上标定后
  由回归测试锁死；换语料/换真编码器必须重标——`has_evidence` docstring 写了
  两轮踩坑史（先余弦噪声、后英文功能词抬分）。
- 负样本泄漏过一次：n02 靠哈希余弦噪声混过关（bm25=0 但 cos>旧阈值），
  修成按分数带标定并加了回归测试 `test_negatives_abstain_offline`。
- gold 判定 = source 后缀 + section 尾部标题匹配，两种切分共享 heading
  真值映射（`chunk.parse_sections`），否则 recall 对其中一种策略无意义。

## 失败模式手册（边跑边补）

| 症状 | 根因 | 修法 |
|---|---|---|
| recall@5=0 但 answer_hit 高 | source 路径含子目录，gold 精确匹配静默失败 | 后缀匹配 + 指标拆开看，矛盾即报警 |
| 负样本误答（两轮） | ①哈希余弦噪声 ②英文功能词把 BM25 全部推过分底线 | 内容词覆盖率闸（`has_evidence`），回归测试锁死 |
| 中文查询全 miss | BM25/hash 向量都不跨语言 | 语料英文→问题改英文；真语义编码器是升级项 |
| 奖金题 structure 反输 fixed | WINNER'S OBLIGATIONS 巨型块反复含 "prize"，词法霸权 | 段内二次切分或真 embedding；已记待办 |
| 小节标题被吞进邻节、gold 漂移 | `<60` 字符块并入前块时标签没跟着并 | 复合标签 `"A + B"`，endswith 匹配仍成立 |
| 表格行 "# PASS forever after." 被当标题 | markdown 标题正则误吃以 `#` 开头的正文行 | `_is_heading` 加 tab/句号护栏 |

## 待办

- [ ] 评测集扩到 30+10，每条标注出处段落（抄本行号）
- [ ] 跑真 embedding（bge-m3）与 hash 的对照，确认阈值重标定后的表；顺手解锁中文查询
- [ ] 巨型块（>4k 字符）段内二次切分，打穿"词法霸权"
- [ ] LLM-as-judge 忠实度评分接进 run_eval（判分规则写进 README）
