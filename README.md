# rag-qa —— 竞赛规则问答 + 引用溯源 + 可分离评测

[![CI](https://github.com/aidless/rag-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/aidless/rag-qa/actions/workflows/ci.yml)

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

## 相关工作与本仓差异

本仓的定位是**竞赛规则问答的 RAG 评测**，不是通用 RAG 框架。与三种常见替代
相比，它的区别是：

1. **检索/生成可分离打分**：`recall@5` 与生成路径无关，`answer_hit` 是抽取式
   下界——比"只报一个端到端准确率"的 RAG 基准更能定位失败在检索层还是生成层
   （本仓用它证明 `structure` 与 `fixed` 的差异在检索层成立，`p11` 严格反输）。
2. **拒答判定而非硬答**：负样本不是"答错扣分"而是 `EVIDENCE_MISSING` 拒答，
   且拒答规则（BM25 分带 + 内容词覆盖率闸）标定在本 16 题上、由回归测试锁死。
3. **一手官方抄本语料 + 强制引用**：语料是带时间戳的官方页面抄本（2026-09-20
   取证），回答强制带 `[S#]` 引用，可追溯到语料来源。

诚实对比：通用 RAG 基准（如端到端准确率或仅 retrieval@k）不分离两层、把
retrieval 失败混进 answer 分；本仓牺牲了"单一漂亮数字"，换取**可定位**的
失败归因。它**不**声称检索质量更高——离线数字只是 plumbing check 与下界。

## 证明什么 / 不证明什么

**证明什么（本仓内可复现）：**

1. **混合检索接线正确**：`--offline` 冒烟表在 `fixed`/`structure` 两种切分策略下
   分别复现 `recall@5 = 9/12` 与 `10/12`、`answer_hit = 2/12` 与 `3/12`、
   `neg_abstain = 4/4`（见下方复算）。该表与 `results/eval-offline.jsonl`
   （32 行逐项明细）逐字一致。
2. **拒答判定有效**：4 个负样本全部 abstain（`neg_abstain = 4/4`）。这是回归测试
   `test_negatives_abstain_offline` 锁死的——n02 曾靠哈希余弦噪声混过关（bm25=0
   但 cos>旧阈值），修成按分数带标定（内容词覆盖率闸）。
3. **检索与生成可分离**：`recall@5` 只依赖切分策略与编码器，与生成路径无关；
   `answer_hit` 是抽取式下界。`recall` 与 `answer_hit` 的**落差本身**才是生成层
   待办存在的证据（`structure`：10/12 recall vs 3/12 answer_hit）。
4. **测试可运行**：`python3 -m pytest tests/ -q` → **6 passed**（`test_rag.py`：
   section 解析、两种切分、tokenize 混合、gold 检索、负样本拒答）。

**不证明什么（诚实边界）：**

- **`--offline` 数字不是真实质量**：hash 三元组编码器 + 抽取式答案是"只证明接线
  正确"的 plumbing check 与下界，行带 `encoder` 字段，跨编码器禁止比较。上线真
  编码器/真生成器后必须重跑回填。
- **不证明跨语言能力**：语料与评测问题都是英文；BM25 与 hash 编码器对中文查询
  无能为力，需要真语义编码器（bge-m3 一类）。
- **不证明检索质量泛化**：拒答阈值在这 16 题上标定，语料或编码器一换必须重标；
  `has_evidence` docstring 记了两轮踩坑史。
- **`structure 输给 fixed` 只对 p11 严格成立**：离线结果中 `p11` 是 structure
  miss / fixed hit（严格输）；`p10` 是两者都 miss（平局）。README 的失败模式表说
  "奖金题 structure 反输 fixed"——精确说是 **p11 严格反输，p10 平局**。
- **`build_corpus_from_evidence.py` 需要外部取证目录**（本仓不含，用 `RAGQA_EVID_DIR`
  指定）。没有该目录它抛 `FileNotFoundError`——这是如实边界，不是缺陷。

## 复算（所有数字从本仓重算）

```bash
# 测试（6 passed）
python3 -m pytest tests/ -q

# 离线冒烟（重现 fixed 9/12 2/12 4/4；structure 10/12 3/12 4/4）
python3 run_eval.py --offline

# 逐项明细（32 行，含每题的 recall@5 / answer_contains / abstained）
cat results/eval-offline.jsonl

# 单题检索（离线）
python3 run_qa.py "How many agents can a team submit per day?" --offline
```

> 注：`build_corpus_from_evidence.py` 需外部证据目录，不在本仓；`corpus/official/`
> 的两个抄本（`overview_20260920.md`、`rules_20260920.md`）已带时间戳与来源页头注，
> 是当前语料的一手来源。

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

## Fail-closed 行为

检索/生成链路是**fail-closed** 的：`has_evidence()` 不过就直接返回
`EVIDENCE_MISSING`（拒答），**绝不**拿弱证据硬答。这是 `test_negatives_abstain_offline`
锁死的行为——4 个负样本全部 abstain（`neg_abstain=4/4`），n02 曾经靠哈希余弦
噪声混过关被修成按分数带标定。换句话说：**证据不足时系统选择"不说"，而不是
"乱说"**；`run_qa.py` 对 `EVIDENCE_MISSING` 的默认输出是明确的拒答消息而非猜测。


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
