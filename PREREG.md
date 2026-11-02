# 预注册规格 — rag-qa 竞赛规则问答 RAG 评测

> 这份文档把本仓**真实**的假设、评测方案、成功标准与边界写下来，独立于任何
> "能力声明"。所有数字均来自本仓可复算的离线冒烟（`run_eval.py --offline`）与
> pytest 套件，不虚构。

## 1. 研究对象

本仓是**小体量但完整的 RAG 管线**：混合检索（BM25 ⊕ 向量，RRF 融合）、
强制引用生成、拒答判定，以及把**检索与生成拆开打分**的评测集。
语料 `corpus/official/` 是官方页面带时间戳的抄本（Overview/Evaluation 与 Rules
赛事特定条款，2026-09-20 登录态浏览器一手取证），由 `scripts/build_corpus_from_evidence.py`
切片生成（只加分节标题、去导航噪声，条文语句未改动）。评测集 16 题
（12 正样本 + 4 负样本，`evalset/qa.jsonl`）。

## 2. 假设（H1–H5）

- **H1（检索与生成可分离）**：`recall@5` 只依赖切分策略与编码器，**不**依赖生成
  路径；`answer_hit` 是抽取式下界。因此 `recall@5` 与 `answer_hit` 的**落差本身**
  是生成层待办的证据，而不是生成器噪声掩盖检索差异。
- **H2（混合检索接线正确）**：在官方抄本语料 + hash 三元组编码器 + 抽取式答案的
  离线冒烟下，`--offline` 表在 `fixed`/`structure` 两策略下分别复现
  `recall@5=9/12` 与 `10/12`、`answer_hit=2/12` 与 `3/12`、`neg_abstain=4/4`，
  且与 `results/eval-offline.jsonl`（32 行逐项）逐字一致。
- **H3（拒答判定有效）**：4 个负样本全部 abstain（`neg_abstain=4/4`）。
  这是回归测试 `test_negatives_abstain_offline` 锁死的——n02 曾靠哈希余弦噪声混
  过关（bm25=0 但 cos>旧阈值），修成按分数带标定（内容词覆盖率闸）。
- **H4（切分策略差异不被生成器掩盖）**：`structure` 的 `recall@5` 高于 `fixed`
  （10/12 vs 9/12）且该差异在检索层（与生成路径无关）成立；`p11` 是 structure
  miss / fixed hit（严格反输），`p10` 两者都 miss（平局）。
- **H5（可复算性）**：测试套件 `python3 -m pytest tests/ -q` → 6 passed；
  离线冒烟 `python3 run_eval.py --offline` 零网络可复现上述表。

## 3. 评测方案

### 3.1 离线冒烟（零网络，plumbing check）

```bash
python3 run_eval.py --offline
```

| strategy | encoder | recall@5 | answer_hit | neg_abstain | n_pos | n_neg |
|---|---|---|---|---|---|---|
| fixed | hash-trigram | 9/12 | 2/12 | 4/4 | 12 | 4 |
| structure | hash-trigram | 10/12 | 3/12 | 4/4 | 12 | 4 |

### 3.2 pytest 套件（`tests/test_rag.py`，6 用例）

section 解析、两种切分、tokenize 混合、gold 检索、负样本拒答。真实执行：
`python3 -m pytest tests/ -q` → **6 passed**。

### 3.3 判定标准（成功 = 满足）

| 标准 | 目标 | 当前实测 |
|---|---|---|
| 离线表逐字一致 | fixed 9/12 2/12 4/4；structure 10/12 3/12 4/4 | 复现（本树） |
| 逐项明细 | 32 行与表一致 | 复现（本树） |
| 负样本拒答 | 4/4 abstain | 复现（本树） |
| pytest | 6 passed | 6 passed（本树） |

## 4. 明确边界（不证明什么）

1. **`--offline` 数字不是真实质量**：hash 三元组编码器 + 抽取式答案是"只证明
   接线正确"的 plumbing check 与下界，行带 `encoder` 字段，跨编码器禁止比较。
   上线真编码器/真生成器后必须重跑回填。
2. **不证明跨语言能力**：语料与评测问题都是英文；BM25 与 hash 编码器对中文查询
   无能为力，需要真语义编码器（bge-m3 一类）。
3. **不证明检索质量泛化**：拒答阈值（bm25≥3 + 内容词覆盖率≥0.75）在这 16 题上
   标定，语料或编码器一换必须重标；`has_evidence` docstring 记了两轮踩坑史。
4. **`structure 输给 fixed` 只对 p11 严格成立**：p10 是平局（都 miss）。
5. **`build_corpus_from_evidence.py` 需要外部取证目录**（本仓不含，用 `RAGQA_EVID_DIR`
   指定）；没有该目录它抛 `FileNotFoundError`——如实边界，不是缺陷。

## 5. 预注册承诺

任何后续**质量**声明（"检索质量 X%""跨语言能力"）必须先在本仓补真实数据
（真语义编码器、真生成器、真实管线基准、重标定阈值）再写；本仓当前只声明
"混合检索接线正确 + 拒答判定有效 + 检索/生成可分离 + 测试可运行"，且明确
该声明**不**外推到真实检索质量或跨语言能力。
