# VectorBridge RAG 架构

> 更新于 2026-06-10。本文档描述当前运行时架构、检索管线和非目标边界。

## 1. 请求路径

当前普通知识库问题默认走 direct RAG，而不是完全交给 Agent 自由决定是否调用工具。

```text
前端 /chat/stream
  -> ChatRequest(message, session_id, query_image_base64?)
  -> 保存查询图片（可选）
  -> chat_with_agent_stream()
  -> 追问识别与 query 补全
  -> run_rag_graph()
  -> 检索、合并、盲区判断
  -> direct answer prompt
  -> FAST_MODEL / MODEL 生成
  -> trace 写入会话
```

天气等明确外部工具问题仍可走 Agent 工具路径。

## 2. 模型分工

| 角色 | 环境变量 | 当前用途 |
| --- | --- | --- |
| 主模型 | `MODEL` | Agent 工具路径、可选 direct RAG 生成 |
| 快速模型 | `FAST_MODEL` | 代码默认 direct RAG 生成、标题、部分轻量任务 |
| 评分模型 | `GRADE_MODEL` | 文档相关性判定 |
| 辅助模型 | `AUX_MODEL` | 幻觉检测、冲突检测、持久化笔记压缩 |

direct RAG 相关变量：

```env
DIRECT_RAG_ANSWER_MODEL=fast
DIRECT_RAG_ANSWER_TIMEOUT=45
DIRECT_RAG_REWRITE_TIMEOUT=20
```

工程选择：代码默认用 `FAST_MODEL` 生成知识库答案，以降低长上下文生成超时的概率。`DIRECT_RAG_ANSWER_MODEL=main` 可用于切换到主模型做质量对比。

## 3. 文档入库架构

```text
UploadFile
  -> sanitize_upload_filename()
  -> data/documents/{filename}
  -> cleanup same filename
  -> DocumentLoader
  -> semantic preprocessing / chunk headers
  -> hierarchical chunks
  -> parent_chunks PostgreSQL
  -> dense / sparse / image_dense
  -> Milvus
```

关键表和存储：

- PostgreSQL `parent_chunks`：L1/L2 父级块。
- Milvus `embeddings_collection`：L3 叶子块和向量字段。
- `data/bm25_state.json`：BM25 统计。
- `data/images/`：PDF 提取图片。
- `data/query_images/`：用户查询图片临时文件。

## 4. 检索架构

```text
query
  -> intent / complexity
  -> RAG-Fusion variants
  -> dense + sparse hybrid recall
  -> UBG fusion by default
  -> optional image_dense recall
  -> L3 leaf candidates
  -> auto-merging to parent chunks
  -> blindspot threshold
  -> final top_k docs
```

默认融合：

```env
RETRIEVAL_FUSION_METHOD=ubg
```

可降级：

```env
RETRIEVAL_FUSION_METHOD=rrf
```

UBG 没有训练权重时会使用启发式 fallback。因此当前实现应理解为”默认 UBG，支持 RRF 显式降级”，而不是已充分训练的学习型融合器。

### 4.1 Conformal 盲区检测

检索融合后，系统需要判断当前检索结果是否可信。传统做法是设固定阈值（如 `BLINDSPOT_MIN_SCORE=0.3`），低于阈值则拒答或降级。但固定阈值的问题是：无法知道”0.3”在不同查询、不同知识库状态下实际意味着什么。

Conformal prediction 提供了一种可校准的替代方案：

**机制**：

1. **校准集**：从已入库文档中采样 N 条查询，对每条查询执行检索，记录其融合分数（UBG 或 RRF 输出）。
2. **非一致性分数**：对每条校准查询，定义非一致性分数 αᵢ = 1 - retrieval_scoreᵢ，表示”检索结果偏离理想匹配的程度”。检索分数越高 → α 越小 → 越”一致”。
3. **标定阈值**：取校准集非一致性分数的 (1-ε) 经验分位数作为阈值 Q₁₋ε。ε 为允许的错误率（如 ε=0.1 对应 90% 目标覆盖率）。
4. **在线判盲**：对新的检索请求，若 1 - fusion_score > Q₁₋ε，则判为盲区，触发拒答或降级重写。
5. **覆盖性解释**：在校准集与在线请求近似可交换的前提下，判为”非盲区”的检索结果可按覆盖率目标 ≥ 1-ε 理解。该解释依赖校准数据质量和线上查询分布。

**实现细节**：

- 校准数据保存于 `data/conformal_state.json`，包含标定阈值和校准统计量。
- 若该文件不存在，回退到 `BLINDSPOT_MIN_SCORE` 固定阈值。
- 校准通过 `scripts/calibrate_conformal.py --cal-size 500` 执行。
- 固定阈值模式下不会加载校准状态，运行时按 `BLINDSPOT_MIN_SCORE` 判盲；如需在 trace 中显式展示 `calibrated=false`，需要额外扩展 trace 字段。

**校准集构建**：

校准集并非每次检索时动态生成，而是一次性离线构建：

1. **数据来源**：从 CRUD_RAG 和 RAGEval 的 gold 标注条目中 `random.shuffle` 后截取前 `--cal-size`（默认 500）条。
2. **逐条跑检索**：对每条校准样本，用其 `question` 做 hybrid 检索（dense + sparse），取 top-5 结果，记录 `max_score`（最高融合分数）和 `hit`（top-5 中是否有 chunk_id 命中 gold 的 `relevant_chunk_ids`）。
3. **非一致性分数**：命中 → `1 - max_score`（分数越低越不确定），未命中 → `1.0`（上限惩罚）。
4. **分位数阈值**：排序后取第 `⌈(n+1)(1-α)⌉` 个作为阈值 Q。α=0.10 → 90% 目标覆盖率。
5. **落盘**：Q、α、校准集大小写入 `data/conformal_state.json`。服务启动时 `load_calibrator()` 加载到内存，在线推理只需一次浮点比较，无额外开销。
6. **验证**：超出 `cal_size` 的剩余条目作为测试集，在校准完成后输出测试覆盖率，用于判断是否接近 90% 目标。

**实际效果**：

Conformal 阈值替代经验固定阈值，在实际使用中有四个好处：

- **阈值自动适配**：固定阈值 0.3 是手动试出来的，换 embedding 模型或分块策略后可能失效。Conformal 重跑校准脚本即可从数据中标定新阈值。
- **覆盖率可预期**：固定阈值下不知道假阴性率（多少”该过的检索被误拒”）。Conformal 校准提供了明确的目标覆盖率口径，实际结果仍需测试集验证。
- **漂移可感知**：大量新文档入库后，固定阈值静默退化。`conformal_state.json` 的时间戳提供了一个明确的”该重校准”信号。
- **透明度**：可以明确说明盲区阈值的统计含义——“α=0.10，校准集 N=500，分位数 Q=X.XX”——而不是”我们设了个 0.3”。

该覆盖性解释的前提是校准集与在线查询分布近似可交换。当用户查询风格与 gold 标注差异过大、或知识库内容剧烈变化时，覆盖率会偏离标称值。

### 4.2 与 CRANE 项目的关系

VectorBridge 借鉴了 [CRANE](https://github.com/fsvov/CRANE)（多模态情感分析可信推理框架）中的两个思想：UBG 多模态融合和 Conformal 置信度校准。两者不是同一任务，也不共享模型权重；这里的关系是设计灵感和方法迁移。

两者的“多模态”含义并不相同。CRANE 处理的是同一个情感分析样本中的文本和音频联合预测；VectorBridge 处理的是知识库中不同内容载体的统一召回，包括文本块、PDF 页面图片、截图和 OCR 文本。两者共享的抽象问题是：不同信号来源可靠性不同，因此需要融合机制处理权重差异，并用校准机制控制低置信输出。

**多模态架构对比**：

| 维度 | CRANE | VectorBridge |
| --- | --- | --- |
| 模态 1 | 文本 (RoBERTa, 768-dim) | 稠密向量 (bge-m3, 1024-dim) |
| 模态 2 | 音频 (Data2Vec, 768-dim) | 稀疏向量 (BM25, 动态维度) |
| 模态 3 | — | 图片 (CLIP ViT-B/32, 512-dim) |
| 融合方式 | UBG 端到端可训练门控 | UBG 启发式权重 + 可选训练 |
| 权重示例 | CRANE 项目实验中观察到 text 0.98 / audio 0.37 | VectorBridge 启发式 fallback 默认约 dense 0.5 / sparse 0.4 / image 0.1 |
| 关键发现 | 文本远超音频，模型自主学习到此 | 稠密 + 稀疏互补，图片为辅助信号 |
| 不确定性 | MC Dropout 10 次推理 | Conformal 盲区阈值 |

两项目的 UBG 门控机制理念相似：为不同信息源分配动态权重，使更可靠的信号在融合中占更大比例。CRANE 端到端学习模态权重；VectorBridge 当前公开默认是启发式 fallback，可选加载训练权重。因此 VectorBridge 中的 UBG 更准确地说是 UBG-style retrieval fusion，而不是 CRANE 中完整神经门控模块的复刻。

**Conformal 应用对比**：

| 维度 | CRANE | VectorBridge |
| --- | --- | --- |
| 任务 | 情感分数回归 | RAG 检索二值判盲 |
| 非一致性 | `|ŷ - y| / uncertainty` | `1 - retrieval_score`（命中时），`1.0`（未命中） |
| 输出 | 预测区间 [ŷ-w, ŷ+w] | 盲区阈值 Q |
| 校准集 | 训练集外的 MOSI val 子集 (~53 条) | CRUD_RAG + RAGEval 随机抽取 (~500 条) |
| 覆盖口径 | CRANE 项目实验中使用预测区间覆盖目标 | VectorBridge 可用校准状态文件设定检索盲区目标覆盖率 |
| 端到端角色 | 后处理校准，不修改模型 | 检索后门控，不修改检索流程 |

尽管任务不同，两者都采用了 conformal calibration 的基本流程：划分校准集、定义非一致性分数、取经验分位数、在运行时应用阈值。这种跨领域可迁移性，是 Conformal Prediction 作为无模型校准工具的核心优势。

需要注意，VectorBridge 的 Conformal 校准不保证最终答案一定正确，而是用于降低“低置信检索仍被当作有依据回答”的风险。最终回答质量仍取决于召回是否命中、上下文排序是否合理、以及生成模型是否忠实使用检索上下文。

**职责分离的发现**：

CRANE 项目实验中观察到 UBG 门控置信度与 Conformal 宽度相关性较弱（text r≈0.13, audio r≈-0.04），由此可以抽象出一个三层职责分工：

1. **UBG 门控** → 模态选择：当前该信哪一路
2. **MC Dropout** → 不确定性估计：预测有多摇摆
3. **Conformal** → 校准门槛：给结果加一个数据驱动的阈值

VectorBridge 保持了类似的分工：UBG 决定检索融合比例，检索分数本身作为置信信号，Conformal 不修改检索逻辑，只在最后加一道校准后的判盲门槛。这让融合策略和校准策略可以相对独立地替换。

**迁移路径总结**：

```text
CRANE (情感分析)                        VectorBridge (RAG)
─────────────────                       ─────────────────
文本 + 音频 + UBG ────────────────→ 稠密 + 稀疏 + 图片 + UBG
MC Dropout 不确定性 ─────────────→ 检索融合分数
Conformal 覆盖区间 ──────────────→ Conformal 盲区阈值
“融合 → 校准” 两阶段范式 ──────────→ 同一范式，不同领域
```

两个项目共同体现了一个可迁移的设计思路：UBG 负责“信哪路”，Conformal 负责“阈值如何校准”，两者解耦且可分别替换。

从 CRANE 到 VectorBridge，迁移的不是具体模型，而是可信多模态系统的设计范式：先把不同来源的信息编码为可比较的表示，再用融合机制处理信号可靠性差异，最后用校准或阈值机制控制不确定输出。CRANE 在文本-音频情感分析中验证了这一范式，VectorBridge 则把它迁移到文本-图片 RAG 检索链路中。

## 5. 图片查询架构

图片查询是最小可用闭环：

```text
前端 FileReader
  -> base64 data URL
  -> ChatRequest.query_image_base64
  -> _save_query_image()
  -> ContextVar query_image_path
  -> CLIP embed_image()
  -> Milvus image_dense search
  -> OCR fallback
  -> image_matches + OCR text in rag_trace
  -> direct RAG prompt
```

关键边界：

- 不做人脸识别，只做图片相似检索和附近内容返回。
- 不做通用 VQA；系统不会直接理解任意图片语义，主要匹配“这张截图可能来自哪个已入库页面、图表或文本区域”。
- 不做 OSS 或长期图片资产管理。
- 不保证所有截图稳定命中；图表、页面截图、含文字截图效果更稳定，人物头像截图能否命中取决于原 PDF 页面图像、邻近文本和 OCR/CLIP 特征是否被成功入库。
- OCR 需要 Tesseract 可执行程序。

## 6. 多轮追问架构

每轮成功 RAG 后记录：

```json
{
  "last_rag_topic": {
    "user_question": "...",
    "effective_query": "...",
    "answer_excerpt": "...",
    "filenames": ["..."]
  }
}
```

追问处理：

- 短追问或上下文依赖问题会基于 `last_rag_topic` 补全 query。
- 不使用 `filenames` 做过滤。
- 仍然全库检索，避免错误地把对话锁死在上一文档。

## 7. Trace 与并发状态

RAG 过程通过 SSE 向前端输出步骤。最新实现已将下面状态从模块级全局变量改为 `contextvars`：

- `rag_step_queue`
- `rag_step_loop`
- `last_rag_context`

流式 direct RAG 在线程池执行时会复制当前 context，避免不同请求之间串 trace 或丢 trace。

## 8. 安全与权限边界

已完成：

- JWT 登录鉴权。
- 管理员才能上传/删除文档。
- 普通用户只能访问自己的会话。
- 上传文件名清洗，避免路径穿越。
- Milvus filename filter 转义。

不作为已完成：

- 文档级 ACL。
- 文件病毒扫描。
- 生产级 PII 脱敏。
- Prompt Injection 阻断。
- 多租户资源隔离。

## 9. 关键环境变量

| 变量 | 说明 |
| --- | --- |
| `MODEL` / `FAST_MODEL` / `GRADE_MODEL` / `AUX_MODEL` | LLM 分工 |
| `DIRECT_RAG_ANSWER_MODEL` | direct RAG 使用 fast 或 main |
| `EMBEDDING_MODEL` | 文本 embedding，默认 bge-m3 |
| `EMBEDDING_DEVICE` | `auto` / `cpu` / `cuda` |
| `MULTIMODAL_EMBEDDING_MODEL` | CLIP 模型 |
| `MULTIMODAL_EMBEDDING_DEVICE` | 图片模型设备 |
| `OCR_ENABLED` / `OCR_LANG` / `TESSERACT_CMD` | OCR 配置 |
| `RETRIEVAL_FUSION_METHOD` | `ubg` 或 `rrf` |
| `RETRIEVAL_CANDIDATE_K` / `RETRIEVAL_CANDIDATE_MULTIPLIER` | 候选池大小 |
| `FINAL_TOP_K` | 最终送入生成模型的片段数 |
| `IMAGE_RETRIEVAL_MIN_SCORE` | 图片检索主阈值 |
| `IMAGE_RETRIEVAL_FALLBACK_MIN_SCORE` | 图片检索兜底阈值 |
| `DATABASE_URL` / `REDIS_URL` | PostgreSQL / Redis |
| `MILVUS_HOST` / `MILVUS_PORT` / `MILVUS_COLLECTION` | Milvus |

## 10. 已知限制

- CDN 依赖尚未本地化。
- 图片查询仍是最小可用，不是完整视觉问答。
- OCR 安装依赖需要用户手动配置。
- 专用 reranker 当前不是稳定默认链路，只保留可选接口和评估入口。
- Conformal 校准需要固定校准集和状态文件，否则只是固定阈值回退。
- 已有自动检索评估和 LLM Judge 抽样脚本，但仍缺少一键端到端评估流水线。
