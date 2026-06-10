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

UBG 没有训练权重时会使用启发式 fallback。因此当前实现应理解为“默认 UBG，支持 RRF 显式降级”，而不是已充分训练的学习型融合器。

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
- 不做 OSS 或长期图片资产管理。
- 不保证所有截图稳定命中；图表、页面截图、含文字截图效果更稳定。
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
