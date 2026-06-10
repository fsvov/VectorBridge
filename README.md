# VectorBridge

本地多模态 RAG 系统 — 文本+图片混合检索、文档上传入库、多轮追问、SSE 流式输出。

## 文档入口

长期维护文档在 `docs/` 目录下：

- [项目总览](docs/project-overview.md)：主链路、运行方式、GPU/OCR 配置、已知边界。
- [RAG 架构](docs/rag-architecture.md)：direct RAG、文档入库、图片查询、多轮追问、trace 架构。
- [评估结果与边界](docs/evaluation-and-limitations.md)：自动评估结果、人工测试结论和已知限制。

当前项目口径：

- 已修通文档上传入库、父子分块、hybrid/UBG 检索、图片截图查询、多轮追问和前端 trace 展示。
- 图片查询是“截图相似检索 + OCR fallback”的最小可用能力，用于匹配截图来自哪个已入库页面、图表或文本区域；不是通用视觉问答或人脸识别。
- 本项目不是生产级安全系统；文档级权限、文件扫描、PII 脱敏等作为未来工程化方向。

快速启动：

```bash
# 启动依赖服务
docker compose up -d

# 启动后端
uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

## 本地部署

### 1) 环境准备
- Python `3.12+`
- 包管理建议：`uv`（也支持 `pip`）
- Docker / Docker Compose（用于启动 Milvus 依赖）

### 2) 使用 pyproject 安装依赖
在项目根目录执行：

```bash
# 方式 A：推荐（uv）
uv sync

# 运行服务
uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

```bash
# 方式 B：pip
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .

# 运行服务
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

### 3) 创建 `.env` 文件

```bash
cp .env.example .env
```

按需编辑 `.env` 中的 API Key、模型名与连接地址；变量说明见 `.env.example` 内注释。

至少需要修改以下安全相关配置后再对外使用：

- `JWT_SECRET_KEY`：必须替换为足够长的随机字符串，不要使用示例值。
- `ADMIN_INVITE_CODE`：用于注册管理员账号，必须替换为私有邀请码；公开仓库中的 `.env.example` 仅提供占位符。

### 首次使用流程

1. 启动依赖服务：`docker compose up -d`。
2. 安装依赖：`uv sync`。
3. 复制并编辑环境变量：`cp .env.example .env`。
4. 启动后端：`uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload`。
5. 打开 `http://127.0.0.1:8000/`，使用 `ADMIN_INVITE_CODE` 注册管理员账号。
6. 上传文档，等待入库完成后开始提问。

### 4) Docker 部署（数据库 + 缓存 + 向量库）
当前仓库的 `docker-compose.yml` 同时承载业务依赖与 Milvus 依赖：
- 业务依赖：`postgres`、`redis`
- 向量依赖：`etcd`、`minio`、`standalone`、`attu`

```bash
# 启动向量库依赖
docker compose up -d

# 查看服务状态
docker compose ps

# 查看日志（可选）
docker compose logs -f standalone
```

端口说明：
- PostgreSQL：`5432`
- Redis：`6379`
- Milvus：`19530`
- Milvus 健康检查：`9091`
- MinIO API：`9000`
- MinIO Console：`9001`
- Attu：`8080`

### 5) 启动应用并访问
在 Milvus 启动后，运行后端应用：

```bash
uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

浏览器访问：
- 前端页面：`http://127.0.0.1:8000/`
- API 文档：`http://127.0.0.1:8000/docs`

## 项目概览
- **核心能力**：
  - direct RAG 主路径 + 可选 LangChain Agent 工具路径。
  - 文档上传后执行三级滑动窗口分块，叶子分块向量化写入 Milvus，父级分块写入 PostgreSQL。
  - 用户注册/登录、JWT 鉴权、基于角色的 RBAC 权限控制（admin/user）。
  - 会话记忆与摘要，聊天与历史记录落地 PostgreSQL，并引入 Redis 缓存热点会话与父文档。
- **运行形态**：FastAPI 后端 + 纯前端（Vue 3 CDN 单页）+ Milvus 向量库。

## 关键设计
- **三塔混合检索**：稠密向量 (bge-m3) + BM25 稀疏向量 + CLIP 图片向量，默认 UBG 加权融合，可降级为 RRF。
- **三级分块 + Auto-merging + Leaf-only 存储**：L1/L2/L3 三层滑窗切分，仅叶子块入库 Milvus，检索时自动合并父块上下文，减少向量冗余。
- **流式可观测 RAG**：`contextvars` 请求级隔离 + SSE 流式推送，检索、评分、重写等步骤在”思考中”阶段即可前端实时展示，思考→回答同一气泡无缝切换。
- **查询重写与相关性门控**：Step-Back / HyDE 双策略重写 + 结构化评分门控，低相关检索自动触发二次召回。
- **BM25 统计持久化与增量同步**：词表、文档频次、全局统计落盘，入库增量添加、删除按文件名从 Milvus 拉取文本后增量扣减，保证向量库与稀疏统计一致。
- **多模态图片查询**：CLIP 图片 embedding → Milvus image_dense 检索 + OCR fallback，前端 base64 上传后端解码，不依赖外部 VQA API。当前目标是截图相似匹配和附近文本返回，不做开放域图像问答。
- **Conformal 盲区检测**：支持用 conformal prediction 标定检索置信度阈值，替代固定经验阈值；在存在校准状态文件且校准集与线上查询分布相近时，可按 α=0.10 解释为 90% 目标覆盖率。该校准作用于检索盲区判断，不保证最终答案一定正确。相关概念源自 [CRANE](https://github.com/fsvov/CRANE) 项目的可信推理方法论。

## 参考项目与数据集

### 参考项目

| 项目 | 借鉴内容                                                                                                                             |
|------|----------------------------------------------------------------------------------------------------------------------------------|
| [CRANE](https://github.com/fsvov/CRANE) | 多模态可信推理方法论：UBG 门控融合 + Conformal Prediction 校准，从文本-音频情感分析迁移为文本-图片 RAG 检索中的融合与校准思路；两者不共享模型或任务，详见 [RAG架构](docs/rag-architecture.md) |
| [Rag_System](https://github.com/CliffsCai/Rag_System) | 多模态 RAG 架构参考：图片 base64 传输方案、三路混合检索、Milvus image_dense 字段设计、跨模态查询流程                                                               |
| [SuperMew](https://github.com/icey1287/SuperMew) | 原始纯文本 RAG 基础，VectorBridge 前身：LangGraph 检索管线、Hybrid Search + RRF、三级分块 + Auto-merging、SSE 流式输出                                     |

### 评估数据集

| 数据集 | 规模 | 内容 |
|------|------|------|
| [CRUD_RAG](https://github.com/IAAR-Shanghai/CRUD_RAG) | 2,394 条 | 中文新闻 QA，单文档/双文档/三文档问答 |
| [RAGEval](https://github.com/OpenBMB/RAGEval) | 6,709 条 | 金融/法律/医疗跨领域中英 QA，合成文档 + 真实标注 |

两个数据集合计 9,103 条 gold 标注样本，用于检索对比评估和 LLM Judge 质量评估。

## 目录与架构

- 后端：`backend/`（分层包结构，统一 `from backend.xxx import`）
  - [app.py](backend/app.py)：FastAPI 入口、CORS、静态资源挂载。
  - `api/`：HTTP 层（`router.py` 路由聚合；`routes/` 下 auth/sessions/chat/documents 分文件；`resources.py` 共享资源）。
  - `chat/`：对话域（`service.py` 聊天入口、`runtime.py` Agent 实例、`storage.py` 会话持久化、`streaming.py` RAG 步骤 SSE 推送、`rag_context.py` trace 暂存）。
  - `rag/`：检索增强（`pipeline.py` LangGraph 工作流、`utils.py` 混合检索/rerank/Auto-merging）。
  - `indexing/`：文档入库与向量（`embedding.py`、`document_loader.py`、`milvus_client.py`、`milvus_writer.py`、`parent_chunk_store.py`）。
  - `tools/`：LangChain Agent 可调用的 `@tool`（天气、知识库检索）。
  - `infra/`：`database.py`、`cache.py`、`auth.py`。
  - `db/`：`models.py` SQLAlchemy ORM 模型。
  - `schemas/`：Pydantic 请求/响应（auth/chat/documents）。
  - `jobs/`：`upload_jobs.py` 异步上传/删除任务进度。
- 前端：`frontend/`
  - `index.html` + `script.js` + `style.css`：Vue 3 + marked + highlight.js，提供聊天、历史会话、文档上传/删除界面。
- 数据：`data/`
  - `bm25_state.json`：BM25 词表与统计（稀疏检索 IDF 与入库、删除同步）。
  - `documents/`：上传文档原文件。
- 向量库：Milvus（可由 `docker-compose` 或自建服务提供）。

## 核心流程

### 1) 项目全链路（端到端）
1. 用户在前端输入问题，调用 `POST /chat/stream`（流式）。
2. FastAPI `api/routes/chat.py` 返回 `StreamingResponse(media_type="text/event-stream")`。
3. 知识库问题默认进入 direct RAG；天气等明确外部工具问题仍可走 Agent 工具路径。
4. direct RAG 进入 `backend/rag/pipeline.py` 执行检索工作流，各阶段通过 `emit_rag_step()` 实时推送到前端。
5. 检索结果与 RAG Trace 一起返回，direct answer 模型流式生成最终回答（逐 token 推送）。
6. 前端 ReadableStream 逐块解析 SSE，打字机效果实时渲染。
7. 同时消息持久化到 PostgreSQL，并通过 Redis 缓存加速历史会话回放。

### 2) RAG 全链路
1. **初次召回**：`retrieve_initial` — Dense + Sparse 混合检索，默认 UBG 融合，可配置 RRF 降级；L3 叶子召回后 Auto-merging 到父块。
2. **相关性打分门控**：`grade_documents` — 结构化输出 yes/no；yes 直接生成，no 进入重写。
3. **查询重写路由**：`rewrite_question` — Step-Back / HyDE 策略选择。
4. **二次召回**：`retrieve_expanded` — 对重写后的查询再次检索，按 chunk_id 去重。
5. **答案生成**：direct answer 模型结合上下文生成最终回答。
6. **可观测追踪**：返回 `rag_trace`，包含评分、重写、检索结果、合并信息。

### 3) 文档入库链路
1. 前端上传 PDF/Word 到 `POST /documents/upload`。
2. 若同名文件已存在：先从 Milvus 分页查询该文件全部叶子 chunk 的 text，对 BM25 统计执行 increment_remove，再删除旧向量与父块缓存。
3. `document_loader.py` 执行三级滑动窗口分块并写入层级元数据。
4. L1/L2 父级分块写入 `parent_chunk_store.py`（DocStore）。
5. L3 叶子分块在 `milvus_writer` 中执行 BM25 increment_add，再经 `embedding.py` 生成 Dense 与 Sparse 向量并写入 Milvus。

### 4) 会话记忆链路
1. 每轮问答按当前登录用户 + session_id 写入 PostgreSQL。
2. 当消息过长时触发摘要压缩，保留长期上下文。
3. Redis 缓存会话列表与会话消息，减少高频读取数据库压力。
4. 前端可通过会话接口读取、删除当前用户自己的历史对话。

## 技术栈

- 后端：FastAPI、LangChain / LangGraph、Pydantic、Uvicorn、SQLAlchemy、PostgreSQL、Redis。
- 向量与检索：Milvus（HNSW 稠密索引 + SPARSE_INVERTED_INDEX 稀疏索引）、UBG/RRF 融合、可选 rerank。
- 嵌入与稀疏：`langchain_huggingface` 本地稠密向量（默认 `BAAI/bge-m3`）；中英混合规则分词 + BM25 手写稀疏向量，统计持久化至 `bm25_state.json`。
- 前端：Vue 3 (CDN)、marked、highlight.js、纯静态部署。

## 环境变量

需在仓库根目录或运行环境配置（详见 `.env.example`）：
- 模型相关：`LLM_API_KEY`、`MODEL`、`BASE_URL`
- 稠密向量：`EMBEDDING_MODEL`、`EMBEDDING_DEVICE`、`DENSE_EMBEDDING_DIM`
- 多模态：`MULTIMODAL_EMBEDDING_MODEL`、`MULTIMODAL_EMBEDDING_DEVICE`、`IMAGE_VECTOR_DIM`
- OCR：`OCR_ENABLED`、`OCR_LANG`、`TESSERACT_CMD`、`OCR_MAX_CHARS`
- 检索融合：`RETRIEVAL_FUSION_METHOD`（ubg / rrf）、`RETRIEVAL_CANDIDATE_K`、`FINAL_TOP_K`
- Auto-merging：`AUTO_MERGE_ENABLED`、`AUTO_MERGE_THRESHOLD`、`LEAF_RETRIEVE_LEVEL`
- Rerank：`RERANK_MODEL`、`RERANK_BINDING_HOST`、`RERANK_API_KEY`（可选）
- Milvus：`MILVUS_HOST`、`MILVUS_PORT`、`MILVUS_COLLECTION`
- 数据库缓存：`DATABASE_URL`、`REDIS_URL`
- 鉴权：`JWT_SECRET_KEY`、`ADMIN_INVITE_CODE`、`JWT_ALGORITHM`、`JWT_EXPIRE_MINUTES`

注意：`JWT_SECRET_KEY` 和 `ADMIN_INVITE_CODE` 不能沿用示例值。前者用于签发登录 token，后者控制管理员注册入口。

OCR 说明：只安装 Python 包 `pytesseract` 不等于启用 OCR。若未安装 Tesseract OCR 可执行程序，图片查询仍可走 CLIP 相似检索，但不会有 OCR 文本增强。

## API 速览

- 鉴权
  - `POST /auth/register`：注册（支持普通用户/管理员邀请码模式）。
  - `POST /auth/login`：登录，返回 Bearer Token。
  - `GET /auth/me`：获取当前登录用户信息。
- 聊天
  - `POST /chat`：聊天（非流式），入参 `message`、`session_id`。
  - `POST /chat/stream`：聊天（流式 SSE），入参同上，返回 `text/event-stream`。
- 会话（用户隔离）
  - `GET /sessions`：列出当前用户会话。
  - `GET /sessions/{session_id}`：拉取当前用户某会话消息。
  - `DELETE /sessions/{session_id}`：删除当前用户会话。
- 文档（管理员权限）
  - `GET /documents`：列出已入库文档及 chunk 数。
  - `POST /documents/upload`：上传并向量化 PDF/Word/Excel。
  - `DELETE /documents/{filename}`：删除指定文档向量数据。

## 流式输出与实时检索过程

### 1. 请求级 Trace 隔离
使用 `contextvars` 保存请求上下文，避免并发请求之间串 trace：
1. 每个 `/chat/stream` 请求创建自己的输出队列。
2. `chat_with_agent_stream()` 设置请求级 RAG step proxy。
3. direct RAG 在线程池执行时复制当前 context，使同步检索代码仍能把步骤写回当前请求队列。
4. 回答完成后发送完整 `rag_trace`，并持久化到当前会话。

相关代码：`backend/chat/streaming.py`（RAG 步骤上下文）、`backend/chat/rag_context.py`（trace 暂存）、`backend/chat/service.py`（流式队列与线程池）。

### 2. 混合检索 (Hybrid Search) 实现
- **Dense Pathway**：`langchain_huggingface.HuggingFaceEmbeddings`（默认 `BAAI/bge-m3`）生成稠密向量。
- **Sparse Pathway**：中英混合规则分词（单字中文 + 英文单词）实现 BM25，统计持久化在 `bm25_state.json`。
- **融合**：Milvus `AnnSearchRequest` 同时发起两路请求，默认 UBG 加权融合，可降级为 RRFRanker (k=60)。

### 3. 前端 Thinking State Machine
1. Idle → 等待用户输入
2. Thinking (Initial) → 创建消息气泡，`isThinking=true`
3. Thinking (Active RAG) → 收到 `rag_step` 事件，动态更新步骤文字
4. Streaming → 收到第一个 `content` token，`isThinking=false`，同一气泡无缝切换为 Markdown 流

从"思考"到"回答"全程在同一个气泡内完成，没有突兀的 UI 抖动。

### 4. 终止功能
- 前端：发送按钮在生成中切换为红色终止按钮，点击调用 `AbortController.abort()`。
- 后端：捕获 `GeneratorExit`，显式 `agent_task.cancel()` 以回收资源并节省 Token 消耗。



