# VectorBridge Project Overview

> Updated on 2026-06-10. VectorBridge is a local multimodal RAG system with document ingestion, hybrid retrieval, screenshot-based image retrieval, streaming trace visualization, and evaluation tooling.

## 1. 项目定位

VectorBridge 是在 SuperMew 基础上改造的本地多模态 RAG 系统，面向本地知识库实验、工程展示和 RAG 检索链路验证。

项目定位：

- 适合展示文档上传、分块入库、混合检索、图片截图检索、多轮追问、SSE 可观测和会话持久化。
- 不是生产级多租户、安全审计或文件安全扫描系统。
- 已实现能力、可选增强和已知边界在文档中分开描述。

## 2. 当前主链路

### 2.1 文档入库

网页上传支持 PDF、Word、Excel、HTML。异步上传流程为：

```text
保存文件
  -> 清理同名旧数据（Milvus / PostgreSQL parent_chunks / BM25 统计）
  -> 解析文档
  -> 三级分块
  -> L1/L2 父级块写 PostgreSQL
  -> L3 叶子块写 Milvus
```

已修复的关键稳定性问题：

- 父级分块 `chunk_id` 重复导致 `parent_chunks_pkey` 冲突。
- Milvus client 被关闭后继续请求导致 `Cannot send a request, as the client has been closed`。
- 同名覆盖上传时清理旧向量、父级块、BM25 统计。
- 上传文件名只接受纯文件名，拒绝路径分隔符和控制字符。
- Milvus filename filter 对引号和反斜杠做转义。

### 2.2 文本检索与回答

默认路径已经从 Agent 自由工具选择改为 direct RAG，减少主模型不调用工具或 JSON 解析失败造成的回答失败。

当前文本检索能力：

- 文本 dense embedding：`BAAI/bge-m3`。
- sparse 检索：持久化 BM25 统计。
- 默认融合：`RETRIEVAL_FUSION_METHOD=ubg`。
- 显式降级：`RETRIEVAL_FUSION_METHOD=rrf`。
- L3 叶子块召回后可 Auto-merging 到 L2/L1。
- RAG-Fusion、CRAG、CTX-Enriched、盲区检测等能力保留在检索链路中。
- direct RAG 代码默认使用 `FAST_MODEL` 生成答案，可通过 `DIRECT_RAG_ANSWER_MODEL=main` 切回主模型；本地 `.env` 若设为 `main`，则以本地配置为准。

回答侧工程约束：

- 默认中文回答。
- 要求依据检索片段回答，不输出大段原文堆砌。
- 回答后记录 `rag_trace`，包含检索片段、图片候选、OCR、幻觉检测等信息。
- 当模型超时或失败时，降级为基于片段的摘要，而不是卡死。

### 2.3 多轮追问

多轮追问已接入最小可用方案：

- 每轮成功 RAG 后保存 `last_rag_topic` 到会话 metadata。
- 对“相关会议呢”“还有哪些”“那 B 类呢”等短追问做上下文补全。
- 补全后的 query 仍在全库检索，不强制限制到上一份文档。
- 生成回答时保留用户原始追问语义，补全 query 只用于检索。

该方案适合解决短追问丢主题问题，但不是完整长期记忆系统。跨很多轮、主题频繁切换时仍应以显式问题为准。

### 2.4 图片查询

图片查询已达到最小可用闭环：

```text
前端选择图片
  -> 以 base64 随 /chat/stream 发送
  -> 后端保存到 data/query_images/{session_id}/
  -> CLIP 对查询图生成 image embedding
  -> 检索 Milvus image_dense
  -> OCR fallback 提取截图文字
  -> 将 image_matches / OCR 文本写入 trace 和回答 prompt
```

当前能力边界：

- 能处理 PDF 中页面截图、图表截图、含文字截图的相似检索。
- 对人物头像只做“相似截图/附近页面内容”检索，不做真实人脸识别。
- OCR 依赖 Tesseract 可执行程序；只安装 Python `pytesseract` 不够。
- 图片结果可能是页级或附近上下文级，不能保证每张任意截图都精确命中。

## 3. 前端与权限

前端为 Vue 3 CDN 单页应用，当前界面为 VectorBridge 工程控制台风格。

已接入功能：

- 登录、注册、JWT 鉴权。
- 管理员上传和删除文档。
- 普通用户聊天和管理自己的历史会话。
- SSE 流式输出、停止生成。
- RAG 步骤、检索详情、来源片段、图片候选、OCR 显示。
- 历史会话删除。
- 图片查询上传和本地历史预览。
- 生成中禁止切换会话，避免流式内容写错消息数组。
- 文档列表搜索、分页、上传/删除进度展示。

注意：Vue、marked、highlight.js、Font Awesome 当前仍来自 CDN。页面已增加依赖加载失败提示，但如需完全离线演示，应把这些静态资源本地化。

## 4. 技术栈

| 层 | 技术 |
| --- | --- |
| 后端 | FastAPI, Uvicorn, LangChain, LangGraph |
| 前端 | Vue 3 CDN, marked, highlight.js |
| 数据库 | PostgreSQL |
| 缓存 | Redis |
| 向量库 | Milvus |
| 文本 embedding | BAAI/bge-m3, sentence-transformers |
| 图片 embedding | CLIP `openai/clip-vit-base-patch32` |
| OCR | pytesseract + Tesseract OCR 可执行程序 |
| 文档解析 | PyPDF / PyMuPDF / docx2txt / Unstructured / HTML parser |

## 5. 运行方式

### 5.1 启动依赖服务

```bat
cd /d path\to\vectorbridge_repo
docker compose up -d
```

依赖端口：

- PostgreSQL: `5432`
- Redis: `6379`
- Milvus: `19530`
- Attu: `8080`

### 5.2 安装依赖

当前 `uv.lock` 以 CPU 可运行环境为基准。GPU 版 PyTorch 可按本地硬件环境单独安装。

```bat
uv sync
```

### 5.3 启动后端

Windows CMD 推荐：

```bat
cd /d path\to\vectorbridge_repo
set "HF_ENDPOINT=https://hf-mirror.com" && .venv\Scripts\python.exe -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

访问：

- 前端：http://127.0.0.1:8000/
- OpenAPI：http://127.0.0.1:8000/docs

### 5.4 GPU Acceleration

`.env.example` 中 `EMBEDDING_DEVICE`、`MULTIMODAL_EMBEDDING_DEVICE`、`UBG_DEVICE` 默认为 `auto`。安装 CUDA 版 PyTorch 后会自动使用 GPU，否则回退 CPU。

公开配置默认保留 `auto`，保证无 GPU 环境也能启动。需要固定 GPU 时，可以在本地 `.env` 中显式设置 `cuda` 或 `cuda:0`。

作者本机评估环境如下，仅作为性能复现实验参考，不是运行要求：

```text
torch 2.11.0+cu128
CUDA 12.8
NVIDIA GeForce RTX 4060 Laptop GPU
```

安装示例：

```bat
.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

验证：

```bat
.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

## 6. OCR 配置

Python 依赖里已有 `pytesseract`，但 Windows 还需要单独安装 Tesseract OCR 程序。

`.env` 示例：

```env
OCR_ENABLED=true
OCR_LANG=chi_sim+eng
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
OCR_MAX_CHARS=800
```

若未安装 Tesseract，图片查询仍可走 CLIP 检索，但 OCR 文本增强不可用。

## 7. 评估数据与常用验证

当前自动评估数据：

| 数据集 | 样本数 | 当前 gold 状态 |
| --- | ---: | --- |
| CRUD_RAG | 2,394 | 已按 source document 重新入库并生成独立 chunk 映射，平均 2.47 个相关 chunk/题 |
| RAGEval | 6,709 | 已重建 gold，并把 271 条无解样本保留为 negative；当前 refined 与 raw relevant chunks 一致 |

这些数据用于文本检索和 LLM Judge 评估，不覆盖网页手动上传的 PDF，也不覆盖图片截图查询。

常用验证命令：

```bat
.venv\Scripts\python.exe test_bug_fixes.py
.venv\Scripts\python.exe test_chat_routing.py
.venv\Scripts\python.exe test_chunk_ingestion.py
.venv\Scripts\python.exe test_catalog_retrieval_utils.py
.venv\Scripts\python.exe test_env_loading.py
.venv\Scripts\python.exe -X utf8 -c "from backend.env import load_env; load_env(); import backend.app; print('backend.app import ok')"
node --check frontend\script.js
```

## 8. 已知边界

- 图片检索是最小可用方案，不保证任意截图都精确定位。
- 当前没有生产级文件安全扫描。
- 没有生产级文档级权限隔离。
- PII / Prompt Injection 相关代码只作为未来治理基础，不作为已完成安全能力。
- CDN 资源未本地化，离线演示可能需要提前处理。
- 本地评估包含自动检索评估、LLM Judge 抽样和人工测试记录，但不代表生产环境指标。

## 9. Documentation Notes

- 能用测试或代码路径确认的能力才写为“已完成”。
- 可选模型、可选 OCR、可选 rerank 均写明依赖和降级行为。
- 评估结果和系统边界见 [evaluation-and-limitations.md](evaluation-and-limitations.md)。
