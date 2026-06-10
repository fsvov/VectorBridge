# 评估结果与边界

> 更新于 2026-06-10。本文档汇总 VectorBridge 的评估配置、结果及已知限制。

## 1. 评估范围

VectorBridge 已在文本检索和回答质量上完成自动评估。图片检索通过人工截图测试验证，不包含在以下自动评估数据中。

当前自动评估覆盖：

| 数据集 | 规模 | 用途 |
| --- | ---: | --- |
| CRUD_RAG | 2,394 条 | 中文新闻检索 QA |
| RAGEval | 6,709 条 | 金融 / 法律 / 医疗检索 QA |

Gold 标注已重建并去重。RAGEval 包含 271 条无解样本作为负例。当前 `gold_chunks_rageval_refined.json` 与原始 RAGEval 的 relevant chunk 集合一致，应视为清洗后的 gold 而非语义收窄后的结果。

## 2. 评估配置

| 字段 | 值 |
| --- | --- |
| 检索模式 | dense / sparse / hybrid / ubg |
| UBG 模式 | 启发式 fallback，无训练权重依赖 |
| LLM Judge | `MODEL`，每组 100 条 |
| 图片检索 | 不参与自动文本评估 |
| 评估日期 | 2026-06-10 |
| 评估硬件 | RTX 4060 Laptop GPU, CUDA 12.8, torch 2.11.0+cu128 |

硬件行仅描述评估环境。项目可在 CPU 上运行，`.env.example` 中 embedding 设备默认为 `auto`。

## 3. 检索结果

### CRUD_RAG

| 模式 | recall@5 | recall@10 | mrr@5 | hit@5 | hit@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | **0.5726** | **0.7946** | **0.6010** | **0.9173** | 0.9716 |
| hybrid | 0.5516 | 0.7786 | 0.5146 | 0.9048 | **0.9766** |
| sparse | 0.5042 | 0.7297 | 0.4798 | 0.8630 | 0.9520 |
| ubg | 0.5011 | 0.7291 | 0.4720 | 0.8601 | 0.9528 |

CRUD_RAG 上 dense 全维度最优。该数据集关键词和实体信号较强，启发式 UBG fallback 在此场景不占优。

### RAGEval

| 模式 | recall@5 | recall@10 | mrr@5 | hit@5 | hit@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ubg | **0.3110** | **0.4266** | 0.4960 | 0.7044 | **0.7773** |
| dense | 0.2930 | 0.4113 | 0.6014 | **0.7207** | 0.7749 |
| hybrid | 0.2805 | 0.3828 | **0.6155** | 0.7038 | 0.7606 |
| sparse | 0.2347 | 0.3084 | 0.5395 | 0.6305 | 0.6979 |

RAGEval 上 UBG 取得最优 recall@5 和 recall@10，hybrid 取得最优 mrr@5。说明路径融合在较难的合成文档上改善了覆盖度，但排序精度仍取决于最终打分策略。

## 4. 回答质量

LLM Judge 每组 100 条结果：

| 数据集 | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
| --- | ---: | ---: | ---: | ---: |
| CRUD_RAG | **0.80** | 0.94 | 0.49 | **0.83** |
| RAGEval Finance | 0.39 | 0.91 | 0.32 | 0.38 |
| RAGEval Law | 0.26 | 0.94 | 0.33 | 0.30 |
| RAGEval Medical | 0.39 | 0.92 | 0.33 | 0.42 |

解读：

- CRUD_RAG 回答可被检索文档支撑，answer relevancy 和 context recall 较高。
- RAGEval 仍较难：context recall 在 0.30-0.42 区间，context precision 普遍偏低。
- LLM Judge 分数应视为近似值，Judge 模型评估自身生成结果存在评分偏宽松风险。

## 5. 人工验证

人工测试覆盖：

- 文档上传、覆盖、删除及进度展示；
- 单轮文本 RAG；
- 短追问的主题补全；
- CLIP 图片 embedding + OCR fallback 截图检索；
- 前端会话删除和 trace 展示。

观察结论：

- 文本 QA 在已入库文档上可稳定演示。
- 追问补全后效果改善，但不是完整长期记忆系统。
- 图片检索适合页面截图、图表和含文字区域的相似检索，不是通用视觉问答；目标是判断截图可能来自哪个已入库页面、图表或文本区域。
- 不支持人脸识别。人物截图仅做相似图片/附近文档检索，能否命中取决于原 PDF 页面图像、邻近文本和 OCR/CLIP 特征是否被成功入库。

## 6. 已知限制

| 领域 | 限制 |
| --- | --- |
| 图片理解 | 截图检索是最小可用方案，非完整 VQA；不做身份识别或开放域图像推理。 |
| Rerank | 精排支持为可选，不是稳定默认依赖。 |
| 安全 | 无生产级文件扫描、PII 治理、文档级权限隔离。 |
| 多用户并发 | 具备基础用户/会话隔离，未做压力测试。 |
| Conformal 校准 | 运行时可使用校准状态文件，未作为默认产物发布正式覆盖度报告；校准对象是检索盲区阈值，不保证最终答案一定正确。 |
| 评估 | 自动评估覆盖文本检索；图片检索当前依赖人工测试。 |

## 7. 可复现性说明

- 复现自动评估需重新入库相应数据集，并使用 `scripts/` 下的脚本重新生成 gold 文件。
- 更换 embedding 模型、LLM 提供商、检索阈值或 gold 生成逻辑会改变评估指标。

参考复现流程：

```bash
# 1. 启动依赖服务并安装依赖
docker compose up -d
uv sync

# 2. 准备 .env 后导入评估数据
uv run python scripts/ingest_crud_rag.py
uv run python scripts/ingest_rageval.py

# 3. 可选：重新生成 / 收窄 RAGEval gold
uv run python scripts/refine_rageval_gold.py

# 4. 可选：生成 Conformal 校准状态
uv run python scripts/calibrate_conformal.py --cal-size 500 --alpha 0.10
```

评估脚本依赖本地数据文件和模型配置。若数据集路径、模型、分块策略或 gold 生成逻辑发生变化，表中指标需要重新生成。
