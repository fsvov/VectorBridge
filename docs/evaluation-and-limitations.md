# Evaluation and Limitations

> Updated on 2026-06-10. This document summarizes the current evaluation setup, results, and known limitations of VectorBridge.

## 1. Evaluation Scope

VectorBridge has been evaluated on text retrieval and answer quality tasks. Image retrieval is validated through manual screenshot tests and is not included in the automatic benchmark numbers below.

Current benchmark coverage:

| Dataset | Size | Usage |
| --- | ---: | --- |
| CRUD_RAG | 2,394 QA samples | Chinese news retrieval QA |
| RAGEval | 6,709 QA samples | Finance / law / medical retrieval QA |

Gold labels were rebuilt and deduplicated. RAGEval includes 271 unanswerable samples as negative cases. The current `gold_chunks_rageval_refined.json` has the same relevant chunk sets as the raw RAGEval gold file, so it should be treated as a cleaned gold file rather than a narrower semantic refinement.

## 2. Configuration

| Field | Value |
| --- | --- |
| Retrieval modes | dense / sparse / hybrid / ubg |
| UBG mode | heuristic fallback, no trained weight dependency |
| LLM Judge | `MODEL`, 100 samples per group |
| Image retrieval | not included in automatic text benchmarks |
| Evaluation date | 2026-06-10 |
| Author hardware | RTX 4060 Laptop GPU, CUDA 12.8, torch 2.11.0+cu128 |

The hardware row describes the author's local evaluation environment only. The project can run on CPU; `.env.example` defaults embedding devices to `auto`.

## 3. Retrieval Results

### CRUD_RAG

| Mode | recall@5 | recall@10 | mrr@5 | hit@5 | hit@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | **0.5726** | **0.7946** | **0.6010** | **0.9173** | 0.9716 |
| hybrid | 0.5516 | 0.7786 | 0.5146 | 0.9048 | **0.9766** |
| sparse | 0.5042 | 0.7297 | 0.4798 | 0.8630 | 0.9520 |
| ubg | 0.5011 | 0.7291 | 0.4720 | 0.8601 | 0.9528 |

On CRUD_RAG, dense retrieval performs best overall. The dataset has strong keyword and entity signals, and the heuristic UBG fallback is not the strongest choice for this setting.

### RAGEval

| Mode | recall@5 | recall@10 | mrr@5 | hit@5 | hit@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ubg | **0.3110** | **0.4266** | 0.4960 | 0.7044 | **0.7773** |
| dense | 0.2930 | 0.4113 | 0.6014 | **0.7207** | 0.7749 |
| hybrid | 0.2805 | 0.3828 | **0.6155** | 0.7038 | 0.7606 |
| sparse | 0.2347 | 0.3084 | 0.5395 | 0.6305 | 0.6979 |

On RAGEval, UBG achieves the best recall@5 and recall@10, while hybrid has the best mrr@5. This suggests that path fusion improves coverage on harder synthetic documents, but ranking quality still depends on the final scoring strategy.

## 4. Answer Quality

LLM Judge results on 100 samples per group:

| Dataset | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
| --- | ---: | ---: | ---: | ---: |
| CRUD_RAG | **0.80** | 0.94 | 0.49 | **0.83** |
| RAGEval Finance | 0.39 | 0.91 | 0.32 | 0.38 |
| RAGEval Law | 0.26 | 0.94 | 0.33 | 0.30 |
| RAGEval Medical | 0.39 | 0.92 | 0.33 | 0.42 |

Interpretation:

- CRUD_RAG has usable answer grounding, with strong answer relevancy and context recall.
- RAGEval remains harder: context recall is in the 0.30-0.42 range and context precision is low.
- LLM Judge scores should be treated as approximate because the judge model can be biased toward model-generated answers.

## 5. Manual Validation

Manual testing covered:

- document upload, overwrite, deletion, and progress reporting;
- single-turn text RAG;
- short follow-up questions with topic completion;
- screenshot retrieval with CLIP image embeddings and OCR fallback;
- session deletion and trace display in the frontend.

Observed behavior:

- Text QA is usable for demonstration on indexed documents.
- Follow-up questions are more stable after query completion, but this is not a full long-term memory system.
- Image retrieval is useful for screenshots of pages, charts, and text-heavy regions, but it is not general visual question answering.
- Face identity recognition is not supported. Person screenshots are handled only as similar-image / nearby-document retrieval.

## 6. Known Limitations

| Area | Limitation |
| --- | --- |
| Image understanding | Screenshot retrieval is minimal viable functionality, not full VQA. |
| Rerank | Reranker support is optional and not a stable default dependency. |
| Security | No production-grade file scanning, PII governance, or document-level ACL. |
| Multi-user production use | Basic user/session isolation exists, but the system has not been stress-tested. |
| Conformal calibration | Runtime can use a calibration state file, but formal coverage reports are not shipped as a default artifact. |
| Evaluation | Automatic benchmarks cover text retrieval; image retrieval is currently evaluated manually. |

## 7. Reproducibility Notes

- Evaluation data and generated gold files are not committed by default.
- `data/`, uploaded documents, model caches, and local reports are ignored by `.gitignore`.
- To reproduce the automatic metrics, ingest the corresponding datasets and regenerate the gold files using scripts under `scripts/`.
- Metrics may change if embedding models, LLM providers, retrieval thresholds, or gold generation logic are changed.
