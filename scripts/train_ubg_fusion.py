"""UBG 检索融合权重训练脚本。

使用 gold chunk 标注数据训练 UBGRetrievalFusion，让模型学习
"何种 query 更应信任 dense/sparse/image 路"。

用法:
    HF_ENDPOINT=https://hf-mirror.com python scripts/train_ubg_fusion.py --samples 500 --epochs 20
"""

import json
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from backend.env import load_env
load_env()

from backend.indexing.device import resolve_torch_device
from backend.indexing.embedding import embedding_service
from backend.indexing.milvus_client import get_milvus_store
from backend.rag.ubg_fusion import UBGRetrievalFusion, compute_score_stats
from backend.eval.gold_utils import gold_file_candidates, load_gold_items

MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "embeddings_collection")
MODEL_OUTPUT = _PROJ / "data" / "ubg_fusion_weights.pt"


def load_training_data(samples: int = 500) -> list[dict]:
    """加载 gold 标注数据。"""
    data = []
    for fp in gold_file_candidates(_PROJ / "data", include_raw_fallback=True):
        if fp.name == "gold_chunks_rageval.json" and (_PROJ / "data" / "gold_chunks_rageval_refined.json").exists():
            continue
        if fp.exists():
            data.extend(load_gold_items(fp))
    np.random.shuffle(data)
    return data[:samples]


def train(samples: int = 500, epochs: int = 20, lr: float = 1e-3):
    print(f"加载 {samples} 条训练数据...", flush=True)
    gold = load_training_data(samples)
    milvus = get_milvus_store()
    device = resolve_torch_device("UBG_TRAIN_DEVICE", default=os.getenv("UBG_DEVICE", "auto"))
    print(f"UBG 训练设备: {device}", flush=True)

    model = UBGRetrievalFusion(query_dim=1024, hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    filter_expr = f"chunk_level == 3"

    for epoch in range(epochs):
        total_loss = 0.0
        count = 0
        for item in gold:
            query = item.get("question", "")
            gold_ids = set(item.get("relevant_chunk_ids", []))
            if not query or not gold_ids:
                continue

            # 获取 query embedding
            try:
                q_emb = embedding_service.get_embeddings([query])[0]
                s_emb = embedding_service.get_sparse_embedding(query)
            except Exception:
                continue

            # 三路检索
            dense_res = milvus.dense_retrieve(q_emb, top_k=10, filter_expr=filter_expr)
            sparse_res = milvus.sparse_retrieve(s_emb, top_k=10, filter_expr=filter_expr)
            image_res = []

            # 计算每路命中率（作为监督信号）
            d_hit = sum(1 for d in dense_res if d.get("chunk_id") in gold_ids) / max(len(dense_res), 1)
            s_hit = sum(1 for d in sparse_res if d.get("chunk_id") in gold_ids) / max(len(sparse_res), 1)
            target_weights = torch.tensor([[d_hit, s_hit, 0.0]], device=device)
            target_weights = target_weights / (target_weights.sum(dim=-1, keepdim=True) + 1e-8)

            # 输入统计量
            d_s = compute_score_stats([d.get("score", 0.0) for d in dense_res])
            s_s = compute_score_stats([d.get("score", 0.0) for d in sparse_res])
            i_s = compute_score_stats([])

            stats = torch.tensor([[
                d_s["mean"], d_s["std"], d_s["min"], d_s["max"],
                s_s["mean"], s_s["std"], s_s["min"], s_s["max"],
                i_s["mean"], i_s["std"], i_s["min"], i_s["max"],
            ]], device=device)
            q_vec = torch.tensor(q_emb, device=device).unsqueeze(0)

            pred = model(q_vec, stats)
            loss = criterion(pred, target_weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        print(f"  epoch {epoch + 1}/{epochs}: loss={total_loss / max(count, 1):.4f}", flush=True)

    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(MODEL_OUTPUT))
    print(f"模型已保存: {MODEL_OUTPUT}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(samples=args.samples, epochs=args.epochs, lr=args.lr)
