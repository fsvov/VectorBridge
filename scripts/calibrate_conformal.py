"""Conformal 检索置信度校准脚本。

从 gold 标注中抽取校准集，运行检索并计算 nonconformity 分位数。
输出 conformal_state.json 供推理时使用。

用法:
    python scripts/calibrate_conformal.py --cal-size 500 --alpha 0.10
    python scripts/calibrate_conformal.py --cal-size 500 --domain-aware
"""

import json
import os
import sys
import argparse
import time
import random
from pathlib import Path

import numpy as np

_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from backend.env import load_env
load_env()

from backend.indexing.embedding import embedding_service
from backend.indexing.milvus_client import get_milvus_store
from backend.eval.gold_utils import gold_file_candidates, load_gold_items
from backend.rag.conformal_retrieval import (
    ConformalRetrievalCalibrator,
    MondrianRetrievalCalibrator,
    save_calibrator,
)

MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "embeddings_collection")
LEAF_LEVEL = int(os.getenv("LEAF_RETRIEVE_LEVEL", "3"))


def load_gold_data(max_items: int | None = None) -> list[dict]:
    data = []
    for fp in gold_file_candidates(_PROJ / "data", include_raw_fallback=True):
        if fp.name == "gold_chunks_rageval.json" and (_PROJ / "data" / "gold_chunks_rageval_refined.json").exists():
            continue
        if fp.exists():
            data.extend(load_gold_items(fp))
    random.shuffle(data)
    if max_items:
        data = data[:max_items]
    return data


def run_calibration_samples(gold_items: list[dict], milvus, top_k: int = 5) -> list[dict]:
    """对每条 gold 执行检索，收集 {max_score, hit, domain}。"""
    filter_expr = f"chunk_level == {LEAF_LEVEL}"
    results = []
    for i, item in enumerate(gold_items):
        query = item.get("question", "")
        gold_ids = set(item.get("relevant_chunk_ids", []))
        domain = item.get("domain", "default")
        if not query or not gold_ids:
            continue

        try:
            q_emb = embedding_service.get_embeddings([query])[0]
            s_emb = embedding_service.get_sparse_embedding(query)
            retrieved = milvus.hybrid_retrieve(q_emb, s_emb, top_k=top_k, filter_expr=filter_expr)
        except Exception:
            continue

        hits = [d for d in retrieved if d.get("chunk_id") in gold_ids]
        max_score = max((d.get("score", 0.0) for d in retrieved), default=0.0)
        results.append({
            "max_score": max_score,
            "hit": len(hits) > 0,
            "domain": domain,
        })

        if (i + 1) % 100 == 0:
            print(f"  校准进度: {i + 1}/{len(gold_items)}", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal-size", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--domain-aware", action="store_true")
    args = parser.parse_args()

    print(f"加载 gold 数据 (最多 {args.cal_size} 条)...", flush=True)
    gold = load_gold_data(max_items=args.cal_size * 2)
    cal_items = gold[:args.cal_size]
    test_items = gold[args.cal_size:args.cal_size + 200]
    print(f"  校准集: {len(cal_items)}, 测试集: {len(test_items)}", flush=True)

    milvus = get_milvus_store()

    print("运行校准检索...", flush=True)
    cal_scores = run_calibration_samples(cal_items, milvus, top_k=args.top_k)

    if args.domain_aware:
        calibrator = MondrianRetrievalCalibrator(args.alpha)
        calibrator.calibrate(cal_scores)
    else:
        calibrator = ConformalRetrievalCalibrator(args.alpha)
        calibrator.calibrate(cal_scores)
        save_calibrator(calibrator)

    # 在测试集上验证
    if test_items:
        print("\n测试集验证...", flush=True)
        test_scores = run_calibration_samples(test_items, milvus, top_k=args.top_k)
        covered = 0
        total = 0
        for item in test_scores:
            if isinstance(calibrator, MondrianRetrievalCalibrator):
                result = calibrator.predict_confidence(item["max_score"], item["domain"])
            else:
                result = calibrator.predict_confidence(item["max_score"])
            if result["covered"]:
                covered += 1
            total += 1
        print(f"  测试覆盖率: {covered}/{total} = {covered/max(total,1):.2%} "
              f"(目标: {1-args.alpha:.2%})", flush=True)

    print(f"\n完成! threshold={calibrator.threshold if hasattr(calibrator, 'threshold') else 'mondrian'}")


if __name__ == "__main__":
    main()
