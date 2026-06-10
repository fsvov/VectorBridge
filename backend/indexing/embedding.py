"""文本向量化服务 - 支持密集向量和稀疏向量（BM25），词表与 df 持久化 + 增量更新 + 嵌入缓存 + 词表清理"""
import json
import math
import os
import re
import threading
import time
from collections import Counter, OrderedDict
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings

from backend.indexing.device import resolve_torch_device

_DEFAULT_STATE_PATH = Path(__file__).resolve().parent.parent / "data" / "bm25_state.json"

# 嵌入缓存最大条目数
_MAX_EMBED_CACHE_SIZE = 5000


def _create_dense_embedder() -> HuggingFaceEmbeddings:
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    device = resolve_torch_device("EMBEDDING_DEVICE", default="auto")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


class EmbeddingService:
    """文本向量化服务 - 密集向量本地模型 + BM25 稀疏向量（持久化统计）+ 嵌入缓存"""

    def __init__(self, state_path: Path | str | None = None):
        self._embedder = None
        self._embedder_lock = threading.Lock()
        self._state_path = Path(state_path or os.getenv("BM25_STATE_PATH", _DEFAULT_STATE_PATH))
        self._lock = threading.Lock()

        # BM25 参数
        self.k1 = 1.5
        self.b = 0.75

        self._vocab: dict[str, int] = {}
        self._vocab_counter = 0
        self._doc_freq: Counter[str] = Counter()
        self._total_docs = 0
        self._sum_token_len = 0
        self._avg_doc_len = 1.0

        # 词表访问时间戳（epoch 秒），用于清理长期未使用的词条
        self._vocab_access_ts: dict[str, float] = {}

        # 嵌入缓存: text → embedding list
        self._embed_cache: OrderedDict[str, list[float]] = OrderedDict()

        self._load_state()

    def _recompute_avg_len(self) -> None:
        self._avg_doc_len = (
            self._sum_token_len / self._total_docs if self._total_docs > 0 else 1.0
        )

    def _load_state(self) -> None:
        path = self._state_path
        if not path.is_file():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if raw.get("version") != 1:
            return
        self._vocab = {str(k): int(v) for k, v in raw.get("vocab", {}).items()}
        self._doc_freq = Counter({str(k): int(v) for k, v in raw.get("doc_freq", {}).items()})
        self._total_docs = int(raw.get("total_docs", 0))
        self._sum_token_len = int(raw.get("sum_token_len", 0))
        # 恢复访问时间戳（旧格式无此字段）
        ts_map = raw.get("vocab_access_ts", {})
        now = time.time()
        self._vocab_access_ts = {str(k): float(v) for k, v in ts_map.items()}
        # 对无时间戳的词条赋予当前时间
        for token in self._vocab:
            if token not in self._vocab_access_ts:
                self._vocab_access_ts[token] = now
        if self._vocab:
            self._vocab_counter = max(self._vocab.values()) + 1
        else:
            self._vocab_counter = 0
        self._recompute_avg_len()

    def _get_dense_embedder(self):
        if self._embedder is None:
            with self._embedder_lock:
                if self._embedder is None:
                    self._embedder = _create_dense_embedder()
        return self._embedder

    def _persist_unlocked(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "total_docs": self._total_docs,
            "sum_token_len": self._sum_token_len,
            "vocab": self._vocab,
            "doc_freq": dict(self._doc_freq),
            "vocab_access_ts": self._vocab_access_ts,
        }
        tmp = self._state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._state_path)

    def _persist(self) -> None:
        with self._lock:
            self._persist_unlocked()

    def _touch_vocab(self, token: str) -> None:
        self._vocab_access_ts[token] = time.time()

    def cleanup_stale_vocab(self, days: int = 30) -> int:
        """清理超过 days 天未访问的词条，释放稀疏向量维度。返回清理数量。"""
        cutoff = time.time() - days * 86400
        stale: list[str] = []
        with self._lock:
            for token, ts in list(self._vocab_access_ts.items()):
                if ts < cutoff and token in self._vocab:
                    stale.append(token)
            if not stale:
                return 0
            for token in stale:
                idx = self._vocab.pop(token, None)
                self._doc_freq.pop(token, None)
                self._vocab_access_ts.pop(token, None)
            # 重建词表索引（压缩空洞）
            self._rebuild_vocab_indices()
            self._persist_unlocked()
        return len(stale)

    def _rebuild_vocab_indices(self) -> None:
        """压缩词表索引，消除已删除词条留下的空洞。"""
        # 注意：这会改变现有词条的索引，需要通知调用方重建 Milvus 稀疏向量
        # 因此此方法仅在 cleanup_stale_vocab 内部使用，且外部需理解其影响
        new_vocab: dict[str, int] = {}
        for i, token in enumerate(self._vocab):
            new_vocab[token] = i
        self._vocab = new_vocab
        self._vocab_counter = len(new_vocab)

    def increment_add_documents(self, texts: list[str]) -> None:
        if not texts:
            return
        with self._lock:
            for text in texts:
                tokens = self.tokenize(text)
                doc_len = len(tokens)
                self._sum_token_len += doc_len
                self._total_docs += 1
                seen: set[str] = set()
                for token in tokens:
                    if token in seen:
                        continue
                    seen.add(token)
                    if token not in self._vocab:
                        self._vocab[token] = self._vocab_counter
                        self._vocab_counter += 1
                    self._doc_freq[token] += 1
                    self._touch_vocab(token)
            self._recompute_avg_len()
            self._persist_unlocked()

    def increment_remove_documents(self, texts: list[str]) -> None:
        if not texts:
            return
        with self._lock:
            for text in texts:
                tokens = self.tokenize(text)
                doc_len = len(tokens)
                self._sum_token_len = max(0, self._sum_token_len - doc_len)
                self._total_docs = max(0, self._total_docs - 1)
                seen: set[str] = set()
                for token in tokens:
                    if token in seen:
                        continue
                    seen.add(token)
                    if token not in self._doc_freq:
                        continue
                    self._doc_freq[token] -= 1
                    if self._doc_freq[token] <= 0:
                        del self._doc_freq[token]
            self._recompute_avg_len()
            self._persist_unlocked()

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embedder = self._get_dense_embedder()

        # 检查缓存
        results: list[list[float]] = []
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []
        for i, text in enumerate(texts):
            cached = self._embed_cache.get(text)
            if cached is not None:
                results.append(cached)
            else:
                results.append([])  # 占位
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            try:
                new_embeddings = embedder.embed_documents(uncached_texts)
            except Exception as e:
                raise Exception(f"本地嵌入模型调用失败: {str(e)}") from e
            for j, idx in enumerate(uncached_indices):
                emb = new_embeddings[j]
                results[idx] = emb
                self._cache_embedding(uncached_texts[j], emb)

        return results

    def _cache_embedding(self, text: str, embedding: list[float]) -> None:
        if len(self._embed_cache) >= _MAX_EMBED_CACHE_SIZE:
            # 淘汰最旧的条目
            self._embed_cache.popitem(last=False)
        self._embed_cache[text] = embedding
        # 同步移动到最后（标记为最近使用）
        self._embed_cache.move_to_end(text)

    def clear_embedding_cache(self) -> int:
        count = len(self._embed_cache)
        self._embed_cache.clear()
        return count

    def tokenize(self, text: str) -> list[str]:
        text = text.lower()
        tokens = []
        chinese_pattern = re.compile(r"[一-鿿]")
        english_pattern = re.compile(r"[a-zA-Z]+")
        i = 0
        while i < len(text):
            char = text[i]
            if chinese_pattern.match(char):
                tokens.append(char)
                i += 1
            elif english_pattern.match(char):
                match = english_pattern.match(text[i:])
                if match:
                    tokens.append(match.group())
                    i += len(match.group())
            else:
                i += 1
        return tokens

    def _sparse_vector_for_text_unlocked(self, text: str) -> tuple[dict, bool]:
        tokens = self.tokenize(text)
        doc_len = len(tokens)
        tf = Counter(tokens)
        sparse_vector: dict[int, float] = {}
        vocab_changed = False
        n = max(self._total_docs, 0)
        avg = max(self._avg_doc_len, 1.0)

        for token, freq in tf.items():
            if token not in self._vocab:
                self._vocab[token] = self._vocab_counter
                self._vocab_counter += 1
                vocab_changed = True

            idx = self._vocab[token]
            self._touch_vocab(token)
            df = self._doc_freq.get(token, 0)
            if df == 0:
                idf = math.log((n + 1) / 1)
            else:
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1)

            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / avg)
            score = idf * numerator / denominator
            if score > 0:
                sparse_vector[idx] = float(score)

        return sparse_vector, vocab_changed

    def get_sparse_embedding(self, text: str) -> dict:
        with self._lock:
            sparse_vector, vocab_changed = self._sparse_vector_for_text_unlocked(text)
            if vocab_changed:
                self._persist_unlocked()
        return sparse_vector

    def get_sparse_embeddings(self, texts: list[str]) -> list[dict]:
        if not texts:
            return []
        with self._lock:
            out: list[dict] = []
            any_new_vocab = False
            for text in texts:
                sparse_vector, vocab_changed = self._sparse_vector_for_text_unlocked(text)
                out.append(sparse_vector)
                any_new_vocab = any_new_vocab or vocab_changed
            if any_new_vocab:
                self._persist_unlocked()
        return out

    def get_all_embeddings(self, texts: list[str]) -> tuple[list[list[float]], list[dict]]:
        dense_embeddings = self.get_embeddings(texts)
        sparse_embeddings = self.get_sparse_embeddings(texts)
        return dense_embeddings, sparse_embeddings


# 全进程唯一实例：写入与检索共用同一份 BM25 持久化状态
embedding_service = EmbeddingService()
