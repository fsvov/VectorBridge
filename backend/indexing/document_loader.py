"""文档加载和分片服务 — 语义分块 + LLM 结构标注 + Chunk Headers + 表格/代码块保护 + 可配置粒度 + 多模态图片提取"""
import hashlib
import json
import os
import re
import math
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ===== 可配置参数 =====
_CHUNK_SIZE_L3 = int(os.getenv("CHUNK_SIZE_L3", "500"))
_CHUNK_OVERLAP_L3 = int(os.getenv("CHUNK_OVERLAP_L3", "80"))
_SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", "0.65"))
_SEMANTIC_MIN_LEN = int(os.getenv("SEMANTIC_MIN_LEN", "1200"))
_LLM_STRUCTURE_ENABLED = os.getenv("LLM_STRUCTURE_ENABLED", "true").lower() != "false"
_CHUNK_HEADERS_ENABLED = os.getenv("CHUNK_HEADERS_ENABLED", "true").lower() != "false"

# ===== 表格/代码块保护 =====
_PROTECT_PLACEHOLDER_NL = "__PROTECTED_NL__"
_PROTECT_PLACEHOLDER_COMMA = "__PROTECTED_CJK_COMMA__"
_PROTECT_PLACEHOLDER_PERIOD = "__PROTECTED_CJK_PERIOD__"

_PROTECT_MAP = {
    "\n": _PROTECT_PLACEHOLDER_NL,
    "，": _PROTECT_PLACEHOLDER_COMMA,
    "。": _PROTECT_PLACEHOLDER_PERIOD,
}
_PROTECT_REVERSE = {v: k for k, v in _PROTECT_MAP.items()}

_SPLIT_SEPARATORS = ["\n\n", "。", "！", "？", "\n", "，", "、", " ", ""]

# 每个语义段落最大字符数（超过则强制再切）
_SEMANTIC_SEGMENT_MAX = 2500


def _run_with_timeout(func, timeout_seconds: float, timeout_message: str):
    """Run a blocking callable without waiting for executor shutdown after timeout."""
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func)
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeout as exc:
        future.cancel()
        raise Exception(timeout_message) from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _protect_regions(text: str) -> str:
    lines = text.split("\n")
    in_table = False
    out_lines = []
    for line in lines:
        stripped = line.strip()
        is_table_line = stripped.startswith("|") and ("|" in stripped[1:])
        if is_table_line and not in_table:
            in_table = True
        elif not is_table_line and in_table:
            if not stripped or not stripped.startswith("|"):
                in_table = False
        if in_table:
            for orig, placeholder in _PROTECT_MAP.items():
                line = line.replace(orig, placeholder)
        out_lines.append(line)
    text = "\n".join(out_lines)

    code_pattern = re.compile(r"```[\s\S]*?```", re.MULTILINE)
    def _protect_code(match):
        block = match.group(0)
        for orig, placeholder in _PROTECT_MAP.items():
            block = block.replace(orig, placeholder)
        return block
    text = code_pattern.sub(_protect_code, text)

    indent_code_pattern = re.compile(r"((?:^    .*\n)+)", re.MULTILINE)
    def _protect_indent(match):
        block = match.group(0)
        for orig, placeholder in _PROTECT_MAP.items():
            block = block.replace(orig, placeholder)
        return block
    text = indent_code_pattern.sub(_protect_indent, text)

    return text


def _restore_regions(text: str) -> str:
    for placeholder, orig in _PROTECT_REVERSE.items():
        text = text.replace(placeholder, orig)
    return text


# ===== 语义分块 =====

def _split_sentences(text: str) -> List[str]:
    """按中英文句子边界切分。"""
    # 先按句号/问号/感叹号/换行切
    raw = re.split(r"(?<=[。！？\.!\?])\s*|(?<=\n\n)", text)
    sentences = []
    for s in raw:
        s = s.strip()
        if s:
            sentences.append(s)
    # 合并过短的句子到前一句
    merged = []
    for s in sentences:
        if merged and len(s) < 20 and len(merged[-1]) < 300:
            merged[-1] = merged[-1] + s
        else:
            merged.append(s)
    return merged


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _split_by_semantic_boundaries(text: str) -> List[str]:
    """
    语义分块：用 bge-m3 计算相邻句子的相似度，在相似度骤降处切分。
    短文档（<_SEMANTIC_MIN_LEN 字符）跳过，直接返回原文。
    """
    if len(text) < _SEMANTIC_MIN_LEN:
        return [text]

    sentences = _split_sentences(text)
    if len(sentences) < 4:
        return [text]

    try:
        from backend.indexing.embedding import embedding_service
        embeddings = embedding_service.get_embeddings(sentences)
    except Exception:
        return [text]

    # 计算相邻相似度
    similarities = []
    for i in range(1, len(sentences)):
        sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
        similarities.append(sim)

    if not similarities:
        return [text]

    # 找语义断点：相似度低于阈值的位置
    mean_sim = sum(similarities) / len(similarities)
    adaptive_threshold = min(_SEMANTIC_THRESHOLD, mean_sim - 0.5 * max(0.05, math.sqrt(
        sum((s - mean_sim) ** 2 for s in similarities) / len(similarities)
    )))

    breakpoints = [0]
    for i, sim in enumerate(similarities):
        if sim < adaptive_threshold:
            breakpoints.append(i + 1)

    # 合并太小的段
    segments = []
    seg_start = 0
    for bp in breakpoints[1:]:
        seg_text = "".join(sentences[seg_start:bp])
        if len(seg_text) > 100:
            segments.append(seg_text)
            seg_start = bp
    # 最后一段
    last_seg = "".join(sentences[seg_start:])
    if last_seg.strip():
        if segments and len(last_seg) < 80:
            segments[-1] = segments[-1] + last_seg
        else:
            segments.append(last_seg)

    # 对超长段落二次切分
    final_segments = []
    for seg in segments:
        if len(seg) > _SEMANTIC_SEGMENT_MAX:
            # 对超长段按相似度再切一次
            sub = _split_by_semantic_boundaries(seg) if len(seg) > _SEMANTIC_MIN_LEN else [seg]
            final_segments.extend(sub)
        else:
            final_segments.append(seg)

    return final_segments if final_segments else [text]


# ===== LLM 结构标注 =====

_STRUCTURE_PROMPT = (
    "为以下文档标注结构标题。在文档中找到2-5个主题转换点，插入 `[## 标题]` 标记。\n"
    "规则：\n"
    "1. 不改变原文任何字词，只在主题转换处插入标题标记\n"
    "2. 标题要简短（3-8字），概括该段主题\n"
    "3. 全文开头不需要标题\n"
    "4. 如果文档主题单一无明显转换，则不插入任何标记\n\n"
    "文档：\n{text}\n\n"
    "输出（直接输出标注后的完整文档，不要解释）："
)


def _annotate_structure_llm(text: str) -> str:
    """用 flash 模型标注文档结构。仅在长文档且启用时调用。"""
    if not _LLM_STRUCTURE_ENABLED or len(text) < 1500:
        return text

    try:
        import os as _os
        from langchain.chat_models import init_chat_model

        api_key = _os.getenv("LLM_API_KEY")
        fast_model = _os.getenv("FAST_MODEL") or _os.getenv("MODEL")
        base_url = _os.getenv("BASE_URL")
        if not (api_key and fast_model and base_url):
            return text

        llm = init_chat_model(
            model=fast_model,
            model_provider="openai",
            api_key=api_key,
            base_url=base_url,
            temperature=0,
        )
        prompt = _STRUCTURE_PROMPT.format(text=text[:6000])
        result = (llm.invoke(prompt).content or "").strip()

        # 验证：标注后的文本应该包含原文的核心内容
        if len(result) < len(text) * 0.5:
            return text
        return result
    except Exception:
        return text


# ===== Contextual Chunk Headers =====

_HEADER_CACHE: dict[str, str] = {}
_HEADER_CACHE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / ".header_cache.json"


def _load_header_cache():
    if _HEADER_CACHE_PATH.exists():
        try:
            data = json.loads(_HEADER_CACHE_PATH.read_text(encoding="utf-8"))
            _HEADER_CACHE.update(data)
        except Exception: pass


def _save_header_cache():
    try:
        _HEADER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _HEADER_CACHE_PATH.write_text(json.dumps(_HEADER_CACHE, ensure_ascii=False), encoding="utf-8")
    except Exception: pass
_HEADER_PROMPT = (
    "为以下文本片段生成一个简短的上下文标题（不超过15字）。\n"
    "标题应包含：实体名称 + 主题内容。如'蓝天娱乐 | 2023年财务表现'\n"
    "直接输出标题，不要解释。\n\n"
    "文本片段：\n{text}"
)


if _CHUNK_HEADERS_ENABLED:
    _load_header_cache()


def _generate_chunk_header(text: str, filename: str) -> str:
    """为单个 L3 chunk 生成上下文标题。额度不足时自动切换备选模型。"""
    cache_key = f"{filename}|{text[:100]}"
    if cache_key in _HEADER_CACHE:
        return _HEADER_CACHE[cache_key]

    if not _CHUNK_HEADERS_ENABLED or len(text) < 50:
        return ""

    from langchain.chat_models import init_chat_model
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("BASE_URL")
    if not (api_key and base_url):
        return ""

    # 模型尝试顺序：FAST_MODEL → 备选列表 → MODEL
    primary = os.getenv("FAST_MODEL") or os.getenv("MODEL")
    backups_raw = os.getenv("MODEL_BACKUPS", "")
    backups = [m.strip() for m in backups_raw.split(",") if m.strip()]
    candidates = [primary] + backups + [os.getenv("MODEL")]
    candidates = list(dict.fromkeys(candidates))  # 去重保序

    for model_name in candidates:
        if not model_name:
            continue
        try:
            llm = init_chat_model(model=model_name, model_provider="openai",
                                  api_key=api_key, base_url=base_url, temperature=0)
            result = _run_with_timeout(
                lambda: (llm.invoke(_HEADER_PROMPT.format(text=text[:800])).content or "").strip(),
                20,
                "Chunk Header 生成超时（20s）",
            )
            result = result.replace('"', '').replace("'", "")
            if len(result) > 30:
                result = result[:30]
            header = f"[{result}] " if result else ""
            _HEADER_CACHE[cache_key] = header
            # 每 100 条持久化一次
            if len(_HEADER_CACHE) % 100 == 0:
                _save_header_cache()
            return header
        except Exception:
            continue

    return ""


# ===== PDF/DOCX 分区 =====

def _partition_pdf_with_pymupdf(file_path: str) -> List:
    """Fast PDF text extraction using PyMuPDF; returns LangChain-like page docs."""
    try:
        import fitz
    except ImportError:
        return []

    docs = []
    pdf = None
    try:
        pdf = fitz.open(file_path)
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = (page.get_text("text") or "").strip()
            if text:
                docs.append(SimpleNamespace(page_content=text, metadata={"page": page_num}))
    except Exception:
        return []
    finally:
        if pdf is not None:
            pdf.close()
    return docs


def _partition_pdf(file_path: str) -> List:
    docs = _partition_pdf_with_pymupdf(file_path)
    if docs:
        return docs

    from langchain_community.document_loaders import PyPDFLoader
    # 先用 PyPDFLoader（快速稳定），unstructured 太慢且容易挂
    try:
        return _run_with_timeout(
            lambda: PyPDFLoader(file_path).load(),
            60,
            "PDF 解析超时（60s）",
        )
    except Exception:
        # 兜底：尝试 unstructured
        try:
            from unstructured.partition.auto import partition
            return _run_with_timeout(
                lambda: partition(filename=file_path, strategy="auto"),
                120,
                "PDF 解析超时（unstructured 120s）",
            )
        except Exception:
            raise


def _partition_docx(file_path: str) -> List:
    try:
        from langchain_community.document_loaders import Docx2txtLoader
        return _run_with_timeout(
            lambda: Docx2txtLoader(file_path).load(),
            60,
            "Word 解析超时（60s）",
        )
    except Exception:
        try:
            from unstructured.partition.auto import partition
            return _run_with_timeout(
                lambda: partition(filename=file_path, strategy="auto"),
                120,
                "Word 解析超时（120s）",
            )
        except Exception:
            raise


def _extract_pdf_images(file_path: str) -> dict:
    """从 PDF 提取图片，并为每页保存渲染截图用于截图式图片检索。"""
    try:
        import fitz
    except ImportError:
        return {}

    page_images: dict[int, list[dict]] = {}
    try:
        from backend.indexing.multimodal_embedding import get_image_dir
        file_hash = hashlib.md5(Path(file_path).read_bytes()[:4096]).hexdigest()[:12]
        img_dir = get_image_dir() / file_hash
        img_dir.mkdir(parents=True, exist_ok=True)

        pdf = fitz.open(file_path)
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            page_imgs = []
            try:
                zoom = float(os.getenv("PDF_PAGE_RENDER_ZOOM", "1.5"))
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                render_path = img_dir / f"p{page_num + 1}_page.png"
                pix.save(str(render_path))
                render_bytes = render_path.read_bytes()
                render_hash = hashlib.md5(render_bytes).hexdigest()[:16]
                page_imgs.append({
                    "placeholder": f"<<PAGE_IMAGE:{render_hash}>>",
                    "path": str(render_path),
                    "page": page_num + 1,
                    "sort_order": -1,
                    "kind": "page_render",
                })
            except Exception:
                pass

            images = page.get_images(full=True)
            for idx, img in enumerate(images):
                xref = img[0]
                base_image = pdf.extract_image(xref)
                img_bytes = base_image["image"]
                ext = base_image["ext"]
                img_path = img_dir / f"p{page_num + 1}_{idx}.{ext}"
                img_path.write_bytes(img_bytes)
                img_hash = hashlib.md5(img_bytes).hexdigest()[:16]
                page_imgs.append({
                    "placeholder": f"<<IMAGE:{img_hash}>>",
                    "path": str(img_path),
                    "page": page_num + 1,
                    "sort_order": idx,
                    "kind": "embedded_image",
                })
            if page_imgs:
                page_images[page_num + 1] = page_imgs
        pdf.close()
    except Exception:
        pass
    return page_images


def _extract_catalog_context(text: str) -> str:
    """Extract section context such as （计算机体系结构/并行与分布计算/存储系统）."""
    if not text:
        return ""
    known_domains = (
        "计算机体系结构",
        "并行与分布计算",
        "存储系统",
        "计算机网络",
        "网络与信息安全",
        "软件工程",
        "系统软件",
        "数据库",
        "数据挖掘",
        "内容检索",
        "计算机科学理论",
        "计算机图形学",
        "多媒体",
        "人工智能",
        "人机交互",
        "交叉",
        "新兴",
    )
    matches = re.findall(r"[（(]([^（）()]{2,120})[）)]", text)
    for item in reversed(matches):
        normalized = item.strip()
        if "http" in normalized.lower() or "dblp" in normalized.lower():
            continue
        if "/" in normalized:
            parts = [part.strip() for part in normalized.split("/") if part.strip()]
            if parts and all(any(domain in part for domain in known_domains) for part in parts):
                return normalized
        elif any(domain == normalized or domain in normalized for domain in known_domains):
            return normalized
    return ""


# ===== DocumentLoader =====

class DocumentLoader:
    """文档加载和分片服务 — 语义分块 + LLM 结构标注 + 三级滑窗"""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        cs = chunk_size or _CHUNK_SIZE_L3
        co = chunk_overlap or _CHUNK_OVERLAP_L3
        level_1_size = max(1600, cs * 3)
        level_1_overlap = max(250, co * 3)
        level_2_size = max(1000, cs * 2)
        level_2_overlap = max(150, co * 2)
        level_3_size = max(400, cs)
        level_3_overlap = max(50, co)

        self._splitter_level_1 = RecursiveCharacterTextSplitter(
            chunk_size=level_1_size, chunk_overlap=level_1_overlap,
            add_start_index=True, separators=_SPLIT_SEPARATORS,
        )
        self._splitter_level_2 = RecursiveCharacterTextSplitter(
            chunk_size=level_2_size, chunk_overlap=level_2_overlap,
            add_start_index=True, separators=_SPLIT_SEPARATORS,
        )
        self._splitter_level_3 = RecursiveCharacterTextSplitter(
            chunk_size=level_3_size, chunk_overlap=level_3_overlap,
            add_start_index=True, separators=_SPLIT_SEPARATORS,
        )

    @staticmethod
    def _build_chunk_id(filename: str, page_number: int, level: int, index: int) -> str:
        return f"{filename}::p{page_number}::l{level}::{index}"

    def _split_page_to_three_levels(
        self,
        text: str,
        base_doc: Dict,
        page_global_chunk_idx: int,
        level_counters: Dict[int, int] | None = None,
    ) -> List[Dict]:
        if not text:
            return []

        protected = _protect_regions(text)
        root_chunks: List[Dict] = []
        page_number = int(base_doc.get("page_number", 0))
        filename = base_doc["filename"]
        if level_counters is None:
            level_counters = {1: 0, 2: 0, 3: 0}

        level_1_docs = self._splitter_level_1.create_documents([protected], [base_doc])
        for level_1_doc in level_1_docs:
            level_1_text = _restore_regions((level_1_doc.page_content or "").strip())
            if not level_1_text:
                continue
            level_1_id = self._build_chunk_id(filename, page_number, 1, level_counters.get(1, 0))
            level_counters[1] = level_counters.get(1, 0) + 1
            level_1_chunk = {**base_doc, "text": level_1_text, "chunk_id": level_1_id,
                             "parent_chunk_id": "", "root_chunk_id": level_1_id,
                             "chunk_level": 1, "chunk_idx": page_global_chunk_idx}
            page_global_chunk_idx += 1
            root_chunks.append(level_1_chunk)

            level_2_docs = self._splitter_level_2.create_documents([level_1_text], [base_doc])
            for level_2_doc in level_2_docs:
                level_2_text = _restore_regions((level_2_doc.page_content or "").strip())
                if not level_2_text:
                    continue
                level_2_id = self._build_chunk_id(filename, page_number, 2, level_counters.get(2, 0))
                level_counters[2] = level_counters.get(2, 0) + 1
                level_2_chunk = {**base_doc, "text": level_2_text, "chunk_id": level_2_id,
                                 "parent_chunk_id": level_1_id, "root_chunk_id": level_1_id,
                                 "chunk_level": 2, "chunk_idx": page_global_chunk_idx}
                page_global_chunk_idx += 1
                root_chunks.append(level_2_chunk)

                level_3_docs = self._splitter_level_3.create_documents([level_2_text], [base_doc])
                for level_3_doc in level_3_docs:
                    level_3_text = _restore_regions((level_3_doc.page_content or "").strip())
                    if not level_3_text:
                        continue
                    # Contextual Chunk Header
                    header = _generate_chunk_header(level_3_text, filename)
                    if header:
                        level_3_text = header + level_3_text
                    level_3_id = self._build_chunk_id(filename, page_number, 3, level_counters.get(3, 0))
                    level_counters[3] = level_counters.get(3, 0) + 1
                    root_chunks.append({**base_doc, "text": level_3_text, "chunk_id": level_3_id,
                                        "parent_chunk_id": level_2_id, "root_chunk_id": level_1_id,
                                        "chunk_level": 3, "chunk_idx": page_global_chunk_idx})
                    page_global_chunk_idx += 1
        return root_chunks

    def _load_from_langchain_docs(
        self, raw_docs: list, file_path: str, filename: str, doc_type: str,
        page_images: dict | None = None,
    ) -> list[dict]:
        documents: list[dict] = []
        page_global_chunk_idx = 0
        page_level_counters: dict[int, dict[int, int]] = {}
        _imgs = page_images or {}
        current_catalog_context = ""
        for doc in raw_docs:
            meta = getattr(doc, "metadata", None) or {}
            page_num = meta.get("page", 0)
            if page_num is None:
                page_num = 0
            try:
                page_num = int(page_num)
            except (TypeError, ValueError):
                page_num = 0

            raw_text = (doc.page_content or "").strip()
            page_catalog_context = _extract_catalog_context(raw_text)
            if page_catalog_context:
                current_catalog_context = page_catalog_context
            if current_catalog_context and current_catalog_context not in raw_text:
                raw_text = f"目录方向：{current_catalog_context}\n{raw_text}"

            # 优化流水线：LLM 结构标注 → 语义分块 → 三级滑窗
            segments = self._preprocess_and_segment(raw_text)
            level_counters = page_level_counters.setdefault(page_num, {1: 0, 2: 0, 3: 0})

            for seg_idx, segment_text in enumerate(segments):
                base_doc = {
                    "filename": filename,
                    "file_path": file_path,
                    "file_type": doc_type,
                    "page_number": page_num,
                }
                page_chunks = self._split_page_to_three_levels(
                    text=segment_text, base_doc=base_doc,
                    page_global_chunk_idx=page_global_chunk_idx,
                    level_counters=level_counters,
                )
                # 关联本页图片到所有 chunk
                imgs = _imgs.get(page_num + 1) or _imgs.get(page_num, [])
                if imgs:
                    for c in page_chunks:
                        c["images"] = imgs
                page_global_chunk_idx += len(page_chunks)
                documents.extend(page_chunks)
        return documents

    def _preprocess_and_segment(self, text: str) -> List[str]:
        """预处理流水线：LLM 结构标注 → 语义分块。"""
        if not text:
            return []
        # Step 1: LLM 结构标注（长文档）
        annotated = _annotate_structure_llm(text)
        # Step 2: 语义分块
        segments = _split_by_semantic_boundaries(annotated)
        return segments

    def load_document(self, file_path: str, filename: str) -> list[dict]:
        file_lower = filename.lower()
        page_images: dict = {}
        if file_lower.endswith(".pdf"):
            doc_type = "PDF"
            raw_docs = _partition_pdf(file_path)
            page_images = _extract_pdf_images(file_path)
        elif file_lower.endswith((".docx", ".doc")):
            doc_type = "Word"
            raw_docs = _partition_docx(file_path)
        elif file_lower.endswith((".xlsx", ".xls")):
            doc_type = "Excel"
            try:
                from langchain_community.document_loaders import UnstructuredExcelLoader

                loader = UnstructuredExcelLoader(file_path)
                raw_docs = loader.load()
            except Exception as e:
                raise Exception(f"处理 Excel 文档失败: {str(e)}") from e
        elif file_lower.endswith((".html", ".htm")):
            doc_type = "HTML"
            from backend.indexing.html_processor import load_html_for_document_loader
            raw_docs = load_html_for_document_loader(file_path, filename)
            return self._load_from_langchain_docs(raw_docs, file_path, filename, doc_type, page_images)
        else:
            raise ValueError(f"不支持的文件类型: {filename}")
        return self._load_from_langchain_docs(raw_docs, file_path, filename, doc_type, page_images)

    def load_document_from_text(self, text: str, filename: str, file_type: str = "TXT") -> list[dict]:
        from langchain_core.documents import Document
        raw_doc = Document(page_content=text, metadata={"source": filename, "page": 1})
        return self._load_from_langchain_docs(
            raw_docs=[raw_doc], file_path=filename, filename=filename, doc_type=file_type,
        )

    def load_documents_from_folder(self, folder_path: str) -> list[dict]:
        all_documents = []
        for filename in os.listdir(folder_path):
            file_lower = filename.lower()
            if not (file_lower.endswith(".pdf") or file_lower.endswith((".docx", ".doc"))
                    or file_lower.endswith((".xlsx", ".xls")) or file_lower.endswith((".html", ".htm"))):
                continue
            file_path = os.path.join(folder_path, filename)
            try:
                documents = self.load_document(file_path, filename)
                all_documents.extend(documents)
            except Exception:
                continue
        return all_documents
