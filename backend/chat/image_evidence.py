def format_image_evidence_for_prompt(rag_trace: dict | None) -> str:
    """Format visual retrieval evidence so the answer model treats image search as evidence."""
    if not rag_trace or not rag_trace.get("has_query_image"):
        return ""

    matches = rag_trace.get("image_matches") or []
    ocr_text = (rag_trace.get("query_image_ocr_text") or "").strip()
    if not matches and not ocr_text:
        return ""

    lines = [
        "IMAGE_QUERY_EVIDENCE:",
        "用户本轮上传了图片。系统已经用图片向量/OCR完成检索；请基于以下候选证据回答。",
        "不要回答“无法查看图片”，不要要求用户重新描述图片。",
        "如果只能定位到文档或页面，请明确说明定位粒度；如果无法精确到图号/人物，也要说明边界。",
    ]
    if ocr_text:
        lines.append(f"OCR text from uploaded image: {ocr_text[:800]}")
    if matches:
        lines.append("Visual matches:")
    for idx, match in enumerate(matches[:5], 1):
        filename = match.get("filename", "Unknown")
        page = match.get("page_number", "N/A")
        image_page = match.get("image_page", page)
        kind = match.get("image_kind", "")
        score = match.get("score", match.get("image_score", ""))
        text = (match.get("text") or "").replace("\n", " ")[:500]
        score_text = f", score {float(score):.3f}" if isinstance(score, (int, float)) else ""
        lines.append(
            f"[image {idx}] {filename} (text page {page}, image page {image_page}, kind {kind}{score_text}): {text}"
        )
    return "\n".join(lines)
