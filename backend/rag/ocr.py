from __future__ import annotations

import os
from pathlib import Path


def extract_text_from_image(image_path: str) -> dict:
    """Best-effort OCR for query images.

    OCR is optional. Missing Python package or missing Tesseract binary returns an
    error field instead of raising, so visual retrieval remains available.
    """
    if os.getenv("OCR_ENABLED", "true").lower() == "false":
        return {"text": "", "error": "ocr_disabled"}

    path = Path(image_path)
    if not path.exists():
        return {"text": "", "error": "image_not_found"}

    try:
        import pytesseract
        from PIL import Image, ImageOps
    except Exception as exc:
        return {"text": "", "error": f"ocr_dependency_missing: {exc}"}

    tesseract_cmd = os.getenv("TESSERACT_CMD", "").strip()
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    try:
        image = Image.open(path)
        image = ImageOps.exif_transpose(image).convert("L")
        image = ImageOps.autocontrast(image)
        lang = os.getenv("OCR_LANG", "chi_sim+eng")
        text = pytesseract.image_to_string(image, lang=lang, config="--psm 6")
        text = " ".join((text or "").split())
        max_chars = int(os.getenv("OCR_MAX_CHARS", "800"))
        return {"text": text[:max_chars], "error": ""}
    except Exception as exc:
        return {"text": "", "error": str(exc)[:300]}
