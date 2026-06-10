import base64
import json
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from backend.chat import chat_with_agent, chat_with_agent_stream
from backend.db.models import User
from backend.infra.auth import get_current_user
from backend.schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])

_QUERY_IMAGE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "query_images"
_MAX_QUERY_IMAGE_BYTES = 8 * 1024 * 1024


def _image_extension(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return ".webp"
    return ".jpg"


def _save_query_image(query_image_base64: str | None, session_id: str) -> str | None:
    if not query_image_base64:
        return None
    payload = query_image_base64.strip()
    if "," in payload and payload.lower().startswith("data:"):
        payload = payload.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(payload, validate=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图片数据不是有效 base64: {e}") from e
    if not image_bytes:
        return None
    if len(image_bytes) > _MAX_QUERY_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="图片过大，请上传 8MB 以内的图片")

    safe_session = re.sub(r"[^a-zA-Z0-9_.-]", "_", session_id or "default_session")
    target_dir = _QUERY_IMAGE_DIR / safe_session
    target_dir.mkdir(parents=True, exist_ok=True)
    ext = _image_extension(image_bytes)
    image_path = target_dir / f"{uuid.uuid4().hex}{ext}"
    image_path.write_bytes(image_bytes)
    return str(image_path)


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, current_user: User = Depends(get_current_user)):
    try:
        session_id = request.session_id or "default_session"
        query_image_path = _save_query_image(request.query_image_base64, session_id)
        resp = chat_with_agent(request.message, current_user.username, session_id, query_image_path)
        if isinstance(resp, dict):
            return ChatResponse(**resp)
        return ChatResponse(response=resp)
    except Exception as e:
        message = str(e)
        match = re.search(r"Error code:\s*(\d{3})", message)
        if match:
            code = int(match.group(1))
            if code == 429:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        "上游模型服务触发限流/额度限制（429）。请检查账号额度/模型状态。\n"
                        f"原始错误：{message}"
                    ),
                )
            if code in (401, 403):
                raise HTTPException(status_code=code, detail=message)
            raise HTTPException(status_code=code, detail=message)
        raise HTTPException(status_code=500, detail=message)


@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, current_user: User = Depends(get_current_user)):
    session_id = request.session_id or "default_session"
    query_image_path = _save_query_image(request.query_image_base64, session_id)

    async def event_generator():
        try:
            async for chunk in chat_with_agent_stream(
                request.message,
                current_user.username,
                session_id,
                query_image_path,
            ):
                yield chunk
        except Exception as e:
            error_data = {"type": "error", "content": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
