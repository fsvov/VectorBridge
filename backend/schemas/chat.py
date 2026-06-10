from typing import List, Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default_session"
    query_image_base64: Optional[str] = None
    query_image_path: Optional[str] = None


class RetrievedChunk(BaseModel):
    filename: str
    page_number: Optional[str | int] = None
    text: Optional[str] = None
    score: Optional[float] = None
    rrf_rank: Optional[int] = None
    rerank_score: Optional[float] = None


class RagTrace(BaseModel):
    tool_used: bool
    tool_name: str
    query: Optional[str] = None
    expanded_query: Optional[str] = None
    step_back_question: Optional[str] = None
    step_back_answer: Optional[str] = None
    expansion_type: Optional[str] = None
    hypothetical_doc: Optional[str] = None
    retrieval_stage: Optional[str] = None
    grade_score: Optional[str] = None
    grade_route: Optional[str] = None
    rewrite_needed: Optional[bool] = None
    rewrite_strategy: Optional[str] = None
    rewrite_query: Optional[str] = None
    rerank_enabled: Optional[bool] = None
    rerank_applied: Optional[bool] = None
    rerank_model: Optional[str] = None
    rerank_endpoint: Optional[str] = None
    rerank_error: Optional[str] = None
    retrieval_mode: Optional[str] = None
    retrieval_pipeline: Optional[str] = None
    candidate_k: Optional[int] = None
    candidate_k_source: Optional[str] = None
    candidate_k_config_error: Optional[str] = None
    retrieval_candidate_multiplier: Optional[int] = None
    retrieval_top_k: Optional[int] = None
    recall_count: Optional[int] = None
    post_merge_candidate_count: Optional[int] = None
    candidate_count: Optional[int] = None
    leaf_retrieve_level: Optional[int] = None
    auto_merge_enabled: Optional[bool] = None
    auto_merge_applied: Optional[bool] = None
    auto_merge_threshold: Optional[int] = None
    auto_merge_replaced_chunks: Optional[int] = None
    auto_merge_steps: Optional[int] = None
    retrieved_chunks: Optional[List[RetrievedChunk]] = None
    initial_retrieved_chunks: Optional[List[RetrievedChunk]] = None
    expanded_retrieved_chunks: Optional[List[RetrievedChunk]] = None
    # 复杂度路由新增字段
    complexity: Optional[str] = None
    complexity_reason: Optional[str] = None
    sub_questions: Optional[List[str]] = None
    sub_agent_count: Optional[int] = None
    synthesis_merged_count: Optional[int] = None
    sub_traces: Optional[List[dict]] = None
    has_query_image: Optional[bool] = None
    image_retrieval_enabled: Optional[bool] = None
    image_retrieval_error: Optional[str] = None
    image_retrieval_min_score: Optional[float] = None
    image_retrieval_fallback_min_score: Optional[float] = None
    image_context_fallback: Optional[bool] = None
    image_matches: Optional[List[dict]] = None
    query_image_ocr_enabled: Optional[bool] = None
    query_image_ocr_text: Optional[str] = None
    query_image_ocr_error: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    rag_trace: Optional[RagTrace] = None


class MessageInfo(BaseModel):
    type: str
    content: str
    timestamp: str
    rag_trace: Optional[RagTrace] = None


class SessionMessagesResponse(BaseModel):
    messages: List[MessageInfo]


class SessionInfo(BaseModel):
    session_id: str
    title: Optional[str] = None
    updated_at: str
    message_count: int


class SessionListResponse(BaseModel):
    sessions: List[SessionInfo]


class SessionDeleteResponse(BaseModel):
    session_id: str
    message: str
