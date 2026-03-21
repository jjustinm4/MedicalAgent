from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    user_query: str
    image_path: str
    analysis_type: str
    need_research: bool
    preferred_tool: str
    next_node: str
    cnn_result: str
    cnn_confidence: float
    cnn_raw_predictions: List[Dict[str, Any]]
    vlm_result: str
    search_results: List[Dict[str, str]]
    tool_calls: List[Dict[str, Any]]
    reasoning_trace: List[str]
    retry_count: int
    final_response: str
    error: Optional[str]
    clarification_questions: List[str]
    clarification_answer: Optional[str]
    needs_clarification: bool


def init_state(image_path: str, user_query: str) -> AgentState:
    return AgentState(
        user_query=user_query or "Describe this uploaded image.",
        image_path=image_path,
        analysis_type="unknown",
        need_research=False,
        preferred_tool="",
        next_node="image_decision",
        cnn_result="",
        cnn_confidence=0.0,
        cnn_raw_predictions=[],
        vlm_result="",
        search_results=[],
        tool_calls=[],
        reasoning_trace=[],
        retry_count=0,
        final_response="",
        error=None,
        clarification_questions=[],
        clarification_answer=None,
        needs_clarification=False,
    )
