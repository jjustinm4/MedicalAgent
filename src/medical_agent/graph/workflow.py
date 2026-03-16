from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from medical_agent.agents.nodes import AgentNodes
from medical_agent.config import Settings, load_settings
from medical_agent.llm import OllamaGemmaClient
from medical_agent.logging_utils import get_logger
from medical_agent.state import AgentState, init_state


logger = get_logger(__name__)


def build_graph(settings: Settings, llm_client: OllamaGemmaClient):
    nodes = AgentNodes(settings=settings, llm_client=llm_client)
    logger.info("Building workflow graph")

    graph = StateGraph(AgentState)
    graph.add_node("planner", nodes.planner)
    graph.add_node("clarification", nodes.clarification)
    graph.add_node("image_decision", nodes.image_decision)
    graph.add_node("cnn_tool", nodes.cnn_tool_node)
    graph.add_node("vlm_tool", nodes.vlm_tool_node)
    graph.add_node("research", nodes.research_node)
    graph.add_node("critic", nodes.critic)
    graph.add_node("response", nodes.response)

    graph.add_edge(START, "planner")
    
    graph.add_conditional_edges(
        "planner",
        lambda state: state.get("next_node", "image_decision"),
        {
            "clarification": "clarification",
            "image_decision": "image_decision",
        },
    )
    graph.add_edge("clarification", END)

    graph.add_conditional_edges(
        "image_decision",
        lambda state: state.get("next_node", "response"),
        {
            "cnn_tool": "cnn_tool",
            "vlm_tool": "vlm_tool",
            "research": "research",
            "response": "response",
        },
    )

    graph.add_edge("cnn_tool", "critic")
    graph.add_edge("vlm_tool", "critic")
    graph.add_edge("research", "critic")

    graph.add_conditional_edges(
        "critic",
        lambda state: state.get("next_node", "response"),
        {
            "image_decision": "image_decision",
            "response": "response",
        },
    )

    graph.add_edge("response", END)

    return graph.compile()


def run_workflow(image_path: str, user_query: str, settings: Settings | None = None) -> Dict[str, Any]:
    app_settings = settings or load_settings()
    logger.info(
        "Workflow started | image_path=%s | user_query=%s",
        image_path,
        user_query,
    )
    llm_client = OllamaGemmaClient(
        base_url=app_settings.ollama_base_url,
        model=app_settings.ollama_model,
    )

    graph = build_graph(app_settings, llm_client)
    state = init_state(image_path=image_path, user_query=user_query)
    final_state = graph.invoke(state)
    logger.info(
        "Workflow finished | analysis_type=%s | next_node=%s | retry_count=%s | error=%s",
        final_state.get("analysis_type"),
        final_state.get("next_node"),
        final_state.get("retry_count"),
        final_state.get("error"),
    )
    return dict(final_state)
