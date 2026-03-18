from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medical_agent.config import load_settings
from medical_agent.graph.workflow import run_workflow
from medical_agent.llm import GeminiClient, OllamaClient, ResilientLLMClient
from medical_agent.logging_utils import configure_logging, get_logger


def _build_clarification_summary(questions: list[str], answers: list[str]) -> str:
    summary_parts: list[str] = []
    for question, answer in zip(questions, answers):
        normalized_answer = answer.strip()
        if not normalized_answer:
            continue

        lower_question = question.lower()
        lower_answer = normalized_answer.lower()
        if any(token in lower_question for token in ["x-ray", "xray", "ct", "mri", "scan"]):
            if any(token in lower_answer for token in ["yes", "scan", "xray", "x-ray", "ct", "mri", "image"]):
                summary_parts.append("The uploaded image is a medical scan.")
        elif any(token in lower_question for token in ["report", "prescription", "lab result", "document"]):
            if any(token in lower_answer for token in ["yes", "report", "document", "lab", "prescription"]):
                summary_parts.append("The uploaded image is a medical report.")
        elif any(token in lower_question for token in ["analyze", "explain", "search", "information"]):
            summary_parts.append(f"Requested task: {normalized_answer}.")
        else:
            summary_parts.append(f"{normalized_answer.rstrip('.')}.")

    return " ".join(summary_parts)


def _build_clarified_query(original_query: str, clarification_summary: str) -> str:
    base_query = original_query.strip() or "Please analyze this uploaded image."
    if not clarification_summary:
        return base_query
    return f"{base_query} Context: {clarification_summary}"


def _resolve_display_result(state: dict) -> dict[str, str]:
    final_response = str(state.get("final_response", "") or "").strip()
    if final_response:
        return {
            "answer": final_response,
            "status": "success",
            "source": "final_response",
        }

    if state.get("clarification_questions"):
        return {
            "answer": "Additional clarification is required before a final answer can be generated.",
            "status": "warning",
            "source": "clarification_required",
        }

    if state.get("error"):
        return {
            "answer": f"Workflow ended with an error: {state.get('error')}",
            "status": "error",
            "source": "workflow_error",
        }

    cnn_result = str(state.get("cnn_result", "") or "").strip()
    cnn_confidence = float(state.get("cnn_confidence", 0.0) or 0.0)
    vlm_result = str(state.get("vlm_result", "") or "").strip()

    if cnn_result or vlm_result:
        return {
            "answer": (
                "Fallback synthesis: "
                f"CNN output: {cnn_result or 'n/a'} (confidence {cnn_confidence:.2f}). "
                f"VLM output: {vlm_result or 'n/a'}."
            ),
            "status": "warning",
            "source": "tool_fallback",
        }

    return {
        "answer": "No final response was produced by the workflow.",
        "status": "info",
        "source": "empty_result",
    }


configure_logging()
logger = get_logger(__name__)


st.set_page_config(page_title="Local Agentic Medical Image Interpreter", page_icon="🧠", layout="wide")

settings = load_settings()
logger.info(
    "App initialized | gemini_model=%s | ollama_fallback_enabled=%s | ollama_model=%s | critic_threshold=%.2f",
    settings.gemini_model,
    settings.ollama_fallback_enabled,
    settings.ollama_model,
    settings.critic_confidence_threshold,
)

st.title("Local Agentic Medical Image Interpreter")
st.caption("LangGraph multi-agent orchestration using Gemini with Ollama fallback, TensorFlow CNN, BLIP VLM, and DuckDuckGo")

with st.sidebar:
    st.subheader("Runtime")
    st.write("Primary provider: Google Gemini")
    st.write(f"Gemini model: {settings.gemini_model}")
    st.write(f"Chest X-ray model: {settings.chest_xray_model}")
    st.write(f"Critic threshold: {settings.critic_confidence_threshold}")

    gemini_health_client = GeminiClient(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        base_url=settings.gemini_base_url,
        timeout_seconds=8,
        probe_timeout_seconds=3,
    )

    if gemini_health_client.ensure_available(force_recheck=True):
        st.success("Gemini status: ready")
    else:
        st.warning(f"Gemini status: unavailable ({gemini_health_client.disabled_reason})")

    ollama_health_client = None
    if settings.ollama_fallback_enabled:
        st.write(f"Fallback provider: Ollama ({settings.ollama_model})")
        ollama_health_client = OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout_seconds=8,
            probe_timeout_seconds=3,
        )
        if ollama_health_client.ensure_available(force_recheck=True):
            st.success("Ollama fallback status: ready")
        else:
            st.warning(f"Ollama fallback status: unavailable ({ollama_health_client.disabled_reason})")
    else:
        st.info("Ollama fallback status: disabled")

    routing_client = ResilientLLMClient(
        gemini_client=gemini_health_client,
        ollama_client=ollama_health_client,
        prefer_gemini=True,
    )
    if routing_client.ensure_available(force_recheck=False):
        st.caption(f"Active LLM route: {routing_client.active_provider}")
    else:
        st.caption("Active LLM route: unavailable")

    show_debug = st.checkbox("Show debug panel", value=True)

uploaded_file = st.file_uploader("Upload one image", type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"])
user_question = st.text_input("Optional question about the image", placeholder="Does this report indicate any abnormality?")
submit = st.button("Submit", type="primary", use_container_width=True)

if "history" not in st.session_state:
    st.session_state.history = []

if "current_state" not in st.session_state:
    st.session_state.current_state = None

if "awaiting_clarification" not in st.session_state:
    st.session_state.awaiting_clarification = False

if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

if submit:
    if uploaded_file is None:
        st.error("Please upload an image before submitting.")
        logger.warning("Submit blocked because no image was uploaded")
    else:
        upload_dir = ROOT_DIR / ".cache" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        image_path = upload_dir / uploaded_file.name
        image_path.write_bytes(uploaded_file.getbuffer())

        query = user_question.strip() or "Please analyze this uploaded image."
        logger.info(
            "Starting workflow from UI | filename=%s | query=%s",
            uploaded_file.name,
            query,
        )

        with st.spinner("Running multi-agent workflow..."):
            final_state = run_workflow(image_path=str(image_path), user_query=query, settings=settings)

        # Check if clarification is needed
        if final_state.get("clarification_questions"):
            logger.info(
                "Workflow requested clarification | questions=%s",
                final_state.get("clarification_questions", []),
            )
            result_payload = _resolve_display_result(final_state)
            st.session_state.latest_result = {
                "question": query,
                "answer": result_payload["answer"],
                "status": result_payload["status"],
                "source": result_payload["source"],
                "state": final_state,
            }
            st.session_state.current_state = final_state
            st.session_state.awaiting_clarification = True
            st.session_state.pending_question = query
        else:
            result_payload = _resolve_display_result(final_state)
            answer = result_payload["answer"]
            logger.info(
                "Workflow completed without clarification | final_response_length=%s",
                len(answer),
            )
            st.session_state.latest_result = {
                "question": query,
                "answer": answer,
                "status": result_payload["status"],
                "source": result_payload["source"],
                "state": final_state,
            }
            st.session_state.history.append(
                {
                    "question": query,
                    "answer": answer,
                    "state": final_state,
                }
            )
            st.session_state.awaiting_clarification = False

# Handle clarification flow
if st.session_state.awaiting_clarification and st.session_state.current_state:
    state = st.session_state.current_state
    st.info("Your question was unclear. Please help clarify:")

    with st.form("clarification_form"):
        clarification_answers: list[str] = []
        for i, question in enumerate(state.get("clarification_questions", [])):
            answer = st.text_input(f"Q{i+1}: {question}", key=f"clarification_{i}")
            clarification_answers.append(answer)

        clarify_submit = st.form_submit_button("Submit Clarification", type="primary")

    if clarify_submit:
        clarification_summary = _build_clarification_summary(
            state.get("clarification_questions", []),
            clarification_answers,
        )
        updated_query = _build_clarified_query(
            st.session_state.pending_question or state.get("user_query", ""),
            clarification_summary,
        )
        logger.info(
            "Submitting clarification | original_query=%s | clarification_summary=%s",
            st.session_state.pending_question or state.get("user_query", ""),
            clarification_summary,
        )

        with st.spinner("Continuing analysis with clarification..."):
            final_state = run_workflow(
                image_path=state.get("image_path", ""),
                user_query=updated_query,
                settings=settings
            )

        final_state["clarification_answer"] = clarification_summary
        result_payload = _resolve_display_result(final_state)
        answer = result_payload["answer"]
        logger.info(
            "Workflow completed after clarification | final_response_length=%s",
            len(answer),
        )
        st.session_state.latest_result = {
            "question": st.session_state.pending_question or state.get("user_query", ""),
            "answer": answer,
            "status": result_payload["status"],
            "source": result_payload["source"],
            "state": final_state,
        }
        st.session_state.history.append(
            {
                "question": st.session_state.pending_question or state.get("user_query", ""),
                "answer": answer,
                "state": final_state,
            }
        )
        st.session_state.awaiting_clarification = False
        st.session_state.current_state = None
        st.session_state.pending_question = ""
        st.rerun()

if st.session_state.latest_result:
    st.subheader("Latest Result")
    latest_answer = st.session_state.latest_result["answer"]
    latest_status = st.session_state.latest_result.get("status", "info")
    latest_source = st.session_state.latest_result.get("source", "unknown")
    if latest_status == "success":
        st.success(latest_answer)
    elif latest_status == "warning":
        st.warning(latest_answer)
    elif latest_status == "error":
        st.error(latest_answer)
    else:
        st.info(latest_answer)
    st.caption(f"Result source: {latest_source}")

for item in st.session_state.history:
    with st.chat_message("user"):
        st.write(item["question"])
    with st.chat_message("assistant"):
        st.write(item["answer"])

if show_debug and st.session_state.history:
    latest = st.session_state.history[-1]["state"]
    with st.expander("Debug: reasoning trace", expanded=True):
        trace = latest.get("reasoning_trace", [])
        if trace:
            for row in trace:
                st.write(f"- {row}")
        else:
            st.write("No trace captured.")

    with st.expander("Debug: tool calls", expanded=True):
        st.json(latest.get("tool_calls", []))

    with st.expander("Debug: intermediate outputs", expanded=False):
        st.json(
            {
                "analysis_type": latest.get("analysis_type"),
                "cnn_result": latest.get("cnn_result"),
                "cnn_confidence": latest.get("cnn_confidence"),
                "vlm_result": latest.get("vlm_result"),
                "search_results": latest.get("search_results"),
                "retry_count": latest.get("retry_count"),
                "error": latest.get("error"),
            }
        )
