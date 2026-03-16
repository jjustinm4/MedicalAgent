from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from medical_agent.config import Settings
from medical_agent.llm import OllamaGemmaClient
from medical_agent.logging_utils import get_logger
from medical_agent.state import AgentState
from medical_agent.tools.cnn_tool import analyze_scan_with_cnn
from medical_agent.tools.search_tool import duckduckgo_search
from medical_agent.tools.vlm_tool import analyze_image_with_vlm


logger = get_logger(__name__)


class AgentNodes:
    def __init__(self, settings: Settings, llm_client: OllamaGemmaClient):
        self.settings = settings
        self.llm = llm_client

    @staticmethod
    def _normalize_query_text(query: str) -> str:
        return " ".join(query.lower().split())

    @classmethod
    def _fallback_planner_decision(cls, query: str) -> tuple[str, bool, bool, str]:
        normalized_query = cls._normalize_query_text(query)
        scan_terms = {
            "scan",
            "xray",
            "x-ray",
            "ct",
            "mri",
            "ultrasound",
            "cxr",
            "chest xray",
            "chest x-ray",
            "image",
        }
        finding_terms = {
            "pneumonia",
            "fracture",
            "effusion",
            "opacity",
            "nodule",
            "tumor",
            "lesion",
            "infection",
            "consolidation",
            "infiltrate",
            "edema",
        }
        document_terms = {
            "report",
            "prescription",
            "document",
            "lab result",
            "lab",
            "blood test",
            "discharge summary",
            "notes",
            "text",
        }
        research_terms = {
            "meaning",
            "explain",
            "condition",
            "why",
            "cause",
            "treatment",
            "symptom",
            "research",
            "more information",
        }
        vague_phrases = {
            "what is this",
            "analyze this",
            "analyze image",
            "check this",
            "help me",
        }

        analysis_type = "unknown"
        if any(term in normalized_query for term in scan_terms | finding_terms):
            analysis_type = "scan"
        elif any(term in normalized_query for term in document_terms):
            analysis_type = "document"

        need_research = any(term in normalized_query for term in research_terms)
        is_vague = analysis_type == "unknown" and (
            any(phrase in normalized_query for phrase in vague_phrases) or len(normalized_query.split()) <= 2
        )
        plan = "Fallback planner heuristics applied."
        return analysis_type, need_research, is_vague, plan

    @classmethod
    def _research_query_seed(cls, state: AgentState) -> str:
        query_seed = state.get("user_query", "").strip()
        if "Context:" in query_seed:
            query_seed = query_seed.split("Context:", 1)[0].strip()
        if query_seed.lower().startswith("original:"):
            query_seed = query_seed.split(":", 1)[1].strip()

        if not query_seed:
            query_seed = state.get("vlm_result", "") or state.get("cnn_result", "")

        query = " ".join(query_seed.split()[:12])
        if state.get("analysis_type") == "scan" and "xray" not in query.lower() and "x-ray" not in query.lower():
            vlm_result = state.get("vlm_result", "").lower()
            if "x ray" in vlm_result or "x-ray" in vlm_result:
                query = f"{query} x-ray".strip()

        return query

    @staticmethod
    def _trace(state: AgentState, message: str) -> List[str]:
        timestamp = datetime.now().strftime("%H:%M:%S")
        traces = list(state.get("reasoning_trace", []))
        traces.append(f"[{timestamp}] {message}")
        logger.info(message)
        return traces

    @staticmethod
    def _record_tool_call(state: AgentState, tool_name: str, details: Dict[str, Any]) -> List[Dict[str, Any]]:
        calls = list(state.get("tool_calls", []))
        calls.append(
            {
                "tool": tool_name,
                "details": details,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )
        return calls

    def planner(self, state: AgentState) -> Dict[str, Any]:
        prompt = (
            "You are PlannerAgent for a local multimodal medical-learning workflow. "
            "Given user query, choose analysis_type and whether web research is useful.\n\n"
            f"User query: {state.get('user_query', '')}\n\n"
            "Return strict JSON: "
            "{\"analysis_type\":\"scan|document|unknown\",\"need_research\":true|false,\"plan\":\"...\",\"is_vague\":true|false}."
        )

        analysis_type = "unknown"
        need_research = False
        plan = "Plan unavailable."
        is_vague = False

        try:
            decision = self.llm.generate_json(prompt=prompt)
            candidate_type = str(decision.get("analysis_type", "unknown")).lower().strip()
            if candidate_type in {"scan", "document", "unknown"}:
                analysis_type = candidate_type
            need_research = bool(decision.get("need_research", False))
            plan = str(decision.get("plan", "Dynamic tool routing based on confidence."))
            is_vague = bool(decision.get("is_vague", False))
        except Exception as exc:
            analysis_type, need_research, is_vague, plan = self._fallback_planner_decision(
                state.get("user_query", "")
            )
            plan = f"{plan} LLM unavailable: {exc}"[:300]

        next_node = "clarification" if is_vague else "image_decision"

        return {
            "analysis_type": analysis_type,
            "need_research": need_research,
            "needs_clarification": is_vague,
            "next_node": next_node,
            "reasoning_trace": self._trace(
                state,
                f"PlannerAgent set analysis_type='{analysis_type}', vague={is_vague}, need_research={need_research}. Plan: {plan}",
            ),
        }

    def clarification(self, state: AgentState) -> Dict[str, Any]:
        """Ask clarifying questions if the user query is vague."""
        prompt = (
            "You are ClarificationAgent. The user's query about their medical image is vague. "
            "Generate 2-3 clarifying yes/no or multiple-choice questions to better understand what they mean.\n\n"
            f"User query: {state.get('user_query', '')}\n\n"
            "Return strict JSON: "
            "{\"questions\":[\"Q1: ...\",\"Q2: ...\",\"Q3: ...\"],\"guidance\":\"...\"}"
        )

        questions = []
        guidance = ""

        try:
            decision = self.llm.generate_json(prompt=prompt)
            questions = decision.get("questions", [])
            guidance = str(decision.get("guidance", ""))
        except Exception as exc:
            # Fallback: ask common disambiguating questions
            questions = [
                "Is the uploaded image a medical scan such as an X-ray, CT, or MRI?",
                "Is it instead a text-based medical report, prescription, or lab result?",
                "What would you like me to do with it: analyze the image, explain medical terms, or search for related information?",
            ]
            guidance = "Fallback clarification questions used."

        return {
            "clarification_questions": questions,
            "needs_clarification": True,
            "next_node": "clarification",
            "reasoning_trace": self._trace(
                state,
                f"ClarificationAgent generated {len(questions)} questions to disambiguate user intent.",
            ),
        }

    def image_decision(self, state: AgentState) -> Dict[str, Any]:
        retry_count = int(state.get("retry_count", 0))
        if retry_count > self.settings.max_retry_loops:
            return {
                "next_node": "response",
                "reasoning_trace": self._trace(
                    state,
                    "ImageDecisionAgent reached retry limit; routing to ResponseAgent.",
                ),
            }

        preferred_tool = str(state.get("preferred_tool", "")).strip()
        if preferred_tool in {"cnn_tool", "vlm_tool", "research"}:
            return {
                "next_node": preferred_tool,
                "preferred_tool": "",
                "reasoning_trace": self._trace(
                    state,
                    f"ImageDecisionAgent obeying CriticAgent suggestion: {preferred_tool}.",
                ),
            }

        prompt = (
            "You are ImageDecisionAgent. Choose the next node in a tool-using workflow.\n"
            "Allowed next_node values: cnn_tool, vlm_tool, research, response.\n\n"
            f"analysis_type={state.get('analysis_type', 'unknown')}\n"
            f"need_research={state.get('need_research', False)}\n"
            f"cnn_result_present={bool(state.get('cnn_result'))}\n"
            f"cnn_confidence={state.get('cnn_confidence', 0.0)}\n"
            f"vlm_result_present={bool(state.get('vlm_result'))}\n"
            f"search_results_count={len(state.get('search_results', []))}\n"
            f"user_query={state.get('user_query', '')}\n\n"
            "Return strict JSON: {\"next_node\":\"...\",\"reason\":\"...\"}."
        )

        next_node = "response"
        reason = "No additional tools needed."

        try:
            decision = self.llm.generate_json(prompt=prompt)
            candidate = str(decision.get("next_node", "response")).strip()
            if candidate in {"cnn_tool", "vlm_tool", "research", "response"}:
                next_node = candidate
            reason = str(decision.get("reason", reason))
        except Exception:
            if not state.get("cnn_result") and state.get("analysis_type") in {"scan", "unknown"}:
                next_node = "cnn_tool"
                reason = "Fallback: start with CNN for scan/unknown images."
            elif not state.get("vlm_result"):
                next_node = "vlm_tool"
                reason = "Fallback: use VLM for text-heavy or uncertain images."
            elif state.get("need_research") and not state.get("search_results"):
                next_node = "research"
                reason = "Fallback: collect external context from DuckDuckGo."
            else:
                next_node = "response"
                reason = "Fallback: enough context for final response."

        if next_node == "cnn_tool" and state.get("cnn_result"):
            next_node = "vlm_tool" if not state.get("vlm_result") else "response"
        if next_node == "vlm_tool" and state.get("vlm_result"):
            next_node = "research" if state.get("need_research") and not state.get("search_results") else "response"
        if next_node == "research" and state.get("search_results"):
            next_node = "response"

        return {
            "next_node": next_node,
            "reasoning_trace": self._trace(
                state,
                f"ImageDecisionAgent routed to '{next_node}'. Reason: {reason}",
            ),
        }

    def cnn_tool_node(self, state: AgentState) -> Dict[str, Any]:
        try:
            result = analyze_scan_with_cnn(state["image_path"])
            return {
                "cnn_result": result["summary"],
                "cnn_confidence": float(result["confidence"]),
                "cnn_raw_predictions": result["raw_predictions"],
                "tool_calls": self._record_tool_call(
                    state,
                    "cnn_tool",
                    {
                        "confidence": float(result["confidence"]),
                        "summary": result["summary"],
                    },
                ),
                "reasoning_trace": self._trace(
                    state,
                    f"CNNToolNode completed with confidence={result['confidence']:.2f}.",
                ),
            }
        except Exception as exc:
            return {
                "error": str(exc),
                "reasoning_trace": self._trace(state, f"CNNToolNode failed: {exc}"),
            }

    def vlm_tool_node(self, state: AgentState) -> Dict[str, Any]:
        try:
            result = analyze_image_with_vlm(
                image_path=state["image_path"],
                user_question=state.get("user_query", ""),
                caption_model_name=self.settings.blip_caption_model,
                vqa_model_name=self.settings.blip_vqa_model,
            )
            return {
                "vlm_result": result["summary"],
                "tool_calls": self._record_tool_call(
                    state,
                    "vlm_tool",
                    {
                        "caption": result.get("caption", ""),
                        "answer": result.get("answer", ""),
                    },
                ),
                "reasoning_trace": self._trace(
                    state,
                    "VLMToolNode completed BLIP caption/QA analysis.",
                ),
            }
        except Exception as exc:
            return {
                "error": str(exc),
                "reasoning_trace": self._trace(state, f"VLMToolNode failed: {exc}"),
            }

    def research_node(self, state: AgentState) -> Dict[str, Any]:
        try:
            query = self._research_query_seed(state)

            payload = duckduckgo_search(query=query, max_results=3)
            return {
                "search_results": payload["results"],
                "tool_calls": self._record_tool_call(
                    state,
                    "research",
                    {
                        "query": payload["query"],
                        "result_count": len(payload["results"]),
                    },
                ),
                "reasoning_trace": self._trace(
                    state,
                    f"ResearchAgent fetched {len(payload['results'])} DuckDuckGo results for query='{payload['query']}'.",
                ),
            }
        except Exception as exc:
            return {
                "error": str(exc),
                "reasoning_trace": self._trace(state, f"ResearchAgent failed: {exc}"),
            }

    def critic(self, state: AgentState) -> Dict[str, Any]:
        retry_count = int(state.get("retry_count", 0))
        cnn_confidence = float(state.get("cnn_confidence", 0.0))

        decision = "proceed"
        suggested_tool = ""
        reason = "Current evidence is sufficient for response synthesis."

        if retry_count >= self.settings.max_retry_loops:
            decision = "proceed"
            reason = "Retry limit reached."
        elif state.get("cnn_result") and cnn_confidence < self.settings.critic_confidence_threshold and not state.get("vlm_result"):
            decision = "retry"
            suggested_tool = "vlm_tool"
            reason = "CNN confidence is low; switch to VLM for document/text interpretation."
        elif not state.get("cnn_result") and state.get("analysis_type") in {"scan", "unknown"}:
            decision = "retry"
            suggested_tool = "cnn_tool"
            reason = "No CNN output yet for scan-like context."
        elif not state.get("vlm_result") and state.get("analysis_type") in {"document", "unknown"}:
            decision = "retry"
            suggested_tool = "vlm_tool"
            reason = "No VLM output yet for document-like context."
        elif state.get("need_research") and not state.get("search_results"):
            decision = "retry"
            suggested_tool = "research"
            reason = "Research requested but no external context gathered yet."

        next_node = "response"
        new_retry_count = retry_count
        if decision == "retry":
            next_node = "image_decision"
            new_retry_count += 1

        return {
            "preferred_tool": suggested_tool,
            "retry_count": new_retry_count,
            "next_node": next_node,
            "reasoning_trace": self._trace(
                state,
                (
                    "CriticAgent decision='{}', suggested_tool='{}', reason='{}', "
                    "retry_count={}."
                ).format(decision, suggested_tool or "none", reason, new_retry_count),
            ),
        }

    def response(self, state: AgentState) -> Dict[str, Any]:
        search_blob = "\n".join(
            [
                f"- {item.get('title', '')}: {item.get('snippet', '')} ({item.get('url', '')})"
                for item in state.get("search_results", [])
            ]
        )
        synthesis_prompt = (
            "You are FinalResponseAgent in a learning-focused local medical multimodal project. "
            "Provide a clear, human-readable explanation combining all tool outputs. "
            "Do not claim diagnosis. Include a short safety note.\n\n"
            f"User query: {state.get('user_query', '')}\n"
            f"CNN summary: {state.get('cnn_result', '')}\n"
            f"CNN confidence: {state.get('cnn_confidence', 0.0)}\n"
            f"VLM summary: {state.get('vlm_result', '')}\n"
            f"Web research:\n{search_blob if search_blob else '- none'}\n"
            f"Reasoning trace:\n{chr(10).join(state.get('reasoning_trace', [])[-8:])}\n"
        )

        try:
            final_text = self.llm.generate_text(prompt=synthesis_prompt)
        except Exception:
            final_text = (
                "The system completed local multimodal analysis. "
                f"CNN output: {state.get('cnn_result', 'n/a')} "
                f"(confidence {state.get('cnn_confidence', 0.0):.2f}). "
                f"VLM output: {state.get('vlm_result', 'n/a')}. "
                "This is educational output and not a medical diagnosis."
            )

        return {
            "final_response": final_text,
            "reasoning_trace": self._trace(state, "FinalResponseAgent generated the final answer."),
        }
