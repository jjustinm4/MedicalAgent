from __future__ import annotations

from typing import Any, Dict, List

from duckduckgo_search import DDGS

from medical_agent.logging_utils import get_logger


logger = get_logger(__name__)


def duckduckgo_search(query: str, max_results: int = 3) -> Dict[str, Any]:
    if not query.strip():
        logger.info("Research skipped because query is empty")
        return {"query": query, "results": []}

    logger.info("Research started | query=%s | max_results=%s", query, max_results)
    results: List[Dict[str, str]] = []
    with DDGS() as ddgs:
        for item in ddgs.text(query, max_results=max_results):
            results.append(
                {
                    "title": item.get("title", ""),
                    "snippet": item.get("body", ""),
                    "url": item.get("href", ""),
                }
            )

    logger.info("Research completed | query=%s | result_count=%s", query, len(results))

    return {
        "query": query,
        "results": results,
    }
