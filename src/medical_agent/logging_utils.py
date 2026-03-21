from __future__ import annotations

import logging
import os
import sys


def configure_logging(level: str | None = None) -> logging.Logger:
    logger = logging.getLogger("medical_agent")
    log_level_name = (level or os.getenv("MEDICAL_AGENT_LOG_LEVEL", "INFO")).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)

    logger.setLevel(log_level)
    logger.propagate = False
    return logger


def get_logger(module_name: str) -> logging.Logger:
    configure_logging()
    if module_name.startswith("medical_agent"):
        return logging.getLogger(module_name)
    return logging.getLogger(f"medical_agent.{module_name}")