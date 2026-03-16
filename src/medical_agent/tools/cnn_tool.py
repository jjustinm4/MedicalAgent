from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import (
    EfficientNetB0,
    decode_predictions,
    preprocess_input,
)

from medical_agent.logging_utils import get_logger


logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_cnn_model() -> tf.keras.Model:
    logger.info("Loading CNN model EfficientNetB0 with ImageNet weights")
    return EfficientNetB0(weights="imagenet")


def _normalize_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    return preprocess_input(image_array)


def _interpret_predictions(decoded: List[Tuple[str, str, float]]) -> Tuple[str, float]:
    best = decoded[0]
    _, label, confidence = best

    if confidence < 0.40:
        summary = (
            "Possible abnormal visual pattern detected with low confidence "
            "(proxy estimate from a generic pretrained CNN)."
        )
    else:
        summary = (
            f"Top CNN visual pattern: '{label}'. This is a generic image classifier result and "
            "not a medical diagnosis."
        )

    return summary, float(confidence)


def analyze_scan_with_cnn(image_path: str) -> Dict[str, Any]:
    logger.info("CNN analysis started | image_path=%s", image_path)
    model = _load_cnn_model()
    tensor = _normalize_image(image_path)
    predictions = model.predict(tensor, verbose=0)
    decoded = decode_predictions(predictions, top=3)[0]
    logger.info(
        "CNN top predictions | %s",
        [
            {
                "label": label,
                "confidence": round(float(score), 4),
            }
            for _, label, score in decoded
        ],
    )

    summary, confidence = _interpret_predictions(decoded)
    raw_predictions = [
        {
            "label": label,
            "confidence": float(score),
        }
        for _, label, score in decoded
    ]

    logger.info(
        "CNN analysis completed | summary=%s | confidence=%.4f",
        summary,
        confidence,
    )

    return {
        "summary": summary,
        "confidence": confidence,
        "raw_predictions": raw_predictions,
        "tool": "cnn",
    }
