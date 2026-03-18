from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from medical_agent.logging_utils import get_logger


logger = get_logger(__name__)
DEFAULT_CHEST_XRAY_MODEL = "dima806/chest_xray_pneumonia_detection"


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=2)
def _load_medical_model(model_name: str) -> tuple[AutoImageProcessor, AutoModelForImageClassification]:
    logger.info("Loading medical image classifier | model=%s", model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.to(_device())
    model.eval()
    return processor, model


def _summarize_medical_prediction(label: str, confidence: float) -> str:
    normalized_label = label.lower().replace("_", " ").strip()

    if "pneumonia" in normalized_label:
        return (
            "Chest X-ray model indicates pneumonia-like visual findings. "
            "This is educational output and not a diagnosis."
        )
    if "normal" in normalized_label or "no finding" in normalized_label:
        return (
            "Chest X-ray model indicates a normal/no-finding pattern. "
            "This is educational output and not a diagnosis."
        )
    if confidence < 0.55:
        return (
            "Chest X-ray model output is low confidence and uncertain. "
            "Use the VLM/LLM path and clinical review for context."
        )

    return (
        f"Chest X-ray model top class: '{label}' with moderate confidence. "
        "This is educational output and not a diagnosis."
    )


def _analyze_with_medical_model(image_path: str, model_name: str) -> Dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    device = _device()
    processor, model = _load_medical_model(model_name)

    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probabilities = torch.softmax(logits, dim=-1)[0].detach().cpu()
    top_k = min(3, int(probabilities.shape[0]))
    top_scores, top_indices = torch.topk(probabilities, k=top_k)
    id2label = getattr(model.config, "id2label", {}) or {}

    raw_predictions = [
        {
            "label": str(id2label.get(int(index.item()), f"class_{int(index.item())}")),
            "confidence": float(score.item()),
        }
        for score, index in zip(top_scores, top_indices)
    ]

    logger.info("Medical CNN top predictions | %s", raw_predictions)

    best = raw_predictions[0]
    summary = _summarize_medical_prediction(best["label"], float(best["confidence"]))
    return {
        "summary": summary,
        "confidence": float(best["confidence"]),
        "raw_predictions": raw_predictions,
        "tool": "cnn_medical_classifier",
        "model_name": model_name,
    }


@lru_cache(maxsize=1)
def _load_generic_fallback_model():
    from tensorflow.keras.applications.efficientnet import EfficientNetB0

    logger.info("Loading fallback CNN model EfficientNetB0 with ImageNet weights")
    return EfficientNetB0(weights="imagenet")


def _analyze_with_generic_fallback(image_path: str) -> Dict[str, Any]:
    import numpy as np
    from tensorflow.keras.applications.efficientnet import decode_predictions, preprocess_input

    image = Image.open(image_path).convert("RGB").resize((224, 224))
    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    tensor = preprocess_input(image_array)

    model = _load_generic_fallback_model()
    predictions = model.predict(tensor, verbose=0)
    decoded = decode_predictions(predictions, top=3)[0]

    raw_predictions = [
        {
            "label": label,
            "confidence": float(score),
        }
        for _, label, score in decoded
    ]

    best = raw_predictions[0]
    confidence = float(best["confidence"])
    if confidence < 0.40:
        summary = (
            "Possible abnormal visual pattern detected with low confidence "
            "(fallback generic CNN; not a medical model)."
        )
    else:
        summary = (
            f"Fallback generic CNN top visual class: '{best['label']}'. "
            "This is educational output and not a diagnosis."
        )

    logger.info("Fallback CNN top predictions | %s", raw_predictions)
    return {
        "summary": summary,
        "confidence": confidence,
        "raw_predictions": raw_predictions,
        "tool": "cnn_generic_fallback",
        "model_name": "EfficientNetB0(ImageNet)",
    }


def analyze_scan_with_cnn(image_path: str, model_name: str = DEFAULT_CHEST_XRAY_MODEL) -> Dict[str, Any]:
    logger.info("CNN analysis started | image_path=%s | model=%s", image_path, model_name)
    try:
        result = _analyze_with_medical_model(image_path=image_path, model_name=model_name)
        logger.info(
            "Medical CNN analysis completed | model=%s | confidence=%.4f",
            result.get("model_name"),
            float(result.get("confidence", 0.0)),
        )
        return result
    except Exception as exc:
        logger.exception(
            "Medical CNN analysis failed, using generic fallback | model=%s | error=%s",
            model_name,
            exc,
        )

    fallback_result = _analyze_with_generic_fallback(image_path=image_path)
    logger.info(
        "Fallback CNN analysis completed | model=%s | confidence=%.4f",
        fallback_result.get("model_name"),
        float(fallback_result.get("confidence", 0.0)),
    )
    return fallback_result
