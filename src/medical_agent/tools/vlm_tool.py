from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Tuple

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipForQuestionAnswering, BlipProcessor

from medical_agent.logging_utils import get_logger


logger = get_logger(__name__)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _load_caption_stack(model_name: str) -> Tuple[BlipProcessor, BlipForConditionalGeneration]:
    logger.info("Loading BLIP caption model | model=%s", model_name)
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    model.to(_device())
    model.eval()
    return processor, model


@lru_cache(maxsize=1)
def _load_vqa_stack(model_name: str) -> Tuple[BlipProcessor, BlipForQuestionAnswering]:
    logger.info("Loading BLIP VQA model | model=%s", model_name)
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForQuestionAnswering.from_pretrained(model_name)
    model.to(_device())
    model.eval()
    return processor, model


def analyze_image_with_vlm(
    image_path: str,
    user_question: str,
    caption_model_name: str,
    vqa_model_name: str,
) -> Dict[str, Any]:
    logger.info(
        "VLM analysis started | image_path=%s | question=%s | device=%s",
        image_path,
        user_question,
        _device(),
    )
    image = Image.open(image_path).convert("RGB")
    device = _device()

    caption_processor, caption_model = _load_caption_stack(caption_model_name)
    caption_inputs = caption_processor(images=image, return_tensors="pt").to(device)
    caption_ids = caption_model.generate(**caption_inputs, max_new_tokens=60)
    caption = caption_processor.decode(caption_ids[0], skip_special_tokens=True)
    logger.info("VLM caption generated | caption=%s", caption)

    answer = ""
    if user_question and user_question.strip():
        vqa_processor, vqa_model = _load_vqa_stack(vqa_model_name)
        vqa_inputs = vqa_processor(image, user_question, return_tensors="pt").to(device)
        answer_ids = vqa_model.generate(**vqa_inputs, max_new_tokens=40)
        answer = vqa_processor.decode(answer_ids[0], skip_special_tokens=True)
        logger.info("VLM answer generated | answer=%s", answer)

    summary = f"Image caption: {caption}"
    if answer:
        summary += f" | Question answer: {answer}"

    logger.info("VLM analysis completed | summary=%s", summary)

    return {
        "summary": summary,
        "caption": caption,
        "answer": answer,
        "tool": "vlm",
    }
