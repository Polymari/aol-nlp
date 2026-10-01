"""Gotcha Clause Extractor: NLP detection of unfair clauses in legal texts."""

from .constants import (
    AVAILABLE_MODELS,
    MODEL_META,
    COLOR_MAP,
    RISK_INK,
    label2id,
    id2label
)
from .preprocessor import clean_text_pipeline, segment_sentences
from .heuristics import (
    clean_boilerplate_header,
    check_pro_user_override,
    has_high_risk_keyword,
    determine_risk_level
)
from .inference import (
    load_model,
    classify_text,
    compare_models,
    get_inference_device
)

__all__ = [
    "AVAILABLE_MODELS",
    "MODEL_META",
    "COLOR_MAP",
    "RISK_INK",
    "label2id",
    "id2label",
    "clean_text_pipeline",
    "segment_sentences",
    "clean_boilerplate_header",
    "check_pro_user_override",
    "has_high_risk_keyword",
    "determine_risk_level",
    "load_model",
    "classify_text",
    "compare_models",
    "get_inference_device"
]
