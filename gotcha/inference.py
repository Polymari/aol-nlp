import os
import time
from typing import List, Tuple, Dict, Any, Optional
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, logging as tf_logging

from .constants import (
    MODEL_DIR,
    AVAILABLE_MODELS,
    MODEL_META,
    MODEL_FALLBACK_MAP,
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

tf_logging.set_verbosity_error()
tf_logging.disable_progress_bar()

MODEL_CACHE: Dict[str, Tuple[Any, Any]] = {}


def get_inference_device() -> torch.device:
    """Return the optimal device for inference."""
    if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_name: str):
    """Load and cache tokenizer and model for the specified model_name."""
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    local_path = os.path.join(MODEL_DIR, model_name)
    has_local = os.path.exists(local_path) and os.path.exists(os.path.join(local_path, "config.json"))

    if has_local:
        model_path = local_path
    else:
        model_path = MODEL_FALLBACK_MAP.get(model_name, "google/electra-small-discriminator")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    device = get_inference_device()
    model = model.to(device)
    model.eval()

    MODEL_CACHE[model_name] = (model, tokenizer)
    return model, tokenizer


def classify_text(
    raw_text: str,
    model_name: str = "electra-small",
    min_risk_tokens: int = 3,
    batch_size: int = 16
) -> List[Tuple[str, Optional[str]]]:
    """Segment, filter, and classify clauses in raw_text using batched neural inference.
    
    Returns:
        List of tuples: [(text_segment, risk_level_or_none), ...]
    """
    if not raw_text or not raw_text.strip():
        return []

    cleaned_text = clean_text_pipeline(raw_text)
    if not cleaned_text:
        return []

    model, tokenizer = load_model(model_name)
    device = model.device

    sentence_spans = segment_sentences(cleaned_text)
    if not sentence_spans:
        return [(cleaned_text, None)]

    # Collect segments to run through the neural model vs pre-filtered segments
    candidates_to_infer = []  # (index, sentence_str)
    sentence_records = []     # (start_idx, end_idx, sentence_str, pre_label)
    prev_end = 0

    for start_idx, end_idx in sentence_spans:
        if start_idx > prev_end:
            # Trailing/leading whitespace between sentences
            sentence_records.append((prev_end, start_idx, cleaned_text[prev_end:start_idx], None))

        sentence = cleaned_text[start_idx:end_idx]
        if not sentence.strip():
            sentence_records.append((start_idx, end_idx, sentence, None))
            prev_end = end_idx
            continue

        if clean_boilerplate_header(sentence) or check_pro_user_override(sentence):
            sentence_records.append((start_idx, end_idx, sentence, None))
            prev_end = end_idx
            continue

        # Candidate for neural forward pass
        candidate_idx = len(sentence_records)
        candidates_to_infer.append((candidate_idx, sentence))
        sentence_records.append((start_idx, end_idx, sentence, "PENDING"))
        prev_end = end_idx

    if prev_end < len(cleaned_text):
        sentence_records.append((prev_end, len(cleaned_text), cleaned_text[prev_end:], None))

    # Batched Neural Inference
    if candidates_to_infer:
        for b_start in range(0, len(candidates_to_infer), batch_size):
            b_chunk = candidates_to_infer[b_start : b_start + batch_size]
            b_indices = [item[0] for item in b_chunk]
            b_sentences = [item[1] for item in b_chunk]

            inputs = tokenizer(
                b_sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits  # shape: (batch_size, seq_len, num_labels)
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)

            for b_i, rec_idx in enumerate(b_indices):
                sentence = b_sentences[b_i]
                sentence_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][b_i])
                seq_preds = predictions[b_i]
                seq_probs = probs[b_i]

                risk_tokens = []
                for t_idx, pred in enumerate(seq_preds):
                    token_str = sentence_tokens[t_idx]
                    if token_str in ('[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>'):
                        continue
                    label = id2label[pred.item()]
                    prob = seq_probs[t_idx][pred.item()].item()
                    if label in ('B-RISK', 'I-RISK'):
                        risk_tokens.append({"token": token_str, "prob": prob})

                # Determine if threshold is met
                assigned_level = None
                if len(risk_tokens) >= min_risk_tokens:
                    max_prob = max(t["prob"] for t in risk_tokens)
                    has_high_kw = has_high_risk_keyword(sentence)

                    keep = False
                    if has_high_kw and max_prob >= 0.55:
                        keep = True
                    elif not has_high_kw and max_prob >= 0.70:
                        keep = True

                    if keep:
                        assigned_level = determine_risk_level(sentence, risk_tokens, has_high_kw)

                # Update the record with assigned level (or None)
                start_c, end_c, sent_c, _ = sentence_records[rec_idx]
                sentence_records[rec_idx] = (start_c, end_c, sent_c, assigned_level)

    highlighted_data = [(rec[2], rec[3]) for rec in sentence_records]
    return highlighted_data


def compare_models(
    text: str,
    min_tokens: int = 3,
    batch_size: int = 16
) -> Tuple[List[Any], List[Any], List[Any], List[Any], pd.DataFrame]:
    """Execute each available model exactly once and produce side-by-side results with metrics."""
    if not text or not text.strip():
        return [], [], [], [], pd.DataFrame()

    results_by_model: Dict[str, Tuple[List[Any], float, int]] = {}
    comparison_rows = []

    for m in AVAILABLE_MODELS:
        start_time = time.time()
        res = classify_text(text, model_name=m, min_risk_tokens=min_tokens, batch_size=batch_size)
        elapsed_ms = (time.time() - start_time) * 1000

        risky_count = sum(1 for _, label in res if label is not None)
        results_by_model[m] = (res, elapsed_ms, risky_count)

        meta = MODEL_META[m]
        comparison_rows.append({
            "Model": meta["name"],
            "Validation F1 (Best)": meta["best_f1"],
            "Parameters": meta["params"],
            "Disk Size": meta["size"],
            "Risks Detected": risky_count,
            "Latency (ms)": f"{elapsed_ms:.1f} ms"
        })

    df_compare = pd.DataFrame(comparison_rows)
    return (
        results_by_model["electra-small"][0],
        results_by_model["tinybert"][0],
        results_by_model["bert-mini"][0],
        results_by_model["bert-tiny"][0],
        df_compare
    )
