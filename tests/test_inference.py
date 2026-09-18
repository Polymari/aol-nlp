import pytest
import pandas as pd
from gotcha.inference import load_model, classify_text, compare_models


def test_load_model():
    model, tokenizer = load_model("bert-tiny")
    assert model is not None
    assert tokenizer is not None
    assert model.training is False


def test_classify_text_empty():
    assert classify_text("") == []
    assert classify_text("   ") == []


def test_classify_text_gotcha_detected():
    sample = "Welcome to the site. By continuing, you agree to forced arbitration in the event of a dispute."
    res = classify_text(sample, model_name="bert-tiny", min_risk_tokens=1)
    assert len(res) >= 2
    # At least one segment should be evaluated
    labels = [label for _, label in res if label is not None]
    assert len(labels) >= 1


def test_compare_models_single_pass():
    sample = "You agree to forced arbitration and class action waiver."
    e, tb, bm, bt, df = compare_models(sample, min_tokens=1)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert list(df["Model"]) == [
        "ELECTRA-Small (Fine-tuned)",
        "TinyBERT (Fine-tuned)",
        "BERT-Mini (Fine-tuned)",
        "BERT-Tiny (Fine-tuned)"
    ]
