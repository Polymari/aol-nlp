import pytest
from gotcha.preprocessor import clean_text_pipeline, segment_sentences


def test_clean_text_mojibake_fix():
    # Broken UTF-8 quotation marks and apostrophes
    raw = "The companyâ€™s services are provided â€œas isâ€."
    cleaned = clean_text_pipeline(raw)
    assert "company’s" in cleaned or "company's" in cleaned
    assert "“as is”" in cleaned or '"as is"' in cleaned


def test_clean_text_hard_wraps():
    raw = "This is a sentence\nsplit across lines.\n\nThis is a new paragraph."
    cleaned = clean_text_pipeline(raw)
    assert "This is a sentence split across lines." in cleaned
    assert "\n\n" in cleaned or "This is a new paragraph." in cleaned


def test_clean_text_whitespace_normalization():
    raw = "Multiple    spaces \t and tabs   normalized."
    cleaned = clean_text_pipeline(raw)
    assert cleaned == "Multiple spaces and tabs normalized."


def test_segment_sentences():
    text = "First sentence here. Second sentence here. Third sentence."
    spans = segment_sentences(text)
    assert len(spans) == 3
    s0 = text[spans[0][0]:spans[0][1]]
    assert s0 == "First sentence here."
