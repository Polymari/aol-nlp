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


def test_clean_text_dehyphenation():
    raw = "You agree to arbi-\n tration and indemni-\n  fication of claims."
    cleaned = clean_text_pipeline(raw)
    assert "arbitration" in cleaned
    assert "indemnification" in cleaned
    assert "arbi-" not in cleaned


def test_clean_text_page_markers():
    raw = "Terms of Service\nPage 1 of 12\n\n1. Introduction\nWelcome.\nPage 2 of 12\n\n2. Agreement\n— 3 —\nDone."
    cleaned = clean_text_pipeline(raw)
    assert "Page 1 of 12" not in cleaned
    assert "Page 2 of 12" not in cleaned
    assert "— 3 —" not in cleaned


def test_segment_sentences_legal_prefix_merge():
    text = "12. Dispute Resolution. All claims shall be settled by arbitration."
    spans = segment_sentences(text)
    assert len(spans) == 2
    s0 = text[spans[0][0]:spans[0][1]]
    assert s0 == "12. Dispute Resolution."
    s1 = text[spans[1][0]:spans[1][1]]
    assert s1 == "All claims shall be settled by arbitration."


def test_segment_sentences_section_prefix():
    text = "Section 4. Updates to Service. We reserve the right to modify these terms."
    spans = segment_sentences(text)
    assert len(spans) == 2
    assert text[spans[0][0]:spans[0][1]] == "Section 4. Updates to Service."

