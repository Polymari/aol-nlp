import pytest
from gotcha.heuristics import (
    check_pro_user_override,
    clean_boilerplate_header,
    has_high_risk_keyword,
    determine_risk_level
)


def test_pro_user_genuine_rights():
    """Verify that pro-user rights clauses are correctly flagged for override."""
    s1 = "You have the right to request deletion of your personal data under GDPR."
    s2 = "You may freely visit our website anonymously without providing any personal information."
    s3 = "You may opt-out of receiving promotional newsletter communications at any time."
    assert check_pro_user_override(s1) is True
    assert check_pro_user_override(s2) is True
    assert check_pro_user_override(s3) is True


def test_pro_user_waiver_hostile_not_overridden():
    """Crucial regression test: Sentences that waive rights or mandate arbitration
    must NEVER be suppressed as pro-user."""
    s_waiver = "You waive your right to access court proceedings and agree to binding arbitration."
    s_class = "You relinquish your right to participate in a class action lawsuit."
    s_dispute = "All disputes regarding your right to access data shall be resolved via binding arbitration."
    assert check_pro_user_override(s_waiver) is False
    assert check_pro_user_override(s_class) is False
    assert check_pro_user_override(s_dispute) is False


def test_boilerplate_headers():
    """Verify standard legal headings are recognized as boilerplate."""
    assert clean_boilerplate_header("PRIVACY POLICY EFFECTIVE DATE") is True
    assert clean_boilerplate_header("This Privacy Policy describes the practices of our service.") is True


def test_boilerplate_preserves_high_risk_clauses():
    """Crucial regression test: All-caps short clauses that contain high-risk terms
    must NOT be dropped as boilerplate."""
    assert clean_boilerplate_header("YOU AGREE TO BINDING ARBITRATION AND WAIVER") is False
    assert clean_boilerplate_header("DISPUTE RESOLUTION AND CLASS ACTION WAIVER") is False


def test_pro_user_permission_with_filler():
    """Regression: rights phrased with filler between the verb and the right
    ('you may request a copy of', 'you may ask us to correct') must still be
    suppressed, otherwise clean privacy policies raise false alarms."""
    s1 = "You may request a copy of the personal data we hold about you."
    s2 = "You may ask us to correct or delete it at any time."
    assert check_pro_user_override(s1) is True
    assert check_pro_user_override(s2) is True


def test_boilerplate_effective_date_and_assent():
    """Standard assent boilerplate is not a risky clause."""
    assert clean_boilerplate_header("These terms take effect on the date shown above.") is True
    assert clean_boilerplate_header("If you do not agree with these terms, do not use the service.") is True


def test_high_risk_keywords():
    assert has_high_risk_keyword("We reserve the right to modify these terms.") is True
    assert has_high_risk_keyword("Any dispute must be settled through arbitration.") is True
    assert has_high_risk_keyword("We sell your location data to third parties.") is True
    assert has_high_risk_keyword("Welcome to our online service.") is False


def test_determine_risk_level():
    high_tokens = [{"token": "arbitrat", "prob": 0.85}]
    assert determine_risk_level("text", high_tokens, has_high_kw=True) == "HIGH RISK"

    med_tokens = [{"token": "modify", "prob": 0.58}]
    assert determine_risk_level("text", med_tokens, has_high_kw=True) == "MEDIUM RISK"

    low_tokens = [{"token": "term", "prob": 0.52}]
    assert determine_risk_level("text", low_tokens, has_high_kw=False) == "LOW RISK"


def test_metadata_and_date_headers_suppressed():
    """Verify that document titles, date metadata, and page numbers are cleanly recognized as boilerplate."""
    assert clean_boilerplate_header("TERMS OF SERVICE") is True
    assert clean_boilerplate_header("Last Updated: January 15, 2024") is True
    assert clean_boilerplate_header("Effective Date: 2024-01-01") is True
    assert clean_boilerplate_header("Page 1 of 12") is True
    assert clean_boilerplate_header("— 4 —") is True
    assert clean_boilerplate_header("1. Acceptance of Terms") is True
    assert clean_boilerplate_header("Section 4. Updates to Service") is True

