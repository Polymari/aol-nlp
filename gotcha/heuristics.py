import re
from typing import List, Dict, Any, Optional
from .constants import (
    KEYWORDS_HIGH,
    BOILERPLATE_PATTERNS,
    KEYWORDS_PRO_USER,
    WAIVER_HOSTILE_INDICATORS,
    RE_DATE_METADATA_STR,
    RE_DOC_TITLE_STR,
    RE_PAGE_MARKERS_STR,
    RE_LEGAL_PREFIX_STR
)

# Pre-compile regexes for high performance
RE_KEYWORDS_HIGH = [re.compile(p, re.IGNORECASE) for p in KEYWORDS_HIGH]
RE_BOILERPLATE = [re.compile(p, re.IGNORECASE) for p in BOILERPLATE_PATTERNS]
RE_PRO_USER = [re.compile(p, re.IGNORECASE) for p in KEYWORDS_PRO_USER]
RE_WAIVER_HOSTILE = [re.compile(p, re.IGNORECASE) for p in WAIVER_HOSTILE_INDICATORS]

RE_DATE_METADATA = re.compile(RE_DATE_METADATA_STR, re.IGNORECASE)
RE_DOC_TITLE = re.compile(RE_DOC_TITLE_STR, re.IGNORECASE)
RE_PAGE_MARKERS = re.compile(RE_PAGE_MARKERS_STR, re.IGNORECASE)
RE_LEGAL_PREFIX = re.compile(RE_LEGAL_PREFIX_STR, re.IGNORECASE)

RE_ALL_CAPS_HEADER = re.compile(r"^[A-Z\s\d/_:,\'\"]{3,50}$")
RE_USER_RIGHTS_PATTERN = re.compile(
    r"\b(right(s)?\s+to|you\s+have\s+the\s+right\s+to)\s+.*?\b(access|correct|correction|delete|deletion|erase|erasure|rectify|rectification|update|portability|restrict|restriction)\b",
    re.IGNORECASE
)
RE_ANONYMOUS_BROWSING = re.compile(r"\b(visit|browse)\b.*\banonymously\b", re.IGNORECASE)
RE_RESTRICT_TERMS = re.compile(r"\b(cannot|unable|restrict)\b", re.IGNORECASE)
RE_GDPR_CCPA = re.compile(r"\b(rights\s+related\s+to|under\s+(the\s+)?)\b.*?\b(gdpr|ccpa|california\s+consumer|protection\s+regulation)\b", re.IGNORECASE)


def has_high_risk_keyword(sentence: str) -> bool:
    """Return True if sentence contains any of the high-risk keywords."""
    return any(p.search(sentence) for p in RE_KEYWORDS_HIGH)


def check_pro_user_override(sentence: str) -> bool:
    """Identify if a sentence grants user rights and should be suppressed as neutral.
    
    Safety guard: Sentences containing waivers, forced arbitration, or hostile keywords
    must NEVER be suppressed as pro-user.
    """
    sentence_clean = sentence.strip()
    if not sentence_clean:
        return False

    # Safety Guard: If hostile or waiver terms exist, never suppress
    for p in RE_WAIVER_HOSTILE:
        if p.search(sentence_clean):
            return False

    # Check explicit pro-user patterns
    for pattern in RE_PRO_USER:
        if pattern.search(sentence_clean):
            return True

    if RE_USER_RIGHTS_PATTERN.search(sentence_clean):
        return True

    if RE_ANONYMOUS_BROWSING.search(sentence_clean) and not RE_RESTRICT_TERMS.search(sentence_clean):
        return True

    if RE_GDPR_CCPA.search(sentence_clean):
        return True

    return False


def clean_boilerplate_header(sentence: str) -> bool:
    """Identify if a sentence is non-informative boilerplate, metadata, or a title header.
    
    Safety guard: If a short heading contains high-risk terms (e.g. ARBITRATION, DISPUTE),
    do NOT discard it.
    """
    sentence_clean = sentence.strip()
    if not sentence_clean:
        return True

    # 1. Structural document metadata, titles, page markers, and prefix fragments
    if (
        RE_DOC_TITLE.match(sentence_clean)
        or RE_DATE_METADATA.match(sentence_clean)
        or RE_PAGE_MARKERS.match(sentence_clean)
        or RE_LEGAL_PREFIX.match(sentence_clean)
    ):
        return True

    # 2. Safety Guard: If high risk keyword is present, keep it even if uppercase
    if has_high_risk_keyword(sentence_clean):
        return False

    if RE_ALL_CAPS_HEADER.match(sentence_clean):
        return True

    for pattern in RE_BOILERPLATE:
        if pattern.search(sentence_clean):
            return True

    return False


def determine_risk_level(sentence: str, risk_tokens: List[Dict[str, Any]], has_high_kw: bool) -> Optional[str]:
    """Map token confidence probabilities and keyword presence to ordinal risk levels:
    - HIGH RISK: max_prob >= 0.80 or (has_high_keyword and max_prob >= 0.68)
    - MEDIUM RISK: (has_high_keyword and max_prob >= 0.55) or max_prob >= 0.62
    - LOW RISK: fallback
    """
    if not risk_tokens:
        return None

    max_prob = max(t["prob"] for t in risk_tokens)

    if max_prob >= 0.80 or (has_high_kw and max_prob >= 0.68):
        return "HIGH RISK"
    elif has_high_kw or max_prob >= 0.62:
        return "MEDIUM RISK"
    else:
        return "LOW RISK"
