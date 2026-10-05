
import re
import ftfy
import nltk
from nltk.tokenize import PunktSentenceTokenizer

from .constants import RE_LEGAL_PREFIX_STR

# Ensure required NLTK tokenizer resources are downloaded
for pkg in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{pkg}')
    except LookupError:
        nltk.download(pkg, quiet=True)

RE_LEGAL_PREFIX = re.compile(RE_LEGAL_PREFIX_STR, re.IGNORECASE)
RE_DEHYPHEN = re.compile(r'([a-zA-Z]{2,})-\s*\n\s*([a-zA-Z]{2,})')
RE_PAGE_STRIP = re.compile(r'(?im)^\s*(page\s+\d+(\s+of\s+\d+)?|[\-—]\s*\d+\s*[\-—]|\[\s*\d+\s*\])\s*$\n?')


def clean_text_pipeline(raw_text: str) -> str:
    """Deterministic cleaning pipeline for legal documents and PDF copies:
    1. Repairs encoding artifacts and Mojibake (via ftfy).
    2. Normalizes line endings (CRLF and old CR -> LF) and removes form feeds (\x0c).
    3. De-hyphenates words fractured across margin line wraps (e.g. 'arbi-\\n tration' -> 'arbitration').
    4. Strips standalone page numbers and PDF pagination markers.
    5. Isolates section headings and date metadata with double newlines.
    6. Normalizes soft line wraps into single spaces while preserving paragraph breaks.
    7. Normalizes consecutive whitespace and tabs.
    """
    if not raw_text:
        return ""
    text = ftfy.fix_text(raw_text)
    text = text.replace('\r\n', '\n').replace('\r', '\n').replace('\x0c', '\n')

    # De-hyphenate broken words across line wraps
    text = RE_DEHYPHEN.sub(r'\1\2', text)

    # Strip standalone page markers
    text = RE_PAGE_STRIP.sub('', text)

    # Ensure section headers and date metadata are separated by double newline
    text = re.sub(r'\n+((?:last\s+updated|effective\s+date|date\s+of\s+revision)[\s:])', r'\n\n\1', text, flags=re.IGNORECASE)
    text = re.sub(r'\n+(\d+(\.\d+)*\.?\s+[A-Z])', r'\n\n\1', text)
    text = re.sub(r'\n+((?:Section|Article|Clause)\s+[A-Z0-9]+)', r'\n\n\1', text, flags=re.IGNORECASE)

    # Collapse soft line wraps within paragraphs into single spaces
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    return text.strip()


def segment_sentences(cleaned_text: str):
    """Tokenize cleaned legal text into character spans: [(start_char, end_char), ...].
    Respects paragraph breaks and re-stitches orphaned legal numbering prefixes (e.g., '12.', 'Section 4.').
    """
    if not cleaned_text or not cleaned_text.strip():
        return []

    tokenizer = PunktSentenceTokenizer()
    para_matches = list(re.finditer(r'[^\n]+(?:\n[^\n]+)*', cleaned_text))
    spans = []

    for pm in para_matches:
        p_text = pm.group(0)
        p_offset = pm.start()

        raw_spans = list(tokenizer.span_tokenize(p_text))
        if not raw_spans:
            continue

        merged = []
        i = 0
        while i < len(raw_spans):
            s_start, s_end = raw_spans[i]
            s_text = p_text[s_start:s_end].strip()

            # If this span is just a legal prefix and there is a subsequent span, merge forward
            if RE_LEGAL_PREFIX.match(s_text) and i + 1 < len(raw_spans):
                next_start, next_end = raw_spans[i + 1]
                merged.append((s_start, next_end))
                i += 2
            else:
                merged.append((s_start, s_end))
                i += 1

        for s, e in merged:
            spans.append((p_offset + s, p_offset + e))

    return spans
