import re
import ftfy
import nltk
from nltk.tokenize import PunktSentenceTokenizer

# Ensure required NLTK tokenizer resources are downloaded
for pkg in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{pkg}')
    except LookupError:
        nltk.download(pkg, quiet=True)


def clean_text_pipeline(raw_text: str) -> str:
    """Deterministic cleaning pipeline for legal documents:
    1. Repairs encoding artifacts and Mojibake (via ftfy).
    2. Normalizes hard line wraps into single spaces while preserving paragraph breaks.
    3. Normalizes consecutive whitespace and tabs.
    """
    if not raw_text:
        return ""
    text = ftfy.fix_text(raw_text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def segment_sentences(cleaned_text: str):
    """Tokenize cleaned legal text into character spans: [(start_char, end_char), ...]."""
    if not cleaned_text or not cleaned_text.strip():
        return []
    tokenizer = PunktSentenceTokenizer()
    return list(tokenizer.span_tokenize(cleaned_text))
