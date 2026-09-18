import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "gotcha-extractor-model")

label2id = {'O': 0, 'B-RISK': 1, 'I-RISK': 2}
id2label = {0: 'O', 1: 'B-RISK', 2: 'I-RISK'}

AVAILABLE_MODELS = ["electra-small", "tinybert", "bert-mini", "bert-tiny"]

MODEL_FALLBACK_MAP = {
    "electra-small": "google/electra-small-discriminator",
    "tinybert": "huawei-noah/TinyBERT_General_4L_312D",
    "bert-tiny": "prajjwal1/bert-tiny",
    "bert-mini": "prajjwal1/bert-mini"
}

MODEL_META = {
    "electra-small": {
        "name": "ELECTRA-Small (Fine-tuned)",
        "params": "13.5M",
        "size": "51.5 MB",
        "desc": "Top overall accuracy & F1 score (79.7%). Excellent balance of 13.5M parameters and 88.0% recall on high-risk clauses.",
        "badge_class": "badge-electra",
        "best_f1": "79.7%"
    },
    "tinybert": {
        "name": "TinyBERT (Fine-tuned)",
        "params": "14.3M",
        "size": "54.4 MB",
        "desc": "Distilled BERT architecture achieving 77.6% F1 and 87.9% recall with balanced inference latency.",
        "badge_class": "badge-tinybert",
        "best_f1": "77.6%"
    },
    "bert-tiny": {
        "name": "BERT-Tiny (Fine-tuned)",
        "params": "4.4M",
        "size": "16.7 MB",
        "desc": "Ultra-compact 2-layer transformer with 75.0% F1 and 91.4% peak recall across legal boundaries.",
        "badge_class": "badge-tiny",
        "best_f1": "75.0%"
    },
    "bert-mini": {
        "name": "BERT-Mini (Fine-tuned)",
        "params": "11.1M",
        "size": "42.4 MB",
        "desc": "Lightweight 4-layer BERT variant delivering 68.4% F1 and 88.9% recall on contract sequences.",
        "badge_class": "badge-mini",
        "best_f1": "68.4%"
    }
}

KEYWORDS_HIGH = [
    r"arbitrat", r"class\s+action", r"waiver", r"dispute",
    r"reserve\s+the\s+right\s+to", r"modify", r"revise", r"update", r"without\s+notice",
    r"sell", r"market", r"advertis", r"third\s+part",
    r"cannot\s+(ensure|warrant|guarantee)", r"no\s+warranty", r"indemni"
]

WAIVER_HOSTILE_INDICATORS = [
    r"\bwaiv", r"\barbitrat", r"\brelinquish", r"\bgive\s+up",
    r"\bsurrender", r"\bbinding", r"\bdispute", r"\bclass\s+action",
    r"\bnot\s+permitted\b", r"\bprohibit", r"\bforfeit"
]

BOILERPLATE_PATTERNS = [
    r"this\s+privacy\s+policy\s+(\([^)]+\)\s+)?describes\s+the\s+practices",
    r"this\s+privacy\s+policy\s+applies\s+only\s+to",
    r"summary\s+the\s+notifications\s+provided\s+by\s+this\s+privacy\s+policy\s+include",
    r"^[a-zA-Z\s]+is\s+data\s+that\s+can\s+be\s+used\s+to\s+identify",
    r"^[a-zA-Z\s]+\s+means\s+any\s+information",
    r"legal\s+grounds\s+for\s+processing\s+personal\s+data",
    r"we\s+restrict\s+access\s+to\s+personal\s+information\s+collected.*to\s+our\s+employees",
    r"please\s+note\s+that\s+we\s+have\s+a\s+separate\s+privacy\s+disclosure\s+statement\s+to\s+address\s+our\s+protocols.*located\s+here",
    r"children\s+under\s+13", r"younger\s+than\s+13", r"receive\s+parental\s+consent",
    r"privacy\s+policy\s+effective\s+date"
]

KEYWORDS_PRO_USER = [
    r"you\s+(may|can)\s+(access|correct|request\s+deletion|delete|port|object)",
    r"request\s+(that\s+we\s+stop|the\s+deletion|deletion|access\s+to)",
    r"freely\s+visit\s+our\s+(website|platform)\s+anonymously",
    r"without\s+being\s+required\s+to\s+provide\s+us\s+with\s+any\s+personal\s+information",
    r"rights\s+related\s+to\s+the\s+european\s+union",
    r"rights\s+related\s+to\s+gdpr",
    r"your\s+right\s+to\s+(access|delete|deletion|rectify|rectification|restrict)",
    r"opt[- ]out\s+of\s+receiving\s+(marketing|promotional|newsletter)",
    r"under\s+(the\s+)?(general\s+data\s+protection\s+regulation|gdpr|ccpa)",
    r"right\s+to\s+request\s+that\s+we\s+(disclose|delete)",
    r"right\s+to\s+know\s+what\s+personal\s+information",
]

COLOR_MAP = {
    "HIGH RISK": "#be123c",
    "MEDIUM RISK": "#b45309",
    "LOW RISK": "#64748b"
}
