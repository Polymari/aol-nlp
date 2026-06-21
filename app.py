import os
import re
import torch
import ftfy
import gradio as gr
from transformers import AutoTokenizer, AutoModelForTokenClassification, logging as tf_logging
tf_logging.set_verbosity_error()
tf_logging.disable_progress_bar()
import nltk
from nltk.tokenize import PunktSentenceTokenizer

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

MODEL_CACHE = {}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

label2id = {'O': 0, 'B-RISK': 1, 'I-RISK': 2}
id2label = {0: 'O', 1: 'B-RISK', 2: 'I-RISK'}

AVAILABLE_MODELS = ["electra-small", "tinybert", "bert-tiny", "bert-mini"]

def load_model(model_name):
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    local_path = os.path.join(BASE_DIR, "gotcha-extractor-model", model_name)
    if not os.path.exists(os.path.join(local_path, "config.json")):
        local_path = os.path.join(BASE_DIR, "models", model_name)
        
    has_local = os.path.exists(local_path) and os.path.exists(os.path.join(local_path, "config.json"))
    
    if has_local:
        model_path = local_path
        print(f"Loading local model from: {model_path}")
    else:
        fallback_map = {
            "electra-small": "google/electra-small-discriminator",
            "tinybert": "huawei-noah/TinyBERT_General_4L_312D",
            "bert-tiny": "prajjwal1/bert-tiny",
            "bert-mini": "prajjwal1/bert-mini"
        }
        model_path = fallback_map.get(model_name, "google/electra-small-discriminator")
        print(f"Local model not found. Falling back to HF Hub: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    MODEL_CACHE[model_name] = (model, tokenizer)
    return model, tokenizer

KEYWORDS_HIGH = [
    r"arbitrat", r"class\s+action", r"waiver", r"dispute",
    r"reserve\s+the\s+right\s+to", r"modify", r"revise", r"update", r"without\s+notice",
    r"sell", r"market", r"advertis", r"third\s+part",
    r"cannot\s+(ensure|warrant|guarantee)", r"no\s+warranty", r"indemni"
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
    r"you\s+may\s+(access|correct|request\s+deletion|delete|port|object)",
    r"request\s+that\s+we\s+stop\s+(any\s+)?processing",
    r"freely\s+visit\s+our\s+(website|platform)\s+anonymously",
    r"without\s+being\s+required\s+to\s+provide\s+us\s+with\s+any\s+personal\s+information",
    r"rights\s+related\s+to\s+the\s+european\s+union",
    r"rights\s+related\s+to\s+gdpr",
    r"your\s+right\s+to\s+(access|delete|rectify|restrict)",
    r"opt[- ]out\s+of\s+receiving\s+(marketing|promotional|newsletter)",
    r"under\s+the\s+general\s+data\s+protection\s+regulation",
    r"right\s+to\s+request\s+that\s+we\s+disclose",
    r"right\s+to\s+know\s+what\s+personal\s+information",
]

def check_pro_user_override(sentence):
    sentence_lower = sentence.strip().lower()

    for pattern in KEYWORDS_PRO_USER:
        if re.search(pattern, sentence_lower):
            return True

    if re.search(r"\b(right(s)?\s+to|you\s+have\s+the\s+right\s+to)\s+.*\b(access|correct|delete|erase|rectify|update|portability|restrict)\b", sentence_lower):
        return True

    if re.search(r"\b(visit|browse)\b.*\banonymously\b", sentence_lower) and not re.search(r"\b(cannot|unable|restrict)\b", sentence_lower):
        return True

    if re.search(r"\brights\s+related\s+to\b.*\b(gdpr|ccpa|california\s+consumer|protection\s+regulation)\b", sentence_lower):
        return True

    return False

def clean_boilerplate_header(sentence):
    sentence_clean = sentence.strip()
    sentence_lower = sentence_clean.lower()

    if re.match(r"^[A-Z\s\d/_:,\'\"]{3,50}$", sentence_clean):
        return True

    for pattern in BOILERPLATE_PATTERNS:
        if re.search(pattern, sentence_lower):
            return True

    return False

def determine_risk_level(sentence, risk_tokens, has_high_keyword):
    if not risk_tokens:
        return None

    probs = [t["prob"] for t in risk_tokens]
    max_prob = max(probs)

    if max_prob >= 0.80 or (has_high_keyword and max_prob >= 0.68):
        return "HIGH RISK"
    elif has_high_keyword or max_prob >= 0.62:
        return "MEDIUM RISK"
    else:
        return "LOW RISK"

def clean_text_pipeline(raw_text):
    text = ftfy.fix_text(raw_text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def classify_text(raw_text, model_name="electra-small", min_risk_tokens=3):
    if not raw_text or not raw_text.strip():
        return []

    cleaned_text = clean_text_pipeline(raw_text)
    model, tokenizer = load_model(model_name)
    device = model.device

    sentence_spans = list(PunktSentenceTokenizer().span_tokenize(cleaned_text))

    highlighted_data = []
    prev_end = 0

    for start_idx, end_idx in sentence_spans:
        if start_idx > prev_end:
            highlighted_data.append((cleaned_text[prev_end:start_idx], None))

        sentence = cleaned_text[start_idx:end_idx]
        if not sentence.strip():
            highlighted_data.append((sentence, None))
            prev_end = end_idx
            continue

        if clean_boilerplate_header(sentence):
            highlighted_data.append((sentence, None))
            prev_end = end_idx
            continue

        if check_pro_user_override(sentence):
            highlighted_data.append((sentence, None))
            prev_end = end_idx
            continue

        inputs = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)

        risk_tokens = []
        for t_idx, pred in enumerate(predictions):
            label = id2label[pred.item()]
            token_str = tokens[t_idx]
            if token_str in ('[CLS]', '[SEP]', '[PAD]'):
                continue
            prob = probs[t_idx][pred.item()].item()
            if label in ('B-RISK', 'I-RISK'):
                risk_tokens.append({"token": token_str, "prob": prob})

        if len(risk_tokens) >= min_risk_tokens:
            max_prob = max(t["prob"] for t in risk_tokens)

            has_high_keyword = False
            sentence_lower = sentence.lower()
            for pattern in KEYWORDS_HIGH:
                if re.search(pattern, sentence_lower):
                    has_high_keyword = True
                    break

            keep = False
            if has_high_keyword:
                if max_prob >= 0.55:
                    keep = True
            else:
                if max_prob >= 0.70:
                    keep = True

            if keep:
                level = determine_risk_level(sentence, risk_tokens, has_high_keyword)
                highlighted_data.append((sentence, level))
            else:
                highlighted_data.append((sentence, None))
        else:
            highlighted_data.append((sentence, None))

        prev_end = end_idx

    if prev_end < len(cleaned_text):
        highlighted_data.append((cleaned_text[prev_end:], None))

    return highlighted_data

demo = gr.Interface(
    fn=classify_text,
    inputs=[
        gr.Textbox(
            lines=6,
            label="Terms of Service Text",
            placeholder="Paste legal agreement clauses, privacy policy paragraphs, or user agreements here..."
        ),
        gr.Dropdown(
            choices=AVAILABLE_MODELS,
            value="electra-small",
            label="Select Extraction Model"
        )
    ],
    outputs=gr.HighlightedText(
        label="Analysis Results (Highlighted Clauses)",
        combine_adjacent=False,
        color_map={
            "HIGH RISK": "#ef4444",
            "MEDIUM RISK": "#f97316",
            "LOW RISK": "#eab308"
        }
    ),
    title="ToS 'Gotcha' Clause Extractor",
    description=(
        "Analyze Terms of Service and Privacy Policies instantly. "
        "This app supports switching between multiple fine-tuned models "
        "to detect user-unfavorable clauses such as forced arbitration, "
        "class action waivers, and data-selling agreements."
    ),
    examples=[
        ["Welcome to the platform. By continuing, you agree to forced arbitration in the event of a dispute. We also reserve the right to sell your location data and usage habits to unverified third parties.", "electra-small"],
        ["You agree to defend, indemnify and hold harmless the Company and its officers from and against any claims, liabilities, damages, losses, and expenses.", "electra-small"],
        ["We may modify these terms at any time without notice. Your continued use of the service constitutes acceptance of the new terms.", "electra-small"]
    ],
    cache_examples=False,
    theme=gr.themes.Soft()
)


if __name__ == "__main__":
    demo.launch()
