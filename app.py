import os
import re
import json as json_lib
import torch
import torch.nn as nn
import ftfy
import gradio as gr
from transformers import AutoTokenizer, AutoModelForTokenClassification, logging as tf_logging
tf_logging.set_verbosity_error()
tf_logging.disable_progress_bar()
import nltk
from nltk.tokenize import PunktSentenceTokenizer

# Ensure NLTK model is downloaded
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# Global cache variables for the loaded models
current_model = None
current_tokenizer = None
current_model_name = None

# Fallback mappings for base pre-trained models
fallback_map = {
    "electra-small": "google/electra-small-discriminator",
    "albert-base": "albert-base-v2",
    "tinybert": "huawei-noah/TinyBERT_General_4L_312D",
    "legal-bert": "nlpaueb/legal-bert-base-uncased",
    "bilstm-crf": None  # No HF fallback; must be trained locally
}

label2id = {'O': 0, 'B-RISK': 1, 'I-RISK': 2}

# ---------- BiLSTM-CRF model definition (must match train_gotcha.ipynb Cell 6) ----------
from torchcrf import CRF as TorchCRF

class DummyConfig:
    def __init__(self, num_labels, id2label, label2id):
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, num_tags, embedding_dim=128, hidden_dim=256):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        self.crf = TorchCRF(num_tags, batch_first=True)
        self.config = DummyConfig(num_tags, id2label, label2id)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        embeds = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embeds)
        emissions = self.hidden2tag(lstm_out)
        if attention_mask is not None:
            mask = attention_mask.to(torch.bool)
        else:
            mask = torch.ones(input_ids.shape, dtype=torch.bool, device=input_ids.device)
        if labels is not None:
            clean_labels = labels.clone()
            clean_labels[clean_labels == -100] = 0
            loss = -self.crf(emissions, clean_labels, mask=mask, reduction='token_mean')
            return {'loss': loss, 'logits': emissions}
        else:
            return {'logits': emissions}

    @classmethod
    def from_pretrained(cls, load_directory, **kwargs):
        with open(os.path.join(load_directory, "config.json"), "r") as f:
            config = json_lib.load(f)
        model = cls(
            vocab_size=config["vocab_size"],
            num_tags=config["num_labels"],
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"]
        )
        weights_path = os.path.join(load_directory, "pytorch_model.bin")
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        return model

def load_model_by_choice(choice):
    global current_model, current_tokenizer, current_model_name
    
    if current_model_name == choice and current_model is not None:
        return current_model, current_tokenizer
        
    local_path = f"./gotcha-extractor-model/{choice}-optuna-search"
    if not os.path.exists(local_path) or not os.path.exists(os.path.join(local_path, "config.json")):
        local_path = f"./gotcha-extractor-model/{choice}"
    has_local = os.path.exists(local_path) and os.path.exists(os.path.join(local_path, "config.json"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if choice == "bilstm-crf":
        if has_local:
            tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")
            model = BiLSTM_CRF.from_pretrained(local_path).to(device)
            model.eval()
            print(f"Loaded BiLSTM-CRF from: {local_path}")
        else:
            raise FileNotFoundError(
                f"BiLSTM-CRF weights not found at {local_path}. "
                "Please train the model first using train_gotcha.ipynb Cell 10."
            )
    else:
        if has_local:
            model_path = local_path
            print(f"Loading locally trained model from: {model_path}")
        else:
            model_path = fallback_map[choice]
            print(f"Local trained model not found. Falling back to base model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        ).to(device)
        model.eval()
    
    current_model = model
    current_tokenizer = tokenizer
    current_model_name = choice
    return model, tokenizer

# Mappings
id2label = {0: 'O', 1: 'B-RISK', 2: 'I-RISK'}

# Critical keywords for risk level assignment and conditional highlighting
KEYWORDS_HIGH = [
    r"arbitrat", r"class\s+action", r"waiver", r"dispute",
    r"reserve\s+the\s+right\s+to", r"modify", r"revise", r"update", r"without\s+notice",
    r"sell", r"market", r"advertis", r"third\s+part",
    r"cannot\s+(ensure|warrant|guarantee)", r"no\s+warranty", r"indemni"
]

# Boilerplate patterns to filter out standard legal declarations
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

# Pro-user keywords/phrases to suppress false positive risk flags (post-processing)
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
    sentence_clean = sentence.strip()
    sentence_lower = sentence_clean.lower()
    
    # Check pro-user keyword suppression list
    for pattern in KEYWORDS_PRO_USER:
        if re.search(pattern, sentence_lower):
            return True
            
    # Check rule-based segment overrides (heuristic patterns)
    # Rule 1: Sentences explicitly defining rights to access, correct, delete, update or restrict processing of personal data
    if re.search(r"\b(right(s)?\s+to|you\s+have\s+the\s+right\s+to)\s+.*\b(access|correct|delete|erase|rectify|update|portability|restrict)\b", sentence_lower):
        return True
        
    # Rule 2: Sentences detailing options to browse/visit anonymously or opt out of optional tracking/cookies without restriction
    if re.search(r"\b(visit|browse)\b.*\banonymously\b", sentence_lower) and not re.search(r"\b(cannot|unable|restrict)\b", sentence_lower):
        return True
        
    # Rule 3: Explicit references to GDPR, CCPA, or other data protection rights that empower the user
    if re.search(r"\brights\s+related\s+to\b.*\b(gdpr|ccpa|california\s+consumer|protection\s+regulation)\b", sentence_lower):
        return True

    return False

def clean_boilerplate_header(sentence):
    sentence_clean = sentence.strip()
    sentence_lower = sentence_clean.lower()
    
    # 1. Skip short all-caps headers
    if re.match(r"^[A-Z\s\d/_:,\'\"]{3,50}$", sentence_clean):
        return True
        
    # 2. Skip matching boilerplate rules
    for pattern in BOILERPLATE_PATTERNS:
        if re.search(pattern, sentence_lower):
            return True
            
    return False

def determine_risk_level(sentence, risk_tokens, has_high_keyword):
    if not risk_tokens:
        return None
        
    probs = [t["prob"] for t in risk_tokens]
    max_prob = max(probs)
    
    # Logic:
    # 1. HIGH RISK (Red):
    #    - Very high confidence prediction (P >= 0.80)
    #    - OR contains a high-risk keyword and has high confidence (P >= 0.68)
    if max_prob >= 0.80 or (has_high_keyword and max_prob >= 0.68):
        return "HIGH RISK"
        
    # 2. MEDIUM RISK (Orange):
    #    - Contains high-risk keyword (moderate confidence P >= 0.55)
    #    - OR moderate confidence prediction (P >= 0.62)
    elif has_high_keyword or max_prob >= 0.62:
        return "MEDIUM RISK"
        
    # 3. LOW RISK (Yellow):
    #    - Standard risk detection (fallback)
    else:
        return "LOW RISK"

def clean_text_pipeline(raw_text):
    # 1. Automatically fix Mojibake and encoding issues
    text = ftfy.fix_text(raw_text)
    
    # 2. Normalize hard wraps (single newlines between text) while preserving paragraph breaks (double newlines)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    # 3. Standardize consecutive spaces
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()

def classify_text(raw_text, model_name="electra-small", min_risk_tokens=3):
    if not raw_text or not raw_text.strip():
        return []
        
    # Clean the input text via the NLP pipeline
    cleaned_text = clean_text_pipeline(raw_text)
    
    # Load chosen model and tokenizer dynamically
    model, tokenizer = load_model_by_choice(model_name)
    device = model.device
    
    # Get sentence spans to classify sentence-by-sentence
    sentence_spans = list(PunktSentenceTokenizer().span_tokenize(cleaned_text))
    
    highlighted_data = []
    prev_end = 0
    
    for start_idx, end_idx in sentence_spans:
        # Append spacing/newlines between sentences
        if start_idx > prev_end:
            highlighted_data.append((cleaned_text[prev_end:start_idx], None))
            
        sentence = cleaned_text[start_idx:end_idx]
        if not sentence.strip():
            highlighted_data.append((sentence, None))
            prev_end = end_idx
            continue
            
        # 1. Boilerplate / Header filter
        if clean_boilerplate_header(sentence):
            highlighted_data.append((sentence, None))
            prev_end = end_idx
            continue
            
        # 1.5 Pro-user override filter (suppresses false positives)
        if check_pro_user_override(sentence):
            highlighted_data.append((sentence, None))
            prev_end = end_idx
            continue
            
        # Tokenize sentence
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
        
        # Handle both dict outputs (BiLSTM-CRF) and ModelOutput (transformers)
        if isinstance(outputs, dict):
            logits = outputs['logits'][0]
        else:
            logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)
        
        # Count and extract risk tokens in this sentence
        risk_tokens = []
        for t_idx, pred in enumerate(predictions):
            label = id2label[pred.item()]
            token_str = tokens[t_idx]
            if token_str in ('[CLS]', '[SEP]', '[PAD]'):
                continue
            prob = probs[t_idx][pred.item()].item()
            if label in ('B-RISK', 'I-RISK'):
                risk_tokens.append({"token": token_str, "prob": prob})
        
        # 2. Heuristics & Conditional Threshold
        if len(risk_tokens) >= min_risk_tokens:
            max_prob = max(t["prob"] for t in risk_tokens)
            
            # Check high-risk keywords
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
        
    # Append any remaining characters at the end of the text
    if prev_end < len(cleaned_text):
        highlighted_data.append((cleaned_text[prev_end:], None))
        
    return highlighted_data

# Create Gradio interface
demo = gr.Interface(
    fn=classify_text,
    inputs=[
        gr.Textbox(
            lines=6, 
            label="Terms of Service Text", 
            placeholder="Paste legal agreement clauses, privacy policy paragraphs, or user agreements here..."
        ),
        gr.Dropdown(
            choices=["electra-small", "albert-base", "tinybert", "legal-bert", "bilstm-crf"],
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
        "Analyze Terms of Service and Privacy Policies instantly. This app supports switching between multiple "
        "trained models to detect user-unfavorable clauses such as forced arbitration, class action waivers, "
        "and data-selling agreements."
    ),
    examples=[
        ["Welcome to the platform. By continuing, you agree to forced arbitration in the event of a dispute. We also reserve the right to sell your location data and usage habits to unverified third parties.", "electra-small"],
        ["You agree to defend, indemnify and hold harmless the Company and its officers from and against any claims, liabilities, damages, losses, and expenses.", "electra-small"],
        ["We may modify these terms at any time without notice. Your continued use of the service constitutes acceptance of the new terms.", "electra-small"]
    ]
)



if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
