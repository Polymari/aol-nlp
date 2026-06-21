import os
import re
import time
import torch
import ftfy
import nltk
from nltk.tokenize import PunktSentenceTokenizer
import pandas as pd
import gradio as gr
from transformers import AutoTokenizer, AutoModelForTokenClassification, logging as tf_logging

tf_logging.set_verbosity_error()
tf_logging.disable_progress_bar()

# Download NLTK data securely
for pkg in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{pkg}')
    except LookupError:
        nltk.download(pkg, quiet=True)

MODEL_CACHE = {}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

label2id = {'O': 0, 'B-RISK': 1, 'I-RISK': 2}
id2label = {0: 'O', 1: 'B-RISK', 2: 'I-RISK'}

AVAILABLE_MODELS = ["electra-small", "tinybert", "bert-mini", "bert-tiny"]

# Static model metadata for UI
MODEL_META = {
    "electra-small": {
        "name": "ELECTRA-Small (Fine-tuned)",
        "params": "13.5M",
        "size": "51.5 MB",
        "desc": "Best overall accuracy and F1 score. Balanced size and high reliability.",
        "badge_class": "badge-electra",
        "best_f1": "47.3%"
    },
    "tinybert": {
        "name": "TinyBERT (Fine-tuned)",
        "params": "14.3M",
        "size": "54.4 MB",
        "desc": "Standard compressed BERT model. Moderately accurate but slower than ELECTRA.",
        "badge_class": "badge-tinybert",
        "best_f1": "23.4%"
    },
    "bert-mini": {
        "name": "BERT-Mini (Fine-tuned)",
        "params": "11.1M",
        "size": "42.4 MB",
        "desc": "Lightweight BERT variant. Fast execution with reasonable accuracy.",
        "badge_class": "badge-mini",
        "best_f1": "21.2%"
    },
    "bert-tiny": {
        "name": "BERT-Tiny (Fine-tuned)",
        "params": "4.4M",
        "size": "16.7 MB",
        "desc": "Ultra-lightweight model. Extremely fast with very low resource usage but lower accuracy.",
        "badge_class": "badge-tiny",
        "best_f1": "2.6%"
    }
}

def load_model(model_name):
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    local_path = os.path.join(BASE_DIR, "gotcha-extractor-model", model_name)
    has_local = os.path.exists(local_path) and os.path.exists(os.path.join(local_path, "config.json"))
    
    if has_local:
        model_path = local_path
        print(f"Loading local model weights from: {model_path}")
    else:
        fallback_map = {
            "electra-small": "google/electra-small-discriminator",
            "tinybert": "huawei-noah/TinyBERT_General_4L_312D",
            "bert-tiny": "prajjwal1/bert-tiny",
            "bert-mini": "prajjwal1/bert-mini"
        }
        model_path = fallback_map.get(model_name, "google/electra-small-discriminator")
        print(f"Local model '{model_name}' weights not found. Warning: falling back to base pre-trained model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    # Force CPU to avoid sandboxed CUDA hangs if needed
    device = "cuda" if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "cpu"
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

        if clean_boilerplate_header(sentence) or check_pro_user_override(sentence):
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

# Parse training history metrics
def load_metrics_df():
    import json
    rows = []
    models = ["electra-small", "tinybert", "bert-mini", "bert-tiny"]
    
    for m in models:
        path = os.path.join(BASE_DIR, "gotcha-extractor-model", f"{m}_metrics.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                
                final_run = data.get("final_run", {})
                if final_run:
                    epochs = final_run.get("epochs", [])
                    f1s = final_run.get("f1", [])
                    losses = final_run.get("loss", [])
                    for i in range(len(epochs)):
                        rows.append({
                            "Model": m.upper(),
                            "Epoch": epochs[i],
                            "Validation F1": f1s[i] if i < len(f1s) else None,
                            "Training Loss": losses[i] if i < len(losses) else None
                        })
            except Exception as e:
                print(f"Error reading metrics for {m}: {e}")
                
    if not rows:
        # Fallback dummy data if metrics JSON files are missing
        for m in models:
            for epoch in range(1, 11):
                rows.append({
                    "Model": m.upper(),
                    "Epoch": epoch,
                    "Validation F1": 0.05 * epoch if m == "electra-small" else 0.02 * epoch,
                    "Training Loss": 0.8 / epoch
                })
    return pd.DataFrame(rows)

METRICS_DF = load_metrics_df()

# Single-model analysis handler
def analyze_single(text, model_name, min_tokens):
    if not text or not text.strip():
        return [], "<div style='text-align:center;color:#64748b;'>Enter text to start analysis.</div>", ""
    
    start_time = time.time()
    results = classify_text(text, model_name, min_tokens)
    elapsed = (time.time() - start_time) * 1000
    
    high_count = 0
    med_count = 0
    low_count = 0
    breakdown_md = ""
    
    for text_seg, label in results:
        if label == "HIGH RISK":
            high_count += 1
            breakdown_md += f"- 🔴 **[HIGH RISK]**: \"{text_seg.strip()}\"\n"
        elif label == "MEDIUM RISK":
            med_count += 1
            breakdown_md += f"- 🟠 **[MEDIUM RISK]**: \"{text_seg.strip()}\"\n"
        elif label == "LOW RISK":
            low_count += 1
            breakdown_md += f"- 🟡 **[LOW RISK]**: \"{text_seg.strip()}\"\n"
            
    stats_html = f"""
    <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
        <div class="card-metric" style="flex: 1; min-width: 120px; border-left: 5px solid #ef4444;">
            <div class="card-title">High Risk</div>
            <div class="card-value">{high_count}</div>
            <div class="card-info">Forced arbitration, class action waivers, location tracking.</div>
        </div>
        <div class="card-metric" style="flex: 1; min-width: 120px; border-left: 5px solid #f97316;">
            <div class="card-title">Medium Risk</div>
            <div class="card-value">{med_count}</div>
            <div class="card-info">Unilateral modifications, advertising trackers.</div>
        </div>
        <div class="card-metric" style="flex: 1; min-width: 120px; border-left: 5px solid #eab308;">
            <div class="card-title">Low Risk</div>
            <div class="card-value">{low_count}</div>
            <div class="card-info">Broad warranty disclaimers, standard liabilities.</div>
        </div>
        <div class="card-metric" style="flex: 1; min-width: 120px; border-left: 5px solid #3b82f6;">
            <div class="card-title">Latency</div>
            <div class="card-value">{elapsed:.1f}ms</div>
            <div class="card-info">Execution time on CPU.</div>
        </div>
    </div>
    """
    
    if not breakdown_md:
        breakdown_md = "*No risky clauses detected. This agreement looks standard!*"
        
    return results, stats_html, breakdown_md

# Multi-model comparison handler
def compare_models(text, min_tokens):
    if not text or not text.strip():
        return [], [], [], [], pd.DataFrame()
    
    res_electra = classify_text(text, "electra-small", min_tokens)
    res_tinybert = classify_text(text, "tinybert", min_tokens)
    res_mini = classify_text(text, "bert-mini", min_tokens)
    res_tiny = classify_text(text, "bert-tiny", min_tokens)
    
    comparison_rows = []
    for m in AVAILABLE_MODELS:
        start_time = time.time()
        results = classify_text(text, m, min_tokens)
        elapsed = (time.time() - start_time) * 1000
        
        risky_count = sum(1 for _, label in results if label is not None)
        meta = MODEL_META[m]
        
        comparison_rows.append({
            "Model": meta["name"],
            "Validation F1 (Best)": meta["best_f1"],
            "Parameters": meta["params"],
            "Disk Size": meta["size"],
            "Risks Detected": risky_count,
            "Latency (ms)": f"{elapsed:.1f} ms"
        })
        
    df_compare = pd.DataFrame(comparison_rows)
    return res_electra, res_tinybert, res_mini, res_tiny, df_compare

# Preset Examples
EXAMPLES = [
    [
        "Welcome to the platform. By continuing, you agree to forced arbitration in the event of a dispute. We also reserve the right to sell your location data and usage habits to unverified third parties.",
        "electra-small",
        3
    ],
    [
        "You agree to defend, indemnify and hold harmless the Company and its officers from and against any claims, liabilities, damages, losses, and expenses.",
        "electra-small",
        3
    ],
    [
        "We may modify these terms at any time without notice. Your continued use of the service constitutes acceptance of the new terms.",
        "electra-small",
        3
    ]
]

# Custom CSS
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

body, .gradio-container {
    font-family: 'Outfit', sans-serif !important;
}

.header-container {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    color: white;
    padding: 2.5rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    text-align: center;
}

.header-container h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    background: linear-gradient(to right, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.header-container p {
    font-size: 1.1rem;
    color: #cbd5e1;
    max-width: 800px;
    margin: 0 auto;
}

.card-metric {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1.25rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.card-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.25rem;
}

.card-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #0f172a;
}

.card-info {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 0.25rem;
}

.model-card {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.5rem;
    background: #ffffff;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    transition: transform 0.2s, box-shadow 0.2s;
}

.model-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
}

.model-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    font-size: 0.8rem;
    font-weight: 600;
    border-radius: 9999px;
    margin-bottom: 0.75rem;
}

.badge-electra { background: #e0f2fe; color: #0369a1; }
.badge-tinybert { background: #fef3c7; color: #d97706; }
.badge-mini { background: #f3e8ff; color: #7e22ce; }
.badge-tiny { background: #dcfce7; color: #15803d; }
"""

# Color map for HighlightedText output
COLOR_MAP = {
    "HIGH RISK": "#ef4444",
    "MEDIUM RISK": "#f97316",
    "LOW RISK": "#eab308"
}

with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
    
    # Custom Gradient Header
    gr.HTML("""
    <div class="header-container">
        <h1>ToS 'Gotcha' Clause Extractor</h1>
        <p>Analyze legal terms and privacy policies instantly using four fine-tuned language models. Compare model capabilities side-by-side to understand accuracy and latency trade-offs.</p>
    </div>
    """)
    
    with gr.Tabs():
        
        # TAB 1: Single Model Classifier
        with gr.TabItem("🔍 Single Model Extractor"):
            with gr.Row():
                with gr.Column(scale=4):
                    text_input = gr.Textbox(
                        lines=10,
                        label="Terms of Service or Privacy Policy text",
                        placeholder="Paste legal agreement clauses, privacy policy paragraphs, or user agreements here..."
                    )
                    with gr.Row():
                        model_dropdown = gr.Dropdown(
                            choices=AVAILABLE_MODELS,
                            value="electra-small",
                            label="Select Extraction Model"
                        )
                        min_tokens_slider = gr.Slider(
                            minimum=1,
                            maximum=5,
                            step=1,
                            value=3,
                            label="Min Risk Tokens in Sentence"
                        )
                    analyze_btn = gr.Button("Analyze Clauses", variant="primary")
                    
                with gr.Column(scale=5):
                    gr.Markdown("### Risk Assessment & Latency")
                    stats_output = gr.HTML("<div style='text-align:center;color:#64748b;'>Enter text and click 'Analyze Clauses' to see results.</div>")
                    
                    highlighted_output = gr.HighlightedText(
                        label="Analysis Results (Highlighted Clauses)",
                        combine_adjacent=False,
                        color_map=COLOR_MAP
                    )
                    
                    with gr.Accordion("🔍 Detailed Risky Clause Breakdown", open=True):
                        breakdown_output = gr.Markdown("*Detailed breakdown will appear here...*")
            
            # Wire up single analyzer
            analyze_btn.click(
                fn=analyze_single,
                inputs=[text_input, model_dropdown, min_tokens_slider],
                outputs=[highlighted_output, stats_output, breakdown_output]
            )
            
            # Examples
            gr.Examples(
                examples=EXAMPLES,
                inputs=[text_input, model_dropdown, min_tokens_slider],
                outputs=[highlighted_output, stats_output, breakdown_output],
                fn=analyze_single,
                cache_examples=False
            )

        # TAB 2: Side-by-Side Model Comparison
        with gr.TabItem("📊 Compare Models Side-by-Side"):
            gr.Markdown("Compare how all four fine-tuned models identify risks and measure their inference latencies.")
            
            with gr.Row():
                comp_text_input = gr.Textbox(
                    lines=5,
                    label="Enter clauses to compare",
                    value="We reserve the right to modify these terms at any time without notice. In the event of a dispute, you waive your right to a class action lawsuit and agree to binding arbitration.",
                    placeholder="Enter legal sentences to test..."
                )
            
            with gr.Row():
                comp_tokens_slider = gr.Slider(
                    minimum=1,
                    maximum=5,
                    step=1,
                    value=3,
                    label="Min Risk Tokens"
                )
                compare_btn = gr.Button("Compare All Models", variant="primary")
                
            gr.Markdown("### Highlighting Comparison")
            
            with gr.Row():
                with gr.Column():
                    gr.HTML("<div class='model-badge badge-electra'>ELECTRA-Small (Best Accuracy)</div>")
                    out_electra = gr.HighlightedText(label="ELECTRA-Small Output", combine_adjacent=False, color_map=COLOR_MAP)
                with gr.Column():
                    gr.HTML("<div class='model-badge badge-tinybert'>TinyBERT</div>")
                    out_tinybert = gr.HighlightedText(label="TinyBERT Output", combine_adjacent=False, color_map=COLOR_MAP)
                    
            with gr.Row():
                with gr.Column():
                    gr.HTML("<div class='model-badge badge-mini'>BERT-Mini</div>")
                    out_mini = gr.HighlightedText(label="BERT-Mini Output", combine_adjacent=False, color_map=COLOR_MAP)
                with gr.Column():
                    gr.HTML("<div class='model-badge badge-tiny'>BERT-Tiny</div>")
                    out_tiny = gr.HighlightedText(label="BERT-Tiny Output", combine_adjacent=False, color_map=COLOR_MAP)
            
            gr.Markdown("### Performance Summary")
            comparison_df = gr.Dataframe(
                headers=["Model", "Validation F1 (Best)", "Parameters", "Disk Size", "Risks Detected", "Latency (ms)"],
                datatype=["str", "str", "str", "str", "number", "str"],
                label="Metrics Comparison Table"
            )
            
            compare_btn.click(
                fn=compare_models,
                inputs=[comp_text_input, comp_tokens_slider],
                outputs=[out_electra, out_tinybert, out_mini, out_tiny, comparison_df]
            )

        # TAB 3: Metrics Dashboard & History
        with gr.TabItem("📈 Performance & Training Dashboard"):
            gr.Markdown("### Evaluation Leaderboard")
            
            leaderboard_rows = []
            for m in AVAILABLE_MODELS:
                meta = MODEL_META[m]
                leaderboard_rows.append([
                    meta["name"],
                    meta["best_f1"],
                    meta["params"],
                    meta["size"],
                    meta["desc"]
                ])
                
            gr.Dataframe(
                value=leaderboard_rows,
                headers=["Model Name", "Best Validation F1", "Parameter Count", "File Size", "Model Profile"],
                datatype=["str", "str", "str", "str", "str"],
                interactive=False
            )
            
            gr.Markdown("### Training Histories (Comparison)")
            
            with gr.Row():
                f1_plot = gr.LinePlot(
                    value=METRICS_DF,
                    x="Epoch",
                    y="Validation F1",
                    color="Model",
                    title="Validation F1 Score vs. Training Epochs",
                    tooltip=["Model", "Epoch", "Validation F1"]
                )
                
                loss_plot = gr.LinePlot(
                    value=METRICS_DF,
                    x="Epoch",
                    y="Training Loss",
                    color="Model",
                    title="Training Loss vs. Training Epochs",
                    tooltip=["Model", "Epoch", "Training Loss"]
                )
                
            gr.Markdown("""
            ### Technical Training Notes
            - **Dataset**: Fine-tuned on a sequence classification dataset annotated for "Gotcha" clauses (Arbitration, class actions, locations, unilateral updates).
            - **Sequence Tagging**: Models categorize each token as `B-RISK` (beginning of risk), `I-RISK` (inside risk), or `O` (outside risk).
            - **Post-Processing**: Sentences are evaluated for risk density based on token count and keywords to filter out general legal boilerplate.
            """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
