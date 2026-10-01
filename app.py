import os
import time
import pandas as pd

# Compatibility patch for Gradio 4 with huggingface_hub>=0.25.0
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "HfFolder"):
        class _HfFolderFallback:
            @staticmethod
            def get_token():
                return huggingface_hub.get_token()
            @staticmethod
            def save_token(token):
                if hasattr(huggingface_hub, "login"):
                    try:
                        huggingface_hub.login(token=token)
                    except Exception:
                        pass
            @staticmethod
            def delete_token():
                if hasattr(huggingface_hub, "logout"):
                    try:
                        huggingface_hub.logout()
                    except Exception:
                        pass
        huggingface_hub.HfFolder = _HfFolderFallback
except Exception:
    pass

import gradio as gr

from gotcha import (
    AVAILABLE_MODELS,
    MODEL_META,
    COLOR_MAP,
    RISK_INK,
    classify_text,
    compare_models as gotcha_compare_models,
    get_inference_device
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Training telemetry ledger (reads the JSON written by train.py)
# ---------------------------------------------------------------------------

def load_comprehensive_metrics():
    import json
    history_rows = []
    optuna_trials_map = {}
    models_summary_map = {}

    for m in AVAILABLE_MODELS:
        path = os.path.join(BASE_DIR, "gotcha-extractor-model", f"{m}_metrics.json")
        data = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading metrics for {m}: {e}")

        final_run = data.get("final_run", {})
        best_hp = data.get("best_hp", {})
        trials = data.get("trials", {})
        test_eval = data.get("test_eval", {})
        summary = data.get("summary", {})

        trial_rows = []
        for t_num, t_info in trials.items():
            params = t_info.get("params", {})
            f1_val = t_info.get("best_f1")
            if f1_val is None and t_info.get("f1"):
                f1_val = max(t_info["f1"])

            trial_rows.append({
                "Trial": t_num,
                "Val F1": f"{f1_val:.3f}" if isinstance(f1_val, (int, float)) else "—",
                "Learning rate": f"{params['learning_rate']:.2e}" if isinstance(params.get("learning_rate"), (int, float)) else "—",
                "Weight decay": f"{params['weight_decay']:.4f}" if isinstance(params.get("weight_decay"), (int, float)) else "—",
                "Batch": str(params.get("per_device_train_batch_size", "—")),
                "Warmup": f"{params['warmup_ratio'] * 100:.0f}%" if isinstance(params.get("warmup_ratio"), (int, float)) else "—",
                "Frozen": str(params.get("num_frozen_layers", "—")),
                "_raw": float(f1_val) if isinstance(f1_val, (int, float)) else 0.0
            })

        trial_rows.sort(key=lambda r: r["_raw"], reverse=True)
        for r in trial_rows:
            del r["_raw"]

        optuna_trials_map[m] = pd.DataFrame(trial_rows) if trial_rows else pd.DataFrame()

        models_summary_map[m] = {
            "name": MODEL_META.get(m, {}).get("name", m.upper()),
            "test_f1": test_eval.get("f1"),
            "test_recall": test_eval.get("recall"),
            "test_precision": test_eval.get("precision"),
            "test_accuracy": test_eval.get("accuracy"),
            "best_hp": best_hp,
            "total_trials": len(trials),
            "desc": MODEL_META.get(m, {}).get("desc", ""),
            "params": MODEL_META.get(m, {}).get("params", "—"),
        }

        epochs = final_run.get("epochs", [])
        f1s = final_run.get("f1", [])
        train_losses = final_run.get("train_loss", [])
        eval_losses = final_run.get("eval_loss", final_run.get("loss", []))
        for i in range(len(epochs)):
            history_rows.append({
                "Model": m,
                "Epoch": epochs[i],
                "Validation F1": f1s[i] if i < len(f1s) else None,
                "Training loss": train_losses[i] if i < len(train_losses) else None,
                "Validation loss": eval_losses[i] if i < len(eval_losses) else None,
            })

    metrics_df = pd.DataFrame(history_rows)
    if not history_rows:
        metrics_df = pd.DataFrame(columns=["Model", "Epoch", "Validation F1", "Training loss", "Validation loss"])
    return metrics_df, optuna_trials_map, models_summary_map


METRICS_DF, OPTUNA_TRIALS_MAP, MODELS_SUMMARY_MAP = load_comprehensive_metrics()


def fmt_pct(value, digits=1):
    if isinstance(value, (int, float)) and value:
        return f"{value * 100:.{digits}f}%"
    return "—"


# Model names and test F1 for the comparison lanes, read from the metrics
# ledger so the header cannot drift from what the ledger reports.
COMPARE_PANELS = [
    (MODEL_META.get(m, {}).get("name", m).replace(" (Fine-tuned)", ""), fmt_pct(
        MODELS_SUMMARY_MAP.get(m, {}).get("test_f1")))
    for m in AVAILABLE_MODELS
]


# ---------------------------------------------------------------------------
# Clause taxonomy
# ---------------------------------------------------------------------------

def categorize_gotcha(sentence: str) -> dict:
    """Classify a flagged clause into one of the risk families the model detects."""
    s = sentence.lower()
    if any(k in s for k in ("arbitrat", "class action", "jury", "court proceeding", "dispute")):
        return {
            "title": "Forced arbitration",
            "family": "Dispute resolution",
            "detail": "Removes your access to court and to collective action, and routes disputes through private arbitration."
        }
    if any(k in s for k in ("sell", "broker", "advertis", "third part", "market your", "telemetry", "location data")):
        return {
            "title": "Data brokerage",
            "family": "Privacy",
            "detail": "Authorizes collecting and syndicating your activity to outside advertisers and data brokers."
        }
    if any(k in s for k in ("modify", "revise", "update these", "without notice", "without prior notice", "at any time", "sole and absolute discretion")):
        return {
            "title": "Unilateral modification",
            "family": "Contract terms",
            "detail": "Lets the company change these terms on its own, without telling you or asking you to agree again."
        }
    if any(k in s for k in ("indemni", "hold harmless", "defend", "liabilit")):
        return {
            "title": "Indemnification",
            "family": "Liability",
            "detail": "Shifts legal costs and third-party claims onto you, including claims caused by the company's own conduct."
        }
    if any(k in s for k in ("no warranty", "as is", "as-is", "cannot ensure", "cannot warrant", "cannot guarantee")):
        return {
            "title": "Warranty disclaimer",
            "family": "Remedies",
            "detail": "Disclaims any promise that the service will work as expected, leaving you without a remedy if it does not."
        }
    return {
        "title": "Restrictive term",
        "family": "General",
        "detail": "Restricts a right or benefit you would reasonably expect to keep."
    }


SEVERITY_ORDER = {"HIGH RISK": 0, "MEDIUM RISK": 1, "LOW RISK": 2}


def render_findings(results):
    """Render flagged clauses as an ordered findings list."""
    flagged = [(seg.strip(), label) for seg, label in results if label and seg.strip()]
    if not flagged:
        return ""
    flagged.sort(key=lambda item: SEVERITY_ORDER.get(item[1], 3))

    items = []
    for i, (segment, label) in enumerate(flagged, 1):
        info = categorize_gotcha(segment)
        ink = RISK_INK.get(label, RISK_INK["LOW RISK"])
        items.append(f"""
        <li class="finding" style="--ink: {ink}">
          <div class="finding-head">
            <span class="finding-num">{i:02d}</span>
            <div>
              <div class="finding-title">{info['title']}</div>
              <div class="finding-family">{info['family']} · {label.replace(' RISK', '').title()} risk</div>
            </div>
          </div>
          <blockquote class="finding-quote">{segment}</blockquote>
          <p class="finding-detail">{info['detail']}</p>
        </li>
        """)
    return f'<ol class="findings">{"".join(items)}</ol>'


def render_summary(counts, elapsed_ms):
    """A single honest sentence about what was found, plus secondary metadata."""
    total = counts["high"] + counts["medium"] + counts["low"]
    if total == 0:
        return ""
    parts = [f"{counts['high']} high risk", f"{counts['medium']} medium", f"{counts['low']} low"]
    device = get_inference_device().type.upper()
    return f"""
    <div class="summary">
      <p class="summary-line">{total} flagged clause{'' if total == 1 else 's'}: {', '.join(parts)}.</p>
      <p class="summary-meta">{elapsed_ms:.0f} ms · {device} · BIO token classification</p>
    </div>
    """


def empty_state(has_text):
    if not has_text:
        return """
        <div class="empty">
          <p class="empty-title">Paste a contract to begin</p>
          <p class="empty-body">Terms of service, a privacy policy, or an EULA. Every sentence is
          classified for risky clauses and the flagged text is marked in place.</p>
        </div>
        """
    return """
    <div class="empty">
      <p class="empty-title">No risky clauses flagged</p>
      <p class="empty-body">Nothing in this text crossed the sensitivity threshold. Try lowering the
      threshold to one if you want a more sensitive pass.</p>
    </div>
    """


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def to_highlight_pairs(results):
    """Normalize classified segments for HighlightedText.

    The pipeline emits inter-sentence whitespace as its own (label=None)
    segment. Gradio hides those spans when show_whitespaces is off, which
    glues adjacent words together and breaks them mid-token. Folding the
    whitespace into the preceding segment keeps the prose intact.
    """
    pairs = []
    for segment, label in results:
        if not pairs:
            if segment.strip():
                pairs.append([segment, label])
            continue
        if not segment.strip():
            pairs[-1][0] += segment
        else:
            pairs.append([segment, label])
    return [(text, label) for text, label in pairs]


def analyze_single(text, model_name, min_tokens):
    """Run the classifier and return every output the workspace needs.

    Returns (annotated_text, summary_html, findings_html, label_update,
    text_update). The two updates reveal the annotated-text heading and
    panel together, and only when there is something to show.
    """
    hidden = (gr.update(visible=False), gr.update(visible=False))

    if not text or not text.strip():
        return [], empty_state(False), "", *hidden

    start = time.time()
    try:
        results = classify_text(text, model_name=model_name, min_risk_tokens=min_tokens)
    except Exception as e:
        error = (
            f'<div class="notice notice-error"><p class="empty-title">Analysis failed</p>'
            f'<p class="empty-body">{e}</p></div>'
        )
        return [], empty_state(True), error, *hidden
    elapsed = (time.time() - start) * 1000

    counts = {"high": 0, "medium": 0, "low": 0}
    for _, label in results:
        if label == "HIGH RISK":
            counts["high"] += 1
        elif label == "MEDIUM RISK":
            counts["medium"] += 1
        elif label == "LOW RISK":
            counts["low"] += 1

    if sum(counts.values()) == 0:
        return [], empty_state(True), "", *hidden

    shown = (gr.update(visible=True), gr.update(visible=True))
    return (
        to_highlight_pairs(results),
        render_summary(counts, elapsed),
        render_findings(results),
        *shown,
    )


def compare_all(text, min_tokens):
    if not text or not text.strip():
        return [], [], [], [], pd.DataFrame()

    e, tb, bm, bt, df = gotcha_compare_models(text, min_tokens=min_tokens)
    return e, tb, bm, bt, df


# ---------------------------------------------------------------------------
# Sample clauses
# ---------------------------------------------------------------------------

PRESET_CASES = {
    "arbitration": (
        "Welcome to the platform. By continuing to use our services, you expressly agree that any and all "
        "disputes, claims, or controversies arising out of or relating to these Terms shall be resolved "
        "exclusively by confidential, binding arbitration administered by the American Arbitration "
        "Association, and you expressly waive any right to a trial by jury or to participate in a class "
        "action lawsuit or class-wide arbitration."
    ),
    "surveillance": (
        "We reserve the right to collect, synthesize, and monetize your precise geographic coordinates, "
        "device identifiers, and browsing habits, and to syndicate such behavioral telemetry to commercial "
        "third parties, ad networks, and data brokers for targeted advertising and market research "
        "without further notice to you."
    ),
    "modification": (
        "We reserve the right, at our sole and absolute discretion, to modify, amend, replace, or update "
        "these Terms of Service at any time without prior notice. Your continued access to or use of the "
        "service following the posting of any modifications constitutes binding and irrevocable "
        "acceptance of the revised covenants."
    ),
    "indemnity": (
        "You agree to defend, indemnify, and hold harmless the Company, its subsidiaries, affiliates, "
        "officers, and directors from and against any and all claims, liabilities, damages, losses, "
        "expenses, and reasonable attorneys' fees arising out of or in any way connected with your "
        "access to or use of the Services, including any claims resulting from our own negligence."
    ),
    "safe": (
        "You may request a copy of the personal data we hold about you, and you may ask us to correct or "
        "delete it at any time. If you have questions about this policy, contact our privacy team and we "
        "will respond within thirty days. These terms take effect on the date shown above and apply to "
        "all users of the service."
    ),
}


# ---------------------------------------------------------------------------
# Design system
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=Inter:wght@400;450;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --paper:        #fbfaf8;
  --paper-sunk:   #f4f2ee;
  --surface:      #ffffff;
  --rule:         #e5e1da;
  --rule-strong:  #d3cec4;

  --ink:          #1c1a17;
  --ink-soft:     #56514a;
  /* 5.15:1 on paper, 4.81:1 on the sunken quote surface. The lighter
     #6f6a60 failed AA at 3.56:1. */
  --ink-faint:    #6f6a60;

  --accent:       #a4262c;
  --accent-wash:  #fdf3f3;
  --focus:        #1c1a17;

  --radius:       6px;
  --gap-sm:       8px;
  --gap:          16px;
  --gap-lg:       36px;

  --font-doc:  'Newsreader', Georgia, serif;
  --font-ui:   'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --font-data: 'JetBrains Mono', ui-monospace, monospace;

  /* ---- Gradio 6 theme variables ----
     Gradio 6 scopes its theme to :root.dark and to an injected body rule that
     outranks a plain .gradio-container selector. Overriding these at equal or
     higher specificity is what actually forces the light surface. */
  --bg:                    #fbfaf8;
  --col:                   #1c1a17;
  --background-fill-primary:   #fbfaf8;
  --body-background-fill:      #fbfaf8;
  --body-text-color:           #1c1a17;
  --body-text-color-subdued:   #56514a;
  --body-text-color-muted:     #6f6a60;
  --neutral-1:                #1c1a17;
  --neutral-100:              #f4f2ee;
  --neutral-200:              #e5e1da;
  --neutral-300:              #d3cec4;
  --neutral-500:              #6f6a60;
  --neutral-700:              #56514a;
  --input-background-fill:    #ffffff;
  --input-background-fill-hover: #ffffff;
  --input-border-color:       #d3cec4;
  --input-border-color-hover: #6f6a60;
  --block-background-fill:    #ffffff;
  --block-background-fill-soft: #f4f2ee;
  --block-border-color:       #e5e1da;
  --block-border-color-soft:  #e5e1da;
  --border-primary:           #e5e1da;
  --border-primary-accent:    #d3cec4;
  --color-accent:             #a4262c;
  --color-accent-soft:        #fdf3f3;
  --color-accent-crisp:       #a4262c;
  --button-primary-background-fill: #a4262c;
  --button-primary-background-fill-hover: #8f1f24;
  --button-primary-text-color: #ffffff;
  --button-secondary-background-fill: #ffffff;
  --button-secondary-background-fill-hover: #f4f2ee;
  --button-secondary-text-color: #1c1a17;
  --button-secondary-border-color: #d3cec4;
  --checkbox-background-fill: #ffffff;
  --slider-color: #a4262c;
  --table-background-fill: #ffffff;
  --table-even-background-fill: #fbfaf8;
  --table-odd-background-fill: #ffffff;
  --dataframe-selected-background-fill: #fdf3f3;
}

html, body {
  background: var(--paper) !important;
  color: var(--ink) !important;
}

body.dark, :root.dark, .dark {
  color-scheme: light;
}

/* Gradio 6 keeps the page in a `.dark` scope whose `:root.dark` rule outranks a
   plain `:root`, so the neutral ramp has to be restated on that selector too
   or inputs keep resolving to their hardcoded dark fills. */
:root.dark, .dark {
  --bg:                    #fbfaf8;
  --col:                   #1c1a17;
  --background-fill-primary:   #fbfaf8;
  --body-background-fill:      #fbfaf8;
  --body-text-color:           #1c1a17;
  --body-text-color-subdued:   #56514a;
  --body-text-color-muted:     #6f6a60;
  --neutral-1:                #1c1a17;
  --neutral-100:              #f4f2ee;
  --neutral-200:              #e5e1da;
  --neutral-300:              #d3cec4;
  --neutral-500:              #6f6a60;
  --neutral-600:              #6f6a60;
  --neutral-700:              #56514a;
  --neutral-800:              #3a3630;
  --neutral-900:              #2a2724;
  --neutral-950:              #1c1a17;
  --input-background-fill:    #ffffff;
  --input-background-fill-hover: #ffffff;
  --input-border-color:       #d3cec4;
  --input-border-color-hover: #6f6a60;
  --block-background-fill:    #ffffff;
  --block-background-fill-soft: #f4f2ee;
  --block-border-color:       #e5e1da;
  --block-border-color-soft:  #e5e1da;
  --border-primary:           #e5e1da;
  --color-accent:             #a4262c;
  --color-accent-soft:        #fdf3f3;
  --color-accent-crisp:       #a4262c;
  --button-primary-background-fill: #a4262c;
  --button-primary-background-fill-hover: #8f1f24;
  --button-primary-text-color: #ffffff;
  --button-secondary-background-fill: #ffffff;
  --button-secondary-background-fill-hover: #f4f2ee;
  --button-secondary-text-color: #1c1a17;
  --button-secondary-border-color: #d3cec4;
  --slider-color: #a4262c;
  --table-background-fill: #ffffff;
  --table-even-background-fill: #fbfaf8;
  --table-odd-background-fill: #ffffff;
  --color-scheme: light;
}

/* Gradio 6's dark scope paints .block, .form and .column with hardcoded dark
   fills that sit behind the input and swallow it. Force the page's own
   surfaces, then re-establish the textbox as the one white control. */
.main, .column, .row, .form, .panel, .block,
.form > *, .block > *, .column > *, .row > * {
  background-color: transparent !important;
  border-color: transparent !important;
}

.textbox, .textbox > *, .textbox textarea {
  background-color: var(--surface) !important;
  color: var(--ink) !important;
}

.textbox {
  border-color: var(--rule-strong) !important;
}

/* ---------- Shell ---------- */

.gradio-container {
  background: var(--paper) !important;
  color: var(--ink) !important;
  font-family: var(--font-ui) !important;
  max-width: 1180px !important;
  padding: 0 24px 72px !important;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

.masthead {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: var(--gap);
  padding: 40px 0 20px;
  border-bottom: 1px solid var(--rule);
  margin-bottom: 32px;
}

.masthead-brand {
  display: flex;
  align-items: baseline;
  gap: 10px;
}

.wordmark {
  font-family: var(--font-doc);
  font-size: 1.5rem;
  font-weight: 600;
  letter-spacing: -0.015em;
  color: var(--ink);
}

.wordmark-sub {
  font-size: 0.82rem;
  color: var(--ink-faint);
  letter-spacing: 0.01em;
}

.masthead-note {
  font-size: 0.82rem;
  color: var(--ink-faint);
  margin: 0;
  max-width: 34ch;
  text-align: right;
  line-height: 1.5;
}

/* ---------- Tabs ---------- */

/* Gradio 6 renders tabs as .tabs > .tab-wrapper > .tab-container. The old
   .gr-tabs / .tab-nav names belong to Gradio 4 and match nothing here. */
.tabs > .tab-wrapper > .tab-container {
  display: flex !important;
  gap: 26px !important;
  border-bottom: 1px solid var(--rule) !important;
  margin-bottom: 32px !important;
  background: transparent !important;
  flex-wrap: wrap;
}

.tabs > .tab-wrapper > .tab-container > button {
  background: none !important;
  border: none !important;
  box-shadow: none !important;
  padding: 8px 0 12px !important;
  border-bottom: 2px solid transparent !important;
  margin-bottom: -1px !important;
  font-family: var(--font-ui) !important;
  font-size: 0.9rem !important;
  font-weight: 450 !important;
  color: var(--ink-faint) !important;
  transition: color 120ms ease, border-color 120ms ease;
}

.tabs > .tab-wrapper > .tab-container > button:hover {
  color: var(--ink-soft) !important;
}

.tabs > .tab-wrapper > .tab-container > button:focus-visible {
  outline: 2px solid var(--focus) !important;
  outline-offset: 3px !important;
  border-radius: 2px;
}

.tabs > .tab-wrapper > .tab-container > button.selected {
  color: var(--ink) !important;
  font-weight: 600 !important;
  border-bottom-color: var(--accent) !important;
}

/* ---------- Field labels ---------- */
/* Gradio paints label + hint text from block_info_text_color, which reads
   near-white on this light surface unless it is restated. */
label, label > span,
.block > label, .block > label > span,
.block_label, .block_info, .block_info > span,
span.block_label, span.block_info,
span.has-info {
  color: var(--ink-soft) !important;
  font-weight: 500 !important;
  letter-spacing: 0 !important;
  text-transform: none !important;
  opacity: 1 !important;
}

/* The hint under a field label is secondary, not primary. */
span.has-info + span,
.block .has-info:not(:only-child) {
  color: var(--ink-faint) !important;
  font-weight: 400 !important;
}

.block > label, .block > label > span, .block_label {
  color: var(--ink) !important;
}

.block_info, .block_info > span, .block_info_text {
  color: var(--ink-faint) !important;
  font-weight: 400 !important;
  font-size: 0.74rem !important;
}

/* ---------- Inputs ---------- */

.gr-textbox textarea, .gr-textbox input {
  background: var(--surface) !important;
  color: var(--ink) !important;
  border: 1px solid var(--rule-strong) !important;
  border-radius: var(--radius) !important;
  font-family: var(--font-doc) !important;
  font-size: 1.02rem !important;
  line-height: 1.65 !important;
  padding: 14px 16px !important;
  transition: border-color 120ms ease, box-shadow 120ms ease;
}

.gr-textbox textarea:focus, .gr-textbox input:focus {
  border-color: var(--focus) !important;
  box-shadow: 0 0 0 3px rgba(28, 26, 23, 0.08) !important;
  outline: none !important;
}

.gr-textbox label, .gr-dropdown label, .gr-slider label, .gr-checkbox label {
  font-family: var(--font-ui) !important;
  font-size: 0.78rem !important;
  font-weight: 500 !important;
  color: var(--ink-soft) !important;
  margin-bottom: 6px !important;
}

.gr-textbox .info, .gr-slider .info, .gr-dropdown .info {
  font-size: 0.74rem !important;
  color: var(--ink-faint) !important;
}

.gr-dropdown > div, .gr-dropdown .wrap {
  background: var(--surface) !important;
  border: 1px solid var(--rule-strong) !important;
  border-radius: var(--radius) !important;
}

input[type=range] { accent-color: var(--accent) !important; }

/* ---------- Buttons ---------- */

.gr-button {
  font-family: var(--font-ui) !important;
  font-weight: 500 !important;
  font-size: 0.88rem !important;
  border-radius: var(--radius) !important;
  transition: background 120ms ease, border-color 120ms ease, color 120ms ease;
}

.btn-primary {
  background: var(--accent) !important;
  border: 1px solid var(--accent) !important;
  color: #fff !important;
}
.btn-primary:hover { background: #8f1f24 !important; border-color: #8f1f24 !important; }

.btn-secondary {
  background: var(--surface) !important;
  border: 1px solid var(--rule-strong) !important;
  color: var(--ink-soft) !important;
}
.btn-secondary:hover { border-color: var(--ink-faint) !important; color: var(--ink) !important; }

.gr-button:focus-visible, button:focus-visible, textarea:focus-visible, select:focus-visible {
  outline: 2px solid var(--focus) !important;
  outline-offset: 2px !important;
}

/* ---------- Control row ---------- */
/* Model and sensitivity sit in one column so each keeps a single-line
   label; stacking them also stops them reading as competing peers. */
.control-row {
  gap: 20px !important;
  align-items: start !important;
}

.control-row > * {
  min-width: 0 !important;
}

/* Gradio's slider header collapses to a narrow column, which wrapped the
   "Sensitivity" label and its hint onto three lines each. Give the header
   the full track width. The hint is allowed to wrap rather than clip. */
.control-row .wrap,
.control-row .head {
  width: 100% !important;
}

.control-row label,
.control-row .head > label {
  white-space: nowrap !important;
}

.control-row .info-text,
.control-row .block_info {
  max-width: none !important;
}

/* ---------- Presets ----------
   These are a convenience, not the main event. They render as small text
   buttons on a hairline row so the review button keeps the accent. */
.preset-bar {
  display: flex !important;
  align-items: center;
  justify-content: flex-start;
  gap: 4px;
  flex-wrap: wrap;
  padding-bottom: 18px;
  margin-bottom: 4px;
  border-bottom: 1px solid var(--rule);
}

.preset-label {
  font-size: 0.78rem;
  color: var(--ink-faint);
  white-space: nowrap;
  margin-inline-end: 10px;
}

.preset-btn {
  background: none !important;
  border: 1px solid transparent !important;
  color: var(--ink-soft) !important;
  font-size: 0.8rem !important;
  font-weight: 450 !important;
  padding: 5px 10px !important;
  min-height: 0 !important;
  width: auto !important;
  flex: 0 0 auto !important;
  border-radius: 4px !important;
  box-shadow: none !important;
  text-decoration: underline;
  text-decoration-color: var(--rule-strong);
  text-underline-offset: 3px;
}

.preset-btn:hover {
  background: var(--paper-sunk) !important;
  border-color: var(--rule-strong) !important;
  color: var(--ink) !important;
  text-decoration-color: transparent;
}

.preset-btn:focus-visible {
  outline: 2px solid var(--focus) !important;
  outline-offset: 2px !important;
}

/* ---------- Document ---------- */

.doc-pane {
  background: var(--surface);
  border: 1px solid var(--rule);
  border-radius: var(--radius);
  padding: 24px 26px;
  min-height: 320px;
}

.highlighted-text {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  font-family: var(--font-doc) !important;
  font-size: 1.02rem !important;
  line-height: 1.72 !important;
  color: var(--ink-soft) !important;
}

/* Token spans are inline; let the browser wrap between them rather than
   letting the component hyphenate words at its own boundaries. */
.highlighted-text .token,
.highlighted-text .token-container,
.highlighted-text .text,
.highlighted-text .textfield,
.highlighted-text span[class*="token"],
.highlighted-text span[class*="text"] {
  white-space: normal !important;
  word-break: normal !important;
  overflow-wrap: break-word !important;
}

/* Gradio marks an HTML container "pending" once it has been toggled visible
   and never clears the class, which pins it at opacity 0.2. The legend is
   static content, so it is never actually pending. */
.html-container.pending {
  opacity: 1 !important;
}

/* Gradio stamps a "processing | Ns" line and an edit affordance on every
   run. The summary already reports latency, and a read-only annotation view
   has nothing to edit. These render in sibling blocks rather than inside
   .highlighted-text, so they are matched on their own class names. */
.progress-text,
[class*="progress-text"] {
  display: none !important;
}

.wrap.full.translucent {
  display: none !important;
}

.highlighted-text span {
  border-radius: 2px !important;
  padding: 1px 2px !important;
  color: var(--ink) !important;
  box-decoration-break: clone;
  -webkit-box-decoration-break: clone;
}

.doc-label {
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--ink-faint);
  margin: 0 0 10px;
}

/* Legend rides on the same line as the heading it explains, so it reads as
   a key to that panel rather than a floating strip of loose swatches. */
.doc-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: var(--gap);
  flex-wrap: wrap;
  padding-bottom: 10px;
  margin-bottom: 14px;
  border-bottom: 1px solid var(--rule);
}

/* The key reads as a caption under the annotated prose, not a floating
   strip, so it keeps its own top margin rather than the panel divider. */
.legend-key {
  padding-top: 12px;
  margin-top: 12px;
  border-top: 1px solid var(--rule);
}

.legend {
  display: flex;
  gap: 18px;
  flex-wrap: wrap;
  padding-top: 16px;
  margin-top: 18px;
  border-top: 1px solid var(--rule);
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: 0.78rem;
  font-weight: 500;
  color: var(--ink);
}

.legend-swatch {
  width: 22px;
  height: 3px;
  border-radius: 2px;
}

/* ---------- Summary ---------- */

.summary {
  padding: 4px 0 20px;
  border-bottom: 1px solid var(--rule);
  margin-bottom: 20px;
}

.summary-line {
  font-family: var(--font-doc);
  font-size: 1.28rem;
  line-height: 1.4;
  color: var(--ink);
  margin: 0 0 6px;
  text-wrap: balance;
}

.summary-meta {
  font-family: var(--font-data);
  font-size: 0.74rem;
  color: var(--ink-faint);
  margin: 0;
}

/* ---------- Findings ---------- */

/* The findings are an <ol> only for semantics; the visible number lives in
   .finding-num, so the native marker has to be suppressed explicitly. */
.findings,
.findings > li {
  list-style: none !important;
}

.findings {
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
}

.finding {
  padding: 18px 0 18px 18px;
  border-left: 2px solid var(--ink);
  border-bottom: 1px solid var(--rule);
}

.finding:last-child { border-bottom: none; padding-bottom: 4px; }

.finding-head {
  display: flex;
  gap: 12px;
  align-items: baseline;
  margin-bottom: 10px;
}

.finding-num {
  font-family: var(--font-data);
  font-size: 0.74rem;
  color: var(--ink);
  border-bottom: 1.5px solid var(--ink);
  padding-bottom: 1px;
  flex-shrink: 0;
}

.finding-title {
  font-family: var(--font-ui);
  font-size: 0.92rem;
  font-weight: 600;
  color: var(--ink);
  line-height: 1.35;
}

.finding-family {
  font-size: 0.76rem;
  color: var(--ink-faint);
  margin-top: 1px;
}

.finding-quote {
  margin: 0 0 10px;
  padding: 10px 14px;
  background: var(--paper-sunk);
  border-radius: 3px;
  font-family: var(--font-doc);
  font-size: 0.98rem;
  line-height: 1.62;
  color: var(--ink-soft);
}

.finding-detail {
  margin: 0;
  font-size: 0.85rem;
  line-height: 1.6;
  color: var(--ink-soft);
  max-width: 62ch;
}

/* ---------- Empty / notice ---------- */

.empty {
  padding: 40px 0;
  max-width: 46ch;
}

.empty-title {
  font-family: var(--font-doc);
  font-size: 1.16rem;
  font-weight: 500;
  color: var(--ink);
  margin: 0 0 8px;
}

.empty-body {
  font-size: 0.88rem;
  line-height: 1.62;
  color: var(--ink-soft);
  margin: 0;
}

.notice {
  border-left: 2px solid var(--accent);
  padding-left: 18px;
}
.notice-error .empty-title { color: var(--accent); }

/* ---------- Ledger ---------- */

.ledger-intro {
  max-width: 62ch;
  margin: 0 0 32px;
}

.ledger-intro h2 {
  font-family: var(--font-doc);
  font-size: 1.4rem;
  font-weight: 600;
  letter-spacing: -0.01em;
  margin: 0 0 8px;
  color: var(--ink);
}

.ledger-intro p {
  font-size: 0.9rem;
  line-height: 1.65;
  color: var(--ink-soft);
  margin: 0;
}

.gr-dataframe {
  border: 1px solid var(--rule) !important;
  border-radius: var(--radius) !important;
  overflow: hidden;
}

.gr-dataframe table {
  font-family: var(--font-data) !important;
  font-size: 0.8rem !important;
  font-variant-numeric: tabular-nums;
}

.gr-dataframe th {
  background: var(--paper-sunk) !important;
  color: var(--ink-soft) !important;
  font-weight: 500 !important;
  border-bottom: 1px solid var(--rule-strong) !important;
}

.gr-dataframe td {
  border-bottom: 1px solid var(--rule) !important;
  color: var(--ink-soft) !important;
}

.hpo-block {
  margin-top: 40px;
  padding-top: 32px;
  border-top: 1px solid var(--rule);
}

.hpo-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: var(--gap);
  flex-wrap: wrap;
  margin-bottom: 20px;
}

.hpo-head h3 {
  font-family: var(--font-doc);
  font-size: 1.12rem;
  font-weight: 600;
  margin: 0;
  color: var(--ink);
}

.hpo-head p {
  font-size: 0.82rem;
  color: var(--ink-faint);
  margin: 0;
}

.compare-grid > div { min-width: 0; }

/* Four equal lanes on desktop so every model gets the same reading width;
   they stay equal as the viewport narrows. */
.compare-grid {
  display: grid !important;
  grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
  gap: 20px !important;
  align-items: start !important;
}

.compare-grid > * { min-width: 0 !important; }

@media (max-width: 1100px) {
  .compare-grid { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
}

@media (max-width: 640px) {
  .compare-grid { grid-template-columns: minmax(0, 1fr) !important; }
}

/* The comparison panels use the same HighlightedText treatment as the main
   workspace, so they need the same wrapping fix even though they do not carry
   the .highlighted-text elem class. */
.compare-grid .token,
.compare-grid .token-container,
.compare-grid .text,
.compare-grid .textfield,
.compare-grid span[class*="token"] {
  white-space: normal !important;
  word-break: normal !important;
  overflow-wrap: break-word !important;
  font-family: var(--font-doc) !important;
  font-size: 0.95rem !important;
  line-height: 1.7 !important;
}

.compare-hint {
  font-size: 0.78rem;
  color: var(--ink-faint);
  margin: 0 0 12px;
}

.model-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 10px;
  padding-bottom: 8px;
  margin-bottom: 10px;
  border-bottom: 1px solid var(--rule);
}

.model-name {
  font-family: var(--font-ui);
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--ink);
}

.model-f1 {
  font-family: var(--font-data);
  font-size: 0.76rem;
  color: var(--ink-faint);
  font-variant-numeric: tabular-nums;
}

/* ---------- Responsive ---------- */

@media (max-width: 900px) {
  .gradio-container { padding: 0 16px 56px !important; }
  .masthead { padding: 28px 0 18px; margin-bottom: 24px; }
  .masthead-note { text-align: left; max-width: none; }
  .doc-pane { padding: 18px; min-height: 0; }
  .highlighted-text { font-size: 1rem !important; }
  .tabs > .tab-wrapper > .tab-container { gap: 18px !important; }
}

/* ---------- Motion ---------- */

@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; animation: none !important; }
}
"""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

gr_version_str = getattr(gr, "__version__", "4.0.0")
gr_major = int(gr_version_str.split(".")[0]) if gr_version_str and gr_version_str[0].isdigit() else 4

# Paper surface, reviewer's red ink, and the same tokens applied to Gradio's
# dark variants so the app reads identically whichever mode the browser picks.
PAPER = "#fbfaf8"
SURFACE = "#ffffff"
INK = "#1c1a17"
INK_SOFT = "#56514a"
INK_FAINT = "#6f6a60"
RULE = "#e5e1da"
RULE_STRONG = "#d3cec4"
SUNK = "#f4f2ee"
ACCENT = "#a4262c"
ACCENT_HOVER = "#8f1f24"


def build_theme():
    """Configure Gradio's token set for both light and dark appearances.

    Gradio 6 ships component stylesheets after the custom CSS block, so
    setting the theme at construction time is what actually wins the cascade.
    """
    theme = gr.themes.Base()
    ramp = {
        "neutral_50": SURFACE,
        "neutral_100": SUNK,
        "neutral_200": RULE,
        "neutral_300": RULE_STRONG,
        "neutral_400": "#b5afa4",
        "neutral_500": INK_FAINT,
        "neutral_600": "#6f6a60",
        "neutral_700": INK_SOFT,
        "neutral_800": "#3a3630",
        "neutral_900": "#2a2724",
        "neutral_950": INK,
    }
    for name, value in ramp.items():
        setattr(theme, name, value)
        dark_name = f"{name}_dark"
        if hasattr(theme, dark_name):
            setattr(theme, dark_name, value)

    pairs = {
        "body_background_fill": PAPER,
        "body_text_color": INK,
        "body_text_color_subdued": INK_SOFT,
        "background_fill_primary": PAPER,
        "background_fill_secondary": SUNK,
        "block_background_fill": SURFACE,
        "block_background_fill_soft": SUNK,
        "block_border_color": RULE,
        "block_border_color_soft": RULE,
        "input_background_fill": SURFACE,
        "input_background_fill_hover": SURFACE,
        "input_background_fill_focus": SURFACE,
        "input_border_color": RULE_STRONG,
        "input_border_color_hover": INK_FAINT,
        "button_primary_background_fill": ACCENT,
        "button_primary_background_fill_hover": ACCENT_HOVER,
        "button_primary_text_color": SURFACE,
        "button_secondary_background_fill": SURFACE,
        "button_secondary_background_fill_hover": SUNK,
        "button_secondary_text_color": INK,
        "button_secondary_border_color": RULE_STRONG,
        "table_background_fill": SURFACE,
        "table_even_background_fill": PAPER,
        "table_odd_background_fill": SURFACE,
        "color_accent": ACCENT,
        "color_accent_soft": "#fdf3f3",
        "checkbox_background_fill": SURFACE,
        "block_label_text_color": INK,
        "block_info_text_color": INK_FAINT,
    }
    for name, value in pairs.items():
        if hasattr(theme, name):
            setattr(theme, name, value)
        dark_name = f"{name}_dark"
        if hasattr(theme, dark_name):
            setattr(theme, dark_name, value)

    for name, value in {
        "block_label_text_size": "0.78rem",
        "block_label_text_weight": 500,
        "block_info_text_size": "0.74rem",
        "block_info_text_weight": 400,
    }.items():
        if hasattr(theme, name):
            setattr(theme, name, value)
        dark_name = f"{name}_dark"
        if hasattr(theme, dark_name):
            setattr(theme, dark_name, value)

    return theme


theme = build_theme()

if gr_major >= 6:
    blocks_kwargs = {}
    launch_kwargs = {"theme": theme, "css": CUSTOM_CSS}
else:
    blocks_kwargs = {"theme": theme, "css": CUSTOM_CSS}
    launch_kwargs = {}

with gr.Blocks(**blocks_kwargs) as demo:

    gr.HTML(f"<style>{CUSTOM_CSS}</style>", visible=False)

    # Masthead — identity only, no telemetry theatre
    gr.HTML("""
    <header class="masthead">
      <div class="masthead-brand">
        <span class="wordmark">Gotcha</span>
        <span class="wordmark-sub">clause extractor</span>
      </div>
      <p class="masthead-note">Reads a contract and marks the clauses that cost you rights.</p>
    </header>
    """)

    with gr.Tabs():

        # -------------------------------------------------------------------
        # Tab 1 — the working surface
        # -------------------------------------------------------------------
        with gr.TabItem("Review a contract"):

            with gr.Row(elem_classes=["preset-bar"]):
                gr.HTML('<span class="preset-label">Load an example</span>')
                btn_arb = gr.Button("Forced arbitration", size="sm", elem_classes=["preset-btn"])
                btn_surv = gr.Button("Data brokerage", size="sm", elem_classes=["preset-btn"])
                btn_mut = gr.Button("Unilateral change", size="sm", elem_classes=["preset-btn"])
                btn_ind = gr.Button("Indemnification", size="sm", elem_classes=["preset-btn"])
                btn_safe = gr.Button("Clean policy", size="sm", elem_classes=["preset-btn"])

            with gr.Row():
                # Input column
                with gr.Column(scale=5):
                    text_input = gr.Textbox(
                        lines=15,
                        label="Contract text",
                        placeholder="Paste the terms of service, privacy policy, or EULA to review...",
                        elem_classes=["contract-input"]
                    )
                    # Stacked rather than side by side: at this column width the
                    # slider label wrapped to three lines and the two controls
                    # looked like competing peers instead of settings.
                    with gr.Row(elem_classes=["control-row"]):
                        model_dropdown = gr.Dropdown(
                            choices=AVAILABLE_MODELS,
                            value="electra-small",
                            label="Model",
                            info="Classifier to run"
                        )
                        min_tokens_slider = gr.Slider(
                            minimum=1,
                            maximum=5,
                            step=1,
                            value=3,
                            label="Sensitivity",
                            info="Risk sub-words to flag"
                        )
                    analyze_btn = gr.Button("Review contract", variant="primary", elem_classes=["btn-primary"])

                # Output column
                with gr.Column(scale=7):
                    summary_output = gr.HTML(empty_state(False))
                    findings_output = gr.HTML("")
                    annotated = gr.HighlightedText(
                        interactive=False,
                        combine_adjacent=False,
                        show_whitespaces=False,
                        show_legend=False,
                        show_inline_category=False,
                        visible=False,
                        color_map=COLOR_MAP,
                        elem_classes=["highlighted-text"]
                    )
                    # The key uses the same severity ink as the finding rails, so
                    # the legend and the marks it explains cannot drift apart.
                    annotated_label = gr.HTML(f"""
                    <div class="legend legend-key">
                      <div class="legend-item"><span class="legend-swatch" style="background:{RISK_INK['HIGH RISK']}"></span>High</div>
                      <div class="legend-item"><span class="legend-swatch" style="background:{RISK_INK['MEDIUM RISK']}"></span>Medium</div>
                      <div class="legend-item"><span class="legend-swatch" style="background:{RISK_INK['LOW RISK']}"></span>Low</div>
                    </div>
                    """, visible=False)

            btn_arb.click(lambda: PRESET_CASES["arbitration"], outputs=text_input)
            btn_surv.click(lambda: PRESET_CASES["surveillance"], outputs=text_input)
            btn_mut.click(lambda: PRESET_CASES["modification"], outputs=text_input)
            btn_ind.click(lambda: PRESET_CASES["indemnity"], outputs=text_input)
            btn_safe.click(lambda: PRESET_CASES["safe"], outputs=text_input)

            analyze_btn.click(
                fn=analyze_single,
                inputs=[text_input, model_dropdown, min_tokens_slider],
                outputs=[annotated, summary_output, findings_output, annotated_label, annotated]
            )

        # -------------------------------------------------------------------
        # Tab 2 — model ledger
        # -------------------------------------------------------------------
        with gr.TabItem("Model ledger"):

            gr.HTML("""
            <div class="ledger-intro">
              <h2>Four classifiers, measured</h2>
              <p>Each model was fine-tuned on the same legal corpus with inverse-sqrt class weighting,
              then tuned with Optuna. Scores below are from the held-out test split, which contains
              human-labeled data only.</p>
            </div>
            """)

            ledger_rows = []
            for m in AVAILABLE_MODELS:
                s = MODELS_SUMMARY_MAP.get(m, {})
                ledger_rows.append([
                    s.get("name", m),
                    s.get("params", "—"),
                    fmt_pct(s.get("test_precision")),
                    fmt_pct(s.get("test_recall")),
                    fmt_pct(s.get("test_f1")),
                    fmt_pct(s.get("test_accuracy")),
                    str(s.get("total_trials", 0)),
                ])

            gr.Dataframe(
                value=ledger_rows,
                headers=["Model", "Parameters", "Precision", "Recall", "F1", "Accuracy", "Optuna trials"],
                datatype=["str", "str", "str", "str", "str", "str", "str"],
                interactive=False,
                wrap=True
            )

            if len(METRICS_DF) > 0:
                with gr.Row():
                    with gr.Column():
                        gr.LinePlot(
                            value=METRICS_DF,
                            x="Epoch",
                            y="Validation F1",
                            color="Model",
                            title="Validation F1 by epoch",
                            tooltip=["Model", "Epoch", "Validation F1"]
                        )
                    with gr.Column():
                        gr.LinePlot(
                            value=METRICS_DF,
                            x="Epoch",
                            y="Training loss",
                            color="Model",
                            title="Training loss by epoch",
                            tooltip=["Model", "Epoch", "Training loss"]
                        )

            with gr.Column(elem_classes=["hpo-block"]):
                gr.HTML("""
                <div class="hpo-head">
                  <h3>Optuna trials</h3>
                  <p>Ranked by validation F1</p>
                </div>
                """)

                hpo_model_select = gr.Dropdown(
                    choices=AVAILABLE_MODELS,
                    value="electra-small",
                    label="Model",
                    info="Inspect a single model's hyperparameter search"
                )

                hpo_trials_table = gr.Dataframe(
                    value=OPTUNA_TRIALS_MAP.get("electra-small", pd.DataFrame()),
                    interactive=False,
                    wrap=True,
                    visible=True
                )

                def on_hpo_model_change(m_key):
                    s = MODELS_SUMMARY_MAP.get(m_key, {})
                    hp = s.get("best_hp", {})
                    lr = hp.get("learning_rate")
                    wd = hp.get("weight_decay")
                    frozen = hp.get("num_frozen_layers", 0)
                    line = (
                        f"{s.get('name', m_key)} · learning rate "
                        f"{f'{lr:.2e}' if isinstance(lr, (int, float)) else '—'} · "
                        f"weight decay {f'{wd:.4f}' if isinstance(wd, (int, float)) else '—'} · "
                        f"{frozen} frozen layers · {s.get('total_trials', 0)} trials"
                    )
                    return line, OPTUNA_TRIALS_MAP.get(m_key, pd.DataFrame())

                hpo_subline = gr.Markdown(value=on_hpo_model_change("electra-small")[0])

                hpo_model_select.change(
                    fn=on_hpo_model_change,
                    inputs=[hpo_model_select],
                    outputs=[hpo_subline, hpo_trials_table]
                )

        # -------------------------------------------------------------------
        # Tab 3 — cross-model comparison
        # -------------------------------------------------------------------
        with gr.TabItem("Compare models"):

            gr.HTML("""
            <div class="ledger-intro">
              <h2>Same text, four opinions</h2>
              <p>Run one contract through every fine-tuned backbone to see where they disagree.
              ELECTRA-Small leads on F1; BERT-Tiny trades precision for the highest recall.</p>
            </div>
            """)

            comp_text_input = gr.Textbox(
                lines=5,
                label="Contract text",
                value="We reserve the right to modify these terms at any time without notice. In the event "
                      "of a dispute, you waive your right to a class action lawsuit and agree to binding arbitration.",
            )

            with gr.Row(elem_classes=["control-row"]):
                comp_tokens_slider = gr.Slider(
                    minimum=1,
                    maximum=5,
                    step=1,
                    value=3,
                    label="Sensitivity",
                    info="Risk sub-words to flag"
                )
                compare_btn = gr.Button("Run all four models", variant="primary", elem_classes=["btn-primary"])

            def compare_cards(text, min_tokens):
                panels = compare_all(text, min_tokens)
                shown = [gr.update(visible=True)] * len(COMPARE_PANELS)
                return *panels, *shown

            gr.HTML('<p class="compare-hint">Run the benchmark to populate these panels.</p>')

            # Panels start hidden so the resting state is a sentence rather
            # than four empty frames, and reveal together after a run.
            compare_outputs = []
            with gr.Row(elem_classes=["compare-grid"]):
                for name, f1 in COMPARE_PANELS:
                    with gr.Column():
                        gr.HTML(
                            f'<div class="model-head"><span class="model-name">{name}</span>'
                            f'<span class="model-f1">F1 {f1}</span></div>'
                        )
                        compare_outputs.append(
                            gr.HighlightedText(
                                interactive=False,
                                combine_adjacent=False,
                                show_inline_category=False,
                                show_whitespaces=False,
                                show_legend=False,
                                visible=False,
                                color_map=COLOR_MAP,
                                elem_classes=["highlighted-text"],
                            )
                        )

            comparison_df = gr.Dataframe(
                headers=["Model", "Validation F1 (Best)", "Parameters", "Disk Size", "Risks Detected", "Latency (ms)"],
                datatype=["str", "str", "str", "str", "number", "str"],
                label="Benchmark metrics",
                interactive=False,
                wrap=True
            )

            compare_btn.click(
                fn=compare_cards,
                inputs=[comp_text_input, comp_tokens_slider],
                outputs=[*compare_outputs, comparison_df, *compare_outputs],
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, **launch_kwargs)