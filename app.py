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

        hw = data.get("hardware_telemetry", {})
        perf = data.get("performance_telemetry", {})
        peak_vram_gb = hw.get("peak_vram_reserved_gb") or hw.get("peak_vram_gb")
        avg_speed = perf.get("avg_samples_per_sec")
        formatted_duration = perf.get("formatted_duration")

        trial_rows = []
        for t_num, t_info in trials.items():
            params = t_info.get("params", {})
            f1_val = t_info.get("best_f1")
            if f1_val is None and t_info.get("f1"):
                f1_val = max(t_info["f1"])

            t_vram = t_info.get("peak_vram_gb")
            if t_vram is None and t_info.get("peak_vram_mb"):
                t_vram = t_info["peak_vram_mb"] / 1024
            t_speed = t_info.get("avg_samples_per_sec")

            trial_rows.append({
                "Trial": t_num,
                "Val F1": f"{f1_val:.3f}" if isinstance(f1_val, (int, float)) else "—",
                "Learning rate": f"{params['learning_rate']:.2e}" if isinstance(params.get("learning_rate"), (int, float)) else "—",
                "Weight decay": f"{params['weight_decay']:.4f}" if isinstance(params.get("weight_decay"), (int, float)) else "—",
                "Batch": str(params.get("per_device_train_batch_size", "—")),
                "Warmup": f"{params['warmup_ratio'] * 100:.0f}%" if isinstance(params.get("warmup_ratio"), (int, float)) else "—",
                "Frozen": str(params.get("num_frozen_layers", "—")),
                "Peak VRAM": f"{t_vram:.2f} GB" if isinstance(t_vram, (int, float)) and t_vram > 0 else "—",
                "Speed": f"{t_speed:.0f} s/s" if isinstance(t_speed, (int, float)) and t_speed > 0 else "—",
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
            "peak_vram_gb": peak_vram_gb,
            "avg_speed": avg_speed,
            "duration": formatted_duration,
            "device_name": hw.get("device_name", "GPU"),
        }

        epochs = final_run.get("epochs", [])
        f1s = final_run.get("f1", [])
        train_losses = final_run.get("train_loss", [])
        eval_losses = final_run.get("eval_loss", final_run.get("loss", []))
        vram_peaks = final_run.get("vram_peak_gb", [])
        for i in range(len(epochs)):
            history_rows.append({
                "Model": m,
                "Epoch": epochs[i],
                "Validation F1": f1s[i] if i < len(f1s) else None,
                "Training loss": train_losses[i] if i < len(train_losses) else None,
                "Validation loss": eval_losses[i] if i < len(eval_losses) else None,
                "VRAM (GB)": vram_peaks[i] if i < len(vram_peaks) else None,
            })

    metrics_df = pd.DataFrame(history_rows)
    if not history_rows:
        metrics_df = pd.DataFrame(columns=["Model", "Epoch", "Validation F1", "Training loss", "Validation loss", "VRAM (GB)"])
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
    if any(k in s for k in ("train our", "train proprietary", "train commercial", "foundation model", "generative ai", "machine learning model", "neural network", "diffusion model", "model pre-training")):
        return {
            "title": "AI model training grab",
            "family": "Intellectual property",
            "detail": "Appropriates your uploaded content, prompts, or code to train and monetize commercial foundation AI models without compensation."
        }
    if any(k in s for k in ("arbitrat", "class action", "jury", "court proceeding", "dispute", "bellwether", "batched in")):
        return {
            "title": "Forced arbitration",
            "family": "Dispute resolution",
            "detail": "Removes your access to court and to collective action, and routes disputes through private arbitration or batching protocols."
        }
    if any(k in s for k in ("biometric", "facial geometry", "voiceprint", "keystroke dynamics", "eye-tracking")):
        return {
            "title": "Biometric surveillance",
            "family": "Privacy",
            "detail": "Monetizes sensitive biometric identifiers, voiceprints, or behavioral keystroke telemetry with third-party brokers."
        }
    if any(k in s for k in ("sell", "broker", "advertis", "third part", "market your", "telemetry", "location data")):
        return {
            "title": "Data brokerage",
            "family": "Privacy",
            "detail": "Authorizes collecting and syndicating your activity to outside advertisers and data brokers."
        }
    if any(k in s for k in ("automatically renew", "auto-renew", "non-refundable", "certified postal mail", "cancellation consultation")):
        return {
            "title": "Predatory auto-renewal",
            "family": "Billing & terms",
            "detail": "Traps you in automatic renewal charges with strict non-refundability or onerous manual cancellation requirements."
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
    """Render flagged clauses as modern, structured legal risk cards."""
    flagged = [(seg.strip(), label) for seg, label in results if label and seg.strip()]
    if not flagged:
        return ""
    flagged.sort(key=lambda item: SEVERITY_ORDER.get(item[1], 3))

    items = []
    for i, (segment, label) in enumerate(flagged, 1):
        info = categorize_gotcha(segment)
        sev_slug = label.split()[0].lower()  # "high", "medium", "low"
        items.append(f"""
        <div class="finding-card finding-{sev_slug}">
          <div class="finding-card-top">
            <div class="finding-badge-row">
              <span class="sev-pill sev-pill-{sev_slug}">{label}</span>
              <span class="family-tag">{info['family']}</span>
            </div>
            <span class="finding-num-tag">Issue #{i:02d}</span>
          </div>
          <h4 class="finding-title">{info['title']}</h4>
          <div class="finding-quote-box">
            <blockquote class="finding-quote">"{segment}"</blockquote>
          </div>
          <div class="finding-impact-box">
            <span class="impact-lead">Rights & Liability Impact:</span> {info['detail']}
          </div>
        </div>
        """)
    return "".join(items)


def render_findings_block(results):
    """Findings section with structured legal risk cards."""
    flagged = [(seg.strip(), label) for seg, label in results if label and seg.strip()]
    if not flagged:
        return ""
    count = len(flagged)
    plural = "" if count == 1 else "s"
    head = (
        f'<div class="findings-header">'
        f'<div class="findings-title-group">'
        f'<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>'
        f'<span class="findings-heading">Legal Risk Breakdown & Rights Impact</span>'
        f'</div>'
        f'<span class="findings-count-pill">{count} Clause{plural} Analyzed</span>'
        f'</div>'
    )
    body = render_findings(results)
    return f'<div class="findings-section">{head}<div class="findings-list">{body}</div></div>'


def render_summary(counts, elapsed_ms):
    """Executive Audit summary bar with severity badges and live telemetry."""
    total = counts["high"] + counts["medium"] + counts["low"]
    if total == 0:
        return ""

    chips = []
    if counts["high"]:
        chips.append(f'<span class="sev-chip sev-chip-high"><span class="chip-dot">●</span> {counts["high"]} High Risk</span>')
    if counts["medium"]:
        chips.append(f'<span class="sev-chip sev-chip-med"><span class="chip-dot">●</span> {counts["medium"]} Medium</span>')
    if counts["low"]:
        chips.append(f'<span class="sev-chip sev-chip-low"><span class="chip-dot">●</span> {counts["low"]} Low</span>')

    chips_html = "".join(chips)
    device = get_inference_device().type.upper()

    return f"""
    <div class="audit-summary-bar">
      <div class="summary-left">
        <div class="summary-status-badge status-alert">
          <svg class="summary-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
          <span>{total} Flagged Clause{'' if total == 1 else 's'}</span>
        </div>
        <div class="sev-chip-group">
          {chips_html}
        </div>
      </div>
      <div class="summary-right">
        <span class="telemetry-pill">⚡ {elapsed_ms:.0f} ms · {device}</span>
      </div>
    </div>
    """


def empty_state(has_text):
    if not has_text:
        return """
        <div class="empty-state-card">
          <div class="empty-icon-wrap">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
              <polyline points="14 2 14 8 20 8"></polyline>
              <line x1="16" y1="13" x2="8" y2="13"></line>
              <line x1="16" y1="17" x2="8" y2="17"></line>
              <polyline points="10 9 9 9 8 9"></polyline>
            </svg>
          </div>
          <h3 class="empty-title">Ready for Contract Audit</h3>
          <p class="empty-body">Paste any Terms of Service, Privacy Policy, or EULA on the left, or select a sample agreement above. Neural token classifiers will audit every clause and highlight predatory stipulations.</p>
          <div class="empty-tips">
            <span class="tip-item"><span class="tip-dot">●</span> 4 Fine-Tuned Models</span>
            <span class="tip-item"><span class="tip-dot">●</span> BIO Token Span Tagging</span>
            <span class="tip-item"><span class="tip-dot">●</span> Hardware Telemetry</span>
          </div>
        </div>
        """
    return """
    <div class="clean-state-card">
      <div class="clean-icon-wrap">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
          <polyline points="22 4 12 14.01 9 11.01"></polyline>
        </svg>
      </div>
      <h3 class="clean-title">No High-Risk Clauses Detected</h3>
      <p class="clean-body">All evaluated clauses fell within standard consumer agreement parameters at the current sensitivity threshold. Try lowering the sensitivity threshold to 1 or 2 if you want a more cautious sweep.</p>
    </div>
    """


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def to_highlight_pairs(results):
    """Normalize and coalesce contiguous segments with identical labels.

    Drastically minimizes DOM nodes created by Gradio HighlightedText,
    preventing UI freezes on large contracts while preserving exact text.
    """
    if not results:
        return []

    pairs = []
    for segment, label in results:
        if not segment:
            continue
        if pairs and pairs[-1][1] == label:
            pairs[-1][0] += segment
        else:
            pairs.append([segment, label])

    return [(text, label) for text, label in pairs]



def analyze_single(text, model_name, min_tokens):
    """Run the classifier and return every output the workspace needs.

    Returns (annotated_update, summary_html, findings_html, label_update).
    The updates reveal the annotated-text heading and panel together, and only
    when there is something to show.
    """
    hidden = (gr.update(value=[], visible=False), empty_state(False), "", gr.update(visible=False))

    if not text or not text.strip():
        return hidden

    start = time.time()
    try:
        results = classify_text(text, model_name=model_name, min_risk_tokens=min_tokens)
    except Exception as e:
        error = (
            f'<div class="notice notice-error"><p class="empty-title">Analysis failed</p>'
            f'<p class="empty-body">{e}</p></div>'
        )
        return gr.update(value=[], visible=False), empty_state(True), error, gr.update(visible=False)
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
        return gr.update(value=[], visible=False), empty_state(True), "", gr.update(visible=False)

    return (
        gr.update(value=to_highlight_pairs(results), visible=True),
        render_summary(counts, elapsed),
        render_findings_block(results),
        gr.update(visible=True),
    )


def compare_all(text, min_tokens):
    if not text or not text.strip():
        return [], [], [], [], pd.DataFrame()

    e, tb, bm, bt, df = gotcha_compare_models(text, min_tokens=min_tokens)
    return (
        to_highlight_pairs(e),
        to_highlight_pairs(tb),
        to_highlight_pairs(bm),
        to_highlight_pairs(bt),
        df
    )



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
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --paper:        #f8fafc;
  --surface:      #ffffff;
  --paper-sunk:   #f1f5f9;
  --rule:         #e2e8f0;
  --rule-strong:  #cbd5e1;

  --ink:          #0f172a;
  --ink-soft:     #334155;
  --ink-faint:    #64748b;

  --brand-navy:   #0f172a;
  --brand-blue:   #2563eb;
  --accent:       #0f172a;
  --focus:        #2563eb;

  --sev-high-bg:  #fef2f2;
  --sev-high-ink: #991b1b;
  --sev-high-bdr: #fecaca;
  --sev-high-bar: #ef4444;

  --sev-med-bg:   #fffbeb;
  --sev-med-ink:  #92400e;
  --sev-med-bdr:  #fde68a;
  --sev-med-bar:  #f59e0b;

  --sev-low-bg:   #f0fdf4;
  --sev-low-ink:  #166534;
  --sev-low-bdr:  #bbf7d0;
  --sev-low-bar:  #22c55e;

  --radius-lg:    12px;
  --radius-md:    8px;
  --radius-sm:    6px;

  --shadow-card:  0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.03);
  --shadow-hover: 0 4px 6px -1px rgba(0, 0, 0, 0.08), 0 2px 4px -2px rgba(0, 0, 0, 0.04);

  --font-ui:   'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --font-doc:  'Newsreader', Georgia, serif;
  --font-data: 'JetBrains Mono', ui-monospace, monospace;

  /* Gradio 6 theme variable overrides */
  --bg:                    #f8fafc;
  --col:                   #0f172a;
  --background-fill-primary:   #f8fafc;
  --body-background-fill:      #f8fafc;
  --body-text-color:           #0f172a;
  --body-text-color-subdued:   #334155;
  --body-text-color-muted:     #64748b;
  --neutral-50:               #ffffff;
  --neutral-100:              #f1f5f9;
  --neutral-200:              #e2e8f0;
  --neutral-300:              #cbd5e1;
  --neutral-500:              #64748b;
  --neutral-700:              #334155;
  --neutral-900:              #0f172a;
  --input-background-fill:    #ffffff;
  --input-background-fill-hover: #ffffff;
  --input-border-color:       #cbd5e1;
  --input-border-color-hover: #94a3b8;
  --block-background-fill:    #ffffff;
  --block-background-fill-soft: #f8fafc;
  --block-border-color:       #e2e8f0;
  --border-primary:           #e2e8f0;
  --color-accent:             #2563eb;
  --button-primary-background-fill: #0f172a;
  --button-primary-background-fill-hover: #1e293b;
  --button-primary-text-color: #ffffff;
  --button-secondary-background-fill: #ffffff;
  --button-secondary-background-fill-hover: #f1f5f9;
  --button-secondary-text-color: #0f172a;
  --button-secondary-border-color: #cbd5e1;
  --table-background-fill: #ffffff;
  --table-even-background-fill: #f8fafc;
  --table-odd-background-fill: #ffffff;
  --dataframe-selected-background-fill: #eff6ff;
}

html, body {
  background: var(--paper) !important;
  color: var(--ink) !important;
  font-family: var(--font-ui) !important;
}

body.dark, :root.dark, .dark {
  color-scheme: light;
}

:root.dark, .dark {
  --bg:                    #f8fafc;
  --col:                   #0f172a;
  --background-fill-primary:   #f8fafc;
  --body-background-fill:      #f8fafc;
  --body-text-color:           #0f172a;
  --body-text-color-subdued:   #334155;
  --body-text-color-muted:     #64748b;
  --neutral-1:                #0f172a;
  --neutral-100:              #f1f5f9;
  --neutral-200:              #e2e8f0;
  --neutral-300:              #cbd5e1;
  --neutral-500:              #64748b;
  --neutral-700:              #334155;
  --input-background-fill:    #ffffff;
  --input-background-fill-hover: #ffffff;
  --input-border-color:       #cbd5e1;
  --block-background-fill:    #ffffff;
  --block-border-color:       #e2e8f0;
  --border-primary:           #e2e8f0;
  --color-accent:             #2563eb;
  --button-primary-background-fill: #0f172a;
  --button-primary-background-fill-hover: #1e293b;
  --button-primary-text-color: #ffffff;
  --table-background-fill: #ffffff;
  --table-even-background-fill: #f8fafc;
  --table-odd-background-fill: #ffffff;
  --color-scheme: light;
}

/* Base resets & clear container */
.main, .column, .row, .form, .panel, .block,
.form > *, .block > *, .column > *, .row > * {
  background-color: transparent !important;
  border-color: transparent !important;
}

/* ---------- Shell & Layout ---------- */

.gradio-container {
  background: var(--paper) !important;
  color: var(--ink) !important;
  font-family: var(--font-ui) !important;
  max-width: none !important;
  padding: 0 32px 72px !important;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

.gradio-container > .main {
  max-width: 1440px;
  margin-inline: auto;
  width: 100%;
}

/* ---------- Masthead Navigation ---------- */

.masthead {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 16px;
  padding: 24px 0 20px;
  border-bottom: 1px solid var(--rule);
  margin-bottom: 24px;
}

.masthead-left {
  display: flex;
  align-items: center;
  gap: 14px;
}

.masthead-logo-shield {
  width: 42px;
  height: 42px;
  border-radius: 10px;
  background: #0f172a;
  color: #ffffff;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 2px 4px rgba(15, 23, 42, 0.15);
}

.masthead-title-wrap {
  display: flex;
  flex-direction: column;
}

.masthead-brand-line {
  display: flex;
  align-items: center;
  gap: 8px;
}

.brand-title {
  font-family: var(--font-ui);
  font-size: 1.3rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--ink);
}

.brand-tag {
  font-family: var(--font-ui);
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  background: #eff6ff;
  color: #1d4ed8;
  border: 1px solid #bfdbfe;
  padding: 2px 7px;
  border-radius: 4px;
  text-transform: uppercase;
}

.brand-subtitle {
  font-size: 0.82rem;
  color: var(--ink-faint);
  margin: 2px 0 0;
}

.masthead-right {
  display: flex;
  align-items: center;
}

.status-badge {
  display: flex;
  align-items: center;
  gap: 8px;
  background: #ffffff;
  border: 1px solid var(--rule);
  border-radius: 20px;
  padding: 6px 14px;
  font-size: 0.78rem;
  font-weight: 500;
  color: var(--ink-soft);
  box-shadow: var(--shadow-card);
}

.status-pulse-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #22c55e;
  box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.2);
}

/* ---------- Tabs ---------- */

.tabs > .tab-wrapper > .tab-container {
  display: flex !important;
  gap: 28px !important;
  border-bottom: 1px solid var(--rule) !important;
  margin-bottom: 24px !important;
  background: transparent !important;
  flex-wrap: wrap;
}

.tabs > .tab-wrapper > .tab-container > button {
  background: none !important;
  border: none !important;
  box-shadow: none !important;
  padding: 10px 2px 14px !important;
  border-bottom: 2px solid transparent !important;
  margin-bottom: -1px !important;
  font-family: var(--font-ui) !important;
  font-size: 0.92rem !important;
  font-weight: 500 !important;
  color: var(--ink-faint) !important;
  transition: all 120ms ease;
}

.tabs > .tab-wrapper > .tab-container > button:hover {
  color: var(--ink) !important;
}

.tabs > .tab-wrapper > .tab-container > button.selected {
  color: var(--ink) !important;
  font-weight: 600 !important;
  border-bottom-color: var(--ink) !important;
}

/* ---------- Preset Bar ---------- */

.preset-bar {
  display: flex !important;
  align-items: center !important;
  gap: 8px !important;
  flex-wrap: wrap !important;
  padding: 12px 18px !important;
  background: #ffffff !important;
  border: 1px solid var(--rule) !important;
  border-radius: var(--radius-md) !important;
  margin-bottom: 20px !important;
  box-shadow: var(--shadow-card) !important;
}

.preset-label-wrap {
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--ink-faint);
  margin-right: 6px;
}

.preset-label {
  font-size: 0.76rem;
  font-weight: 600;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--ink-faint);
  white-space: nowrap;
}

.preset-btn {
  background: #f8fafc !important;
  border: 1px solid var(--rule) !important;
  color: var(--ink-soft) !important;
  font-size: 0.8rem !important;
  font-weight: 500 !important;
  padding: 5px 12px !important;
  min-height: 0 !important;
  width: auto !important;
  flex: 0 0 auto !important;
  border-radius: 20px !important;
  box-shadow: none !important;
  transition: all 120ms ease !important;
}

.preset-btn:hover {
  background: #ffffff !important;
  border-color: var(--rule-strong) !important;
  color: var(--ink) !important;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06) !important;
  transform: translateY(-1px) !important;
}

/* ---------- Workspace Dual Cards ---------- */

.workspace-row {
  gap: 24px !important;
  align-items: stretch !important;
}

.workspace-card {
  background: var(--surface) !important;
  border: 1px solid var(--rule) !important;
  border-radius: var(--radius-lg) !important;
  padding: 24px !important;
  box-shadow: var(--shadow-card) !important;
  display: flex !important;
  flex-direction: column !important;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-bottom: 12px;
  margin-bottom: 14px;
  border-bottom: 1px solid var(--rule);
}

.card-title-group {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--ink);
}

.card-heading {
  font-family: var(--font-ui);
  font-size: 0.96rem;
  font-weight: 600;
  color: var(--ink);
  margin: 0;
}

.card-tag {
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--ink-faint);
  background: var(--paper-sunk);
  padding: 2px 8px;
  border-radius: 4px;
}

/* Textarea input */
.contract-input {
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
}

.contract-input .input-container {
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.contract-input textarea {
  flex: 1 1 auto;
  min-height: 320px;
  max-height: 480px !important;
  overflow-y: auto !important;
  background: #ffffff !important;
  border: 1px solid var(--rule-strong) !important;
  border-radius: var(--radius-md) !important;
  font-family: var(--font-doc) !important;
  font-size: 0.98rem !important;
  line-height: 1.65 !important;
  color: var(--ink) !important;
  padding: 14px 16px !important;
  transition: border-color 140ms ease, box-shadow 140ms ease !important;
  box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.02) !important;
  scrollbar-width: thin;
  scrollbar-color: #cbd5e1 #f8fafc;
}

.contract-input textarea::-webkit-scrollbar {
  width: 6px;
}

.contract-input textarea::-webkit-scrollbar-track {
  background: #f8fafc;
}

.contract-input textarea::-webkit-scrollbar-thumb {
  background: #cbd5e1;
  border-radius: 4px;
}

.contract-input textarea::-webkit-scrollbar-thumb:hover {
  background: #94a3b8;
}

.contract-input textarea:focus {
  border-color: #3b82f6 !important;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
  outline: none !important;
}

/* Controls box inside input card */
.control-box {
  background: var(--paper-sunk) !important;
  border: 1px solid var(--rule) !important;
  border-radius: var(--radius-md) !important;
  padding: 14px 16px !important;
  margin-top: 16px !important;
  margin-bottom: 16px !important;
  gap: 16px !important;
}

.control-box .wrap, .control-box .head {
  width: 100% !important;
}

.control-box label {
  font-family: var(--font-ui) !important;
  font-size: 0.78rem !important;
  font-weight: 600 !important;
  color: var(--ink-soft) !important;
  margin-bottom: 4px !important;
}

.control-box .info {
  font-size: 0.74rem !important;
  color: var(--ink-faint) !important;
}

/* Primary CTA Button */
.btn-primary {
  background: #0f172a !important;
  border: 1px solid #0f172a !important;
  color: #ffffff !important;
  font-family: var(--font-ui) !important;
  font-size: 0.92rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.01em !important;
  padding: 12px 24px !important;
  border-radius: var(--radius-md) !important;
  box-shadow: 0 2px 4px rgba(15, 23, 42, 0.12) !important;
  cursor: pointer !important;
  transition: all 140ms ease !important;
  width: 100% !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
}

.btn-primary:hover {
  background: #1e293b !important;
  border-color: #1e293b !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 8px rgba(15, 23, 42, 0.18) !important;
}

.btn-primary:active {
  transform: translateY(0) !important;
}

/* ---------- Results Panel & Document Viewer ---------- */

.audit-summary-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
  background: #ffffff;
  border: 1px solid var(--rule);
  border-radius: var(--radius-md);
  padding: 12px 18px;
  margin-bottom: 18px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
}

.summary-left {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.summary-status-badge {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.88rem;
  font-weight: 600;
  color: #0f172a;
}

.summary-icon {
  color: #dc2626;
}

.sev-chip-group {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}

.sev-chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 0.74rem;
  font-weight: 600;
  padding: 3px 9px;
  border-radius: 20px;
}

.sev-chip-high {
  background: var(--sev-high-bg);
  color: var(--sev-high-ink);
  border: 1px solid var(--sev-high-bdr);
}

.sev-chip-med {
  background: var(--sev-med-bg);
  color: var(--sev-med-ink);
  border: 1px solid var(--sev-med-bdr);
}

.sev-chip-low {
  background: var(--sev-low-bg);
  color: var(--sev-low-ink);
  border: 1px solid var(--sev-low-bdr);
}

.chip-dot {
  font-size: 0.65rem;
}

.summary-right {
  display: flex;
  align-items: center;
}

.telemetry-pill {
  font-family: var(--font-data);
  font-size: 0.76rem;
  color: var(--ink-faint);
  background: var(--paper-sunk);
  padding: 4px 10px;
  border-radius: 6px;
  border: 1px solid var(--rule);
}

/* Document Header & Legend */
.doc-viewer-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
  padding-bottom: 10px;
  margin-bottom: 12px;
  border-bottom: 1px solid var(--rule);
}

.doc-viewer-title {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.85rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--ink-soft);
}

.legend-row {
  display: flex;
  align-items: center;
  gap: 12px;
}

.legend-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 0.76rem;
  font-weight: 500;
  color: var(--ink-soft);
}

.legend-indicator {
  width: 14px;
  height: 4px;
  border-radius: 2px;
}

.legend-high .legend-indicator { background: var(--sev-high-bar); }
.legend-med .legend-indicator  { background: var(--sev-med-bar); }
.legend-low .legend-indicator  { background: var(--sev-low-bar); }

/* Document Reader Scroll Pane */
.doc-reader-scroll {
  max-height: 480px !important;
  overflow-y: auto !important;
  background: #ffffff !important;
  border: 1px solid var(--rule) !important;
  border-radius: var(--radius-md) !important;
  padding: 20px 22px !important;
  box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.02) !important;
  scrollbar-width: thin;
  scrollbar-color: #cbd5e1 #f8fafc;
}

.doc-reader-scroll::-webkit-scrollbar {
  width: 6px;
}

.doc-reader-scroll::-webkit-scrollbar-track {
  background: #f8fafc;
}

.doc-reader-scroll::-webkit-scrollbar-thumb {
  background: #cbd5e1;
  border-radius: 4px;
}

.doc-reader-scroll::-webkit-scrollbar-thumb:hover {
  background: #94a3b8;
}

/* HighlightedText Typography and Highlighting Stems */
.highlighted-text {
  font-family: var(--font-doc) !important;
  font-size: 1.02rem !important;
  line-height: 1.75 !important;
  color: #1e293b !important;
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
}

.highlighted-text span[style*="background"] {
  border-radius: 3px !important;
  padding: 2px 4px !important;
  margin: 0 1px !important;
  font-weight: 500 !important;
  box-decoration-break: clone;
  -webkit-box-decoration-break: clone;
}

/* High Risk Highlighter (Matches #fde8ea) */
.highlighted-text span[style*="fde8ea"],
.highlighted-text span[style*="253, 232, 234"] {
  background-color: var(--sev-high-bg) !important;
  color: var(--sev-high-ink) !important;
  box-shadow: inset 0 -2px 0 var(--sev-high-bar) !important;
}

/* Medium Risk Highlighter (Matches #fdf0dd) */
.highlighted-text span[style*="fdf0dd"],
.highlighted-text span[style*="253, 240, 221"] {
  background-color: var(--sev-med-bg) !important;
  color: var(--sev-med-ink) !important;
  box-shadow: inset 0 -2px 0 var(--sev-med-bar) !important;
}

/* Low Risk Highlighter (Matches #eef1f4) */
.highlighted-text span[style*="eef1f4"],
.highlighted-text span[style*="238, 241, 244"] {
  background-color: var(--sev-low-bg) !important;
  color: var(--sev-low-ink) !important;
  box-shadow: inset 0 -2px 0 var(--sev-low-bar) !important;
}

/* Wrap tokens cleanly */
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

/* ---------- Findings Section & Legal Risk Cards ---------- */

.findings-section {
  margin-top: 28px;
  padding-top: 22px;
  border-top: 1px solid var(--rule);
}

.findings-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
}

.findings-title-group {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--ink);
}

.findings-heading {
  font-family: var(--font-ui);
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--ink);
}

.findings-count-pill {
  font-size: 0.74rem;
  font-weight: 600;
  color: var(--ink-faint);
  background: var(--paper-sunk);
  padding: 3px 9px;
  border-radius: 12px;
  border: 1px solid var(--rule);
}

.findings-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.finding-card {
  background: #ffffff;
  border: 1px solid var(--rule);
  border-left-width: 4px;
  border-radius: var(--radius-md);
  padding: 16px 18px;
  box-shadow: var(--shadow-card);
  transition: all 120ms ease;
}

.finding-card:hover {
  border-color: var(--rule-strong);
  box-shadow: var(--shadow-hover);
}

.finding-card.finding-high {
  border-left-color: var(--sev-high-bar);
}

.finding-card.finding-medium {
  border-left-color: var(--sev-med-bar);
}

.finding-card.finding-low {
  border-left-color: var(--sev-low-bar);
}

.finding-card-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.finding-badge-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.sev-pill {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.03em;
  text-transform: uppercase;
  padding: 2px 8px;
  border-radius: 4px;
}

.sev-pill-high {
  background: var(--sev-high-bg);
  color: var(--sev-high-ink);
  border: 1px solid var(--sev-high-bdr);
}

.sev-pill-medium {
  background: var(--sev-med-bg);
  color: var(--sev-med-ink);
  border: 1px solid var(--sev-med-bdr);
}

.sev-pill-low {
  background: var(--sev-low-bg);
  color: var(--sev-low-ink);
  border: 1px solid var(--sev-low-bdr);
}

.family-tag {
  font-size: 0.75rem;
  color: var(--ink-faint);
  font-weight: 500;
}

.finding-num-tag {
  font-family: var(--font-data);
  font-size: 0.72rem;
  color: var(--ink-faint);
  font-weight: 500;
}

.finding-title {
  font-family: var(--font-ui);
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--ink);
  margin: 0 0 10px;
}

.finding-quote-box {
  background: #f8fafc;
  border: 1px solid #f1f5f9;
  border-radius: 6px;
  padding: 10px 14px;
  margin-bottom: 10px;
}

.finding-quote {
  font-family: var(--font-doc);
  font-size: 0.95rem;
  line-height: 1.55;
  color: #1e293b;
  margin: 0;
  font-style: normal;
}

.finding-impact-box {
  font-size: 0.82rem;
  color: #475569;
  line-height: 1.55;
}

.impact-lead {
  font-weight: 600;
  color: var(--ink);
  margin-right: 4px;
}

/* ---------- Empty State & Clean State Cards ---------- */

.empty-state-card, .clean-state-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 48px 32px;
  background: #f8fafc;
  border: 1px dashed var(--rule-strong);
  border-radius: var(--radius-md);
  margin-bottom: 12px;
}

.empty-icon-wrap, .clean-icon-wrap {
  width: 56px;
  height: 56px;
  border-radius: 50%;
  background: #ffffff;
  border: 1px solid var(--rule);
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 16px;
  color: var(--ink-faint);
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}

.clean-icon-wrap {
  background: #f0fdf4;
  border-color: #bbf7d0;
  color: #16a34a;
}

.empty-title, .clean-title {
  font-family: var(--font-ui);
  font-size: 1.12rem;
  font-weight: 600;
  color: var(--ink);
  margin: 0 0 8px;
}

.empty-body, .clean-body {
  font-family: var(--font-ui);
  font-size: 0.88rem;
  color: var(--ink-faint);
  line-height: 1.6;
  max-width: 48ch;
  margin: 0 0 20px;
}

.empty-tips {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  justify-content: center;
}

.tip-item {
  font-size: 0.76rem;
  font-weight: 500;
  color: var(--ink-soft);
  background: #ffffff;
  border: 1px solid var(--rule);
  padding: 4px 10px;
  border-radius: 20px;
}

.tip-dot {
  color: #2563eb;
  font-size: 0.7rem;
  margin-right: 4px;
}

/* ---------- Multi-Model Comparison Grid ---------- */

.compare-grid {
  display: grid !important;
  grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
  gap: 20px !important;
  align-items: start !important;
  margin-top: 14px !important;
}

.compare-grid > * { min-width: 0 !important; }

@media (max-width: 860px) {
  .compare-grid { grid-template-columns: minmax(0, 1fr) !important; }
}

.compare-card {
  background: #ffffff !important;
  border: 1px solid var(--rule) !important;
  border-radius: var(--radius-md) !important;
  padding: 16px 18px !important;
  box-shadow: var(--shadow-card) !important;
}

.compare-reader-scroll {
  max-height: 420px !important;
  overflow-y: auto !important;
  background: #ffffff !important;
  padding: 4px 0 !important;
  scrollbar-width: thin;
  scrollbar-color: #cbd5e1 #f8fafc;
}

.compare-reader-scroll::-webkit-scrollbar {
  width: 6px;
}

.compare-reader-scroll::-webkit-scrollbar-track {
  background: #f8fafc;
}

.compare-reader-scroll::-webkit-scrollbar-thumb {
  background: #cbd5e1;
  border-radius: 4px;
}

.compare-reader-scroll::-webkit-scrollbar-thumb:hover {
  background: #94a3b8;
}

.compare-hint {
  font-family: var(--font-ui);
  font-size: 0.86rem;
  font-weight: 500;
  color: var(--ink-faint);
  margin: 18px 0 8px;
}

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
  font-size: 0.86rem;
  font-weight: 600;
  color: var(--ink);
}

.model-f1 {
  font-family: var(--font-data);
  font-size: 0.76rem;
  color: var(--ink-faint);
  font-variant-numeric: tabular-nums;
}

/* Hide Gradio internal progress indicators */
.progress-text, [class*="progress-text"], .wrap.full.translucent {
  display: none !important;
}

/* Responsive */
@media (max-width: 900px) {
  .gradio-container { padding: 0 16px 56px !important; }
  .workspace-row { flex-direction: column !important; }
  .masthead { padding: 20px 0 16px; margin-bottom: 20px; }
  .contract-input textarea { min-height: 220px !important; }
}

@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; animation: none !important; }
}
"""


# ---------------------------------------------------------------------------
# UI Theme & Tokens
# ---------------------------------------------------------------------------

gr_version_str = getattr(gr, "__version__", "4.0.0")
gr_major = int(gr_version_str.split(".")[0]) if gr_version_str and gr_version_str[0].isdigit() else 4

PAPER = "#f8fafc"
SURFACE = "#ffffff"
INK = "#0f172a"
INK_SOFT = "#334155"
INK_FAINT = "#64748b"
RULE = "#e2e8f0"
RULE_STRONG = "#cbd5e1"
SUNK = "#f1f5f9"
ACCENT = "#0f172a"
ACCENT_HOVER = "#1e293b"


def build_theme():
    """Configure Gradio's token set with modern slate legal-tech appearance."""
    theme = gr.themes.Base()
    ramp = {
        "neutral_50": SURFACE,
        "neutral_100": SUNK,
        "neutral_200": RULE,
        "neutral_300": RULE_STRONG,
        "neutral_400": "#94a3b8",
        "neutral_500": INK_FAINT,
        "neutral_600": "#475569",
        "neutral_700": INK_SOFT,
        "neutral_800": "#1e293b",
        "neutral_900": INK,
        "neutral_950": "#020617",
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
        "color_accent": "#2563eb",
        "color_accent_soft": "#eff6ff",
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
        "block_label_text_weight": 600,
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

    # Masthead — modern legal auditor navigation
    gr.HTML("""
    <header class="masthead">
      <div class="masthead-left">
        <div class="masthead-logo-shield">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
          </svg>
        </div>
        <div class="masthead-title-wrap">
          <div class="masthead-brand-line">
            <span class="brand-title">GOTCHA</span>
            <span class="brand-tag">LEGAL AUDITOR</span>
          </div>
          <p class="brand-subtitle">Automated Terms of Service & EULA Predatory Clause Extractor</p>
        </div>
      </div>
      <div class="masthead-right">
        <div class="status-badge">
          <span class="status-pulse-dot"></span>
          <span class="status-badge-text">4 Fine-Tuned Models Online</span>
        </div>
      </div>
    </header>
    """)

    with gr.Tabs():

        # -------------------------------------------------------------------
        # Tab 1 — the working surface
        # -------------------------------------------------------------------
        with gr.TabItem("Review a contract"):

            with gr.Row(elem_classes=["preset-bar"]):
                gr.HTML("""
                <div class="preset-label-wrap">
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
                  <span class="preset-label">Sample Agreements</span>
                </div>
                """)
                btn_arb = gr.Button("Forced Arbitration", size="sm", elem_classes=["preset-btn"])
                btn_surv = gr.Button("Data Brokerage", size="sm", elem_classes=["preset-btn"])
                btn_mut = gr.Button("Unilateral Change", size="sm", elem_classes=["preset-btn"])
                btn_ind = gr.Button("Indemnification", size="sm", elem_classes=["preset-btn"])
                btn_safe = gr.Button("Clean Policy", size="sm", elem_classes=["preset-btn"])

            with gr.Row(elem_classes=["workspace-row"]):
                # Input column
                with gr.Column(scale=5, elem_classes=["workspace-card", "input-card"]):
                    gr.HTML("""
                    <div class="card-header">
                      <div class="card-title-group">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
                        <h3 class="card-heading">Contract Text</h3>
                      </div>
                      <span class="card-tag">ToS / EULA</span>
                    </div>
                    """)
                    text_input = gr.Textbox(
                        lines=18,
                        label="Contract text",
                        show_label=False,
                        placeholder="Paste terms of service, privacy policy, or EULA text here...",
                        elem_classes=["contract-input"]
                    )
                    with gr.Row(elem_classes=["control-box"]):
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
                            info="Risk token threshold"
                        )
                    analyze_btn = gr.Button("Review Contract", variant="primary", elem_classes=["btn-primary"])

                # Output column
                with gr.Column(scale=7, elem_classes=["workspace-card", "results-col"]):
                    summary_output = gr.HTML(empty_state(False))
                    annotated_label = gr.HTML(f"""
                    <div class="doc-viewer-header">
                      <div class="doc-viewer-title">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                        <span>Marked Agreement Document</span>
                      </div>
                      <div class="legend-row">
                        <span class="legend-chip legend-high"><span class="legend-indicator"></span>High Risk</span>
                        <span class="legend-chip legend-med"><span class="legend-indicator"></span>Medium</span>
                        <span class="legend-chip legend-low"><span class="legend-indicator"></span>Low</span>
                      </div>
                    </div>
                    """, visible=False)
                    annotated = gr.HighlightedText(
                        interactive=False,
                        combine_adjacent=False,
                        show_whitespaces=False,
                        show_legend=False,
                        show_inline_category=False,
                        visible=False,
                        color_map=COLOR_MAP,
                        elem_classes=["highlighted-text", "doc-reader-scroll"]
                    )
                    findings_output = gr.HTML("")

            btn_arb.click(lambda: PRESET_CASES["arbitration"], outputs=text_input)
            btn_surv.click(lambda: PRESET_CASES["surveillance"], outputs=text_input)
            btn_mut.click(lambda: PRESET_CASES["modification"], outputs=text_input)
            btn_ind.click(lambda: PRESET_CASES["indemnity"], outputs=text_input)
            btn_safe.click(lambda: PRESET_CASES["safe"], outputs=text_input)

            analyze_btn.click(
                fn=analyze_single,
                inputs=[text_input, model_dropdown, min_tokens_slider],
                outputs=[annotated, summary_output, findings_output, annotated_label]
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
                vram_str = f"{s['peak_vram_gb']:.2f} GB" if isinstance(s.get("peak_vram_gb"), (int, float)) and s["peak_vram_gb"] > 0 else "—"
                speed_str = f"{s['avg_speed']:.0f} s/s" if isinstance(s.get("avg_speed"), (int, float)) and s["avg_speed"] > 0 else "—"
                ledger_rows.append([
                    s.get("name", m),
                    s.get("params", "—"),
                    fmt_pct(s.get("test_precision")),
                    fmt_pct(s.get("test_recall")),
                    fmt_pct(s.get("test_f1")),
                    fmt_pct(s.get("test_accuracy")),
                    vram_str,
                    speed_str,
                    str(s.get("total_trials", 0)),
                ])

            gr.Dataframe(
                value=ledger_rows,
                headers=["Model", "Parameters", "Precision", "Recall", "F1", "Accuracy", "Peak VRAM", "Speed", "Optuna trials"],
                datatype=["str", "str", "str", "str", "str", "str", "str", "str", "str"],
                interactive=False,
                wrap=True
            )

            if len(METRICS_DF) > 0:
                has_vram_plot = "VRAM (GB)" in METRICS_DF.columns and METRICS_DF["VRAM (GB)"].dropna().count() > 0
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
                    if has_vram_plot:
                        with gr.Column():
                            gr.LinePlot(
                                value=METRICS_DF,
                                x="Epoch",
                                y="VRAM (GB)",
                                color="Model",
                                title="VRAM Footprint (GB) by epoch",
                                tooltip=["Model", "Epoch", "VRAM (GB)"]
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
                    vram_part = f" · Peak VRAM {s['peak_vram_gb']:.2f} GB" if s.get("peak_vram_gb") else ""
                    speed_part = f" · {s['avg_speed']:.0f} samples/s" if s.get("avg_speed") else ""
                    dur_part = f" · Duration {s['duration']}" if s.get("duration") else ""
                    line = (
                        f"{s.get('name', m_key)} · learning rate "
                        f"{f'{lr:.2e}' if isinstance(lr, (int, float)) else '—'} · "
                        f"weight decay {f'{wd:.4f}' if isinstance(wd, (int, float)) else '—'} · "
                        f"{frozen} frozen layers · {s.get('total_trials', 0)} trials"
                        f"{vram_part}{speed_part}{dur_part}"
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
                elem_classes=["contract-input"],
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

            comparison_df = gr.Dataframe(
                headers=["Model", "Validation F1 (Best)", "Parameters", "Disk Size", "Risks Detected", "Latency (ms)"],
                datatype=["str", "str", "str", "str", "number", "str"],
                label="Benchmark metrics",
                interactive=False,
                wrap=True
            )

            gr.HTML('<p class="compare-hint">Annotated clause comparison across backbones:</p>')

            def compare_cards(text, min_tokens):
                e, tb, bm, bt, df = compare_all(text, min_tokens)
                return (
                    gr.update(value=e, visible=True),
                    gr.update(value=tb, visible=True),
                    gr.update(value=bm, visible=True),
                    gr.update(value=bt, visible=True),
                    df
                )

            # Panels reveal together with bounded scrollable panes
            compare_outputs = []
            with gr.Row(elem_classes=["compare-grid"]):
                for name, f1 in COMPARE_PANELS:
                    with gr.Column(elem_classes=["compare-card"]):
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
                                elem_classes=["highlighted-text", "compare-reader-scroll"],
                            )
                        )

            compare_btn.click(
                fn=compare_cards,
                inputs=[comp_text_input, comp_tokens_slider],
                outputs=[*compare_outputs, comparison_df],
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True, **launch_kwargs)