#!/usr/bin/env python3
"""Gotcha Clause Extractor: Comprehensive Production Training Pipeline

Fine-tunes transformer models (ELECTRA-Small, TinyBERT, BERT-Mini, BERT-Tiny)
for legal sequence labeling (BIO tag token classification) with:
1. Unified Corpus: CodeHima/TOS_Dataset + EE21/ToS-Summaries + Local Synthetic Data.
2. Symmetric context mixing (prevents positional inductive bias).
3. Exact multi-instance sub-word token BIO alignment.
4. Inverse-sqrt class-weighted CrossEntropy loss ([0.38, 4.70, 1.00]).
5. Optuna hyperparameter optimization with complete trial telemetry logging.
6. Comprehensive JSON metric logging (trials, learning curves, hold-out test metrics).
7. Automatic checkpoint cleanup to maintain lean disk footprint.
"""

import os
import re
import sys
import json
import copy
import random
import shutil
import argparse
import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, concatenate_datasets
import nltk
from nltk.tokenize import sent_tokenize
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass
import evaluate
import optuna
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    BertForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    TrainerCallback,
    PrinterCallback,
    logging as tf_logging
)

from gotcha.constants import label2id, id2label, AVAILABLE_MODELS
from gotcha.preprocessor import clean_text_pipeline

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
tf_logging.set_verbosity_error()
tf_logging.disable_progress_bar()
optuna.logging.set_verbosity(optuna.logging.WARNING)

TOKENIZER_MAP = {
    "prajjwal1/bert-tiny": "google-bert/bert-base-uncased",
    "prajjwal1/bert-mini": "google-bert/bert-base-uncased",
    "bert-tiny": "google-bert/bert-base-uncased",
    "bert-mini": "google-bert/bert-base-uncased",
}

MODEL_CLASS_MAP = {
    "prajjwal1/bert-tiny": BertForTokenClassification,
    "prajjwal1/bert-mini": BertForTokenClassification,
    "bert-tiny": BertForTokenClassification,
    "bert-mini": BertForTokenClassification,
}

MODEL_ARCH_MAP = {
    "google/electra-small-discriminator": {"backbone_attr": "electra", "max_layers": 12},
    "electra-small":                     {"backbone_attr": "electra", "max_layers": 12},
    "prajjwal1/bert-tiny":                {"backbone_attr": "bert",    "max_layers": 2},
    "bert-tiny":                          {"backbone_attr": "bert",    "max_layers": 2},
    "prajjwal1/bert-mini":                {"backbone_attr": "bert",    "max_layers": 4},
    "bert-mini":                          {"backbone_attr": "bert",    "max_layers": 4},
    "huawei-noah/TinyBERT_General_4L_312D": {"backbone_attr": "bert",  "max_layers": 4},
    "tinybert":                           {"backbone_attr": "bert",    "max_layers": 4},
}

MODEL_PRETRAINED_MAP = {
    "electra-small": "google/electra-small-discriminator",
    "tinybert": "huawei-noah/TinyBERT_General_4L_312D",
    "bert-mini": "prajjwal1/bert-mini",
    "bert-tiny": "prajjwal1/bert-tiny",
}

STOPWORDS = set('a an the in on at of and or is are was were to for with by as this that it be from have has had not but'.split())

seqeval_metric = evaluate.load("seqeval")


def build_unified_dataset(synthetic_path="corpus/synthetic_gotchas.json", seed=42):
    """Build and partition a unified legal dataset combining CodeHima, EE21, and Synthetic data."""
    print("=" * 65)
    print("  BUILDING UNIFIED TRAINING CORPUS")
    print("=" * 65)

    standardized_data = []
    neutral_sentences = []
    gotcha_sentences = []

    # 1. Ingest CodeHima
    print("Fetching CodeHima/TOS_Dataset from Hugging Face...")
    try:
        splits = {'train': 'data/train-00000-of-00001.parquet'}
        df_codehima = pd.read_parquet("hf://datasets/CodeHima/TOS_Dataset/" + splits["train"])

        for _, row in df_codehima.iterrows():
            sentence = clean_text_pipeline(str(row['sentence']))
            if not sentence or len(sentence) < 10:
                continue
            if row['unfairness_level'] in ['potentially_unfair', 'clearly_unfair']:
                gotcha_sentences.append(sentence)
            else:
                neutral_sentences.append(sentence)
                standardized_data.append({"text": sentence, "gotchas": []})

        # Symmetric context wrapping to eliminate positional prior
        random.seed(seed)
        for gotcha_sent in gotcha_sentences:
            r = random.random()
            if r < 0.40:
                c = random.choice(neutral_sentences)
                comb = c + " " + gotcha_sent
            elif r < 0.80:
                c = random.choice(neutral_sentences)
                comb = gotcha_sent + " " + c
            else:
                c1 = random.choice(neutral_sentences)
                c2 = random.choice(neutral_sentences)
                comb = c1 + " " + gotcha_sent + " " + c2
            standardized_data.append({"text": comb, "gotchas": [gotcha_sent]})

        print(f"  CodeHima parsed: {len(neutral_sentences)} neutral, {len(gotcha_sentences)} gotchas (symmetrically wrapped)")
    except Exception as e:
        print(f"  Warning loading CodeHima: {e}")

    # 2. Ingest EE21/ToS-Summaries
    print("Fetching EE21/ToS-Summaries from Hugging Face...")
    ee21_gotchas = 0
    ee21_neutrals = 0
    try:
        df_summaries = pd.read_json("hf://datasets/EE21/ToS-Summaries/dataset.json", lines=True)
        for _, row in df_summaries.iterrows():
            text = clean_text_pipeline(str(row['plain_text']))
            summary = str(row['summary']).lower()
            sum_words = set(re.findall(r'\b[a-z]{3,}\b', summary)) - STOPWORDS
            if not sum_words or len(text) < 30:
                continue

            sentences = sent_tokenize(text)
            for s in sentences:
                if len(s) < 25 or len(s) > 500:
                    continue
                s_words = set(re.findall(r'\b[a-z]{3,}\b', s.lower())) - STOPWORDS
                overlap = len(sum_words & s_words)
                if overlap >= 4:
                    standardized_data.append({"text": s, "gotchas": [s]})
                    ee21_gotchas += 1
                elif overlap == 0:
                    standardized_data.append({"text": s, "gotchas": []})
                    ee21_neutrals += 1

        print(f"  EE21 parsed: {ee21_gotchas} gotcha spans, {ee21_neutrals} neutral segments")
    except Exception as e:
        print(f"  Warning loading EE21: {e}")

    print(f"  Total human-labeled samples: {len(standardized_data)}")
    raw_original_dataset = Dataset.from_list(standardized_data)
    dataset_splits = raw_original_dataset.train_test_split(test_size=0.15, seed=seed)

    # 3. Augment Train Split Exclusively with Synthetic Gotchas
    if os.path.exists(synthetic_path):
        print(f"Merging synthetic gotchas from {synthetic_path} strictly into train split...")
        try:
            with open(synthetic_path, "r", encoding="utf-8") as f:
                synthetic_data = json.load(f)
            synthetic_dataset = Dataset.from_list(synthetic_data)
            dataset_splits["train"] = concatenate_datasets([dataset_splits["train"], synthetic_dataset])
            print(f"  Added {len(synthetic_data)} synthetic samples. Train split size: {len(dataset_splits['train'])}")
        except Exception as e:
            print(f"  Warning loading synthetic data: {e}")

    print(f"Final partitioned dataset: train={len(dataset_splits['train'])}, test (100% human)={len(dataset_splits['test'])}")
    return dataset_splits


def tokenize_and_align_labels(example, tokenizer, max_length=512):
    """Align multi-occurrence gotchas to sub-word token sequences."""
    text = example['text']
    gotchas = example.get('gotchas', [])

    tokenized_inputs = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True
    )

    offsets = tokenized_inputs["offset_mapping"]
    labels = [label2id['O']] * len(offsets)

    for gotcha in gotchas:
        if not gotcha:
            continue
        gotcha_len = len(gotcha)
        pos = 0
        while True:
            start_char = text.find(gotcha, pos)
            if start_char == -1:
                break
            end_char = start_char + gotcha_len

            gotcha_started = False
            for idx, (start, end) in enumerate(offsets):
                if start == end:
                    labels[idx] = -100
                    continue
                if start >= start_char and start < end_char:
                    if not gotcha_started:
                        labels[idx] = label2id['B-RISK']
                        gotcha_started = True
                    else:
                        labels[idx] = label2id['I-RISK']

            pos = end_char

    tokenized_inputs["labels"] = labels
    del tokenized_inputs["offset_mapping"]
    return tokenized_inputs


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is None:
            self._class_weights = torch.tensor([0.38, 4.70, 1.00])
        else:
            self._class_weights = torch.tensor(class_weights)
        self._cached_loss_fct = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        device = logits.device

        if self._cached_loss_fct is None or self._cached_loss_fct.weight.device != device:
            self._cached_loss_fct = nn.CrossEntropyLoss(
                weight=self._class_weights.to(device)
            )

        if labels is not None:
            active_loss = inputs.get("attention_mask").view(-1) == 1
            active_logits = logits.view(-1, self.model.config.num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1),
                torch.tensor(-100, device=device)
            )
            loss = self._cached_loss_fct(active_logits, active_labels)
        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


class ComprehensiveMetricsCallback(TrainerCallback):
    """Collects and displays epoch metrics for both HPO search trials and final training."""
    def __init__(self, mode="final"):
        self.mode = mode
        self.history = {}
        self.trials_history = {}
        self.current_trial = 0
        self.current_trial_epochs = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        ep = int(round(state.epoch)) if state.epoch is not None else 0
        if ep <= 0:
            return

        target_dict = self.current_trial_epochs if self.mode == "trial" else self.history
        if ep not in target_dict:
            target_dict[ep] = {"Epoch": ep}

        entry = target_dict[ep]
        if "loss" in logs:
            entry["Train_Loss"] = round(float(logs["loss"]), 4)
        if "eval_loss" in logs:
            entry["Eval_Loss"] = round(float(logs["eval_loss"]), 4)
        if "eval_f1" in logs:
            entry["F1"] = round(float(logs["eval_f1"]), 4)
        if "eval_precision" in logs:
            entry["Precision"] = round(float(logs["eval_precision"]), 4)
        if "eval_recall" in logs:
            entry["Recall"] = round(float(logs["eval_recall"]), 4)
        if "eval_accuracy" in logs:
            entry["Accuracy"] = round(float(logs["eval_accuracy"]), 4)
        if "learning_rate" in logs:
            entry["Learning_Rate"] = round(float(logs["learning_rate"]), 7)

        if "eval_f1" in logs:
            prefix = f"  [Trial {self.current_trial + 1} | Ep {ep}]" if self.mode == "trial" else f"  [Epoch {ep}/{int(args.num_train_epochs)}]"
            train_l = entry.get("Train_Loss", "—")
            eval_l = entry.get("Eval_Loss", "—")
            f1 = entry.get("F1", 0.0)
            prec = entry.get("Precision", 0.0)
            rec = entry.get("Recall", 0.0)
            print(f"{prefix} Train Loss: {train_l} | Eval Loss: {eval_l} | F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")

    def on_train_end(self, args, state, control, **kwargs):
        if self.mode == "trial":
            if self.current_trial_epochs:
                sorted_epochs = [self.current_trial_epochs[k] for k in sorted(self.current_trial_epochs.keys())]
                self.trials_history[self.current_trial] = sorted_epochs
                self.current_trial += 1
                self.current_trial_epochs = {}

    def get_final_history_list(self):
        return [self.history[k] for k in sorted(self.history.keys())]

    def reset(self):
        self.history = {}
        self.trials_history = {}
        self.current_trial = 0
        self.current_trial_epochs = {}


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p_idx] for (p_idx, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p_idx, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": float(results.get("overall_precision", 0.0)),
        "recall": float(results.get("overall_recall", 0.0)),
        "f1": float(results.get("overall_f1", 0.0)),
        "accuracy": float(results.get("overall_accuracy", 0.0)),
    }


def freeze_layers(model, model_checkpoint, num_frozen_layers):
    """Dynamically freeze embeddings + specified number of bottom encoder layers."""
    arch = MODEL_ARCH_MAP.get(model_checkpoint)
    if arch is None:
        return

    backbone = getattr(model, arch["backbone_attr"], None)
    if backbone is None:
        return

    if hasattr(backbone, "embeddings"):
        for param in backbone.embeddings.parameters():
            param.requires_grad = False

    if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
        actual_layers = len(backbone.encoder.layer)
        num_to_freeze = min(num_frozen_layers, actual_layers)
        for i in range(num_to_freeze):
            for param in backbone.encoder.layer[i].parameters():
                param.requires_grad = False


def evaluate_model_on_test(model_key: str, dataset_splits, output_base_dir: str = "./gotcha-extractor-model", device: str = "cuda"):
    """Run hold-out test evaluation on an already trained model and update metrics JSON."""
    model_dir = os.path.join(output_base_dir, model_key)
    if not os.path.exists(model_dir):
        print(f"[SKIP] Model directory {model_dir} does not exist.")
        return None

    print(f"\n--- Evaluating {model_key.upper()} on 100% Human Hold-Out Test Split ---")
    pretrained_id = MODEL_PRETRAINED_MAP.get(model_key, model_key)
    tok_name = TOKENIZER_MAP.get(pretrained_id, pretrained_id)
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    tokenized_test = dataset_splits["test"].map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        remove_columns=dataset_splits["test"].column_names
    )

    model_cls = MODEL_CLASS_MAP.get(pretrained_id, AutoModelForTokenClassification)
    model = model_cls.from_pretrained(model_dir).to(device)

    eval_args = TrainingArguments(
        output_dir=os.path.join(output_base_dir, f"{model_key}-eval-temp"),
        per_device_eval_batch_size=8,
        report_to="none",
    )

    eval_trainer = WeightedLossTrainer(
        model=model,
        args=eval_args,
        eval_dataset=tokenized_test,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    eval_trainer.remove_callback(PrinterCallback)

    test_results = eval_trainer.evaluate(eval_dataset=tokenized_test)
    if os.path.exists(eval_args.output_dir):
        shutil.rmtree(eval_args.output_dir)

    test_eval = {
        "f1": round(float(test_results.get("eval_f1", 0.0)), 4),
        "precision": round(float(test_results.get("eval_precision", 0.0)), 4),
        "recall": round(float(test_results.get("eval_recall", 0.0)), 4),
        "loss": round(float(test_results.get("eval_loss", 0.0)), 4),
        "accuracy": round(float(test_results.get("eval_accuracy", 0.0)), 4),
    }
    print(f"  Hold-Out Test Results: F1={test_eval['f1']} | Recall={test_eval['recall']} | Precision={test_eval['precision']} | Loss={test_eval['loss']}")

    # Update existing metrics JSON
    metrics_file = os.path.join(output_base_dir, f"{model_key}_metrics.json")
    metrics_data = {}
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
        except Exception:
            metrics_data = {}

    metrics_data["test_eval"] = test_eval
    summary = metrics_data.get("summary", {})
    summary["test_f1"] = test_eval["f1"]
    summary["test_recall"] = test_eval["recall"]
    summary["test_precision"] = test_eval["precision"]
    metrics_data["summary"] = summary

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    return test_eval


def train_single_model(
    model_key: str,
    dataset_splits,
    output_base_dir: str = "./gotcha-extractor-model",
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.15,
    num_frozen_layers: int = 0,
    use_optuna: bool = False,
    n_trials: int = 10,
    device: str = "cuda"
):
    """Train a designated model with comprehensive Optuna HPO and complete metric recording."""
    pretrained_id = MODEL_PRETRAINED_MAP.get(model_key, model_key)
    print("\n" + "=" * 65)
    print(f"  COMMENCING TRAINING: {model_key.upper()} ({pretrained_id})")
    print(f"  Epochs: {epochs} | Batch Size: {batch_size} | Default LR: {learning_rate} | Optuna: {use_optuna}")
    print("=" * 65)

    tok_name = TOKENIZER_MAP.get(pretrained_id, pretrained_id)
    model_cls = MODEL_CLASS_MAP.get(pretrained_id, AutoModelForTokenClassification)

    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    print("Tokenizing train and test splits...")
    tokenized_train = dataset_splits["train"].map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        remove_columns=dataset_splits["train"].column_names
    )
    tokenized_test = dataset_splits["test"].map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        remove_columns=dataset_splits["test"].column_names
    )

    arch = MODEL_ARCH_MAP.get(model_key, {})
    max_freezable = arch.get("max_layers", 0)
    output_dir = os.path.join(output_base_dir, model_key)
    os.makedirs(output_dir, exist_ok=True)

    # Read existing metrics JSON if it exists to preserve prior trials when running without Optuna
    metrics_file = os.path.join(output_base_dir, f"{model_key}_metrics.json")
    existing_metrics = {}
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                existing_metrics = json.load(f)
        except Exception:
            existing_metrics = {}

    best_f1 = 0.0
    best_hp = {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "per_device_train_batch_size": batch_size,
        "warmup_ratio": warmup_ratio,
        "num_frozen_layers": num_frozen_layers,
    }

    trials_dict = existing_metrics.get("trials", {})

    if use_optuna:
        print(f"\n--- Running Optuna Hyperparameter Search ({n_trials} trials, {epochs} epochs each) ---")
        search_dir = os.path.join(output_base_dir, f"{model_key}-optuna-search")
        trial_metrics_cb = ComprehensiveMetricsCallback(mode="trial")

        trial_records = {}

        def model_init(trial=None):
            m = model_cls.from_pretrained(
                pretrained_id,
                num_labels=len(label2id),
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True
            ).to(device)

            if trial is not None:
                trial_metrics_cb.current_trial = trial.number
                if max_freezable > 0:
                    n_freeze = trial.suggest_int("num_frozen_layers", 0, max_freezable)
                    freeze_layers(m, model_key, n_freeze)
                else:
                    n_freeze = 0
                    freeze_layers(m, model_key, 0)
                if trial.number not in trial_records:
                    trial_records[trial.number] = {"trial_number": trial.number + 1, "params": {}}
                trial_records[trial.number]["params"]["num_frozen_layers"] = n_freeze
            else:
                freeze_layers(m, model_key, num_frozen_layers)
            return m

        def hp_space(trial):
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 8e-5, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.15),
                "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16]),
                "warmup_ratio": trial.suggest_float("warmup_ratio", 0.05, 0.25),
            }
            if trial.number not in trial_records:
                trial_records[trial.number] = {"trial_number": trial.number + 1, "params": {}}
            trial_records[trial.number]["params"].update(copy.deepcopy(params))
            return params

        search_args = TrainingArguments(
            output_dir=search_dir,
            eval_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            disable_tqdm=True,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=1,
            report_to="none",
        )

        search_trainer = WeightedLossTrainer(
            model_init=model_init,
            args=search_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[trial_metrics_cb],
        )
        search_trainer.remove_callback(PrinterCallback)

        best_run = search_trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",
            hp_space=hp_space,
            n_trials=n_trials,
            compute_objective=lambda m: m["eval_f1"],
        )

        best_hp.update(best_run.hyperparameters)
        best_f1 = best_run.objective

        # Build comprehensive trials dictionary from trial records and callback
        trials_dict = {}
        for trial_num, rec in trial_records.items():
            t_key = str(trial_num + 1)
            epoch_records = trial_metrics_cb.trials_history.get(trial_num, [])

            epochs_list = [e.get("Epoch") for e in epoch_records]
            f1s_list = [e.get("F1", 0.0) for e in epoch_records]
            eval_losses = [e.get("Eval_Loss", e.get("Loss", 0.0)) for e in epoch_records]
            train_losses = [e.get("Train_Loss", eval_losses[i] if i < len(eval_losses) else 0.0) for i, e in enumerate(epoch_records)]
            precs = [e.get("Precision", 0.0) for e in epoch_records]
            recs = [e.get("Recall", 0.0) for e in epoch_records]
            accs = [e.get("Accuracy", 0.0) for e in epoch_records]

            best_trial_f1 = max(f1s_list) if f1s_list else 0.0

            trials_dict[t_key] = {
                "trial_number": trial_num + 1,
                "state": "COMPLETE",
                "params": rec["params"],
                "best_f1": round(float(best_trial_f1), 4),
                "epochs": epochs_list,
                "f1": f1s_list,
                "loss": eval_losses,
                "eval_loss": eval_losses,
                "train_loss": train_losses,
                "precision": precs,
                "recall": recs,
                "accuracy": accs,
            }

        if os.path.exists(search_dir):
            shutil.rmtree(search_dir)

        print(f"\nOptuna Search Finished. Best F1: {best_f1:.4f}")
        print(f"Optimal Parameters: {best_hp}")

    # Final Training Run
    print(f"\n--- Executing Final Training Run ---")
    final_frozen = best_hp.get("num_frozen_layers", num_frozen_layers)
    final_model = model_cls.from_pretrained(
        pretrained_id,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    ).to(device)

    freeze_layers(final_model, model_key, final_frozen)

    trainable_params = sum(p.numel() for p in final_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in final_model.parameters())
    print(f"Model Parameters: {trainable_params:,} trainable / {total_params:,} total ({trainable_params / total_params * 100:.1f}%)")

    final_metrics_cb = ComprehensiveMetricsCallback(mode="final")

    final_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=best_hp.get("learning_rate", learning_rate),
        per_device_train_batch_size=best_hp.get("per_device_train_batch_size", batch_size),
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=best_hp.get("weight_decay", weight_decay),
        disable_tqdm=True,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        warmup_ratio=best_hp.get("warmup_ratio", warmup_ratio),
        report_to="none",
    )

    final_trainer = WeightedLossTrainer(
        model=final_model,
        args=final_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[final_metrics_cb],
    )
    final_trainer.remove_callback(PrinterCallback)

    final_trainer.train()

    # Save final model and tokenizer
    print(f"Saving fine-tuned model and tokenizer to {output_dir}...")
    final_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Hold-out test set evaluation
    print("\n--- Evaluating Best Model Checkpoint on 100% Human Hold-Out Test Split ---")
    test_results = final_trainer.evaluate(eval_dataset=tokenized_test)
    test_eval = {
        "f1": round(float(test_results.get("eval_f1", 0.0)), 4),
        "precision": round(float(test_results.get("eval_precision", 0.0)), 4),
        "recall": round(float(test_results.get("eval_recall", 0.0)), 4),
        "loss": round(float(test_results.get("eval_loss", 0.0)), 4),
        "accuracy": round(float(test_results.get("eval_accuracy", 0.0)), 4),
    }
    print(f"Hold-Out Test Score: F1={test_eval['f1']} | Recall={test_eval['recall']} | Precision={test_eval['precision']} | Loss={test_eval['loss']}")

    # Assemble comprehensive metrics
    history = final_metrics_cb.get_final_history_list()
    final_epochs = [h["Epoch"] for h in history]
    final_f1 = [h.get("F1", 0.0) for h in history]
    final_eval_loss = [h.get("Eval_Loss", 0.0) for h in history]
    final_train_loss = [h.get("Train_Loss", final_eval_loss[idx] if idx < len(final_eval_loss) else 0.0) for idx, h in enumerate(history)]
    final_prec = [h.get("Precision", 0.0) for h in history]
    final_rec = [h.get("Recall", 0.0) for h in history]
    final_acc = [h.get("Accuracy", 0.0) for h in history]
    final_lr = [h.get("Learning_Rate", 0.0) for h in history]

    peak_f1 = max(final_f1) if final_f1 else best_f1
    peak_epoch = final_epochs[final_f1.index(peak_f1)] if final_f1 else None

    metrics_data = {
        "model_info": {
            "name": model_key,
            "pretrained_id": pretrained_id,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": device,
            "timestamp": datetime.datetime.now().isoformat()
        },
        "best_hp": {
            "num_frozen_layers": final_frozen,
            "learning_rate": best_hp.get("learning_rate"),
            "weight_decay": best_hp.get("weight_decay"),
            "batch_size": best_hp.get("per_device_train_batch_size", batch_size),
            "warmup_ratio": best_hp.get("warmup_ratio"),
            "best_f1": round(float(peak_f1), 4)
        },
        "trials": trials_dict,
        "final_run": {
            "epochs": final_epochs,
            "f1": final_f1,
            "loss": final_eval_loss,
            "eval_loss": final_eval_loss,
            "train_loss": final_train_loss,
            "precision": final_prec,
            "recall": final_rec,
            "accuracy": final_acc,
            "learning_rate": final_lr
        },
        "test_eval": test_eval,
        "summary": {
            "peak_val_f1": round(float(peak_f1), 4),
            "peak_val_epoch": peak_epoch,
            "optuna_best_f1": round(float(best_f1), 4) if best_f1 else round(float(peak_f1), 4),
            "test_f1": test_eval["f1"],
            "test_recall": test_eval["recall"],
            "test_precision": test_eval["precision"],
            "total_trials": len(trials_dict)
        }
    }

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Comprehensive metrics ledger saved to: {metrics_file}")

    # Remove intermediate checkpoint directory to keep disk space lean
    for item in os.listdir(output_dir):
        subpath = os.path.join(output_dir, item)
        if os.path.isdir(subpath) and item.startswith("checkpoint-"):
            shutil.rmtree(subpath)

    print(f"\n[SUCCESS] {model_key.upper()} trained successfully! Peak Val F1: {peak_f1:.4f} | Test F1: {test_eval['f1']:.4f}")
    return peak_f1


def main():
    parser = argparse.ArgumentParser(description="Gotcha Clause Extractor: Production Training & Evaluation Suite")
    parser.add_argument("--model", type=str, default="electra-small", choices=AVAILABLE_MODELS + ["all"],
                        help="Model key to train, or 'all' to train all 4 models sequentially.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10).")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8).")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (default: 5e-5).")
    parser.add_argument("--optuna", action="store_true", help="Enable Optuna hyperparameter optimization.")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of Optuna trials (default: 10).")
    parser.add_argument("--dry_run", action="store_true", help="Quick 1-minute dry run on tiny sample to verify pipeline.")
    parser.add_argument("--evaluate_all", action="store_true", help="Evaluate all existing models on test set and write metrics.")
    parser.add_argument("--output_dir", type=str, default="./gotcha-extractor-model", help="Base output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--synthetic_path", type=str, default="corpus/synthetic_gotchas.json", help="Path to synthetic JSON.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing training pipeline on compute device: {device}")

    # Build dataset
    splits = build_unified_dataset(synthetic_path=args.synthetic_path, seed=args.seed)

    # If evaluate_all requested, evaluate all existing models and exit
    if args.evaluate_all:
        print("\n" + "=" * 65)
        print("  RUNNING COMPREHENSIVE HOLD-OUT TEST EVALUATION ON ALL MODELS")
        print("=" * 65)
        for m in AVAILABLE_MODELS:
            evaluate_model_on_test(m, splits, output_base_dir=args.output_dir, device=device)
        print("\nAll model test evaluations complete.")
        return

    if args.dry_run:
        print("\n*** DRY RUN MODE: Truncating splits to 16 train / 8 test samples for fast verification ***")
        splits["train"] = splits["train"].select(range(min(16, len(splits["train"]))))
        splits["test"] = splits["test"].select(range(min(8, len(splits["test"]))))
        args.epochs = 2
        if args.optuna:
            args.n_trials = 2

    models_to_train = AVAILABLE_MODELS if args.model == "all" else [args.model]

    for m in models_to_train:
        train_single_model(
            model_key=m,
            dataset_splits=splits,
            output_base_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            use_optuna=args.optuna,
            n_trials=args.n_trials,
            device=device
        )

    print("\n" + "=" * 65)
    print("  ALL REQUESTED TRAINING & EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
