#!/usr/bin/env python3
"""Gotcha Clause Extractor: Production Retraining Pipeline

Fine-tunes transformer models (ELECTRA, TinyBERT, BERT-Mini, BERT-Tiny)
for legal sequence labeling (BIO tag classification) with:
1. Connected EE21/ToS-Summaries + CodeHima/TOS_Dataset + Local Synthetic data.
2. Symmetric context mixing (prevents positional inductive bias).
3. Multi-instance token alignment.
4. Class-weighted CrossEntropy loss ([0.38, 4.70, 1.00]).
5. Optional Optuna hyperparameter optimization.
6. Automatic checkpoint cleanup.
"""

import os
import re
import json
import copy
import random
import shutil
import argparse
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, concatenate_datasets
from nltk.tokenize import sent_tokenize
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


def build_unified_dataset(synthetic_path="corpus/synthetic_gotchas.json", seed=42):
    """Build and partition a unified legal dataset combining CodeHima, EE21, and Synthetic data."""
    print("=" * 60)
    print("  BUILDING UNIFIED TRAINING CORPUS")
    print("=" * 60)

    # 1. Ingest CodeHima
    print("Fetching CodeHima/TOS_Dataset from Hugging Face...")
    splits = {'train': 'data/train-00000-of-00001.parquet'}
    df_codehima = pd.read_parquet("hf://datasets/CodeHima/TOS_Dataset/" + splits["train"])

    neutral_sentences = []
    gotcha_sentences = []
    standardized_data = []

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

    # 2. Ingest EE21/ToS-Summaries
    print("Fetching EE21/ToS-Summaries from Hugging Face...")
    df_summaries = pd.read_json("hf://datasets/EE21/ToS-Summaries/dataset.json", lines=True)

    ee21_gotchas = 0
    ee21_neutrals = 0
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pre-allocate weights to avoid tensor allocations on each step
        self._class_weights = torch.tensor([0.38, 4.70, 1.00])
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


class EpochMetricsCallback(TrainerCallback):
    """Log and format clean epoch validation metrics."""
    def __init__(self):
        self.history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "eval_loss" in logs:
            row = {
                "Epoch": int(state.epoch) if state.epoch == int(state.epoch) else round(state.epoch, 2),
                "Eval_Loss": round(logs.get("eval_loss", 0), 4),
                "F1": round(logs.get("eval_f1", 0), 4),
                "Precision": round(logs.get("eval_precision", 0), 4),
                "Recall": round(logs.get("eval_recall", 0), 4),
                "Accuracy": round(logs.get("eval_accuracy", 0), 4),
            }
            self.history.append(row)
            print(f"  [Epoch {row['Epoch']}] Loss: {row['Eval_Loss']} | F1: {row['F1']} | Precision: {row['Precision']} | Recall: {row['Recall']}")


seqeval_metric = evaluate.load("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results.get("overall_precision", 0.0),
        "recall": results.get("overall_recall", 0.0),
        "f1": results.get("overall_f1", 0.0),
        "accuracy": results.get("overall_accuracy", 0.0),
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
    """Train a designated model with optional Optuna HPO."""
    pretrained_id = MODEL_PRETRAINED_MAP.get(model_key, model_key)
    print("\n" + "=" * 60)
    print(f"  COMMENCING TRAINING: {model_key.upper()} ({pretrained_id})")
    print(f"  Epochs: {epochs} | Batch Size: {batch_size} | LR: {learning_rate} | Frozen: {num_frozen_layers}")
    print("=" * 60)

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

    best_f1 = 0.0
    best_hp = {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "per_device_train_batch_size": batch_size,
        "warmup_ratio": warmup_ratio,
        "num_frozen_layers": num_frozen_layers,
    }

    if use_optuna:
        print(f"\n--- Running Optuna Hyperparameter Search ({n_trials} trials) ---")
        search_dir = os.path.join(output_base_dir, f"{model_key}-optuna-search")

        def model_init(trial=None):
            m = model_cls.from_pretrained(
                pretrained_id,
                num_labels=len(label2id),
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True
            ).to(device)

            if trial is not None and max_freezable > 0:
                n_freeze = trial.suggest_int("num_frozen_layers", 0, max_freezable)
                freeze_layers(m, model_key, n_freeze)
            else:
                freeze_layers(m, model_key, num_frozen_layers)
            return m

        def hp_space(trial):
            return {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 8e-5, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.15),
                "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16]),
                "warmup_ratio": trial.suggest_float("warmup_ratio", 0.05, 0.25),
            }

        search_args = TrainingArguments(
            output_dir=search_dir,
            eval_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            logging_strategy="no",
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
        if os.path.exists(search_dir):
            shutil.rmtree(search_dir)

        print(f"Optuna Best F1: {best_f1:.4f} with params: {best_hp}")

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
    print(f"Trainable Parameters: {trainable_params:,} / {total_params:,} ({trainable_params / total_params * 100:.1f}%)")

    metrics_callback = EpochMetricsCallback()

    final_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=best_hp.get("learning_rate", learning_rate),
        per_device_train_batch_size=best_hp.get("per_device_train_batch_size", batch_size),
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=best_hp.get("weight_decay", weight_decay),
        logging_strategy="no",
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
        callbacks=[metrics_callback],
    )
    final_trainer.remove_callback(PrinterCallback)

    final_trainer.train()

    # Save final model and tokenizer
    print(f"Saving final model to {output_dir}...")
    final_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save metrics JSON
    history = metrics_callback.history
    peak_f1 = max([h["F1"] for h in history]) if history else best_f1
    metrics_data = {
        "final_run": {
            "epochs": [h["Epoch"] for h in history],
            "f1": [h["F1"] for h in history],
            "loss": [h["Eval_Loss"] for h in history],
            "precision": [h["Precision"] for h in history],
            "recall": [h["Recall"] for h in history]
        },
        "best_hp": {
            "num_frozen_layers": final_frozen,
            "learning_rate": best_hp.get("learning_rate"),
            "weight_decay": best_hp.get("weight_decay"),
            "batch_size": best_hp.get("per_device_train_batch_size"),
            "warmup_ratio": best_hp.get("warmup_ratio"),
            "best_f1": peak_f1
        }
    }
    metrics_file = os.path.join(output_base_dir, f"{model_key}_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    # Remove intermediate checkpoint directory to keep disk space lean
    for item in os.listdir(output_dir):
        subpath = os.path.join(output_dir, item)
        if os.path.isdir(subpath) and item.startswith("checkpoint-"):
            shutil.rmtree(subpath)

    print(f"\n[SUCCESS] {model_key.upper()} trained successfully! Peak F1: {peak_f1:.4f}")
    return peak_f1


def main():
    parser = argparse.ArgumentParser(description="Train ToS Gotcha Extractor Models")
    parser.add_argument("--model", type=str, default="electra-small", choices=AVAILABLE_MODELS + ["all"],
                        help="Model key to train, or 'all' to train all 4 models sequentially.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10).")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8).")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (default: 5e-5).")
    parser.add_argument("--optuna", action="store_true", help="Enable Optuna hyperparameter search.")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of Optuna trials (default: 10).")
    parser.add_argument("--dry_run", action="store_true", help="Quick 1-step dry run on tiny sample to verify pipeline.")
    parser.add_argument("--output_dir", type=str, default="./gotcha-extractor-model", help="Base output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Build dataset
    splits = build_unified_dataset(seed=args.seed)

    if args.dry_run:
        print("\n*** DRY RUN MODE: Truncating splits to 16 samples for pipeline verification ***")
        splits["train"] = splits["train"].select(range(min(16, len(splits["train"]))))
        splits["test"] = splits["test"].select(range(min(8, len(splits["test"]))))
        args.epochs = 1
        args.optuna = False

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

    print("\n" + "=" * 60)
    print("  ALL REQUESTED TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
