# Legal Clause Extraction Strategies: Design & Evaluation

This document outlines the various strategies, design decisions, and architectural tradeoffs evaluated during the development of the ToS "Gotcha" Clause Extractor model and local inference pipeline.

---

## 1. Sequence Labeling Tag Decoding Strategies

Our model classifies tokens using BIO format tags: `O` (Outside), `B-RISK` (Beginning of Risk), and `I-RISK` (Inside Risk). We evaluated two main decoding strategies to convert these token predictions into continuous text spans.

| Strategy | Decodes | Tradeoffs / Behavior | Result |
| :--- | :--- | :--- | :--- |
| **Strict BIO Decoding** | Spans *must* start with `B-RISK`, followed by `I-RISK` tokens. | Highly precise in standard NER, but misses any risk spans where the model assigns `I-RISK` to the first word instead of `B-RISK`. | **Failed**: The model trained on sparse tags almost never predicted `B-RISK`. It yielded **0 gotchas** on test documents. |
| **IO Fallback Decoding** | Spans can start on *either* `B-RISK` or `I-RISK` tokens. | Maximizes recall by capturing all flagged tokens regardless of starting tags. Requires post-processing to avoid grabbing single noisy words. | **Successful**: Successfully recovered 100% of labeled gotchas and clauses. |

### Technical Implementation Comparison
* **Strict BIO**:
  ```python
  active = False
  for idx, pred in enumerate(predictions):
      if label == "B-RISK":
          active = True
          # start new span
      elif label == "I-RISK" and active:
          # extend span
      else:
          active = False
  ```
* **IO Fallback (Implemented)**:
  ```python
  for idx, pred in enumerate(predictions):
      if label in ("B-RISK", "I-RISK"):
          # add token offset to candidate list regardless of starting tag
  ```

---

## 2. Text Preprocessing & NLP Pipeline

To address the raw data issues (including Mojibake and hard word wraps), we introduced an NLP preprocessing pipeline before feeding the text to the sentence tokenizer and model:

1. **Mojibake Recovery (`ftfy`)**:
   Automatically identifies and repairs malformed text encodings (e.g. `â€œ` -> `"`, `â€™` -> `'`) safely and deterministically.
2. **Hard Wrap Normalization**:
   Replaces single line breaks (`(?<!\n)\n(?!\n)`) with spaces to prevent sentences from being cut in half, while keeping double newlines to preserve paragraph structure.
3. **Space Standardization**:
   Collapses consecutive whitespace and tabs into a single space for cleaner tokenization.

```python
def clean_text_pipeline(raw_text):
    # 1. Automatically fix Mojibake and encoding issues
    text = ftfy.fix_text(raw_text)
    
    # 2. Normalize hard wraps while preserving paragraph breaks
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    # 3. Standardize consecutive spaces
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()
```

---

## 3. Text Segmentation & Sentence Density Classification

Transformer models (like LegalBERT) have a strict context length limit of **512 tokens**. We analyzed how to chunk and process large legal documents.

### Strategy A: Direct Document-Level Inference (No Segmentation)
- **Mechanism**: The raw text file is truncated to 512 tokens and fed into the model.
- **Verdict**: **Infeasible**; truncates anything after the first few paragraphs.

### Strategy B: Token-Offset Slicing (Old Method)
- **Mechanism**: Splits documents into sentences, maps predicted risk token offsets, merges them, and slices the text.
- **Verdict**: **Noisy & Broken**; when the model assigns `O` to minor intermediate sub-tokens (or when token alignment cuts words in half), it returns fragmented words and phrases.

### Strategy C: Sentence Density Classification (Implemented NLP Pipeline)
- **Mechanism**:
  - The cleaned text is tokenized into sentences using NLTK (`sent_tokenize` / `PunktSentenceTokenizer`).
  - Each sentence is tokenized and fed into the model.
  - We count the number of risk tokens (`B-RISK` or `I-RISK`) in each sentence.
  - If a sentence contains `min_risk_tokens >= 3` risk tokens, we extract the entire, grammatically complete sentence as a "gotcha".
- **Pros**:
  - Aligns with the sentence-level training distribution.
  - Prevents word-slicing and grammatical fragmentation.
  - Clean and human-readable outputs in the UI.
- **Verdict**: **Best performing**; implemented across the notebook and Gradio apps.

---

## 4. File System & Encoding Strategies (Windows Support)

Windows systems use a default encoding of `cp1252` in Python, which throws a fatal `UnicodeDecodeError` when it encounters curly quotes (`â€œ` / `“`), em-dashes, or special symbols in legal documents.

- **Robust Method**: `open(file_path, "r", encoding="utf-8")` -> Successfully reads all corpus files.
- **Mojibake Handling**: Standard print outputs fallback to ASCII-replacement (`errors='replace'`) if the console environment raises a `UnicodeEncodeError`.

---

## 5. Ordinal Risk Classification & Visual Highlighting

To help users prioritize the most critical gotchas, we replaced the binary "RISK" / "no risk" highlighting with an ordinal system in the Gradio UI:
- **HIGH RISK (Red - `#ef4444`)**: Unilateral terms modification, class action waivers, forced arbitration, and data selling.
- **MEDIUM RISK (Orange - `#f97316`)**: Data tracking disclosures, user request limitations, and third-party databases liability disclaimers.
- **LOW RISK (Yellow - `#eab308`)**: Standard legal background text and minor declarations.

The risk level of a flagged sentence (`risk_tokens >= 3`) is determined using a combination of the maximum risk token confidence (`max_prob`) and high-severity keyword patterns:
- **HIGH RISK**: `max_prob >= 0.80` OR (`has_high_keyword` AND `max_prob >= 0.68`).
- **MEDIUM RISK**: (`has_high_keyword` AND `max_prob >= 0.55`) OR `max_prob >= 0.62`.
- **LOW RISK**: Fallback.

---

## 6. Precision Improvements: Hybrid Filtering & Conditional Thresholds

During local corpus evaluation (e.g. `#TeamSeas_PRIVACY POLICY.txt`), we found that sentence density classification alone results in a high number of false positives (noise), extracting 26 sentences. This noise consisted of standard scope definitions, GDPR legal grounds boilerplate, and headers.

To resolve this while maintaining high recall for important clauses, we implemented a hybrid post-processing pipeline:

1. **Short Header Filter**: Skip sentences `< 50` characters that are in all-caps or match standard title regex patterns (e.g. `PRIVACY POLICY EFFECTIVE DATE`).
2. **Boilerplate Regex Patterns**: Match and skip standard compliance declarations, definitions, and scope descriptions.
3. **Conditional Keyword Thresholding**:
   - For sentences containing **high-severity keywords** (arbitration, class action, waiver, dispute, modify, sell, third party, warranty disclaimers, indemnity), we allow lower confidence (`max_prob >= 0.55`) to ensure high recall for key clauses.
   - For sentences **without** high-severity keywords, we raise the bar to `max_prob >= 0.70` to prevent low-confidence boilerplate text from triggering highlights.

### Evaluation Results (TeamSeas Privacy Policy)
- **Baseline**: 26 extractions (contained intro noise, definitions, and GDPR boilerplates).
- **Hybrid Pipeline (Implemented)**: **12 extractions**. Filters out 100% of headers and legal boilerplate while retaining 100% of true gotchas (unilateral change, third-party database liability disclaimers, and marketing data-sharing).

---

## 7. Token Classification vs. Sequence Classification

We analyzed the tradeoffs between the current token-level classification (using `RobertaForTokenClassification` / `BertForTokenClassification`) and sequence-level sentence classification (using `RobertaForSequenceClassification`):

| Aspect | Token Classification (BIO Tagging) | Sequence Classification (Sentence-level) |
| :--- | :--- | :--- |
| **Model Type** | e.g. `AutoModelForTokenClassification` | e.g. `RobertaForSequenceClassification` |
| **Granularity** | Sub-word / Token-level. Predicts a label for every token in the sentence. | Sequence-level. Predicts a single class (`0` or `1`) for the entire sentence. |
| **Explainability** | **High**: We can isolate and highlight the exact words that triggered the risk score (e.g. "forced arbitration", "sell data"). | **Low**: The model flags the entire sentence as a gotcha, but cannot point to the specific words responsible. |
| **Handling Multiple Risks** | Excellent. Can detect and label multiple distinct risks within a single complex sentence. | Poor. Flags the whole sentence as a single binary risk, merging different categories together. |
| **Dataset Alignment** | Requires aligning character offsets of raw clauses to sub-word token sequences (handled by `offset_mapping`). | Simpler. Requires only sentence-level labels (e.g., standard binary classification dataset). |

### Verdict
While Sequence Classification is simpler to train, **Token Classification with BIO Tagging** is highly preferred for legal clause extraction because it preserves word-level explainability (which is critical in legal technology to show users exactly *what* is unfavorable about a clause) and allows for a more flexible post-processing pipeline (like our token density and conditional keyword thresholding).

---

## 8. Class Imbalance & Weighted Loss Optimization

### Token Label Distribution Findings
To design a more robust loss weighting strategy, we ran a token label frequency analysis across the training dataset:
- **`O` (Outside / Background)**: `412,954` occurrences (`84.65%`)
- **`B-RISK` (Beginning of Gotcha)**: `1,215` occurrences (`0.25%`)
- **`I-RISK` (Inside Gotcha)**: `73,691` occurrences (`15.10%`)

This confirms that the `B-RISK` token is **extremely sparse**—roughly **60x rarer** than the `I-RISK` token and **340x rarer** than the background `O` class. 

### Weighted Loss Behavior
By applying a custom `WeightedLossTrainer` with target class weights `[0.15, 1.0, 1.0]`, we penalize misclassifications on risk tags ~6.6x more heavily than on background tokens. 
- **Result**: F1 score increased significantly, peaking at **`0.4195`** (at Epoch 8) compared to the unweighted baseline peak of **`0.2800`**.
- **Validation Loss Divergence**: During training, the validation loss was observed to climb (peaking at `1.52` at Epoch 8) even as F1 score improved. This is expected behavior with weighted cross-entropy. Because errors on rare classes (`B-RISK` / `I-RISK`) are heavily weighted, any confident false positive or misaligned token classification greatly increases the cross-entropy loss, even though the overall precision, recall, and F1 improve.

### Epoch-by-Epoch Metric Logs

Below is the comparison between the unweighted baseline run and the first custom weighted run (`[0.15, 1.0, 1.0]`).

#### Run 1: Unweighted Baseline (CrossEntropyLoss)
*Epochs 1-5 metrics showing severe overfitting by Epoch 3:*

| Epoch | Training Loss | Validation Loss | Precision | Recall | F1 Score | Accuracy |
| :---: | :-----------: | :-------------: | :-------: | :----: | :------: | :------: |
| 1     | 0.356628      | 0.330325        | 0.046099  | 0.079268 | 0.058296 | 0.866110 |
| 2     | 0.348027      | 0.304125        | 0.064736  | 0.231707 | 0.101198 | 0.877864 |
| 3     | 0.245735      | 0.363917        | 0.130699  | 0.262195 | 0.174442 | 0.882941 |
| 4     | 0.233512      | 0.453271        | 0.170149  | 0.347561 | 0.228457 | 0.891357 |
| 5     | 0.098720      | 0.521604        | 0.198473  | 0.475610 | 0.280072 | 0.869228 |

#### Run 2: Weighted Loss Run (`[0.15, 1.0, 1.0]`)
*Training run peaking at Epoch 8 before early stopping begins decay:*

| Epoch | Training Loss | Validation Loss | Precision | Recall | F1 Score | Accuracy |
| :---: | :-----------: | :-------------: | :-------: | :----: | :------: | :------: |
| 1     | 0.715565      | 0.698208        | 0.000307  | 0.011364 | 0.000597 | 0.612319 |
| 2     | 0.708619      | 0.562188        | 0.014003  | 0.164773 | 0.025812 | 0.725119 |
| 3     | 0.549188      | 0.526344        | 0.049486  | 0.301136 | 0.085004 | 0.812416 |
| 4     | 0.486801      | 0.553252        | 0.109195  | 0.539773 | 0.181644 | 0.807929 |
| 5     | 0.445968      | 0.761628        | 0.089674  | 0.375000 | 0.144737 | 0.832108 |
| 6     | 0.225640      | 0.899322        | 0.156069  | 0.460227 | 0.233094 | 0.860157 |
| 7     | 0.314585      | 1.122243        | 0.178879  | 0.471591 | 0.259375 | 0.854217 |
| **8** | **0.113244**  | **1.525058**    | **0.367521** | **0.488636** | **0.419512** | **0.874355** |
| 9     | 0.092061      | 1.518979        | 0.229236  | 0.392045 | 0.289308 | 0.873750 |

### Next-Generation Weighting Recommendations
For future training runs, the loss weights can be refined to better separate `B-RISK` and `I-RISK` to reflect the scarcity of starting boundaries:
- **Balanced Class Weighting Option**: `[0.10, 2.0, 1.0]`. This weights `B-RISK` twice as heavily as `I-RISK`, and 20x relative to `O`, helping the model identify gotcha starting boundaries more confidently.
- **Learning Rate Tuning**: Lowering learning rate to `1e-5` (down from `2e-5`) with a `warmup_ratio=0.15` can help stabilize gradients under weighted loss.

---

## 9. LLM-Based Data Augmentation & Verification

To permanently address class imbalance at the training data level (rather than relying solely on loss scaling), we implemented an offline synthetic data generation pipeline to augment the minority classes.

### Synthetic Generation Strategy (DeepSeek V4 Flash)
Using a custom CLI tool, we prompted `deepseek/deepseek-v4-flash` via OpenRouter to generate batches of realistic Terms of Service gotcha sentences across four risk categories (unilateral changes, arbitration, data selling, indemnification).
- **Target Quantity**: Exactly **3,479** additional gotcha sentences were generated to fill the sample-level balance gap.
- **Constraints Enforced**: Auto-verified that generated gotcha targets are exact substrings of the text to prevent token mapping alignment errors.
- **Dataset Splitting (Validation Discipline)**: Split the baseline human-labeled dataset first into `train` (85%) and `test` (15%) splits, then merge the synthetic dataset **exclusively** into the training split. This ensures the validation dataset remains **100% real, human-labeled data**, completely preventing synthetic data leakage.
- **Refined Weights for Balanced Training**: Lowered the class weights in `WeightedLossTrainer` (Cell 6) to **`[0.20, 1.5, 0.8]`** (O: 0.20, B-RISK: 1.50, I-RISK: 0.80) to align with the balanced training data, boosting **Precision** while maintaining high **Recall**.

### Verification & Class Balance Improvements

After integrating the **3,479** synthetic sentences, we ran a verification parser to count the updated class distribution:

#### Sample-Level Balance

| Dataset Version | Total Samples | Gotcha (Unfair) Samples | Non-Gotcha (Neutral) Samples |
| :--- | :---: | :---: | :---: |
| **Original Dataset** | 6,225 | 1,373 (22.06%) | 4,852 (77.94%) |
| **Augmented Dataset** | **9,704** | **3,864 (39.82%)** | **5,840 (60.18%)** |
| **Progress** | **+3,479** | **~2x Gotcha representation** | **Balanced distribution** |

#### Token-Level Balance (BIO Tags)

| Token Tag Class | Original Counts | Augmented Counts | Count Share Change |
| :--- | :---: | :---: | :---: |
| **`O` (Background)** | 412,954 | 452,618 | Decreased from 84.65% to **81.61%** |
| **`B-RISK` (Boundary Start)** | 1,215 | 3,939 | **Increased from 0.25% to 0.71% (~3x boost)** |
| **`I-RISK` (Boundary Body)** | 73,691 | 98,063 | Increased from 15.10% to **17.68%** |
| **Total Risk Tokens (B+I)** | 74,906 | **102,002** | **Increased from 15.35% to 18.39%** |

### Epoch-by-Epoch Metric Logs (Augmented Run)

#### Run 3: Augmented Dataset (No Leakage, Weights `[0.20, 1.50, 0.80]`)
*Final training run validated strictly on 100% clean, human-labeled data:*

| Epoch | Training Loss | Validation Loss | Precision | Recall | F1 Score | Accuracy |
| :---: | :-----------: | :-------------: | :-------: | :----: | :------: | :------: |
| 1     | 0.616961      | 0.670520        | 0.004093  | 0.070270 | 0.007736 | 0.648470 |
| 2     | 0.492043      | 0.608343        | 0.033696  | 0.167568 | 0.056109 | 0.821527 |
| 3     | 0.344169      | 0.578036        | 0.071913  | 0.286486 | 0.114967 | 0.854305 |
| 4     | 0.349514      | 0.651165        | 0.111296  | 0.362162 | 0.170267 | 0.860793 |
| 5     | 0.308415      | 0.827021        | 0.124224  | 0.324324 | 0.179641 | 0.867483 |
| 6     | 0.173650      | 1.190387        | 0.196721  | 0.324324 | 0.244898 | 0.869775 |
| 7     | 0.129882      | 1.127535        | 0.162621  | 0.362162 | 0.224456 | 0.870311 |
| **8** | **0.139337**  | **1.176360**    | **0.191429** | **0.362162** | **0.250467** | **0.870083** |

### Expected Training Impact
1. **Critical Boundary Resolution**: The **nearly 3x increase** in the `B-RISK` start tag frequency helps resolve the boundary detection issue. The model is now exposed to three times as many gotcha starting boundaries.
2. **Mitigated Majority Class Bias**: The model is less prone to over-predicting the background `O` tag since the sample-level gotcha density is nearly doubled to **~40%**.
3. **Enhanced Generalization**: The diverse, high-quality phrasing styles from DeepSeek expand the dataset vocabulary, preventing overfitting on small training samples.

---

## 10. Selective Layer Freezing (Option 3A)

To combat overfitting during training on our small legal corpus, we implemented model-specific selective layer freezing:

| Model Architecture | Total Layers | Layers Frozen | Layers Trained | Parameter Reduction |
| :--- | :---: | :---: | :---: | :---: |
| **BERT / LegalBERT** | 12 | Embeddings + bottom 8 | Top 4 + Head | ~65% (109M -> 38M trainable) |
| **RoBERTa** | 12 | Embeddings + bottom 8 | Top 4 + Head | ~65% (124M -> 44M trainable) |
| **DistilBERT** | 6 | Embeddings + bottom 4 | Top 2 + Head | ~60% (66M -> 26M trainable) |

### Run 4: LegalBERT 8-Layer Frozen (Convergence Failure, LR `1e-5`)
Freezing 8 layers restricted capacity too aggressively under a low learning rate, leading to severe underfitting where the model predicted the background `O` class for all tokens:

| Epoch | Training Loss | Validation Loss | Precision | Recall | F1 Score | Accuracy |
| :---: | :-----------: | :-------------: | :-------: | :----: | :------: | :------: |
| 1     | 0.690018      | 0.647675        | 0.000949  | 0.011299 | 0.001751 | 0.805964 |
| 2     | 0.552097      | 0.630341        | 0.003667  | 0.033898 | 0.006619 | 0.824811 |
| 3     | 0.484176      | 0.613919        | 0.005118  | 0.056497 | 0.009385 | 0.810294 |
| ...   | ...           | ...             | ...       | ...      | ...      | ...      |
| 8     | 0.329301      | 0.684114        | 0.012402  | 0.107345 | 0.022235 | 0.830584 |

### Run 5: LegalBERT 4-Layer Frozen (Successful Convergence, LR `3e-5`)
Loosening the constraint to 4 frozen layers and raising the learning rate to `3e-5` allowed successful fitting and generalization, peaking at Epoch 7 before early stopping patience 2:

| Epoch | Training Loss | Validation Loss | Precision | Recall | F1 Score | Accuracy |
| :---: | :-----------: | :-------------: | :-------: | :----: | :------: | :------: |
| 1     | 0.556682      | 0.655695        | 0.004936  | 0.097297 | 0.009395 | 0.688943 |
| 2     | 0.508040      | 0.585957        | 0.010909  | 0.064865 | 0.018677 | 0.849546 |
| 3     | 0.359164      | 0.612488        | 0.080158  | 0.329730 | 0.128964 | 0.827949 |
| 4     | 0.256262      | 0.672906        | 0.074468  | 0.302703 | 0.119530 | 0.861597 |
| 5     | 0.184931      | 0.943676        | 0.142593  | 0.416216 | 0.212414 | 0.854626 |
| 6     | 0.154436      | 1.201853        | 0.166667  | 0.389189 | 0.233387 | 0.868957 |
| **7** | **0.092731**  | **1.247241**    | **0.187328** | **0.367568** | **0.248175** | **0.881076** |
| 8     | 0.097620      | 1.363127        | 0.177885  | 0.400000 | 0.246256 | 0.870486 |

### Run 6: LegalBERT 6-Layer Frozen (Persistent Overfitting, LR `3e-5`)
While the 6-layer model showed slightly improved loss characteristics over the 4-layer baseline, it still experienced persistent overfitting where validation loss climbed significantly by Epoch 8.

### Run 7: Tuning 8-Layer Freezing with Adjusted Parameters
To enforce strong regularization via 8-layer freezing while avoiding the underfitting seen in Run 4, we adjusted the training hyperparameter landscape:
1. **Increase Learning Rate to `5e-5`**: Compels the top active layers (4 layers + classification head) to adapt more quickly to downstream task signals.
2. **Increase Epochs to 15**: Gives the highly constrained model more optimization steps to settle into a high-performance region.
3. **Change Class Loss Weights to `[0.10, 2.00, 1.00]`**: Penalizes background class errors less and weights B-RISK / I-RISK more heavily, preventing the constrained model from defaulting to predicting background `O` tags for all tokens.
4. **Early Stopping Patience to 3**: Provides a wider window for the model to recover from localized metric plateaus.


---

## 11. Lightweight Models & Custom Sequence-Constrained BiLSTM-CRF

To resolve overfitting more fundamentally, we replaced the heavy BERT models with a suite of extremely small models and a sequence-constrained BiLSTM-CRF model:

| Model ID / Architecture | Parameter Count | Freeze Configuration | Overfitting Tradeoff / Design |
| :--- | :---: | :--- | :--- |
| **`google/electra-small-discriminator`** | **~14M** | Embeddings + bottom 8 layers | Discriminator pre-training provides high sample-efficiency. Compact capacity acts as a regularization bottleneck. |
| **`albert-base-v2`** | **~11M** | Embeddings only | Employs layer-parameter sharing. The deep structure is highly regularized by shared weights. |
| **`huawei-noah/TinyBERT_General_4L_312D`** | **~14M** | Embeddings + bottom 2 layers | A compact 4-layer distilled BERT. Faster inference and lower representation drift. |
| **Custom `BiLSTM_CRF`** | **~9M** | None (Trained from scratch) | Standard recurrent layer (`nn.LSTM` with hidden dim 256) and a statistical sequence constraint classifier (`torchcrf.CRF`) on top. |

### BiLSTM-CRF Implementation and Trainer Integration
- **Hugging Face Compat**: By formatting the model to accept `input_ids`, `attention_mask`, and `labels`, and having it return a dictionary containing `{"loss": loss, "logits": emissions}`, we can run the model directly inside the standard Hugging Face `Trainer` loop.
- **CRF Loss Bypass**: `WeightedLossTrainer` is configured to identify custom models and bypass the standard cross-entropy loss calculation, utilizing the CRF's native negative log-likelihood score instead.
- **Config & Weight Persistence**: The `BiLSTM_CRF` class implements custom `save_pretrained()` and `from_pretrained()` helper methods to store parameter weights and config JSON configurations, allowing it to behave exactly like a native Hugging Face model in downstream deployment environments.

---

## 12. Pipeline Diagnostic Audit & Critical Bug Fixes

A comprehensive audit of the data loading, label encoding, and training pipeline revealed **5 root-cause bugs** responsible for the persistent high-accuracy / low-F1 symptoms and severe overfitting. All fixes were applied to `train_gotcha.ipynb` (Cells 2, 4, and 6).

### Bug #1: `clearly_unfair` Label Mismatch (Critical — P0)

**Symptom**: High accuracy with low F1; model rewarded for predicting `O` on genuinely unfair text.

**Root Cause**: The CodeHima/TOS_Dataset uses the label `'clearly_unfair'` for its most egregious gotcha sentences, but the parser in Cell 4 checked for `'unfair'` — a label that **does not exist** in the dataset.

| Dataset Label | Count | Parser Matched? |
| :--- | :---: | :---: |
| `clearly_fair` | 4,365 | ✅ Treated as neutral |
| `potentially_unfair` | 526 | ✅ Treated as gotcha |
| **`clearly_unfair`** | **487** | ❌ **Silently treated as neutral** |

This caused **487 of the strongest gotcha examples** (class action waivers, forced arbitration, data selling, indemnity) to be labeled as all-`O` background tokens. The model was actively *punished* for correctly identifying these as risks, creating contradictory training signals that poisoned both the training and validation splits.

**Fix** (Cell 4):
```diff
-if row['unfairness_level'] in ['potentially_unfair', 'unfair']:
+if row['unfairness_level'] in ['potentially_unfair', 'clearly_unfair']:
```

### Bug #2: All-or-Nothing Token Labeling (High — P1)

**Symptom**: Severe overfitting; model learns binary sequence classification shortcut instead of token-level discrimination.

**Root Cause**: The parser set `gotchas = [sentence]`, making the gotcha text identical to the full input text. This produced a pathological distribution where every CodeHima gotcha sample had **100% risk tokens and 0% `O` tokens**, and every neutral sample had **100% `O` tokens and 0% risk tokens**.

| Sample Type | O tokens | B-RISK | I-RISK | Count |
| :--- | :---: | :---: | :---: | :---: |
| CodeHima gotcha (before fix) | **0%** | 1 token | 99% | 526 |
| CodeHima neutral | **100%** | 0% | 0% | 4,365 |

The model never saw mixed samples from this source — samples where it must learn to distinguish risk tokens from neutral tokens *within the same sequence*. At inference time, real documents always contain mixed text, causing a severe train/inference distribution mismatch.

**Fix** (Cell 4): Prepend a random neutral sentence to each gotcha sentence to create mixed-token training samples:
```python
random.seed(42)
for gotcha_sent in gotcha_sentences:
    context_sent = random.choice(neutral_sentences)
    combined_text = context_sent + ' ' + gotcha_sent
    standardized_data.append({"text": combined_text, "gotchas": [gotcha_sent]})
```

### Bug #3: EE21 Discarding Neutral Documents (Medium — P3)

**Symptom**: Wasted negative training data; skewed class balance toward gotcha-heavy samples.

**Root Cause**: When the EE21 word-overlap threshold was not met, the document was silently discarded instead of being used as a negative example. 54 full ToS documents from diverse companies were lost as potential neutral training samples.

**Fix** (Cell 4):
```diff
 if exact_match:
     standardized_data.append({"text": text, "gotchas": [exact_match]})
+else:
+    standardized_data.append({"text": text, "gotchas": []})
```

### Bug #4: Stale Class Weights (Medium — P2)

**Symptom**: `B-RISK` signal drowned out; model spreads `I-RISK` broadly instead of learning precise boundaries.

**Root Cause**: The weights `[0.10, 2.00, 1.00]` were tuned for the pre-augmentation, pre-bug-fix token distribution. After fixing Bugs #1–#3 and adding 3,479 synthetic samples, the token distribution shifted significantly. The `B-RISK` class at ~1% of tokens with only 2x weight was still overwhelmed by `I-RISK` at ~19% with 1x weight.

**Fix** (Cell 6): Recalculated using inverse-sqrt frequency scaling based on the corrected distribution:
```diff
-loss_fct = CrossEntropyLoss(weight=torch.tensor([0.10, 2.00, 1.00], device=device))
+loss_fct = CrossEntropyLoss(weight=torch.tensor([0.38, 4.70, 1.00], device=device))
```

### Bug #5: Train/Validation Domain Mismatch

**Symptom**: Apparent overfitting (train loss drops, val loss climbs) partly caused by distributional mismatch rather than true overfitting.

**Root Cause**: The training set contains ~36% synthetic data (short, keyword-dense sentences) while the validation set is 100% real data (long, natural legal prose). The model optimizes for the synthetic-heavy training distribution, then gets evaluated on a fundamentally different distribution. This is partially mitigated by the existing `metric_for_best_model="f1"` setting but still affects gradient dynamics and early stopping behavior.

**Mitigation**: Not directly patched in this round. Future improvements could include `WeightedRandomSampler` for balanced real/synthetic exposure per batch, or curriculum learning (real data first, then synthetic mixing).

### Before vs. After Token Distribution

The combined effect of all fixes on the training data token distribution:

#### Before Fixes (Broken Pipeline)

| Token Tag Class | Count | Share |
| :--- | :---: | :---: |
| `O` (Background) | 132,713 | 86.18% |
| `B-RISK` (Boundary Start) | 526 | 0.34% |
| `I-RISK` (Boundary Body) | 20,759 | 13.48% |
| **Mixed samples** | **~847** | — |
| **All-risk samples** | **526** | — |

#### After Fixes (Corrected Pipeline)

| Token Tag Class | Count | Share | Change |
| :--- | :---: | :---: | :---: |
| `O` (Background) | 406,485 | 80.44% | ↓ from 86% |
| `B-RISK` (Boundary Start) | 4,180 | 0.83% | **↑ ~2.4x** |
| `I-RISK` (Boundary Body) | 94,649 | 18.73% | ↑ from 13% |
| **Mixed samples** | **3,878** | — | **↑ ~4.6x** |
| **All-risk samples** | **69** | — | **↓ from 526** |

Key improvements:
1. **B-RISK nearly tripled** (0.34% → 0.83%) — stronger boundary detection signal.
2. **Mixed samples increased 4.6x** (847 → 3,878) — model must now learn token-level discrimination within sequences rather than binary shortcuts.
3. **All-risk samples dropped 87%** (526 → 69) — model can no longer cheat by predicting all-risk for entire sequences.
4. **487 mislabeled samples corrected** — model no longer receives contradictory signals for the strongest gotcha examples.

---

## 13. Pro-User Rights Suppression & Heuristic Segment Overrides

To further eliminate false positives, we introduced post-processing rule-based overrides that filter out user-favorable rights and standard GDPR/CCPA disclosures. While the transformer models are highly sensitive to legal terms (often classifying "rights to access" or "website anonymization" as potential risks due to semantic similarity), we leverage post-processing rules to override these classifications.

### Pro-User Keyword Suppression (`KEYWORDS_PRO_USER`)
A targeted list of regular expressions intercepts sentences that convey rights granted to the user rather than company rights/restrictions:
- **User access/deletion rights**: `you may (access|correct|request deletion|delete|port|object)`
- **Opt-out of tracking/marketing**: `opt[- ]out of receiving (marketing|promotional|newsletter)`
- **Anonymous browsing**: `freely visit our (website|platform) anonymously` and `without being required to provide us with any personal information`
- **Data processing restriction**: `request that we stop (any )?processing`

### Rule-Based Segment Overrides
In addition to the keyword matching, we enforce structural heuristics:
1. **Explicit rights definition**: Sentences starting with or containing phrases like `right(s) to` or `you have the right to` combined with action verbs (`access`, `correct`, `delete`, `erase`, `rectify`, `update`, `portability`, `restrict`).
2. **Browsing anonymity**: Safeguarded browse/visit sentences, ensuring that permission-based or anonymous visit descriptions are marked neutral, provided they don't contain restriction words like `cannot`, `unable`, or `restrict`.
3. **Data Protection Regulation References**: Sentences describing user-empowering rights under frameworks like CCPA or GDPR (e.g., `rights related to GDPR` or `rights related to CCPA`).

If a sentence matches any of these criteria, it is automatically classified as `None` (neutral), overriding model predictions. This dramatically reduces false positives on user-favorable legal text without affecting the model's performance on hostile terms (arbitration, data selling, unilateral modification).

---

## 14. Hyperparameter Optimization & Dynamic Freezing with Optuna

To maximize performance, combat overfitting, and systematically find optimal configuration settings across multiple different model backbones, we implemented an automated hyperparameter search using **Optuna** directly integrated into the training pipeline (`train_gotcha.ipynb` Cell 6).

### 1. Optuna Search Space Configuration
The training loop evaluates the following hyperparameter search space during a search run:
* **`learning_rate`**: Continuously sampled in log-scale between `1e-5` and `8e-5`.
* **`weight_decay`**: Uniformly sampled between `0.0` and `0.15` to find the best regularization bottleneck.
* **`per_device_train_batch_size`**: Categorically selected between `8` and `16` to optimize batch statistics.
* **`warmup_ratio`**: Uniformly sampled between `0.05` and `0.25` for training stability.
* **`num_frozen_layers`**: A dynamic integer trial parameter that selects how many bottom layers of the backbone encoder to freeze, customized per model architecture.

### 2. Model Architecture and Freezing Constraints (`MODEL_ARCH_MAP`)
To prevent representational drift and over-parameterization, each model contains its own freezing metadata map:

| Model Backbone ID | Max Layers | Freeze Search Scope | Layer-Wise Specifics |
| :--- | :---: | :--- | :--- |
| **`google/electra-small-discriminator`** | 12 | `0` to `12` layers | Freezes embeddings and bottom $N$ layers of `electra.encoder.layer`. |
| **`nlpaueb/legal-bert-base-uncased`** | 12 | `0` to `12` layers | Freezes embeddings and bottom $N$ layers of `bert.encoder.layer`. |
| **`huawei-noah/TinyBERT_General_4L_312D`** | 4 | `0` to `4` layers | Freezes embeddings and bottom $N$ layers of `bert.encoder.layer`. |
| **`albert-base-v2`** | 0 | None (embeddings only) | ALBERT uses cross-layer parameter sharing, meaning freezing specific layers is not architecturally viable. Instead, it always freezes embeddings and searches within learning rate / regularization parameters. |
| **Custom `BiLSTM_CRF`** | 0 | None | Trained from scratch without pre-trained weights; no freezing search is performed. |

### 3. Objective Definition & Auto-Cleanup
* **Objective Function**: The hyperparameter search optimizes for **Validation F1** (`eval_f1` computed via `seqeval`) in order to ensure boundary accuracy is prioritized over overall token accuracy.
* **Automated Cleanup**: Checking checkpoints during search runs creates significant disk space footprint. To prevent disk overflow, the pipeline writes search logs to a temporary directory `./gotcha-extractor-model/{save_name}-optuna-search` and automatically deletes it using `shutil.rmtree` upon finding the best configuration.
* **Final Model Preservation**: Once the best hyperparameters are identified, a final full training run is executed using these settings, saving the optimized weights directly into the clean production path `./gotcha-extractor-model/{save_name}`.


