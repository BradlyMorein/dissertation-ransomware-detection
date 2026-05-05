# ADR 0004 — Random Forest baseline methodology

**Status:** Accepted
**Date:** 5 May 2026
**Decided by:** Bradly Morein (supervisor confirmed verbally on 2 May 2026: Comfortable to support my decisions made)

## Context

Random Forest is the first of three model architectures evaluated in this
dissertation, serving as the supervised classifier baseline against which
the Autoencoder (unsupervised) and LSTM (supervised, temporal) models are
compared. The Random Forest training pipeline produces:

1. The headline numbers for RQ1 (binary attack classification on CIC-IDS2017).
2. The trained model that will be applied to CIC-UNSW-NB15 in Week 7 for
   RQ3 (cross-dataset generalisation).

Several methodology choices needed to be made: hyperparameter approach,
class-imbalance handling, probability calibration, feature subset, and
inference latency reporting. This ADR documents these decisions and their
rationale.

## Decisions

### 1. Hyperparameters

`n_estimators=200`, `max_depth=20`, all other parameters at sklearn defaults.

### 2. Class imbalance handling

`class_weight="balanced"` — sklearn auto-weights classes inversely
proportional to their frequency. The dataset's 78.89% / 21.11%
benign/attack ratio is moderate (3.7:1) and balanced weighting is the
standard tool for it.

### 3. Probability calibration

None applied. Default Random Forest probabilities are used directly for
ROC-AUC and PR-AUC computation, and the standard 0.5 threshold for hard
classification.

### 4. Feature subset

71 features. Drops 9 columns from the cleaned CIC-IDS2017 (79 columns):
- 5 string identifiers: `Flow ID`, `Src IP`, `Dst IP`, `Timestamp`, `Label`
- 1 derived label: `is_attack` (the target)
- 3 routing/protocol fields: `Src Port`, `Dst Port`, `Protocol`

The 3 routing fields are absent from CIC-UNSW-NB15. Dropping them from
training preserves cross-dataset feature alignment for RQ3, allowing the
same trained model to be applied to UNSW without reshaping.

### 5. Inference latency reporting

Two latency measurements are reported, with explicit acknowledgement of
their interpretive limits:

- **Bulk inference latency:** 314,761 flows predicted in 1.03s
  → 3.26 μs/flow average.
- **Per-call inference latency:** 100 single-flow predictions, hot cache,
  median 24.8 ms/flow, p95 26.9 ms/flow.

The 7,000× gap between the two measurements is due to sklearn's per-call
overhead (Python interpreter, parallel-backend setup) being amortised
across batch predictions. Real NIDS deployment falls between the two
extremes, depending on the polling interval and batch size used by the
flow-collection layer.

### 6. Random seed

`random_state=42`, matching the splitting seed in `split.py` and ADR 0003.
Ensures full reproducibility from clean repository state.

## Rationale

### Hyperparameters

Three approaches considered: sklearn defaults (`n_estimators=100,
max_depth=None`), sensible reasonable parameters (chosen), and grid
search with cross-validation.

`n_estimators=200` rather than the 100 default: more trees reduce
variance in predictions at modest training-time cost. With 1.46M
training rows and 71 features, 200 trees fitted in 234s — within budget.

`max_depth=20` rather than unlimited: prevents over-fitting. With
unlimited depth and 1.46M training rows, individual trees would memorise
the training data and degrade test performance. 20 is conventional for
tabular data of this size.

Grid search would have improved metrics by perhaps 1-2% but added
~3 hours of training time and complicated the methodology description.
Out of scope for an MSc with the schedule constraint.

### Class weighting

The 3.7:1 imbalance is moderate, where `class_weight="balanced"` is the
standard tool. SMOTE oversampling was considered and rejected as overkill
at this imbalance ratio (typically used at 50:1 or worse).

### Probability calibration

Random Forest probabilities are natively poorly calibrated (they tend to
overestimate confidence). Platt scaling or isotonic regression on the
validation fold would improve threshold tuning and PR-AUC interpretation.
Skipped for MSc scope; mentioned as future work in §7.4.

### Feature subset alignment

71 features for the trained model rather than the 79 in cleaned
CIC-IDS2017 prioritises cross-dataset evaluation in Week 7 over
maximum CIC-IDS2017-only performance. Quantitatively the trade-off is
small — the 3 dropped routing fields (`Src Port`, `Dst Port`, `Protocol`)
provide identifying information rather than discriminative information
about flow behaviour.

### Latency reporting

The 7,000× gap between bulk and per-call latency is a real methodological
issue, not a bug. Two principled options were considered:

- Build a batch-size sweep curve (1, 10, 100, 1000, 10000 flow batches)
  and report the curve. Most rigorous.
- Report both bulk and per-call extremes with explicit explanation.
  Honest, less work.

Option B was chosen for schedule reasons. The Methodology chapter §4.6
explicitly acknowledges that real-world deployment latency falls between
the two extremes and is determined by the deployment layer's batching
policy, which is outside the scope of this work.

## Consequences

- **Reproducibility.** Anyone with the repository, environment, and
  data can produce the same model and same metrics. Random seed 42 is
  fixed in `random_forest.py` and recorded in `_metrics_manifest.json`.

- **Cross-dataset evaluation enabled.** The trained model accepts
  71-feature input matching CIC-UNSW-NB15's schema. Week 7's
  cross-dataset experiment loads `experiments/rf/model.joblib` and
  applies it directly to the UNSW Parquet without retraining.

- **Methodology chapter 4.6** must report all six decisions explicitly,
  with rationale. Each decision becomes a paragraph or bullet point in
  the chapter draft.

- **Limitations section 7.4** must mention probability calibration as
  future work, latency-batch-size sweep as future work, and the
  trade-off between cross-dataset alignment and maximum
  CIC-IDS2017-only performance.

## Implementation

- Module: `src/models/random_forest.py`
- Shared metrics: `src/evaluation/metrics.py`
- Tests: `tests/test_random_forest.py` (21 tests covering feature
  selection, hyperparameter contract, binary metric arithmetic, and
  per-class metric edge cases)
- Manifest: `experiments/rf/_metrics_manifest.json`
- Trained model: `experiments/rf/model.joblib` (21MB, gitignored;
  reproducible via `python -m src.models.random_forest`)
- Predictions: `experiments/rf/predictions.npz` (847KB, committed)

## Headline results (test set, n=314,761)

| Metric | Value |
|---|---:|
| Accuracy | 0.9917 |
| Precision (attack) | 0.9626 |
| Recall (attack) | 0.9995 |
| F1 (attack) | 0.9807 |
| ROC-AUC | 0.9998 |
| PR-AUC | 0.9993 |
| FPR | 0.0104 |
| Bulk inference latency | 3.26 μs/flow |
| Per-call inference latency (median) | 24.8 ms/flow |
| Training time (1.47M rows × 71 features × 200 trees) | 234s |