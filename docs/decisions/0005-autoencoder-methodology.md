# ADR 0005 — Autoencoder methodology

**Status:** Accepted
**Date:** 6 May 2026
**Decided by:** Bradly Morein (supervisor confirmed verbally on 2 May 2026: Comfortable to support my decisions made)

## Context

The autoencoder is the second of three model architectures evaluated in
my dissertation, providing an unsupervised contrast to the supervised
Random Forest baseline (ADR 0004) and the supervised temporal LSTM model
(ADR 0006, forthcoming). The autoencoder learns a compressed
representation of benign network flow features and detects attacks as
flows that reconstruct poorly through the learned manifold.

Key methodology decisions: framework choice, network architecture,
training-data filter, feature scaling, and anomaly threshold rule.

## Decisions

### 1. Framework

`sklearn.neural_network.MLPRegressor`, used as a symmetric autoencoder
by passing the input as both X and y. Same library as the Random Forest
baseline, which preserves consistency in the comparative framework.

### 2. Architecture

Symmetric encoder-decoder: `71 → 32 → 16 → 32 → 71`.

The 16-dimensional bottleneck (≈22% of input dimension) follows the
convention of compressing to between 1/4 and 1/8 of the input
dimensionality. ReLU activations on hidden layers, identity on output.

### 3. Training data filter

**Monday-only benign flows** (~260K rows in the train fold of
CIC-IDS2017). Monday is the only attack-free day in the published
dataset, so this filter eliminates the risk of training-data
contamination from misslabeled or borderline flows on Tue–Fri.

The filter `filter_monday_benign()` defensively checks both
day-of-week == Monday AND `is_attack == 0`, even though the second
condition should be redundant.

### 4. Feature preprocessing

A two-step pipeline: `log1p` followed by `StandardScaler`.

Network flow features are heavily right-skewed: most values cluster
near zero, with long tails of large values (typical for byte counts,
packet counts, and durations). `StandardScaler` alone produces
catastrophic R² values during model fitting because feature means are
dominated by outliers. `RobustScaler` fails for the opposite reason:
features with near-zero IQR get amplified.

`log1p(x) = log(1 + x)` flattens the right skew while preserving
zero-valued inputs. Subsequent `StandardScaler` produces well-conditioned
inputs to the network.

**Five features required clipping to non-negative values before log1p:**
`Flow Duration`, `Flow Packets/s`, `Flow IAT Mean`, `Flow IAT Max`,
`Flow IAT Min`. These are CICFlowMeter packet-ordering artifacts where
out-of-order packets produce mathematically nonsensical negative timing
values. The Engelen et al. (2021) paper documents these artifacts
without correcting them. Clipping to zero replaces nonsense values with
the closest sensible value.

The fitted pipeline (FunctionTransformer + StandardScaler) is saved as
part of the model bundle so the same transform can be applied to
unseen data, including the cross-dataset CIC-UNSW-NB15 evaluation in
Week 7.

### 5. Anomaly threshold rule

The 95th percentile of validation benign reconstruction errors. By
construction, this gives FPR ≈ 0.05.

This rule never sees attack labels in validation, preserving the
unsupervised premise: the threshold is set without knowledge of what
attacks "look like." This is the cleanest defensible threshold for
unsupervised anomaly detection.

The downside is acknowledged: more aggressive thresholds (Youden's J,
ROC knee-point) using validation attack labels would improve recall but
weaken the unsupervised story.

### 6. Random seed

`random_state=42`, matching the splitting seed in `split.py` and the
RF seed in ADR 0004.

## Consequences

- **Headline test-set metrics (n=314,761):**

  | Metric | Value |
  |---|---:|
  | Accuracy | 0.8318 |
  | Precision (attack) | 0.6749 |
  | Recall (attack) | 0.3922 |
  | F1 (attack) | 0.4961 |
  | ROC-AUC | 0.9204 |
  | PR-AUC | 0.5925 |
  | FPR | 0.0506 |
  | Threshold | 0.0970 |

  The ROC-AUC of 0.92 indicates that the autoencoder's reconstruction
  errors effectively rank attack flows above benign flows in the
  validation set — the model has learned to distinguish "normal" from
  "anomalous." However, the threshold rule produces conservative recall
  (39%): many attack flows fall within the benign reconstruction-error
  distribution. This is the **expected cost of unsupervised threshold
  setting**, not a model failure.

- **Comparative implications for RQ1.** The headline contrast with RF
  (recall 99.95%) demonstrates the value of supervision when labelled
  attack data is available during training. The AE's ROC-AUC of 0.92
  is the rigorous comparable metric — both models rank flows similarly
  well, but the binary thresholding regime is what differs.

- **Methodology chapter 4.7 must explicitly report:**
  - The five-feature CICFlowMeter clipping decision.
  - The log1p preprocessing rationale.
  - The 95th percentile threshold rule and its trade-off.
  - The interpretive distinction between rank-based metrics (ROC-AUC,
    PR-AUC) and threshold-dependent metrics (precision, recall, F1).

- **Limitations section 7.4** mentions:
  - Probability calibration of reconstruction error (e.g., Platt
    scaling on a separate held-out set) as future work.
  - Threshold sensitivity analysis (sweep across percentiles) as
    future work.
  - Comparison with deeper or wider AE architectures as future work.

## Alternatives considered

**Alternative 1 — RobustScaler instead of log1p+StandardScaler.**
Tested empirically. Produced loss values in the billions because
features with near-zero IQR (e.g., `Bwd PSH Flags` with mostly-zero
values) get amplified by 1/IQR ≈ 8.5×, while broad features get
divided down. Caused training instability. Rejected.

**Alternative 2 — StandardScaler alone (no log1p).**
Tested empirically. Produced catastrophic R² values during sklearn's
internal validation (R² ≈ -10²⁵). Outliers in feature distributions
dominated reconstruction error and triggered early stopping after only
12 iterations. Rejected.

**Alternative 3 — Youden's J or ROC-knee threshold.** Would use
validation attack labels to optimise the threshold for maximum
(TPR - FPR). Would substantially improve binary metrics (estimated
recall 0.85+) but weaken the unsupervised story. Treated as future
work in 7.4.

**Alternative 4 — Deeper architecture (extra bottleneck layers).**
Would force a sharper anomaly signal but increase training time and
risk of non-convergence. Rejected for schedule reasons; mentioned in
future work.

**Alternative 5 — All benign rows from train fold (not Monday-only).**
Would provide more training data (~1.16M flows vs 260K) but introduce
risk of contamination from borderline-labelled flows on Tue–Fri.
Rejected — the unsupervised story is cleaner with Monday-only.

## Implementation

- Module: `src/models/autoencoder.py`
- Tests: `tests/test_autoencoder.py` (21 tests covering Monday filtering,
  feature selection, reconstruction-error helper, threshold rule, and
  hyperparameter contract)
- Manifest: `experiments/ae/_metrics_manifest.json`
- Trained model + scaler bundle: `experiments/ae/model.joblib` (gitignored;
  reproducible via `python -m src.models.autoencoder`)
- Predictions: `experiments/ae/predictions.npz` (committed)
- Reconstruction errors: `experiments/ae/errors.npz` (committed; required
  for the threshold-sensitivity analysis in notebook 06)

## References

- Engelen, G., Rimmer, V., & Joosen, W. (2021). Troubleshooting an
  Intrusion Detection Dataset: the CICIDS2017 Case Study. IEEE Security
  and Privacy Workshops.
- Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward
  generating a new intrusion detection dataset and intrusion traffic
  characterization. ICISSP.