"""
autoencoder.py — Unsupervised autoencoder for NIDS anomaly detection.

Trains a feed-forward autoencoder on Monday-only benign flows from
Engelen-cleaned CIC-IDS2017. Attacks are detected at evaluation time
as flows whose reconstruction error exceeds a threshold derived from
the validation set's benign reconstruction-error distribution.

Methodology decisions (see ADR 0005 — written after this module):
- Framework: sklearn MLPRegressor (consistency with the RF baseline).
- Architecture: 71 -> 32 -> 16 -> 32 -> 71 (compressed bottleneck).
- Training data: Monday-only benign (~260K flows in train fold);
  Monday is the only attack-free day in CIC-IDS2017, eliminating
  contamination from misslabeled attack flows.
- Feature scaling: StandardScaler fitted on training data only.
- Threshold: 95th percentile of validation benign reconstruction error.
  This rule never sees attack labels, preserving the unsupervised story.
- Random seed: 42 (matches RF and split.py).

Run from the repository root:

    python -m src.models.autoencoder

Outputs:
- Trained model + scaler at experiments/ae/model.joblib (joblib bundle)
- Reconstruction errors at experiments/ae/errors.npz
- Predictions and probabilities at experiments/ae/predictions.npz
- Metrics manifest at experiments/ae/_metrics_manifest.json
"""

from pathlib import Path
from dataclasses import asdict
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.evaluation.metrics import (
    BinaryMetrics,
    compute_binary_metrics,
    compute_per_class_metrics,
    measure_inference_latency,
)


# ---------- Configuration ---------- #

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "data" / "processed" / "cicids2017-engelen"
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "ae"

TRAIN_PATH = SPLITS_DIR / "train.parquet"
VAL_PATH   = SPLITS_DIR / "val.parquet"
TEST_PATH  = SPLITS_DIR / "test.parquet"

MODEL_PATH       = EXPERIMENT_DIR / "model.joblib"
ERRORS_PATH      = EXPERIMENT_DIR / "errors.npz"
PREDICTIONS_PATH = EXPERIMENT_DIR / "predictions.npz"
MANIFEST_PATH    = EXPERIMENT_DIR / "_metrics_manifest.json"

# Same drop list as the RF baseline (ADR 0004) — preserves cross-dataset alignment.
DROP_FOR_FEATURES = [
    "Flow ID", "Src IP", "Dst IP", "Timestamp",
    "Label", "is_attack",
    "Src Port", "Dst Port", "Protocol",
]

# Methodology per Decision 1-4 / ADR 0005.
HYPERPARAMS = dict(
    hidden_layer_sizes=(32, 16, 32),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=200,
    early_stopping=False,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42,
    verbose=True,
)

# Threshold rule (Decision 5 / ADR 0005)
BENIGN_PERCENTILE_FOR_THRESHOLD = 95.0

# Timestamp format used in CIC-IDS2017 / Engelen output
TIMESTAMP_FORMAT = "%d/%m/%Y %I:%M:%S %p"


# ---------- Feature selection and Monday filter ---------- #

def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Same contract as random_forest.select_features, kept here for clarity.

    Returns
    -------
    X : DataFrame of 71 features.
    y_binary : Series of 0/1 (is_attack).
    y_multi : Series of strings (original Label column).
    """
    if "is_attack" not in df.columns:
        raise ValueError("Input DataFrame must have an 'is_attack' column.")
    if "Label" not in df.columns:
        raise ValueError("Input DataFrame must have a 'Label' column.")
    feature_cols = [c for c in df.columns if c not in DROP_FOR_FEATURES]
    X = df[feature_cols].copy()
    y_binary = df["is_attack"].astype(int)
    y_multi = df["Label"].astype(str)
    return X, y_binary, y_multi


def filter_monday_benign(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to Monday-only benign rows for autoencoder training.

    Parses the Timestamp column, filters to (1) day-of-week == Monday
    AND (2) is_attack == 0. Both conditions are applied even though
    Monday is supposed to be attack-free, as a defensive check.

    Returns the filtered DataFrame.
    """
    if "Timestamp" not in df.columns:
        raise ValueError("Input DataFrame must have a 'Timestamp' column.")

    timestamps = pd.to_datetime(df["Timestamp"], format=TIMESTAMP_FORMAT, errors="coerce")
    if timestamps.isna().any():
        n_failed = int(timestamps.isna().sum())
        raise ValueError(f"{n_failed} timestamps failed to parse.")

    mask = (timestamps.dt.day_name() == "Monday") & (df["is_attack"] == 0)
    return df.loc[mask].copy()


# ---------- Reconstruction-error helpers ---------- #

def reconstruction_error(model: MLPRegressor, X_scaled: np.ndarray) -> np.ndarray:
    """Per-sample mean-squared reconstruction error.

    For each row, returns the mean of squared per-feature deltas between
    the input and the model's reconstruction. Lower = better fit.
    """
    X_recon = model.predict(X_scaled)
    sq_err = (X_scaled - X_recon) ** 2
    return sq_err.mean(axis=1)


def threshold_from_benign_percentile(
    val_errors: np.ndarray,
    val_is_attack: np.ndarray,
    percentile: float = BENIGN_PERCENTILE_FOR_THRESHOLD,
) -> float:
    """Derive an anomaly threshold from the validation set.

    Computes the k-th percentile of reconstruction errors *over benign
    validation rows only*. Attacks rows in val are not used to select
    the threshold, preserving the unsupervised story.
    """
    benign_errors = val_errors[val_is_attack == 0]
    if len(benign_errors) == 0:
        raise ValueError("No benign rows in validation set.")
    return float(np.percentile(benign_errors, percentile))

# ---------- Inference wrapper ---------- #

class AutoencoderClassifier:
    """Wraps model, scaler and threshold behind a predict() interface.

    Latency is measured end-to-end so the figure is comparable with the
    Random Forest, whose predict() likewise covers its full inference path.
    Timing the MLPRegressor alone would omit scaling and error computation,
    which a deployed detector would still have to perform per flow.
    """

    def __init__(self, model: MLPRegressor, scaler: Pipeline, threshold: float):
        self.model = model
        self.scaler = scaler
        self.threshold = threshold

    def predict(self, X) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        errors = reconstruction_error(self.model, X_scaled)
        return (errors > self.threshold).astype(int)


# ---------- Training ---------- #

def train_autoencoder(
    X_train_scaled: np.ndarray,
    hyperparams: dict | None = None,
) -> tuple[MLPRegressor, float]:
    """Fit MLPRegressor as an autoencoder by setting target = input.

    Returns
    -------
    model : fitted MLPRegressor
    fit_seconds : float
    """
    params = dict(HYPERPARAMS)
    if hyperparams is not None:
        params.update(hyperparams)

    print(f"Training MLPRegressor (autoencoder) with parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    model = MLPRegressor(**params)
    t0 = time.time()
    # Autoencoder: target == input
    model.fit(X_train_scaled, X_train_scaled)
    fit_seconds = time.time() - t0
    print(f"Training complete in {fit_seconds:.1f}s.")
    print(f"  Final training loss: {model.loss_:.6f}")
    print(f"  Iterations completed: {model.n_iter_}")
    return model, fit_seconds


# ---------- CLI entry point ---------- #

def main() -> int:
    if not TRAIN_PATH.exists():
        print(f"ERROR: train split not found at {TRAIN_PATH}", file=sys.stderr)
        return 1

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Splits directory: {SPLITS_DIR}")
    print(f"Experiment directory: {EXPERIMENT_DIR}")
    print()

    # ---- Load splits ---- #
    print("Loading splits...")
    df_train = pd.read_parquet(TRAIN_PATH)
    df_val   = pd.read_parquet(VAL_PATH)
    df_test  = pd.read_parquet(TEST_PATH)
    print(f"  train: {len(df_train):,} rows")
    print(f"  val:   {len(df_val):,} rows")
    print(f"  test:  {len(df_test):,} rows")

    # ---- Filter Monday benign for AE training ---- #
    print(f"\nFiltering training data: Monday-only benign...")
    df_monday = filter_monday_benign(df_train)
    print(f"  Monday benign in train: {len(df_monday):,} rows ({100*len(df_monday)/len(df_train):.2f}%)")

    X_monday, _, _ = select_features(df_monday)
    X_val,    y_val_bin,   _           = select_features(df_val)
    X_test,   y_test_bin,  y_test_multi = select_features(df_test)
    print(f"  Feature count: {X_monday.shape[1]}")

    # Clip negative-valued features to >= 0 — these are CICFlowMeter
    # packet-ordering artifacts (out-of-order packets producing negative
    # durations, IATs, and rates). log1p requires non-negative input.
    # See Engelen et al. 2021 for documentation of these artifacts.
    NEGATIVE_VALUE_FEATURES = [
        "Flow Duration",
        "Flow Packets/s",
        "Flow IAT Mean",
        "Flow IAT Max",
        "Flow IAT Min",
    ]
    for X in (X_monday, X_val, X_test):
        for col in NEGATIVE_VALUE_FEATURES:
            if col in X.columns:
                X[col] = X[col].clip(lower=0)
    print(f"  Clipped {len(NEGATIVE_VALUE_FEATURES)} CICFlowMeter-artifact features to >= 0 in all three folds.")

    # ---- Fit scaler on Monday training only ---- #
    print(f"\nFitting log1p + StandardScaler pipeline on Monday training data...")
    # Network flow features are right-skewed (heavy tail of large values).
    # log1p flattens the skew; StandardScaler then centres+scales.
    # log1p(x) = log(1 + x), accepts non-negative input and treats zero as zero.
    scaler = Pipeline([
        ("log1p", FunctionTransformer(np.log1p, validate=True)),
        ("standard", StandardScaler()),
    ])
    X_monday_scaled = scaler.fit_transform(X_monday)
    X_val_scaled    = scaler.transform(X_val)
    X_test_scaled   = scaler.transform(X_test)

    # Inspect post-pipeline distribution
    inner = scaler.named_steps["standard"]
    print(f"  After log1p+StandardScaler:")
    print(f"    Scaler mean range:  [{inner.mean_.min():.2f}, {inner.mean_.max():.2f}]")
    print(f"    Scaler scale range: [{inner.scale_.min():.4g}, {inner.scale_.max():.4g}]")
    print(f"    Train data range:   [{X_monday_scaled.min():.2f}, {X_monday_scaled.max():.2f}]")

    # ---- Train ---- #
    print()
    model, fit_seconds = train_autoencoder(X_monday_scaled)

    # ---- Compute reconstruction errors ---- #
    print(f"\nComputing reconstruction errors...")
    train_errors = reconstruction_error(model, X_monday_scaled)
    val_errors   = reconstruction_error(model, X_val_scaled)
    test_errors  = reconstruction_error(model, X_test_scaled)
    print(f"  Train (benign only): median {np.median(train_errors):.5f}, p95 {np.percentile(train_errors, 95):.5f}")
    print(f"  Val (mixed):         median {np.median(val_errors):.5f}, p95 {np.percentile(val_errors, 95):.5f}")
    print(f"  Test (mixed):        median {np.median(test_errors):.5f}, p95 {np.percentile(test_errors, 95):.5f}")

    # ---- Threshold from validation benign 95th percentile ---- #
    threshold = threshold_from_benign_percentile(
        val_errors=val_errors,
        val_is_attack=y_val_bin.values,
        percentile=BENIGN_PERCENTILE_FOR_THRESHOLD,
    )
    print(f"\nAnomaly threshold (val benign p{BENIGN_PERCENTILE_FOR_THRESHOLD:g}): {threshold:.5f}")

    # ---- Predict on test set ---- #
    y_test_pred = (test_errors > threshold).astype(int)
    # Use raw error as the "probability"-equivalent for ROC-AUC / PR-AUC
    # ROC-AUC and PR-AUC are rank-based and don't require values in [0, 1]
    y_test_proba = test_errors

        # ---- Metrics ---- #
    print(f"\nEvaluating on test set (n={len(X_test):,})...")

    print(f"  Measuring per-flow inference latency (n=100)...")
    classifier = AutoencoderClassifier(model, scaler, threshold)
    inference_ms = measure_inference_latency(classifier, X_test, n_repeats=100)
    print(f"  Median per-flow latency: {np.median(inference_ms):.3f} ms")

    binary_metrics = compute_binary_metrics(
        y_true=y_test_bin.values,
        y_pred=y_test_pred,
        y_proba=y_test_proba,
        inference_times_ms=inference_ms,
    )

    # Per-class breakdown
    multi_labels = sorted(y_test_multi.unique(), key=lambda lbl: -(y_test_multi == lbl).sum())
    per_class = compute_per_class_metrics(
        y_true_multi=y_test_multi.values,
        y_pred_binary=y_test_pred,
        multi_class_labels=multi_labels,
    )

    # ---- Save artefacts ---- #
    print(f"\nSaving artefacts...")

    # Model + scaler bundled together so they always travel as a pair
    print(f"  Model + scaler -> {MODEL_PATH.relative_to(REPO_ROOT)}")
    joblib.dump({"model": model, "scaler": scaler, "threshold": threshold}, MODEL_PATH)

    print(f"  Errors -> {ERRORS_PATH.relative_to(REPO_ROOT)}")
    np.savez_compressed(
        ERRORS_PATH,
        train_errors=train_errors,
        val_errors=val_errors,
        test_errors=test_errors,
    )

    print(f"  Predictions -> {PREDICTIONS_PATH.relative_to(REPO_ROOT)}")
    np.savez_compressed(
        PREDICTIONS_PATH,
        predictions=y_test_pred,
        probabilities=y_test_proba,
        y_true=y_test_bin.values,
    )

    manifest = {
        "tool": "src/models/autoencoder.py",
        "splits_directory": str(SPLITS_DIR.relative_to(REPO_ROOT)),
        "experiment_directory": str(EXPERIMENT_DIR.relative_to(REPO_ROOT)),
        "hyperparameters": {k: v for k, v in HYPERPARAMS.items() if k != "verbose"},
        "n_features": int(X_monday.shape[1]),
        "training": {
            "filter": "Monday-only benign",
            "n_train_samples": int(len(X_monday)),
            "fit_seconds": round(fit_seconds, 2),
            "iterations_completed": int(model.n_iter_),
            "final_loss": float(model.loss_),
        },
        "threshold": {
            "method": f"{BENIGN_PERCENTILE_FOR_THRESHOLD}th percentile of validation benign error",
            "value": float(threshold),
            "n_val_benign_used": int((y_val_bin == 0).sum()),
        },
        "test": {
            "binary": asdict(binary_metrics),
            "per_class": per_class,
        },
    }

    print(f"  Manifest -> {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    # ---- Headline summary ---- #
    bm = manifest["test"]["binary"]
    print(f"\n{'=' * 70}")
    print(f"HEADLINE METRICS — Autoencoder on CIC-IDS2017 test set")
    print(f"{'=' * 70}")
    print(f"  Threshold:          {threshold:.5f}")
    print(f"  Accuracy:           {bm['accuracy']:.4f}")
    print(f"  Precision (attack): {bm['precision']:.4f}")
    print(f"  Recall (attack):    {bm['recall']:.4f}")
    print(f"  F1 (attack):        {bm['f1']:.4f}")
    print(f"  ROC-AUC:            {bm['roc_auc']:.4f}")
    print(f"  PR-AUC:             {bm['pr_auc']:.4f}")
    print(f"  FPR:                {bm['fpr']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())