"""
random_forest.py — Random Forest baseline for NIDS binary classification.

Trains a stratified Random Forest on the 71-feature aligned subset of
Engelen-cleaned CIC-IDS2017 (matching the feature schema of CIC-UNSW-NB15
for cross-dataset evaluation).

Methodology decisions (see ADR 0004 — written after this module):
- Hyperparameters: n_estimators=200, max_depth=20.
- Class weighting: balanced (handles 3.7:1 imbalance).
- Probability calibration: none (default RF output, future work).
- Random seed: 42 for reproducibility.

Run from the repository root:

    python -m src.models.random_forest

Outputs:
- Trained model serialised at experiments/rf/model.joblib
- Predictions and probabilities at experiments/rf/predictions.npz
- Metrics manifest at experiments/rf/_metrics_manifest.json
"""

from pathlib import Path
import json
import sys
import time
from dataclasses import asdict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.evaluation.metrics import (
    BinaryMetrics,
    compute_binary_metrics,
    compute_per_class_metrics,
    measure_inference_latency,
)


# ---------- Configuration ---------- #

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "data" / "processed" / "cicids2017-engelen"
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "rf"

TRAIN_PATH = SPLITS_DIR / "train.parquet"
VAL_PATH   = SPLITS_DIR / "val.parquet"
TEST_PATH  = SPLITS_DIR / "test.parquet"

MODEL_PATH       = EXPERIMENT_DIR / "model.joblib"
PREDICTIONS_PATH = EXPERIMENT_DIR / "predictions.npz"
MANIFEST_PATH    = EXPERIMENT_DIR / "_metrics_manifest.json"

# Columns to drop from training data: identifiers, derived labels,
# and routing fields not present in CIC-UNSW-NB15.
DROP_FOR_FEATURES = [
    "Flow ID", "Src IP", "Dst IP", "Timestamp",   # identifiers
    "Label", "is_attack",                         # labels (target, derived)
    "Src Port", "Dst Port", "Protocol",           # absent in UNSW; drop for cross-dataset alignment
]

# Hyperparameters per Decision 2 / ADR 0004
HYPERPARAMS = dict(
    n_estimators=200,
    max_depth=20,
    class_weight="balanced",
    n_jobs=-1,           # use all CPU cores
    random_state=42,
    verbose=1,           # prints progress during fit
)


# ---------- Feature selection ---------- #

def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split a DataFrame into features / binary target / multi-class label.

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


# ---------- Training ---------- #

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    hyperparams: dict | None = None,
) -> tuple[RandomForestClassifier, float]:
    """Fit a RandomForestClassifier with the configured hyperparameters.

    Returns
    -------
    model : trained estimator
    fit_seconds : float
        Wall-clock time spent fitting.
    """
    params = dict(HYPERPARAMS)
    if hyperparams is not None:
        params.update(hyperparams)

    print(f"Training RandomForestClassifier with parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    model = RandomForestClassifier(**params)
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_seconds = time.time() - t0
    print(f"Training complete in {fit_seconds:.1f}s.")
    return model, fit_seconds


# ---------- Evaluation ---------- #

def evaluate_on_test(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test_binary: pd.Series,
    y_test_multi: pd.Series,
) -> dict:
    """Run the trained model against the test set and compute all metrics.

    Returns a dict suitable for inclusion in the manifest.
    """
    print(f"Evaluating on test set (n={len(X_test):,})...")

    # Predict probabilities and hard labels
    t0 = time.time()
    y_proba = model.predict_proba(X_test)[:, 1]  # column 1 = attack class
    bulk_predict_seconds = time.time() - t0

    y_pred = (y_proba >= 0.5).astype(int)

    # Headline binary metrics
    print(f"  Bulk inference: {len(X_test):,} flows in "
          f"{bulk_predict_seconds:.2f}s "
          f"({bulk_predict_seconds * 1e6 / len(X_test):.2f} us/flow average)")

    # Per-flow latency measurement (more careful, hot-cache)
    print(f"  Measuring per-flow inference latency (n=100)...")
    inference_ms = measure_inference_latency(model, X_test, n_repeats=100)
    print(f"  Median per-flow latency: {np.median(inference_ms):.3f} ms")

    binary_metrics = compute_binary_metrics(
        y_true=y_test_binary.values,
        y_pred=y_pred,
        y_proba=y_proba,
        inference_times_ms=inference_ms,
    )

    # Per-class breakdown
    multi_labels = sorted(y_test_multi.unique(), key=lambda lbl: -(y_test_multi == lbl).sum())
    per_class = compute_per_class_metrics(
        y_true_multi=y_test_multi.values,
        y_pred_binary=y_pred,
        multi_class_labels=multi_labels,
    )

    return {
        "binary": asdict(binary_metrics),
        "per_class": per_class,
        "bulk_predict_seconds": round(bulk_predict_seconds, 3),
        "predictions": y_pred,
        "probabilities": y_proba,
    }


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
    print(f"  train: {len(df_train):,} rows × {df_train.shape[1]} columns")
    print(f"  val:   {len(df_val):,} rows × {df_val.shape[1]} columns")
    print(f"  test:  {len(df_test):,} rows × {df_test.shape[1]} columns")

    X_train, y_train_bin, _           = select_features(df_train)
    X_val,   y_val_bin,   _           = select_features(df_val)
    X_test,  y_test_bin,  y_test_multi = select_features(df_test)
    print(f"  Feature count after selection: {X_train.shape[1]}\n")

    # ---- Train ---- #
    model, fit_seconds = train_random_forest(X_train, y_train_bin)

    # Quick val-set sanity check (we don't tune on val for this baseline,
    # but it's useful to record val performance as a sanity check)
    print(f"\nQuick sanity check on validation set...")
    y_val_proba = model.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_proba >= 0.5).astype(int)
    val_acc = (y_val_pred == y_val_bin.values).mean()
    print(f"  Validation accuracy: {val_acc:.4f}")

    # ---- Evaluate on test ---- #
    print()
    eval_results = evaluate_on_test(model, X_test, y_test_bin, y_test_multi)

    # ---- Save artefacts ---- #
    print(f"\nSaving artefacts...")

    print(f"  Model -> {MODEL_PATH.relative_to(REPO_ROOT)}")
    joblib.dump(model, MODEL_PATH)

    print(f"  Predictions -> {PREDICTIONS_PATH.relative_to(REPO_ROOT)}")
    np.savez_compressed(
        PREDICTIONS_PATH,
        predictions=eval_results["predictions"],
        probabilities=eval_results["probabilities"],
        y_true=y_test_bin.values,
    )

    manifest = {
        "tool": "src/models/random_forest.py",
        "splits_directory": str(SPLITS_DIR.relative_to(REPO_ROOT)),
        "experiment_directory": str(EXPERIMENT_DIR.relative_to(REPO_ROOT)),
        "hyperparameters": HYPERPARAMS,
        "n_features": int(X_train.shape[1]),
        "feature_names": list(X_train.columns),
        "training": {
            "n_train_samples": int(len(X_train)),
            "fit_seconds": round(fit_seconds, 2),
        },
        "validation": {
            "n_val_samples": int(len(X_val)),
            "accuracy": round(float(val_acc), 4),
        },
        "test": {
            "binary": eval_results["binary"],
            "per_class": eval_results["per_class"],
            "bulk_predict_seconds": eval_results["bulk_predict_seconds"],
        },
    }

    print(f"  Manifest -> {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    # ---- Headline summary ---- #
    bm = eval_results["binary"]
    print(f"\n{'=' * 70}")
    print(f"HEADLINE METRICS — Random Forest on CIC-IDS2017 test set")
    print(f"{'=' * 70}")
    print(f"  Accuracy:           {bm['accuracy']:.4f}")
    print(f"  Precision (attack): {bm['precision']:.4f}")
    print(f"  Recall (attack):    {bm['recall']:.4f}")
    print(f"  F1 (attack):        {bm['f1']:.4f}")
    print(f"  ROC-AUC:            {bm['roc_auc']:.4f}")
    print(f"  PR-AUC:             {bm['pr_auc']:.4f}")
    print(f"  FPR:                {bm['fpr']:.4f}")
    print(f"  Latency (median):   {bm['inference_ms_median']:.3f} ms/flow")
    print(f"  Latency (p95):      {bm['inference_ms_p95']:.3f} ms/flow")

    return 0


if __name__ == "__main__":
    sys.exit(main())