"""
evaluate_cross_dataset.py — Evaluate CIC-IDS2017-trained RF and AE models on
the held-out CIC-UNSW-NB15 dataset.

This is the genuine generalisation test for my dissertation's three-way
comparison study. RF and AE were trained on CIC-IDS2017; this script applies
them unchanged to CIC-UNSW-NB15 (different sources, different attack types,
no LSTM because CIC-UNSW-NB15 was distributed without identifier columns,
so Src IP and Timestamp are absent).

Run from the repository root:

    python -m src.models.evaluate_cross_dataset
"""

from pathlib import Path
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)


# ---------- Configuration ---------- #

REPO_ROOT = Path(__file__).resolve().parents[2]
UNSW_PATH = REPO_ROOT / "data" / "processed" / "cic-unsw-nb15" / "cic-unsw-nb15.parquet"

RF_BUNDLE = REPO_ROOT / "experiments" / "rf" / "model.joblib"
AE_BUNDLE = REPO_ROOT / "experiments" / "ae" / "model.joblib"

OUTPUT_DIR = REPO_ROOT / "experiments" / "cross_dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Same drop list used in RF and AE training — preserves cross-dataset alignment
DROP_FOR_FEATURES = [
    "Flow ID", "Src IP", "Dst IP", "Timestamp",
    "Label", "LabelName", "is_attack",
    "Src Port", "Dst Port", "Protocol",
]

# AE-style negative-value clipping (same as in autoencoder.py)
NEGATIVE_VALUE_FEATURES = [
    "Flow Duration", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Max", "Flow IAT Min",
]


def load_and_prep_unsw() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load UNSW Parquet, derive is_attack from Label, prepare X and y.

    Returns
    -------
    X : DataFrame of features (71 columns aligned with CIC-IDS2017 training)
    y_binary : np.ndarray of binary labels (0=benign, 1=attack)
    y_multi : np.ndarray of UNSW LabelName strings (for per-class breakdown if desired)
    """
    print(f"Loading UNSW from {UNSW_PATH.relative_to(REPO_ROOT)}...")
    df = pd.read_parquet(UNSW_PATH)
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")

    # Derive is_attack: 0 in Label = benign, anything else = attack
    df["is_attack"] = (df["Label"] != 0).astype(int)
    n_attack = int(df["is_attack"].sum())
    print(f"  Attack rate: {df['is_attack'].mean():.4f} ({n_attack:,} attacks of {len(df):,})")

    # Clip CICFlowMeter artifacts (same as AE training)
    for col in NEGATIVE_VALUE_FEATURES:
        if col in df.columns:
            n_clipped = (df[col] < 0).sum()
            if n_clipped > 0:
                print(f"  Clipping {col}: {n_clipped:,} negative values to 0")
            df[col] = df[col].clip(lower=0)

    # Select features
    feature_cols = [c for c in df.columns if c not in DROP_FOR_FEATURES]
    X = df[feature_cols].astype(np.float32)
    y_binary = df["is_attack"].values
    y_multi = df["LabelName"].values

    print(f"  Feature count: {X.shape[1]}")
    return X, y_binary, y_multi


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Standard binary metrics, matching the format used in earlier RF/AE manifests."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "accuracy":     float(accuracy_score(y_true, y_pred)),
        "precision":    float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":       float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":           float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":      float(roc_auc_score(y_true, y_proba)),
        "pr_auc":       float(average_precision_score(y_true, y_proba)),
        "fpr":          float(fpr),
        "n_test_samples": int(len(y_true)),
        "n_test_attack":  int((y_true == 1).sum()),
        "n_test_benign":  int((y_true == 0).sum()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def evaluate_rf(X: pd.DataFrame, y_binary: np.ndarray) -> dict:
    """Apply the trained RF model to UNSW data."""
    print("\n" + "=" * 78)
    print("RANDOM FOREST — Cross-dataset evaluation on UNSW")
    print("=" * 78)

    if not RF_BUNDLE.exists():
        raise FileNotFoundError(f"RF bundle not found at {RF_BUNDLE}")

    # RF was saved as a raw RandomForestClassifier (not a dict bundle)
    model = joblib.load(RF_BUNDLE)
    print(f"  Loaded RF: n_estimators={model.n_estimators}, max_depth={model.max_depth}")

    t0 = time.time()
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    elapsed = time.time() - t0
    print(f"  Prediction time: {elapsed:.2f}s ({elapsed * 1e6 / len(X):.2f} us/flow)")

    metrics = compute_binary_metrics(y_binary, y_pred, y_proba)
    metrics["bulk_predict_seconds"] = round(elapsed, 3)

    # Save predictions
    np.savez_compressed(
        OUTPUT_DIR / "rf_predictions.npz",
        y_true=y_binary, predictions=y_pred, probabilities=y_proba,
    )
    print(f"  Predictions saved to {OUTPUT_DIR / 'rf_predictions.npz'}")

    return metrics


def evaluate_ae(X: pd.DataFrame, y_binary: np.ndarray) -> dict:
    """Apply the trained AE model + scaler + threshold to UNSW data."""
    print("\n" + "=" * 78)
    print("AUTOENCODER — Cross-dataset evaluation on UNSW")
    print("=" * 78)

    if not AE_BUNDLE.exists():
        raise FileNotFoundError(f"AE bundle not found at {AE_BUNDLE}")

    bundle = joblib.load(AE_BUNDLE)
    model = bundle["model"]
    scaler = bundle["scaler"]
    threshold = bundle["threshold"]
    print(f"  Loaded AE: hidden_layers={model.hidden_layer_sizes}")
    print(f"  Loaded scaler: {type(scaler).__name__}")
    print(f"  Anomaly threshold: {threshold:.5f}")

    # Apply preprocessing pipeline (log1p + StandardScaler) fitted on CIC-IDS2017 train
    X_scaled = scaler.transform(X)

    # Reconstruction errors
    t0 = time.time()
    X_reconstructed = model.predict(X_scaled)
    errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
    elapsed = time.time() - t0
    print(f"  Inference time: {elapsed:.2f}s ({elapsed * 1e6 / len(X):.2f} us/flow)")
    print(f"  Error distribution:")
    print(f"    Median: {np.median(errors):.5f}")
    print(f"    p95:    {np.percentile(errors, 95):.5f}")
    print(f"    Max:    {errors.max():.5f}")

    # Apply threshold
    y_pred = (errors > threshold).astype(int)
    y_proba = errors  # raw reconstruction error as score

    metrics = compute_binary_metrics(y_binary, y_pred, y_proba)
    metrics["bulk_predict_seconds"] = round(elapsed, 3)
    metrics["threshold_used"] = float(threshold)
    metrics["error_median"] = float(np.median(errors))
    metrics["error_p95"] = float(np.percentile(errors, 95))

    # Save predictions and errors
    np.savez_compressed(
        OUTPUT_DIR / "ae_predictions.npz",
        y_true=y_binary, predictions=y_pred, probabilities=y_proba, errors=errors,
    )
    print(f"  Predictions saved to {OUTPUT_DIR / 'ae_predictions.npz'}")

    return metrics


def print_headline(model_name: str, metrics: dict):
    """Pretty-print headline metrics."""
    print(f"\n{'=' * 70}")
    print(f"HEADLINE METRICS — {model_name} on UNSW")
    print(f"{'=' * 70}")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Precision (attack): {metrics['precision']:.4f}")
    print(f"  Recall (attack):    {metrics['recall']:.4f}")
    print(f"  F1 (attack):        {metrics['f1']:.4f}")
    print(f"  ROC-AUC:            {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:             {metrics['pr_auc']:.4f}")
    print(f"  FPR:                {metrics['fpr']:.4f}")
    print(f"  TN={metrics['tn']:,}  FP={metrics['fp']:,}  FN={metrics['fn']:,}  TP={metrics['tp']:,}")


def main() -> int:
    if not UNSW_PATH.exists():
        print(f"ERROR: UNSW Parquet not found at {UNSW_PATH}", file=sys.stderr)
        return 1

    print(f"Cross-dataset evaluation: CIC-IDS2017 -> CIC-UNSW-NB15")
    print(f"Output directory: {OUTPUT_DIR.relative_to(REPO_ROOT)}")
    print()

    # Load UNSW
    X, y_binary, y_multi = load_and_prep_unsw()

    # Run evaluations
    rf_metrics = evaluate_rf(X, y_binary)
    print_headline("Random Forest", rf_metrics)

    ae_metrics = evaluate_ae(X, y_binary)
    print_headline("Autoencoder", ae_metrics)

    # Combined manifest
    manifest = {
        "tool": "src/models/evaluate_cross_dataset.py",
        "training_dataset": "CIC-IDS2017 (Engelen-cleaned)",
        "evaluation_dataset": "CIC-UNSW-NB15 (CIC-augmented)",
        "n_evaluation_samples": int(len(X)),
        "n_attack_samples": int(y_binary.sum()),
        "n_benign_samples": int(len(y_binary) - y_binary.sum()),
        "attack_rate": float(y_binary.mean()),
        "lstm_excluded_reason": "CIC-UNSW-NB15 was distributed without identifier columns; Src IP and Timestamp are absent, so sequence construction is not possible",
        "rf": rf_metrics,
        "ae": ae_metrics,
    }

    manifest_path = OUTPUT_DIR / "_metrics_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to: {manifest_path.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())