"""
metrics.py — Shared evaluation metrics for binary NIDS classification.

Computes the 9 metrics specified for the dissertation (see Methodology
chapter 4.6 / dissertation companion pack):

1.  Accuracy
2.  Precision (attack class)
3.  Recall (attack class)
4.  F1 (attack class)
5.  ROC-AUC
6.  PR-AUC (Average Precision)
7.  False Positive Rate (FPR)
8.  Inference latency (median ms per flow)
9.  Per-class precision/recall/F1 (multi-class breakdown)

All three model types (RF, AE, LSTM) call into this module, so reported
metrics are computed identically and are directly comparable.

Conventions:
- Binary positive class is "attack" (is_attack == 1). Precision, recall,
  F1, FPR refer to this class.
- Predictions arrive as either probabilities (continuous, 0-1) or hard
  labels (0/1). Threshold for binary decisions defaults to 0.5.
"""

from dataclasses import dataclass, asdict, field
import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


# ---------- Data structures ---------- #

@dataclass
class BinaryMetrics:
    """Headline binary classification metrics (attack vs benign)."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    fpr: float
    inference_ms_median: float
    inference_ms_p95: float
    n_test_samples: int
    n_test_attack: int
    n_test_benign: int


# ---------- Core metric functions ---------- #

def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    inference_times_ms: Optional[np.ndarray] = None,
) -> BinaryMetrics:
    """Compute the 8 headline binary metrics plus inference latency.

    Parameters
    ----------
    y_true : array of 0/1
        Ground-truth attack labels.
    y_pred : array of 0/1
        Hard predicted labels (after applying threshold).
    y_proba : array of floats in [0, 1]
        Predicted probabilities for the attack class (used for ROC-AUC, PR-AUC).
    inference_times_ms : array of floats, optional
        Per-flow inference times in milliseconds. If None, latency
        statistics are reported as NaN.

    Returns
    -------
    BinaryMetrics
        All metrics as a dataclass; convertible to dict via asdict().
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    # Confusion matrix gives us TN, FP, FN, TP for FPR
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Inference latency
    if inference_times_ms is not None and len(inference_times_ms) > 0:
        latency_median = float(np.median(inference_times_ms))
        latency_p95 = float(np.percentile(inference_times_ms, 95))
    else:
        latency_median = float("nan")
        latency_p95 = float("nan")

    return BinaryMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_proba)),
        pr_auc=float(average_precision_score(y_true, y_proba)),
        fpr=float(fpr),
        inference_ms_median=latency_median,
        inference_ms_p95=latency_p95,
        n_test_samples=len(y_true),
        n_test_attack=int(np.sum(y_true == 1)),
        n_test_benign=int(np.sum(y_true == 0)),
    )


def compute_per_class_metrics(
    y_true_multi: np.ndarray,
    y_pred_binary: np.ndarray,
    multi_class_labels: list[str],
) -> dict:
    """Per-class evaluation: how does each multi-class label fare under
    the binary classifier?

    For each class label (e.g., 'DoS Hulk', 'Heartbleed'), reports:
    - n_samples in test set
    - recall (fraction of this class correctly classified as attack)
    - misclassification details (how many were missed entirely vs just predicted as benign)

    BENIGN gets reported as recall = (1 - FPR) restricted to benign rows.

    Parameters
    ----------
    y_true_multi : array of strings
        Original multi-class labels (e.g., 'BENIGN', 'DoS Hulk', 'Heartbleed').
    y_pred_binary : array of 0/1
        Binary predictions from the classifier.
    multi_class_labels : list[str]
        Ordered list of class names to report. Typically the unique values
        in y_true_multi, sorted by frequency descending.

    Returns
    -------
    dict
        Keyed by class name; each value is a dict with 'n', 'recall',
        'n_correct', 'n_missed'.
    """
    results = {}
    y_pred_binary = np.asarray(y_pred_binary).astype(int)

    for label in multi_class_labels:
        mask = y_true_multi == label
        n = int(mask.sum())
        if n == 0:
            continue

        if label == "BENIGN":
            # For benign, "correct" means predicted as 0 (benign)
            n_correct = int((y_pred_binary[mask] == 0).sum())
        else:
            # For attacks, "correct" means predicted as 1 (attack)
            n_correct = int((y_pred_binary[mask] == 1).sum())

        n_missed = n - n_correct
        recall = n_correct / n if n > 0 else 0.0

        results[label] = {
            "n": n,
            "n_correct": n_correct,
            "n_missed": n_missed,
            "recall": round(recall, 4),
        }

    return results


def measure_inference_latency(
    model,
    X: np.ndarray,
    n_repeats: int = 100,
    batch_size: int = 1,
) -> np.ndarray:
    """Measure per-flow inference latency with hot cache.

    The first 10 calls are discarded (cold cache); the remaining n_repeats
    calls are timed. With batch_size=1 we get per-flow latency, which is
    what matters for real-time NIDS deployment.

    Parameters
    ----------
    model : sklearn-compatible estimator with .predict() method
    X : array
        Test features. We sample one row at a time, n_repeats times.
    n_repeats : int
        Number of timing measurements to take.
    batch_size : int
        Predict-batch size. 1 for per-flow latency.

    Returns
    -------
    array of floats
        Inference time per flow in milliseconds.
    """
    rng = np.random.default_rng(42)
    indices = rng.choice(len(X), size=n_repeats + 10, replace=True)

    times_ms = []
    for i, idx in enumerate(indices):
     # Use DataFrame iloc to preserve column names (avoids sklearn warning)
        if hasattr(X, "iloc"):
            sample = X.iloc[idx:idx + batch_size]
        else:
            sample = np.asarray(X)[idx:idx + batch_size]
            
        t0 = time.perf_counter()
        _ = model.predict(sample)
        t1 = time.perf_counter()

        if i >= 10:  # discard first 10 (cold cache)
            times_ms.append((t1 - t0) * 1000)

    return np.array(times_ms)