#!/usr/bin/env python3
"""
verify_predictions.py - recompute every reported metric from the saved
per-flow predictions.

This is a stronger check than reconciling the figures recorded in the
manifests against each other. Here the confusion matrix is rebuilt from
the raw y_true and prediction arrays, every metric is recalculated from
first principles, and ROC-AUC is recomputed from the predicted
probabilities using the rank-sum identity rather than by calling
scikit-learn. The reported figures are therefore checked against an
independent implementation operating on the original per-flow outputs.

Run from the repository root:

    python3 verify_predictions.py

Appends to verification_report.md and prints to stdout. Exit status 0
if every check passes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "verification_report.md"
TOL = 1e-6
AUC_TOL = 1e-5          # looser: parallel float summation moves the last bits

lines: list[str] = []
failures = 0
checks = 0


def emit(s: str = "") -> None:
    lines.append(s)
    print(s)


def check(label: str, computed: float, reported, tol: float = TOL) -> None:
    global failures, checks
    checks += 1
    if reported is None:
        emit(f"  {label}: computed {computed:.10f} — not recorded in manifest")
        return
    ok = abs(computed - reported) <= tol
    if not ok:
        failures += 1
    emit(f"  {label}: computed {computed:.10f} · reported {reported:.10f} · "
         f"{'PASS' if ok else 'FAIL'}")


def check_int(label: str, computed: int, reported) -> None:
    global failures, checks
    checks += 1
    ok = computed == reported
    if not ok:
        failures += 1
    emit(f"  {label}: computed {computed:,} · reported {reported:,} · "
         f"{'PASS' if ok else 'FAIL'}")


def average_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks of x, 1-based, with ties given their average rank."""
    n = x.size
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)

    xs = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    ROC-AUC via the rank-sum identity:

        AUC = (sum of ranks of positives - n_pos(n_pos + 1)/2) / (n_pos * n_neg)

    Equivalent to the probability that a randomly chosen positive is
    ranked above a randomly chosen negative, with ties counted as half.
    """
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = average_ranks(scores)
    return (r[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Average precision: sum over thresholds of (R_n - R_{n-1}) * P_n,
    which is how scikit-learn defines PR-AUC.
    """
    order = np.argsort(-scores, kind="mergesort")
    y = y_true[order]
    s = scores[order]

    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return float("nan")

    # keep only the last index of each run of equal scores
    distinct = np.r_[np.where(np.diff(s))[0], s.size - 1]
    tp, fp = tp[distinct], fp[distinct]

    precision = tp / (tp + fp)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def verify(title: str, npz_path: Path, reported: dict,
           threshold: float | None = None,
           error_key: str | None = None) -> None:
    global failures, checks

    emit(f"### {title}")
    emit()

    if not npz_path.exists():
        emit(f"  MISSING: {npz_path}")
        emit()
        return

    d = np.load(npz_path)
    y = d["y_true"].astype(np.int64)
    p = d["predictions"].astype(np.int64)
    probs = d["probabilities"].astype(np.float64)

    emit(f"Source: `{npz_path.relative_to(ROOT)}` · {y.size:,} flows")
    emit()

    tp = int(np.sum((y == 1) & (p == 1)))
    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))

    emit("Confusion matrix counted directly from the saved arrays:")
    emit()
    emit("```")
    emit(f"TP {tp:>10,}    FP {fp:>10,}")
    emit(f"FN {fn:>10,}    TN {tn:>10,}")
    emit("```")
    emit()

    for key, val in (("tp", tp), ("tn", tn), ("fp", fp), ("fn", fn)):
        if key in reported:
            check_int(f"{key.upper()} vs manifest", val, reported[key])
    if "n_test_samples" in reported:
        check_int("total flows      ", tp + tn + fp + fn, reported["n_test_samples"])
    emit()

    n = tp + tn + fp + fn
    acc = (tp + tn) / n
    pre = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f_1 = 2 * pre * rec / (pre + rec) if (tp + fp) and (pre + rec) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")

    emit("```")
    emit(f"accuracy  = ({tp:,} + {tn:,}) / {n:,} = {acc:.10f}")
    if tp + fp:
        emit(f"precision = {tp:,} / {tp + fp:,} = {pre:.10f}")
        emit(f"F1        = 2PR / (P + R)      = {f_1:.10f}")
    else:
        emit("precision = 0 / 0 = UNDEFINED (no positive predictions made)")
        emit("F1        = UNDEFINED for the same reason")
    emit(f"recall    = {tp:,} / {tp + fn:,} = {rec:.10f}")
    emit(f"FPR       = {fp:,} / {fp + tn:,} = {fpr:.10f}")
    emit("```")
    emit()

    check("accuracy ", acc, reported.get("accuracy"))
    if tp + fp:
        check("precision", pre, reported.get("precision"))
        check("F1       ", f_1, reported.get("f1"))
    else:
        emit("  precision and F1 are undefined, not zero; the manifest's 0.0 is")
        emit("  scikit-learn's substitution for the undefined case.")
    check("recall   ", rec, reported.get("recall"))
    check("FPR      ", fpr, reported.get("fpr"))
    emit()

    emit("Threshold-independent metrics, recomputed from the probabilities:")
    emit()
    auc = roc_auc(y, probs)
    ap = average_precision(y, probs)
    emit(f"  ROC-AUC by rank sum   : {auc:.10f}")
    emit(f"  Average precision     : {ap:.10f}")
    emit()
    check("ROC-AUC  ", auc, reported.get("roc_auc"), AUC_TOL)
    check("PR-AUC   ", ap, reported.get("pr_auc"), AUC_TOL)
    emit()

    if threshold is not None:
        key = error_key if error_key and error_key in d.files else "probabilities"
        err = d[key].astype(np.float64)
        implied = (err > threshold).astype(np.int64)
        mismatches = int(np.sum(implied != p))
        checks += 1
        if mismatches:
            failures += 1
        emit(f"Threshold rule: a flow is flagged when reconstruction error exceeds")
        emit(f"{threshold:.10f}. Applying that rule to the saved errors reproduces")
        emit(f"the saved predictions with {mismatches:,} mismatches · "
             f"{'PASS' if not mismatches else 'FAIL'}")
        emit()


# ======================================================================
emit()
emit("---")
emit()
emit("## 6. Metrics recomputed from saved per-flow predictions")
emit()
emit("Each confusion matrix below is counted directly from the stored")
emit("`y_true` and `predictions` arrays rather than taken from the manifest.")
emit("ROC-AUC is recomputed from the stored probabilities using the rank-sum")
emit("identity, and average precision by summing precision over recall")
emit("increments. Neither calls scikit-learn, so the reported values are")
emit("compared against an independent implementation.")
emit()


def man(rel: str) -> dict:
    p = ROOT / rel
    return json.load(p.open()) if p.exists() else {}


rf = man("experiments/rf/_metrics_manifest.json")
ae = man("experiments/ae/_metrics_manifest.json")
lstm = man("experiments/lstm/_metrics_manifest.json")
xd = man("experiments/cross_dataset/_metrics_manifest.json")

emit("### Within-dataset, CIC-IDS2017")
emit()

if rf:
    verify("Random Forest", ROOT / "experiments/rf/predictions.npz",
           rf["test"]["binary"])

if ae:
    verify("Autoencoder", ROOT / "experiments/ae/predictions.npz",
           ae["test"]["binary"],
           threshold=ae.get("threshold", {}).get("value"))

if lstm:
    verify("LSTM", ROOT / "experiments/lstm/predictions.npz",
           lstm["test"]["binary"])

    d = np.load(ROOT / "experiments/lstm/predictions.npz")
    if "val_accuracies" in d.files:
        va = d["val_accuracies"]
        tl = d["train_losses"]
        vl = d["val_losses"]
        emit("#### LSTM training history")
        emit()
        emit(f"  epochs recorded            : {va.size}")
        emit(f"  best validation accuracy   : {va.max():.6f} at epoch {int(va.argmax()) + 1}")
        emit(f"  final validation accuracy  : {va[-1]:.6f} at epoch {va.size}")
        emit(f"  final training loss        : {tl[-1]:.6f}")
        emit(f"  final validation loss      : {vl[-1]:.6f}")
        emit(f"  lowest validation loss     : {vl.min():.6f} at epoch {int(vl.argmin()) + 1}")
        emit()
        if int(va.argmax()) + 1 != va.size:
            emit("  The best validation accuracy did not occur at the final epoch.")
            emit("  No early stopping or best-checkpoint selection was used, so the")
            emit("  model evaluated on the test set is the final-epoch model rather")
            emit("  than the best-performing one.")
        else:
            emit("  The final epoch also gave the best validation accuracy.")
        emit()

emit("### Cross-dataset, CIC-UNSW-NB15")
emit()

if xd:
    if "rf" in xd:
        verify("Random Forest", ROOT / "experiments/cross_dataset/rf_predictions.npz",
               xd["rf"])
    if "ae" in xd:
        verify("Autoencoder", ROOT / "experiments/cross_dataset/ae_predictions.npz",
               xd["ae"], threshold=xd["ae"].get("threshold_used"),
               error_key="errors")

emit("---")
emit()
emit("### Summary of section 6")
emit()
emit(f"{checks} checks run, {checks - failures} passed, {failures} failed.")
emit()

with REPORT.open("a") as fh:
    fh.write("\n".join(lines) + "\n")

print(f"\nAppended to {REPORT}")
sys.exit(1 if failures else 0)
