#!/usr/bin/env python3
"""
threshold_sensitivity.py - show what the Autoencoder would have produced
at percentiles other than the 95th.

The 95th percentile was fixed in advance by a stated rule. This script
does not re-calibrate it. It reports what the alternatives would have
given, so the consequence of that choice can be stated rather than
assumed.

IMPORTANT, read before using the output.

The threshold at each percentile is derived from BENIGN reconstruction
error, and the metrics are then measured on the full test set. Reporting
whichever percentile produced the best F1 would be tuning a
hyperparameter on the test set, which is not a legitimate result. The
output below belongs in a sensitivity or limitations discussion, not in
the results table.

Run from the repository root:

    python3 threshold_sensitivity.py

Writes reports/threshold_sensitivity.md and prints the same content.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "reports"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "threshold_sensitivity.md"

PERCENTILES = [50.0, 60.0, 70.0, 75.0, 80.0, 85.0, 90.0, 92.5,
               95.0, 97.5, 99.0, 99.5, 99.9]

lines: list[str] = []


def emit(s: str = "") -> None:
    lines.append(s)
    print(s)


def metrics(y: np.ndarray, err: np.ndarray, thresh: float) -> dict:
    """Confusion matrix and derived metrics at one threshold."""
    pred = err > thresh
    pos = y == 1
    tp = int(np.sum(pos & pred))
    fn = int(np.sum(pos & ~pred))
    fp = int(np.sum(~pos & pred))
    tn = int(np.sum(~pos & ~pred))

    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else float("nan")
    pre = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f_1 = (2 * pre * rec / (pre + rec)
           if (tp + fp) and (tp + fn) and (pre + rec) else float("nan"))
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")

    return dict(tp=tp, tn=tn, fp=fp, fn=fn, accuracy=acc, precision=pre,
                recall=rec, f1=f_1, fpr=fpr)


def table(y: np.ndarray, err: np.ndarray, calib: np.ndarray,
          actual: float | None) -> None:
    """Sweep percentiles of `calib` and measure on `y` / `err`."""
    emit("| percentile | threshold | TP | FP | FN | TN | accuracy | "
         "precision | recall | F1 | FPR |")
    emit("|---|---|---|---|---|---|---|---|---|---|---|")

    rows = []
    for p in PERCENTILES:
        t = float(np.percentile(calib, p))
        rows.append((p, t, metrics(y, err, t)))

    if actual is not None:
        rows.append((None, actual, metrics(y, err, actual)))
        rows.sort(key=lambda r: r[1])

    for p, t, m in rows:
        label = f"{p:g}th" if p is not None else "**as run**"
        emit(f"| {label} | {t:.6f} | {m['tp']:,} | {m['fp']:,} | "
             f"{m['fn']:,} | {m['tn']:,} | {m['accuracy']:.4f} | "
             f"{m['precision']:.4f} | {m['recall']:.4f} | "
             f"{m['f1']:.4f} | {m['fpr']:.4f} |")
    emit()


# ------------------------------------------------------------------ load
ae_man = json.load((ROOT / "experiments/ae/_metrics_manifest.json").open())
THRESH = ae_man["threshold"]["value"]

within = np.load(ROOT / "experiments/ae/predictions.npz")
y_in = within["y_true"].astype(int)
e_in = within["probabilities"].astype(float)

# The threshold as run was calibrated on the VALIDATION benign errors.
# Use those if they were saved, otherwise fall back to the test benign
# errors and say so, because the two are not the same thing.
val_err = None
for key in ("val_errors", "validation_errors", "val_benign_errors"):
    if key in within.files:
        val_err = within[key].astype(float)
        break
if val_err is None:
    for name in ("val_errors.npy", "validation_errors.npy"):
        p = ROOT / "experiments/ae" / name
        if p.exists():
            val_err = np.load(p).astype(float)
            break

if val_err is not None:
    calib_in = val_err
    calib_note = (f"Percentiles are taken over the {val_err.size:,} saved "
                  "validation benign reconstruction errors, which is the "
                  "set the threshold as run was calibrated on.")
else:
    calib_in = e_in[y_in == 0]
    calib_note = (f"The saved validation errors were not found, so "
                  f"percentiles are taken over the {calib_in.size:,} test "
                  "benign errors instead. The threshold as run was "
                  "calibrated on validation benign error, so the "
                  "percentile column below is indicative of the shape of "
                  "the trade-off rather than an exact reconstruction of "
                  "the calibration procedure.")

# ------------------------------------------------------------------ write
emit("# Autoencoder threshold sensitivity")
emit()
emit("The decision threshold was set at the 95th percentile of benign")
emit("reconstruction error, a rule fixed before the results were seen.")
emit("This table reports what other percentiles would have produced. It is")
emit("a sensitivity analysis, not a calibration. Selecting a percentile on")
emit("the basis of the metrics below would amount to tuning on the test")
emit("set and no figure here should be presented as a result.")
emit()
emit(f"Threshold as run: `{THRESH:.16f}`")
emit()
emit(calib_note)
emit()
emit("## Within-dataset, CIC-IDS2017 test set")
emit()
emit(f"{y_in.size:,} flows · {int((y_in == 0).sum()):,} benign · "
     f"{int((y_in == 1).sum()):,} attack")
emit()
table(y_in, e_in, calib_in, THRESH)

# ------------------------------------------------------- cross-dataset
cross_path = ROOT / "experiments/cross_dataset/ae_predictions.npz"
if cross_path.exists():
    cross = np.load(cross_path)
    y_cr = cross["y_true"].astype(int)
    e_cr = (cross["errors"] if "errors" in cross.files
            else cross["probabilities"]).astype(float)

    emit("## Cross-dataset, CIC-UNSW-NB15")
    emit()
    emit(f"{y_cr.size:,} flows · {int((y_cr == 0).sum()):,} benign · "
         f"{int((y_cr == 1).sum()):,} attack")
    emit()
    emit("Two separate questions are asked here, and they must not be")
    emit("confused.")
    emit()
    emit("The first table applies thresholds derived from **CIC-IDS2017**")
    emit("benign error to CIC-UNSW-NB15. This is the deployment scenario,")
    emit("a model calibrated on one network and carried to another.")
    emit()
    table(y_cr, e_cr, calib_in, THRESH)

    emit("The second table re-derives the percentiles from")
    emit("**CIC-UNSW-NB15's own** benign error. This is not a scenario that")
    emit("could occur in practice, because it requires labelled benign")
    emit("traffic from the target network. It is included only to separate")
    emit("two possible explanations of the collapse, a threshold placed in")
    emit("the wrong position against scores that are still informative, or")
    emit("scores that carry no information at all.")
    emit()
    table(y_cr, e_cr, e_cr[y_cr == 0], None)

    # --------------------------------------------- ranking diagnostic
    pos = y_cr == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    order = np.argsort(e_cr, kind="mergesort")
    ranks = np.empty(e_cr.size, dtype=np.float64)
    ranks[order] = np.arange(1, e_cr.size + 1, dtype=np.float64)
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    emit("### Is any threshold capable of working here")
    emit()
    emit("ROC-AUC does not depend on the threshold. It measures only")
    emit("whether attack flows received higher reconstruction errors than")
    emit("benign flows, across every possible cut point. A value of 0.5 is")
    emit("what random ordering produces.")
    emit()
    emit("```")
    emit(f"ROC-AUC on CIC-UNSW-NB15 = {auc:.6f}")
    emit("```")
    emit()
    if auc < 0.55:
        emit("The ranking is close to uninformative. No threshold can")
        emit("separate a list that is not ordered usefully in the first")
        emit("place, so the cross-dataset failure cannot be attributed to")
        emit("the threshold having been placed badly. The second table")
        emit("above confirms this, re-deriving the threshold from the")
        emit("target network's own traffic does not recover performance.")
    else:
        emit("The ranking retains information, so the position of the")
        emit("threshold is implicated in the cross-dataset result rather")
        emit("than the scores themselves.")
    emit()

with OUT.open("w") as fh:
    fh.write("\n".join(lines) + "\n")

print(f"\nWritten to {OUT}")
