#!/usr/bin/env python3
"""
check_table_5_11.py - verify every figure in Table 5.11 against source.

Table 5.11 compares a trivial classifier that uses nothing but the source
address against the two per-flow models. This script does not read the
table. It recomputes the source address rule from the test partition,
reads the Random Forest and Autoencoder figures from their manifests,
compares all fifteen cells against the values written into the table,
and then prints the table as generated from source so it can be pasted
in directly.

Metrics are computed from first principles. Nothing here calls
scikit-learn, consistent with verify_results.py and verify_predictions.py.

Run from the repository root:

    python3 check_table_5_11.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SPLITS = ROOT / "data" / "processed" / "cicids2017-engelen"

# the figures currently written into Table 5.11, to 4 decimal places
EXPECTED = {
    "Source address rule": dict(accuracy=0.9984, precision=0.9971,
                                recall=0.9953, f1=0.9962, fpr=0.0008),
    "Random Forest":       dict(accuracy=0.9917, precision=0.9626,
                                recall=0.9995, f1=0.9807, fpr=0.0104),
    "Autoencoder":         dict(accuracy=0.8318, precision=0.6749,
                                recall=0.3922, f1=0.4961, fpr=0.0506),
}

METRICS = ("accuracy", "precision", "recall", "f1", "fpr")

failures = 0


def metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> dict:
    n = tp + tn + fp + fn
    acc = (tp + tn) / n
    pre = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f_1 = 2 * pre * rec / (pre + rec) if (pre + rec) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return dict(accuracy=acc, precision=pre, recall=rec, f1=f_1, fpr=fpr)


# ------------------------------------------------ 1. source address rule
print("=" * 68)
print("1. Source address rule, recomputed from the test partition")
print("=" * 68)
print()

test = pd.read_parquet(SPLITS / "test.parquet", columns=["Src IP", "is_attack"])

ct = pd.crosstab(test["Src IP"], test["is_attack"])
for c in (0, 1):
    if c not in ct.columns:
        ct[c] = 0
dominant = ct[1].idxmax()

pred = test["Src IP"] == dominant
pos = test["is_attack"] == 1

tp = int((pos & pred).sum())
fp = int((~pos & pred).sum())
fn = int((pos & ~pred).sum())
tn = int((~pos & ~pred).sum())

print(f"  dominant source address : {dominant}")
print(f"  rule                    : predict attack if and only if "
      f"Src IP == {dominant}")
print()
print(f"  TP {tp:>10,}    FP {fp:>10,}")
print(f"  FN {fn:>10,}    TN {tn:>10,}")
print()

computed = {"Source address rule": metrics_from_counts(tp, tn, fp, fn)}

# ------------------------------------------- 2. the two per-flow models
print("=" * 68)
print("2. Random Forest and Autoencoder, read from their manifests")
print("=" * 68)
print()

for label, rel in (("Random Forest", "experiments/rf/_metrics_manifest.json"),
                   ("Autoencoder", "experiments/ae/_metrics_manifest.json")):
    p = ROOT / rel
    if not p.exists():
        print(f"  MISSING: {rel}")
        failures += 1
        continue
    with p.open() as fh:
        block = json.load(fh)["test"]["binary"]
    computed[label] = {m: block.get(m) for m in METRICS}
    print(f"  {label:<22} read from {rel}")
print()

# ------------------------------------------------------- 3. comparison
print("=" * 68)
print("3. Comparison against the figures written into Table 5.11")
print("=" * 68)
print()
print(f"  {'row':<22}{'metric':<12}{'computed':>14}{'in table':>11}   verdict")
print(f"  {'-' * 64}")

for row, expected in EXPECTED.items():
    got = computed.get(row)
    if got is None:
        print(f"  {row:<22}{'ALL':<12}{'not available':>14}")
        failures += 1
        continue
    for m in METRICS:
        c = got.get(m)
        e = expected[m]
        if c is None:
            print(f"  {row:<22}{m:<12}{'not recorded':>14}{e:>11.4f}   CHECK")
            failures += 1
            continue
        ok = round(c, 4) == round(e, 4)
        if not ok:
            failures += 1
        print(f"  {row:<22}{m:<12}{c:>14.10f}{e:>11.4f}   "
              f"{'PASS' if ok else 'FAIL'}")
    print()

# -------------------------------------------- 4. the table, from source
print("=" * 68)
print("4. Table 5.11 as generated from source")
print("=" * 68)
print()
print("Table 5.11 - Source Address Baseline Compared with the Per-Flow Models")
print()
header = f"{'':<22}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>9}{'FPR':>9}"
print(header)
for row in EXPECTED:
    got = computed.get(row)
    if got is None:
        continue
    cells = "".join(
        f"{got[m]:>{w}.4f}" if got[m] is not None else f"{'n/a':>{w}}"
        for m, w in zip(METRICS, (10, 11, 9, 9, 9)))
    print(f"{row:<22}{cells}")
print()
print("The LSTM is excluded because it classifies sequences of ten flows")
print("rather than individual flows, so its figures have a different")
print("denominator.")
print()

print("=" * 68)
if failures:
    print(f"{failures} cell(s) did not match. Use the figures in section 4 above.")
else:
    print("Every figure in Table 5.11 matches source. Safe to paste.")
print("=" * 68)
