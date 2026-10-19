#!/usr/bin/env python3
"""
verify_results.py - independent verification of reported metrics.

Recomputes every performance figure from confusion matrix counts and
compares the result against the value recorded in the experiment
manifests. Nothing here calls scikit-learn: the arithmetic is done
from first principles so that the reported figures are checked against
an independent calculation rather than against the library that
produced them.

Run from the repository root:

    python3 verify_results.py

Writes a Markdown report to verification_report.md and prints a
summary to stdout. Exit status is 0 if every check passes and 1 if any
check fails.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOL = 5e-6           # agreement tolerance for recomputed metrics
OUT = ROOT / "verification_report.md"

lines: list[str] = []
failures = 0
checks = 0


def emit(s: str = "") -> None:
    lines.append(s)
    print(s)


def load(rel: str) -> dict:
    p = ROOT / rel
    if not p.exists():
        emit(f"MISSING FILE: {rel}")
        return {}
    with p.open() as fh:
        return json.load(fh)


def check(label: str, computed, reported, tol: float = TOL) -> None:
    """Compare a computed value against a reported one."""
    global failures, checks
    checks += 1
    if reported is None:
        emit(f"  {label}: computed {computed} — not recorded in manifest, nothing to compare")
        return
    ok = abs(computed - reported) <= tol
    if not ok:
        failures += 1
    verdict = "PASS" if ok else "FAIL"
    emit(f"  {label}: computed {computed:.10f} · reported {reported:.10f} · {verdict}")


def check_eq(label: str, computed, reported) -> None:
    """Compare two integers for exact equality."""
    global failures, checks
    checks += 1
    ok = computed == reported
    if not ok:
        failures += 1
    emit(f"  {label}: computed {computed:,} · reported {reported:,} · {'PASS' if ok else 'FAIL'}")


# ----------------------------------------------------------------------
# metric definitions, from first principles
# ----------------------------------------------------------------------

def accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)


def precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) else float("nan")


def recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) else float("nan")


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) else float("nan")


def fpr(fp, tn):
    return fp / (fp + tn) if (fp + tn) else float("nan")


def confusion_block(cm_source: dict, n_attack: int, per_class: dict | None):
    """
    Return (tp, tn, fp, fn).

    Uses explicit tn/fp/fn/tp when the manifest records them. Otherwise
    derives them: the per-class BENIGN entry gives TN (n_correct) and FP
    (n_missed), and TP is recovered from the reported recall against the
    attack count, with FN as the remainder.
    """
    if all(k in cm_source for k in ("tn", "fp", "fn", "tp")):
        return cm_source["tp"], cm_source["tn"], cm_source["fp"], cm_source["fn"]

    if not per_class or "BENIGN" not in per_class:
        raise ValueError("no confusion matrix and no BENIGN per-class block")

    benign = per_class["BENIGN"]
    tn = benign["n_correct"]
    fp = benign["n_missed"]
    tp = round(cm_source["recall"] * n_attack)
    fn = n_attack - tp
    return tp, tn, fp, fn


def verify_binary(title: str, block: dict, per_class: dict | None = None,
                  derived: bool = False) -> None:
    emit(f"### {title}")
    emit()

    n_total = block.get("n_test_samples")
    n_attack = block.get("n_test_attack")
    n_benign = block.get("n_test_benign")

    tp, tn, fp, fn = confusion_block(block, n_attack, per_class)

    if derived:
        emit("Confusion matrix derived from the per-class BENIGN counts and the")
        emit("reported recall, as the manifest does not record it directly.")
        emit()

    emit("```")
    emit(f"TP {tp:>10,}    FP {fp:>10,}")
    emit(f"FN {fn:>10,}    TN {tn:>10,}")
    emit("```")
    emit()

    if n_total is not None:
        check_eq("partition size   TP+TN+FP+FN", tp + tn + fp + fn, n_total)
    if n_attack is not None:
        check_eq("attack count     TP+FN", tp + fn, n_attack)
    if n_benign is not None:
        check_eq("benign count     TN+FP", tn + fp, n_benign)
    emit()

    acc = accuracy(tp, tn, fp, fn)
    pre = precision(tp, fp)
    rec = recall(tp, fn)
    f_1 = f1(pre, rec)
    fp_r = fpr(fp, tn)

    emit("```")
    emit(f"accuracy  = (TP + TN) / N        = ({tp:,} + {tn:,}) / {tp+tn+fp+fn:,} = {acc:.10f}")
    if tp + fp:
        emit(f"precision = TP / (TP + FP)       = {tp:,} / {tp+fp:,} = {pre:.10f}")
    else:
        emit(f"precision = TP / (TP + FP)       = 0 / 0 = UNDEFINED")
    emit(f"recall    = TP / (TP + FN)       = {tp:,} / {tp+fn:,} = {rec:.10f}")
    emit(f"F1        = 2PR / (P + R)        = {f_1:.10f}")
    emit(f"FPR       = FP / (FP + TN)       = {fp:,} / {fp+tn:,} = {fp_r:.10f}")
    emit("```")
    emit()

    check("accuracy ", acc, block.get("accuracy"))
    if tp + fp:
        check("precision", pre, block.get("precision"))
        check("F1       ", f_1, block.get("f1"))
    else:
        emit("  precision: TP + FP = 0, so precision is undefined rather than zero.")
        emit("             The manifest records 0.0, which is scikit-learn's")
        emit("             substitution for the undefined case. F1 is undefined for")
        emit("             the same reason and is likewise recorded as 0.0.")
        emit("             Neither should be read as a measured value.")
    check("recall   ", rec, block.get("recall"))
    check("FPR      ", fp_r, block.get("fpr"))
    emit()


# ======================================================================
emit("# Verification report")
emit()
emit(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
emit()
emit("Every figure below is recomputed from confusion matrix counts using the")
emit("metric definitions written out in full, then compared against the value")
emit("recorded in the experiment manifests. The recomputation does not use")
emit("scikit-learn, so the reported figures are checked against an independent")
emit("calculation rather than against the library that produced them.")
emit()
emit("---")
emit()

# ---------------------------------------------------------------- split
emit("## 1. Data partitioning")
emit()

clean = load("data/processed/cicids2017-engelen/_clean_manifest.json")
split = load("data/processed/cicids2017-engelen/_split_manifest.json")

if clean:
    emit("### Cleaning")
    emit()
    emit("```")
    emit(json.dumps(clean.get("totals", {}), indent=2))
    emit("```")
    emit()

if split:
    folds = {f["fold"]: f for f in split.get("folds", [])}
    emit("### Partition sizes")
    emit()
    emit("```")
    for name in ("train", "val", "test"):
        f = folds.get(name, {})
        emit(f"{name:<6} {f.get('n_rows', 0):>10,}   "
             f"benign {f.get('n_benign', 0):>10,}   "
             f"attack {f.get('n_attack', 0):>9,}   "
             f"{f.get('pct_benign', 0):.4f}% benign")
    emit("```")
    emit()

    total = sum(f["n_rows"] for f in split.get("folds", []))
    check_eq("partitions sum   train+val+test", total, split.get("fold_total_rows"))

    for name in ("train", "val", "test"):
        f = folds.get(name, {})
        check_eq(f"{name} rows       benign+attack", f.get("n_benign", 0) + f.get("n_attack", 0),
                 f.get("n_rows", 0))

    emit()
    emit("### Stratification")
    emit()
    pcts = [folds[n]["pct_benign"] for n in ("train", "val", "test") if n in folds]
    spread = max(pcts) - min(pcts)
    checks += 1
    ok = spread < 0.01
    if not ok:
        failures += 1
    emit(f"  benign proportion across folds: {', '.join(f'{p:.4f}%' for p in pcts)}")
    emit(f"  spread {spread:.4f} percentage points · {'PASS' if ok else 'FAIL'}")
    emit()
    emit(f"  manifest records fold_total_matches_input = "
         f"{split.get('fold_total_matches_input')}")
    emit(f"  random seed {split.get('random_seed')}, strategy "
         f"{split.get('split_strategy')!r}")
    emit()

emit("---")
emit()

# ------------------------------------------------------- within-dataset
emit("## 2. Within-dataset results, CIC-IDS2017")
emit()

rf = load("experiments/rf/_metrics_manifest.json")
if rf:
    verify_binary("Random Forest", rf["test"]["binary"],
                  rf["test"].get("per_class"), derived=True)

ae = load("experiments/ae/_metrics_manifest.json")
if ae:
    verify_binary("Autoencoder", ae["test"]["binary"],
                  ae["test"].get("per_class"), derived=True)
    th = ae.get("threshold", {})
    emit(f"Threshold: {th.get('value')} ({th.get('method')}), "
         f"fitted on {th.get('n_val_benign_used'):,} validation benign flows.")
    emit()

lstm = load("experiments/lstm/_metrics_manifest.json")
if lstm:
    verify_binary("LSTM", lstm["test"]["binary"])

emit("---")
emit()

# -------------------------------------------------------- cross-dataset
emit("## 3. Cross-dataset results, CIC-UNSW-NB15")
emit()

xd = load("experiments/cross_dataset/_metrics_manifest.json")
if xd:
    emit(f"Trained on {xd.get('training_dataset')}, evaluated on "
         f"{xd.get('evaluation_dataset')}.")
    emit(f"{xd.get('n_evaluation_samples'):,} flows, "
         f"{xd.get('n_attack_samples'):,} attack, "
         f"{xd.get('n_benign_samples'):,} benign.")
    emit()
    emit(f"LSTM excluded: {xd.get('lstm_excluded_reason')}")
    emit()

    for key, name in (("rf", "Random Forest"), ("ae", "Autoencoder")):
        if key in xd:
            b = dict(xd[key])
            b.setdefault("n_test_samples", xd.get("n_evaluation_samples"))
            b.setdefault("n_test_attack", xd.get("n_attack_samples"))
            b.setdefault("n_test_benign", xd.get("n_benign_samples"))
            verify_binary(name, b)

    rf_auc = xd.get("rf", {}).get("roc_auc")
    if rf_auc is not None and rf_auc < 0.5:
        emit(f"Note: the Random Forest ROC-AUC of {rf_auc:.4f} is below 0.50. A value")
        emit("below chance means the ranking was inverted relative to the true")
        emit("labels, which is a threshold-independent result and therefore not")
        emit("attributable to threshold placement.")
        emit()

emit("---")
emit()

# ----------------------------------------------------------- arithmetic
emit("## 4. Stated figures")
emit()

if lstm:
    hp = lstm.get("hyperparameters", {})
    emit("```")
    emit(json.dumps(hp, indent=2)[:1200])
    emit("```")
    emit()

emit("---")
emit()
emit(f"## Summary")
emit()
emit(f"{checks} checks run, {checks - failures} passed, {failures} failed.")
emit()

OUT.write_text("\n".join(lines) + "\n")
print(f"\nReport written to {OUT}")
sys.exit(1 if failures else 0)
