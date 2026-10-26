#!/usr/bin/env python3
"""
host_concentration.py - quantify how far the source address determines the
label in CIC-IDS2017, and what that implies for the sequence model.

Sequences for the LSTM are grouped by Src IP. If source address is strongly
correlated with the label, the grouping key carries much of the answer, and
per-sequence performance cannot be attributed to temporal modelling alone.

This script measures that correlation directly. It reports, for each fold,
how attack flows are distributed across source addresses, how many sources
survive the ten-flow minimum needed to form a window, how far the sources
overlap between folds, and what a trivial classifier using nothing but the
single most attack-heavy source address would score.

Src IP is NOT a model feature. It is excluded from the 71 by
DROP_FOR_FEATURES. The point of this analysis is not that any model used
it, but that the dataset contains a near-perfect trivial separator which
the 71 features are necessarily correlated with.

Run from the repository root:

    python3 host_concentration.py

Writes reports/host_concentration.md and prints the same content.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SPLITS = ROOT / "data" / "processed" / "cicids2017-engelen"
OUTDIR = ROOT / "reports"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "host_concentration.md"

FOLDS = ("train", "val", "test")
SEQ_LEN = 10          # must match sequence_length in _sequences_manifest.json
TOP_N = 12

lines: list[str] = []


def emit(s: str = "") -> None:
    lines.append(s)
    print(s)


def load(fold: str) -> pd.DataFrame:
    return pd.read_parquet(SPLITS / f"{fold}.parquet",
                           columns=["Src IP", "is_attack"])


emit("# Source address concentration in CIC-IDS2017")
emit()
emit("Sequences for the LSTM are grouped by `Src IP` and cut into")
emit(f"non-overlapping windows of {SEQ_LEN} flows. This analysis measures how")
emit("far the grouping key determines the label.")
emit()
emit("`Src IP` is not a model feature. It is excluded from the 71 features")
emit("used by every model. The question here is not whether a model used the")
emit("address, but whether the dataset is trivially separable by it.")
emit()

frames = {f: load(f) for f in FOLDS}

# ------------------------------------------------- per-fold distribution
for fold in FOLDS:
    df = frames[fold]
    ct = pd.crosstab(df["Src IP"], df["is_attack"])
    for c in (0, 1):
        if c not in ct.columns:
            ct[c] = 0
    ct = ct[[0, 1]]
    ct.columns = ["benign", "attack"]
    ct["total"] = ct["benign"] + ct["attack"]
    ct["pct_attack"] = 100 * ct["attack"] / ct["total"]
    ct = ct.sort_values("attack", ascending=False)

    n_attack = int(ct["attack"].sum())
    top = ct.iloc[0]

    emit(f"## {fold}")
    emit()
    emit(f"{len(df):,} flows · {n_attack:,} attack · "
         f"{len(ct):,} distinct source addresses")
    emit()
    emit("```")
    emit(f"{'Src IP':<18}{'benign':>12}{'attack':>12}{'% attack':>11}")
    for ip, row in ct.head(TOP_N).iterrows():
        emit(f"{str(ip):<18}{int(row['benign']):>12,}"
             f"{int(row['attack']):>12,}{row['pct_attack']:>10.2f}%")
    if len(ct) > TOP_N:
        rest = ct.iloc[TOP_N:]
        emit(f"{'(' + str(len(rest)) + ' more)':<18}"
             f"{int(rest['benign'].sum()):>12,}{int(rest['attack'].sum()):>12,}")
    emit("```")
    emit()
    emit(f"The single most attack-heavy source, `{top.name}`, accounts for "
         f"{100 * top['attack'] / n_attack:.2f}% of all attack flows in this "
         f"fold, and {top['pct_attack']:.2f}% of its own traffic is attack.")
    emit()

    eligible = ct[ct["total"] >= SEQ_LEN]
    emit(f"Sources with at least {SEQ_LEN} flows, and therefore able to form "
         f"at least one window: **{len(eligible)} of {len(ct)}**.")
    emit()

# ------------------------------------------------------ source overlap
emit("## Source overlap between folds")
emit()
sets = {f: set(frames[f]["Src IP"].unique()) for f in FOLDS}
for f in FOLDS:
    emit(f"  {f:<6} {len(sets[f]):>4} distinct sources")
emit()
emit(f"  present in all three folds : {len(sets['train'] & sets['val'] & sets['test'])}")
emit(f"  in test but not in train   : {len(sets['test'] - sets['train'])}")
emit()

te = frames["test"]["Src IP"].value_counts()
big = te[te >= SEQ_LEN]
seen = sum(ip in sets["train"] for ip in big.index)
emit(f"Of the {len(big)} test sources able to form a window, {seen} were also "
     f"present in the training fold.")
emit()
if seen == len(big):
    emit("Every source that produced a test sequence had already been seen")
    emit("during training. No held-out host exists.")
    emit()

# -------------------------------------------------- trivial baseline
emit("## A trivial classifier using only the source address")
emit()
test = frames["test"]
ct_test = pd.crosstab(test["Src IP"], test["is_attack"])
for c in (0, 1):
    if c not in ct_test.columns:
        ct_test[c] = 0
dominant = ct_test[1].idxmax()

pred = (test["Src IP"] == dominant)
pos = test["is_attack"] == 1
tp = int((pos & pred).sum())
fp = int((~pos & pred).sum())
fn = int((pos & ~pred).sum())
tn = int((~pos & ~pred).sum())

n = tp + tn + fp + fn
acc = (tp + tn) / n
pre = tp / (tp + fp) if (tp + fp) else float("nan")
rec = tp / (tp + fn) if (tp + fn) else float("nan")
f_1 = 2 * pre * rec / (pre + rec)
fpr = fp / (fp + tn)

emit(f"Rule: predict attack if and only if `Src IP == {dominant}`.")
emit("No features, no training, no model.")
emit()
emit("```")
emit(f"TP {tp:>10,}    FP {fp:>10,}")
emit(f"FN {fn:>10,}    TN {tn:>10,}")
emit()
emit(f"accuracy   {acc:.4f}")
emit(f"precision  {pre:.4f}")
emit(f"recall     {rec:.4f}")
emit(f"F1         {f_1:.4f}")
emit(f"FPR        {fpr:.4f}")
emit("```")
emit()
emit("Compare against the per-flow models reported in Chapter 5. Any model")
emit("that does not clearly exceed this baseline has not demonstrated that it")
emit("learned anything beyond what a single address lookup provides.")
emit()

# ----------------------------------------- host-disjoint feasibility
emit("## Feasibility of a host-disjoint split")
emit()
share = 100 * int(ct_test[1].max()) / int(ct_test[1].sum())
emit(f"Holding `{dominant}` out of the evaluation set would remove "
     f"{share:.2f}% of the attack flows along with it.")
emit()
emit("A host-disjoint train and test split is therefore not possible on this")
emit("dataset. The confound between source identity and label cannot be")
emit("removed by resampling, only by using a capture with more than one")
emit("attacking host.")
emit()

with OUT.open("w") as fh:
    fh.write("\n".join(lines) + "\n")

print(f"\nWritten to {OUT}")
