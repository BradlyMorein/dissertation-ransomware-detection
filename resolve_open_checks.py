#!/usr/bin/env python3
"""
resolve_open_checks.py - resolve the two unresolved findings in
verification_report.md.

Section 5 of the report does not end in PASS. It ends in two unresolved
states, and both need settling before the report can be cited.

  Finding 1  Rows shared between partitions. VERDICT: REVIEW.
             1,463 row hashes appear in more than one fold. The script
             could not tell whether these are duplicate records that
             already existed in the source capture, which is harmless,
             or evidence that the split placed the same row in more
             than one fold, which is not.

  Finding 2  Sequence identifiers shared between all three partitions.
             VERDICT: FAIL on all three pairs. Note that train and test
             share 31,470 identifiers and test contains exactly 31,470
             sequences, so every single test identifier is reported as
             shared. A result that extreme usually means the check is
             wrong rather than the data.

This script settles both.

Run from the repository root:

    python3 resolve_open_checks.py

Writes reports/open_checks_resolved.md and prints the same content.
Nothing is modified.
"""

from __future__ import annotations

from pathlib import Path

import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
SPLITS = ROOT / "data" / "processed" / "cicids2017-engelen"
OUTDIR = ROOT / "reports"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "open_checks_resolved.md"

FOLDS = ("train", "val", "test")

lines: list[str] = []


def emit(s: str = "") -> None:
    lines.append(s)
    print(s)


emit("# Resolution of the two open checks in verification_report.md")
emit()

# ======================================================================
# FINDING 1 - are the shared rows source duplicates or a faulty split?
# ======================================================================
emit("## Finding 1 — rows shared between partitions")
emit()
emit("The question is whether the split copied any row, or whether the")
emit("source capture simply contains identical flow records that a random")
emit("split then scattered across folds.")
emit()

clean_manifest = SPLITS / "_clean_manifest.json"
rows_out = None
if clean_manifest.exists():
    with clean_manifest.open() as fh:
        rows_out = json.load(fh).get("totals", {}).get("rows_out")

hashes = {}
counts = {}
for fold in FOLDS:
    df = pd.read_parquet(SPLITS / f"{fold}.parquet")
    counts[fold] = len(df)
    hashes[fold] = pd.util.hash_pandas_object(df, index=False)
    del df
    emit(f"  {fold:<6} {counts[fold]:>10,} rows")

total_rows = sum(counts.values())
emit()
emit(f"  rows across all three folds      : {total_rows:,}")
emit(f"  rows_out recorded by the cleaner : "
     f"{rows_out:,}" if rows_out is not None else
     "  rows_out recorded by the cleaner : NOT FOUND")
emit()

if rows_out is not None:
    if total_rows == rows_out:
        emit("  The fold sizes sum to exactly the number of rows the cleaning")
        emit("  stage produced. The split therefore allocated each cleaned row")
        emit("  to exactly one fold and copied none of them. PASS")
    else:
        emit(f"  MISMATCH of {abs(total_rows - rows_out):,} rows. The split did not")
        emit("  partition the cleaned data cleanly. INVESTIGATE.")
emit()

# the union of all three folds is the cleaned dataset, so duplicates in
# that union are duplicates that were present before the split
all_hashes = pd.concat([hashes[f] for f in FOLDS], ignore_index=True)
n_union_distinct = all_hashes.nunique()
dupes_total = total_rows - n_union_distinct

within = {f: counts[f] - hashes[f].nunique() for f in FOLDS}
dupes_within = sum(within.values())
dupes_between = dupes_total - dupes_within

emit("### Where the duplicate rows sit")
emit()
emit("```")
emit(f"duplicate rows in the cleaned data overall : {dupes_total:>8,}")
emit(f"  of which duplicated inside one fold      : {dupes_within:>8,}")
emit(f"  of which duplicated across folds         : {dupes_between:>8,}")
emit("```")
emit()
for f in FOLDS:
    emit(f"  {f:<6} {within[f]:>7,} duplicate rows within the fold")
emit()

pct = 100 * dupes_total / total_rows
emit(f"Duplicates are {pct:.4f}% of the cleaned dataset.")
emit()

if dupes_within > 0:
    emit("Identical flow records do occur inside individual folds, where no")
    emit("split decision could have produced them. They are therefore present")
    emit("in the source capture, and the rows shared between folds are the")
    emit("same phenomenon seen across a partition boundary rather than a")
    emit("faulty split.")
else:
    emit("No fold contains an internal duplicate, so the shared rows cannot")
    emit("be explained by repeated records in the source. INVESTIGATE.")
emit()

# how many test rows have a twin in train - the only leakage that matters
train_set = set(hashes["train"].to_numpy().tolist())
test_arr = hashes["test"].to_numpy()
n_test_with_train_twin = int(sum(1 for h in test_arr if h in train_set))

emit("### The only form that matters for the reported results")
emit()
emit("A test row identical to a training row is the case that could inflate")
emit("a score, because the model has seen that exact feature vector with")
emit("that exact label.")
emit()
emit("```")
emit(f"test rows with an identical twin in train : {n_test_with_train_twin:>8,}")
emit(f"test rows in total                        : {counts['test']:>8,}")
emit(f"proportion                                : "
     f"{100 * n_test_with_train_twin / counts['test']:>8.4f}%")
emit("```")
emit()

del all_hashes, train_set, test_arr, hashes

# ======================================================================
# FINDING 2 - the sequence_id FAIL
# ======================================================================
emit("---")
emit()
emit("## Finding 2 — sequence identifiers shared between partitions")
emit()
emit("verify_partitions.py tests whether a sequence_id appears in more than")
emit("one fold. That test is only meaningful if sequence_id is unique across")
emit("the whole dataset. This checks whether it is.")
emit()

seq_paths = {f: SPLITS / f"{f}_sequences.parquet" for f in FOLDS}

if not all(p.exists() for p in seq_paths.values()):
    emit("Sequence files not found. Skipped.")
else:
    ids = {}
    emit("```")
    emit(f"{'fold':<8}{'rows':>12}{'distinct id':>14}{'min':>10}{'max':>10}")
    for fold, p in seq_paths.items():
        df = pd.read_parquet(p, columns=["sequence_id"])
        s = df["sequence_id"]
        ids[fold] = set(s.unique().tolist())
        emit(f"{fold:<8}{len(df):>12,}{s.nunique():>14,}"
             f"{int(s.min()):>10,}{int(s.max()):>10,}")
        del df, s
    emit("```")
    emit()

    starts_at_zero = all(min(v) == 0 for v in ids.values())
    contiguous = all(max(v) - min(v) + 1 == len(v) for v in ids.values())
    nested = (ids["test"] <= ids["train"]) and (ids["val"] <= ids["train"])

    emit("```")
    emit(f"every fold numbers its sequences from 0   : {starts_at_zero}")
    emit(f"every fold's identifiers are contiguous   : {contiguous}")
    emit(f"val and test id sets are subsets of train : {nested}")
    emit("```")
    emit()

    if starts_at_zero and contiguous and nested:
        emit("**The FAIL is an artefact of the check, not a property of the data.**")
        emit()
        emit("Each fold numbers its own sequences from zero upward. sequence_id")
        emit("is unique within a fold and is reused across folds, so identifier")
        emit("0 names a different sequence in each of the three partitions. The")
        emit("overlap the check reports is a collision of numbering ranges, not")
        emit("sequences appearing in more than one fold. The smaller folds'")
        emit("ranges sit entirely inside the larger one, which is why every")
        emit("identifier in val and test is reported as shared.")
        emit()
        emit("Sequence disjointness follows from flow disjointness instead.")
        emit("Sequences are constructed inside each fold after the split, from")
        emit("that fold's flows alone, so no sequence can span two folds.")
    else:
        emit("The identifiers do NOT follow a simple per-fold numbering scheme.")
        emit("The reported overlap may be real. INVESTIGATE before relying on")
        emit("any LSTM figure.")
    emit()

    # corroborate: do the sequence row counts match flows minus remainders?
    emit("### Corroboration from the row counts")
    emit()
    emit("If sequences were built inside each fold, each fold's sequence rows")
    emit("should be its flow count rounded down to a whole number of windows.")
    emit()
    emit("```")
    for fold in FOLDS:
        df = pd.read_parquet(seq_paths[fold], columns=["sequence_id"])
        n_seq_rows = len(df)
        n_seq = df["sequence_id"].nunique()
        del df
        emit(f"{fold:<8}flows {counts[fold]:>10,}   sequence rows {n_seq_rows:>10,}"
             f"   dropped {counts[fold] - n_seq_rows:>5,}   sequences {n_seq:>8,}"
             f"   rows/seq {n_seq_rows / n_seq:>5.1f}")
    emit("```")
    emit()

emit("---")
emit()
emit("Generated by resolve_open_checks.py")
emit()

with OUT.open("w") as fh:
    fh.write("\n".join(lines) + "\n")

print(f"\nWritten to {OUT}")
