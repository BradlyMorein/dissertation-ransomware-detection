#!/usr/bin/env python3
"""
verify_partitions.py - check that the train, validation and test
partitions are disjoint.

Counts summing correctly is not the same as the partitions being
disjoint: a row duplicated across two folds would satisfy a count check
and still leak training data into the test set. This script hashes every
row and tests the three partitions for intersection directly.

Duplicate rows that already exist in the source data are distinguished
from genuine leakage, because identical flow records can legitimately
occur more than once in a capture and would otherwise raise a false
alarm.

Run from the repository root:

    python3 verify_partitions.py

Appends to verification_report.md and prints to stdout. Exit status 0
if the partitions are disjoint, 1 otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SPLITS = ROOT / "data" / "processed" / "cicids2017-engelen"
REPORT = ROOT / "verification_report.md"

FOLDS = ("train", "val", "test")

lines: list[str] = []
failures = 0


def emit(s: str = "") -> None:
    lines.append(s)
    print(s)


def row_hashes(path: Path) -> pd.Series:
    """Hash every row of a partition, returning one 64-bit value per row."""
    df = pd.read_parquet(path)
    n = len(df)
    h = pd.util.hash_pandas_object(df, index=False)
    del df
    return h, n


emit()
emit("---")
emit()
emit("## 5. Partition disjointness")
emit()
emit("Each row is hashed over all of its columns. Two partitions sharing a")
emit("hash means they contain an identical row. Because identical flow")
emit("records can legitimately occur more than once in the source capture,")
emit("any shared hashes are checked against the number of duplicate rows")
emit("already present within the partitions themselves.")
emit()

hashes: dict[str, pd.Series] = {}
counts: dict[str, int] = {}

for fold in FOLDS:
    p = SPLITS / f"{fold}.parquet"
    if not p.exists():
        emit(f"MISSING: {p}")
        sys.exit(1)
    h, n = row_hashes(p)
    hashes[fold] = h
    counts[fold] = n
    dup_within = n - h.nunique()
    emit(f"  {fold:<6} {n:>10,} rows · {h.nunique():>10,} distinct · "
         f"{dup_within:>7,} duplicate rows within the fold")

emit()

sets = {f: set(h.to_numpy().tolist()) for f, h in hashes.items()}

emit("### Pairwise intersections")
emit()

pairs = (("train", "val"), ("train", "test"), ("val", "test"))
overlap_found = False

for a, b in pairs:
    shared = sets[a] & sets[b]
    k = len(shared)
    if k:
        overlap_found = True
    emit(f"  {a} ∩ {b}: {k:,} shared row hashes · {'CHECK' if k else 'PASS'}")

emit()

total_rows = sum(counts.values())
union = set().union(*sets.values())

emit("### Reconciliation")
emit()
emit(f"  rows across all three partitions : {total_rows:,}")
emit(f"  distinct row hashes in the union : {len(union):,}")
emit(f"  duplicate rows overall           : {total_rows - len(union):,}")
emit()

if not overlap_found:
    emit("  No row hash appears in more than one partition. The three folds")
    emit("  are disjoint and no training data is present in the test set.")
    emit("  VERDICT: PASS")
else:
    dups_within = sum(counts[f] - hashes[f].nunique() for f in FOLDS)
    emit(f"  Shared row hashes were found across partitions.")
    emit(f"  Duplicate rows within individual folds total {dups_within:,}.")
    emit()
    if dups_within == 0:
        emit("  No partition contains any internal duplicate, so the source data")
        emit("  appears to hold distinct rows throughout. Rows shared BETWEEN")
        emit("  partitions therefore cannot be explained by repeated records in")
        emit("  the capture, and indicate that the split placed the same row in")
        emit("  more than one fold.")
        emit("  VERDICT: FAIL — investigate before relying on any reported metric.")
    else:
        emit("  Internal duplicates are present, so identical flow records do")
        emit("  occur in the source. Rows shared between partitions may be")
        emit("  genuine repeats rather than a faulty split. Confirm against the")
        emit("  source by checking whether the shared rows are duplicated in the")
        emit("  cleaned dataset before the split was applied.")
        emit("  VERDICT: REVIEW")
    failures = 1

emit()

# ------------------------------------------------------------ sequences
seq_paths = {f: SPLITS / f"{f}_sequences.parquet" for f in FOLDS}
if all(p.exists() for p in seq_paths.values()):
    emit("### Sequence partitions")
    emit()
    seq_ids: dict[str, set] = {}
    for fold, p in seq_paths.items():
        df = pd.read_parquet(p, columns=["sequence_id"])
        ids = set(df["sequence_id"].unique().tolist())
        seq_ids[fold] = ids
        emit(f"  {fold:<6} {len(df):>9,} rows · {len(ids):>8,} distinct sequence_id")
        del df
    emit()
    seq_overlap = False
    for a, b in pairs:
        shared = seq_ids[a] & seq_ids[b]
        if shared:
            seq_overlap = True
            failures = 1
        emit(f"  {a} ∩ {b}: {len(shared):,} shared sequence_id · "
             f"{'FAIL' if shared else 'PASS'}")
    emit()
    if not seq_overlap:
        emit("  No sequence identifier appears in more than one partition.")
    emit()

with REPORT.open("a") as fh:
    fh.write("\n".join(lines) + "\n")

print(f"\nAppended to {REPORT}")
sys.exit(1 if failures else 0)
