"""
sequences.py — Build flow sequences for LSTM training/evaluation.

Transforms per-flow Parquet files into sequence Parquets where each
"sequence" is N consecutive flows from the same source IP. The LSTM
in Week 5 will learn temporal patterns within these sequences.

Methodology decisions (see ADR 0006):
- Sequence length: 10 flows (NIDS-LSTM literature standard).
- Grouping: by Src IP.
- Sort: by Timestamp ascending within each source.
- Window: non-overlapping (each flow appears in exactly one sequence).
- Drop: sources with fewer than 10 flows in their fold.
- Label rule: majority vote — sequence is attack if >= 6 of 10 flows
  are attacks.
- Identifier columns (Flow ID, Src IP, Dst IP, Timestamp, Src Port,
  Dst Port, Protocol) are preserved in the output for diagnostics but
  will be dropped at LSTM training time, matching RF and AE.

Run from the repository root:

    python -m src.preprocessing.sequences

Outputs (one row per flow within each sequence, with sequence_id):
- data/processed/cicids2017-engelen/train_sequences.parquet
- data/processed/cicids2017-engelen/val_sequences.parquet
- data/processed/cicids2017-engelen/test_sequences.parquet
- data/processed/cicids2017-engelen/_sequences_manifest.json
"""

from pathlib import Path
from dataclasses import dataclass, asdict
import json
import sys
import time

import numpy as np
import pandas as pd


# ---------- Configuration ---------- #

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "data" / "processed" / "cicids2017-engelen"

TRAIN_PATH = SPLITS_DIR / "train.parquet"
VAL_PATH   = SPLITS_DIR / "val.parquet"
TEST_PATH  = SPLITS_DIR / "test.parquet"

OUTPUT_PATHS = {
    "train": SPLITS_DIR / "train_sequences.parquet",
    "val":   SPLITS_DIR / "val_sequences.parquet",
    "test":  SPLITS_DIR / "test_sequences.parquet",
}

MANIFEST_PATH = SPLITS_DIR / "_sequences_manifest.json"

# Methodology constants
SEQUENCE_LENGTH = 10
LABEL_RULE_THRESHOLD = 6  # majority: >= 6 attacks out of 10
SOURCE_IP_COLUMN = "Src IP"
TIMESTAMP_COLUMN = "Timestamp"
TIMESTAMP_FORMAT = "%d/%m/%Y %I:%M:%S %p"


# ---------- Data structures ---------- #

@dataclass
class FoldSequenceStats:
    """Per-fold output stats for the manifest."""
    fold: str
    n_flows_in: int
    n_sources_in: int
    n_sources_with_min_flows: int
    n_sources_dropped: int
    n_sequences_out: int
    n_flows_used: int
    n_flows_dropped: int
    n_attack_sequences: int
    n_benign_sequences: int
    pct_attack_sequences: float


# ---------- Core sequence-building logic ---------- #

def build_sequences(
    df: pd.DataFrame,
    sequence_length: int = SEQUENCE_LENGTH,
    label_threshold: int = LABEL_RULE_THRESHOLD,
) -> tuple[pd.DataFrame, dict]:
    """Build non-overlapping sequences of `sequence_length` flows.

    Process:
    1. Group flows by `Src IP`.
    2. Sort each group by `Timestamp` ascending.
    3. Within each group, take non-overlapping windows of `sequence_length`.
    4. Discard the trailing partial window.
    5. Discard sources with fewer than `sequence_length` flows total.
    6. Compute the majority-vote label for each sequence.

    Parameters
    ----------
    df : pd.DataFrame
        Flow-level DataFrame with at least Src IP, Timestamp, is_attack columns.
    sequence_length : int
        Window size.
    label_threshold : int
        Minimum number of attacks in a window for the sequence to be
        labelled as attack.

    Returns
    -------
    df_sequences : pd.DataFrame
        Long-format: each sequence becomes `sequence_length` rows with a
        common `sequence_id` and `sequence_label`. Original flow columns
        are preserved.
    stats : dict
        Diagnostics: counts of input flows, sources, sequences out, etc.
    """
    if SOURCE_IP_COLUMN not in df.columns:
        raise ValueError(f"Input DataFrame must have a '{SOURCE_IP_COLUMN}' column.")
    if TIMESTAMP_COLUMN not in df.columns:
        raise ValueError(f"Input DataFrame must have a '{TIMESTAMP_COLUMN}' column.")
    if "is_attack" not in df.columns:
        raise ValueError("Input DataFrame must have an 'is_attack' column.")

    n_flows_in = len(df)
    n_sources_in = df[SOURCE_IP_COLUMN].nunique()

    # Parse timestamps once
    timestamps = pd.to_datetime(
        df[TIMESTAMP_COLUMN], format=TIMESTAMP_FORMAT, errors="coerce"
    )
    if timestamps.isna().any():
        n_failed = int(timestamps.isna().sum())
        raise ValueError(f"{n_failed} timestamps failed to parse.")

    df = df.copy()
    df["_parsed_timestamp"] = timestamps

    # Filter sources with enough flows
    flows_per_source = df.groupby(SOURCE_IP_COLUMN).size()
    eligible_sources = flows_per_source[flows_per_source >= sequence_length].index
    n_sources_with_min_flows = len(eligible_sources)
    n_sources_dropped = n_sources_in - n_sources_with_min_flows

    df = df[df[SOURCE_IP_COLUMN].isin(eligible_sources)].copy()

    # Build sequences per source
    sequence_rows = []
    next_seq_id = 0

    for src, group in df.groupby(SOURCE_IP_COLUMN, sort=False):
        group_sorted = group.sort_values("_parsed_timestamp", kind="stable").reset_index(drop=True)
        n = len(group_sorted)
        n_complete = n // sequence_length

        for i in range(n_complete):
            window = group_sorted.iloc[i * sequence_length : (i + 1) * sequence_length].copy()
            n_attack = int(window["is_attack"].sum())
            seq_label = 1 if n_attack >= label_threshold else 0

            window["sequence_id"] = next_seq_id
            window["sequence_position"] = range(sequence_length)
            window["sequence_label"] = seq_label

            sequence_rows.append(window)
            next_seq_id += 1

    if not sequence_rows:
        raise RuntimeError("No sequences produced. Check input data.")

    df_sequences = pd.concat(sequence_rows, ignore_index=True)
    df_sequences = df_sequences.drop(columns=["_parsed_timestamp"])

    n_sequences_out = next_seq_id
    n_flows_used = n_sequences_out * sequence_length
    n_flows_dropped = n_flows_in - n_flows_used

    # Sequence-level label distribution
    sequence_labels = df_sequences.drop_duplicates(subset=["sequence_id"])["sequence_label"]
    n_attack_seq = int((sequence_labels == 1).sum())
    n_benign_seq = int((sequence_labels == 0).sum())

    stats = {
        "n_flows_in": n_flows_in,
        "n_sources_in": n_sources_in,
        "n_sources_with_min_flows": n_sources_with_min_flows,
        "n_sources_dropped": n_sources_dropped,
        "n_sequences_out": n_sequences_out,
        "n_flows_used": n_flows_used,
        "n_flows_dropped": n_flows_dropped,
        "n_attack_sequences": n_attack_seq,
        "n_benign_sequences": n_benign_seq,
        "pct_attack_sequences": round(100 * n_attack_seq / n_sequences_out, 4) if n_sequences_out > 0 else 0.0,
    }
    return df_sequences, stats


# ---------- CLI entry point ---------- #

def main() -> int:
    if not TRAIN_PATH.exists():
        print(f"ERROR: train split not found at {TRAIN_PATH}", file=sys.stderr)
        return 1

    print(f"Splits directory: {SPLITS_DIR}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Majority-vote threshold: {LABEL_RULE_THRESHOLD} of {SEQUENCE_LENGTH}")
    print()

    fold_stats = []

    for fold_name, in_path, out_path in [
        ("train", TRAIN_PATH, OUTPUT_PATHS["train"]),
        ("val",   VAL_PATH,   OUTPUT_PATHS["val"]),
        ("test",  TEST_PATH,  OUTPUT_PATHS["test"]),
    ]:
        print(f"--- {fold_name} ---")
        t0 = time.time()
        df = pd.read_parquet(in_path)
        print(f"  Loaded {len(df):,} flows from {in_path.name}")

        df_seq, stats = build_sequences(df)
        elapsed = time.time() - t0

        print(f"  {stats['n_sources_in']} sources -> {stats['n_sources_with_min_flows']} eligible "
              f"({stats['n_sources_dropped']} dropped, < {SEQUENCE_LENGTH} flows)")
        print(f"  {stats['n_sequences_out']:,} sequences from {stats['n_flows_used']:,} flows "
              f"({stats['n_flows_dropped']:,} dropped as trailing partial)")
        print(f"  Sequence labels: {stats['n_attack_sequences']:,} attack, "
              f"{stats['n_benign_sequences']:,} benign "
              f"({stats['pct_attack_sequences']:.2f}% attack)")
        print(f"  Writing to {out_path.name}...", end=" ", flush=True)
        df_seq.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
        print(f"done ({elapsed:.1f}s)")
        print()

        fold_stats.append(FoldSequenceStats(fold=fold_name, **stats))

    # Manifest
    manifest = {
        "tool": "src/preprocessing/sequences.py",
        "splits_directory": str(SPLITS_DIR.relative_to(REPO_ROOT)),
        "sequence_length": SEQUENCE_LENGTH,
        "label_rule": f"majority vote: >= {LABEL_RULE_THRESHOLD} of {SEQUENCE_LENGTH} flows = attack",
        "grouping": f"by '{SOURCE_IP_COLUMN}', sorted by '{TIMESTAMP_COLUMN}'",
        "windowing": "non-overlapping",
        "folds": [asdict(s) for s in fold_stats],
    }

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to: {MANIFEST_PATH.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())