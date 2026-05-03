"""
split.py — Train/validation/test split for cleaned CIC-IDS2017.

Reads the five per-day cleaned Parquet files, concatenates them, and
applies a stratified random split into train/validation/test folds.

Methodology decisions (see docs/decisions/0003-split-strategy.md):
- 70 / 15 / 15 split ratio.
- Stratified random per-flow (literature standard for NIDS benchmark studies).
- Stratify on binary is_attack (BENIGN vs any attack); the original multi-class
  Label is preserved in the output for per-class evaluation in Chapter 5.
- Random seed 42 for reproducibility.

Identifier columns (Flow ID, Src IP, Dst IP, Timestamp) are preserved in
the output; they will be dropped at model-training time. This script is
strictly about partitioning, not feature selection.

Run from the repository root:

    python -m src.preprocessing.split
"""

from pathlib import Path
import json
import sys
import time
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ---------- Configuration ---------- #

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR   = REPO_ROOT / "data" / "processed" / "cicids2017-engelen"
DST_DIR   = REPO_ROOT / "data" / "processed" / "cicids2017-engelen"

DAY_FILES = [
    "Monday-WorkingHours.parquet",
    "Tuesday-WorkingHours.parquet",
    "Wednesday-WorkingHours.parquet",
    "Thursday-WorkingHours.parquet",
    "Friday-WorkingHours.parquet",
]

OUTPUT_NAMES = {
    "train": "train.parquet",
    "val":   "val.parquet",
    "test":  "test.parquet",
}

MANIFEST_PATH = DST_DIR / "_split_manifest.json"

RANDOM_SEED = 42

# 70 / 15 / 15. Implemented as two binary splits:
#   1. Split off 30% for val+test combined.
#   2. Split that 30% in half: 15% val, 15% test.
TRAIN_FRACTION = 0.70
# After the first split, the remaining 30% is split 50/50 into val/test.
VAL_FRACTION_OF_REMAINDER = 0.50


# ---------- Data structures ---------- #

@dataclass
class FoldStats:
    """Per-fold size and class distribution."""
    fold: str
    n_rows: int
    n_benign: int
    n_attack: int
    pct_benign: float
    pct_attack: float


# ---------- Core split logic ---------- #

def derive_is_attack(df: pd.DataFrame) -> pd.Series:
    """Derive a binary is_attack column from the multi-class Label."""
    if "Label" not in df.columns:
        raise ValueError("DataFrame must have a 'Label' column.")
    return (df["Label"] != "BENIGN").astype(int)


def stratified_split(
    df: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction_of_remainder: float = VAL_FRACTION_OF_REMAINDER,
    random_seed: int = RANDOM_SEED,
) -> dict[str, pd.DataFrame]:
    """Apply 70/15/15 stratified random split.

    Returns a dict with keys 'train', 'val', 'test', each mapping to a
    DataFrame. Stratification is on the binary is_attack derived from
    the Label column.
    """
    is_attack = derive_is_attack(df)

    # First split: separate train (70%) from val+test (30%)
    df_train, df_remainder, _, y_remainder = train_test_split(
        df,
        is_attack,
        test_size=(1 - train_fraction),
        stratify=is_attack,
        random_state=random_seed,
        shuffle=True,
    )

    # Second split: separate val from test, each 50% of the remainder
    df_val, df_test = train_test_split(
        df_remainder,
        test_size=(1 - val_fraction_of_remainder),
        stratify=y_remainder,
        random_state=random_seed,
        shuffle=True,
    )

    return {"train": df_train, "val": df_val, "test": df_test}


def fold_stats(name: str, df: pd.DataFrame) -> FoldStats:
    """Compute summary statistics for a single fold."""
    n_rows = len(df)
    is_attack = derive_is_attack(df)
    n_attack = int(is_attack.sum())
    n_benign = n_rows - n_attack
    return FoldStats(
        fold=name,
        n_rows=n_rows,
        n_benign=n_benign,
        n_attack=n_attack,
        pct_benign=round(100 * n_benign / n_rows, 4),
        pct_attack=round(100 * n_attack / n_rows, 4),
    )


# ---------- CLI entry point ---------- #

def main() -> int:
    if not SRC_DIR.exists():
        print(f"ERROR: source directory not found: {SRC_DIR}", file=sys.stderr)
        return 1

    DST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Source:      {SRC_DIR}")
    print(f"Destination: {DST_DIR}")
    print(f"Random seed: {RANDOM_SEED}")
    print()

    t0 = time.time()

    # Load and concatenate all five days
    print(f"Loading {len(DAY_FILES)} cleaned Parquet files...", end=" ", flush=True)
    dfs = []
    for fname in DAY_FILES:
        path = SRC_DIR / fname
        if not path.exists():
            print(f"\nERROR: missing input file: {path}", file=sys.stderr)
            return 1
        dfs.append(pd.read_parquet(path))
    df = pd.concat(dfs, ignore_index=True)
    print(f"done ({len(df):,} rows, {df.shape[1]} columns)")

    # Add the binary is_attack column (preserved in the output Parquets)
    print(f"Deriving binary is_attack column...", end=" ", flush=True)
    df["is_attack"] = derive_is_attack(df)
    print("done")

    # Apply the split
    print(f"Splitting {TRAIN_FRACTION:.0%} / "
          f"{(1-TRAIN_FRACTION)*VAL_FRACTION_OF_REMAINDER:.0%} / "
          f"{(1-TRAIN_FRACTION)*(1-VAL_FRACTION_OF_REMAINDER):.0%}...", end=" ", flush=True)
    folds = stratified_split(df)
    print("done")

    # Save each fold + collect stats
    stats: list[FoldStats] = []
    for fold_name, df_fold in folds.items():
        out_path = DST_DIR / OUTPUT_NAMES[fold_name]
        df_fold.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
        stat = fold_stats(fold_name, df_fold)
        stats.append(stat)
        print(f"  {fold_name:<6} -> {OUTPUT_NAMES[fold_name]} "
              f"({stat.n_rows:>8,} rows, "
              f"{stat.pct_attack:>5.2f}% attack)")

    duration = time.time() - t0

    # Manifest
    manifest = {
        "tool": "src/preprocessing/split.py",
        "source_directory": str(SRC_DIR.relative_to(REPO_ROOT)),
        "destination_directory": str(DST_DIR.relative_to(REPO_ROOT)),
        "random_seed": RANDOM_SEED,
        "split_strategy": "stratified random per-flow on binary is_attack",
        "train_fraction": TRAIN_FRACTION,
        "val_fraction_of_remainder": VAL_FRACTION_OF_REMAINDER,
        "total_rows_in": len(df),
        "duration_seconds": round(duration, 2),
        "folds": [asdict(s) for s in stats],
    }

    # Sanity check: folds sum to total
    total_out = sum(s.n_rows for s in stats)
    manifest["fold_total_rows"] = total_out
    manifest["fold_total_matches_input"] = (total_out == len(df))

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nAll splitting complete in {duration:.1f}s.")
    print(f"Manifest written to: {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    print(f"\nTotal: {len(df):,} rows -> "
          f"{stats[0].n_rows:,} train + "
          f"{stats[1].n_rows:,} val + "
          f"{stats[2].n_rows:,} test")

    return 0


if __name__ == "__main__":
    sys.exit(main())