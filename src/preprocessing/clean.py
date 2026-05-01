"""
clean.py — Apply data-integrity filters to Engelen-cleaned CIC-IDS2017.

Reads a Parquet file (one day's flows), applies the row and column drop
rules established by EDA notebook 02, and writes a cleaned Parquet output.

Drop rules (rationale in notebook 02 and Methodology chapter §4.3):

ROW DROPS (applied first, in order):
1. Rows where Protocol == 0 (non-IP frame artefacts from CICFlowMeter,
   1,628 rows in the full dataset).
2. Rows where any numeric feature is NaN or +/-Inf (zero-duration or
   single-packet flows; 793 rows in the full dataset).

COLUMN DROPS (applied after row drops):
6 near-constant columns (modal value zero in >=99.9% of rows):
    Bwd URG Flags, URG Flag Count, Fwd URG Flags,
    Subflow Bwd Packets, ECE Flag Count, CWR Flag Count

Order of operations matters for audit trail: row-drop counts in the manifest
are independent and meaningful only if columns are still present when we
count NaN/Inf rows.

Run from the repository root:

    python -m src.preprocessing.clean

This processes all five days, writing cleaned outputs to
data/processed/cicids2017-engelen/ and a JSON manifest summarising the
operation.
"""

from pathlib import Path
import json
import sys
import time
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


# ---------- Configuration ---------- #

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "data" / "interim" / "cicids2017-engelen"
DST_DIR = REPO_ROOT / "data" / "processed" / "cicids2017-engelen"
MANIFEST_PATH = DST_DIR / "_clean_manifest.json"

DAY_FILES = [
    "Monday-WorkingHours.parquet",
    "Tuesday-WorkingHours.parquet",
    "Wednesday-WorkingHours.parquet",
    "Thursday-WorkingHours.parquet",
    "Friday-WorkingHours.parquet",
]

# 6 near-constant columns identified in notebook 02
NEAR_CONSTANT_COLS = [
    "Bwd URG Flags",
    "URG Flag Count",
    "Fwd URG Flags",
    "Subflow Bwd Packets",
    "ECE Flag Count",
    "CWR Flag Count",
]


# ---------- Data structures ---------- #

@dataclass
class CleanResult:
    """Per-file outcome of the cleaning operation. Aggregated into the manifest."""
    file: str
    rows_in: int
    rows_out: int
    cols_in: int
    cols_out: int
    n_dropped_protocol_zero: int
    n_dropped_nan_inf: int
    n_dropped_total: int
    duration_seconds: float


# ---------- Cleaning logic ---------- #

def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply row and column drops to a flow DataFrame. Returns (cleaned_df, stats).

    Stats dict has keys: rows_in, rows_out, cols_in, cols_out,
    n_dropped_protocol_zero, n_dropped_nan_inf.
    """
    rows_in = len(df)
    cols_in = df.shape[1]

    # --- Row drops, sequentially with audit ---

    # 1. Drop Protocol == 0 (non-IP frame artefacts)
    protocol_zero_mask = df["Protocol"] == 0
    n_dropped_protocol = int(protocol_zero_mask.sum())
    df = df.loc[~protocol_zero_mask].copy()

    # 2. Drop rows with NaN or +/-Inf in any numeric column
    numeric = df.select_dtypes(include="number")
    nan_mask = numeric.isna().any(axis=1)
    inf_mask = numeric.apply(lambda c: np.isinf(c)).any(axis=1)
    bad_row_mask = nan_mask | inf_mask
    n_dropped_nan_inf = int(bad_row_mask.sum())
    df = df.loc[~bad_row_mask].copy()

    # --- Column drops ---

    cols_to_drop = [c for c in NEAR_CONSTANT_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    rows_out = len(df)
    cols_out = df.shape[1]

    stats = {
        "rows_in": rows_in,
        "rows_out": rows_out,
        "cols_in": cols_in,
        "cols_out": cols_out,
        "n_dropped_protocol_zero": n_dropped_protocol,
        "n_dropped_nan_inf": n_dropped_nan_inf,
    }
    return df, stats


def clean_one(src_path: Path, dst_path: Path) -> CleanResult:
    """Read one Parquet file, clean it, write the result."""
    t0 = time.time()
    df = pd.read_parquet(src_path)
    df_clean, stats = clean_dataframe(df)
    df_clean.to_parquet(dst_path, engine="pyarrow", compression="snappy", index=False)
    duration = time.time() - t0

    return CleanResult(
        file=src_path.name,
        rows_in=stats["rows_in"],
        rows_out=stats["rows_out"],
        cols_in=stats["cols_in"],
        cols_out=stats["cols_out"],
        n_dropped_protocol_zero=stats["n_dropped_protocol_zero"],
        n_dropped_nan_inf=stats["n_dropped_nan_inf"],
        n_dropped_total=stats["rows_in"] - stats["rows_out"],
        duration_seconds=round(duration, 2),
    )


# ---------- CLI entry point ---------- #

def main() -> int:
    if not SRC_DIR.exists():
        print(f"ERROR: source directory not found: {SRC_DIR}", file=sys.stderr)
        return 1

    DST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Source:      {SRC_DIR}")
    print(f"Destination: {DST_DIR}")
    print(f"Cleaning {len(DAY_FILES)} files...\n")

    results: list[CleanResult] = []
    total_t0 = time.time()

    for fname in DAY_FILES:
        src = SRC_DIR / fname
        dst = DST_DIR / fname

        if not src.exists():
            print(f"SKIP: {fname} not found in source directory")
            continue

        print(f"Cleaning {fname} ...", end=" ", flush=True)
        result = clean_one(src, dst)
        results.append(result)
        print(
            f"done ({result.rows_in:,} -> {result.rows_out:,} rows, "
            f"{result.cols_in} -> {result.cols_out} cols, "
            f"{result.duration_seconds}s)"
        )

    total_t = time.time() - total_t0

    # ---------- Aggregate manifest ---------- #
    manifest = {
        "tool": "src/preprocessing/clean.py",
        "source_directory": str(SRC_DIR.relative_to(REPO_ROOT)),
        "destination_directory": str(DST_DIR.relative_to(REPO_ROOT)),
        "near_constant_columns_dropped": NEAR_CONSTANT_COLS,
        "row_filters_applied": [
            "Protocol == 0",
            "any numeric feature is NaN or +/-Inf",
        ],
        "duration_seconds_total": round(total_t, 2),
        "files": [asdict(r) for r in results],
    }

    # Aggregate totals
    manifest["totals"] = {
        "rows_in":                 sum(r.rows_in for r in results),
        "rows_out":                sum(r.rows_out for r in results),
        "n_dropped_protocol_zero": sum(r.n_dropped_protocol_zero for r in results),
        "n_dropped_nan_inf":       sum(r.n_dropped_nan_inf for r in results),
        "n_dropped_total":         sum(r.n_dropped_total for r in results),
    }
    manifest["totals"]["pct_retained"] = round(
        100 * manifest["totals"]["rows_out"] / manifest["totals"]["rows_in"], 4
    )

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nAll cleaning complete in {total_t:.1f}s.")
    print(f"Manifest written to: {MANIFEST_PATH.relative_to(REPO_ROOT)}")

    # Summary line
    print(
        f"\nTotals: {manifest['totals']['rows_in']:,} -> "
        f"{manifest['totals']['rows_out']:,} rows "
        f"(retained {manifest['totals']['pct_retained']:.4f}%)"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())