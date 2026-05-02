"""
clean_unsw.py — Convert and clean CIC-UNSW-NB15 for use as the secondary
cross-dataset evaluation set.

CIC-UNSW-NB15 differs structurally from CIC-IDS2017 in three ways:

1. Source format is CSV (Data.csv, Label.csv), not Parquet.
2. Labels live in a separate Label.csv file, joined to Data.csv by row index.
3. No Protocol column to filter on (CIC team has stripped identifier columns).

This script handles all three:

- Reads Data.csv and Label.csv from data/raw/cic-unsw-nb15/.
- Joins them on row index, attaching label codes (0-9) and decoded label
  strings to each row.
- Applies the row + column drop pipeline from _common.py with the protocol
  filter disabled.
- Writes a single cleaned Parquet file to data/processed/cic-unsw-nb15/.
- Writes an audit manifest.

Run from the repository root:

    python -m src.preprocessing.clean_unsw
"""

from pathlib import Path
import json
import sys
import time
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from src.preprocessing._common import NEAR_CONSTANT_COLS, clean_dataframe


# ---------- Configuration ---------- #

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR   = REPO_ROOT / "data" / "raw" / "cic-unsw-nb15"
DST_DIR   = REPO_ROOT / "data" / "processed" / "cic-unsw-nb15"

DATA_CSV    = SRC_DIR / "Data.csv"
LABEL_CSV   = SRC_DIR / "Label.csv"
OUTPUT_NAME = "cic-unsw-nb15.parquet"

MANIFEST_PATH = DST_DIR / "_clean_manifest.json"

# Label encoding from the CIC team's Readme.txt
LABEL_NAMES = {
    0: "Benign",
    1: "Analysis",
    2: "Backdoor",
    3: "DoS",
    4: "Exploits",
    5: "Fuzzers",
    6: "Generic",
    7: "Reconnaissance",
    8: "Shellcode",
    9: "Worms",
}


# ---------- Data structures ---------- #

@dataclass
class CleanResult:
    """Outcome of cleaning the UNSW dataset."""
    rows_in: int
    rows_out: int
    cols_in: int                    # columns after Data.csv + Label.csv merge, before cleaning
    cols_out: int                   # columns after cleaning
    n_dropped_protocol_zero: int    # always 0 for UNSW; field kept for parallel manifest format
    n_dropped_nan_inf: int
    n_dropped_total: int
    duration_seconds: float
    label_distribution: dict        # decoded label name -> count


# ---------- Loading ---------- #

def load_and_merge(data_path: Path, label_path: Path) -> pd.DataFrame:
    """Read Data.csv and Label.csv, merge them by row index.

    Both files have headers. Data.csv has 76 feature columns; Label.csv has
    one column 'Label' containing integers 0-9. The CSVs are aligned by row
    index — row i in Data corresponds to row i in Label.

    Returns a single DataFrame with the 76 feature columns plus 'Label'
    (integer) and 'LabelName' (decoded string).
    """
    df_data = pd.read_csv(data_path, low_memory=False)
    df_label = pd.read_csv(label_path)

    if len(df_data) != len(df_label):
        raise ValueError(
            f"Row count mismatch between Data.csv ({len(df_data):,}) "
            f"and Label.csv ({len(df_label):,}). Cannot join."
        )

    label_col = df_label.columns[0]  # the file has a single column called 'Label'
    df_data = df_data.copy()
    df_data["Label"]     = df_label[label_col].values
    df_data["LabelName"] = df_data["Label"].map(LABEL_NAMES)

    if df_data["LabelName"].isna().any():
        unknown = df_data.loc[df_data["LabelName"].isna(), "Label"].unique()
        raise ValueError(f"Unknown label codes in Label.csv: {unknown}")

    return df_data


# ---------- Float downcast (mirroring csv_to_parquet.py) ---------- #

def downcast_floats(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float64 columns to float32 to halve memory footprint."""
    float_cols = df.select_dtypes(include="float64").columns
    for col in float_cols:
        df[col] = df[col].astype("float32")
    return df


# ---------- CLI entry point ---------- #

def main() -> int:
    if not DATA_CSV.exists() or not LABEL_CSV.exists():
        print(f"ERROR: missing input files in {SRC_DIR}", file=sys.stderr)
        print(f"  Data.csv  exists: {DATA_CSV.exists()}", file=sys.stderr)
        print(f"  Label.csv exists: {LABEL_CSV.exists()}", file=sys.stderr)
        return 1

    DST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Source:      {SRC_DIR}")
    print(f"Destination: {DST_DIR}")
    print()

    t0 = time.time()

    print(f"Loading Data.csv and Label.csv...", end=" ", flush=True)
    df = load_and_merge(DATA_CSV, LABEL_CSV)
    print(f"done ({len(df):,} rows, {df.shape[1]} columns)")

    print(f"Downcasting float64 to float32...", end=" ", flush=True)
    df = downcast_floats(df)
    print("done")

    print(f"Applying clean_dataframe (protocol filter disabled)...", end=" ", flush=True)
    df_clean, stats = clean_dataframe(df, apply_protocol_filter=False)
    print(
        f"done ({stats['rows_in']:,} -> {stats['rows_out']:,} rows, "
        f"{stats['cols_in']} -> {stats['cols_out']} cols)"
    )

    out_path = DST_DIR / OUTPUT_NAME
    print(f"Writing Parquet output...", end=" ", flush=True)
    df_clean.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    print(f"done -> {out_path.relative_to(REPO_ROOT)}")

    duration = time.time() - t0

    # Compute label distribution for the manifest
    label_dist = (
        df_clean["LabelName"]
        .value_counts()
        .to_dict()
    )

    result = CleanResult(
        rows_in=stats["rows_in"],
        rows_out=stats["rows_out"],
        cols_in=stats["cols_in"],
        cols_out=stats["cols_out"],
        n_dropped_protocol_zero=stats["n_dropped_protocol_zero"],
        n_dropped_nan_inf=stats["n_dropped_nan_inf"],
        n_dropped_total=stats["rows_in"] - stats["rows_out"],
        duration_seconds=round(duration, 2),
        label_distribution=label_dist,
    )

    manifest = {
        "tool": "src/preprocessing/clean_unsw.py",
        "source_directory": str(SRC_DIR.relative_to(REPO_ROOT)),
        "destination_directory": str(DST_DIR.relative_to(REPO_ROOT)),
        "near_constant_columns_dropped": NEAR_CONSTANT_COLS,
        "row_filters_applied": [
            "any numeric feature is NaN or +/-Inf",
        ],
        "row_filters_skipped": [
            "Protocol == 0 (no Protocol column in CIC-UNSW-NB15)",
        ],
        "label_encoding": LABEL_NAMES,
        "result": asdict(result),
    }
    manifest["result"]["pct_retained"] = round(
        100 * result.rows_out / result.rows_in, 4
    )

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nAll cleaning complete in {duration:.1f}s.")
    print(f"Manifest written to: {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    print(
        f"\nTotals: {result.rows_in:,} -> {result.rows_out:,} rows "
        f"(retained {manifest['result']['pct_retained']:.4f}%)"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())