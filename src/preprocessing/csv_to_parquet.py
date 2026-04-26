"""
csv_to_parquet.py — Convert Engelen-cleaned CIC-IDS2017 CSVs to Parquet.

One-shot conversion script. Reads the five CSV files from
data/raw/cicids2017-engelen/ and writes Parquet equivalents to
data/interim/cicids2017-engelen/.

Run from the repository root:

    python -m src.preprocessing.csv_to_parquet

Notes on dtype handling:
- CICFlowMeter outputs use mixed int / float / string columns.
- We let pandas infer dtypes on first read, then optimise:
  - int columns kept as int64 (no risk of overflow on flow features)
  - float columns downcast to float32 (halves memory, no NIDS-relevant
    precision loss)
  - object (string) columns kept as-is (label, flow ID, timestamps)

The conversion is idempotent: re-running it overwrites existing Parquet
files. Source CSVs are never modified.
"""

from pathlib import Path
import sys
import time

import pandas as pd


# Resolve paths relative to repository root, regardless of where the
# script is invoked from.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "data" / "raw" / "cicids2017-engelen"
DST_DIR = REPO_ROOT / "data" / "interim" / "cicids2017-engelen"

CSV_FILES = [
    "Monday-WorkingHours.csv",
    "Tuesday-WorkingHours.csv",
    "Wednesday-WorkingHours.csv",
    "Thursday-WorkingHours.csv",
    "Friday-WorkingHours.csv",
]


def downcast_floats(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float64 columns to float32. Halves memory, no relevant
    precision loss for IDS flow features (precision well below 1e-6
    matters nowhere in this dataset)."""
    float_cols = df.select_dtypes(include="float64").columns
    for col in float_cols:
        df[col] = df[col].astype("float32")
    return df


def convert_one(src_path: Path, dst_path: Path) -> dict:
    """Convert a single CSV to Parquet, returning a stats dict."""
    t0 = time.time()
    df = pd.read_csv(src_path, low_memory=False)
    t_read = time.time() - t0

    rows, cols = df.shape
    mem_before = df.memory_usage(deep=True).sum() / 1024**2

    df = downcast_floats(df)
    mem_after = df.memory_usage(deep=True).sum() / 1024**2

    t1 = time.time()
    df.to_parquet(dst_path, engine="pyarrow", compression="snappy", index=False)
    t_write = time.time() - t1

    src_size = src_path.stat().st_size / 1024**2
    dst_size = dst_path.stat().st_size / 1024**2

    return {
        "file": src_path.name,
        "rows": rows,
        "cols": cols,
        "csv_mb": round(src_size, 1),
        "parquet_mb": round(dst_size, 1),
        "compression_ratio": round(src_size / dst_size, 1),
        "memory_mb_before": round(mem_before, 0),
        "memory_mb_after": round(mem_after, 0),
        "read_seconds": round(t_read, 1),
        "write_seconds": round(t_write, 1),
    }


def main() -> int:
    if not SRC_DIR.exists():
        print(f"ERROR: source directory not found: {SRC_DIR}", file=sys.stderr)
        return 1

    DST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Source: {SRC_DIR}")
    print(f"Destination: {DST_DIR}")
    print(f"Converting {len(CSV_FILES)} files...\n")

    results = []
    total_t0 = time.time()

    for csv_name in CSV_FILES:
        src_path = SRC_DIR / csv_name
        dst_name = csv_name.replace(".csv", ".parquet")
        dst_path = DST_DIR / dst_name

        if not src_path.exists():
            print(f"SKIP: {csv_name} not found in source directory")
            continue

        print(f"Converting {csv_name} ...", end=" ", flush=True)
        stats = convert_one(src_path, dst_path)
        results.append(stats)
        print(f"done ({stats['rows']:,} rows, "
              f"{stats['csv_mb']} MB → {stats['parquet_mb']} MB, "
              f"{stats['read_seconds']}s read + {stats['write_seconds']}s write)")

    total_t = time.time() - total_t0

    print(f"\nAll conversions complete in {total_t:.1f}s.")

    # Summary table
    print("\nSummary:")
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())