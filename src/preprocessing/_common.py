"""
_common.py — Shared cleaning logic used by clean.py (CIC-IDS2017) and
clean_unsw.py (CIC-UNSW-NB15).

Contains:
- NEAR_CONSTANT_COLS — the 5-column drop list that applies to both datasets.
- clean_dataframe() — the row + column drop pipeline.

Dataset-specific orchestration (file I/O, label merging, path resolution)
lives in clean.py and clean_unsw.py respectively.
"""

import numpy as np
import pandas as pd


# 5 near-constant columns identified in notebook 02 (CIC-IDS2017) and
# verified on CIC-UNSW-NB15 in notebook 04. `Subflow Bwd Packets` was
# originally on this list but was found to be informative on UNSW
# (n_unique=551, modal_pct=97.0%); see notebook 02 follow-up cell and
# Methodology chapter 4.3.
NEAR_CONSTANT_COLS = [
    "Bwd URG Flags",
    "URG Flag Count",
    "Fwd URG Flags",
    "ECE Flag Count",
    "CWR Flag Count",
]


def clean_dataframe(
    df: pd.DataFrame,
    apply_protocol_filter: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Apply row and column drops to a flow DataFrame.

    Drops applied in this order:

    1. (CIC-IDS2017 only) Rows where Protocol == 0 — non-IP frame artefacts.
       Skipped for UNSW (no Protocol column).
    2. Rows where any numeric feature is NaN or +/-Inf — zero-duration or
       single-packet flows.
    3. Columns from NEAR_CONSTANT_COLS that are present in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input flows.
    apply_protocol_filter : bool, default True
        Whether to apply the Protocol == 0 row filter. CIC-IDS2017 has a
        Protocol column and uses this filter; CIC-UNSW-NB15 does not have
        the column and sets this to False.

    Returns
    -------
    df_clean : pd.DataFrame
    stats : dict
        Keys: rows_in, rows_out, cols_in, cols_out,
        n_dropped_protocol_zero, n_dropped_nan_inf.
        n_dropped_protocol_zero is 0 when apply_protocol_filter is False.
    """
    rows_in = len(df)
    cols_in = df.shape[1]

    # --- Row drop 1: Protocol == 0 (CIC-IDS2017 only) ---
    if apply_protocol_filter and "Protocol" in df.columns:
        protocol_zero_mask = df["Protocol"] == 0
        n_dropped_protocol = int(protocol_zero_mask.sum())
        df = df.loc[~protocol_zero_mask].copy()
    else:
        n_dropped_protocol = 0

    # --- Row drop 2: NaN or +/-Inf in any numeric column ---
    numeric = df.select_dtypes(include="number")
    nan_mask = numeric.isna().any(axis=1)
    inf_mask = numeric.apply(lambda c: np.isinf(c)).any(axis=1)
    bad_row_mask = nan_mask | inf_mask
    n_dropped_nan_inf = int(bad_row_mask.sum())
    df = df.loc[~bad_row_mask].copy()

    # --- Column drops: near-constant features ---
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