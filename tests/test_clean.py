"""
test_clean.py — Unit tests for src/preprocessing/clean.py

Tests use small synthetic DataFrames designed so that the expected outcome
is hand-verifiable. The objective is to catch regressions where the cleaning
logic does the wrong thing — not to test that pandas works.

Run from the repository root:

    pytest tests/test_clean.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing._common import clean_dataframe, NEAR_CONSTANT_COLS


# ---------- Fixtures ---------- #

def make_minimal_df(n_rows: int = 10) -> pd.DataFrame:
    """Construct a minimal CICFlowMeter-shaped DataFrame for testing.

    All rows are 'good' by default — Protocol > 0, no NaN/Inf. Tests
    inject specific bad rows by mutating cells before calling the
    cleaning function.
    """
    df = pd.DataFrame({
        # Identifiers (pass-through; cleaning does not touch them)
        "Flow ID":          [f"flow{i}" for i in range(n_rows)],
        "Src IP":           ["10.0.0.1"] * n_rows,
        "Dst IP":           ["10.0.0.2"] * n_rows,
        "Src Port":         [12345] * n_rows,
        "Dst Port":         [80] * n_rows,
        "Protocol":         [6] * n_rows,        # 6 = TCP, valid
        "Timestamp":        [f"03/07/2017 0{i % 9}:00:00 PM" for i in range(n_rows)],
        # A handful of numeric features
        "Flow Duration":    [1000.0 * (i + 1) for i in range(n_rows)],
        "Total Fwd Packet": list(range(1, n_rows + 1)),
        "Flow Bytes/s":     [100.0 + i for i in range(n_rows)],
        # The 6 near-constant columns (mostly 0; they should be dropped)
        "Bwd URG Flags":      [0] * n_rows,
        "URG Flag Count":     [0] * n_rows,
        "Fwd URG Flags":      [0] * n_rows,
        "Subflow Bwd Packets":[0] * n_rows,
        "ECE Flag Count":     [0] * n_rows,
        "CWR Flag Count":     [0] * n_rows,
        # Label
        "Label":            ["BENIGN"] * n_rows,
    })
    return df


# ---------- Tests ---------- #

class TestRowDrops:
    """Row-level filter tests."""

    def test_drops_rows_with_protocol_zero(self):
        df = make_minimal_df(n_rows=10)
        df.loc[3, "Protocol"] = 0
        df.loc[7, "Protocol"] = 0
        df_clean, stats = clean_dataframe(df)
        assert stats["n_dropped_protocol_zero"] == 2
        assert stats["rows_in"] == 10
        assert stats["rows_out"] == 8

    def test_drops_rows_with_nan_in_numeric(self):
        df = make_minimal_df(n_rows=10)
        df.loc[2, "Flow Duration"] = np.nan
        df.loc[5, "Flow Bytes/s"] = np.nan
        df_clean, stats = clean_dataframe(df)
        assert stats["n_dropped_nan_inf"] == 2
        assert stats["rows_out"] == 8

    def test_drops_rows_with_positive_inf(self):
        df = make_minimal_df(n_rows=10)
        df.loc[4, "Flow Bytes/s"] = np.inf
        df_clean, stats = clean_dataframe(df)
        assert stats["n_dropped_nan_inf"] == 1
        assert stats["rows_out"] == 9

    def test_drops_rows_with_negative_inf(self):
        df = make_minimal_df(n_rows=10)
        df.loc[4, "Flow Bytes/s"] = -np.inf
        df_clean, stats = clean_dataframe(df)
        assert stats["n_dropped_nan_inf"] == 1
        assert stats["rows_out"] == 9

    def test_protocol_filter_runs_before_nan_filter(self):
        """A row with Protocol==0 AND a NaN should count once, against Protocol filter."""
        df = make_minimal_df(n_rows=10)
        df.loc[3, "Protocol"] = 0
        df.loc[3, "Flow Bytes/s"] = np.nan  # same row gets both issues
        df_clean, stats = clean_dataframe(df)
        # The row is dropped by Protocol filter; NaN filter sees 9 rows after
        assert stats["n_dropped_protocol_zero"] == 1
        assert stats["n_dropped_nan_inf"] == 0  # already gone before NaN check
        assert stats["rows_out"] == 9

    def test_no_bad_rows_means_no_drops(self):
        df = make_minimal_df(n_rows=10)
        df_clean, stats = clean_dataframe(df)
        assert stats["n_dropped_protocol_zero"] == 0
        assert stats["n_dropped_nan_inf"] == 0
        assert stats["rows_out"] == 10


class TestColumnDrops:
    """Column-level drop tests."""

    def test_drops_all_five_near_constant_columns(self):
        df = make_minimal_df(n_rows=10)
        df_clean, stats = clean_dataframe(df)
        for col in NEAR_CONSTANT_COLS:
            assert col not in df_clean.columns, f"{col} should have been dropped"
        # cols_in includes the 5 near-constant; cols_out should be 5 fewer
        assert stats["cols_out"] == stats["cols_in"] - 5

    def test_keeps_other_columns(self):
        df = make_minimal_df(n_rows=10)
        df_clean, _ = clean_dataframe(df)
        for col in ("Flow ID", "Src IP", "Protocol", "Flow Duration",
                    "Total Fwd Packet", "Subflow Bwd Packets", "Label"):
            assert col in df_clean.columns

    def test_handles_missing_near_constant_column_gracefully(self):
        """If a near-constant column is already missing, cleaning should not error."""
        df = make_minimal_df(n_rows=10)
        df = df.drop(columns=["URG Flag Count"])  # remove one of them
        df_clean, stats = clean_dataframe(df)
        # The other 5 should still be dropped without error
        for col in NEAR_CONSTANT_COLS:
            assert col not in df_clean.columns


class TestStatsManifest:
    """The stats dict must accurately reflect the operation."""

    def test_stats_keys_are_complete(self):
        df = make_minimal_df(n_rows=10)
        _, stats = clean_dataframe(df)
        expected_keys = {
            "rows_in", "rows_out", "cols_in", "cols_out",
            "n_dropped_protocol_zero", "n_dropped_nan_inf",
        }
        assert set(stats.keys()) == expected_keys

    def test_stats_arithmetic_consistency(self):
        df = make_minimal_df(n_rows=10)
        df.loc[3, "Protocol"] = 0
        df.loc[5, "Flow Bytes/s"] = np.nan
        df.loc[7, "Flow Duration"] = np.inf
        df_clean, stats = clean_dataframe(df)
        # Total drops = protocol drops + NaN/Inf drops
        assert (stats["rows_in"] - stats["rows_out"]
                == stats["n_dropped_protocol_zero"] + stats["n_dropped_nan_inf"])

    def test_empty_dataframe_returns_zero_drops(self):
        df = make_minimal_df(n_rows=0)
        df_clean, stats = clean_dataframe(df)
        assert stats["rows_in"] == 0
        assert stats["rows_out"] == 0
        assert stats["n_dropped_protocol_zero"] == 0
        assert stats["n_dropped_nan_inf"] == 0


class TestDeterminism:
    """Same input must always produce same output."""

    def test_idempotent(self):
        df = make_minimal_df(n_rows=10)
        df.loc[3, "Protocol"] = 0
        df.loc[5, "Flow Bytes/s"] = np.nan
        df_clean1, stats1 = clean_dataframe(df.copy())
        df_clean2, stats2 = clean_dataframe(df.copy())
        assert stats1 == stats2
        pd.testing.assert_frame_equal(df_clean1, df_clean2)