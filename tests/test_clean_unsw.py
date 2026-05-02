"""
test_clean_unsw.py — Unit tests for src/preprocessing/clean_unsw.py

The shared `clean_dataframe` logic is already tested in test_clean.py;
these tests focus on UNSW-specific behaviour:

- The CSV loading + label merging pipeline.
- The protocol filter being correctly disabled for UNSW.
- The label distribution accounting in the manifest.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing._common import clean_dataframe, NEAR_CONSTANT_COLS
from src.preprocessing.clean_unsw import LABEL_NAMES, load_and_merge


# ---------- Fixtures ---------- #

def make_unsw_minimal_df(n_rows: int = 10) -> pd.DataFrame:
    """Construct a minimal UNSW-shaped DataFrame for testing.

    UNSW has 76 feature columns plus Label (integer) and LabelName (string).
    No identifier columns (no Flow ID, IPs, ports, Protocol, Timestamp).
    """
    df = pd.DataFrame({
        # A handful of numeric features that exist in UNSW
        "Flow Duration":         [1000.0 * (i + 1) for i in range(n_rows)],
        "Total Fwd Packet":      list(range(1, n_rows + 1)),
        "Total Bwd packets":     list(range(1, n_rows + 1)),
        "Flow Bytes/s":          [100.0 + i for i in range(n_rows)],
        "Subflow Bwd Packets":   list(range(n_rows)),  # informative on UNSW; must be kept
        # The 5 near-constant columns we drop
        "Bwd URG Flags":         [0] * n_rows,
        "URG Flag Count":        [0] * n_rows,
        "Fwd URG Flags":         [0] * n_rows,
        "ECE Flag Count":        [0] * n_rows,
        "CWR Flag Count":        [0] * n_rows,
        # Labels — 0 = Benign, others are attack codes
        "Label":                 [0] * n_rows,
        "LabelName":             ["Benign"] * n_rows,
    })
    return df


# ---------- Tests for load_and_merge ---------- #

class TestLoadAndMerge:
    """Tests for the CSV-loading-and-label-merging logic."""

    def test_merge_attaches_label_and_labelname(self, tmp_path):
        # Build small Data.csv and Label.csv on disk
        data = pd.DataFrame({
            "Flow Duration":     [100, 200, 300],
            "Total Fwd Packet":  [1, 2, 3],
        })
        label = pd.DataFrame({"Label": [0, 4, 9]})

        data_path  = tmp_path / "Data.csv"
        label_path = tmp_path / "Label.csv"
        data.to_csv(data_path, index=False)
        label.to_csv(label_path, index=False)

        merged = load_and_merge(data_path, label_path)
        assert "Label"     in merged.columns
        assert "LabelName" in merged.columns
        assert list(merged["Label"])     == [0, 4, 9]
        assert list(merged["LabelName"]) == ["Benign", "Exploits", "Worms"]

    def test_merge_raises_on_row_count_mismatch(self, tmp_path):
        data = pd.DataFrame({"Flow Duration": [100, 200, 300]})
        label = pd.DataFrame({"Label": [0, 4]})  # one row short

        data_path  = tmp_path / "Data.csv"
        label_path = tmp_path / "Label.csv"
        data.to_csv(data_path, index=False)
        label.to_csv(label_path, index=False)

        with pytest.raises(ValueError, match="Row count mismatch"):
            load_and_merge(data_path, label_path)

    def test_merge_raises_on_unknown_label_code(self, tmp_path):
        data = pd.DataFrame({"Flow Duration": [100, 200]})
        label = pd.DataFrame({"Label": [0, 99]})  # 99 is not in LABEL_NAMES

        data_path  = tmp_path / "Data.csv"
        label_path = tmp_path / "Label.csv"
        data.to_csv(data_path, index=False)
        label.to_csv(label_path, index=False)

        with pytest.raises(ValueError, match="Unknown label codes"):
            load_and_merge(data_path, label_path)


# ---------- Tests for cleaning behaviour on UNSW shape ---------- #

class TestUnswCleaning:
    """Confirm clean_dataframe behaves correctly on UNSW-shaped data."""

    def test_protocol_filter_disabled_does_not_drop_rows(self):
        """With apply_protocol_filter=False, no rows are dropped for protocol reasons
        even if a Protocol column happens to exist with value 0."""
        df = make_unsw_minimal_df(n_rows=10)
        # Inject a Protocol column with some zeros, just to be sure
        df["Protocol"] = [0, 0, 6, 17, 6, 6, 17, 6, 0, 6]
        df_clean, stats = clean_dataframe(df, apply_protocol_filter=False)
        assert stats["n_dropped_protocol_zero"] == 0
        assert stats["rows_out"] == 10  # nothing dropped

    def test_nan_inf_filter_still_applies_on_unsw(self):
        df = make_unsw_minimal_df(n_rows=10)
        df.loc[2, "Flow Duration"] = np.nan
        df.loc[5, "Flow Bytes/s"] = np.inf
        df_clean, stats = clean_dataframe(df, apply_protocol_filter=False)
        assert stats["n_dropped_nan_inf"] == 2
        assert stats["rows_out"] == 8

    def test_subflow_bwd_packets_is_kept_on_unsw_data(self):
        """Critical regression test: Subflow Bwd Packets must NOT be dropped."""
        df = make_unsw_minimal_df(n_rows=10)
        df_clean, _ = clean_dataframe(df, apply_protocol_filter=False)
        assert "Subflow Bwd Packets" in df_clean.columns

    def test_five_near_constant_cols_still_dropped_on_unsw(self):
        df = make_unsw_minimal_df(n_rows=10)
        df_clean, stats = clean_dataframe(df, apply_protocol_filter=False)
        for col in NEAR_CONSTANT_COLS:
            assert col not in df_clean.columns
        # 12 cols in (76 features minus most + Label, LabelName + 5 near-constant);
        # 5 dropped means 7 out
        assert stats["cols_out"] == stats["cols_in"] - 5

    def test_label_columns_preserved_on_unsw(self):
        """Label and LabelName must survive cleaning (used downstream for stratification)."""
        df = make_unsw_minimal_df(n_rows=10)
        df_clean, _ = clean_dataframe(df, apply_protocol_filter=False)
        assert "Label"     in df_clean.columns
        assert "LabelName" in df_clean.columns


# ---------- Sanity check on the LABEL_NAMES mapping ---------- #

class TestLabelMapping:
    """The label encoding must match the CIC team's published Readme."""

    def test_label_names_complete_zero_to_nine(self):
        assert set(LABEL_NAMES.keys()) == set(range(10))

    def test_zero_is_benign(self):
        assert LABEL_NAMES[0] == "Benign"

    def test_known_attack_codes(self):
        assert LABEL_NAMES[4] == "Exploits"
        assert LABEL_NAMES[5] == "Fuzzers"
        assert LABEL_NAMES[6] == "Generic"
        assert LABEL_NAMES[9] == "Worms"