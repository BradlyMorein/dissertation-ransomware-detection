"""
test_sequences.py — Unit tests for src/preprocessing/sequences.py

Tests the contract:
- Sources with < N flows are dropped.
- Sequences are non-overlapping.
- Sequences are correctly labelled by majority vote.
- Trailing partial windows are discarded.
- Output preserves sequence_id and sequence_position correctly.
- Sort is by timestamp ascending within source.
- Determinism: same input produces same output.

Run from the repository root:

    pytest tests/test_sequences.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.sequences import (
    LABEL_RULE_THRESHOLD,
    SEQUENCE_LENGTH,
    build_sequences,
)


# ---------- Fixtures ---------- #

def make_flow_df(
    source_specs: list[tuple[str, int, int]],
    base_timestamp_str: str = "01/01/2017 09:00:00 AM",
) -> pd.DataFrame:
    """Build a small flow DataFrame for sequence testing.

    source_specs: list of (src_ip, n_flows, n_attacks) tuples.
    Within each source, n_attacks come first, then n_benigns.
    Timestamps are 1 second apart, sequential per source then offset by source.
    """
    rows = []
    base = pd.Timestamp(base_timestamp_str)

    for src_idx, (src_ip, n_flows, n_attacks) in enumerate(source_specs):
        for i in range(n_flows):
            is_attack = 1 if i < n_attacks else 0
            label = "DoS Hulk" if is_attack else "BENIGN"
            ts = base + pd.Timedelta(seconds=i + src_idx * 100)
            rows.append({
                "Src IP": src_ip,
                "Timestamp": ts.strftime("%d/%m/%Y %I:%M:%S %p"),
                "Flow Duration": 1000 + i,
                "Total Fwd Packet": (i % 50) + 1,
                "Label": label,
                "is_attack": is_attack,
            })
    return pd.DataFrame(rows)


# ---------- Tests for source filtering ---------- #

class TestSourceFiltering:
    """Sources with fewer than SEQUENCE_LENGTH flows are dropped."""

    def test_drops_source_with_too_few_flows(self):
        # One source with 5 flows (insufficient), one with 10 (sufficient)
        df = make_flow_df([
            ("10.0.0.1", 5, 0),    # too few
            ("10.0.0.2", 10, 0),   # sufficient
        ])
        df_seq, stats = build_sequences(df)

        assert stats["n_sources_in"] == 2
        assert stats["n_sources_with_min_flows"] == 1
        assert stats["n_sources_dropped"] == 1
        assert df_seq["Src IP"].nunique() == 1
        assert "10.0.0.1" not in df_seq["Src IP"].values
        assert "10.0.0.2" in df_seq["Src IP"].values

    def test_keeps_all_sources_with_enough_flows(self):
        df = make_flow_df([
            ("10.0.0.1", 10, 0),
            ("10.0.0.2", 20, 0),
        ])
        df_seq, stats = build_sequences(df)
        assert stats["n_sources_dropped"] == 0
        assert df_seq["Src IP"].nunique() == 2


# ---------- Tests for sequence construction ---------- #

class TestSequenceConstruction:
    """Sequences are non-overlapping and complete."""

    def test_non_overlapping_windows(self):
        # 30 flows from one source -> 3 non-overlapping sequences of 10
        df = make_flow_df([("10.0.0.1", 30, 0)])
        df_seq, stats = build_sequences(df)

        assert stats["n_sequences_out"] == 3
        assert stats["n_flows_used"] == 30
        assert stats["n_flows_dropped"] == 0
        # Three distinct sequence IDs
        assert df_seq["sequence_id"].nunique() == 3

    def test_trailing_partial_window_dropped(self):
        # 25 flows -> 2 sequences (20 flows used) + 5 trailing dropped
        df = make_flow_df([("10.0.0.1", 25, 0)])
        df_seq, stats = build_sequences(df)
        assert stats["n_sequences_out"] == 2
        assert stats["n_flows_used"] == 20
        assert stats["n_flows_dropped"] == 5

    def test_sequence_position_is_zero_to_nine(self):
        df = make_flow_df([("10.0.0.1", 10, 0)])
        df_seq, _ = build_sequences(df)
        positions = sorted(df_seq[df_seq["sequence_id"] == 0]["sequence_position"].tolist())
        assert positions == list(range(SEQUENCE_LENGTH))

    def test_each_sequence_has_exactly_n_rows(self):
        df = make_flow_df([("10.0.0.1", 30, 0)])
        df_seq, _ = build_sequences(df)
        sizes = df_seq.groupby("sequence_id").size()
        assert (sizes == SEQUENCE_LENGTH).all()

    def test_sequences_unique_per_source(self):
        # Each source gets distinct sequence_ids
        df = make_flow_df([
            ("10.0.0.1", 10, 0),
            ("10.0.0.2", 10, 0),
        ])
        df_seq, _ = build_sequences(df)
        assert df_seq["sequence_id"].nunique() == 2


# ---------- Tests for label rule ---------- #

class TestLabelRule:
    """Majority vote: >= 6 of 10 attacks = attack sequence."""

    def test_all_benign_gives_benign_sequence(self):
        df = make_flow_df([("10.0.0.1", 10, 0)])  # 0 attacks
        df_seq, _ = build_sequences(df)
        labels = df_seq.drop_duplicates("sequence_id")["sequence_label"].tolist()
        assert labels == [0]

    def test_all_attack_gives_attack_sequence(self):
        df = make_flow_df([("10.0.0.1", 10, 10)])  # all attacks
        df_seq, _ = build_sequences(df)
        labels = df_seq.drop_duplicates("sequence_id")["sequence_label"].tolist()
        assert labels == [1]

    def test_threshold_exactly_majority_gives_attack(self):
        df = make_flow_df([("10.0.0.1", 10, LABEL_RULE_THRESHOLD)])  # 6 attacks
        df_seq, _ = build_sequences(df)
        labels = df_seq.drop_duplicates("sequence_id")["sequence_label"].tolist()
        assert labels == [1]

    def test_below_threshold_gives_benign(self):
        df = make_flow_df([("10.0.0.1", 10, LABEL_RULE_THRESHOLD - 1)])  # 5 attacks
        df_seq, _ = build_sequences(df)
        labels = df_seq.drop_duplicates("sequence_id")["sequence_label"].tolist()
        assert labels == [0]


# ---------- Tests for sort order ---------- #

class TestSortOrder:
    """Sequences are built from flows sorted by timestamp ascending."""

    def test_timestamps_ascending_within_sequence(self):
        df = make_flow_df([("10.0.0.1", 10, 0)])
        # Shuffle the input order to verify sort happens
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        df_seq, _ = build_sequences(df)

        # Within sequence_id == 0, parsed timestamps should be ascending
        seq_0 = df_seq[df_seq["sequence_id"] == 0].sort_values("sequence_position")
        timestamps = pd.to_datetime(seq_0["Timestamp"], format="%d/%m/%Y %I:%M:%S %p")
        assert (timestamps.values == sorted(timestamps.values)).all()


# ---------- Tests for error handling ---------- #

class TestErrorHandling:
    """Required columns checked; bad timestamps caught."""

    def test_raises_on_missing_src_ip(self):
        df = pd.DataFrame({
            "Timestamp": ["01/01/2017 09:00:00 AM"] * 10,
            "is_attack": [0] * 10,
        })
        with pytest.raises(ValueError, match="Src IP"):
            build_sequences(df)

    def test_raises_on_missing_timestamp(self):
        df = pd.DataFrame({
            "Src IP": ["10.0.0.1"] * 10,
            "is_attack": [0] * 10,
        })
        with pytest.raises(ValueError, match="Timestamp"):
            build_sequences(df)

    def test_raises_on_missing_is_attack(self):
        df = pd.DataFrame({
            "Src IP": ["10.0.0.1"] * 10,
            "Timestamp": ["01/01/2017 09:00:00 AM"] * 10,
        })
        with pytest.raises(ValueError, match="is_attack"):
            build_sequences(df)

    def test_raises_on_unparseable_timestamp(self):
        df = pd.DataFrame({
            "Src IP": ["10.0.0.1"] * 10,
            "Timestamp": ["not a date"] * 10,
            "is_attack": [0] * 10,
        })
        with pytest.raises(ValueError, match="failed to parse"):
            build_sequences(df)


# ---------- Tests for determinism ---------- #

class TestDeterminism:
    """Same input -> same output, regardless of input order."""

    def test_input_shuffle_same_output(self):
        df = make_flow_df([("10.0.0.1", 20, 0)])

        df_seq_1, _ = build_sequences(df)
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df_seq_2, _ = build_sequences(df_shuffled)

        # Compare the sequence labels — should be identical
        labels_1 = df_seq_1.drop_duplicates("sequence_id")["sequence_label"].tolist()
        labels_2 = df_seq_2.drop_duplicates("sequence_id")["sequence_label"].tolist()
        assert labels_1 == labels_2