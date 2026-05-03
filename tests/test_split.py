"""
test_split.py — Unit tests for src/preprocessing/split.py

Tests focus on the splitting contract:
- Deterministic with a fixed seed.
- Folds sum to the input size (no rows lost or duplicated).
- Stratification preserves class balance to within tolerance.
- The is_attack derivation handles edge cases.

Run from the repository root:

    pytest tests/test_split.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.split import (
    derive_is_attack,
    stratified_split,
    fold_stats,
)


# ---------- Fixtures ---------- #

def make_split_test_df(n_rows: int = 1000, attack_pct: float = 0.21) -> pd.DataFrame:
    """Construct a DataFrame for split testing.

    n_rows total, with attack_pct fraction labelled as attacks.
    Attack labels are spread across two classes; benign rows have label 'BENIGN'.
    The row order has benign and attack interleaved so naive splits without
    stratification would distribute classes unevenly.
    """
    n_attack = int(n_rows * attack_pct)
    n_benign = n_rows - n_attack
    labels = ["BENIGN"] * n_benign + ["DoS Hulk"] * (n_attack // 2) + ["PortScan"] * (n_attack - n_attack // 2)

    # Shuffle so benign and attack are interleaved
    rng = np.random.default_rng(seed=0)
    order = rng.permutation(len(labels))
    labels = [labels[i] for i in order]

    df = pd.DataFrame({
        "Flow Duration":     [1000 + i for i in range(n_rows)],
        "Total Fwd Packet":  [i % 50 + 1 for i in range(n_rows)],
        "Label":             labels,
    })
    return df


# ---------- Tests for derive_is_attack ---------- #

class TestDeriveIsAttack:
    """Binary attack label derivation."""

    def test_benign_maps_to_zero(self):
        df = pd.DataFrame({"Label": ["BENIGN"] * 5})
        result = derive_is_attack(df)
        assert (result == 0).all()

    def test_attack_maps_to_one(self):
        df = pd.DataFrame({"Label": ["DoS Hulk", "PortScan", "Heartbleed"]})
        result = derive_is_attack(df)
        assert (result == 1).all()

    def test_mixed_labels(self):
        df = pd.DataFrame({"Label": ["BENIGN", "DoS Hulk", "BENIGN", "PortScan"]})
        result = derive_is_attack(df)
        assert list(result) == [0, 1, 0, 1]

    def test_attempted_labels_count_as_attack(self):
        """The '-Attempted' suffix labels should count as attack."""
        df = pd.DataFrame({"Label": ["BENIGN", "DoS Hulk - Attempted", "BENIGN"]})
        result = derive_is_attack(df)
        assert list(result) == [0, 1, 0]

    def test_raises_when_label_column_missing(self):
        df = pd.DataFrame({"Flow Duration": [100, 200]})
        with pytest.raises(ValueError, match="Label"):
            derive_is_attack(df)


# ---------- Tests for stratified_split ---------- #

class TestStratifiedSplit:
    """The core splitting contract."""

    def test_folds_sum_to_input_size(self):
        df = make_split_test_df(n_rows=1000)
        folds = stratified_split(df)
        total = sum(len(f) for f in folds.values())
        assert total == len(df)

    def test_no_row_appears_in_multiple_folds(self):
        """Each input row must end up in exactly one fold."""
        df = make_split_test_df(n_rows=1000)
        folds = stratified_split(df)

        # Use the row indices to verify no overlap
        train_idx = set(folds["train"].index)
        val_idx   = set(folds["val"].index)
        test_idx  = set(folds["test"].index)

        assert len(train_idx & val_idx) == 0
        assert len(train_idx & test_idx) == 0
        assert len(val_idx & test_idx) == 0
        assert len(train_idx | val_idx | test_idx) == len(df)

    def test_split_ratio_close_to_70_15_15(self):
        """Folds should be 70%/15%/15% within rounding."""
        df = make_split_test_df(n_rows=10_000)
        folds = stratified_split(df)

        assert 0.69 <= len(folds["train"]) / len(df) <= 0.71
        assert 0.14 <= len(folds["val"])   / len(df) <= 0.16
        assert 0.14 <= len(folds["test"])  / len(df) <= 0.16

    def test_stratification_preserves_attack_ratio(self):
        """Each fold should have approximately the same attack ratio as input."""
        df = make_split_test_df(n_rows=10_000, attack_pct=0.21)
        folds = stratified_split(df)

        input_attack_pct = derive_is_attack(df).mean()
        for fold_name, fold_df in folds.items():
            fold_attack_pct = derive_is_attack(fold_df).mean()
            # Within 0.5 percentage points
            assert abs(fold_attack_pct - input_attack_pct) < 0.005, (
                f"{fold_name} attack pct {fold_attack_pct:.4f} "
                f"differs from input {input_attack_pct:.4f}"
            )

    def test_deterministic_with_fixed_seed(self):
        """Same input, same seed -> same split."""
        df = make_split_test_df(n_rows=1000)
        folds1 = stratified_split(df, random_seed=42)
        folds2 = stratified_split(df, random_seed=42)

        for key in ("train", "val", "test"):
            pd.testing.assert_frame_equal(
                folds1[key].sort_index(),
                folds2[key].sort_index(),
            )

    def test_different_seeds_produce_different_splits(self):
        """Sanity: two different seeds should give non-identical splits."""
        df = make_split_test_df(n_rows=1000)
        folds_a = stratified_split(df, random_seed=1)
        folds_b = stratified_split(df, random_seed=2)

        # Train sets should differ
        train_a_idx = set(folds_a["train"].index)
        train_b_idx = set(folds_b["train"].index)
        assert train_a_idx != train_b_idx

    def test_label_column_preserved_in_all_folds(self):
        df = make_split_test_df(n_rows=1000)
        folds = stratified_split(df)
        for fold_df in folds.values():
            assert "Label" in fold_df.columns


# ---------- Tests for fold_stats ---------- #

class TestFoldStats:
    """The reporting / manifest helper."""

    def test_fold_stats_arithmetic(self):
        df = make_split_test_df(n_rows=1000, attack_pct=0.21)
        stat = fold_stats("train", df)
        assert stat.fold == "train"
        assert stat.n_rows == 1000
        assert stat.n_benign + stat.n_attack == 1000
        # pct values should sum to 100 (within rounding)
        assert abs(stat.pct_benign + stat.pct_attack - 100.0) < 0.01

    def test_fold_stats_zero_attack_ok(self):
        df = pd.DataFrame({"Label": ["BENIGN"] * 100})
        stat = fold_stats("train", df)
        assert stat.n_attack == 0
        assert stat.pct_attack == 0.0
        assert stat.pct_benign == 100.0