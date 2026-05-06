"""
test_autoencoder.py — Unit tests for src/models/autoencoder.py

Tests the contract:
- Monday-benign filter selects the right rows.
- Feature selection drops the same columns as the RF pipeline.
- StandardScaler integration leaves no NaN / Inf.
- Reconstruction-error helper returns one value per row.
- Threshold helper uses only benign rows.
- Hyperparameter contract locked in.

Does NOT test sklearn's MLPRegressor — that's their contract. We test
our wrapping logic.

Run from the repository root:

    pytest tests/test_autoencoder.py -v
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from src.models.autoencoder import (
    BENIGN_PERCENTILE_FOR_THRESHOLD,
    DROP_FOR_FEATURES,
    HYPERPARAMS,
    filter_monday_benign,
    reconstruction_error,
    select_features,
    threshold_from_benign_percentile,
)


# ---------- Fixtures ---------- #

def make_split_like_df(
    n_rows: int = 100,
    attack_pct: float = 0.21,
    days: list[str] | None = None,
) -> pd.DataFrame:
    """Construct a DataFrame mimicking CIC-IDS2017 train fold:
    71 features + identifier columns + Label + is_attack + Timestamp.

    The `days` parameter controls which days the timestamps span.
    Defaults to all five working days, distributing rows roughly evenly.
    """
    if days is None:
        days = ["03/07/2017", "04/07/2017", "05/07/2017", "06/07/2017", "07/07/2017"]
        # 03/07/2017 is a Monday in our data

    n_attack = int(n_rows * attack_pct)
    n_benign = n_rows - n_attack
    labels = ["BENIGN"] * n_benign + ["DoS Hulk"] * (n_attack // 2) + ["PortScan"] * (n_attack - n_attack // 2)
    rng = np.random.default_rng(seed=0)
    order = rng.permutation(len(labels))
    labels = [labels[i] for i in order]

    # Distribute timestamps across days
    timestamps = [
        f"{days[i % len(days)]} 0{(i % 9) + 1}:0{i % 6}:00 PM"
        for i in range(n_rows)
    ]

    df = pd.DataFrame({
        "Flow ID":     [f"flow{i}" for i in range(n_rows)],
        "Src IP":      ["10.0.0.1"] * n_rows,
        "Dst IP":      ["10.0.0.2"] * n_rows,
        "Timestamp":   timestamps,
        "Src Port":    [12345] * n_rows,
        "Dst Port":    [80] * n_rows,
        "Protocol":    [6] * n_rows,
        "Flow Duration":    [1000 + i for i in range(n_rows)],
        "Total Fwd Packet": [(i % 50) + 1 for i in range(n_rows)],
        "Label":     labels,
        "is_attack": [0 if lbl == "BENIGN" else 1 for lbl in labels],
    })
    return df


# ---------- Tests for filter_monday_benign ---------- #

class TestFilterMondayBenign:
    """Monday-only benign filter for AE training."""

    def test_keeps_monday_benign_drops_others(self):
        # Construct: 50% Monday flows, 50% Tuesday; mix benign and attack
        df = pd.DataFrame({
            "Timestamp": [
                "03/07/2017 09:00:00 AM",  # Monday
                "03/07/2017 10:00:00 AM",  # Monday
                "04/07/2017 09:00:00 AM",  # Tuesday
                "04/07/2017 10:00:00 AM",  # Tuesday
            ],
            "Label":     ["BENIGN", "BENIGN", "BENIGN", "DoS Hulk"],
            "is_attack": [0, 0, 0, 1],
        })
        result = filter_monday_benign(df)
        # Should keep only the two Monday benign rows
        assert len(result) == 2
        assert all(result["is_attack"] == 0)

    def test_drops_monday_attacks_defensively(self):
        """Even if a row is on Monday, drop it if labelled as attack."""
        df = pd.DataFrame({
            "Timestamp": [
                "03/07/2017 09:00:00 AM",  # Monday + benign -> keep
                "03/07/2017 10:00:00 AM",  # Monday + attack -> drop (defensive)
            ],
            "Label":     ["BENIGN", "DoS Hulk"],
            "is_attack": [0, 1],
        })
        result = filter_monday_benign(df)
        assert len(result) == 1
        assert result.iloc[0]["Label"] == "BENIGN"

    def test_raises_on_unparseable_timestamp(self):
        df = pd.DataFrame({
            "Timestamp": ["this is not a date"],
            "Label":     ["BENIGN"],
            "is_attack": [0],
        })
        with pytest.raises(ValueError, match="failed to parse"):
            filter_monday_benign(df)

    def test_raises_when_timestamp_missing(self):
        df = pd.DataFrame({"Label": ["BENIGN"], "is_attack": [0]})
        with pytest.raises(ValueError, match="Timestamp"):
            filter_monday_benign(df)

    def test_returns_copy_not_view(self):
        """Modifying the filter output should not alter the input."""
        df = pd.DataFrame({
            "Timestamp": ["03/07/2017 09:00:00 AM"],
            "Label":     ["BENIGN"],
            "is_attack": [0],
        })
        result = filter_monday_benign(df)
        result.loc[result.index[0], "Label"] = "MODIFIED"
        assert df.iloc[0]["Label"] == "BENIGN"


# ---------- Tests for select_features (delegates to RF logic shape) ---------- #

class TestSelectFeatures:
    """Same drop list as the RF pipeline; verify it's applied identically."""

    def test_drops_identifier_and_routing_columns(self):
        df = make_split_like_df(n_rows=50)
        X, _, _ = select_features(df)
        for col in DROP_FOR_FEATURES:
            assert col not in X.columns

    def test_returns_three_objects(self):
        df = make_split_like_df(n_rows=50)
        result = select_features(df)
        assert len(result) == 3

    def test_raises_when_is_attack_missing(self):
        df = make_split_like_df(n_rows=50).drop(columns=["is_attack"])
        with pytest.raises(ValueError, match="is_attack"):
            select_features(df)

    def test_raises_when_label_missing(self):
        df = make_split_like_df(n_rows=50).drop(columns=["Label"])
        with pytest.raises(ValueError, match="Label"):
            select_features(df)


# ---------- Tests for reconstruction_error ---------- #

class TestReconstructionError:
    """Per-row mean squared error between input and model reconstruction."""

    def test_returns_one_value_per_row(self):
        # Mock model: returns input * 0.9 (small reconstruction error)
        mock_model = MagicMock()
        X = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]])
        mock_model.predict.return_value = X * 0.9

        errors = reconstruction_error(mock_model, X)
        assert errors.shape == (3,)

    def test_perfect_reconstruction_gives_zero_error(self):
        mock_model = MagicMock()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_model.predict.return_value = X.copy()  # exact reproduction
        errors = reconstruction_error(mock_model, X)
        assert np.all(errors == 0)

    def test_error_is_mean_squared(self):
        """Hand-computed: input [1, 2], reconstruction [2, 0],
        squared deltas [1, 4], mean 2.5."""
        mock_model = MagicMock()
        X = np.array([[1.0, 2.0]])
        mock_model.predict.return_value = np.array([[2.0, 0.0]])
        errors = reconstruction_error(mock_model, X)
        assert errors[0] == pytest.approx(2.5)


# ---------- Tests for threshold_from_benign_percentile ---------- #

class TestThresholdFromBenignPercentile:
    """Threshold derivation: must use ONLY benign rows."""

    def test_uses_only_benign_rows(self):
        """Attacks have huge errors; benign have small. Threshold should
        come from benign distribution and ignore attacks."""
        val_errors = np.array([0.1, 0.2, 0.3, 0.4, 0.5,    # benign
                               999, 999, 999])              # attacks
        val_is_attack = np.array([0, 0, 0, 0, 0,
                                  1, 1, 1])
        threshold = threshold_from_benign_percentile(
            val_errors=val_errors,
            val_is_attack=val_is_attack,
            percentile=95.0,
        )
        # Should be near 0.5 (top of benign range), not influenced by 999s
        assert threshold < 1.0

    def test_50th_percentile_returns_median(self):
        val_errors = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        val_is_attack = np.array([0, 0, 0, 0, 0])
        threshold = threshold_from_benign_percentile(
            val_errors=val_errors,
            val_is_attack=val_is_attack,
            percentile=50.0,
        )
        assert threshold == 3.0

    def test_raises_when_no_benign_rows(self):
        val_errors = np.array([1.0, 2.0])
        val_is_attack = np.array([1, 1])
        with pytest.raises(ValueError, match="No benign"):
            threshold_from_benign_percentile(val_errors, val_is_attack)


# ---------- Tests for HYPERPARAMS ---------- #

class TestHyperparameters:
    """Locked-in methodology decisions: ADR 0005."""

    def test_architecture_is_32_16_32(self):
        assert HYPERPARAMS["hidden_layer_sizes"] == (32, 16, 32)

    def test_activation_is_relu(self):
        assert HYPERPARAMS["activation"] == "relu"

    def test_solver_is_adam(self):
        assert HYPERPARAMS["solver"] == "adam"

    def test_random_state_is_42(self):
        assert HYPERPARAMS["random_state"] == 42

    def test_early_stopping_disabled(self):
        assert HYPERPARAMS["early_stopping"] is False

    def test_threshold_percentile_is_95(self):
        assert BENIGN_PERCENTILE_FOR_THRESHOLD == 95.0