"""
test_random_forest.py — Unit tests for src/models/random_forest.py and
src/evaluation/metrics.py

Tests the contract:
- Feature selection drops the right columns.
- Binary metrics arithmetic is correct on hand-computed examples.
- Per-class metrics handle BENIGN specially.
- Inference latency measurement returns sensible numbers.

Does NOT test sklearn's RandomForestClassifier — that's their contract.
I test my wrapping logic, not their fitting logic.

Run from the repository root:

    pytest tests/test_random_forest.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    BinaryMetrics,
    compute_binary_metrics,
    compute_per_class_metrics,
)
from src.models.random_forest import (
    DROP_FOR_FEATURES,
    HYPERPARAMS,
    select_features,
)


# ---------- Fixtures ---------- #

def make_split_like_df(n_rows: int = 100, attack_pct: float = 0.21) -> pd.DataFrame:
    """Construct a DataFrame mimicking the structure of train.parquet:
    71 features + identifier columns + Label + is_attack.
    """
    n_attack = int(n_rows * attack_pct)
    n_benign = n_rows - n_attack
    labels = ["BENIGN"] * n_benign + ["DoS Hulk"] * (n_attack // 2) + ["PortScan"] * (n_attack - n_attack // 2)
    rng = np.random.default_rng(seed=0)
    order = rng.permutation(len(labels))
    labels = [labels[i] for i in order]

    df = pd.DataFrame({
        # Identifier columns
        "Flow ID":     [f"flow{i}" for i in range(n_rows)],
        "Src IP":      ["10.0.0.1"] * n_rows,
        "Dst IP":      ["10.0.0.2"] * n_rows,
        "Timestamp":   [f"03/07/2017 13:0{i % 9}:00 PM" for i in range(n_rows)],
        "Src Port":    [12345] * n_rows,
        "Dst Port":    [80] * n_rows,
        "Protocol":    [6] * n_rows,
        # Two illustrative numeric features (not actually 71 — shape doesn't matter for these tests)
        "Flow Duration":    [1000 + i for i in range(n_rows)],
        "Total Fwd Packet": [i % 50 + 1 for i in range(n_rows)],
        # Labels
        "Label":     labels,
        "is_attack": [0 if lbl == "BENIGN" else 1 for lbl in labels],
    })
    return df


# ---------- Tests for select_features ---------- #

class TestSelectFeatures:
    """Feature selection drops the right columns and returns three structured outputs."""

    def test_drops_identifier_and_routing_columns(self):
        df = make_split_like_df(n_rows=100)
        X, y_binary, y_multi = select_features(df)
        for col in DROP_FOR_FEATURES:
            assert col not in X.columns, f"{col} should have been dropped from features"

    def test_returns_three_objects(self):
        df = make_split_like_df(n_rows=100)
        result = select_features(df)
        assert len(result) == 3

    def test_y_binary_matches_is_attack(self):
        df = make_split_like_df(n_rows=100)
        _, y_binary, _ = select_features(df)
        assert y_binary.dtype.kind in ("i", "u")  # integer
        assert set(y_binary.unique()).issubset({0, 1})
        np.testing.assert_array_equal(y_binary.values, df["is_attack"].values)

    def test_y_multi_matches_label(self):
        df = make_split_like_df(n_rows=100)
        _, _, y_multi = select_features(df)
        assert y_multi.dtype == object
        np.testing.assert_array_equal(y_multi.values, df["Label"].values)

    def test_X_keeps_real_features(self):
        df = make_split_like_df(n_rows=100)
        X, _, _ = select_features(df)
        assert "Flow Duration"     in X.columns
        assert "Total Fwd Packet"  in X.columns

    def test_raises_when_is_attack_missing(self):
        df = make_split_like_df(n_rows=100).drop(columns=["is_attack"])
        with pytest.raises(ValueError, match="is_attack"):
            select_features(df)

    def test_raises_when_label_missing(self):
        df = make_split_like_df(n_rows=100).drop(columns=["Label"])
        with pytest.raises(ValueError, match="Label"):
            select_features(df)


# ---------- Tests for HYPERPARAMS ---------- #

class TestHyperparameters:
    """Locked-in methodology decisions: ADR 0004."""

    def test_n_estimators_is_200(self):
        assert HYPERPARAMS["n_estimators"] == 200

    def test_max_depth_is_20(self):
        assert HYPERPARAMS["max_depth"] == 20

    def test_class_weight_is_balanced(self):
        assert HYPERPARAMS["class_weight"] == "balanced"

    def test_random_state_is_42(self):
        assert HYPERPARAMS["random_state"] == 42


# ---------- Tests for compute_binary_metrics ---------- #

class TestBinaryMetrics:
    """Hand-computed metrics on tiny examples."""

    def test_perfect_classifier(self):
        y_true  = np.array([0, 0, 1, 1])
        y_pred  = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.9, 0.8])
        m = compute_binary_metrics(y_true, y_pred, y_proba)
        assert m.accuracy == 1.0
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.fpr == 0.0

    def test_all_predict_benign(self):
        """Classifier that says everything is benign — high accuracy on majority,
        but recall=0 on attack class, FPR=0."""
        y_true  = np.array([0, 0, 0, 1])
        y_pred  = np.array([0, 0, 0, 0])
        y_proba = np.array([0.1, 0.1, 0.1, 0.1])
        m = compute_binary_metrics(y_true, y_pred, y_proba)
        assert m.accuracy == 0.75
        assert m.recall == 0.0
        assert m.fpr == 0.0

    def test_fpr_calculation(self):
        """4 benign, 2 attack. 1 false positive (benign predicted as attack).
        Expected FPR = 1/4 = 0.25."""
        y_true  = np.array([0, 0, 0, 0, 1, 1])
        y_pred  = np.array([0, 1, 0, 0, 1, 1])
        y_proba = np.array([0.1, 0.6, 0.2, 0.1, 0.9, 0.8])
        m = compute_binary_metrics(y_true, y_pred, y_proba)
        assert m.fpr == 0.25

    def test_class_count_fields(self):
        y_true  = np.array([0, 0, 0, 1, 1])
        y_pred  = np.array([0, 0, 0, 1, 1])
        y_proba = np.array([0.1, 0.1, 0.1, 0.9, 0.9])
        m = compute_binary_metrics(y_true, y_pred, y_proba)
        assert m.n_test_samples == 5
        assert m.n_test_attack == 2
        assert m.n_test_benign == 3

    def test_inference_times_aggregation(self):
        y_true  = np.array([0, 1])
        y_pred  = np.array([0, 1])
        y_proba = np.array([0.1, 0.9])
        times_ms = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m = compute_binary_metrics(y_true, y_pred, y_proba, inference_times_ms=times_ms)
        assert m.inference_ms_median == 3.0   # median of [1,2,3,4,5]
        assert m.inference_ms_p95 == pytest.approx(4.8, abs=0.01)

    def test_no_inference_times_returns_nan(self):
        y_true  = np.array([0, 1])
        y_pred  = np.array([0, 1])
        y_proba = np.array([0.1, 0.9])
        m = compute_binary_metrics(y_true, y_pred, y_proba)
        assert np.isnan(m.inference_ms_median)
        assert np.isnan(m.inference_ms_p95)


# ---------- Tests for compute_per_class_metrics ---------- #

class TestPerClassMetrics:
    """Per-class breakdown logic — BENIGN is special-cased."""

    def test_benign_recall_is_correct_when_predicted_zero(self):
        y_true_multi  = np.array(["BENIGN", "BENIGN", "BENIGN", "DoS Hulk"])
        y_pred_binary = np.array([0, 0, 1, 1])  # 1 benign falsely flagged as attack
        result = compute_per_class_metrics(y_true_multi, y_pred_binary, ["BENIGN", "DoS Hulk"])
        assert result["BENIGN"]["n"] == 3
        assert result["BENIGN"]["n_correct"] == 2  # two were correctly labelled benign
        assert result["BENIGN"]["recall"] == pytest.approx(2 / 3, abs=0.001)

    def test_attack_recall_is_correct_when_predicted_one(self):
        y_true_multi  = np.array(["BENIGN", "DoS Hulk", "DoS Hulk", "DoS Hulk"])
        y_pred_binary = np.array([0, 1, 1, 0])  # 2 of 3 DoS correctly caught
        result = compute_per_class_metrics(y_true_multi, y_pred_binary, ["BENIGN", "DoS Hulk"])
        assert result["DoS Hulk"]["n"] == 3
        assert result["DoS Hulk"]["n_correct"] == 2
        assert result["DoS Hulk"]["n_missed"] == 1
        assert result["DoS Hulk"]["recall"] == pytest.approx(2 / 3, abs=0.001)

    def test_class_with_zero_samples_omitted(self):
        y_true_multi  = np.array(["BENIGN", "DoS Hulk"])
        y_pred_binary = np.array([0, 1])
        result = compute_per_class_metrics(
            y_true_multi, y_pred_binary,
            ["BENIGN", "DoS Hulk", "Heartbleed"]   # Heartbleed has 0 samples
        )
        assert "Heartbleed" not in result

    def test_micro_class_with_one_sample(self):
        """Micro-class scenario: only 1 Heartbleed flow in test set."""
        y_true_multi  = np.array(["BENIGN", "BENIGN", "Heartbleed"])
        y_pred_binary = np.array([0, 0, 1])
        result = compute_per_class_metrics(y_true_multi, y_pred_binary, ["BENIGN", "Heartbleed"])
        assert result["Heartbleed"]["n"] == 1
        assert result["Heartbleed"]["recall"] == 1.0