"""
ransomware_subset.py — Per-class recall on the ransomware-indicative subset.

Reports detection performance restricted to the attack classes that map onto
ransomware kill-chain phases (Methodology chapter §3.3.3). Neither dataset
contains ransomware samples, so the mapped classes stand proxy for the network
behaviours a ransomware campaign produces: brute-force credential access,
command-and-control beaconing, lateral movement, reconnaissance and exploit
delivery.

Reporting decisions:
- Recall only. The subset contains no benign flows, so precision, F1 and FPR
  cannot be computed over it without pairing the subset against a benign
  population that corresponds to no real network. Recall answers the §3.3.3
  claim directly: a model that fails to detect these patterns would also fail
  to detect ransomware.
- Per-class as well as aggregate. PortScan is ~95% of the CIC-IDS2017 subset,
  so the aggregate is dominated by one class and conceals the per-phase
  picture. Both are reported.
- "- Attempted" variants are included as instances of their parent class.
  Attempted brute-force and attempted C2 contact are still the network
  behaviours the mapping targets, and excluding them would discard most of
  the Bot evidence (209 of 303 flows).
- Cross-dataset recall is reported alongside each model's FPR. Both models
  are degenerate on UNSW in opposite directions, so recall alone is not
  interpretable there.

Unit of prediction differs by model. RF and AE predict per flow; the LSTM
predicts per sequence of 10 flows. LSTM counts are therefore roughly a tenth
of the RF/AE counts and are not directly comparable in magnitude.

LSTM class attribution: each attack sequence is assigned the modal non-BENIGN
label among its 10 constituent flows. 97.3% of attack sequences contain a
single attack class, so the attribution is unambiguous for almost all of them;
the homogeneity figures are recorded in the manifest.

Run from the repository root:

    python -m src.evaluation.ransomware_subset

Outputs:
- experiments/ransomware_subset/_metrics_manifest.json
"""

from pathlib import Path
from dataclasses import dataclass, asdict
import json
import sys

import numpy as np
import pandas as pd


# ---------- Configuration ---------- #

REPO_ROOT = Path(__file__).resolve().parents[2]

SPLITS_DIR = REPO_ROOT / "data" / "processed" / "cicids2017-engelen"
UNSW_PATH = REPO_ROOT / "data" / "processed" / "cic-unsw-nb15" / "cic-unsw-nb15.parquet"

TEST_SEQUENCES_PATH = SPLITS_DIR / "test_sequences.parquet"

RF_MANIFEST = REPO_ROOT / "experiments" / "rf" / "_metrics_manifest.json"
AE_MANIFEST = REPO_ROOT / "experiments" / "ae" / "_metrics_manifest.json"
LSTM_PREDICTIONS = REPO_ROOT / "experiments" / "lstm" / "predictions.npz"

CROSS_DIR = REPO_ROOT / "experiments" / "cross_dataset"
CROSS_MANIFEST = CROSS_DIR / "_metrics_manifest.json"

OUTPUT_DIR = REPO_ROOT / "experiments" / "ransomware_subset"
MANIFEST_PATH = OUTPUT_DIR / "_metrics_manifest.json"

# Ransomware-indicative class mapping (Methodology chapter §3.3.3).
# Parent class -> kill-chain phase. "- Attempted" variants are included as
# instances of their parent class.
CICIDS_MAPPING = {
    "PortScan": "reconnaissance",
    "FTP-Patator": "brute-force credential access",
    "SSH-Patator": "brute-force credential access",
    "Bot": "command-and-control beaconing",
    "Infiltration": "lateral movement",
}

UNSW_MAPPING = {
    "Reconnaissance": "internal network scanning",
    "Exploits": "exploitation for initial access",
    "Shellcode": "exploit payload delivery",
    "Backdoor": "persistent unauthorised access",
    "Worms": "self-propagation",
}

ATTEMPTED_SUFFIX = " - Attempted"


# ---------- Data structures ---------- #

@dataclass
class ClassRecall:
    """Recall for a single mapped class."""
    label: str
    phase: str
    n: int
    n_detected: int
    recall: float | None  # None when n == 0


# ---------- Helpers ---------- #

def expand_mapping(mapping: dict[str, str]) -> dict[str, str]:
    """Add the '- Attempted' variant of each mapped class.

    Returns a label -> phase dict covering both the parent class and its
    attempted variant, so that e.g. 'Bot - Attempted' inherits the
    command-and-control phase from 'Bot'.
    """
    expanded: dict[str, str] = {}
    for label, phase in mapping.items():
        expanded[label] = phase
        expanded[label + ATTEMPTED_SUFFIX] = phase
    return expanded


def recall_or_none(n_detected: int, n: int) -> float | None:
    """Recall, or None when the class has no instances."""
    return round(n_detected / n, 4) if n > 0 else None


def summarise(results: list[ClassRecall]) -> dict:
    """Aggregate a list of per-class results, and the same excluding the
    largest class.

    The subset is dominated by one class in both datasets, so the aggregate
    alone is misleading. The 'excluding_largest' figure makes the dominance
    visible.
    """
    present = [r for r in results if r.n > 0]
    total_n = sum(r.n for r in present)
    total_detected = sum(r.n_detected for r in present)

    summary = {
        "n": total_n,
        "n_detected": total_detected,
        "recall": recall_or_none(total_detected, total_n),
    }

    if present:
        largest = max(present, key=lambda r: r.n)
        rest_n = total_n - largest.n
        rest_detected = total_detected - largest.n_detected
        summary["largest_class"] = largest.label
        summary["excluding_largest"] = {
            "n": rest_n,
            "n_detected": rest_detected,
            "recall": recall_or_none(rest_detected, rest_n),
        }

    return summary


def print_table(title: str, results: list[ClassRecall], summary: dict) -> None:
    """Print one model's per-class table."""
    print(f"\n{title}")
    print("-" * len(title))
    for r in results:
        recall = f"{r.recall:.4f}" if r.recall is not None else "   -  "
        print(f"  {r.label:<32} {r.n_detected:>6}/{r.n:<6}  recall={recall}")
    total_recall = summary["recall"]
    total_recall = f"{total_recall:.4f}" if total_recall is not None else "   -  "
    print(f"  {'SUBSET TOTAL':<32} "
          f"{summary['n_detected']:>6}/{summary['n']:<6}  recall={total_recall}")
    if "excluding_largest" in summary:
        excl = summary["excluding_largest"]
        excl_recall = excl["recall"]
        excl_recall = f"{excl_recall:.4f}" if excl_recall is not None else "   -  "
        label = f"EXCLUDING {summary['largest_class']}"
        print(f"  {label:<32} "
              f"{excl['n_detected']:>6}/{excl['n']:<6}  recall={excl_recall}")


# ---------- Per-flow models (RF, AE) on CIC-IDS2017 ---------- #

def recall_from_manifest(manifest_path: Path, mapping: dict[str, str]) -> list[ClassRecall]:
    """Extract per-class recall for the mapped classes from an existing
    model manifest.

    RF and AE already record per-class counts over the full CIC-IDS2017 test
    fold, so the subset is a filter over those counts rather than a fresh
    computation.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    per_class = manifest["test"]["per_class"]

    results: list[ClassRecall] = []
    for label, phase in mapping.items():
        entry = per_class.get(label)
        if entry is None:
            results.append(ClassRecall(label, phase, 0, 0, None))
            continue
        results.append(
            ClassRecall(
                label=label,
                phase=phase,
                n=int(entry["n"]),
                n_detected=int(entry["n_correct"]),
                recall=recall_or_none(int(entry["n_correct"]), int(entry["n"])),
            )
        )
    return results


# ---------- Sequence model (LSTM) on CIC-IDS2017 ---------- #

def lstm_modal_classes() -> tuple[pd.Series, dict]:
    """Assign each attack sequence its modal non-BENIGN label.

    Returns
    -------
    modal : Series indexed by sequence_id, values are class labels
    homogeneity : dict recording how many attack sequences contain 1, 2 or
        more distinct attack classes, justifying the modal attribution
    """
    df = pd.read_parquet(
        TEST_SEQUENCES_PATH,
        columns=["sequence_id", "Label", "sequence_label"],
    )

    attack_rows = df[df["sequence_label"] == 1]
    attack_only = attack_rows[attack_rows["Label"] != "BENIGN"]

    distinct_per_sequence = attack_only.groupby("sequence_id")["Label"].nunique()
    counts = distinct_per_sequence.value_counts().sort_index()
    n_sequences = int(distinct_per_sequence.size)
    n_single = int(counts.get(1, 0))

    homogeneity = {
        "n_attack_sequences": n_sequences,
        "distinct_attack_classes_per_sequence": {
            str(k): int(v) for k, v in counts.items()
        },
        "pct_single_class": round(100 * n_single / n_sequences, 2) if n_sequences else 0.0,
    }

    modal = attack_only.groupby("sequence_id")["Label"].agg(
        lambda s: s.value_counts().idxmax()
    )
    return modal, homogeneity


def lstm_recall(mapping: dict[str, str]) -> tuple[list[ClassRecall], dict]:
    """Per-class recall for the LSTM, at sequence level."""
    modal, homogeneity = lstm_modal_classes()

    predictions = np.load(LSTM_PREDICTIONS)["predictions"]
    frame = pd.DataFrame({"label": modal})
    frame["predicted"] = predictions[frame.index.values]

    results: list[ClassRecall] = []
    for label, phase in mapping.items():
        subset = frame[frame["label"] == label]
        n = int(len(subset))
        detected = int(subset["predicted"].sum())
        results.append(ClassRecall(label, phase, n, detected, recall_or_none(detected, n)))

    return results, homogeneity


# ---------- Cross-dataset models on CIC-UNSW-NB15 ---------- #

def unsw_recall(model: str, mapping: dict[str, str]) -> list[ClassRecall]:
    """Per-class recall on UNSW for one cross-dataset model.

    Predictions are in the same row order as the UNSW Parquet, since
    evaluate_cross_dataset.py reads the file once and predicts over it
    directly.
    """
    labels = pd.read_parquet(UNSW_PATH, columns=["LabelName"])["LabelName"]
    predictions = np.load(CROSS_DIR / f"{model}_predictions.npz")["predictions"]

    if len(labels) != len(predictions):
        raise ValueError(
            f"Row count mismatch for {model}: {len(labels):,} labels vs "
            f"{len(predictions):,} predictions."
        )

    results: list[ClassRecall] = []
    for label, phase in mapping.items():
        mask = (labels == label).values
        n = int(mask.sum())
        detected = int(predictions[mask].sum())
        results.append(ClassRecall(label, phase, n, detected, recall_or_none(detected, n)))

    return results


def unsw_fpr(model: str) -> float:
    """False positive rate on UNSW, read from the cross-dataset manifest.

    Recall on the UNSW subset is not interpretable without this: one model
    detected nothing and the other flagged nearly everything.
    """
    with open(CROSS_MANIFEST) as f:
        manifest = json.load(f)
    return float(manifest[model]["fpr"])


# ---------- CLI entry point ---------- #

def main() -> int:
    required = [
        TEST_SEQUENCES_PATH, UNSW_PATH, RF_MANIFEST, AE_MANIFEST,
        LSTM_PREDICTIONS, CROSS_MANIFEST,
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("ERROR: required inputs not found:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cicids_mapping = expand_mapping(CICIDS_MAPPING)
    unsw_mapping = UNSW_MAPPING  # no "- Attempted" variants in UNSW

    print("Ransomware-indicative subset — per-class recall")
    print(f"Output directory: {OUTPUT_DIR.relative_to(REPO_ROOT)}")

    # ---- Within-dataset: CIC-IDS2017 ---- #
    print("\n" + "=" * 70)
    print("WITHIN-DATASET — CIC-IDS2017 test fold")
    print("=" * 70)

    rf_results = recall_from_manifest(RF_MANIFEST, cicids_mapping)
    rf_summary = summarise(rf_results)
    print_table("Random Forest (per flow)", rf_results, rf_summary)

    ae_results = recall_from_manifest(AE_MANIFEST, cicids_mapping)
    ae_summary = summarise(ae_results)
    print_table("Autoencoder (per flow)", ae_results, ae_summary)

    lstm_results, homogeneity = lstm_recall(cicids_mapping)
    lstm_summary = summarise(lstm_results)
    print_table("LSTM (per sequence)", lstm_results, lstm_summary)
    print(f"\n  Sequence attribution: {homogeneity['pct_single_class']}% of "
          f"{homogeneity['n_attack_sequences']:,} attack sequences contain a "
          f"single attack class.")

    # ---- Cross-dataset: CIC-UNSW-NB15 ---- #
    print("\n" + "=" * 70)
    print("CROSS-DATASET — CIC-UNSW-NB15 (all rows)")
    print("=" * 70)

    cross = {}
    for model, title in [("rf", "Random Forest"), ("ae", "Autoencoder")]:
        results = unsw_recall(model, unsw_mapping)
        summary = summarise(results)
        fpr = unsw_fpr(model)
        print_table(f"{title} (per flow)", results, summary)
        print(f"  {'FALSE POSITIVE RATE':<32} {fpr:>19.5f}")
        cross[model] = {
            "per_class": [asdict(r) for r in results],
            "summary": summary,
            "fpr": fpr,
        }

    # ---- Manifest ---- #
    manifest = {
        "tool": "src/evaluation/ransomware_subset.py",
        "mapping_source": "Methodology chapter §3.3.3",
        "attempted_variants": "included as instances of their parent class",
        "metric": "recall only; the subset contains no benign flows",
        "class_mapping": {
            "cicids2017": cicids_mapping,
            "cic-unsw-nb15": unsw_mapping,
        },
        "within_dataset": {
            "dataset": "CIC-IDS2017 (Engelen-cleaned) test fold",
            "rf": {
                "unit": "flow",
                "per_class": [asdict(r) for r in rf_results],
                "summary": rf_summary,
            },
            "ae": {
                "unit": "flow",
                "per_class": [asdict(r) for r in ae_results],
                "summary": ae_summary,
            },
            "lstm": {
                "unit": "sequence of 10 flows",
                "class_attribution": "modal non-BENIGN label within each sequence",
                "sequence_homogeneity": homogeneity,
                "per_class": [asdict(r) for r in lstm_results],
                "summary": lstm_summary,
            },
        },
        "cross_dataset": {
            "dataset": "CIC-UNSW-NB15 (CIC-augmented), all rows",
            "lstm_excluded_reason": (
                "UNSW is anonymised; lacks Src IP/Timestamp required for "
                "sequence construction"
            ),
            "rf": cross["rf"],
            "ae": cross["ae"],
        },
    }

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written to: {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())