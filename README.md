# MSc Dissertation — Network-Based Ransomware Anomaly Detection

A comparative evaluation of Random Forest, Autoencoder, and LSTM models for
network-based ransomware anomaly detection, using the Engelen-cleaned
CIC-IDS2017 dataset (primary) and CIC-UNSW-NB15 (secondary, for cross-dataset
validation). See `docs/decisions/` for ADRs documenting the dataset choices.

**Author:** Bradly Morein (UP2088701)
**Programme:** MSc Cybersecurity and Data Forensics, University of Portsmouth
**Supervisor:** Dr Linda Yang

---

## Research questions

- **RQ1** — How do Random Forest, Autoencoder, and LSTM compare in detecting
  ransomware-indicative network anomalies on the Engelen-cleaned CIC-IDS2017
  dataset?
- **RQ2** — Which architecture achieves the best balance between detection rate
  and false positive rate for practical deployment?
- **RQ3** — To what extent do trained models generalise to unseen distributions
  when validated on CIC-UNSW-NB15 (cross-dataset evaluation)?
- **RQ4** — What is the inference latency of each model, and does it support
  real-time network traffic inspection?

---

## Repository structure

```
.
├── data/              # raw, interim, processed — gitignored, see data/README.md
├── docs/decisions/    # architecture decision records (ADRs)
├── notebooks/         # exploratory analysis and the Colab LSTM notebook
├── src/               # pipeline code
├── experiments/       # run manifests and saved predictions
├── reports/           # figures and tables used in the dissertation
├── tests/             # unit tests for pipeline code
├── verify_results.py        # reconciles reported metrics against manifests
├── verify_partitions.py     # checks the train/val/test folds are disjoint
├── verify_predictions.py    # recomputes every metric from saved predictions
├── environment.yml          # human-readable environment spec
├── environment.lock.yml     # conda lockfile (exact versions)
├── requirements.lock.txt    # pip lockfile
└── ENVIRONMENT.md           # host machine and setup documentation
```

---

## What can be reproduced from a clone

The datasets and the trained model binaries are excluded from version control
because of their size. What is included determines what a fresh clone can do
without downloading anything further.

**Runs from a clone, no datasets required**

| Command | What it does |
|---|---|
| `python -m pytest tests/ -v` | 97 unit tests over the preprocessing, split, sequence construction and model configuration code. Uses synthetic fixtures. |
| `python3 verify_results.py` | Recomputes every reported metric from the confusion matrix counts in the experiment manifests and compares them against the recorded values. |
| `python3 verify_predictions.py` | Recounts each confusion matrix from the saved per-flow predictions, recomputes all metrics, and recalculates ROC-AUC and PR-AUC from the stored probabilities without calling scikit-learn. |

**Requires the datasets**

| Command | What it does |
|---|---|
| `python3 verify_partitions.py` | Hashes every row and confirms the three folds are disjoint. Reads the partition files. |
| The pipeline below | Regenerates everything from the raw captures. |

---

## Reproducing this work in full

### 1. Environment

```bash
# Miniforge: https://github.com/conda-forge/miniforge
git clone git@github.com:BradlyMorein/dissertation-ransomware-detection.git
cd dissertation-ransomware-detection
conda env create -f environment.lock.yml
conda activate ransomware-detection
```

See `ENVIRONMENT.md` for the host specification the results were produced on.

### 2. Datasets

Acquisition instructions are in `data/README.md`. Two sources are needed:

- **CIC-IDS2017**, with the corrections published by Engelen et al. (2021)
- **CIC-UNSW-NB15**, the CICFlowMeter-derived release

### 3. Pipeline

Run from the repository root, in this order:

```bash
python -m src.preprocessing.csv_to_parquet      # raw CSV -> interim parquet
python -m src.preprocessing.clean               # clean CIC-IDS2017
python -m src.preprocessing.clean_unsw          # clean CIC-UNSW-NB15
python -m src.preprocessing.split               # stratified train/val/test split
python -m src.preprocessing.sequences           # build 10-flow sequences for the LSTM
```

Each stage writes a manifest to its output directory recording row counts,
drops and timings.

### 4. Models

```bash
python -m src.models.random_forest              # Random Forest, local
python -m src.models.autoencoder                # Autoencoder, local
```

The LSTM was trained on Google Colab. Open
`notebooks/lstm_train_colab.ipynb` and run it there, then place the resulting
`model.pt` and `predictions.npz` in `experiments/lstm/`.

### 5. Evaluation

```bash
python -m src.models.evaluate_cross_dataset     # apply RF and AE to CIC-UNSW-NB15
python -m src.evaluation.ransomware_subset      # per-class recall on the ransomware-indicative subset
```

### 6. Verification

```bash
python -m pytest tests/ -v
python3 verify_results.py
python3 verify_partitions.py
python3 verify_predictions.py
```

The three verification scripts append to `verification_report.md` in the order
listed. `test_results.txt` holds the captured test suite output.

---

## Determinism

The split, the Random Forest and the Autoencoder all use seed 42 and reproduce
exactly between runs.

The Random Forest's *predicted probabilities* are the one exception. It
averages across 200 trees using eight parallel threads, and floating-point
addition is not associative, so the summation order varies between runs and the
probabilities differ in their final bits. The resulting class predictions are
identical, and the threshold-dependent metrics are therefore exactly
reproducible, but ROC-AUC and PR-AUC — which are computed from the
probabilities — vary in the seventh decimal place. The Autoencoder is a single
forward pass with no parallel reduction and is bit-for-bit deterministic.

---

## Licence

Academic work prepared for assessment at the University of Portsmouth.
Not licensed for external use or redistribution.
