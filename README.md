# MSc Dissertation — Network-Based Ransomware Anomaly Detection

A comparative evaluation of Random Forest, Autoencoder, and LSTM models for
network-based ransomware anomaly detection, using the CSE-CIC-IDS2018 and
UNSW-NB15 datasets.

**Author:** Bradly Morein (UP2088701)
**Programme:** MSc Cybersecurity and Data Forensics, University of Portsmouth
**Supervisor:** Dr Linda Yang
**Submission deadline:** 8 July 2026 (working target 6 July)

---

## Research questions

- **RQ1** — How do Random Forest, Autoencoder, and LSTM compare in detecting
  ransomware-indicative network anomalies on CSE-CIC-IDS2018?
- **RQ2** — Which architecture achieves the best balance between detection rate
  and false positive rate for practical deployment?
- **RQ3** — To what extent do trained models generalise to unseen distributions
  when validated on UNSW-NB15?
- **RQ4** — What is the inference latency of each model, and does it support
  real-time network traffic inspection?

---

## Repository structure

.
├── data/              # raw, interim, processed — gitignored except READMEs
├── notebooks/         # exploratory analysis (Jupyter)
├── src/               # production code
├── experiments/       # configuration-driven runs, logs, checkpoints
├── reports/           # figures and tables destined for the dissertation
├── tests/             # unit tests for pipeline code
├── scripts/           # one-off scripts (data download, setup)
├── environment.yml            # human-readable environment spec
├── environment.lock.yml       # conda lockfile (exact versions)
├── requirements.lock.txt      # pip lockfile
└── ENVIRONMENT.md             # host machine and setup documentation

---

## Reproducing this work

1. Install Miniforge: https://github.com/conda-forge/miniforge
2. Clone this repo: `git clone <repo-url>`
3. Create the environment: `conda env create -f environment.lock.yml`
4. Activate: `conda activate ransomware-detection`
5. Acquire the datasets: see `data/README.md`
6. Run the pipeline: (documented in later weeks as it's built)

---

## Status

- [x] Week 1 — Environment and repository setup
- [ ] Week 1 — Dataset acquisition, EDA
- [ ] Week 2 — Preprocessing pipeline
- [ ] Week 3 — Random Forest training and tuning
- [ ] Week 4 — Autoencoder training and tuning
- [ ] Week 5 — LSTM training and tuning
- [ ] Week 6 — Evaluation pipeline, statistical tests
- [ ] Week 7 — Cross-dataset validation
- [ ] Weeks 8–10 — Writing
- [ ] Week 11 — Polish and submission

---

## Licence

This is academic work prepared for assessment at the University of Portsmouth.
Not licensed for external use or redistribution during the assessment period.
