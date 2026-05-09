# ADR 0006 — LSTM methodology

**Status:** Accepted
**Date:** 9 May 2026
**Decided by:** Bradly Morein (supervisor confirmed verbally on 2 May 2026: happy to support whatever decisions made)

## Context

The LSTM is the third of three model architectures evaluated in this
dissertation, providing a temporal/sequence-based contrast to the
supervised Random Forest baseline (ADR 0004) and the unsupervised
Autoencoder (ADR 0005). The LSTM learns multi-flow temporal patterns
within sequences of consecutive flows from the same source IP, with
the hypothesis that attacks producing characteristic multi-flow
signatures (volume-based DoS, brute force, Bot command-and-control)
should be more reliably detected by sequence-level modelling than by
per-flow classification.

This ADR documents seven methodology decisions: framework, sequence
construction, label rule, architecture, output head, training
configuration, and dataset-structure constraints.

## Decisions

### 1. Framework

**PyTorch 2.10.0+cu128 on Google Colab Pro (NVIDIA A100 GPU).**

PyTorch was chosen over TensorFlow/Keras for cleaner LSTM API and
better integration with Colab Pro's GPU runtime. The decision is
operational rather than methodologically loaded — both frameworks
support the same architecture; PyTorch's syntax and Colab integration
are slightly more ergonomic. The model is trained on Colab Pro's
A100 GPU rather than locally because the development laptop (Dell XPS,
Intel Core i7 11th-gen) lacks an NVIDIA GPU and CPU-only LSTM training
on 146,876 sequences would take 8+ hours; A100 reduces this to
45 seconds.

### 2. Sequence construction

**Group flows by Src IP, sort by Timestamp, slide non-overlapping
windows of length 10.**

This is the standard NIDS-LSTM literature convention. Each sequence
captures 10 consecutive flows from one source — the natural unit for
"what is this host doing over time." Sources with fewer than 10 flows
in their fold are dropped (cannot form a complete sequence).

The non-overlapping windowing means each flow appears in exactly one
sequence. This eliminates within-fold temporal correlation between
sequences while preserving the temporal ordering within each sequence.
Overlapping windows would multiply the training data ~10× but would
introduce strong correlation between adjacent sequences, weakening
the inferential validity of the train/val/test split.

**Sequence length N=10** balances context (enough flows to capture an
attack pattern) against data volume (enough sequences to train on).
Common values in the literature are 10, 16, 20, 32; 10 is the
cleanest choice given the dataset's flow volume.

### 3. Label rule

**Majority vote: a sequence is labelled "attack" if 6 or more of its
10 constituent flows are attack flows.**

This is the cleanest defensible rule. Alternatives considered:

- **Any-attack** (>= 1 attack flow → attack sequence): more permissive,
  but conflates pure-attack sequences with mostly-benign sequences
  containing one attack flow. The latter case is operationally less
  meaningful (a single anomalous flow is a different detection problem
  from a sustained attack).

- **Last-flow** (label by the final flow's class): operationally
  realistic — predicting whether the next flow will be malicious given
  the history — but produces a heavily asymmetric problem (most
  sequences are "predict benign next given benign history") and
  doesn't match the per-flow classification framing of RF and AE.

The majority-vote rule preserves the dataset's per-flow class balance
to within 0.07 percentage points across folds (per-flow attack rate
21.11% vs. per-sequence attack rate 21.04-21.06%), indicating that
attacks arrive in coherent runs rather than scattered randomly.

### 4. Architecture

**1-layer unidirectional LSTM, hidden_size=128, dropout=0.2.**

Decisions within this:

- **Bidirectional vs unidirectional**: unidirectional. Reflects
  real-time deployment scenarios where future flows are not yet
  observed. Bidirectional models would see both past and future within
  each sequence and produce slightly stronger metrics, but at the cost
  of operational realism.

- **Number of layers**: 1. Sufficient for sequences of length 10 in a
  homogeneous feature space. Multi-layer LSTMs are more useful for
  longer sequences (50+) or hierarchical patterns; for our case, the
  extra parameters would offer diminishing returns.

- **Hidden size**: 128. Standard middle-ground choice. Lower (32, 64)
  would underfit the 71-dimensional input; higher (256, 512) would
  overfit on this dataset size.

- **Dropout**: 0.2. PyTorch's `nn.LSTM` only applies dropout *between*
  LSTM layers, not on the output. With `n_layers=1`, the LSTM dropout
  argument is silently ignored; we apply dropout via a standalone
  `nn.Dropout(0.2)` layer between the LSTM output and the linear head.

### 5. Output head

**Final hidden state of the last LSTM layer → dropout → linear → BCE
with logits loss.**

The standard PyTorch idiom for sequence-to-binary classification.
Alternatives considered:

- **Mean-pool over all hidden states**: more position-invariant but
  loses the temporal-progression signal that's the point of LSTM.

- **Attention-pooling**: more sophisticated but adds parameters and
  code complexity for limited expected benefit on length-10 sequences.

The output is a raw logit; sigmoid is applied at inference time to
produce probabilities. `nn.BCEWithLogitsLoss` combines sigmoid + BCE
in a numerically stable way during training.

### 6. Training configuration

| Parameter | Value | Rationale |
|---|---|---|
| Optimizer | Adam | Standard for LSTM training |
| Learning rate | 0.001 | PyTorch default; never needed adjustment |
| Batch size | 256 | A100 memory not a constraint; 256 is fast |
| Epochs | 20 | Empirically converged; min val loss at epoch 18 |
| Loss function | BCEWithLogitsLoss(pos_weight=3.749) | Balances 21/79 class imbalance |
| Random seed | 42 | Matches RF (ADR 0004) and AE (ADR 0005) |

The class weighting (`pos_weight = n_negative / n_positive ≈ 3.75`)
mirrors the `class_weight='balanced'` setting used in RF. Each attack
sequence contributes 3.75× the gradient of a benign sequence,
compensating for the minority-class imbalance.

### 7. Dataset-structure constraint: group-by-source split not feasible

**Decision: do not implement a group-by-source robustness check on
CIC-IDS2017.**

The original plan (Day 11) was to evaluate the LSTM under both per-flow
splits (matching RF/AE) and group-by-source splits (where whole source
IPs are held out from training). Empirical investigation revealed:

- The dataset has only **14 active source IPs** (those with ≥10 flows).
- One source — **172.16.0.1** — generates 440,693 of 503,200 attack
  flows (87%). The other 7 "active attackers" each generate <700
  attack flows (likely noise-level misclassifications, not deliberate
  attackers in the experimental design).

A clean group-by-source split is therefore **not feasible**:
- If 172.16.0.1 is in the train fold, the test fold contains
  essentially zero attacks — pathological.
- If 172.16.0.1 is in the test fold, the train fold contains
  essentially zero attacks — also pathological.

This is a structural property of CIC-IDS2017's experimental setup
(controlled testbed with one designated attacker) and cannot be
worked around within this single dataset.

**Mitigation**: The cross-dataset evaluation on CIC-UNSW-NB15 in Week 7
provides the genuine generalisation test. UNSW has different source
IPs entirely, so applying RF, AE, and LSTM trained on CIC-IDS2017 to
UNSW directly tests whether the models have learned attack patterns
that generalise, or whether they have learned dataset-specific source
signatures.

## Consequences

### Headline test-set metrics (n=31,470 sequences)

| Metric | Value |
|---|---:|
| Accuracy | 0.9994 |
| Precision (attack) | 0.9979 |
| Recall (attack) | 0.9992 |
| F1 (attack) | 0.9986 |
| ROC-AUC | 0.99997 |
| PR-AUC | 0.9999 |
| FPR | 0.0006 |

Of 6,620 attack sequences, 5 are missed (4 PortScan, 1 Bot-Attempted).
Of 24,850 benign sequences, 14 are flagged as attacks.

### Per-class results

14 of 16 attack classes are detected at 100% recall. The two
exceptions:

- **PortScan**: 4 of 2,394 missed (recall 0.998).
- **Bot - Attempted**: 1 of 5 missed (recall 0.800; micro-class).

Critically, the volume-based DoS classes that the AE failed at (Hulk:
AE 0.024 → LSTM 1.000; DDoS: AE 0.026 → LSTM 1.000) are detected
perfectly. The brute-force classes (FTP-Patator and SSH-Patator: AE
0.000 → LSTM 1.000) are also detected perfectly. This confirms the
core hypothesis: **temporal modelling captures the multi-flow
signatures that per-flow autoencoder reconstruction misses**.

### Training behaviour

- 45.4 seconds total wall-clock time on NVIDIA A100.
- Validation accuracy 0.9987 after epoch 1; 0.9994 after epoch 20.
- Minimum validation loss at epoch 18 (loss 0.0036) — close to the end
  of the 20-epoch budget but not at the boundary, indicating
  appropriate training duration.
- No overfitting: train and val loss converge to similar values
  (0.0039 vs 0.0046).

### Comparative implications for RQ1

Three-way comparison on CIC-IDS2017:

| Metric | RF | AE (95th pct) | AE (86.5th F1-opt) | LSTM |
|---|---:|---:|---:|---:|
| Accuracy | 0.9917 | 0.8318 | — | 0.9994 |
| Precision | 0.9626 | 0.6749 | 0.6386 | 0.9979 |
| Recall | 0.9995 | 0.3922 | 0.9106 | 0.9992 |
| F1 | 0.9807 | 0.4961 | 0.7507 | 0.9986 |
| ROC-AUC | 0.9998 | 0.9204 | 0.9204 | 0.99997 |
| FPR | 0.0104 | 0.0506 | 0.1379 | 0.0006 |

The story for Chapter 6: **supervision and information availability
dominate architecture choice on this dataset.** RF and LSTM (both
supervised, both saw attack labels in training) achieve near-ceiling
performance; AE (unsupervised, never saw attack labels) achieves much
lower recall under the unsupervised threshold rule. The LSTM's
sequence-level prediction has more information per decision (10 flows
vs. 1) but the marginal benefit over RF is small because RF is already
near ceiling.

### Generalisation caveat

Both supervised models (RF and LSTM) may have benefited from the
dataset's source-IP concentration: 87% of attack flows come from one
source (172.16.0.1), and the per-flow stratified split distributes the
same sources across train/val/test folds. The cross-dataset UNSW
evaluation in Week 7 will test whether the learned patterns generalise
to data with completely different sources.

### Operational characteristics

- Bulk inference: 0.337 seconds for 31,470 sequences (10.7
  microseconds per sequence) on A100 GPU.
- Model size: 103,041 trainable parameters (~408KB serialised).
- Memory footprint: trivial; training peaked at <1GB GPU memory.
- Real-time NIDS deployment feasible: a network with 100K flows/min
  produces ~10K sequences/min at length 10, classifiable in 0.1
  seconds on GPU or ~10 seconds on CPU.

## Implementation

- Module: `notebooks/lstm_train_colab.ipynb` (executed on Colab Pro,
  not locally).
- Sequence preprocessing: `src/preprocessing/sequences.py` (ADR
  references the methodology already documented in `_sequences_manifest.json`).
- Tests: `tests/test_sequences.py` (17 tests, all passing).
- Manifest: `experiments/lstm/_metrics_manifest.json`.
- Trained model: `experiments/lstm/model.pt` (committed; 408KB, small
  enough for git).
- Predictions: `experiments/lstm/predictions.npz` (committed).

The trained model and predictions are committed to git because they
fit comfortably under git's preferred file-size limits (both <1MB),
unlike the larger sequence Parquets which remain gitignored.

## Alternatives considered

**Alternative 1 — Bidirectional LSTM.** Would see both past and future
context within each sequence and likely improve metrics by ~0.1-0.2
percentage points. Rejected on the principle that real-time NIDS
deployment cannot see future flows.

**Alternative 2 — Multi-layer LSTM (n_layers=2).** Would add ~100K
more parameters and ~50% more training time. Rejected — single layer
already achieves 99.94% accuracy; deeper architecture would offer
diminishing returns.

**Alternative 3 — Attention pooling over all hidden states.**
Sophisticated but unnecessary for length-10 sequences where the final
hidden state already captures the full pattern.

**Alternative 4 — GRU instead of LSTM.** Marginally faster, similar
performance. Chosen LSTM because it's the more canonical choice in
the NIDS literature; standardisation aids comparison.

**Alternative 5 — Group-by-source split for robustness.** Considered
and rejected after empirical investigation revealed the single-attacker
structure of CIC-IDS2017 (one source generates 87% of attacks). See
Decision 7 above.

## References

- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.
  *Neural Computation*, 9(8), 1735-1780.
- Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward
  generating a new intrusion detection dataset and intrusion traffic
  characterization. *ICISSP*.
- Engelen, G., Rimmer, V., & Joosen, W. (2021). Troubleshooting an
  Intrusion Detection Dataset: the CICIDS2017 Case Study. *IEEE
  Security and Privacy Workshops*.
- Yin, C., Zhu, Y., Fei, J., & He, X. (2017). A deep learning approach
  for intrusion detection using recurrent neural networks. *IEEE
  Access*, 5, 21954-21961. (Standard reference for NIDS-LSTM
  methodology.)