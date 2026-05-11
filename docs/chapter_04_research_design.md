# Chapter 4 — Research Design

This chapter describes the concrete design and training of the three
machine learning models used in this dissertation: a Random Forest
classifier, an Autoencoder, and an LSTM. It also documents the
preprocessing pipeline shared across all three, the cross-dataset
evaluation methodology, and the computational environment supporting
reproducibility. Where this chapter touches on the high-level
methodological rationale, that rationale is set out in Chapter 3;
here the focus is on implementation.

The chapter is structured around the three models in the order they
were developed. Section 4.1 documents the shared preprocessing
pipeline that produces aligned 71-feature datasets from
CIC-IDS2017 and CIC-UNSW-NB15. Sections 4.2 through 4.4 cover the
Random Forest, Autoencoder, and LSTM in turn. Section 4.5 documents
the cross-dataset evaluation methodology and the methodological
constraint that excluded the LSTM from the cross-dataset analysis.
Section 4.6 documents the computational environment and reproducibility
infrastructure.

## 4.1 Preprocessing Pipeline

A shared preprocessing pipeline was applied to both the primary dataset
(CIC-IDS2017) and the secondary dataset (CIC-UNSW-NB15) to produce
feature-aligned inputs suitable for all three models. The pipeline
consisted of three stages: row-level cleaning, column-level pruning,
and dataset splitting.

Row-level cleaning removed records that could not be processed by
downstream models. Three categories of row were dropped: rows with a
Protocol value of zero (a CICFlowMeter artefact indicating an
unparseable flow record); rows containing any NaN value across numeric
features; and rows containing positive or negative infinity in any
numeric feature. Applied to CIC-IDS2017, this filter retained
2,098,400 of 2,100,769 rows (99.89%). Applied to CIC-UNSW-NB15, the
filter retained all 447,915 rows (100%).

Column-level pruning removed five features identified during
exploratory data analysis as near-constant across both datasets, and
therefore non-informative for binary classification. These were:
`Bwd PSH Flags`, `Bwd URG Flags`, `Fwd Header Length.1`, `Fwd Bytes/Bulk
Avg`, and `Bwd Bytes/Bulk Avg`. The pruning was applied uniformly to
both datasets. After cleaning and pruning, the two datasets shared 71
feature columns with identical names and types, supporting direct
cross-dataset evaluation in Section 4.5.

Five additional features (`Flow Duration`, `Flow Packets/s`,
`Flow IAT Mean`, `Flow IAT Max`, `Flow IAT Min`) were identified during
the autoencoder training as containing negative values in a small
proportion of rows. These are CICFlowMeter packet-ordering artefacts:
when packets within a flow arrive out of order, the timing-derived
features can take mathematically nonsensical negative values
(Engelen et al., 2021). For the autoencoder, which uses a `log1p`
transformation incompatible with negative inputs, these features were
clipped to zero before scaling.

Splitting was applied only to CIC-IDS2017, which served as the
training dataset. A stratified random split was applied at the flow
level using a 70/15/15 ratio for training, validation, and test folds,
stratified on the binary attack indicator to preserve the 21.11%
attack rate across all three folds (random seed 42). CIC-UNSW-NB15
was held back entirely for the cross-dataset evaluation in Section
4.5 and was not split.

The full preprocessing pipeline was implemented in `src/preprocessing/`
as four modules: `clean.py` (row and column filtering), `clean_unsw.py`
(UNSW-specific schema mapping), `split.py` (stratified random
splitting), and `sequences.py` (sequence construction for the LSTM,
described in Section 4.4). Each module is accompanied by unit tests
covering the canonical edge cases; the test suite of 97 tests
runs in under two seconds.

## 4.2 Random Forest

The Random Forest (Breiman, 2001) was the first model trained, serving
both as a strong supervised baseline and as a reference for the
subsequent unsupervised and temporal models. Implementation used
`sklearn.ensemble.RandomForestClassifier` (Pedregosa et al., 2011).

The model was configured with 200 trees and a maximum tree depth of
20. The 200-tree count was chosen as a balance between predictive
stability and training time: ensembles below 100 trees showed
appreciable variance across random seeds during exploratory runs,
while ensembles above 300 trees produced diminishing improvements at
substantial cost. The depth limit of 20 prevented individual trees
from overfitting to specific flow-level patterns in the training
fold; deeper unconstrained trees produced near-perfect training
accuracy but degraded validation performance.

To address the 79% benign / 21% attack class imbalance in the
training fold, the `class_weight='balanced'` argument was used. This
re-weights samples inversely to class frequency during tree
construction, effectively up-weighting attack flows by approximately
3.75 times relative to benign flows. The Random Forest's split
criteria therefore consider attack patterns with weight comparable to
the majority class, despite the four-to-one imbalance in raw counts.

Feature selection followed the alignment principle established in
Section 4.1: all 71 features common to CIC-IDS2017 and CIC-UNSW-NB15
were used. The flow identifier columns (`Flow ID`, `Src IP`, `Dst IP`,
`Timestamp`), routing fields (`Src Port`, `Dst Port`, `Protocol`),
and label columns were excluded from the feature set, consistent with
the cross-dataset evaluation requirement that the model use only
features available in both datasets.

Training was performed on the 1,468,879-row training fold of
CIC-IDS2017 with a random seed of 42. Training completed in 234
seconds on a Dell XPS Intel Core i7 11th-generation laptop using
eight parallel workers. The resulting model contained 200 decision
trees with a total serialised size of 9.4 MB; the model is preserved
under `experiments/rf/model.joblib` along with a metrics manifest
recording the hyperparameters, training time, validation metrics, and
per-class breakdowns. All design decisions and their rationale are
documented in ADR 0004.

## 4.3 Autoencoder

The Autoencoder served as the unsupervised contrast to the supervised
Random Forest and LSTM. Its design hypothesis was that a model trained
exclusively on benign traffic to reconstruct its inputs would assign
high reconstruction error to traffic patterns it had not encountered
during training. Attack flows, by virtue of being structurally
different from benign flows, would reconstruct poorly and could be
flagged on that basis alone.

The architecture was a symmetric multi-layer perceptron with four
hidden layers compressing 71 input features to a 16-dimensional
bottleneck and back: 71 → 32 → 16 → 32 → 71. Implementation used
`sklearn.neural_network.MLPRegressor` with `relu` activations, configured
to use the input as both feature and target. The 16-dimensional
bottleneck represents a compression ratio of approximately 22%, within
the conventional range for autoencoder anomaly detection (Sakurada &
Yairi, 2014).

Training data were filtered to Monday-only benign flows from the
CIC-IDS2017 training fold. Monday is the only day in CIC-IDS2017
containing no recorded attacks (Sharafaldin et al., 2018), so this
filter eliminates the risk of training-data contamination from
mislabelled or borderline-classified attack flows on Tuesday through
Friday. The filter produced 259,858 rows for training, approximately
18% of the full training fold.

Feature preprocessing required substantial iteration before producing
stable training behaviour. A direct application of `StandardScaler`
to the network flow features produced chaotic validation scores
during training (R² values in the negative tens of millions), driven
by extreme outliers in heavy-tailed features such as flow byte counts
and packet rates. Substituting `RobustScaler` instead produced
training losses in the billions, caused by features with near-zero
inter-quartile range being amplified by the scaler. The eventual
solution was a two-stage `log1p` followed by `StandardScaler`
pipeline. The `log1p` transformation compresses heavy-tailed
distributions toward Gaussianity while preserving zero values; the
standard scaler then centres and unit-scales the result. Five features
required clipping to non-negative values before the `log1p` step, as
documented in Section 4.1.

The Adam optimiser was used at a learning rate of 0.001 for a maximum
of 200 iterations, with early stopping disabled to allow the model to
fit fully. Training converged at iteration 48 with a final training
loss of 0.0040, completing in 45.7 seconds.

The anomaly detection threshold was set as the 95th percentile of
reconstruction errors on the validation fold's benign rows. This rule
deliberately uses no attack labels, preserving the unsupervised
character of the model. The resulting threshold of 0.097 produces a
false positive rate of approximately 5% on the test fold by
construction. The trade-off implicit in this choice — between
preserving the unsupervised premise and maximising attack recall — is
analysed in Chapter 5 (Section 5.3). The model state, scaler, and
threshold are bundled under `experiments/ae/model.joblib`; the full
design rationale is documented in ADR 0005.

## 4.4 Long Short-Term Memory Network

The LSTM (Hochreiter & Schmidhuber, 1997) was the third model trained
and the only deep learning architecture in the comparative study. Its
design hypothesis was that attack flows often produce characteristic
multi-flow signatures over time — repeated authentication attempts in
brute force attacks, sustained packet bursts in DoS attacks, or
sequential command-and-control beacons in botnet traffic — and that a
recurrent network operating on flow sequences could detect these
temporal patterns more effectively than per-flow classifiers.

The LSTM's input is a sequence of consecutive flow records from the
same source host, rather than a single flow. This required a separate
preprocessing step beyond that of Section 4.1: the construction of
flow sequences from the per-flow Parquet files. Implementation lives
in `src/preprocessing/sequences.py` and was applied to each of the
three CIC-IDS2017 folds independently.

### Sequence construction

Sequences were built by grouping flows by source IP, sorting within
each group by timestamp in ascending order, and applying a non-
overlapping sliding window of length ten. Each sequence therefore
captured ten consecutive flows from a single source — the natural unit
for "what does this host do over time" and the standard convention in
the NIDS-LSTM literature (Yin et al., 2017). Sources with fewer than
ten flows in their fold were excluded, as they cannot form a complete
sequence. The non-overlapping window choice means each flow appears
in exactly one sequence; this eliminates within-fold temporal
correlation between sequences while preserving temporal ordering
inside each sequence. Overlapping windows would have multiplied
training data approximately tenfold but at the cost of strongly
correlated sequences.

Sequence-level labels were assigned by majority vote: a sequence was
labelled as attack if six or more of its ten constituent flows were
labelled attacks at the per-flow level. This rule preserved the
per-flow class balance to within 0.07 percentage points across all
three folds (21.06% attack at the sequence level versus 21.11% at the
per-flow level), confirming that attack flows arrive in temporally
coherent runs rather than scattered randomly across each source's
activity. Three sequence Parquet files resulted: 146,876 training
sequences, 31,468 validation sequences, and 31,470 test sequences.

### Architecture

The model was a single-layer unidirectional LSTM with a hidden size of
128 and a final linear layer producing a single binary logit. The
final hidden state of the LSTM was passed through a dropout layer (p
= 0.2) and then through the linear head; the sigmoid was applied
implicitly via `BCEWithLogitsLoss` during training, and explicitly at
inference time. A unidirectional rather than bidirectional design was
chosen deliberately to reflect real-time deployment realities, in
which future flows are not yet observed. The resulting model contained
103,041 trainable parameters.

### Training

PyTorch (Paszke et al., 2019) was used as the implementation
framework, with training performed on a Google Colab Pro NVIDIA A100
GPU because the development laptop lacks a CUDA-compatible GPU and
CPU-only training on 146,876 sequences would require approximately
eight hours. The class imbalance was handled with `BCEWithLogitsLoss`
configured with `pos_weight = 3.749`, the negative-to-positive ratio
in the training fold; this is the deep-learning equivalent of the
Random Forest's `class_weight='balanced'` setting.

Training used the Adam optimiser at learning rate 0.001, a batch size
of 256, and a budget of 20 epochs. The minimum validation loss
occurred at epoch 18, indicating that the 20-epoch budget was
appropriate. Total training time was 45.4 seconds; validation accuracy
after epoch 1 was already 0.9987, refining to 0.9994 by the final
epoch. The trained model state dict, architecture metadata, and
scaler statistics are bundled under `experiments/lstm/model.pt`; the
full design rationale is documented in ADR 0006.

### A note on dataset structure

The sequence-construction decision interacted with a property of
CIC-IDS2017 that became apparent only during preprocessing. The
dataset contains 34 source IP addresses, of which only 14 have ten or
more flows. Of these 14, one source (172.16.0.1) generates 87% of all
attack flows; the remaining 13 sources are network background traffic.
This concentration prevented a group-by-source robustness evaluation
within CIC-IDS2017, because partitioning sources into separate
training and test groups would place either all attacks in one fold
or none. This constraint is documented in ADR 0006 and is discussed
in Chapter 5 (Section 5.4) in the context of cross-dataset evaluation.

## 4.5 Cross-Dataset Evaluation Methodology

The within-CIC-IDS2017 evaluation answers the research question
directly, but with an important caveat: the same dataset's flows are
used for training, validation, and test. To test whether the models
learned generalisable attack patterns rather than dataset-specific
signatures, the Random Forest and Autoencoder were applied unchanged
to the held-out CIC-UNSW-NB15 dataset.

CIC-UNSW-NB15 was selected as the cross-dataset evaluation target
because it shares the CICFlowMeter feature schema with CIC-IDS2017
(76 of 80 columns identical; 71 features after the column pruning of
Section 4.1) yet differs in every other dimension. It was generated
from a different network testbed (the UNSW Cyber Range), it features
a different attack taxonomy (Exploits, Fuzzers, Reconnaissance,
Backdoors, Shellcode, Worms — none of which overlap with CIC-IDS2017's
DoS, Brute Force, Bot, or Port Scan classes), and it captures a
different background-traffic distribution.

The evaluation procedure applied the trained models to the entire
447,915-row CIC-UNSW-NB15 dataset without any retraining, fine-tuning,
or threshold recalibration. The same 71 features were used; the same
five negative-value clipping rules were applied; and the same scaler
fitted on CIC-IDS2017 training data was reused for the Autoencoder.
The Random Forest used its default decision threshold of 0.5; the
Autoencoder used its 95th-percentile threshold of 0.097, fitted on
CIC-IDS2017 validation data. This represents the strictest possible
generalisation test: the models receive no information about UNSW
during evaluation.

The LSTM was excluded from the cross-dataset evaluation. UNSW is an
anonymised dataset, lacking the `Src IP`, `Dst IP`, `Timestamp`,
`Flow ID`, source-and-destination port, and protocol fields. Without
`Src IP` and `Timestamp`, the sequence construction of Section 4.4
cannot be applied: sequences cannot be grouped by source or sorted by
time. Three alternative approaches were considered: building sequences
from row order alone (rejected because the resulting evaluation would
conflate dataset shift with sequence-construction shift, telling us
nothing useful about generalisation); training a fresh LSTM on UNSW
(rejected because it defeats the purpose of testing CIC-IDS2017-
trained model generalisation); and applying the LSTM to single flows
through a degenerate length-one sequence (rejected because the model
was not trained on length-one sequences and the resulting evaluation
would be uninterpretable). The exclusion is documented as a
methodological constraint of the dataset combination in ADR 0007 and
discussed alongside the within-CIC-IDS2017 dataset-structure
constraint in Chapter 5.

Binary classification metrics were used exclusively in the cross-
dataset evaluation. Per-class breakdowns were not produced because
UNSW's attack taxonomy does not overlap with CIC-IDS2017's, making
per-class comparisons unintelligible: there is no meaningful
counterpart to "Bot" or "FTP-Patator" in UNSW, and no counterpart to
"Exploits" or "Worms" in CIC-IDS2017. The binary view answers the
generalisation question directly: did the model trained to detect
attacks in CIC-IDS2017 still detect attacks in UNSW?

The evaluation script (`src/models/evaluate_cross_dataset.py`)
produces a manifest at `experiments/cross_dataset/_metrics_manifest.json`
and prediction files for each model. The detailed methodology rationale
including the LSTM exclusion is documented in ADR 0007. The
cross-dataset results themselves are presented in Chapter 5 (Section
5.4).

## 4.6 Computational Environment

The project's computational infrastructure was deliberately partitioned
between local CPU-based work and remote GPU-based work, with all
artefacts committed to a single version-controlled repository for
reproducibility.

Local development used a Dell XPS 13 laptop with an Intel Core i7
11th-generation CPU (eight logical cores), 16 GB RAM, and Fedora
Workstation 43. All preprocessing, the Random Forest training, the
Autoencoder training, and all results notebooks ran locally. The
Python environment was managed via Miniforge with conda, locked to
Python 3.11.15 and the following primary dependencies: pandas 2.x for
data handling, scikit-learn 1.x for the Random Forest and Autoencoder,
matplotlib and seaborn for visualisation, and pyarrow for Parquet I/O.
The complete environment specification is preserved in
`environment.lock.yml` at the repository root.

GPU-based work — the LSTM training only — used Google Colab Pro with
an NVIDIA A100 GPU (40 GB memory), accessed via a personal access
token-authenticated clone of the repository. PyTorch 2.10.0 with CUDA
12.8 was used for the LSTM implementation. The decision to perform
LSTM training on Colab rather than locally was operational: the
development laptop lacks a CUDA-compatible GPU, and CPU-only training
on 146,876 sequences was estimated to require approximately eight
hours, against 45 seconds on the A100. The trained model state was
downloaded to local storage immediately after training and is preserved
under `experiments/lstm/model.pt`.

Reproducibility was supported by four practices applied throughout
the project. First, a uniform random seed of 42 was used across all
randomness-sensitive operations (train/val/test splitting, Random
Forest training, Autoencoder weight initialisation, LSTM weight
initialisation, and PyTorch's CUDA random state). Second, every
experimental output was accompanied by a JSON manifest recording
hyperparameters, runtime, dataset SHA-256 hashes, and headline metrics.
Third, a unit test suite of 97 tests across the preprocessing and
modelling code was maintained, running in under two seconds.
Fourth, seven Architectural Decision Records (ADRs) documented the
major methodological choices and their alternatives.

All code, data manifests, trained model bundles, and figures are
available at `github.com/BradlyMorein/dissertation-ransomware-detection`.