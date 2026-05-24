# ADR 0003 — Train/validation/test split strategy: stratified random per-flow

**Status:** Accepted
**Date:** 3 May 2026
**Decided by:** Bradly Morein (supervisor confirmed verbally on 2 May 2026: Comfortable to support decisions made)

## Context

After cleaning, the primary dataset (Engelen-corrected CIC-IDS2017) consists
of 2,098,400 flows across five days of capture (3–7 July 2017), with a
21.11% binary attack rate. To train and evaluate the three model
architectures (Random Forest, Autoencoder, LSTM) we need a deterministic
partitioning of these flows into training, validation, and test sets.

Two methodology decisions are required:

1. **Splitting strategy** — random vs chronological.
2. **Granularity** — per-flow vs per-day.

The choices interact. Random per-flow means individual flows are randomly
assigned to folds regardless of when they occurred. Chronological per-day
means whole days are assigned to folds (e.g., Mon–Wed train, Thu val, Fri
test). Random per-day would distribute days randomly without temporal
structure; chronological per-flow is incoherent (a flow's timestamp would
need to determine its fold, which is just per-day disguised).

## Decision

**Stratified random per-flow split** with a 70 / 15 / 15 ratio for
train / validation / test, stratified on binary `is_attack` (BENIGN vs
any-attack), with random seed 42 for reproducibility.

The original multi-class `Label` column is preserved in all output Parquets
to support per-class evaluation in Chapter 5.

## Rationale

1. **Comparability with the published literature.** Stratified random
   per-flow is the dominant approach in NIDS benchmark studies on CIC-IDS2017
   (Sharafaldin et al. 2018; Engelen et al. 2021; the broader CIC-IDS2017
   literature). Using the same approach makes our results directly
   comparable rather than introducing methodological variation that
   confounds comparison.

2. **Class balance preservation.** Stratification on `is_attack` ensures
   each fold has the same 21.11% attack rate as the full dataset (preserved
   to four decimal places in the actual run). Without stratification, the
   minority attack class could end up unevenly distributed.

3. **Schedule fit.** Stratified random is a single well-tested function
   call (`sklearn.model_selection.train_test_split` with `stratify=`).
   The chronological alternative would require additional methodology
   description and result reporting, eating into time better spent on the
   modelling experiments and writing.

4. **Stratification on binary, not multi-class.** Eight micro-classes
   (Heartbleed: 11, SSH-Patator-Attempted: 8, etc.) are too small for
   sklearn's stratification to handle reliably. Binary stratification
   covers the headline RQ1 framing (BENIGN vs attack); multi-class
   per-class analysis in Chapter 6 will use the preserved `Label` column
   and acknowledge wide confidence intervals on micro-class metrics.

## Consequences

- **Repository structure.** `src/preprocessing/split.py` produces
  `train.parquet` / `val.parquet` / `test.parquet` in
  `data/processed/cicids2017-engelen/`. Models load from these Parquets
  rather than from the per-day cleaned files.

- **Methodology chapter §3.5.1** must explicitly state and justify the split
  approach, including the rationale for stratified random over chronological
  and for binary over multi-class stratification.

- **Limitations section.** The chronological-deployment case is not directly
  evaluated. This will be acknowledged in §7.4 (Limitations) and named as
  potential future work.

- **Reproducibility.** Random seed 42 is fixed in `split.py` and recorded
  in `_split_manifest.json`. Anyone re-running the pipeline gets the same
  split byte-for-byte.

## Alternatives considered

**Alternative 1 — Chronological per-day split.**
Train Mon–Wed, validate Thursday, test Friday (or similar). Reflects
deployment reality (model trained on past data, evaluated on future).
Rejected because:
- Wednesday's DoS attacks would be entirely in training; their detection
  performance could not be measured on the test fold.
- Friday's attacks (PortScan, DDoS, Bot) differ from earlier-week attacks,
  giving an unrepresentative test set.
- Breaks comparability with the published literature, which is
  predominantly random-split-based.

**Alternative 2 — Stratified random as headline + chronological as
robustness check (both).**
Most rigorous methodologically. Rejected on schedule grounds; the
chronological evaluation would add ~1 working day to implementation plus
additional results and discussion in Chapter 6. Acknowledged as future
work in §7.4 (Limitations).

**Alternative 3 — Multi-class stratification with manual handling for
micro-classes.**
Custom code that distributes Heartbleed (11 flows) and similar micro-classes
deliberately across folds, then stratifies the rest on binary. Rejected as
overkill: the headline RQ1 is binary classification; per-class analysis in
Chapter 6 can rely on the preserved `Label` column without requiring
multi-class stratification at split time.

## References

- Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward
  generating a new intrusion detection dataset and intrusion traffic
  characterization. In *4th International Conference on Information
  Systems Security and Privacy (ICISSP)* (pp. 108–116).
- Engelen, G., Rimmer, V., & Joosen, W. (2021). Troubleshooting an
  Intrusion Detection Dataset: the CICIDS2017 Case Study. In *2021 IEEE
  Security and Privacy Workshops (SPW)* (pp. 7–12).

## Implementation

- Module: `src/preprocessing/split.py`
- Tests: `tests/test_split.py` (14 tests)
- Manifest: `data/processed/cicids2017-engelen/_split_manifest.json`
- Key parameters: `RANDOM_SEED=42`, `TRAIN_FRACTION=0.70`,
  `VAL_FRACTION_OF_REMAINDER=0.50`.