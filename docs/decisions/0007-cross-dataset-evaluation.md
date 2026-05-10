# ADR 0007 — Cross-dataset evaluation methodology

**Status:** Accepted
**Date:** 10 May 2026
**Decided by:** Bradly Morein (supervisor confirmed verbally on 2 May 2026: will support decisions made)

## Context

The within-CIC-IDS2017 evaluation of the three models — RF (recall 0.9995),
AE at chosen threshold (recall 0.3922), LSTM (recall 0.9992) — produced
strong but suspect headline metrics. ADR 0006 §7 documented the
underlying concern: CIC-IDS2017 has only 14 active source IPs with
172.16.0.1 generating 87% of attack flows. The within-dataset metrics
may be measuring "the model recognises 172.16.0.1's traffic" rather
than "the model detects attacks in the abstract."

The genuine generalisation test requires applying the trained models
to a different dataset with different sources, different attack types,
and different network characteristics. CIC-UNSW-NB15 (CIC-augmented)
fulfils this requirement.

## Decisions

### 1. Evaluation dataset: CIC-UNSW-NB15

CIC-augmented version of the UNSW-NB15 dataset (1.5M flows reduced to
447,915 after the same cleaning pipeline applied to CIC-IDS2017).
Different network architecture, different attacker profile, different
attack taxonomy. Selected per ADR 0002.

Attack taxonomy (10 classes, 0=Benign):
- Analysis (385), Backdoor (452), DoS (4,467), Exploits (30,951),
  Fuzzers (29,613), Generic (4,632), Reconnaissance (16,735),
  Shellcode (2,102), Worms (246).

**Total**: 358,332 benign + 89,583 attack = 447,915 flows. Attack
rate 20.00% (very close to CIC-IDS2017's 21.11%, useful for direct
comparability).

No overlap with CIC-IDS2017's attack taxonomy (DoS Hulk, FTP-Patator,
Bot, etc.). The cross-dataset evaluation is therefore a genuine test
of whether models learned generalisable attack patterns.

### 2. Evaluation methodology

Apply RF and AE — both trained on CIC-IDS2017 — unchanged to the
entire UNSW dataset (no train/val/test split on UNSW because no
training is performed on it). All 447,915 flows treated as the
held-out evaluation set.

For RF: load saved `model.joblib`, predict on UNSW features. Use the
default 0.5 decision threshold. Save predictions and probabilities.

For AE: load saved bundle (`model.joblib` containing model + scaler +
threshold). Apply the CIC-IDS2017-fitted scaler to UNSW data. Compute
reconstruction errors. Apply the 95th-percentile threshold (0.097)
fitted on CIC-IDS2017 validation benigns.

Same preprocessing applied as during training:
- 71 features after dropping identifiers and routing fields.
- CICFlowMeter artifact features (Flow Duration, Flow Packets/s,
  Flow IAT Mean/Max/Min) clipped to >= 0.

### 3. LSTM excluded

The LSTM cannot be applied to CIC-UNSW-NB15 because UNSW is anonymised:
the `Src IP`, `Dst IP`, `Timestamp`, `Flow ID`, `Src Port`, `Dst Port`,
and `Protocol` fields are removed for privacy. The LSTM's sequence
construction requires `Src IP` (to group flows by source) and
`Timestamp` (to sort within each source). Without these, sequences
cannot be constructed in a methodologically meaningful way.

**Alternatives considered and rejected:**

- **Treat row-order as sequential.** Build sequences from rows in
  Parquet order, treating the dataset as a single source. This loses
  the per-source temporal grouping the LSTM was trained on. Rejected
  because the resulting LSTM evaluation would tell us nothing about
  generalisation — it would conflate dataset shift with sequence-
  construction shift.

- **Use a single-step prediction.** Reduce the LSTM to predicting on
  windows of 10 flows from the same fold. Same issue — the LSTM
  expects per-source sequences.

- **Train a fresh LSTM on UNSW.** Defeats the purpose of cross-dataset
  evaluation, which is to test generalisation of CIC-IDS2017-trained
  models.

The LSTM exclusion is documented as a methodological limitation in
§7.4 of the dissertation. We expect the LSTM would exhibit similar
generalisation failure to RF given the same underlying mechanism
(learning dataset-specific signatures), though the precise failure
mode might differ.

### 4. Binary metrics only

UNSW's attack taxonomy doesn't overlap with CIC-IDS2017's. We cannot
compute per-class recall comparisons (the natural unit of the
within-dataset analyses). The cross-dataset evaluation is therefore
**binary classification only** — "did the model flag this as attack
or not."

This is methodologically cleaner. Binary metrics are exactly what
generalisation testing requires.

## Consequences

### Headline finding: complete generalisation failure for both models

| Metric | RF on CIC-IDS17 | RF on UNSW | AE on CIC-IDS17 | AE on UNSW |
|---|---:|---:|---:|---:|
| Accuracy | 0.9917 | 0.8000 | 0.8318 | 0.1999 |
| Recall | 0.9995 | 0.0000 | 0.3922 | 0.9997 |
| Precision | 0.9626 | 0.0000 | 0.6749 | 0.1999 |
| FPR | 0.0104 | 0.0000 | 0.0506 | 1.0000 |
| **ROC-AUC** | **0.9998** | **0.3836** | **0.9204** | **0.4944** |
| F1 | 0.9807 | 0.0000 | 0.4961 | 0.3332 |

The most diagnostic finding: **RF's ROC-AUC of 0.38 is sub-random.**
The supervised RF doesn't merely fail to generalise; it generalises
to the *wrong* signal. UNSW benign flows receive higher mean
attack-probability (0.0756) than UNSW attack flows (0.0627).

**RF's threshold sweep** confirms the failure is not threshold-related.
Across 50 thresholds spanning [0.01, 0.50], the maximum F1 attained
is 0.3124 (at threshold 0.010), which is **worse than a random
classifier's baseline F1 (0.333) on a 20%-attack dataset**.

**AE's threshold-shift** mechanism is structural rather than
calibration-related. UNSW benign flows have median reconstruction
error 0.539 (366× higher than CIC-IDS17's 0.001); UNSW attack flows
have median 0.515. **Both classes lie far above the threshold of
0.097**, leading to the FPR = 1.0 outcome. Critically, **UNSW benign
and attack reconstruct with similar quality** (medians 2.6 percentage
points apart, against 35× separation on CIC-IDS17), meaning the AE
has lost class-discriminative signal entirely.

### Mechanism analysis

**For RF**: feature-importance analysis on the trained RF reveals
the top-5 features account for 27% of model importance:

1. RST Flag Count (12%): CIC-IDS17 attack median 1, UNSW attack
   median 0 — **uninformative on UNSW**.
2. ACK Flag Count (6%): both directions consistent but UNSW has all
   non-zero values, putting all flows in CIC-IDS17 "attack"
   territory.
3. SYN Flag Count (5%): consistent direction, weaker discrimination.
4. FWD Init Win Bytes (4%): consistent direction.
5. Bwd Packet Length Mean (3%): **inverted between datasets**.

Two of the top-10 features show direct inversion (Bwd Packet Length
Mean, Fwd Seg Size Min). The combination of feature absence and
feature inversion produces the sub-random ranking.

The supervised RF therefore learned **CIC-IDS2017-specific TCP-flag
signatures** (driven by FTP-Patator/SSH-Patator brute force attacks
and DoS attacks). UNSW attacks are different in kind: exploits target
software vulnerabilities without characteristic flag patterns;
fuzzers generate random inputs; reconnaissance may resemble legitimate
scanning.

**For AE**: the AE was trained on Monday-only benign CIC-IDS2017
flows. UNSW flows have different feature distributions (different
network architecture, different applications, different time periods).
The AE cannot reconstruct UNSW flows accurately because it has never
seen this distribution. Both classes reconstruct similarly poorly,
leaving no discriminative signal.

### Implications for RQ1

The original research question — *which model best detects
ransomware-like network traffic?* — has a more nuanced answer:

- **Within CIC-IDS2017**: LSTM >= RF >> AE.
- **Cross-dataset on UNSW**: all three approaches fail completely
  (RF and AE empirically; LSTM by methodological extension).

Architecture choice doesn't change the fundamental issue. The
conclusion is **methodological**: within-dataset NIDS evaluation
dramatically overstates real-world deployment capability. Models
trained on a single dataset's network characteristics learn
dataset-specific patterns, which is a more fundamental limitation
than architecture choice.

### §7.4 (Limitations) implications

This finding directly informs the dissertation's limitations section:

1. **Within-dataset metrics are upper bounds, not deployment estimates.**
   The within-CIC-IDS2017 LSTM recall of 99.92% should be interpreted
   as "the LSTM can perfectly distinguish CIC-IDS2017 attacks from
   CIC-IDS2017 benigns when trained on CIC-IDS2017" — not as a
   prediction of real-world detection capability.

2. **Cross-dataset evaluation reveals generalisation failure invisible
   to within-dataset metrics.** A model with 99%+ within-dataset
   accuracy can have sub-random cross-dataset accuracy. This is a
   substantial finding in the methodology of NIDS evaluation.

3. **Cross-dataset generalisation requires fundamentally different
   approaches.** Future work directions include:
   - Adversarial training on synthetic cross-domain examples.
   - Domain adaptation using few-shot examples from the target network.
   - Unsupervised pre-training on broader benign distributions.
   - Feature engineering focused on protocol-agnostic signatures
     rather than specific TCP-flag patterns.

## Alternatives considered

**Alternative 1 — Skip cross-dataset evaluation.** Within-dataset
metrics on three models would suffice for the dissertation's
comparative study. Rejected — without cross-dataset testing, the
strong within-dataset metrics could lead readers to believe NIDS
deployment is easier than it actually is. The cross-dataset finding
is itself a contribution.

**Alternative 2 — Train on combined CIC-IDS2017 + UNSW.** Would test
multi-dataset generalisation rather than zero-shot transfer. Rejected
because it doesn't answer the same question — it tests "can a model
trained on multiple datasets learn generalisable patterns" rather
than "does a single-dataset-trained model generalise."

**Alternative 3 — Run LSTM via row-order sequences on UNSW.** See
"LSTM excluded" above.

**Alternative 4 — Lower decision threshold for RF.** Threshold sweep
shows no operating point produces