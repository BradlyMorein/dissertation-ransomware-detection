# Dissertation Outline

**Title (working):** Comparative Evaluation of Machine Learning and Deep Learning Models for Network-Based Ransomware Detection

**Author:** Bradly Morein (UP2088701)
**Programme:** MSc Cybersecurity and Data Forensics, University of Portsmouth
**Supervisor:** Linda Yang
**Submission target:** 6 July 2026 (deadline 8 July 2026)
**Word budget:** 12,000 words
**Referencing style:** APA7

## Structure

Per supervisor's brief, six chapters plus References and a 10-minute Zoom
presentation.

| # | Chapter | Words | % |
|---|---|---:|---:|
| 1 | Introduction (written last) | 1,000 | 8% |
| 2 | Literature Review | 2,500 | 21% |
| 3 | Methodology | 2,000 | 17% |
| 4 | Research Design | 2,500 | 21% |
| 5 | Results Analysis | 3,000 | 25% |
| 6 | Conclusion | 1,000 | 8% |
| | **Total** | **12,000** | **100%** |

## Primary contribution framing

**Comparative ML/DL study primary; methodological observation secondary.**

The dissertation argues that the research project built and evaluated three
models (Random Forest, Autoencoder, LSTM) for network-based ransomware
detection on CIC-IDS2017, with cross-dataset evaluation on CIC-UNSW-NB15
as a methodological extension. The cross-dataset finding (sub-random RF
performance, complete AE failure) is presented as an emergent result that
informs how within-dataset NIDS metrics should be interpreted.

---

## Chapter 1 — Introduction (1,000 words; write last)

**Purpose**: Frame the research question, motivation, contributions, and
roadmap. Written last because the introduction is clearest when the
deliverables are known.

### Sections

**1.1 Motivation** (250 words)
- Ransomware as a current cybersecurity threat (recent statistics,
  high-profile incidents).
- The role of network-based intrusion detection in the defence stack.
- Why ML-based NIDS specifically warrants attention.

**1.2 Research Question** (200 words)
- RQ1: How do Random Forest, Autoencoder, and LSTM models compare in
  network-based ransomware detection on the CIC-IDS2017 benchmark?
- Secondary question (emergent): To what extent do within-dataset NIDS
  metrics predict cross-dataset detection capability?

**1.3 Contributions** (250 words)
- A reproducible comparative evaluation framework for three ML/DL NIDS
  architectures on CIC-IDS2017.
- An empirical demonstration that within-dataset accuracy of 99%+ can
  coexist with sub-random cross-dataset performance.
- A documented methodology trail (7 ADRs, 97 unit tests, manifest-
  driven artefacts) supporting reproducibility.

**1.4 Dissertation Structure** (300 words)
- Brief preview of each chapter's contents and how they connect.

---

## Chapter 2 — Literature Review (2,500 words)

**Purpose**: Establish the research context and identify the gap this
work fills. Single chapter combining background and related work
(per supervisor's structure).

### Sections

**2.1 Ransomware as a Threat** (450 words)
- Definition, lifecycle, propagation mechanisms.
- Network indicators: command-and-control, lateral movement, exfiltration.
- Recent incidents (briefly; for motivation, not exhaustive).

**2.2 Network-Based Intrusion Detection** (450 words)
- Signature-based vs anomaly-based detection.
- Flow-level NIDS vs packet-level NIDS.
- The shift from rule-based to ML-based detection (last ~10 years).

**2.3 Machine Learning Approaches** (500 words)
- Random Forest in NIDS: ensemble strengths, common feature sets,
  reported metrics. Reference 2-3 representative papers.
- Limitations of supervised ML for novel attacks.

**2.4 Deep Learning Approaches** (500 words)
- Autoencoder-based anomaly detection: reconstruction error as
  anomaly signal. Reference 2-3 NIDS-AE papers.
- LSTM and sequence modelling for NIDS: temporal patterns,
  multi-flow signatures. Reference 2-3 NIDS-LSTM papers.

**2.5 Benchmark Datasets and Evaluation Practices** (300 words)
- CIC-IDS2017, CIC-UNSW-NB15: what they are, why they're used.
- Reference to Engelen et al. (2021) on CIC-IDS2017 quality issues.
- Common evaluation practices in the literature.

**2.6 The Cross-Dataset Evaluation Gap** (300 words)
- Most published NIDS papers report single-dataset metrics.
- Few studies test generalisation across datasets.
- This gap motivates the cross-dataset extension of the comparative
  study (Chapter 5, Section 5.4).

### Material sources
- Engelen et al. (2021); Sharafaldin et al. (2018); Moustafa & Slay
  (2015); Yin et al. (2017); Hochreiter & Schmidhuber (1997).
- Plus 5-10 additional NIDS papers for Sections 2.3-2.4.

---

## Chapter 3 — Methodology (2,000 words)

**Purpose**: Document the high-level approach. *Not* implementation
details — those go in Research Design (Chapter 4). This chapter answers
"what is the comparative study and why."

### Sections

**3.1 Research Approach** (300 words)
- Comparative evaluation as the primary methodology.
- Three model families chosen to span the supervised/unsupervised and
  per-flow/per-sequence axes.

**3.2 Dataset Selection** (450 words)
- CIC-IDS2017 (Engelen-cleaned) as the primary dataset (ADR 0001).
- CIC-UNSW-NB15 as the secondary dataset (ADR 0002).
- Why these two specifically: feature alignment, attack diversity,
  cross-dataset comparability.
- Acknowledgement of CIC-IDS2017 known quality issues (Engelen).

**3.3 Evaluation Strategy** (400 words)
- Stratified random 70/15/15 split for within-dataset (ADR 0003).
- Whole-dataset cross-dataset evaluation (ADR 0007).
- Metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, FPR.
- Rationale for each metric's inclusion.

**3.4 Reproducibility Methodology** (450 words)
- ADR approach: 7 architectural decision records documenting major
  methodological choices.
- Manifest pattern: every experimental output includes a JSON manifest
  with hyperparameters and metrics.
- Unit testing: 97 tests across the cleaning, splitting, sequencing,
  and modelling pipeline.
- Random seed: 42 throughout (consistent across split.py, RF, AE, LSTM).

**3.5 Methodological Constraints** (400 words)
- Dataset structure constraint: CIC-IDS2017 has only 14 active source
  IPs; one source generates 87% of attack flows. Group-by-source
  robustness split not feasible (ADR 0006 §7).
- LSTM cross-dataset evaluation not feasible: UNSW is anonymised
  (ADR 0007 §3).
- These constraints are documented and discussed in Chapter 5 §5.4.

### Material sources
- ADRs 0001, 0002, 0003, 0006 §7, 0007.

---

## Chapter 4 — Research Design (2,500 words)

**Purpose**: Concrete model design and training. Implementation depth
on each of the three models. Per supervisor's brief, LSTM gets the most
attention here.

### Sections

**4.1 Preprocessing Pipeline** (350 words)
- Cleaning: row drops (NaN, infinite, protocol-zero), column drops
  (5 near-constant columns).
- Train/val/test split: stratified random per-flow, seed 42.
- Feature alignment between CIC-IDS2017 and CIC-UNSW-NB15: 71 shared
  features after dropping identifiers and routing fields.
- Documented CICFlowMeter artefacts: 5 features clipped to >= 0.

**4.2 Random Forest** (400 words; ADR 0004)
- Hyperparameters: n_estimators=200, max_depth=20, class_weight=
  'balanced', random_state=42.
- Feature selection rationale.
- Training procedure.

**4.3 Autoencoder** (450 words; ADR 0005)
- Architecture: 71→32→16→32→71 symmetric MLP.
- Training data filter: Monday-only benign flows (~260K rows).
- Preprocessing: log1p + StandardScaler (justification: heavy-tailed
  flow features).
- Threshold rule: 95th percentile of validation benign reconstruction
  errors. Trade-off discussion deferred to Chapter 5.

**4.4 LSTM** (700 words — largest section, per supervisor's brief; ADR 0006)
- Framework choice: PyTorch on Google Colab Pro (A100 GPU).
- Sequence construction: group by Src IP, sort by Timestamp, length
  10, non-overlapping, majority-vote labels.
- Architecture: 1-layer unidirectional LSTM, hidden=128, dropout=0.2,
  final hidden state → linear → sigmoid.
- Training: 20 epochs, Adam (lr=0.001), class-weighted BCE
  (pos_weight=3.749), batch size 256.
- Hyperparameter justification: each choice's rationale.

**4.5 Cross-Dataset Evaluation Methodology** (400 words; ADR 0007)
- Application of trained models to entire UNSW dataset.
- No retraining or threshold recalibration.
- LSTM exclusion rationale (UNSW anonymisation → no Src IP/Timestamp →
  no sequence construction).
- Binary classification metrics only (no class taxonomy overlap).

**4.6 Computational Environment** (200 words)
- Hardware: Dell XPS i7 11th-gen (cleaning, RF, AE locally).
- Google Colab Pro NVIDIA A100 (LSTM training, 45s wall-clock).
- Software: Python 3.11, scikit-learn 1.x, PyTorch 2.10.0+cu128,
  pandas 2.x. All managed via conda + pyproject.toml.
- Reproducibility: 97-test unit test suite, SHA-256 input hashes,
  manifests per experiment.

### Material sources
- ADRs 0001-0007 (concrete implementation details).
- README.md.
- experiments/*/.

---

## Chapter 5 — Results Analysis (3,000 words; biggest chapter)

**Purpose**: Present the comparative findings. Within-CIC-IDS2017 first,
cross-dataset second. Pie charts as recommended.

### Sections

**5.1 Within-CIC-IDS2017 Headline Metrics** (500 words)
- Three-way comparison table: RF, AE (95th pct + F1-optimal), LSTM.
- Headline numbers: each model's accuracy, precision, recall, F1,
  ROC-AUC, PR-AUC, FPR.
- Visual: comparison bar chart of all metrics across models.

**5.2 Per-Class Recall Analysis** (700 words)
- RF per-class recall: most classes 99%+ (~10 classes shown).
- AE per-class recall: PortScan and slow DoS caught; volume DoS and
  brute force missed.
- LSTM per-class recall: 14 of 16 classes at 100%; only PortScan (4
  missed of 2,394) and Bot-Attempted (1 missed of 5) are imperfect.
- **Pie chart 1**: Within-CIC-IDS2017 attack class distribution in
  the test fold.
- **Pie chart 2**: AE recall distribution across attack classes
  (showing the dominant "catches none" vs "catches all" split).
- Honest framing: the volume-DoS and brute-force gap between AE
  (0.02-0.04) and LSTM (1.00) is the main story.

**5.3 Threshold Sensitivity (AE)** (400 words)
- The 95th-percentile rule produces recall 0.39; F1-optimal at 86.5th
  percentile produces recall 0.91.
- 52-percentage-point trade-off between unsupervised and supervised
  thresholding regimes.
- Reference Figure: AE threshold sensitivity sweep.

**5.4 Cross-Dataset Evaluation on CIC-UNSW-NB15** (900 words)
- The headline finding: both RF and AE fail completely on UNSW.
- Within vs. cross-dataset table (RF and AE in each).
- **Pie chart 3**: UNSW attack class distribution.
- RF mechanism: ROC-AUC 0.38 (sub-random); threshold sweep shows no
  operating point beats random classifier baseline.
- Feature importance inversion: top-5 features were TCP flag counts
  that don't transfer; two features show direct inversion.
- AE mechanism: 366× reconstruction error shift between datasets;
  both UNSW classes reconstruct similarly poorly (no
  class-discriminative signal).
- LSTM excluded with explanation (UNSW anonymisation).

**5.5 Synthesis** (500 words)
- All three within-dataset models work; RF and LSTM at near-ceiling.
- Cross-dataset evaluation reveals these metrics are upper bounds,
  not deployment estimates.
- The supervised models learned dataset-specific TCP-flag signatures.
- The unsupervised model learned dataset-specific benign distributions.
- Architecture choice doesn't change the underlying generalisation
  failure.

### Material sources
- Results notebooks 05, 06, 07, 08.
- ADRs 0004, 0005, 0006, 0007.
- All figures in reports/figures/.

---

## Chapter 6 — Conclusion (1,000 words)

**Purpose**: Synthesise findings, discuss limitations, suggest future
work, end strongly.

### Sections

**6.1 Summary of Findings** (300 words)
- Direct answer to RQ1: within-dataset, LSTM ≥ RF >> AE; cross-dataset,
  all models fail; architecture choice doesn't address generalisation.
- Direct answer to secondary RQ: within-dataset NIDS metrics
  dramatically overstate cross-dataset capability.

**6.2 Limitations** (250 words)
- Dataset structure constraint of CIC-IDS2017 (single attacker source).
- LSTM cross-dataset evaluation infeasibility on UNSW.
- Single training dataset; no multi-dataset training experiments.
- Binary-only cross-dataset metrics (no class taxonomy overlap).

**6.3 Future Work** (250 words)
- Domain adaptation for cross-dataset NIDS.
- Adversarial training on synthetic cross-domain examples.
- Sequence-construction strategies for anonymised datasets.
- Multi-dataset training for generalisable representations.

**6.4 Conclusion** (200 words)
- The comparative study answers RQ1 within CIC-IDS2017.
- The cross-dataset finding informs how to interpret NIDS evaluation
  metrics in the literature.
- Final reflection on the methodological contribution.

### Material sources
- Synthesis across all prior chapters.

---

## Figures budget

Approximate figure count per chapter:

| Chapter | Figures | Sources |
|---|---:|---|
| 2 (Lit Rev) | 1-2 | Conceptual diagrams (e.g., NIDS architecture) |
| 3 (Methodology) | 1-2 | Pipeline diagram, ADR map |
| 4 (Research Design) | 2-3 | Architecture diagrams (AE, LSTM) |
| 5 (Results) | 8-10 | Headline tables, confusion matrices, ROC/PR, threshold sweep, per-class breakdown, **3 pie charts**, cross-dataset comparisons |
| 6 (Conclusion) | 0-1 | Optional summary diagram |

Total: ~15-20 figures. All currently in `reports/figures/`; a few new
ones may be needed for Chapter 4 architecture diagrams and Chapter 2
conceptual figures.

## Pie chart specifics (per supervisor's recommendation)

Three pie charts in Chapter 5:

1. **Attack class distribution in CIC-IDS2017 test fold** — shows the
   dataset structure that drives within-dataset comparisons.
2. **AE recall distribution across attack classes on CIC-IDS2017** —
   visually conveys the "catches some classes well, fails on others"
   pattern.
3. **Attack class distribution in CIC-UNSW-NB15** — shows the
   cross-dataset difference from CIC-IDS2017's class structure.

These will be generated in a follow-up notebook or added to existing
notebooks.

## Referencing strategy

APA7 format throughout. Reference manager: probably Zotero (set up
during writing).

Required references (minimum):
- Engelen et al. (2021) — CIC-IDS2017 cleaning
- Sharafaldin et al. (2018) — original CIC-IDS2017
- Moustafa & Slay (2015) — UNSW-NB15
- Hochreiter & Schmidhuber (1997) — LSTM
- Breiman (2001) — Random Forest

Plus 5-10 more NIDS-specific papers for the Literature Review.

## Writing timeline

| Week | Days | Focus |
|---|---|---|
| 6 (this week) | Today + 1-2 more | Chapter outline ✓, then Chapter 4 (Research Design) draft |
| 6 | rest | Chapter 4 finish, Chapter 3 (Methodology) draft |
| 7 | all | Chapter 5 (Results) draft, Chapter 2 (Literature Review) start |
| 8 | all | Chapter 2 finish, Chapter 6 (Conclusion) draft |
| 9 | all | Chapter 1 (Introduction) draft, full pass, references |
| 10 | all | Copy-edit, format check, figures finalised |
| 11 | first half | Final polish, submission target 6 July |

## Notes on supervisor's brief

- "Methodology" and "Research Design" overlap in conventional usage.
  Here they are separate per supervisor: Methodology = comparative
  study design and rationale; Research Design = concrete model design
  and training. Care needed to avoid repetition between chapters.
- Pie charts recommended for Chapter 5 model comparisons.
- Introduction written last.
- APA7 referencing throughout.
- 10-minute Zoom presentation prepared separately after submission.