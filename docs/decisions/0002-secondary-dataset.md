# ADR 0002 — Secondary dataset: CIC-UNSW-NB15 (CIC-augmented version)

**Status:** Accepted
**Date:** 27 April 2026
**Decided by:** Bradly Morein (supervisor sign-off pending; will be raised at Saturday 2 May meeting)

## Context

ADR 0001 designated CSE-CIC-IDS2018 as the cross-dataset validation set.
On commencing acquisition, two practical issues emerged:

1. CSE-CIC-IDS2018 acquired as raw CSVs uses CICFlowMeter-V3 (the original,
   unfixed version). The primary dataset (Engelen-cleaned CIC-IDS2017) uses
   Engelen's fixed CICFlowMeter. Cross-dataset evaluation between the two
   would conflate **data-distribution shift** with **feature-pipeline shift**,
   weakening the experimental claim.

2. To eliminate the feature-pipeline confound, CSE-CIC-IDS2018 would need
   re-extraction from raw PCAPs using Engelen's fixed tool — a 220GB download
   plus a substantial Java toolchain setup, incompatible with the compressed
   11-week schedule.

A cleaner alternative was identified: **CIC-UNSW-NB15** — the CIC team's
re-extraction of UNSW-NB15 raw PCAPs using their CICFlowMeter tool, hosted
at https://www.unb.ca/cic/datasets/cic-unsw-nb15.html.

## Decision

Replace CSE-CIC-IDS2018 with CIC-UNSW-NB15 as the secondary dataset for
cross-dataset validation in Week 7 experiments.

## Rationale

1. **Feature-pipeline alignment.** CIC-UNSW-NB15 uses the same CICFlowMeter
   feature schema as CIC-IDS2017. Cross-dataset evaluation therefore isolates
   **data-distribution shift** as the sole experimental variable, producing
   a sharper experimental claim than a comparison conflating data and
   feature pipelines.

2. **Attack diversity orthogonal to primary dataset.** CIC-UNSW-NB15 contains
   nine attack categories distinct from CIC-IDS2017's attacks: Fuzzers,
   Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode,
   Worms. Genuine cross-domain generalisation can be observed.

3. **Comparable class balance to primary dataset.** CIC-UNSW-NB15's 80/20
   benign-attack ratio is close to CIC-IDS2017's 78.91/21.09 ratio. Any
   performance changes in Week 7 reflect data-distribution shifts rather
   than imbalance shifts.

4. **Smaller and more manageable** (~448k flows in the balanced subset vs
   the ~2.1M of the primary), well-suited to evaluation-only use.

## Consequences

- Chapter 2 (Literature Review) §2.6 — discussion of dataset critiques
  expanded to cover CIC-UNSW-NB15 alongside the primary dataset and the
  rationale for CIC-augmented over original UNSW-NB15.
- Chapter 3 (Methodology) §3.3.2 — secondary dataset description updated.
  Cross-dataset claim must explicitly note that the experimental design
  isolates data-distribution shift by holding feature pipeline constant.
- Repository structure — `data/raw/cic-unsw-nb15/` replaces the previous
  `data/raw/cse-cic-ids2018/` as the secondary data location.
- Citation obligation — Mohammadian, Lashkari & Ghorbani (2024) must be
  cited per the CIC team's attribution requirement.

## Alternatives considered

**Alternative 1 — Original UNSW-NB15 (Argus/Bro features, 49 columns).**
Literature-standard; appears in many published cross-dataset NIDS studies.
Rejected because the heterogeneous feature schema (49 columns, different
extraction tooling) would confound the experimental claim. Producing a
feature-aligned subset would impose an additional preprocessing step and
remove ~20-50 features the primary models were trained on.

**Alternative 2 — Use CSE-CIC-IDS2018 as published (no PCAP re-extraction).**
Original choice from ADR 0001. Rejected on the discovery that CSE-CIC-IDS2018
uses original (unfixed) CICFlowMeter, which would conflate data-distribution
and feature-pipeline shifts in cross-dataset evaluation. CSE-CIC-IDS2018
remains a viable optional tertiary dataset for a robustness check if time
permits in Week 7.

**Alternative 3 — Both CIC-UNSW-NB15 and original UNSW-NB15.**
Most rigorous: use CIC-UNSW-NB15 as primary cross-dataset comparator and
original UNSW-NB15 (with feature alignment) as a robustness check.
Rejected on schedule grounds; can be added in Week 7 if experimental
results permit time.

## References

- Mohammadian, H., Lashkari, A. H., & Ghorbani, A. A. (2024). Poisoning
  and Evasion: Deep Learning-Based NIDS under Adversarial Attacks. In
  *21st Annual International Conference on Privacy, Security and Trust
  (PST)*.
- Moustafa, N., & Slay, J. (2015). UNSW-NB15: a comprehensive data set for
  network intrusion detection systems. In *2015 Military Communications
  and Information Systems Conference (MilCIS)* (pp. 1–6). IEEE.

## Supersedes

Partially supersedes ADR 0001's commitment to CSE-CIC-IDS2018 as the
secondary dataset. ADR 0001's primary-dataset decision (CIC-IDS2017 +
Engelen corrections) remains in force.