# ADR 0001 — Primary dataset: CIC-IDS2017 with Engelen corrections

**Status:** Accepted
**Date:** 22 April 2026
**Decided by:** Bradly Morein (supervisor consulted by email; awaiting reply)

## Context

The approved dissertation proposal cites Engelen, Rimmer & Joosen (2021)
"Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study"
as the source of corrections to be applied to a cleaned version of
CSE-CIC-IDS2018.

On commencing data acquisition, closer inspection of Engelen et al.'s
published artefacts reveals that:

- Their fixed CICFlowMeter tool and cleaned CSV release target CICIDS2017,
  not CSE-CIC-IDS2018.
- The findings about CICFlowMeter implementation issues apply to both
  datasets (the same tool produced both), but no equivalent cleaned CSV
  release for CSE-CIC-IDS2018 has been published.
- Producing an Engelen-cleaned CSE-CIC-IDS2018 would require re-extracting
  flow features from the 220GB raw PCAP archive using the fixed
  CICFlowMeter — a substantial undertaking incompatible with the
  compressed 11-week project schedule.

## Decision

Primary dataset: CIC-IDS2017 with Engelen et al. (2021) corrections applied.
Secondary (cross-dataset validation): CSE-CIC-IDS2018, used as published
with documented limitations.

## Rationale

1. **Methodological strength.** CIC-IDS2017+Engelen is the most rigorously
   cleaned publicly available NIDS dataset in the CIC family. Using it as
   primary strengthens the empirical basis of all reported findings.

2. **Schedule preservation.** No PCAP re-extraction step required;
   schedule unchanged.

3. **Comparative design preserved.** The three-model comparison (Random
   Forest, Autoencoder, LSTM) and the nine-metric evaluation framework are
   structurally identical regardless of which CIC dataset is primary.

4. **Cross-dataset validation strengthened.** Using CSE-CIC-IDS2018 as the
   secondary dataset for generalisation testing is methodologically sound:
   the secondary dataset's known imperfections become a feature of the
   robustness test, not a weakness of the primary findings.

5. **Literature alignment.** CIC-IDS2017 remains heavily used in
   contemporary NIDS research, ensuring straightforward benchmarking
   against published results.

## Consequences

- Chapter 2 (Literature Review) §2.6 — discussion of dataset critiques
  expanded to cover both 2017 and 2018, with Engelen positioned as the
  primary cleaning authority for 2017.
- Chapter 4 (Methodology) §4.2 — primary dataset description updated.
- Repository structure — data/raw/cicids2017-engelen/ replaces
  data/raw/cse-cic-ids2018/ as the primary data location;
  data/raw/cse-cic-ids2018/ retained for the cross-dataset role.
- Proposal amendment — supervisor notified by email 22 April 2026.

## Alternatives considered

**Option 1 — Stick with CSE-CIC-IDS2018 as published.**
Defensible but methodologically weaker. Most comparable studies do this;
none are penalised, but none are particularly strengthened by the choice.
Rejected in favour of higher methodological rigour now achievable at zero
schedule cost.

**Option 3 — PCAP re-extraction with Engelen's fixed CICFlowMeter on
CSE-CIC-IDS2018.**
Most rigorous in principle, but requires 220GB download, Java toolchain
setup, and approximately one week of additional work. Rejected on
schedule grounds.

## References

- Engelen, G., Rimmer, V. & Joosen, W. (2021). Troubleshooting an
  Intrusion Detection Dataset: the CICIDS2017 Case Study. In *2021 IEEE
  Security and Privacy Workshops (SPW)* (pp. 7–12). IEEE.

## Supervisor input (added 26 April 2026)

Supervisor reviewed by email and endorsed Option 2 on 26 April 2026. Two
refinements offered:

1. **Framing of cross-dataset role.** Use of CSE-CIC-IDS2018 as the
   secondary set should be framed as a deliberate test on "noisier
   real-world data" — i.e. a positive feature of the methodology rather
   than a limitation. This framing should be reflected in §2.6 (dataset
   review) and §4.2 (primary dataset choice) of the dissertation.

2. **Methodology chapter must explicitly explain the reasoning.** The
   dataset choice and the corrections applied are to be described in
   detail in the methodology chapter, with the rationale of this ADR
   reflected in chapter prose.

Status updated: **Accepted (supervisor confirmed 26 April 2026)**
