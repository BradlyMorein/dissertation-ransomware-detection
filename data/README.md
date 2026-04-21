# Data acquisition

Raw datasets are NOT committed to this repository. Acquire them following the
procedure below. All integrity checksums are recorded in `sha256sums.txt`.

## CSE-CIC-IDS2018 (primary dataset)

- **Source:** Canadian Institute for Cybersecurity
- **URL:** https://www.unb.ca/cic/datasets/ids-2018.html
- **Acquisition date:** [to be filled in]
- **SHA-256 of downloaded archives:** see `sha256sums.txt`
- **Corrections applied:** Engelen et al. (2021) label corrections and
  ICMP-as-UDP fixes. See `scripts/apply_corrections.sh` (to be written).

Raw data lands in `data/raw/cse-cic-ids2018/`. Corrected data lands in
`data/interim/cse-cic-ids2018/`.

## UNSW-NB15 (secondary, for cross-dataset validation)

- **Source:** University of New South Wales
- **URL:** https://research.unsw.edu.au/projects/unsw-nb15-dataset
- **Acquisition date:** [to be filled in]
- **SHA-256 of downloaded archives:** see `sha256sums.txt`

Raw data lands in `data/raw/unsw-nb15/`.

## Integrity verification

After downloading, verify each archive:

```bash
sha256sum -c data/sha256sums.txt
```

Every line should report `OK`. Any line reporting `FAILED` means the download
was corrupted; re-download before proceeding.
