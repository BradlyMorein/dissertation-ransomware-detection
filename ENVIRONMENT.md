# Development environment

## Hardware
- Dell XPS, Intel Core i7 (TigerLake-LP), Intel Iris Xe integrated graphics
- 16 GB RAM
- ~929 GB free disk

## Operating system
- Fedora Workstation 43 (clean install, April 2026)

## Python environment
- Conda distribution: Miniforge3 (Linux x86_64, latest as of April 2026)
- Channel priority: strict, conda-forge primary
- Environment name: `ransomware-detection`
- Python version: 3.11

## Reproducing this environment

From a machine with Miniforge installed:

```bash
conda env create -f environment.lock.yml
conda activate ransomware-detection
```

The lockfile captures exact versions of every package; `environment.yml`
records the human-readable specification with major.minor pins.

## GPU
- No NVIDIA GPU. TensorFlow is the conda-forge CPU build (`tensorflow=2.16.*`).
- LSTM training will be performed on Portsmouth's Sciama HPC cluster (or
  Google Colab Pro as fallback) — see Methodology chapter for details.

## Last verified working
- 21 April 2026
