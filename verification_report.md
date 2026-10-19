# Verification report

Generated 2026-10-19 17:25:09 UTC

Every figure below is recomputed from confusion matrix counts using the
metric definitions written out in full, then compared against the value
recorded in the experiment manifests. The recomputation does not use
scikit-learn, so the reported figures are checked against an independent
calculation rather than against the library that produced them.

---

## 1. Data partitioning

### Cleaning

```
{
  "rows_in": 2100814,
  "rows_out": 2098400,
  "n_dropped_protocol_zero": 1628,
  "n_dropped_nan_inf": 786,
  "n_dropped_total": 2414,
  "pct_retained": 99.8851
}
```

### Partition sizes

```
train   1,468,879   benign  1,158,815   attack   310,064   78.8911% benign
val       314,760   benign    248,318   attack    66,442   78.8912% benign
test      314,761   benign    248,318   attack    66,443   78.8910% benign
```

  partitions sum   train+val+test: computed 2,098,400 · reported 2,098,400 · PASS
  train rows       benign+attack: computed 1,468,879 · reported 1,468,879 · PASS
  val rows       benign+attack: computed 314,760 · reported 314,760 · PASS
  test rows       benign+attack: computed 314,761 · reported 314,761 · PASS

### Stratification

  benign proportion across folds: 78.8911%, 78.8912%, 78.8910%
  spread 0.0002 percentage points · PASS

  manifest records fold_total_matches_input = True
  random seed 42, strategy 'stratified random per-flow on binary is_attack'

---

## 2. Within-dataset results, CIC-IDS2017

### Random Forest

Confusion matrix derived from the per-class BENIGN counts and the
reported recall, as the manifest does not record it directly.

```
TP     66,412    FP      2,579
FN         31    TN    245,739
```

  partition size   TP+TN+FP+FN: computed 314,761 · reported 314,761 · PASS
  attack count     TP+FN: computed 66,443 · reported 66,443 · PASS
  benign count     TN+FP: computed 248,318 · reported 248,318 · PASS

```
accuracy  = (TP + TN) / N        = (66,412 + 245,739) / 314,761 = 0.9917079943
precision = TP / (TP + FP)       = 66,412 / 68,991 = 0.9626183125
recall    = TP / (TP + FN)       = 66,412 / 66,443 = 0.9995334347
F1        = 2PR / (P + R)        = 0.9807286206
FPR       = FP / (FP + TN)       = 2,579 / 248,318 = 0.0103858762
```

  accuracy : computed 0.9917079943 · reported 0.9917079943 · PASS
  precision: computed 0.9626183125 · reported 0.9626183125 · PASS
  F1       : computed 0.9807286206 · reported 0.9807286206 · PASS
  recall   : computed 0.9995334347 · reported 0.9995334347 · PASS
  FPR      : computed 0.0103858762 · reported 0.0103858762 · PASS

### Autoencoder

Confusion matrix derived from the per-class BENIGN counts and the
reported recall, as the manifest does not record it directly.

```
TP     26,062    FP     12,553
FN     40,381    TN    235,765
```

  partition size   TP+TN+FP+FN: computed 314,761 · reported 314,761 · PASS
  attack count     TP+FN: computed 66,443 · reported 66,443 · PASS
  benign count     TN+FP: computed 248,318 · reported 248,318 · PASS

```
accuracy  = (TP + TN) / N        = (26,062 + 235,765) / 314,761 = 0.8318279584
precision = TP / (TP + FP)       = 26,062 / 38,615 = 0.6749190729
recall    = TP / (TP + FN)       = 26,062 / 66,443 = 0.3922459853
F1        = 2PR / (P + R)        = 0.4961449866
FPR       = FP / (FP + TN)       = 12,553 / 248,318 = 0.0505521146
```

  accuracy : computed 0.8318279584 · reported 0.8318279584 · PASS
  precision: computed 0.6749190729 · reported 0.6749190729 · PASS
  F1       : computed 0.4961449866 · reported 0.4961449866 · PASS
  recall   : computed 0.3922459853 · reported 0.3922459853 · PASS
  FPR      : computed 0.0505521146 · reported 0.0505521146 · PASS

Threshold: 0.0970091284551957 (95.0th percentile of validation benign error), fitted on 248,318 validation benign flows.

### LSTM

```
TP      6,615    FP         14
FN          5    TN     24,836
```

  partition size   TP+TN+FP+FN: computed 31,470 · reported 31,470 · PASS
  attack count     TP+FN: computed 6,620 · reported 6,620 · PASS
  benign count     TN+FP: computed 24,850 · reported 24,850 · PASS

```
accuracy  = (TP + TN) / N        = (6,615 + 24,836) / 31,470 = 0.9993962504
precision = TP / (TP + FP)       = 6,615 / 6,629 = 0.9978880676
recall    = TP / (TP + FN)       = 6,615 / 6,620 = 0.9992447130
F1        = 2PR / (P + R)        = 0.9985659295
FPR       = FP / (FP + TN)       = 14 / 24,850 = 0.0005633803
```

  accuracy : computed 0.9993962504 · reported 0.9993962504 · PASS
  precision: computed 0.9978880676 · reported 0.9978880676 · PASS
  F1       : computed 0.9985659295 · reported 0.9985659295 · PASS
  recall   : computed 0.9992447130 · reported 0.9992447130 · PASS
  FPR      : computed 0.0005633803 · reported 0.0005633803 · PASS

---

## 3. Cross-dataset results, CIC-UNSW-NB15

Trained on CIC-IDS2017 (Engelen-cleaned), evaluated on CIC-UNSW-NB15 (CIC-augmented).
447,915 flows, 89,583 attack, 358,332 benign.

LSTM excluded: UNSW is anonymised; lacks Src IP/Timestamp required for sequence construction

### Random Forest

```
TP          0    FP          0
FN     89,583    TN    358,332
```

  partition size   TP+TN+FP+FN: computed 447,915 · reported 447,915 · PASS
  attack count     TP+FN: computed 89,583 · reported 89,583 · PASS
  benign count     TN+FP: computed 358,332 · reported 358,332 · PASS

```
accuracy  = (TP + TN) / N        = (0 + 358,332) / 447,915 = 0.8000000000
precision = TP / (TP + FP)       = 0 / 0 = UNDEFINED
recall    = TP / (TP + FN)       = 0 / 89,583 = 0.0000000000
F1        = 2PR / (P + R)        = nan
FPR       = FP / (FP + TN)       = 0 / 358,332 = 0.0000000000
```

  accuracy : computed 0.8000000000 · reported 0.8000000000 · PASS
  precision: TP + FP = 0, so precision is undefined rather than zero.
             The manifest records 0.0, which is scikit-learn's
             substitution for the undefined case. F1 is undefined for
             the same reason and is likewise recorded as 0.0.
             Neither should be read as a measured value.
  recall   : computed 0.0000000000 · reported 0.0000000000 · PASS
  FPR      : computed 0.0000000000 · reported 0.0000000000 · PASS

### Autoencoder

```
TP     89,553    FP    358,330
FN         30    TN          2
```

  partition size   TP+TN+FP+FN: computed 447,915 · reported 447,915 · PASS
  attack count     TP+FN: computed 89,583 · reported 89,583 · PASS
  benign count     TN+FP: computed 358,332 · reported 358,332 · PASS

```
accuracy  = (TP + TN) / N        = (89,553 + 2) / 447,915 = 0.1999374881
precision = TP / (TP + FP)       = 89,553 / 447,883 = 0.1999473077
recall    = TP / (TP + FN)       = 89,553 / 89,583 = 0.9996651150
F1        = 2PR / (P + R)        = 0.3332415446
FPR       = FP / (FP + TN)       = 358,330 / 358,332 = 0.9999944186
```

  accuracy : computed 0.1999374881 · reported 0.1999374881 · PASS
  precision: computed 0.1999473077 · reported 0.1999473077 · PASS
  F1       : computed 0.3332415446 · reported 0.3332415446 · PASS
  recall   : computed 0.9996651150 · reported 0.9996651150 · PASS
  FPR      : computed 0.9999944186 · reported 0.9999944186 · PASS

Note: the Random Forest ROC-AUC of 0.3836 is below 0.50. A value
below chance means the ranking was inverted relative to the true
labels, which is a threshold-independent result and therefore not
attributable to threshold placement.

---

## 4. Stated figures

```
{
  "n_features": 71,
  "hidden_size": 128,
  "n_layers": 1,
  "dropout": 0.2,
  "n_epochs": 20,
  "batch_size": 256,
  "learning_rate": 0.001,
  "random_state": 42,
  "pos_weight": 3.748965263366699,
  "bidirectional": false
}
```

---

## Summary

43 checks run, 43 passed, 0 failed.

