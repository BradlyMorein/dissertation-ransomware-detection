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


---

## 5. Partition disjointness

Each row is hashed over all of its columns. Two partitions sharing a
hash means they contain an identical row. Because identical flow
records can legitimately occur more than once in the source capture,
any shared hashes are checked against the number of duplicate rows
already present within the partitions themselves.

  train   1,468,879 rows ·  1,467,345 distinct ·   1,534 duplicate rows within the fold
  val       314,760 rows ·    314,687 distinct ·      73 duplicate rows within the fold
  test      314,761 rows ·    314,688 distinct ·      73 duplicate rows within the fold

### Pairwise intersections

  train ∩ val: 681 shared row hashes · CHECK
  train ∩ test: 639 shared row hashes · CHECK
  val ∩ test: 143 shared row hashes · CHECK

### Reconciliation

  rows across all three partitions : 2,098,400
  distinct row hashes in the union : 2,095,257
  duplicate rows overall           : 3,143

  Shared row hashes were found across partitions.
  Duplicate rows within individual folds total 1,680.

  Internal duplicates are present, so identical flow records do
  occur in the source. Rows shared between partitions may be
  genuine repeats rather than a faulty split. Confirm against the
  source by checking whether the shared rows are duplicated in the
  cleaned dataset before the split was applied.
  VERDICT: REVIEW

### Sequence partitions

  train  1,468,760 rows ·  146,876 distinct sequence_id
  val      314,680 rows ·   31,468 distinct sequence_id
  test     314,700 rows ·   31,470 distinct sequence_id

  train ∩ val: 31,468 shared sequence_id · FAIL
  train ∩ test: 31,470 shared sequence_id · FAIL
  val ∩ test: 31,468 shared sequence_id · FAIL



---

## 6. Metrics recomputed from saved per-flow predictions

Each confusion matrix below is counted directly from the stored
`y_true` and `predictions` arrays rather than taken from the manifest.
ROC-AUC is recomputed from the stored probabilities using the rank-sum
identity, and average precision by summing precision over recall
increments. Neither calls scikit-learn, so the reported values are
compared against an independent implementation.

### Within-dataset, CIC-IDS2017

### Random Forest

Source: `experiments/rf/predictions.npz` · 314,761 flows

Confusion matrix counted directly from the saved arrays:

```
TP     66,412    FP      2,579
FN         31    TN    245,739
```

  total flows      : computed 314,761 · reported 314,761 · PASS

```
accuracy  = (66,412 + 245,739) / 314,761 = 0.9917079943
precision = 66,412 / 68,991 = 0.9626183125
F1        = 2PR / (P + R)      = 0.9807286206
recall    = 66,412 / 66,443 = 0.9995334347
FPR       = 2,579 / 248,318 = 0.0103858762
```

  accuracy : computed 0.9917079943 · reported 0.9917079943 · PASS
  precision: computed 0.9626183125 · reported 0.9626183125 · PASS
  F1       : computed 0.9807286206 · reported 0.9807286206 · PASS
  recall   : computed 0.9995334347 · reported 0.9995334347 · PASS
  FPR      : computed 0.0103858762 · reported 0.0103858762 · PASS

Threshold-independent metrics, recomputed from the probabilities:

  ROC-AUC by rank sum   : 0.9998087304
  Average precision     : 0.9992590510

  ROC-AUC  : computed 0.9998087304 · reported 0.9998087304 · PASS
  PR-AUC   : computed 0.9992590510 · reported 0.9992590510 · PASS

### Autoencoder

Source: `experiments/ae/predictions.npz` · 314,761 flows

Confusion matrix counted directly from the saved arrays:

```
TP     26,062    FP     12,553
FN     40,381    TN    235,765
```

  total flows      : computed 314,761 · reported 314,761 · PASS

```
accuracy  = (26,062 + 235,765) / 314,761 = 0.8318279584
precision = 26,062 / 38,615 = 0.6749190729
F1        = 2PR / (P + R)      = 0.4961449866
recall    = 26,062 / 66,443 = 0.3922459853
FPR       = 12,553 / 248,318 = 0.0505521146
```

  accuracy : computed 0.8318279584 · reported 0.8318279584 · PASS
  precision: computed 0.6749190729 · reported 0.6749190729 · PASS
  F1       : computed 0.4961449866 · reported 0.4961449866 · PASS
  recall   : computed 0.3922459853 · reported 0.3922459853 · PASS
  FPR      : computed 0.0505521146 · reported 0.0505521146 · PASS

Threshold-independent metrics, recomputed from the probabilities:

  ROC-AUC by rank sum   : 0.9203888578
  Average precision     : 0.5925172412

  ROC-AUC  : computed 0.9203888578 · reported 0.9203888578 · PASS
  PR-AUC   : computed 0.5925172412 · reported 0.5925172412 · PASS

Threshold rule: a flow is flagged when reconstruction error exceeds
0.0970091285. Applying that rule to the saved errors reproduces
the saved predictions with 0 mismatches · PASS

### LSTM

Source: `experiments/lstm/predictions.npz` · 31,470 flows

Confusion matrix counted directly from the saved arrays:

```
TP      6,615    FP         14
FN          5    TN     24,836
```

  TP vs manifest: computed 6,615 · reported 6,615 · PASS
  TN vs manifest: computed 24,836 · reported 24,836 · PASS
  FP vs manifest: computed 14 · reported 14 · PASS
  FN vs manifest: computed 5 · reported 5 · PASS
  total flows      : computed 31,470 · reported 31,470 · PASS

```
accuracy  = (6,615 + 24,836) / 31,470 = 0.9993962504
precision = 6,615 / 6,629 = 0.9978880676
F1        = 2PR / (P + R)      = 0.9985659295
recall    = 6,615 / 6,620 = 0.9992447130
FPR       = 14 / 24,850 = 0.0005633803
```

  accuracy : computed 0.9993962504 · reported 0.9993962504 · PASS
  precision: computed 0.9978880676 · reported 0.9978880676 · PASS
  F1       : computed 0.9985659295 · reported 0.9985659295 · PASS
  recall   : computed 0.9992447130 · reported 0.9992447130 · PASS
  FPR      : computed 0.0005633803 · reported 0.0005633803 · PASS

Threshold-independent metrics, recomputed from the probabilities:

  ROC-AUC by rank sum   : 0.9999662385
  Average precision     : 0.9998667483

  ROC-AUC  : computed 0.9999662385 · reported 0.9999662385 · PASS
  PR-AUC   : computed 0.9998667483 · reported 0.9998667483 · PASS

#### LSTM training history

  epochs recorded            : 20
  best validation accuracy   : 0.999460 at epoch 15
  final validation accuracy  : 0.999364 at epoch 20
  final training loss        : 0.003858
  final validation loss      : 0.004641
  lowest validation loss     : 0.003634 at epoch 18

  The best validation accuracy did not occur at the final epoch.
  No early stopping or best-checkpoint selection was used, so the
  model evaluated on the test set is the final-epoch model rather
  than the best-performing one.

### Cross-dataset, CIC-UNSW-NB15

### Random Forest

Source: `experiments/cross_dataset/rf_predictions.npz` · 447,915 flows

Confusion matrix counted directly from the saved arrays:

```
TP          0    FP          0
FN     89,583    TN    358,332
```

  TP vs manifest: computed 0 · reported 0 · PASS
  TN vs manifest: computed 358,332 · reported 358,332 · PASS
  FP vs manifest: computed 0 · reported 0 · PASS
  FN vs manifest: computed 89,583 · reported 89,583 · PASS
  total flows      : computed 447,915 · reported 447,915 · PASS

```
accuracy  = (0 + 358,332) / 447,915 = 0.8000000000
precision = 0 / 0 = UNDEFINED (no positive predictions made)
F1        = UNDEFINED for the same reason
recall    = 0 / 89,583 = 0.0000000000
FPR       = 0 / 358,332 = 0.0000000000
```

  accuracy : computed 0.8000000000 · reported 0.8000000000 · PASS
  precision and F1 are undefined, not zero; the manifest's 0.0 is
  scikit-learn's substitution for the undefined case.
  recall   : computed 0.0000000000 · reported 0.0000000000 · PASS
  FPR      : computed 0.0000000000 · reported 0.0000000000 · PASS

Threshold-independent metrics, recomputed from the probabilities:

  ROC-AUC by rank sum   : 0.3835530074
  Average precision     : 0.1848399212

  ROC-AUC  : computed 0.3835530074 · reported 0.3835530074 · PASS
  PR-AUC   : computed 0.1848399212 · reported 0.1848399212 · PASS

### Autoencoder

Source: `experiments/cross_dataset/ae_predictions.npz` · 447,915 flows

Confusion matrix counted directly from the saved arrays:

```
TP     89,553    FP    358,330
FN         30    TN          2
```

  TP vs manifest: computed 89,553 · reported 89,553 · PASS
  TN vs manifest: computed 2 · reported 2 · PASS
  FP vs manifest: computed 358,330 · reported 358,330 · PASS
  FN vs manifest: computed 30 · reported 30 · PASS
  total flows      : computed 447,915 · reported 447,915 · PASS

```
accuracy  = (89,553 + 2) / 447,915 = 0.1999374881
precision = 89,553 / 447,883 = 0.1999473077
F1        = 2PR / (P + R)      = 0.3332415446
recall    = 89,553 / 89,583 = 0.9996651150
FPR       = 358,330 / 358,332 = 0.9999944186
```

  accuracy : computed 0.1999374881 · reported 0.1999374881 · PASS
  precision: computed 0.1999473077 · reported 0.1999473077 · PASS
  F1       : computed 0.3332415446 · reported 0.3332415446 · PASS
  recall   : computed 0.9996651150 · reported 0.9996651150 · PASS
  FPR      : computed 0.9999944186 · reported 0.9999944186 · PASS

Threshold-independent metrics, recomputed from the probabilities:

  ROC-AUC by rank sum   : 0.4944154638
  Average precision     : 0.2284088525

  ROC-AUC  : computed 0.4944154638 · reported 0.4944154638 · PASS
  PR-AUC   : computed 0.2284088525 · reported 0.2284088525 · PASS

Threshold rule: a flow is flagged when reconstruction error exceeds
0.0970091285. Applying that rule to the saved errors reproduces
the saved predictions with 0 mismatches · PASS

---

### Summary of section 6

52 checks run, 52 passed, 0 failed.

