# Autoencoder threshold sensitivity

The decision threshold was set at the 95th percentile of benign
reconstruction error, a rule fixed before the results were seen.
This table reports what other percentiles would have produced. It is
a sensitivity analysis, not a calibration. Selecting a percentile on
the basis of the metrics below would amount to tuning on the test
set and no figure here should be presented as a result.

Threshold as run: `0.0970091284551957`

The saved validation errors were not found, so percentiles are taken over the 248,318 test benign errors instead. The threshold as run was calibrated on validation benign error, so the percentile column below is indicative of the shape of the trade-off rather than an exact reconstruction of the calibration procedure.

## Within-dataset, CIC-IDS2017 test set

314,761 flows · 248,318 benign · 66,443 attack

| percentile | threshold | TP | FP | FN | TN | accuracy | precision | recall | F1 | FPR |
|---|---|---|---|---|---|---|---|---|---|---|
| 50th | 0.001470 | 66,443 | 124,156 | 0 | 124,162 | 0.6056 | 0.3486 | 1.0000 | 0.5170 | 0.5000 |
| 60th | 0.003884 | 66,443 | 99,327 | 0 | 148,991 | 0.6844 | 0.4008 | 1.0000 | 0.5723 | 0.4000 |
| 70th | 0.007385 | 66,404 | 74,496 | 39 | 173,822 | 0.7632 | 0.4713 | 0.9994 | 0.6405 | 0.3000 |
| 75th | 0.009763 | 65,717 | 62,080 | 726 | 186,238 | 0.8005 | 0.5142 | 0.9891 | 0.6767 | 0.2500 |
| 80th | 0.012930 | 63,947 | 49,664 | 2,496 | 198,654 | 0.8343 | 0.5629 | 0.9624 | 0.7103 | 0.2000 |
| 85th | 0.017576 | 61,583 | 37,248 | 4,860 | 211,070 | 0.8662 | 0.6231 | 0.9269 | 0.7452 | 0.1500 |
| 90th | 0.027980 | 48,697 | 24,832 | 17,746 | 223,486 | 0.8647 | 0.6623 | 0.7329 | 0.6958 | 0.1000 |
| 92.5th | 0.039800 | 39,199 | 18,624 | 27,244 | 229,694 | 0.8543 | 0.6779 | 0.5900 | 0.6309 | 0.0750 |
| **as run** | 0.097009 | 26,062 | 12,553 | 40,381 | 235,765 | 0.8318 | 0.6749 | 0.3922 | 0.4961 | 0.0506 |
| 95th | 0.099331 | 25,801 | 12,416 | 40,642 | 235,902 | 0.8314 | 0.6751 | 0.3883 | 0.4930 | 0.0500 |
| 97.5th | 0.188127 | 1,286 | 6,208 | 65,157 | 242,110 | 0.7733 | 0.1716 | 0.0194 | 0.0348 | 0.0250 |
| 99th | 0.283051 | 466 | 2,484 | 65,977 | 245,834 | 0.7825 | 0.1580 | 0.0070 | 0.0134 | 0.0100 |
| 99.5th | 0.359329 | 324 | 582 | 66,119 | 247,736 | 0.7881 | 0.3576 | 0.0049 | 0.0096 | 0.0023 |
| 99.9th | 0.407116 | 96 | 127 | 66,347 | 248,191 | 0.7888 | 0.4305 | 0.0014 | 0.0029 | 0.0005 |

## Cross-dataset, CIC-UNSW-NB15

447,915 flows · 358,332 benign · 89,583 attack

Two separate questions are asked here, and they must not be
confused.

The first table applies thresholds derived from **CIC-IDS2017**
benign error to CIC-UNSW-NB15. This is the deployment scenario,
a model calibrated on one network and carried to another.

| percentile | threshold | TP | FP | FN | TN | accuracy | precision | recall | F1 | FPR |
|---|---|---|---|---|---|---|---|---|---|---|
| 50th | 0.001470 | 89,583 | 358,332 | 0 | 0 | 0.2000 | 0.2000 | 1.0000 | 0.3333 | 1.0000 |
| 60th | 0.003884 | 89,583 | 358,332 | 0 | 0 | 0.2000 | 0.2000 | 1.0000 | 0.3333 | 1.0000 |
| 70th | 0.007385 | 89,583 | 358,332 | 0 | 0 | 0.2000 | 0.2000 | 1.0000 | 0.3333 | 1.0000 |
| 75th | 0.009763 | 89,583 | 358,332 | 0 | 0 | 0.2000 | 0.2000 | 1.0000 | 0.3333 | 1.0000 |
| 80th | 0.012930 | 89,583 | 358,332 | 0 | 0 | 0.2000 | 0.2000 | 1.0000 | 0.3333 | 1.0000 |
| 85th | 0.017576 | 89,583 | 358,332 | 0 | 0 | 0.2000 | 0.2000 | 1.0000 | 0.3333 | 1.0000 |
| 90th | 0.027980 | 89,583 | 358,332 | 0 | 0 | 0.2000 | 0.2000 | 1.0000 | 0.3333 | 1.0000 |
| 92.5th | 0.039800 | 89,576 | 358,332 | 7 | 0 | 0.2000 | 0.2000 | 0.9999 | 0.3333 | 1.0000 |
| **as run** | 0.097009 | 89,553 | 358,330 | 30 | 2 | 0.1999 | 0.1999 | 0.9997 | 0.3332 | 1.0000 |
| 95th | 0.099331 | 89,553 | 358,330 | 30 | 2 | 0.1999 | 0.1999 | 0.9997 | 0.3332 | 1.0000 |
| 97.5th | 0.188127 | 89,297 | 358,197 | 286 | 135 | 0.1997 | 0.1995 | 0.9968 | 0.3325 | 0.9996 |
| 99th | 0.283051 | 62,940 | 209,317 | 26,643 | 149,015 | 0.4732 | 0.2312 | 0.7026 | 0.3479 | 0.5841 |
| 99.5th | 0.359329 | 56,192 | 206,103 | 33,391 | 152,229 | 0.4653 | 0.2142 | 0.6273 | 0.3194 | 0.5752 |
| 99.9th | 0.407116 | 54,553 | 204,977 | 35,030 | 153,355 | 0.4642 | 0.2102 | 0.6090 | 0.3125 | 0.5720 |

The second table re-derives the percentiles from
**CIC-UNSW-NB15's own** benign error. This is not a scenario that
could occur in practice, because it requires labelled benign
traffic from the target network. It is included only to separate
two possible explanations of the collapse, a threshold placed in
the wrong position against scores that are still informative, or
scores that carry no information at all.

| percentile | threshold | TP | FP | FN | TN | accuracy | precision | recall | F1 | FPR |
|---|---|---|---|---|---|---|---|---|---|---|
| 50th | 0.538716 | 42,395 | 179,166 | 47,188 | 179,166 | 0.4946 | 0.1913 | 0.4732 | 0.2725 | 0.5000 |
| 60th | 0.598402 | 35,446 | 143,333 | 54,137 | 214,999 | 0.5591 | 0.1983 | 0.3957 | 0.2642 | 0.4000 |
| 70th | 0.630874 | 31,286 | 107,496 | 58,297 | 250,836 | 0.6299 | 0.2254 | 0.3492 | 0.2740 | 0.3000 |
| 75th | 0.641887 | 28,106 | 89,583 | 61,477 | 268,749 | 0.6627 | 0.2388 | 0.3137 | 0.2712 | 0.2500 |
| 80th | 0.667997 | 24,688 | 71,667 | 64,895 | 286,665 | 0.6951 | 0.2562 | 0.2756 | 0.2656 | 0.2000 |
| 85th | 0.703301 | 21,919 | 53,750 | 67,664 | 304,582 | 0.7289 | 0.2897 | 0.2447 | 0.2653 | 0.1500 |
| 90th | 0.868711 | 16,556 | 35,834 | 73,027 | 322,498 | 0.7570 | 0.3160 | 0.1848 | 0.2332 | 0.1000 |
| 92.5th | 1.020664 | 14,683 | 26,875 | 74,900 | 331,457 | 0.7728 | 0.3533 | 0.1639 | 0.2239 | 0.0750 |
| 95th | 1.369436 | 8,444 | 17,917 | 81,139 | 340,415 | 0.7789 | 0.3203 | 0.0943 | 0.1457 | 0.0500 |
| 97.5th | 322.187792 | 1,441 | 8,959 | 88,142 | 349,373 | 0.7832 | 0.1386 | 0.0161 | 0.0288 | 0.0250 |
| 99th | 1416.999597 | 155 | 3,584 | 89,428 | 354,748 | 0.7923 | 0.0415 | 0.0017 | 0.0033 | 0.0100 |
| 99.5th | 1798.144906 | 91 | 1,792 | 89,492 | 356,540 | 0.7962 | 0.0483 | 0.0010 | 0.0020 | 0.0050 |
| 99.9th | 2211.151512 | 66 | 359 | 89,517 | 357,973 | 0.7993 | 0.1553 | 0.0007 | 0.0015 | 0.0010 |

### Is any threshold capable of working here

ROC-AUC does not depend on the threshold. It measures only
whether attack flows received higher reconstruction errors than
benign flows, across every possible cut point. A value of 0.5 is
what random ordering produces.

```
ROC-AUC on CIC-UNSW-NB15 = 0.494416
```

The ranking is close to uninformative. No threshold can
separate a list that is not ordered usefully in the first
place, so the cross-dataset failure cannot be attributed to
the threshold having been placed badly. The second table
above confirms this, re-deriving the threshold from the
target network's own traffic does not recover performance.

