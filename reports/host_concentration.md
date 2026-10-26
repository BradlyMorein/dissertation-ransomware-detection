# Source address concentration in CIC-IDS2017

Sequences for the LSTM are grouped by `Src IP` and cut into
non-overlapping windows of 10 flows. This analysis measures how
far the grouping key determines the label.

`Src IP` is not a model feature. It is excluded from the 71 features
used by every model. The question here is not whether a model used the
address, but whether the dataset is trivially separable by it.

## train

1,468,879 flows · 310,064 attack · 34 distinct source addresses

```
Src IP                  benign      attack   % attack
172.16.0.1                 861     308,467     99.72%
192.168.10.15           91,172         468      0.51%
192.168.10.8           143,019         407      0.28%
192.168.10.9            95,768         261      0.27%
192.168.10.14           87,165         234      0.27%
192.168.10.5           105,626         227      0.21%
109.201.130.57               3           0      0.00%
104.97.95.20                 1           0      0.00%
159.203.79.231               1           0      0.00%
109.201.130.58               3           0      0.00%
192.168.10.12           87,456           0      0.00%
185.49.84.72                 9           0      0.00%
(22 more)              547,731           0
```

The single most attack-heavy source, `172.16.0.1`, accounts for 99.48% of all attack flows in this fold, and 99.72% of its own traffic is attack.

Sources with at least 10 flows, and therefore able to form at least one window: **14 of 34**.

## val

314,760 flows · 66,442 attack · 21 distinct source addresses

```
Src IP                  benign      attack   % attack
172.16.0.1                 165      66,095     99.75%
192.168.10.15           19,438          93      0.48%
192.168.10.8            30,456          89      0.29%
192.168.10.14           18,700          69      0.37%
192.168.10.9            20,307          50      0.25%
192.168.10.5            22,489          45      0.20%
192.168.10.12           18,603           1      0.01%
123.130.127.12               2           0      0.00%
109.201.130.57               1           0      0.00%
185.49.84.72                 1           0      0.00%
159.203.79.231               1           0      0.00%
192.168.10.17           16,087           0      0.00%
(9 more)               102,068           0
```

The single most attack-heavy source, `172.16.0.1`, accounts for 99.48% of all attack flows in this fold, and 99.75% of its own traffic is attack.

Sources with at least 10 flows, and therefore able to form at least one window: **14 of 21**.

## test

314,761 flows · 66,443 attack · 20 distinct source addresses

```
Src IP                  benign      attack   % attack
172.16.0.1                 193      66,131     99.71%
192.168.10.8            30,319          89      0.29%
192.168.10.15           19,509          79      0.40%
192.168.10.9            20,390          50      0.24%
192.168.10.5            22,583          47      0.21%
192.168.10.14           18,998          46      0.24%
192.168.10.17           15,890           1      0.01%
172.217.12.147               1           0      0.00%
192.168.10.12           18,893           0      0.00%
185.49.84.72                 2           0      0.00%
192.168.10.16           15,443           0      0.00%
192.168.10.19           15,243           0      0.00%
(8 more)                70,854           0
```

The single most attack-heavy source, `172.16.0.1`, accounts for 99.53% of all attack flows in this fold, and 99.71% of its own traffic is attack.

Sources with at least 10 flows, and therefore able to form at least one window: **14 of 20**.

## Source overlap between folds

  train    34 distinct sources
  val      21 distinct sources
  test     20 distinct sources

  present in all three folds : 17
  in test but not in train   : 2

Of the 14 test sources able to form a window, 14 were also present in the training fold.

Every source that produced a test sequence had already been seen
during training. No held-out host exists.

## A trivial classifier using only the source address

Rule: predict attack if and only if `Src IP == 172.16.0.1`.
No features, no training, no model.

```
TP     66,131    FP        193
FN        312    TN    248,125

accuracy   0.9984
precision  0.9971
recall     0.9953
F1         0.9962
FPR        0.0008
```

Compare against the per-flow models reported in Chapter 5. Any model
that does not clearly exceed this baseline has not demonstrated that it
learned anything beyond what a single address lookup provides.

## Feasibility of a host-disjoint split

Holding `172.16.0.1` out of the evaluation set would remove 99.53% of the attack flows along with it.

A host-disjoint train and test split is therefore not possible on this
dataset. The confound between source identity and label cannot be
removed by resampling, only by using a capture with more than one
attacking host.

