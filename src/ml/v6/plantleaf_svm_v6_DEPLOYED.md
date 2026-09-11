# SVM training report — v6

`plantleaf_svm_v6_6bestfeatures_plusR2_27082026_report.md` · generated 2026-08-28T09:48:43 · git `7b9ea2e`

Model: `/Volumes/Lexar 1TB/PlantLeaf/Analysis v6/Training/Models and Reports/finals/6best-plusR2/plantleaf_svm_v6_6bestfeatures_plusR2_27082026.pkl`

## Configuration

| setting | value |
|---|---|
| Mode | `v6` |
| Feature set | `(explicit --features)` |
| NaN policy | `nan` |
| Scaler | `yeo-johnson` |
| Grid scoring | `roc_auc` |
| Imputer | `median` |
| class_weight | `auto` |
| Noise pre-filter | `False` |
| Ambiguous (label 2) | `exclude` |
| Recall target | `0.9` |
| Hard-negative weight | `1.0` |
| Seed | `42` |
| Source CSV | `/Volumes/Lexar 1TB/PlantLeaf/Analysis v6/Training/training_set_26082026.csv` |
| sklearn / numpy / pandas | 1.6.1 / 2.0.2 / 2.3.3 |

**Features (7):** `peak_SNR`, `pre_SNR`, `post_SNR`, `fall_time_ms`, `rise_time_ms`, `fit_valid`, `R2`

## Dataset

| set | rows | clicks | noise | sessions |
|---|---:|---:|---:|---:|
| Set A (training) | 905 | 158 | 747 | 28 |
| Set B (held out) | 231 | 31 | 200 | 2 |

Held-out session(s): `stimolomeccanico_aloe_misurazione1_03032026_13.49`, `stimolomeccanico_cactus_misurazione1_03032026_10.39mattina_final`

> ⚠️ Set B holds **31 clicks**. Every Set B figure below carries a wide interval at that count — treat the cross-validated numbers as the decision-grade ones and Set B as corroboration.

## Single-feature AUC-ROC

How well each feature separates click from noise **on its own**, before any model sees it — model-free, kernel-free, and therefore comparable across runs and feature sets. Ranked by distance from 0.5: an AUC *below* 0.5 is inversely predictive and just as informative; only 0.5 itself is nothing.

### Set A

| feature | AUC | \|Δ0.5\| | direction | coverage |
|---|---:|---:|:--:|---:|
| `peak_SNR` | 0.790 | 0.290 | ↑ | 905/905 (100 %) |
| `fall_time_ms` | 0.787 | 0.287 | ↑ | 905/905 (100 %) |
| `fit_valid` | 0.724 | 0.224 | ↑ | 905/905 (100 %) |
| `pre_SNR` | 0.293 | 0.207 | ↓ | 905/905 (100 %) |
| `R2` | 0.685 | 0.185 | ↑ | 473/905 (52 %) |
| `rise_time_ms` | 0.585 | 0.085 | ↑ | 905/905 (100 %) |
| `post_SNR` | 0.457 | 0.043 | ↓ | 905/905 (100 %) _~chance_ |

### Set B

| feature | AUC | \|Δ0.5\| | direction | coverage |
|---|---:|---:|:--:|---:|
| `peak_SNR` | 0.880 | 0.380 | ↑ | 231/231 (100 %) |
| `fall_time_ms` | 0.835 | 0.335 | ↑ | 231/231 (100 %) |
| `fit_valid` | 0.732 | 0.232 | ↑ | 231/231 (100 %) |
| `R2` | 0.710 | 0.210 | ↑ | 138/231 (60 %) |
| `rise_time_ms` | 0.701 | 0.201 | ↑ | 231/231 (100 %) |
| `post_SNR` | 0.557 | 0.057 | ↑ | 231/231 (100 %) |
| `pre_SNR` | 0.477 | 0.023 | ↓ | 231/231 (100 %) _~chance_ |

> ⚠️ **Set A → Set B drift.** A feature that separates in training and not in the held-out session is a property of that session, not of clicks.
>
> | feature | Set A | Set B | Δ |
> |---|---:|---:|---:|
> | `pre_SNR` | 0.293 | 0.477 | +0.184 |
> | `rise_time_ms` | 0.585 | 0.701 | +0.115 |
> | `post_SNR` | 0.457 | 0.557 | +0.100 |

## Kernel: linear

Best params `{'C': 0.1, 'class_weight': 'balanced'}` · CV roc_auc 0.933 · **CV AUC-ROC 0.920** · threshold **0.142**

The grid selects hyperparameters on `roc_auc`; the threshold is tuned separately afterwards, from the out-of-fold ROC curve, to recall ≥ 0.9.

| metric | CV @ 0.50 | CV @ 0.142 | Set B |
|---|---:|---:|---:|
| Recall | 0.620 | 0.905 | 0.935 |
| Precision | 0.710 | 0.504 | 0.358 |
| Specificity | 0.946 | 0.811 | 0.740 |
| F1 | 0.662 | 0.647 | 0.518 |
| AUC-ROC | 0.920 | 0.920 | 0.940 |
| Accuracy | 0.890 | 0.828 | 0.766 |
| Confusion | TP 98 · FP 40 · FN 60 · TN 707 | TP 143 · FP 141 · FN 15 · TN 606 | TP 29 · FP 52 · FN 2 · TN 148 |

### Feature importance (linear weights)

| # | feature | weight | \|weight\| |
|---:|---|---:|---:|
| 1 | `peak_SNR` | +0.9429 | 0.9429 |
| 2 | `pre_SNR` | -0.6795 | 0.6795 |
| 3 | `fall_time_ms` | +0.6281 | 0.6281 |
| 4 | `post_SNR` | -0.5586 | 0.5586 |
| 5 | `fit_valid` | +0.4767 | 0.4767 |
| 6 | `rise_time_ms` | -0.3134 | 0.3134 |
| 7 | `R2` | +0.1929 | 0.1929 |

_Positive pushes toward click. Weights are in scaled space, so they are comparable to each other but not to raw units._

### Feature importance on Set B (permutation, held out)

Δroc_auc when each feature is shuffled on the **held-out** session(s), 30 repeats. The Set A table above says what the fitted model leans on; this says what still carries signal where it has never looked. A feature that ranks high there and near zero here is a property of the training sessions, not of clicks.

| # | feature | Δroc_auc | ± |
|---:|---|---:|---:|
| 1 | `peak_SNR` | +0.2208 | 0.0252 |
| 2 | `fall_time_ms` | +0.0886 | 0.0216 |
| 3 | `fit_valid` | +0.0479 | 0.0156 |
| 4 | `pre_SNR` | +0.0238 | 0.0136 |
| 5 | `R2` | +0.0147 | 0.0053 |
| 6 | `post_SNR` | +0.0077 | 0.0111 |
| 7 | `rise_time_ms` | -0.0035 | 0.0070 |

_Measured on 231 rows, 31 clicks. Noisy at that count — read the ordering, not the magnitudes._

### Set B, per session

| session | clicks | detected | false positives | recall |
|---|---:|---:|---:|---:|
| `stimolomeccanico_aloe_misurazione1_03032026_13.49` | 11 | 11 | 13 | 1.00 |
| `stimolomeccanico_cactus_misurazione1_03032026_10.39mattina_final` | 20 | 18 | 39 | 0.90 |

## Kernel: rbf  ← selected

Best params `{'C': 50, 'class_weight': 'balanced', 'gamma': 0.01}` · CV roc_auc 0.945 · **CV AUC-ROC 0.929** · threshold **0.121**

The grid selects hyperparameters on `roc_auc`; the threshold is tuned separately afterwards, from the out-of-fold ROC curve, to recall ≥ 0.9.

| metric | CV @ 0.50 | CV @ 0.121 | Set B |
|---|---:|---:|---:|
| Recall | 0.601 | 0.905 | 0.968 |
| Precision | 0.766 | 0.505 | 0.333 |
| Specificity | 0.961 | 0.813 | 0.700 |
| F1 | 0.674 | 0.649 | 0.496 |
| AUC-ROC | 0.929 | 0.929 | 0.958 |
| Accuracy | 0.898 | 0.829 | 0.736 |
| Confusion | TP 95 · FP 29 · FN 63 · TN 718 | TP 143 · FP 140 · FN 15 · TN 607 | TP 30 · FP 60 · FN 1 · TN 140 |

### Feature importance (permutation)

| # | feature | Δroc_auc | ± |
|---:|---|---:|---:|
| 1 | `peak_SNR` | +0.1657 | 0.0093 |
| 2 | `fall_time_ms` | +0.0612 | 0.0085 |
| 3 | `pre_SNR` | +0.0609 | 0.0053 |
| 4 | `post_SNR` | +0.0568 | 0.0088 |
| 5 | `fit_valid` | +0.0219 | 0.0030 |
| 6 | `rise_time_ms` | +0.0130 | 0.0039 |
| 7 | `R2` | +0.0076 | 0.0023 |

_Drop in roc_auc when the feature is shuffled. Near zero can mean unimportant **or** that a correlated feature carries the same information._

### Feature importance on Set B (permutation, held out)

Δroc_auc when each feature is shuffled on the **held-out** session(s), 30 repeats. The Set A table above says what the fitted model leans on; this says what still carries signal where it has never looked. A feature that ranks high there and near zero here is a property of the training sessions, not of clicks.

| # | feature | Δroc_auc | ± |
|---:|---|---:|---:|
| 1 | `peak_SNR` | +0.2472 | 0.0290 |
| 2 | `fall_time_ms` | +0.0855 | 0.0205 |
| 3 | `fit_valid` | +0.0421 | 0.0133 |
| 4 | `post_SNR` | +0.0271 | 0.0168 |
| 5 | `pre_SNR` | +0.0264 | 0.0154 |
| 6 | `R2` | +0.0236 | 0.0069 |
| 7 | `rise_time_ms` | +0.0030 | 0.0063 |

_Measured on 231 rows, 31 clicks. Noisy at that count — read the ordering, not the magnitudes._

### Set B, per session

| session | clicks | detected | false positives | recall |
|---|---:|---:|---:|---:|
| `stimolomeccanico_aloe_misurazione1_03032026_13.49` | 11 | 11 | 15 | 1.00 |
| `stimolomeccanico_cactus_misurazione1_03032026_10.39mattina_final` | 20 | 19 | 45 | 0.95 |

## Kernel comparison

| kernel | CV AUC-ROC | threshold | Set B AUC |
|---|---:|---:|---:|
| linear | 0.920 | 0.142 | 0.940 |
| rbf ← | 0.929 | 0.121 | 0.958 |

## Reproducing this run

```
src/ml/train_svm.py --csv '/Volumes/Lexar 1TB/PlantLeaf/Analysis v6/Training/training_set_26082026.csv' --class-weight auto --set-b-from-column --features peak_SNR pre_SNR post_SNR fall_time_ms rise_time_ms fit_valid R2 --report --output '/Volumes/Lexar 1TB/PlantLeaf/Analysis v6/Training/Models and Reports/finals/6best-plusR2/plantleaf_svm_v6_6bestfeatures_plusR2_27082026.pkl'
```

---

_Generated by `src/ml/train_svm.py`. Numbers are the ones printed to the terminal during the run, carried through as data rather than re-derived._
