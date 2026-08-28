#!/usr/bin/env python3
# Copyright (C) 2026 Tommaso Vaninetti
#
# This file is part of PlantLeaf.
#
# PlantLeaf is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# PlantLeaf is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with PlantLeaf. If not, see <https://www.gnu.org/licenses/>.

"""
train_svm.py — PlantLeaf SVM click classifier training (v6, with v5 selectable)
==============================================================================

Trains a click/noise SVM on the labelled CSV from src/ml/collect_training_set.py
(or a raw export from data_collection_dialog_v5.py). Set A / Set B split,
StratifiedGroupKFold + GridSearchCV, threshold optimisation from the ROC curve,
and feature importance.

    python3 src/ml/collect_training_set.py DATASET -o training_set.csv \
        --set-b test_aloe_1 --v5
    python3 src/ml/train_svm.py --csv training_set.csv --set-b-from-column

-- v6 IS THE DEFAULT; --v5 REPRODUCES THE OLD MODEL -------------------------
--v5 sets three things at once, because changing any one of them means the run
is no longer a v5 run:

                    v6 (default)              --v5
    features        v6-core (9)               the frozen 17
    noise filter    OFF                       ON  (R2>0.1, SPR<100)
    NaN policy      sentinels -> NaN          sentinels kept
    scaler          yeo-johnson               standard
    grid scoring    roc_auc                   recall

WARNING: THE NOISE PRE-FILTER IS THE FIT GATE v6 DELETED FROM STAGE 2. Applied
to label-0 rows, it keeps only well-fitting noise -- which on the v6 corpus
throws away the hardest negatives (1046 -> 470 in one measured run) and leaves
the model trained against a negative class that is not the population Stage 2
passes. It stays available, and stays ON under --v5, but it is off by default.

-- THE FEATURE SET IS MEANT TO BE CHANGED -----------------------------------
Four questions are open and are answered by RUNNING this, not by argument:
fit_coverage in or out; n_seg in or out; shape_novelty in or out (AUC 0.493 --
chance -- on the population that reaches Stage 3); FPE_hz vs FPE_hz_region.

    --feature-set v6-core | v6-core-region | v6 | v5
    --features peak_SNR fall_time_ms ...        explicit list
    --exclude-features shape_novelty            ablation

--exclude-features is validated against the ACTIVE set. It used to be
choices=FEATURE_NAMES, which silently made every v6 feature un-ablatable.

-- SENTINELS, AND WHY BOTH ENDS MUST AGREE ----------------------------------
The exporter writes tau_ms = -1.0 and R2 = 0.0 when the decay fit fails, on
90.2 % of candidates. Those are sentinels, not measurements: -1 sits far outside
the real range and inflates StandardScaler's std; R2 = 0.0 is worse because zero
is INSIDE the valid range, so "no fit" and "a terrible fit" are the same value.

--nan-policy nan (the v6 default) converts them, keyed on fit_valid -- never on
the value, because 202 rows have fit_valid == 1 AND R2 == 0.0 and are real. The
choice is saved as model['nan_policy'] and read back by
click_pipeline_v5.run_stage3, so inference applies the same encoding. A model
without that key is treated as v5-era, which is what the shipped one is.

fit_coverage is NOT converted: only 3 802 of 363 552 failed-fit rows have
fit_coverage == 0, the rest carry a real n_fit/decay_len measurement.

-- OPTIONS ------------------------------------------------------------------
    --csv PATH                  Labelled CSV (required)
    --set-b SESSION [...]       Session IDs to hold out
    --set-b-from-column         Take the split from the `set` column instead
    --v5                        Train the v5 model (see above)
    --feature-set NAME          v5 | v6 | v6-core | v6-core-region | v6-final
    --features F [F ...]        Explicit feature list
    --exclude-features F [...]  Ablation
    --class-weight MODE         none (default) | balanced | auto
    --scale MODE                yeo-johnson (default) | log10 | standard | quantile | robust
    --scoring MODE              roc_auc (default) | recall | average_precision
    --hard-negative-weight W    Weight for hard_negative=1 rows (default 1.0)
    --nan-policy MODE           sentinel | nan
    --impute STRATEGY           median (default) | mean
    --noise-filter / --no-noise-filter
    --ambiguous MODE            exclude (default) | click | noise
    --kernels KERNEL [...]      linear, rbf (default: both)
    --recall-target FLOAT       default 0.90
    --predict-output PATH       Input CSV + svm_probability + svm_prediction
    --report                    Also write a Markdown report beside the model
    --report-output PATH        Report location override (implies --report)
    --output PATH               Model .pkl
    --seed INT                  default 42

-- THE REPORT AND THE EXPLORER ----------------------------------------------
--report writes TWO files beside the model, so a model and the documents
explaining it never drift apart in a folder:

    foo.pkl  ->  foo_report.md        the written record
             ->  foo_explorer.html    the interactive one

The Markdown carries the configuration, the dataset split, the single-feature
AUC screen for BOTH sets with a drift warning, per kernel the CV metrics at 0.50
and at the tuned threshold, feature importance on Set A AND permutation
importance on held-out Set B, the per-session breakdown, and the exact command
that produced it. It is built from the numbers the run returned, never from
captured stdout: a scraped report drifts the first time a print is reworded.

The HTML is self-contained -- open it by double-clicking, no server. It carries
this run's actual out-of-fold scores and labels and recomputes every confusion
matrix in the browser, so you can drag the operating point along the ROC curve
and watch recall, precision, F1 and MCC move. Every training run produces its
own, so a new model means a new explorer automatically.

-- TWO IMPORTANCE MEASURES, AND WHY BOTH ------------------------------------
Set A importance (linear weights, or permutation) says what the FITTED MODEL
LEANS ON. Set B permutation says what STILL CARRIES SIGNAL on a session the
model never saw. A feature ranking high in the first and near zero in the second
is a property of the training sessions, not of clicks -- the report prints the
rank movement between them for exactly that reason.

-- SCALING, AND WHY NOT RobustScaler --------------------------------------
These features are ratios with long tails. Measured on training_set_26082026,
v6-core, same CV protocol (cvAUC / Set B AUC):

                  linear            rbf
    standard      0.705 / 0.910    0.867 / 0.874     <- v5 behaviour
    robust        0.547 / 0.850    0.805 / 0.879     <- WORST
    log10         0.892 / 0.955    0.919 / 0.960
    yeo-johnson   0.917 / 0.934    0.928 / 0.963     <- default
    quantile      0.902 / 0.944    0.916 / 0.962

RobustScaler is the trap: it divides by the IQR, and this data packs its bulk
into a tiny IQR with a long tail above it, so the tail is AMPLIFIED, not tamed --
peak_SNR |z|max goes 17.7 -> 198.9, local_crest 19.6 -> 2914. It is kept
selectable only so that result stays reproducible.

Bad scaling also made libsvm crawl: the first run of this comparison took over
10 minutes for one linear GridSearchCV. yeo-johnson and log10 finish in 0.4 s
with zero non-convergent fits; robust took 5.1 s with six.

-- WHY THE GRID SCORES ON roc_auc -----------------------------------------
The grid selects HYPERPARAMETERS; the decision threshold is tuned afterwards from
the out-of-fold ROC curve against --recall-target. AUC measures ranking quality,
which is exactly what that tuning consumes. recall@0.5 measures a rule that is
then discarded -- and recall alone cannot tell a good model from one that
predicts everything click (recall 1.000, precision 0.175, AUC 0.500).

It matters most with --class-weight balanced, which pushes toward exactly that
degenerate corner. Measured, rbf + v6-core: recall scoring reports a CV score of
0.894 while selecting a model whose true out-of-fold AUC is 0.830; roc_auc
reports 0.911 and selects one at 0.885.

class_weight at 947 noise : 189 clicks is w_click = 3.005, w_noise = 0.600 -- an
effective 5.01x. 'auto' puts [None, 'balanced'] in the grid and lets CV choose.
Run with and without and compare AUC at the same tuned threshold: if it does not
move, the threshold was already doing the work and the simpler config is the one
to report.

-- WHY THE CV SPLIT IS CHECKED BEFORE THE GRID RUNS ------------------------
Sessions are the CV groups, and on this corpus 13 of 28 Set A sessions contain no
clicks at all. StratifiedGroupKFold keeps whole sessions together, so it can hand
back a validation fold made only of click-free sessions -- with --ambiguous noise
and seed 42 it produced one of 25 rows and 0 clicks.

A click-free fold is not a rounding error. roc_auc_score returns NaN for a
single-class y_true, np.average propagates that to every candidate, and sklearn
(_search.py:1128) then ranks all candidates equal and takes best_index_ = 0 --
so best_params_ is whatever came FIRST IN GRID ORDER. It still prints like a
normal answer; only best_score_ = nan gives it away. Measured cost: the grid
returned C=0.1/gamma=scale where the repaired search picks C=5/gamma=0.01,
out-of-fold AUC 0.9019 vs 0.9144.

Only roc_auc is exposed to this. recall (via make_scorer(zero_division=0)) and
average_precision both return 0.0 for such a fold -- wrong, but finite, so the
ranking still happens. roc_auc is the v6 default, which is why this appeared with
v6 and not v5.

make_group_cv therefore walks the split seed forward (seed, seed+1, ...) until
every validation fold holds at least --min-fold-clicks clicks, and says which
seed it used. The floor is 5, not 1, because an AUC over a single positive is the
rank of one row yet carries the same 1/5 weight as a fold with 86 clicks. The
MODEL seed never changes; only the partition does, and cv_split_seed is recorded
in the .pkl so the run stays reproducible.

--min-fold-clicks 0 restores the old unrepaired behaviour exactly, which is what
reproduces pre-fix numbers (the v5 baseline CV AUC 0.754 needs it).

After the grid, a non-finite best_score_ is a hard error rather than a warning:
a model selected by grid position is not a selected model, and shipping one
quietly is the failure this whole section exists to prevent.

-- OUTPUT -------------------------------------------------------------------
A .pkl holding 'pipeline', 'threshold', 'kernel', 'features', 'all_results',
plus provenance: mode, nan_policy, feature_set, class_weight, noise_filter,
ambiguous, seed, row and session counts, sklearn/numpy/pandas versions, git SHA.

    model = joblib.load('plantleaf_svm_v6.pkl')
    X = ...   # (n, len(model['features'])), columns in model['features'] order
    proba = model['pipeline'].predict_proba(X)[:, 1]
    pred  = (proba >= model['threshold']).astype(int)   # 1 = click

model['features'] is authoritative at inference -- schema membership is not model
membership. NOTE: with add_indicator the fitted estimator sees MORE columns than
model['features'] lists; the imputer inside the Pipeline appends them, so callers
must keep passing exactly model['features'] and let the Pipeline do the rest.
"""

from __future__ import annotations

import sys
import argparse
import datetime as _dt
import json
import shlex
import subprocess as _sp
from pathlib import Path

import numpy as np

import pandas as pd

import joblib

import sklearn
from sklearn.svm import SVC
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer,
    FunctionTransformer,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    StratifiedGroupKFold,
    GridSearchCV,
    cross_val_predict,
)
from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
    make_scorer,
)
from sklearn.inspection import permutation_importance


# Bare-name import, matching the convention in evaluate_candidates.py:97 and
# analyze_dataset.py:81. Both names imported here are used at TRAINING time only
# — LOG_COLUMNS is data and assert_log_safe is called directly — so neither ends
# up inside a pickle. The transform itself uses numpy.log10 for exactly that
# reason; see build_scaler.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_transforms import LOG_COLUMNS, assert_log_safe   # noqa: E402


def _git_sha() -> str:
    """Short SHA of the tree that produced this model, or '' outside a repo."""
    try:
        out = _sp.run(['git', 'rev-parse', '--short', 'HEAD'],
                      cwd=Path(__file__).resolve().parent,
                      capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ''
    except (OSError, _sp.SubprocessError):
        return ''


# ── Constants ─────────────────────────────────────────────────────────────────

#: The v5 feature vector, frozen. --v5 trains on exactly this, so a v5 result
#: stays reproducible without checking out an old commit.
FEATURE_NAMES_V5 = [
    'peak_SNR', 'pre_SNR', 'post_SNR',
    'rise_time_ms', 'fall_time_ms', 'asymmetry_integral',
    'ZCR_pre', 'ZCR_click', 'ZCR_post',
    'kurtosis', 'centroid_shift_hz',
    'tau_ms', 'R2', 'fit_coverage',
    'SPR', 'R_spectral', 'FPE_hz',
]

#: The v6 proposal's set — SPECTRAL_FEATURES_v6_PROPOSAL.md §1, decisions
#: D10/D15/D16 (out: R_spectral, SPR, centroid_shift_hz), D5/D6/D20 (in:
#: spectral_entropy, shape_novelty, fit_valid), D17 (FPE_hz → region).
FEATURE_NAMES_V6 = [
    'peak_SNR', 'pre_SNR', 'post_SNR',
    'rise_time_ms', 'fall_time_ms', 'asymmetry_integral',
    'ZCR_pre', 'ZCR_click', 'ZCR_post',
    'kurtosis',
    'tau_ms', 'R2', 'fit_coverage', 'fit_valid',
    'FPE_hz_region', 'spectral_entropy', 'shape_novelty',
]

#: The nine slots that survived measurement against the labelled corpus. This is
#: the DEFAULT because it is the only list where every member has evidence:
#:
#:   peak_SNR      AUC 0.806 click-vs-noise; beat k_ratio (0.609) for slot 1,
#:                 and carries LOWER session ICC (0.064 vs 0.134, log scale),
#:                 so it is also the safer choice against session leakage.
#:   fit_valid     structural — the non-optional companion to tau_ms/R2. Without
#:                 it the model cannot tell "no decay was fitted" from "a decay
#:                 was fitted and was terrible" (R2 = 0.0 means both).
#:   FPE_hz        vs FPE_hz_region is an open A/B: use the -region preset.
#:
#: Everything excluded was excluded on a number, not a hunch: SPR AUC 0.454,
#: shape_novelty 0.493, asymmetry_integral 0.575 — all at or near chance on the
#: population that actually reaches Stage 3. See STAGE2_FINDINGS_AND_STAGE3_BRIEF.md.
FEATURE_NAMES_V6_CORE = [
    'peak_SNR', 'pre_SNR', 'post_SNR',
    'rise_time_ms', 'fall_time_ms',
    'tau_ms', 'R2', 'fit_valid',
    'FPE_hz',
]

#: The FPE_hz / FPE_hz_region A/B partner (proposal D17). Identical otherwise, so
#: the AUC difference between the two runs is attributable to that one swap.
FEATURE_NAMES_V6_CORE_REGION = [
    ('FPE_hz_region' if f == 'FPE_hz' else f) for f in FEATURE_NAMES_V6_CORE
]

# The final, confirmed 7 features option that places on top of the v6-core set.
# It is the deployed version and default for v6.
FEATURES_NAMES_V6_FINAL = [
    'peak_SNR', 'pre_SNR', 'post_SNR',
    'rise_time_ms', 'fall_time_ms',
    'R2', 'fit_valid'
]

FEATURE_SETS = {
    'v5':             FEATURE_NAMES_V5,
    'v6':             FEATURE_NAMES_V6,
    'v6-core':        FEATURE_NAMES_V6_CORE,
    'v6-core-region': FEATURE_NAMES_V6_CORE_REGION,
    'v6-final':       FEATURES_NAMES_V6_FINAL
}

#: Back-compat alias. Kept because --predict-output and the docstring refer to
#: "the 17 features"; nothing should read this to decide what to train on.
FEATURE_NAMES = FEATURE_NAMES_V5

#: Scaling strategies. Default is yeo-johnson on measurement, not preference:
#: on training_set_26082026 it wins CV AUC on both kernels (linear 0.917 vs 0.705
#: for standard, rbf 0.928 vs 0.866), needs no column subsetting because it
#: handles zero and negative values, and is a pure sklearn built-in so it pickles
#: with no import coupling at all.
#:
#: ⚠️ 'robust' is MEASURED WORST and is kept only so that result stays
#: reproducible. RobustScaler divides by the IQR; this data packs its bulk into a
#: tiny IQR and puts a long tail above it, so the tail explodes rather than being
#: tamed — peak_SNR |z|max goes 17.7 -> 198.9, local_crest 19.6 -> 2914, and
#: linear CV AUC collapses to 0.547. Do not reach for it as "the outlier-robust
#: one" without re-reading that.
SCALERS = ('yeo-johnson', 'log10', 'standard', 'quantile', 'robust')

#: Scalers that transform EVERY column, versus log10 which touches only
#: LOG_COLUMNS. This decides which columns the nan-policy guard has to check.
SCALE_TRANSFORMS_ALL = ('yeo-johnson', 'quantile')

#: Scoring for the hyperparameter grid. Default roc_auc because the grid selects
#: hyperparameters and the DECISION THRESHOLD IS TUNED AFTERWARDS from the ROC
#: curve: AUC measures ranking quality, which is what that tuning consumes, while
#: recall@0.5 measures a rule that is then discarded. Measured, rbf + v6-core:
#: with class_weight='balanced', recall scoring reports a better-looking CV score
#: (0.894 vs 0.911) while selecting a model 0.055 AUC worse (0.830 vs 0.885).
SCORINGS = ('roc_auc', 'recall', 'average_precision')

#: Minimum clicks that must land in EVERY validation fold before a grid search is
#: allowed to run.
#:
#: This corpus groups by session, and 13 of its 28 Set A sessions contain no
#: clicks at all. StratifiedGroupKFold is greedy — it keeps whole sessions
#: together and balances as it goes — so it can and does emit a fold made
#: entirely of noise-only sessions. On
#: training_set_27082026_ambiguousincluded with --ambiguous noise and seed 42,
#: fold 0 came out 25 rows / 0 clicks.
#:
#: That single fold is not a small error. roc_auc_score returns NaN for a
#: single-class y_true (sklearn warns UndefinedMetricWarning and hands back nan);
#: np.average propagates it, so EVERY candidate's mean_test_score becomes NaN;
#: and sklearn then takes the branch at _search.py:1128 —
#:
#:     if np.isnan(array_means).all():
#:         rank_result = np.ones_like(array_means)      # every candidate rank 1
#:
#: after which best_index_ = rank.argmin() returns index 0. The "best" model is
#: whichever combination happens to be FIRST IN GRID ORDER. Measured cost on that
#: file: the grid returned C=0.1/gamma=scale with best_score_ = nan, where the
#: repaired search picks C=5/gamma=0.01 — OOF AUC 0.9019 vs 0.9144.
#:
#: Only roc_auc is exposed to this. recall (make_scorer(zero_division=0)) and
#: average_precision both return 0.0 for a click-free fold, which is wrong but
#: finite, so the mean survives and the ranking still happens. roc_auc is the v6
#: default, which is why this surfaced with v6 and not v5.
#:
#: The floor is 5 rather than 1 because an AUC over one positive is not a
#: measurement — it is the rank of a single row — and it carries the same 1/5
#: weight in the mean as a fold with 86 clicks.
MIN_FOLD_POSITIVES = 5

#: How far to walk from --seed looking for a split that clears the floor. Seeds
#: are tried in order (seed, seed+1, ...) so the choice stays deterministic and
#: reproducible from what the model records. Measured: every training CSV in the
#: corpus clears MIN_FOLD_POSITIVES within one step of 42.
MAX_SPLIT_SEED_TRIES = 50

MODE_V5, MODE_V6 = 'v5', 'v6'

#: What each mode implies when the user does not say otherwise. --v5 has to set
#: three things at once (features, noise filter, NaN policy) or a "v5 run" is not
#: actually a v5 run.
MODE_DEFAULTS = {
    MODE_V5: {'feature_set': 'v5',      'noise_filter': True,  'nan_policy': 'sentinel',
              'scale': 'standard',    'scoring': 'recall'},
    MODE_V6: {'feature_set': 'v6-core', 'noise_filter': False, 'nan_policy': 'nan',
              'scale': 'yeo-johnson', 'scoring': 'roc_auc'},
}

#: Columns the decay-fit sentinels live in. tau_ms = -1.0 and R2 = 0.0 on every
#: fit_valid == 0 row (363 552 of 363 552 measured). fit_coverage is NOT here —
#: it stays a real measurement; see click_pipeline_v5.fit_result_to_nan.
FIT_SENTINEL_COLS = ('tau_ms', 'R2')

#: The seven region-FFT features share one missingness cause: the region was too
#: short to transform. The cliff is exact — 100.00 % NaN below n_seg = 8, 0.34 %
#: at or above — so a single indicator column describes 99.0 % of the pattern.
REGION_FFT_COLS = ('spectral_entropy', 'shape_novelty', 'spectral_tilt',
                   'FPE_hz_region', 'SPR_region', 'f_50_hz', 'IQR_f')

# Noise sample pre-filtering gates (SVM_TRAINING_DATA_GUIDE.md §3.3)
# Applied to label=0 rows only. Clicks (label=1) are always kept.
NOISE_FILTER_R2_MIN  = 0.10
NOISE_FILTER_SPR_MAX = 100.0


def build_scaler(scale: str, feature_names: list[str]):
    """
    The ('scaler', ...) step of the Pipeline.

    Living INSIDE the Pipeline is what makes this correct and what makes
    inference free: GridSearchCV and cross_val_predict refit the whole Pipeline
    per fold, so the transform's statistics come from that fold's training half
    only, and joblib carries the fitted transform inside model['pipeline'] — so
    click_pipeline_v5.run_stage3 needs no change whatsoever. model['features']
    still describes the RAW input columns; the pipeline does the rest.
    """
    if scale == 'standard':
        return StandardScaler()
    if scale == 'robust':
        return RobustScaler()
    if scale == 'yeo-johnson':
        return PowerTransformer(method='yeo-johnson', standardize=True)
    if scale == 'quantile':
        return QuantileTransformer(output_distribution='normal', random_state=0)
    if scale == 'log10':
        # ⚠️ np.log10, NOT a project-local function. FunctionTransformer pickles a
        # REFERENCE to its callable, so a function of ours would bake an import
        # path into every saved model and any loader would need src/ml on
        # sys.path — which the detection worker does not have. Measured: it fails
        # with `ModuleNotFoundError: No module named 'feature_transforms'`.
        # numpy.log10 is a ufunc that pickles as `numpy.log10` and resolves
        # anywhere numpy is installed, which is anywhere the model can load at all.
        #
        # ColumnTransformer does the subsetting with built-ins only; positivity is
        # enforced upstream, loudly, by assert_log_safe() at load time.
        log_idx  = [i for i, f in enumerate(feature_names) if f in LOG_COLUMNS]
        rest_idx = [i for i in range(len(feature_names)) if i not in log_idx]
        return Pipeline([
            ('log', ColumnTransformer([
                ('log10', FunctionTransformer(np.log10, validate=False), log_idx),
                ('rest',  'passthrough',                                 rest_idx),
            ])),
            ('scale', StandardScaler()),
        ])
    raise ValueError(f"unknown scale {scale!r}; expected one of {SCALERS}")


def check_nan_policy_vs_scale(nan_policy: str, scale: str,
                              feature_names: list[str]) -> list[str]:
    """
    Return the sentinel-bearing columns this (nan_policy, scale) pair would
    corrupt. Empty list means the combination is safe.

    ⚠️ WHY THIS IS AN ERROR AND NOT A WARNING. --nan-policy and --scale are
    independent flags, so `--nan-policy sentinel --scale log10` is reachable
    without going through the --v5 bundle. In that state tau_ms still holds the
    literal -1.0 on every failed fit — 46.2 % of rows in the first real training
    export — and:

      log10       clipping -1.0 would yield -12.0, more than TEN DECADES below the
                  real log10(tau_ms) minimum of -1.796. feature_transforms raises
                  rather than clipping, so this one is at least loud.
      yeo-johnson does NOT raise, which is the dangerous case. The lambda MLE for
                  tau_ms fits to -0.1118 with the -1 cluster present against
                  -2.4125 on the real values alone: a 2.30 shift in the exponent,
                  driven entirely by an artefact, reported by nothing.
      quantile    ranks are monotone so -1 stays lowest, but it becomes a large
                  artificial tie block at the bottom of the distribution.
      standard    the pre-existing behaviour (-1 inflates the std). Not a new
                  failure, and it is what --v5 uses, so --v5 stays legal.

    A silent failure is worse than a loud one, and a wrong lambda is invisible in
    every metric the trainer prints.
    """
    if nan_policy != 'sentinel':
        return []
    if scale in SCALE_TRANSFORMS_ALL:
        candidates = FIT_SENTINEL_COLS
    elif scale == 'log10':
        candidates = tuple(c for c in FIT_SENTINEL_COLS if c in LOG_COLUMNS)
    else:                                   # standard, robust: no transform
        return []
    return [c for c in candidates if c in feature_names]


# ── Data loading and preparation ──────────────────────────────────────────────

def load_and_prepare(
    csv_path:       Path,
    set_b_sessions: list[str],
    noise_filter:   bool,
    ambiguous:      str = 'exclude',
    set_b_from_column: bool = False,
    feature_names:  list[str] | None = None,
    nan_policy:     str = 'sentinel',
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Load the labeled CSV, apply noise pre-filtering, and split Set A / Set B.

    Returns (df_a, df_b) where df_b is None if no Set B was requested.

    `set_b_from_column` reads the split from a `set` column ('A'/'B') instead of
    matching session IDs — the format scripts/v6/collect_training_set.py writes.
    The split is then stated once, in the data, rather than re-typed on every
    training run and silently diverging from the one the corpus was built for.

    `ambiguous` decides what happens to label = 2 rows (the reviewer's "could be
    either"): 'exclude' (default), 'click' or 'noise'. The two non-default values
    exist for a sensitivity check — if recall moves a lot between them, the
    ambiguous set is carrying real signal and deserves a weighting scheme rather
    than a hard assignment.
    """
    df = pd.read_csv(csv_path)

    active = list(feature_names) if feature_names else list(FEATURE_NAMES_V5)

    # Verify required columns. This follows the ACTIVE feature list: a v6 preset
    # needs columns a v5 CSV does not have, and the failure has to name them
    # rather than surfacing later as a KeyError inside the Pipeline.
    missing = [c for c in active + ['label', 'session_id'] if c not in df.columns]
    if missing:
        print(f"ERROR: CSV is missing columns required by this feature set: {missing}")
        print(f"  Feature set in use ({len(active)}): {active}")
        print(f"  A v6 preset needs a v6-schema CSV — export with "
              f"data_collection_dialog_v5.py, or collect with "
              f"scripts/v6/collect_training_set.py.")
        sys.exit(1)

    # Defensive numeric coercion. Hand-editing the CSV in Excel under an Italian
    # locale can introduce decimal commas ("12,73") and stray whitespace, which
    # make pandas read feature columns as strings and crash on .astype(float).
    # Repair them here so training works regardless of how the CSV was produced.
    # v6 CSVs carry additional numeric columns; coerce them with the same
    # Italian-locale repair, but only when present so v5 CSVs still load.
    # `n_seg_valid` is kept for development-era CSVs; the v6 schema has no such
    # column (it is n_seg >= V6_MIN_NSEG). Every entry is guarded by
    # `if c in df.columns` below, so listing both is harmless.
    _V6_NUMERIC = [
        'spectral_entropy', 'shape_novelty', 'spectral_tilt',
        'temporal_concentration', 'FPE_hz_region', 'SPR_region', 'f_50_hz',
        'IQR_f', 'fit_valid', 'decay_len', 'n_seg', 'n_seg_valid', 'b3_frames',
        'gibbs_fired',
        # ── Stage 1 v5.1 ──
        # local_crest is the one that matters: it is a MODEL feature, so an
        # Italian-locale comma here would make it load as object dtype and crash
        # .astype(float) exactly like the v5 features this repair exists for.
        'local_crest', 'k_ratio',
        'run_id', 'run_length', 'run_crest', 'pos_in_run', 'would_pass_v5',
        # ── harmonic_confinement ──
        'harmonic_confinement', 'hc_f1_hz', 'hc_r_A', 'hc_r_B',
    ]
    _to_coerce = list(dict.fromkeys(
        active
        + [c for c in FEATURE_NAMES_V5 if c in df.columns]
        + [c for c in _V6_NUMERIC if c in df.columns]
    ))
    for col in _to_coerce:
        if df[col].dtype == object:
            df[col] = (
                df[col].astype(str)
                       .str.strip()
                       .str.replace(',', '.', regex=False)
            )
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop unlabeled rows; tolerate whitespace / "1.0"-style / comma labels.
    df['label'] = pd.to_numeric(
        df['label'].astype(str).str.strip().str.replace(',', '.', regex=False)
                   .replace('', np.nan),
        errors='coerce',
    )
    df = df[df['label'].notna()]
    df['label'] = df['label'].astype(int)

    # ── label = 2 is AMBIGUOUS: judged, but not a class ──────────────────────
    # The reviewer marks a row 2 when it could be a click and could be noise.
    # Forcing such rows into a class injects label noise into BOTH, and at ~100
    # positives a handful of wrongly-forced ones is a large fraction of the
    # positive class.
    #
    # ⚠️ This MUST be explicit. Left alone, a 2 survives notna(), casts to int,
    # is counted in neither tally above, escapes the label==0 noise pre-filter,
    # and then reaches the estimator as a THIRD CLASS — silently turning a binary
    # problem into a 3-class one and making every reported metric meaningless.
    n_ambig = int((df['label'] == 2).sum())
    if ambiguous == 'click':
        df.loc[df['label'] == 2, 'label'] = 1
    elif ambiguous == 'noise':
        df.loc[df['label'] == 2, 'label'] = 0
    else:                                    # 'exclude' (default)
        df = df[df['label'] != 2].copy()
    # Anything that is neither 0 nor 1 by now is a corrupt cell, not a decision.
    stray = df[~df['label'].isin((0, 1))]
    if len(stray):
        print(f"  ⚠️  dropping {len(stray)} row(s) with a label outside 0/1/2: "
              f"{sorted(stray['label'].unique())[:5]}")
        df = df[df['label'].isin((0, 1))].copy()

    n_clicks = (df['label'] == 1).sum()
    n_noise  = (df['label'] == 0).sum()
    print(f"Labeled rows loaded:  {len(df)}")
    print(f"  label=1 (clicks) :  {n_clicks}")
    print(f"  label=0 (noise)  :  {n_noise}")
    if n_ambig:
        _what = {'exclude': 'EXCLUDED from training',
                 'click':   'counted as CLICKS  (--ambiguous click)',
                 'noise':   'counted as NOISE   (--ambiguous noise)'}[ambiguous]
        print(f"  label=2 (ambiguous): {n_ambig}  → {_what}")
    print(f"  Sessions         :  {df['session_id'].nunique()}")

    # Apply noise pre-filtering gates (label=0 rows only)
    #
    # NaN behaviour is unchanged from v5, deliberately. A v6 CSV writes NaN for R2
    # where a v5 CSV wrote the 0.0 sentinel, and both fail `R2 > NOISE_FILTER_R2_MIN`
    # — NaN because every comparison against NaN is False, 0.0 because it is below
    # the threshold. So unfittable noise rows are excluded from the noise class
    # exactly as before, and this filter needed no change. (That is NOT true of the
    # Stage-2 gates, where the same NaN semantics INVERT the outcome and had to be
    # handled explicitly — see _stage2_reason and evaluate_candidates.apply_stage2.)
    if noise_filter:
        noise_mask  = df['label'] == 0
        noise_valid = noise_mask & (df['R2'] > NOISE_FILTER_R2_MIN) & (df['SPR'] < NOISE_FILTER_SPR_MAX)
        before = noise_mask.sum()
        df = df[~noise_mask | noise_valid].copy()
        after = (df['label'] == 0).sum()
        print(f"\nNoise pre-filter  R²>{NOISE_FILTER_R2_MIN}, SPR<{NOISE_FILTER_SPR_MAX}:")
        print(f"  noise samples:  {before} → {after}")
        ratio = (df['label'] == 1).sum() / max(after, 1)
        print(f"  class ratio clicks:noise = 1:{1/ratio:.1f}" if ratio > 0 else "")

    # ── SENTINEL → NaN  (nan_policy='nan', the v6 default) ───────────────────
    # The exporter writes tau_ms = -1.0 and R2 = 0.0 when the decay fit fails,
    # on 90.2 % of candidates (363 552 of 402 861 measured). Those are sentinels,
    # not measurements, and feeding them to StandardScaler is actively harmful:
    # tau_ms = -1 sits far outside the real range (~0.1-0.6 ms), so it inflates
    # the standard deviation and compresses all genuine variation toward zero.
    # R2 = 0.0 is worse because it is SILENT — zero is inside the valid range, so
    # "no fit" and "a terrible fit" are indistinguishable.
    #
    # Converting to NaN hands both cases to the Pipeline's imputer, which is
    # refit per CV fold, and lets fit_valid carry the distinction explicitly.
    #
    # ⚠️ Keyed on fit_valid, never on the values. 202 rows have fit_valid == 1
    # AND R2 == 0.0 — a genuinely flat log-envelope, a real and terrible fit. A
    # blanket "R2 == 0 → NaN" rewrite would corrupt exactly those.
    #
    # ⚠️ fit_coverage is NOT converted. Of the 363 552 failed-fit rows only 3 802
    # have fit_coverage == 0; the rest carry a real n_fit/decay_len measurement.
    #
    # This choice is stamped into the model as 'nan_policy' and read back by
    # click_pipeline_v5.run_stage3, so inference applies the same encoding. The
    # two must agree or the model is fitted on medians and served -1.0.
    if nan_policy == 'nan':
        if 'fit_valid' not in df.columns:
            print("ERROR: nan_policy='nan' needs a fit_valid column; this CSV has none.")
            print("  Use --nan-policy sentinel, or export with the v6 schema.")
            sys.exit(1)
        failed = df['fit_valid'].fillna(0).astype(int) == 0
        n_conv = int(failed.sum())
        present = [c for c in FIT_SENTINEL_COLS if c in df.columns]
        for col in present:
            df.loc[failed, col] = np.nan
        print(f"\nNaN policy 'nan':  {n_conv} row(s) with fit_valid == 0 "
              f"→ {', '.join(present)} set to NaN")
        print(f"  fit_coverage left intact (a real measurement on failed fits)")
    elif nan_policy != 'sentinel':
        print(f"ERROR: unknown nan_policy {nan_policy!r}")
        sys.exit(1)

    # Split Set A (training) / Set B (held-out test)
    if set_b_from_column:
        if 'set' not in df.columns:
            print("ERROR: --set-b-from-column given but the CSV has no 'set' column.")
            print("  That column is written by scripts/v6/collect_training_set.py.")
            print(f"  Columns present: {sorted(df.columns)[:12]} ...")
            sys.exit(1)
        sets = df['set'].astype(str).str.strip().str.upper()
        stray = sorted(set(sets.unique()) - {'A', 'B'})
        if stray:
            print(f"ERROR: 'set' column holds values outside A/B: {stray}")
            sys.exit(1)
        df_b = df[sets == 'B'].copy()
        df_a = df[sets == 'A'].copy()

        if len(df_b) == 0:
            print("\nWarning: the 'set' column marks no rows as B — "
                  "no held-out test set.")
            df_b = None
        else:
            print(f"\nSet B (held-out test, from 'set' column): {len(df_b)} rows  "
                  f"(clicks={(df_b['label']==1).sum()}, noise={(df_b['label']==0).sum()})")
            print(f"  Sessions: {sorted(df_b['session_id'].unique())}")
        print(f"Set A (training):      {len(df_a)} rows  "
              f"(clicks={(df_a['label']==1).sum()}, noise={(df_a['label']==0).sum()})")
        print(f"  Sessions: {sorted(df_a['session_id'].unique())}")
        return df_a, df_b

    if set_b_sessions:
        b_mask = df['session_id'].isin(set_b_sessions)
        df_b   = df[b_mask].copy()
        df_a   = df[~b_mask].copy()

        if len(df_b) == 0:
            print(f"\nWarning: no rows found for Set B sessions: {set_b_sessions}")
            print(f"  Available sessions: {sorted(df['session_id'].unique())}")

        print(f"\nSet B (held-out test): {len(df_b)} rows  "
              f"(clicks={( df_b['label']==1).sum()}, noise={(df_b['label']==0).sum()})")
        print(f"  Sessions: {sorted(df_b['session_id'].unique())}")
        print(f"Set A (training):      {len(df_a)} rows  "
              f"(clicks={(df_a['label']==1).sum()}, noise={(df_a['label']==0).sum()})")
        print(f"  Sessions: {sorted(df_a['session_id'].unique())}")
    else:
        df_a = df.copy()
        df_b = None
        print("\nNo Set B specified — cross-validation only (no held-out test).")

    return df_a, df_b


def print_univariate_auc(df: pd.DataFrame, feature_names: list[str],
                         label: str = 'Set A') -> dict:
    """
    Single-feature AUC-ROC: how well each feature separates click from noise ON
    ITS OWN, before any model sees it.

    This answers a different question from the two importance measures below, and
    the difference is the point:

        univariate AUC      is this feature informative BY ITSELF? Model-free,
                            kernel-free, hyperparameter-free — so it is stable
                            across runs and comparable between feature sets.
        linear |coef_|      how much does the FITTED model lean on it, given all
                            the others? Changes when a correlated feature is
                            added or removed.
        permutation Δscore  how much does the fitted model LOSE without it?
                            Near zero for a feature whose information a
                            correlated partner also carries.

    A feature can score 0.50 here and still earn its slot through an interaction,
    and one can score high and be redundant. Read it as a screen, not a verdict —
    it is what ruled out SPR (0.454), shape_novelty (0.493) and
    asymmetry_integral (0.575), and what put peak_SNR (0.806) ahead of k_ratio
    (0.609) for slot 1.

    Ranked by |AUC - 0.5|, because an AUC BELOW 0.5 is just as informative as one
    above — it means the feature is inversely predictive, not useless. Only 0.5
    itself is uninformative.

    Coverage is reported per feature because it is not comparable otherwise: an
    AUC over the 71 % of rows where spectral_entropy exists is measured on a
    different, easier population than one over all rows.
    """
    y_all = df['label'].to_numpy(dtype=int)
    rows = []
    for f in feature_names:
        if f not in df.columns:
            continue
        v = pd.to_numeric(df[f], errors='coerce').to_numpy(dtype=np.float64)
        ok = np.isfinite(v)
        y, x = y_all[ok], v[ok]
        n_cov = int(ok.sum())
        if n_cov == 0 or len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
            rows.append((f, float('nan'), n_cov, len(y_all)))
            continue
        rows.append((f, float(roc_auc_score(y, x)), n_cov, len(y_all)))

    rows.sort(key=lambda r: -(abs(r[1] - 0.5) if r[1] == r[1] else -1))
    print(f"\n  Single-feature AUC-ROC ({label}, click vs noise, no model):")
    print(f"    {'feature':<24} {'AUC':>6} {'|Δ0.5|':>7} {'dir':>4} {'coverage':>10}")
    out = {}
    for f, auc, n_cov, n_tot in rows:
        if auc != auc:
            print(f"    {f:<24} {'n/a':>6} {'':>7} {'':>4} "
                  f"{n_cov}/{n_tot} ({100*n_cov/max(n_tot,1):.0f}%)")
            continue
        out[f] = auc
        d = abs(auc - 0.5)
        arrow = '↑' if auc >= 0.5 else '↓'
        bar = '█' * int(d / 0.5 * 20)
        flag = '' if d >= 0.05 else '   ~chance'
        print(f"    {f:<24} {auc:6.3f} {d:7.3f} {arrow:>4} "
              f"{n_cov:>5}/{n_tot} ({100*n_cov/max(n_tot,1):3.0f}%)  {bar}{flag}")
    print(f"    (↑ higher value = more click-like, ↓ lower = more click-like;"
          f" 0.500 = no separation)")
    return out


#: How the decision threshold is chosen once the model is fitted. A SEPARATE
#: decision from hyperparameter selection: the grid picks the model, this picks
#: where on that model's ROC curve you stand. One fitted model yields a different
#: confusion matrix at every threshold — on this corpus the same model spans
#: precision 0.37 to 0.76 — and AUC-ROC is the area under all of them, so it
#: summarises the whole curve and names no single operating point.
THRESHOLD_METRICS = ('recall-target', 'f1', 'youden', 'precision-target')


def choose_threshold(y, probs, metric='recall-target', recall_target=0.90,
                     precision_target=0.50):
    """
    Pick the operating point on an already-fitted model.

        recall-target     lowest FPR among thresholds reaching recall >= target.
                          Right when a missed click costs more than a false one —
                          the historical default here.
        f1                maximises F1: balanced, when neither error dominates.
        youden            maximises tpr - fpr, the point furthest from chance.
                          Prevalence-independent, so it does not chase precision
                          when positives are rare.
        precision-target  highest recall among thresholds reaching precision >=
                          target. Right when review time is the binding constraint.

    Returns (threshold, note). An unreachable target falls back to 0.5 WITH a note
    saying so, rather than silently returning the nearest miss.
    """
    fpr, tpr, thr = roc_curve(y, probs)
    if metric == 'recall-target':
        ok = tpr >= recall_target
        if not ok.any():
            return 0.5, f'recall {recall_target} unreachable - fell back to 0.5'
        return float(thr[ok][fpr[ok].argmin()]), f'lowest FPR at recall >= {recall_target}'
    if metric == 'youden':
        return float(thr[int(np.argmax(tpr - fpr))]), 'max Youden J (tpr - fpr)'
    if metric == 'f1':
        scores = [f1_score(y, (probs >= t).astype(int), zero_division=0) for t in thr]
        i = int(np.argmax(scores))
        return float(thr[i]), f'max F1 = {scores[i]:.3f}'
    if metric == 'precision-target':
        best_t, best_r = None, -1.0
        for t in thr:
            pred = (probs >= t).astype(int)
            if precision_score(y, pred, zero_division=0) >= precision_target:
                r = recall_score(y, pred, zero_division=0)
                if r > best_r:
                    best_t, best_r = float(t), r
        if best_t is None:
            return 0.5, f'precision {precision_target} unreachable - fell back to 0.5'
        return best_t, f'max recall ({best_r:.3f}) at precision >= {precision_target}'
    raise ValueError(f'unknown threshold metric {metric!r}; expected {THRESHOLD_METRICS}')


def operating_points(y, probs, chosen_thr=None, recall_target=0.90):
    """
    The confusion matrix this ONE model produces at each of several thresholds.

    Exists because "AUC is 0.93 but precision is poor" is almost always a
    threshold statement, not a model statement. Showing the row this run picked
    beside the alternatives makes the trade visible instead of leaving it to be
    re-derived by hand.
    """
    marks = {}

    def _add(t, note):
        if t is None or not np.isfinite(t):
            return
        marks.setdefault(round(float(t), 4), []).append(note)

    for tgt in (0.95, 0.90, 0.80, 0.70):
        _add(choose_threshold(y, probs, 'recall-target', recall_target=tgt)[0],
             f'recall >= {tgt:.2f}')
    _add(choose_threshold(y, probs, 'f1')[0], 'max F1')
    _add(choose_threshold(y, probs, 'youden')[0], 'max Youden J')
    _add(0.5, 'sklearn default')
    if chosen_thr is not None:
        _add(chosen_thr, 'THIS RUN')

    rows = []
    for t in sorted(marks, reverse=True):
        pred = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append({
            'threshold': t, 'note': ', '.join(marks[t]),
            'recall': float(recall_score(y, pred, zero_division=0)),
            'precision': float(precision_score(y, pred, zero_division=0)),
            'f1': float(f1_score(y, pred, zero_division=0)),
            'specificity': float(tn / (tn + fp)) if (tn + fp) else 0.0,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        })
    return rows


def print_operating_points(rows, label='Set A, cross-validated'):
    print(f"\n  Operating points - the SAME model at different thresholds:")
    print(f"    ({label}; AUC-ROC is the area under all of these and names none)")
    print(f"    {'thr':>6} {'recall':>7} {'prec':>6} {'F1':>6} {'spec':>6} "
          f"{'TP':>4} {'FP':>5} {'FN':>4} {'TN':>5}   note")
    for r in rows:
        star = '  <<<' if 'THIS RUN' in r['note'] else ''
        print(f"    {r['threshold']:6.3f} {r['recall']:7.3f} {r['precision']:6.3f} "
              f"{r['f1']:6.3f} {r['specificity']:6.3f} {r['tp']:4d} {r['fp']:5d} "
              f"{r['fn']:4d} {r['tn']:5d}   {r['note']}{star}")


# ── Metrics helper ────────────────────────────────────────────────────────────

def _print_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   probs: np.ndarray) -> dict:
    """
    Print recall, precision, specificity, F1, AUC-ROC, accuracy, confusion matrix,
    and RETURN them.

    It returns rather than only printing so --report can be built from the same
    numbers the terminal showed, instead of re-deriving them (which drifts) or
    scraping stdout (which rots).
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    recall      = recall_score(y_true, y_pred, zero_division=0)
    precision   = precision_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1          = f1_score(y_true, y_pred, zero_division=0)
    auc         = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else float('nan')
    accuracy    = (tp + tn) / len(y_true)

    print(f"  Confusion matrix  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Recall       [PRIMARY] :  {recall:.3f}   ({tp}/{tp+fn} clicks detected)")
    print(f"  Precision              :  {precision:.3f}")
    print(f"  Specificity            :  {specificity:.3f}")
    print(f"  F1                     :  {f1:.3f}")
    print(f"  AUC-ROC                :  {auc:.3f}")
    print(f"  Accuracy               :  {accuracy:.3f}   (not primary metric)")
    return {'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
            'recall': float(recall), 'precision': float(precision),
            'specificity': float(specificity), 'f1': float(f1),
            'auc': float(auc), 'accuracy': float(accuracy)}


# ── Cross-validation splitting ────────────────────────────────────────────────

def fold_composition(cv, y, groups) -> list[tuple[int, int]]:
    """(n_rows, n_clicks) of each validation fold, in fold order."""
    X_dummy = np.zeros((len(y), 1))
    return [(len(te), int((y[te] == 1).sum()))
            for _, te in cv.split(X_dummy, y, groups)]


def make_group_cv(
    y:         np.ndarray,
    groups:    np.ndarray,
    seed:      int,
    n_splits:  int = 5,
    min_pos:   int = MIN_FOLD_POSITIVES,
    label:     str = 'CV',
) -> tuple[StratifiedGroupKFold, int]:
    """
    Build a StratifiedGroupKFold whose every validation fold actually contains
    clicks, and say out loud which seed produced it.

    Grouping by session is not negotiable — rows from one recording are not
    independent, and a split that lets them straddle the fold boundary reports a
    number the model cannot reproduce on a new session. But with 13 of 28 Set A
    sessions click-free, honouring the groups means the stratification can fail:
    see MIN_FOLD_POSITIVES for what a click-free fold does to the grid.

    So: walk seed, seed+1, ... until a split clears the floor, and return the
    seed that worked. Deterministic, reproducible from the recorded --seed, and
    loud — a silently reseeded experiment is its own kind of bug.

    Raises RuntimeError rather than returning a bad split. There is no useful
    fallback here: a grid searched on NaN does not select, it returns whatever
    was first in the parameter list, and shipping that quietly is exactly the
    failure this function exists to prevent.
    """
    tried = []
    for offset in range(MAX_SPLIT_SEED_TRIES):
        cand = seed + offset
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cand)
        comp = fold_composition(cv, y, groups)
        worst = min(c for _, c in comp)
        tried.append(worst)
        if worst >= min_pos:
            shape = '  '.join(f'{n}/{c}' for n, c in comp)
            print(f"  {label} folds (rows/clicks):  {shape}")
            if offset:
                n_empty = sum(1 for g in set(groups) if not (y[groups == g] == 1).any())
                print(f"    seed {seed} put only {tried[0]} click(s) in a validation "
                      f"fold (floor is {min_pos}); using split seed {cand} instead.")
                print(f"      {n_empty} of {len(set(groups))} sessions contain no "
                      f"clicks and sessions are the CV groups, so the stratifier "
                      f"cannot always spread them. The MODEL seed is unchanged.")
            return cv, cand
        if offset == 0 and worst == 0:
            print(f"  split seed {cand}: a validation fold has NO clicks — roc_auc "
                  f"is NaN there and the grid would stop selecting. Searching...")

    n_click_sess = sum(1 for g in set(groups) if (y[groups == g] == 1).any())
    raise RuntimeError(
        f"No {n_splits}-fold session-grouped split puts at least {min_pos} click(s) "
        f"in every validation fold (tried seeds {seed}..{seed + MAX_SPLIT_SEED_TRIES - 1}; "
        f"best worst-fold was {max(tried)} click(s)).\n"
        f"  {int((y == 1).sum())} clicks live in {n_click_sess} of {len(set(groups))} "
        f"sessions - too concentrated to spread across {n_splits} folds.\n"
        f"  Fixes: label clicks in more sessions, lower --cv-folds, or lower "
        f"--min-fold-clicks (below ~5 clicks a fold's AUC is really just the rank "
        f"of a handful of rows)."
    )


# ── Single-kernel training ────────────────────────────────────────────────────

def train_kernel(
    df_a:          pd.DataFrame,
    kernel:        str,
    seed:          int,
    recall_target: float,
    feature_names: list[str],
    class_weight:  str = 'none',
    impute:        str = 'median',
    sample_weight: np.ndarray | None = None,
    scale:         str = 'yeo-johnson',
    scoring:       str = 'roc_auc',
    threshold_metric: str = 'recall-target',
    precision_target: float = 0.50,
    cv_folds:         int = 5,
    min_fold_clicks:  int = MIN_FOLD_POSITIVES,
) -> tuple[Pipeline, float, float, dict]:
    """
    Train one SVM kernel with StratifiedGroupKFold + GridSearchCV.

    Returns:
        pipeline        — fitted Pipeline (imputer + scaler + SVM), best params, on ALL Set A
        threshold       — optimal decision threshold from ROC curve
        auc_cv          — cross-validated AUC-ROC (for model comparison)
        chosen          — dict of the settings the grid actually selected

    class_weight:
        'none'      — SVC default, every sample equal (v5 behaviour)
        'balanced'  — weights inversely proportional to class frequency, recomputed
                      per CV fold by GridSearchCV. At 947 noise : 189 clicks this is
                      w_click = 3.005, w_noise = 0.600, an effective 5.01x.
        'auto'      — put [None, 'balanced'] in the grid and let cross-validation
                      pick on recall, rather than choosing in advance.
    """
    print(f"\n{'='*60}")
    print(f"  Kernel: {kernel.upper()}")
    print(f"{'='*60}")
    print(f"  Features used ({len(feature_names)}): {feature_names}")

    X      = df_a[feature_names].values.astype(np.float64)
    y      = df_a['label'].values.astype(int)
    groups = df_a['session_id'].values

    # ── Imputer ──────────────────────────────────────────────────────────────
    # median, not mean: under nan_policy='nan' this is no longer a rare safety
    # net but the primary path for the decay features (90.2 % of rows), and the
    # surviving distributions are heavy-tailed — peak_SNR reaches 1e40 on the
    # reconstruction outliers, so a mean is not a location estimate there.
    #
    # It sits INSIDE the Pipeline, which is what makes it correct: GridSearchCV
    # and cross_val_predict refit the whole Pipeline per fold, so the imputation
    # statistics come from that fold's training half only. Imputing once over the
    # full frame before splitting would leak the validation half into training.
    #
    # add_indicator: one binary column per feature that had a missing value in
    # training. For the seven region-FFT features this is the shared "region too
    # short to transform" mask (100 % NaN below n_seg = 8), which is real signal.
    # ⚠️ It widens the matrix past len(feature_names); the indicator columns are
    # unnamed, so they are reported separately and never written to
    # model['features'], which must keep describing the INPUT columns.
    region_present = [f for f in feature_names if f in REGION_FFT_COLS]
    add_indicator  = bool(region_present)

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy=impute, add_indicator=add_indicator)),
        ('scaler',  build_scaler(scale, feature_names)),
        ('svm',     SVC(kernel=kernel, probability=True, random_state=seed)),
    ])
    print(f"  Scaler         :  {scale}")
    if scale == 'robust':
        print(f"    ⚠️  measured WORST on this corpus (linear CV AUC 0.547 vs 0.917 "
              f"for yeo-johnson) — RobustScaler's IQR divisor amplifies these tails")
    if add_indicator:
        print(f"  Missingness indicator ON for: {region_present}")

    cv, split_seed = make_group_cv(y, groups, seed, n_splits=cv_folds,
                                  min_pos=min_fold_clicks, label='CV')

    if kernel == 'linear':
        param_grid = {'svm__C': [0.1, 1, 5, 10, 50]}
    else:
        param_grid = {
            'svm__C':     [0.1, 1, 5, 10, 50],
            'svm__gamma': ['scale', 'auto', 0.01, 0.1],
        }

    if class_weight == 'balanced':
        param_grid['svm__class_weight'] = ['balanced']
    elif class_weight == 'auto':
        param_grid['svm__class_weight'] = [None, 'balanced']
    print(f"  class_weight   :  {class_weight}")
    if class_weight != 'none':
        _nc, _nn = int((y == 1).sum()), int((y == 0).sum())
        _tot = _nc + _nn
        if _nc and _nn:
            print(f"    'balanced' at {_nn} noise : {_nc} clicks → "
                  f"w_click={_tot/(2*_nc):.3f}  w_noise={_tot/(2*_nn):.3f}  "
                  f"({(_tot/(2*_nc))/(_tot/(2*_nn)):.2f}x)")

    n_combos = 1
    for v in param_grid.values():
        n_combos *= len(v)
    print(f"  GridSearchCV: {n_combos} combinations × {cv_folds} folds = "
          f"{n_combos * cv_folds} fits")
    # The grid selects HYPERPARAMETERS. The decision threshold is tuned further
    # down, from the out-of-fold ROC curve, against --recall-target. Those are two
    # separate mechanisms and this line says so, because selecting on recall@0.5
    # optimises a rule the threshold tuning then throws away.
    print(f"  Scoring: {scoring} (hyperparameter selection)  |  Groups: session_id")
    print(f"    threshold is tuned separately, after the grid, "
          f"to recall >= {recall_target}")

    if scoring == 'recall':
        # make_scorer with zero_division=0 so folds whose validation set contains
        # no clicks (noise-only sessions) return 0 rather than warning.
        scorer = make_scorer(recall_score, zero_division=0)
        print(f"    ⚠️  recall alone cannot separate a good model from one that "
              f"predicts everything click (recall 1.000, AUC 0.500).")
    else:
        scorer = scoring

    grid = GridSearchCV(
        pipe, param_grid,
        cv=cv, scoring=scorer,
        refit=True, n_jobs=-1, verbose=0,
    )
    fit_params = {}
    if sample_weight is not None:
        # Hard-negative upweighting rides through the Pipeline to the estimator.
        fit_params['svm__sample_weight'] = sample_weight
    grid.fit(X, y, groups=groups, **fit_params)

    # ── Did the grid actually SELECT anything? ───────────────────────────────
    # A non-finite best_score_ means every candidate's mean was NaN, and sklearn
    # (_search.py:1128) then ranks them all 1 and best_index_ = argmin() picks
    # index 0 — the first entry in the parameter list, by position rather than by
    # merit. best_params_ still looks like a normal answer, which is what makes it
    # dangerous. make_group_cv should have prevented the known cause; this catches
    # the rest, because shipping an unselected model is worse than not shipping.
    n_nan = int(np.isnan(grid.cv_results_['mean_test_score']).sum())
    n_cand = len(grid.cv_results_['mean_test_score'])
    if not np.isfinite(grid.best_score_):
        raise RuntimeError(
            f"GridSearchCV did not select: every one of the {n_cand} candidates "
            f"scored NaN on {scoring}, so sklearn ranked them all equal and "
            f"returned the FIRST in grid order "
            f"({ {k.replace('svm__', ''): v for k, v in grid.best_params_.items()} }) "
            f"rather than the best one.\n"
            f"  Usual cause: a validation fold with a single class — roc_auc is "
            f"undefined there and NaN propagates through the mean.\n"
            f"  Fold shape was {fold_composition(cv, y, groups)} (rows, clicks)."
        )
    if n_nan:
        print(f"  ⚠️  {n_nan}/{n_cand} candidates scored NaN and were ranked last; "
              f"the selection below is over the remaining {n_cand - n_nan}.")

    best_params_clean = {k.replace('svm__', ''): v for k, v in grid.best_params_.items()}
    print(f"\n  Best params    :  {best_params_clean}")
    print(f"  Best CV {scoring:6s}:  {grid.best_score_:.3f}")

    # ── Out-of-fold probabilities for threshold optimisation ──────────────────
    # clone of best_estimator_ is re-fitted per fold — correct for OOF probs.
    print(f"  Computing out-of-fold probabilities ({cv_folds} more fits)...")
    # Same split the grid used. Re-deriving it from `seed` here would put the
    # OOF probabilities on a different partition than the one the
    # hyperparameters were chosen on.
    cv_oof = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True,
                                  random_state=split_seed)
    probs_oof = cross_val_predict(
        grid.best_estimator_, X, y,
        cv=cv_oof, groups=groups,
        method='predict_proba',
    )[:, 1]

    auc_cv = roc_auc_score(y, probs_oof) if len(np.unique(y)) > 1 else float('nan')

    # CV metrics at default threshold
    preds_default = (probs_oof >= 0.5).astype(int)
    print(f"\n  Cross-validated metrics  (threshold = 0.50):")
    m_default = _print_metrics(y, preds_default, probs_oof)

    # ── Operating point ──────────────────────────────────────────────────────
    # A SECOND, independent decision. The grid above chose the MODEL; this chooses
    # where on its ROC curve to stand. Precision and recall trade against each
    # other along that curve at fixed AUC, so "high recall, poor precision" is a
    # statement about this line, not about the model.
    opt_thr, thr_note = choose_threshold(
        y, probs_oof, threshold_metric,
        recall_target=recall_target, precision_target=precision_target)
    if 'unreachable' in thr_note:
        print(f"\n  WARNING: {thr_note}")

    preds_opt = (probs_oof >= opt_thr).astype(int)
    print(f"\n  Cross-validated metrics  "
          f"(threshold = {opt_thr:.3f}, {threshold_metric}: {thr_note}):")
    m_opt = _print_metrics(y, preds_opt, probs_oof)

    op_rows = operating_points(y, probs_oof, opt_thr, recall_target)
    print_operating_points(op_rows)

    # ── Feature importance ────────────────────────────────────────────────────
    print()
    importance = _print_feature_importance(grid.best_estimator_, kernel, X, y,
                                          groups, seed, feature_names, scoring)

    chosen = {
        'params':       best_params_clean,
        'class_weight': best_params_clean.get('class_weight', None),
        'impute':       impute,
        'add_indicator': add_indicator,
        'scale':        scale,
        'scoring':      scoring,
        'cv_best_score': float(grid.best_score_),
        # The split seed can differ from --seed when make_group_cv had to walk
        # forward past a click-free fold. Recording it is what makes the run
        # reproducible; --seed alone no longer determines the partition.
        'cv_split_seed': int(split_seed),
        'cv_folds':      int(cv_folds),
        'cv_fold_shape': fold_composition(cv, y, groups),
        'cv_at_0.5':    m_default,
        'cv_at_thr':    m_opt,
        'threshold_metric': threshold_metric,
        'threshold_note':   thr_note,
        'operating_points': op_rows,
        'probs_oof':        probs_oof,
        'importance':   importance,
    }
    return grid.best_estimator_, opt_thr, auc_cv, chosen


def _print_feature_importance(
    pipeline:      Pipeline,
    kernel:        str,
    X:             np.ndarray,
    y:             np.ndarray,
    groups:        np.ndarray,
    seed:          int,
    feature_names: list[str],
    scoring:       str = 'roc_auc',
) -> list[tuple]:
    """
    Feature importance:
      linear  → weight vector (interpretable in scaled space, directly comparable)
      rbf     → permutation importance on Set A with cross-validation
                (note: use Set B permutation importance for the final published result)
    """
    print(f"  Feature importance ({kernel}):")
    svm_model = pipeline.named_steps['svm']

    # SVC.coef_ is a PROPERTY: every access recomputes dual_coef_ @
    # support_vectors_, and `hasattr` evaluates it too — so the guard has to wrap
    # the hasattr, not just the read. Under numpy 2.x on Apple's Accelerate BLAS
    # that matmul reports spurious divide-by-zero / overflow / invalid
    # RuntimeWarnings from stale FP status flags. Verified benign: the operands
    # are finite (|dual_coef_| <= C, |support_vectors_| ~ 17 — a product that
    # cannot overflow float64), the result is finite, and it matches a manual
    # recomputation to 0.0 absolute difference. Silenced around this one access
    # only, so a real numerical problem anywhere else still surfaces — and
    # finiteness is asserted below rather than assumed.
    weights = None
    if kernel == 'linear':
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            if hasattr(svm_model, 'coef_'):
                weights = np.asarray(svm_model.coef_[0], dtype=np.float64)

    ranked: list[tuple] = []
    if weights is not None:
        if not np.isfinite(weights).all():
            print("    ⚠️  non-finite weights — that BLAS warning was NOT spurious")

        # ⚠️ coef_ can be WIDER than feature_names. SimpleImputer(add_indicator=True)
        # appends one binary column per input column that had a missing value in
        # training, so the estimator sees len(feature_names) + n_indicator columns
        # while feature_names still describes only the inputs. Indexing
        # feature_names by the coef_ position is then an IndexError — which is
        # exactly what the v6 and v6-core-region presets hit, because they are the
        # only ones carrying region-FFT features.
        #
        # The imputer records which input columns it made indicators for, so name
        # them rather than printing bare indices: "<feature> [missing]" is the
        # answer to "does the model care that this was unmeasurable", which for
        # the region-FFT family is a real question, not bookkeeping.
        names = list(feature_names)
        imputer = pipeline.named_steps.get('imputer')
        indicator = getattr(imputer, 'indicator_', None)
        if indicator is not None and getattr(indicator, 'features_', None) is not None:
            for col in indicator.features_:
                src = feature_names[col] if col < len(feature_names) else f'col{col}'
                names.append(f'{src} [missing]')
        while len(names) < len(weights):            # never index out of range
            names.append(f'col{len(names)}')

        importance = np.abs(weights)
        order      = np.argsort(importance)[::-1]
        top        = importance[order[0]] if importance[order[0]] > 0 else 1.0
        for rank, idx in enumerate(order, 1):
            direction = '↑' if weights[idx] > 0 else '↓'
            bar = '█' * int(importance[idx] / top * 20)
            print(f"    {rank:2d}. {names[idx]:<28}  {direction}{importance[idx]:.4f}  {bar}")
            ranked.append((names[idx], float(weights[idx]), float(importance[idx])))
        print(f"    (↑ = pushes toward click,  ↓ = pushes toward noise)")
        if len(names) > len(feature_names):
            print(f"    [missing] rows are imputer indicators, not input features — "
                  f"they are NOT in model['features']")

    else:
        # RBF: permutation importance on Set A
        # This is an approximation — use Set B if available for the published result.
        print(f"    Computing permutation importance on Set A "
              f"(n_repeats=15, scoring={scoring})...")
        result = permutation_importance(
            pipeline, X, y,
            n_repeats=15, random_state=seed,
            scoring=(scoring if scoring != 'recall'
                     else make_scorer(recall_score, zero_division=0)),
        )
        order = np.argsort(result.importances_mean)[::-1]
        for rank, idx in enumerate(order, 1):
            mean = result.importances_mean[idx]
            std  = result.importances_std[idx]
            bar  = '█' * max(0, int(mean / (result.importances_mean[order[0]] + 1e-9) * 20))
            print(f"    {rank:2d}. {feature_names[idx]:<22}  Δ{scoring}={mean:+.3f} "
                  f"± {std:.3f}  {bar}")
            ranked.append((feature_names[idx], float(mean), float(std)))
        print(f"    (Δ: drop in {scoring} when the feature is shuffled; "
              f"larger = more important)")
        print(f"    Note: this is Set A. Set B permutation importance is computed "
              f"in the held-out evaluation below.")
    return ranked


# ── Set B evaluation ──────────────────────────────────────────────────────────

def evaluate_on_set_b(
    df_b:          pd.DataFrame,
    pipeline:      Pipeline,
    thr:           float,
    feature_names: list[str],
    scoring:       str = 'roc_auc',
    seed:          int = 42,
    n_repeats:     int = 30,
) -> dict:
    """Evaluate the final model on the held-out Set B test set, and return it."""
    print(f"\n{'='*60}")
    print(f"  Set B — held-out test set evaluation")
    print(f"{'='*60}")

    X_b = df_b[feature_names].values.astype(np.float64)
    y_b = df_b['label'].values.astype(int)

    probs_b = pipeline.predict_proba(X_b)[:, 1]
    preds_b = (probs_b >= thr).astype(int)

    print(f"  Threshold used:  {thr:.3f}")
    metrics = _print_metrics(y_b, preds_b, probs_b)
    per_session = []

    # Per-session breakdown
    print(f"\n  Per-session breakdown (Set B):")
    print(f"    {'Session':<35}  clicks  detected  FP    recall")
    print(f"    {'-'*70}")
    for sid, grp in df_b.groupby('session_id'):
        X_s = grp[feature_names].values.astype(np.float64)
        y_s = grp['label'].values.astype(int)
        pr_s = pipeline.predict_proba(X_s)[:, 1]
        pd_s = (pr_s >= thr).astype(int)
        n_clicks = (y_s == 1).sum()
        n_det    = ((y_s == 1) & (pd_s == 1)).sum()
        n_fp     = ((y_s == 0) & (pd_s == 1)).sum()
        rec_s    = n_det / n_clicks if n_clicks > 0 else float('nan')
        print(f"    {str(sid):<35}  {n_clicks:6d}  {n_det:8d}  {n_fp:4d}  {rec_s:.2f}")
        per_session.append({'session': str(sid), 'clicks': int(n_clicks),
                            'detected': int(n_det), 'fp': int(n_fp),
                            'recall': float(rec_s)})

    # ── Permutation importance ON THE HELD-OUT SET ───────────────────────────
    # The Set A version answers "what does the fitted model lean on?". This one
    # answers the question that actually matters: "what still carries signal on a
    # session the model never saw?" A feature the model leans on heavily in
    # training and that contributes nothing here is a session artefact, and the
    # gap between the two rankings is where that shows up.
    #
    # Scored on the run's own metric so the numbers mean the same thing at both
    # ends. roc_auc is threshold-free, so this measure does not move when the
    # operating point does.
    perm_b = []
    if len(np.unique(y_b)) > 1:
        scorer = (scoring if scoring != 'recall'
                  else make_scorer(recall_score, zero_division=0))
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            r = permutation_importance(pipeline, X_b, y_b, n_repeats=n_repeats,
                                       random_state=seed, scoring=scorer)
        order = np.argsort(r.importances_mean)[::-1]
        top = r.importances_mean[order[0]] if r.importances_mean[order[0]] > 0 else 1.0
        n_click_b = int((y_b == 1).sum())
        print(f"\n  Permutation importance on SET B "
              f"(n_repeats={n_repeats}, scoring={scoring}):")
        for rank, idx in enumerate(order, 1):
            mean, std = float(r.importances_mean[idx]), float(r.importances_std[idx])
            bar = '█' * max(0, int(mean / top * 18))
            print(f"    {rank:2d}. {feature_names[idx]:<22}  Δ{scoring}={mean:+.4f} "
                  f"± {std:.4f}  {bar}")
            perm_b.append((feature_names[idx], mean, std))
        print(f"    (measured on {len(y_b)} held-out rows, {n_click_b} clicks — "
              f"noisy at that count; read the ORDER, not the values)")
    else:
        print(f"\n  Permutation importance on Set B: skipped "
              f"(the set has a single class)")

    return {'threshold': float(thr), 'n_rows': int(len(y_b)),
            'metrics': metrics, 'per_session': per_session,
            'permutation': perm_b, 'perm_scoring': scoring,
            'perm_n_repeats': n_repeats, 'n_clicks': int((y_b == 1).sum()),
            'probs': probs_b, 'y': y_b}


# ── Prediction export ────────────────────────────────────────────────────────

def save_prediction_csv(
    csv_path:       Path,
    pipeline:       Pipeline,
    threshold:      float,
    feature_names:  list[str],
    set_b_sessions: list[str],
    output_path:    Path,
) -> None:
    """
    Re-read the original CSV, run the trained pipeline on every row that has
    feature data, and write a new CSV with two added columns:

        svm_probability  — raw model score (0.0 – 1.0)
        svm_prediction   — binary decision at the chosen threshold (0 or 1)

    Rows without any feature values (empty spreadsheet rows) are kept in the
    output but get NaN in both new columns.  Unlabeled rows are included too,
    so you can inspect what the model would predict for unannotated candidates.
    """
    df = pd.read_csv(csv_path)

    # Same coercion as load_and_prepare — handle Italian decimal commas, etc.
    for col in feature_names:
        if col in df.columns and df[col].dtype == object:
            df[col] = (
                df[col].astype(str)
                       .str.strip()
                       .str.replace(',', '.', regex=False)
            )
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Tag each row so the reader knows which split it belongs to
    df['svm_set'] = 'A'
    if set_b_sessions:
        df.loc[df['session_id'].isin(set_b_sessions), 'svm_set'] = 'B'
    label_numeric = pd.to_numeric(df['label'].astype(str).str.strip(), errors='coerce')
    df.loc[label_numeric.isna(), 'svm_set'] = 'unlabeled'

    # Predict only rows that have at least one non-NaN feature
    available = [c for c in feature_names if c in df.columns]
    has_data  = df[available].notna().any(axis=1)

    df['svm_probability'] = np.nan
    df['svm_prediction']  = pd.array([pd.NA] * len(df), dtype='Int64')

    if has_data.any():
        X     = df.loc[has_data, available].values.astype(np.float64)
        probs = pipeline.predict_proba(X)[:, 1]
        preds = (probs >= threshold).astype(int)
        df.loc[has_data, 'svm_probability'] = np.round(probs, 4)
        df.loc[has_data, 'svm_prediction']  = preds

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Summary
    labeled = label_numeric.notna()
    n_agree = ((label_numeric == df['svm_prediction'].astype(float)) & labeled).sum()
    n_total = labeled.sum()
    n_fp    = ((label_numeric == 0) & (df['svm_prediction'] == 1)).sum()
    n_fn    = ((label_numeric == 1) & (df['svm_prediction'] == 0)).sum()

    print(f"\n  Prediction CSV saved:  {output_path}")
    print(f"  Rows written:          {len(df)}  "
          f"(labeled={n_total}, unlabeled={len(df)-n_total})")
    print(f"  Agreement with label:  {n_agree}/{n_total}  "
          f"(FP={n_fp}, FN={n_fn})")
    print(f"  New columns:  svm_set | svm_probability | svm_prediction")


def _md_auc_table(auc: dict, df: pd.DataFrame, feature_names: list[str]) -> list[str]:
    """One Markdown row per feature: AUC, distance from chance, direction, coverage."""
    if not auc:
        return ['_not computed (a set with a single class cannot be scored)_', '']
    out = ['| feature | AUC | \\|Δ0.5\\| | direction | coverage |',
           '|---|---:|---:|:--:|---:|']
    for f, a in sorted(auc.items(), key=lambda kv: -abs(kv[1] - 0.5)):
        cov = ''
        if f in df.columns:
            v = pd.to_numeric(df[f], errors='coerce')
            cov = f'{int(v.notna().sum())}/{len(df)} ({100 * v.notna().mean():.0f} %)'
        arrow = '↑' if a >= 0.5 else '↓'
        near = ' _~chance_' if abs(a - 0.5) < 0.05 else ''
        out.append(f'| `{f}` | {a:.3f} | {abs(a - 0.5):.3f} | {arrow} | {cov}{near} |')
    out.append('')
    return out


def _md_metrics_table(*named) -> list[str]:
    """Metrics side by side; `named` is (title, metrics-dict) pairs."""
    named = [(t, m) for t, m in named if m]
    if not named:
        return []
    heads = ' | '.join(t for t, _ in named)
    out = [f'| metric | {heads} |', '|---|' + '---:|' * len(named)]
    for key, lab in (('recall', 'Recall'), ('precision', 'Precision'),
                     ('specificity', 'Specificity'), ('f1', 'F1'),
                     ('auc', 'AUC-ROC'), ('accuracy', 'Accuracy')):
        vals = ' | '.join(f'{m[key]:.3f}' for _, m in named)
        out.append(f'| {lab} | {vals} |')
    cms = ' | '.join(f"TP {m['tp']} · FP {m['fp']} · FN {m['fn']} · TN {m['tn']}"
                     for _, m in named)
    out.append(f'| Confusion | {cms} |')
    out.append('')
    return out


#: Shipped alongside this module. Self-contained apart from Google Fonts, with a
#: single /*__DATA__*/ placeholder where the scores and labels go.
EXPLORER_TEMPLATE = Path(__file__).resolve().parent / 'roc_explorer_template.html'


def write_explorer(path: Path, cfg: dict, y_a, p_a, y_b, p_b) -> None:
    """
    Write the interactive threshold explorer: one self-contained HTML file
    carrying this run's actual scores and labels.

    The page recomputes every confusion matrix in the browser from y and p, so it
    cannot drift from the model it came with — and it needs no server, so it opens
    by double-clicking the file.

    y_a / p_a are the OUT-OF-FOLD probabilities, not in-sample ones. That matters:
    in-sample scores would draw a flattering curve that no held-out session could
    reproduce, and the page's whole purpose is choosing an operating point you can
    actually deploy at.
    """
    tpl = EXPLORER_TEMPLATE.read_text(encoding='utf-8')
    if '/*__DATA__*/' not in tpl:
        raise ValueError(f'{EXPLORER_TEMPLATE.name} has no /*__DATA__*/ placeholder')

    def _pack(y, probs, name, sessions):
        y = np.asarray(y).astype(int)
        probs = np.asarray(probs, dtype=float)
        return {'name': name,
                'auc': float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else float('nan'),
                'y': y.tolist(),
                'p': [round(float(v), 6) for v in probs],
                'sessions': int(sessions)}

    sets = {'A': _pack(y_a, p_a, 'Set A — cross-validated (out-of-fold)',
                       len(cfg['sessions_train']))}
    if y_b is not None and len(y_b):
        sets['B'] = _pack(y_b, p_b, 'Set B — held out', len(cfg['sessions_test']))

    data = {'model': {'kernel': cfg['kernel'],
                      'C': cfg.get('best_params', {}).get('C', '?'),
                      'gamma': cfg.get('best_params', {}).get('gamma', '—'),
                      'scale': cfg['scale'], 'nan_policy': cfg['nan_policy'],
                      'features': cfg['features'],
                      'source': Path(cfg['source_csv']).name},
            'sets': sets}

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tpl.replace('/*__DATA__*/',
                                json.dumps(data, separators=(',', ':'))),
                    encoding='utf-8')


def write_report(path: Path, cfg: dict, df_a, df_b, univariate_a: dict,
                 univariate_b: dict, results: dict, best_kernel: str) -> None:
    """
    Write the run as Markdown, from the same numbers the terminal printed.

    Built from returned data, never from captured stdout: a scraped report drifts
    silently the first time a print statement is reworded, and this is a document
    that will be read months later next to a model file.
    """
    L: list[str] = []
    A = L.append
    A(f"# SVM training report — {cfg['mode']}")
    A('')
    A(f"`{path.name}` · generated {cfg['trained_at']} · git `{cfg['git_sha'] or 'n/a'}`")
    A('')
    A(f"Model: `{cfg['model_path']}`")
    A('')

    A('## Configuration')
    A('')
    A('| setting | value |')
    A('|---|---|')
    for lab, key in (('Mode', 'mode'), ('Feature set', 'feature_set'),
                     ('NaN policy', 'nan_policy'), ('Scaler', 'scale'),
                     ('Grid scoring', 'scoring'), ('Imputer', 'impute'),
                     ('class_weight', 'class_weight'),
                     ('Noise pre-filter', 'noise_filter'),
                     ('Ambiguous (label 2)', 'ambiguous'),
                     ('Recall target', 'recall_target'),
                     ('Hard-negative weight', 'hard_negative_weight'),
                     ('Seed', 'seed')):
        A(f'| {lab} | `{cfg.get(key)}` |')
    A(f"| Source CSV | `{cfg['source_csv']}` |")
    A(f"| sklearn / numpy / pandas | {cfg['sklearn_version']} / "
      f"{cfg['numpy_version']} / {cfg['pandas_version']} |")
    A('')
    A(f"**Features ({len(cfg['features'])}):** "
      + ', '.join(f'`{f}`' for f in cfg['features']))
    if cfg.get('excluded_features'):
        A('')
        A(f"**Excluded:** " + ', '.join(f'`{f}`' for f in cfg['excluded_features']))
    A('')

    A('## Dataset')
    A('')
    A('| set | rows | clicks | noise | sessions |')
    A('|---|---:|---:|---:|---:|')
    A(f"| Set A (training) | {cfg['n_train']} | {cfg['n_train_clicks']} | "
      f"{cfg['n_train_noise']} | {len(cfg['sessions_train'])} |")
    if df_b is not None and len(df_b):
        A(f"| Set B (held out) | {cfg['n_test']} | {int((df_b['label'] == 1).sum())} | "
          f"{int((df_b['label'] == 0).sum())} | {len(cfg['sessions_test'])} |")
    A('')
    if cfg['sessions_test']:
        A(f"Held-out session(s): " + ', '.join(f'`{x}`' for x in cfg['sessions_test']))
        A('')
        n_click_b = int((df_b['label'] == 1).sum()) if df_b is not None else 0
        if n_click_b < 50:
            A(f"> ⚠️ Set B holds **{n_click_b} clicks**. Every Set B figure below "
              f"carries a wide interval at that count — treat the cross-validated "
              f"numbers as the decision-grade ones and Set B as corroboration.")
            A('')

    A('## Single-feature AUC-ROC')
    A('')
    A('How well each feature separates click from noise **on its own**, before any '
      'model sees it — model-free, kernel-free, and therefore comparable across '
      'runs and feature sets. Ranked by distance from 0.5: an AUC *below* 0.5 is '
      'inversely predictive and just as informative; only 0.5 itself is nothing.')
    A('')
    A('### Set A')
    A('')
    L.extend(_md_auc_table(univariate_a, df_a, cfg['features']))
    if univariate_b:
        A('### Set B')
        A('')
        L.extend(_md_auc_table(univariate_b, df_b, cfg['features']))
        drift = sorted(((f, univariate_a[f], univariate_b[f])
                        for f in univariate_a if f in univariate_b),
                       key=lambda t: -abs(t[1] - t[2]))
        if drift and abs(drift[0][1] - drift[0][2]) >= 0.15:
            A('> ⚠️ **Set A → Set B drift.** A feature that separates in training '
              'and not in the held-out session is a property of that session, not '
              'of clicks.')
            A('>')
            A('> | feature | Set A | Set B | Δ |')
            A('> |---|---:|---:|---:|')
            for f, a, b in drift[:5]:
                if abs(a - b) < 0.10:
                    break
                A(f'> | `{f}` | {a:.3f} | {b:.3f} | {b - a:+.3f} |')
            A('')

    for kernel, r in results.items():
        ch = r['chosen']
        A(f"## Kernel: {kernel}" + ('  ← selected' if kernel == best_kernel else ''))
        A('')
        A(f"Best params `{ch['params']}` · CV {cfg['scoring']} "
          f"{ch['cv_best_score']:.3f} · **CV AUC-ROC {r['auc']:.3f}** · "
          f"threshold **{r['threshold']:.3f}**")
        A('')
        A('The grid selects hyperparameters on '
          f"`{cfg['scoring']}`; the threshold is tuned separately afterwards, from "
          f"the out-of-fold ROC curve, to recall ≥ {cfg['recall_target']}.")
        A('')
        L.extend(_md_metrics_table(
            ('CV @ 0.50', ch.get('cv_at_0.5')),
            (f"CV @ {r['threshold']:.3f}", ch.get('cv_at_thr')),
            ('Set B', (r.get('set_b') or {}).get('metrics'))))

        if ch.get('importance'):
            is_linear = kernel == 'linear'
            A(f"### Feature importance ({'linear weights' if is_linear else 'permutation'})")
            A('')
            A('| # | feature | ' + ('weight | \\|weight\\| |' if is_linear
                                    else f"Δ{cfg['scoring']} | ± |"))
            A('|---:|---|---:|---:|')
            for i, (name, a, b) in enumerate(ch['importance'], 1):
                A(f'| {i} | `{name}` | {a:+.4f} | {b:.4f} |')
            A('')
            if is_linear:
                A('_Positive pushes toward click. Weights are in scaled space, so '
                  'they are comparable to each other but not to raw units._')
            else:
                A(f"_Drop in {cfg['scoring']} when the feature is shuffled. Near "
                  f"zero can mean unimportant **or** that a correlated feature "
                  f"carries the same information._")
            A('')

        sb = r.get('set_b')
        if sb and sb.get('permutation'):
            A('### Feature importance on Set B (permutation, held out)')
            A('')
            A(f"Δ{sb['perm_scoring']} when each feature is shuffled on the "
              f"**held-out** session(s), {sb['perm_n_repeats']} repeats. The Set A "
              f"table above says what the fitted model leans on; this says what "
              f"still carries signal where it has never looked. A feature that "
              f"ranks high there and near zero here is a property of the training "
              f"sessions, not of clicks.")
            A('')
            A(f"| # | feature | Δ{sb['perm_scoring']} | ± |")
            A('|---:|---|---:|---:|')
            for i, (name, mean, std) in enumerate(sb['permutation'], 1):
                A(f'| {i} | `{name}` | {mean:+.4f} | {std:.4f} |')
            A('')
            A(f"_Measured on {sb['n_rows']} rows, {sb['n_clicks']} clicks. "
              f"Noisy at that count — read the ordering, not the magnitudes._")
            A('')
            # rank movement between the two, which is the point of having both
            a_rank = {n: i for i, (n, *_ ) in enumerate(ch.get('importance') or [], 1)}
            b_rank = {n: i for i, (n, *_ ) in enumerate(sb['permutation'], 1)}
            moved = sorted(((n, a_rank[n], b_rank[n]) for n in b_rank if n in a_rank),
                           key=lambda t: -abs(t[1] - t[2]))
            if moved and abs(moved[0][1] - moved[0][2]) >= 3:
                A('> **Rank movement, Set A → Set B.** The largest disagreements '
                  'between what the model relies on and what generalises:')
                A('>')
                A('> | feature | Set A rank | Set B rank | move |')
                A('> |---|---:|---:|---:|')
                for n, ra, rb in moved[:4]:
                    if abs(ra - rb) < 2:
                        break
                    A(f'> | `{n}` | {ra} | {rb} | {ra - rb:+d} |')
                A('')

        if sb and sb['per_session']:
            A('### Set B, per session')
            A('')
            A('| session | clicks | detected | false positives | recall |')
            A('|---|---:|---:|---:|---:|')
            for row in sb['per_session']:
                A(f"| `{row['session']}` | {row['clicks']} | {row['detected']} | "
                  f"{row['fp']} | {row['recall']:.2f} |")
            A('')

    if len(results) > 1:
        A('## Kernel comparison')
        A('')
        A('| kernel | CV AUC-ROC | threshold | Set B AUC |')
        A('|---|---:|---:|---:|')
        for k, r in results.items():
            sb = (r.get('set_b') or {}).get('metrics', {})
            A(f"| {k}{' ←' if k == best_kernel else ''} | {r['auc']:.3f} | "
              f"{r['threshold']:.3f} | "
              f"{sb.get('auc', float('nan')):.3f} |")
        A('')

    A('## Reproducing this run')
    A('')
    A('```')
    A(cfg['command'])
    A('```')
    A('')
    A('---')
    A('')
    A('_Generated by `src/ml/train_svm.py`. Numbers are the ones printed to the '
      'terminal during the run, carried through as data rather than re-derived._')

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(L) + '\n', encoding='utf-8')


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Train PlantLeaf v5 SVM click classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--csv',           required=True,   type=Path,
                        help='Labeled features CSV')
    parser.add_argument('--set-b',         nargs='*',       default=[],
                        metavar='SESSION',
                        help='Session IDs to hold out as Set B test set')
    parser.add_argument('--set-b-from-column', action='store_true',
                        help="Take the A/B split from the CSV's `set` column "
                             "instead of --set-b. This is the format written by "
                             "scripts/v6/collect_training_set.py, where the split "
                             "is recorded in the data rather than re-typed here.")
    parser.add_argument('--v5', dest='mode', action='store_const',
                        const=MODE_V5, default=MODE_V6,
                        help="Train the v5 model: the 17 v5 features, the v5 noise "
                             "pre-filter ON, and the decay sentinels left as-is. "
                             "Default is v6. This sets all three together — a v5 run "
                             "with a v6 NaN policy is not a v5 run.")
    parser.add_argument('--feature-set', choices=sorted(FEATURE_SETS),
                        default=None, metavar='NAME',
                        help=f"Named feature preset: {', '.join(sorted(FEATURE_SETS))}. "
                             f"Defaults to the mode's set (v6 → v6-core, --v5 → v5).")
    parser.add_argument('--features', nargs='+', default=None, metavar='FEATURE',
                        help='Explicit feature list, overriding --feature-set. '
                             'Any column in the CSV is allowed.')
    parser.add_argument('--class-weight', choices=('none', 'balanced', 'auto'),
                        default='none',
                        help="'none' (default, = v5 behaviour); 'balanced' weights "
                             "inversely to class frequency (5.01x at 947:189); "
                             "'auto' puts both in the grid and lets CV choose.")
    parser.add_argument('--nan-policy', choices=('sentinel', 'nan'), default=None,
                        help="How to treat tau_ms/R2 on fit_valid==0 rows. 'nan' "
                             "converts the sentinels so the Pipeline imputes them "
                             "(v6 default); 'sentinel' leaves -1.0/0.0 (v5 default). "
                             "Stamped into the model and honoured at inference.")
    parser.add_argument('--hard-negative-weight', type=float, default=1.0,
                        metavar='W',
                        help="Weight for rows the CSV marks hard_negative=1 — noise "
                             "the previous model called a click. 1.0 (default) is off. "
                             "Try 2-3; these are 36 rows, so a large weight makes the "
                             "model fit a handful of examples. Needs a hard_negative "
                             "column (collect_training_set.py --v5-model).")
    parser.add_argument('--scale', choices=SCALERS, default=None,
                        help="Feature scaling. 'yeo-johnson' (v6 default) is a "
                             "built-in power transform, measured best on both "
                             "kernels; 'log10' transforms ratio-type features only "
                             "and is the more explainable choice; 'standard' is the "
                             "v5 default. ⚠️ 'robust' measured WORST (linear CV AUC "
                             "0.547) — its IQR divisor amplifies these tails.")
    parser.add_argument('--scoring', choices=SCORINGS, default=None,
                        help="Metric the hyperparameter grid selects on (NOT the "
                             "decision threshold, which --recall-target tunes "
                             "afterwards). 'roc_auc' is the v6 default; 'recall' is "
                             "the v5 default and interacts badly with "
                             "--class-weight balanced.")
    parser.add_argument('--impute', choices=('median', 'mean'), default='median',
                        help='Pipeline imputation strategy (default: median).')
    parser.add_argument('--kernels',       nargs='+',       default=['linear', 'rbf'],
                        choices=['linear', 'rbf'],
                        help='Kernels to train (default: linear rbf)')
    parser.add_argument('--recall-target', type=float,      default=0.90,
                        help='Target recall for threshold optimisation (default: 0.90)')
    parser.add_argument('--output',        type=Path,       default=Path('plantleaf_svm_v5.pkl'),
                        help='Output model path — must be a .pkl file (default: plantleaf_svm_v5.pkl)')
    parser.add_argument('--ambiguous', choices=('exclude', 'click', 'noise'),
                        default='exclude',
                        help="What to do with label=2 rows (the reviewer's "
                             "'could be either'). 'exclude' (default) keeps them out "
                             "of training; 'click'/'noise' fold them into that class. "
                             "The latter two are a SENSITIVITY CHECK: if recall moves "
                             "much between the three, the ambiguous set carries real "
                             "signal and wants a weighting scheme, not a hard call.")
    parser.add_argument('--no-noise-filter', dest='noise_filter',
                        action='store_false', default=None,
                        help='Skip the R²>0.1 / SPR<100 noise pre-filter '
                             '(already the v6 default).')
    parser.add_argument('--noise-filter', dest='noise_filter',
                        action='store_true', default=None,
                        help='Force the v5 noise pre-filter ON. ⚠️ It IS the fit gate '
                             'v6 removed from Stage 2: on the v6 corpus it deletes the '
                             'hardest negatives (1046 → 470 noise in a test run).')
    parser.add_argument('--predict-output',  type=Path,       default=None,
                        metavar='PATH',
                        help='If given, write the input CSV with two extra columns '
                             '(svm_probability, svm_prediction) to this path')
    parser.add_argument('--exclude-features', nargs='+', default=[], metavar='FEATURE',
                        help='Features to drop from the training vector (ablation). '
                             'Validated against the ACTIVE feature set — it used to be '
                             'choices=FEATURE_NAMES, which silently made every v6 '
                             'feature un-ablatable.')
    parser.add_argument('--threshold-metric', choices=THRESHOLD_METRICS,
                        default='recall-target',
                        help="How to pick the decision threshold ON the fitted "
                             "model - a separate choice from --scoring, which picks "
                             "the model itself. 'recall-target' (default) uses "
                             "--recall-target; 'f1' balances the two errors; "
                             "'youden' maximises tpr-fpr; 'precision-target' uses "
                             "--precision-target. Every option gives the SAME AUC.")
    parser.add_argument('--precision-target', type=float, default=0.50,
                        help='Target precision for --threshold-metric '
                             'precision-target (default: 0.50)')
    parser.add_argument('--report', action='store_true',
                        help='Also write a Markdown report next to the model, named '
                             'after it (foo.pkl -> foo_report.md). Implied by '
                             '--report-output.')
    parser.add_argument('--report-output', type=Path, default=None, metavar='PATH',
                        help='Write the Markdown report here instead of the default '
                             'location. Implies --report.')
    parser.add_argument('--seed',          type=int,        default=42)
    parser.add_argument('--cv-folds', type=int, default=5, metavar='K',
                        help='Cross-validation folds (default: 5). Lower it when '
                             'clicks are concentrated in too few sessions to spread '
                             'across 5 groups.')
    parser.add_argument('--min-fold-clicks', type=int, default=MIN_FOLD_POSITIVES,
                        metavar='N',
                        help=f"Minimum clicks required in EVERY validation fold "
                             f"(default: {MIN_FOLD_POSITIVES}). Sessions are the CV "
                             f"groups and many contain no clicks, so the stratifier "
                             f"can emit a click-free fold; with --scoring roc_auc "
                             f"that fold scores NaN, which propagates through the "
                             f"mean and makes GridSearchCV return the first "
                             f"candidate in grid order instead of the best. The "
                             f"split seed is walked forward until this floor is met.")
    args = parser.parse_args()

    # Catch the most common mistake: passing a directory instead of a .pkl file
    if args.output.is_dir() or args.output.suffix == '':
        print(f"ERROR: --output must be a file path ending in .pkl, not a directory.")
        print(f"  Got: {args.output}")
        print(f"  Example: --output src/ml/plantleaf_svm_v5.pkl")
        sys.exit(1)

    if args.set_b_from_column and args.set_b:
        print("ERROR: --set-b and --set-b-from-column both given. "
              "Pick one source of truth for the split.")
        sys.exit(1)

    # ── Resolve the mode into effective settings ─────────────────────────────
    # Each explicit flag wins over the mode default, so `--v5 --class-weight
    # balanced` or `--nan-policy sentinel` on a v6 run both work.
    defaults     = MODE_DEFAULTS[args.mode]
    nan_policy   = args.nan_policy if args.nan_policy is not None else defaults['nan_policy']
    noise_filter = args.noise_filter if args.noise_filter is not None else defaults['noise_filter']
    scale        = args.scale if args.scale is not None else defaults['scale']
    scoring      = args.scoring if args.scoring is not None else defaults['scoring']

    if args.features:
        base, set_name = list(args.features), '(explicit --features)'
    else:
        set_name = args.feature_set or defaults['feature_set']
        base     = list(FEATURE_SETS[set_name])

    unknown = [f for f in args.exclude_features if f not in base]
    if unknown:
        print(f"ERROR: --exclude-features names features that are not in the active set: "
              f"{unknown}")
        print(f"  Active set '{set_name}' ({len(base)}): {base}")
        sys.exit(1)

    feature_names = [f for f in base if f not in args.exclude_features]
    if not feature_names:
        print("ERROR: --exclude-features removed every feature. Nothing to train on.")
        sys.exit(1)

    # ── nan_policy x scale guard ─────────────────────────────────────────────
    # Refuse, do not warn: with sentinels still in place, yeo-johnson silently
    # fits a lambda contaminated by an artificial cluster at -1.0, and nothing the
    # trainer prints would reveal it. See check_nan_policy_vs_scale.
    unsafe = check_nan_policy_vs_scale(nan_policy, scale, feature_names)
    if unsafe:
        print(f"ERROR: --nan-policy sentinel cannot be combined with --scale {scale}.")
        print(f"  Affected column(s) in the active feature set: {unsafe}")
        print(f"  Under 'sentinel', a failed decay fit leaves tau_ms = -1.0 and "
              f"R2 = 0.0 in the data — 46.2 % of rows in the v6 training export.")
        if scale == 'log10':
            print(f"  log10 is undefined at -1.0, and clipping it would map -1.0 to "
                  f"-12.0: ten decades below the real log10(tau_ms) minimum "
                  f"of -1.796.")
        else:
            print(f"  {scale} will not error — it will silently fit its transform to "
                  f"that artificial cluster. Measured on tau_ms: lambda = -0.112 "
                  f"with the sentinels vs -2.412 without, a 2.30 shift.")
        print(f"\n  Fix with EITHER:")
        print(f"    --nan-policy nan       convert the sentinels (recommended)")
        print(f"    --scale standard       keep the sentinels, skip the transform")
        sys.exit(1)

    print(f"PlantLeaf {args.mode} — SVM Training")
    print(f"  CSV            :  {args.csv}")
    print(f"  Mode           :  {args.mode}"
          f"{'  (--v5: legacy features, noise filter, sentinels)' if args.mode == MODE_V5 else ''}")
    print(f"  Feature set    :  {set_name}  ({len(feature_names)} features)")
    print(f"  Features       :  {feature_names}")
    print(f"  NaN policy     :  {nan_policy}")
    print(f"  Scaler         :  {scale}")
    print(f"  Grid scoring   :  {scoring}   (picks the MODEL)")
    print(f"  Threshold      :  {args.threshold_metric}"
          + (f" >= {args.recall_target}" if args.threshold_metric == 'recall-target'
             else f" >= {args.precision_target}"
                  if args.threshold_metric == 'precision-target' else '')
          + "   (picks the OPERATING POINT)")
    print(f"  class_weight   :  {args.class_weight}")
    print(f"  Imputer        :  {args.impute}")
    print(f"  Kernels        :  {args.kernels}")
    print(f"  Recall target  :  {args.recall_target}")
    print(f"  Set B sessions :  "
          f"{'(from `set` column)' if args.set_b_from_column else (args.set_b or '(none)')}")
    print(f"  Ambiguous      :  label=2 rows -> {args.ambiguous}")
    print(f"  Noise filter   :  "
          f"{f'R²>{NOISE_FILTER_R2_MIN}, SPR<{NOISE_FILTER_SPR_MAX}' if noise_filter else 'OFF'}")
    if args.exclude_features:
        print(f"  Excluded       :  {args.exclude_features}  "
              f"({len(feature_names)}/{len(base)} features active)")
    print(f"  Seed           :  {args.seed}")
    print()

    df_a, df_b = load_and_prepare(
        args.csv,
        set_b_sessions=args.set_b,
        noise_filter=noise_filter,
        ambiguous=args.ambiguous,
        set_b_from_column=args.set_b_from_column,
        feature_names=feature_names,
        nan_policy=nan_policy,
    )

    # Validate minimum data requirements
    n_clicks_a = (df_a['label'] == 1).sum()
    n_noise_a  = (df_a['label'] == 0).sum()
    n_sessions = df_a['session_id'].nunique()

    if n_clicks_a < 10:
        print(f"\nERROR: Set A has only {n_clicks_a} click samples. Need at least 10 to train.")
        sys.exit(1)
    if n_sessions < 2:
        print(f"\nWarning: Set A has only {n_sessions} session(s). "
              f"StratifiedGroupKFold requires at least 5 — results may be unreliable.")
    if n_clicks_a < 50:
        print(f"\nWarning: {n_clicks_a} clicks is below the recommended 50 minimum for "
              f"a linear kernel (see SVM_TRAINING_DATA_GUIDE.md §2.2).")
    if n_clicks_a / max(n_noise_a, 1) < 1 / 3:
        print(f"\nWarning: class ratio 1:{n_noise_a // max(n_clicks_a, 1)} exceeds the "
              f"recommended 1:3 maximum — consider reducing noise samples.")

    # ── log10 positivity, checked on the real data ───────────────────────────
    # The flag guard above catches the KNOWN sentinel combination from the flags
    # alone. This catches everything else: a column added to LOG_COLUMNS that
    # turns out to reach zero, or a CSV whose values are not what the policy
    # assumed. Loud, before any fitting, rather than -inf flowing into the scaler.
    if scale == 'log10':
        try:
            assert_log_safe(df_a, feature_names, 'Set A')
            if df_b is not None and len(df_b):
                assert_log_safe(df_b, feature_names, 'Set B')
        except ValueError as exc:
            print(f"\nERROR: {exc}")
            sys.exit(1)

    # ── Hard-negative upweighting ────────────────────────────────────────────
    # These are label-0 rows the PREVIOUS model scored as clicks: the negatives
    # the last generation could not separate, and so the ones worth more than
    # another easy row. An upweight, not a separate pool — there are only 36 of
    # them, and a large weight would have the model fitting a handful of points.
    #
    # This composes with class_weight rather than replacing it: sklearn multiplies
    # sample_weight by the class weight, so --class-weight balanced already
    # weights all 947 negatives and this then re-weights 36 of them.
    sample_weight = None
    if args.hard_negative_weight != 1.0:
        if 'hard_negative' not in df_a.columns:
            print("\nERROR: --hard-negative-weight needs a hard_negative column.")
            print("  Produce one with: collect_training_set.py --v5-model MODEL.pkl")
            sys.exit(1)
        hn = pd.to_numeric(df_a['hard_negative'], errors='coerce').fillna(0) == 1
        sample_weight = np.where(hn, args.hard_negative_weight, 1.0).astype(np.float64)
        print(f"\nHard negatives:  {int(hn.sum())} row(s) at weight "
              f"{args.hard_negative_weight}")
        if int(hn.sum()) == 0:
            print("  ⚠️  none found — the column is present but all zero; "
                  "was --v5-model passed to the collector?")

    # Model-free feature screen. Printed once, before any kernel, because it does
    # not depend on kernel or hyperparameters — and printed for BOTH sets, since a
    # feature that separates on Set A and not on Set B is a session artefact.
    univariate_a = print_univariate_auc(df_a, feature_names, 'Set A')
    if df_b is not None and len(df_b) and df_b['label'].nunique() > 1:
        univariate_b = print_univariate_auc(df_b, feature_names, 'Set B')
        drift = sorted(
            ((f, univariate_a[f], univariate_b[f]) for f in univariate_a
             if f in univariate_b),
            key=lambda t: -abs(t[1] - t[2]))
        if drift and abs(drift[0][1] - drift[0][2]) >= 0.15:
            print(f"\n  ⚠️  Largest Set A → Set B shift in single-feature AUC:")
            for f, a, b in drift[:3]:
                print(f"        {f:<24} {a:.3f} → {b:.3f}   (Δ {b - a:+.3f})")
            print(f"      A feature that separates in training and not in the "
                  f"held-out session is a session artefact, not a click property.")
    else:
        univariate_b = {}

    # Train all requested kernels
    results: dict = {}
    for kernel in args.kernels:
        pipeline, threshold, auc_cv, chosen = train_kernel(
            df_a, kernel, args.seed, args.recall_target, feature_names,
            class_weight=args.class_weight, impute=args.impute,
            sample_weight=sample_weight, scale=scale, scoring=scoring,
            threshold_metric=args.threshold_metric,
            precision_target=args.precision_target,
            cv_folds=args.cv_folds, min_fold_clicks=args.min_fold_clicks,
        )
        results[kernel] = {
            'pipeline' : pipeline,
            'threshold': threshold,
            'auc'      : auc_cv,
            'chosen'   : chosen,
        }
        if df_b is not None and len(df_b) > 0:
            results[kernel]['set_b'] = evaluate_on_set_b(
                df_b, pipeline, threshold, feature_names,
                scoring=scoring, seed=args.seed)

    # Summary and best model selection
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  {'Kernel':<10}  CV AUC-ROC  Threshold")
    for kernel, res in results.items():
        print(f"  {kernel:<10}  {res['auc']:.3f}       {res['threshold']:.3f}")

    best_kernel = max(results, key=lambda k: results[k]['auc'])
    best        = results[best_kernel]
    print(f"\n  Best kernel: {best_kernel}  (CV AUC-ROC = {best['auc']:.3f})")

    if len(results) > 1:
        kernels = list(results.keys())
        diff = abs(results[kernels[0]]['auc'] - results[kernels[1]]['auc'])
        if diff < 0.02:
            print(f"  Note: AUC difference is {diff:.3f} — models are comparable.")
            print(f"  Prefer 'linear' for interpretability if performance is similar.")

    # Save best model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # ── Provenance ───────────────────────────────────────────────────────────
    # A model is only reproducible if it records what produced it. 'nan_policy'
    # in particular is not documentation: click_pipeline_v5.run_stage3 READS it to
    # decide whether to convert sentinels before scoring, so a model saved without
    # it is treated as v5-era (sentinels) at inference.
    save_dict = {
        'pipeline'   : best['pipeline'],
        'threshold'  : best['threshold'],
        'kernel'     : best_kernel,
        'features'   : feature_names,
        'all_results': {k: {'threshold': v['threshold'], 'auc': v['auc'],
                            'chosen': {kk: vv for kk, vv in v['chosen'].items()
                                       if kk != 'probs_oof'}}
                        for k, v in results.items()},
        # ── provenance ──
        'mode'            : args.mode,
        'nan_policy'      : nan_policy,
        'feature_set'     : set_name,
        'excluded_features': list(args.exclude_features),
        'class_weight'    : args.class_weight,
        'class_weight_chosen': best['chosen'].get('class_weight'),
        'impute'          : args.impute,
        'scale'           : scale,
        'scoring'         : scoring,
        'univariate_auc'  : {'set_a': univariate_a, 'set_b': univariate_b},
        'threshold_metric': args.threshold_metric,
        'precision_target': args.precision_target,
        'add_indicator'   : best['chosen'].get('add_indicator', False),
        'noise_filter'    : noise_filter,
        'hard_negative_weight': args.hard_negative_weight,
        'ambiguous'       : args.ambiguous,
        'recall_target'   : args.recall_target,
        'seed'            : args.seed,
        'trained_at'      : _dt.datetime.now().isoformat(timespec='seconds'),
        'source_csv'      : str(args.csv),
        'n_train'         : int(len(df_a)),
        'n_train_clicks'  : int((df_a['label'] == 1).sum()),
        'n_train_noise'   : int((df_a['label'] == 0).sum()),
        'n_test'          : int(len(df_b)) if df_b is not None else 0,
        'sessions_train'  : sorted(df_a['session_id'].astype(str).unique().tolist()),
        'sessions_test'   : (sorted(df_b['session_id'].astype(str).unique().tolist())
                             if df_b is not None else []),
        'sklearn_version' : sklearn.__version__,
        'numpy_version'   : np.__version__,
        'pandas_version'  : pd.__version__,
        'git_sha'         : _git_sha(),
    }
    joblib.dump(save_dict, args.output)
    print(f"\n  Model saved:  {args.output}")

    # ── Markdown report ──────────────────────────────────────────────────────
    # Default sits beside the .pkl and is named after it, so a model and the
    # report explaining it never drift apart in a folder.
    if args.report or args.report_output is not None:
        report_path = (args.report_output if args.report_output is not None
                       else args.output.with_name(args.output.stem + '_report.md'))
        cfg = dict(save_dict)
        cfg.pop('pipeline', None)
        cfg['model_path'] = str(args.output)
        cfg['command'] = ' '.join(shlex.quote(a) for a in sys.argv)
        try:
            write_report(report_path, cfg, df_a, df_b, univariate_a, univariate_b,
                         results, best_kernel)
            print(f"  Report saved: {report_path}")
        except Exception as exc:                                # noqa: BLE001
            print(f"  ⚠️  Report could not be written: {type(exc).__name__}: {exc}")
            print(f"      The model itself saved fine at {args.output}")

        # The interactive threshold explorer, beside the Markdown. Separate
        # try/except: a failure here must not cost the report, and vice versa.
        explorer_path = report_path.with_name(
            report_path.stem.replace('_report', '') + '_explorer.html')
        try:
            sb = (results[best_kernel].get('set_b') or {})
            ecfg = dict(cfg)
            ecfg['best_params'] = results[best_kernel]['chosen']['params']
            write_explorer(explorer_path, ecfg,
                           df_a['label'].values,
                           results[best_kernel]['chosen']['probs_oof'],
                           sb.get('y'), sb.get('probs'))
            print(f"  Explorer:     {explorer_path}   (open it in a browser)")
        except Exception as exc:                                # noqa: BLE001
            # Neither a report nor an explorer failure may cost a trained model
            # that is already on disk — say what broke and carry on.
            print(f"  ⚠️  Explorer could not be written: "
                  f"{type(exc).__name__}: {exc}")
            print(f"      The model and report are unaffected.")

    if args.predict_output:
        save_prediction_csv(
            args.csv,
            best['pipeline'],
            best['threshold'],
            feature_names,
            args.set_b or [],
            args.predict_output,
        )

    print(f"\n  Inference usage:")
    print(f"    import joblib, numpy as np")
    print(f"    model = joblib.load('{args.output}')")
    print(f"    pipe  = model['pipeline']")
    print(f"    thr   = model['threshold']   # {best['threshold']:.3f}")
    print(f"    # X: (n, {len(feature_names)}) array, columns = model['features']")
    print(f"    proba = pipe.predict_proba(X)[:, 1]")
    print(f"    pred  = (proba >= thr).astype(int)  # 1=click, 0=noise")


if __name__ == '__main__':
    main()
