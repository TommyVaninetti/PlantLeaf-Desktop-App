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
train_svm.py — PlantLeaf v5 SVM classifier training
=====================================================

Trains a click/noise SVM classifier on the labeled CSV produced by
data_collection_dialog_v5.py. Implements the full protocol from
SVM_TRAINING_DATA_GUIDE.md: noise pre-filtering, Set A / Set B split,
StratifiedGroupKFold + GridSearchCV, threshold optimisation from the ROC
curve, and feature importance reporting.

Usage (run from the project root):
    python scripts/train_svm.py --csv labeled_data.csv [options]

Basic example:
    python src/ml/train_svm.py --csv features_export.csv \\
        --set-b aloe_water_session_1 ferrocactus_outdoor_rec2

Full example:
    python src/ml/train_svm.py --csv labeled.csv \\
        --set-b aloe_water_session_1 \\
        --kernels linear rbf \\
        --recall-target 0.92 \\
        --exclude-features fit_coverage \\
        --output src/ml/plantleaf_svm_v5.pkl

Options:
    --csv PATH              Labeled CSV from data_collection_dialog_v5.py (required)
    --set-b SESSION [...]   Session IDs (file stems) to hold out as Set B
                            If omitted: cross-validate only, no held-out test.
    --kernels KERNEL [...]  Kernels to train: linear, rbf (default: both)
    --recall-target FLOAT   Target recall for threshold optimisation (default: 0.90)
    --output PATH           Output model path (default: plantleaf_svm_v5.pkl)
    --no-noise-filter       Skip R²>0.1 and SPR<100 pre-filter for noise samples
    --predict-output PATH   Write the input CSV + svm_probability + svm_prediction
                            columns to PATH (all rows, including unlabeled ones)
    --exclude-features F [F ...]  Features to drop from the training vector.
                            Useful for diagnosing which features help or hurt.
                            Example: --exclude-features fit_coverage centroid_shift_hz
    --seed INT              Random seed (default: 42)

Output:
    A .pkl file containing a dict with keys:
        'pipeline'   — fitted sklearn Pipeline (StandardScaler + SVM)
        'threshold'  — optimal decision threshold
        'kernel'     — 'linear' or 'rbf'
        'features'   — ordered list of the 17 feature names
        'all_results'— dict with threshold and AUC for every trained kernel

    Inference:
        import joblib
        model = joblib.load('plantleaf_svm_v5.pkl')
        pipe  = model['pipeline']
        thr   = model['threshold']
        # X: numpy array, shape (n_candidates, 17), columns in FEATURE_NAMES order
        proba = pipe.predict_proba(X)[:, 1]
        pred  = (proba >= thr).astype(int)   # 1 = click, 0 = noise
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np

import pandas as pd

import joblib

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
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


# ── Constants ─────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    'peak_SNR', 'pre_SNR', 'post_SNR',
    'rise_time_ms', 'fall_time_ms', 'asymmetry_integral',
    'ZCR_pre', 'ZCR_click', 'ZCR_post',
    'kurtosis', 'centroid_shift_hz',
    'tau_ms', 'R2', 'fit_coverage',
    'SPR', 'R_spectral', 'FPE_hz',
]

# Noise sample pre-filtering gates (SVM_TRAINING_DATA_GUIDE.md §3.3)
# Applied to label=0 rows only. Clicks (label=1) are always kept.
NOISE_FILTER_R2_MIN  = 0.10
NOISE_FILTER_SPR_MAX = 100.0


# ── Data loading and preparation ──────────────────────────────────────────────

def load_and_prepare(
    csv_path:       Path,
    set_b_sessions: list[str],
    noise_filter:   bool,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Load the labeled CSV, apply noise pre-filtering, and split Set A / Set B.

    Returns (df_a, df_b) where df_b is None if set_b_sessions is empty.
    """
    df = pd.read_csv(csv_path)

    # Verify required columns
    missing = [c for c in FEATURE_NAMES + ['label', 'session_id'] if c not in df.columns]
    if missing:
        print(f"ERROR: CSV is missing columns: {missing}")
        sys.exit(1)

    # Defensive numeric coercion. Hand-editing the CSV in Excel under an Italian
    # locale can introduce decimal commas ("12,73") and stray whitespace, which
    # make pandas read feature columns as strings and crash on .astype(float).
    # Repair them here so training works regardless of how the CSV was produced.
    for col in FEATURE_NAMES:
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

    n_clicks = (df['label'] == 1).sum()
    n_noise  = (df['label'] == 0).sum()
    print(f"Labeled rows loaded:  {len(df)}")
    print(f"  label=1 (clicks) :  {n_clicks}")
    print(f"  label=0 (noise)  :  {n_noise}")
    print(f"  Sessions         :  {df['session_id'].nunique()}")

    # Apply noise pre-filtering gates (label=0 rows only)
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

    # Split Set A (training) / Set B (held-out test)
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


# ── Metrics helper ────────────────────────────────────────────────────────────

def _print_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> None:
    """Print recall, precision, specificity, F1, AUC-ROC, accuracy, confusion matrix."""
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


# ── Single-kernel training ────────────────────────────────────────────────────

def train_kernel(
    df_a:          pd.DataFrame,
    kernel:        str,
    seed:          int,
    recall_target: float,
    feature_names: list[str],
) -> tuple[Pipeline, float, float]:
    """
    Train one SVM kernel with StratifiedGroupKFold + GridSearchCV.

    Returns:
        pipeline        — fitted Pipeline (imputer + scaler + SVM), best params, on ALL Set A
        threshold       — optimal decision threshold from ROC curve
        auc_cv          — cross-validated AUC-ROC (for model comparison)
    """
    print(f"\n{'='*60}")
    print(f"  Kernel: {kernel.upper()}")
    print(f"{'='*60}")
    print(f"  Features used ({len(feature_names)}): {feature_names}")

    X      = df_a[feature_names].values.astype(np.float64)
    y      = df_a['label'].values.astype(int)
    groups = df_a['session_id'].values

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),   # safety net for any NaN
        ('scaler',  StandardScaler()),
        ('svm',     SVC(kernel=kernel, probability=True, random_state=seed)),
    ])

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

    if kernel == 'linear':
        param_grid = {'svm__C': [0.1, 1, 5, 10, 50]}
    else:
        param_grid = {
            'svm__C':     [0.1, 1, 5, 10, 50],
            'svm__gamma': ['scale', 'auto', 0.01, 0.1],
        }

    n_combos = 1
    for v in param_grid.values():
        n_combos *= len(v)
    print(f"  GridSearchCV: {n_combos} combinations × 5 folds = {n_combos * 5} fits")
    print(f"  Scoring: recall (primary metric)  |  Groups: session_id")

    # Use make_scorer with zero_division=0 so folds whose validation set
    # contains no click samples (noise-only sessions) cleanly return 0
    # instead of triggering UndefinedMetricWarning.
    recall_scorer = make_scorer(recall_score, zero_division=0)

    grid = GridSearchCV(
        pipe, param_grid,
        cv=cv, scoring=recall_scorer,
        refit=True, n_jobs=-1, verbose=0,
    )
    grid.fit(X, y, groups=groups)

    best_params_clean = {k.replace('svm__', ''): v for k, v in grid.best_params_.items()}
    print(f"\n  Best params    :  {best_params_clean}")
    print(f"  Best CV recall :  {grid.best_score_:.3f}")

    # ── Out-of-fold probabilities for threshold optimisation ──────────────────
    # clone of best_estimator_ is re-fitted per fold — correct for OOF probs.
    print(f"  Computing out-of-fold probabilities (5 more fits)...")
    cv_oof = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    probs_oof = cross_val_predict(
        grid.best_estimator_, X, y,
        cv=cv_oof, groups=groups,
        method='predict_proba',
    )[:, 1]

    auc_cv = roc_auc_score(y, probs_oof) if len(np.unique(y)) > 1 else float('nan')

    # CV metrics at default threshold
    preds_default = (probs_oof >= 0.5).astype(int)
    print(f"\n  Cross-validated metrics  (threshold = 0.50):")
    _print_metrics(y, preds_default, probs_oof)

    # ── Optimal threshold from ROC curve ─────────────────────────────────────
    fpr_arr, tpr_arr, thresholds = roc_curve(y, probs_oof)

    # Find lowest FPR where recall >= recall_target
    valid = tpr_arr >= recall_target
    if valid.any():
        fpr_valid = fpr_arr[valid]
        thr_valid = thresholds[valid]
        opt_thr   = float(thr_valid[fpr_valid.argmin()])
    else:
        opt_thr = 0.5
        print(f"\n  Warning: recall target {recall_target} not reachable — using threshold=0.5")

    preds_opt = (probs_oof >= opt_thr).astype(int)
    print(f"\n  Cross-validated metrics  (threshold = {opt_thr:.3f}, target recall ≥ {recall_target}):")
    _print_metrics(y, preds_opt, probs_oof)

    # ── Feature importance ────────────────────────────────────────────────────
    print()
    _print_feature_importance(grid.best_estimator_, kernel, X, y, groups, seed, feature_names)

    return grid.best_estimator_, opt_thr, auc_cv


def _print_feature_importance(
    pipeline:      Pipeline,
    kernel:        str,
    X:             np.ndarray,
    y:             np.ndarray,
    groups:        np.ndarray,
    seed:          int,
    feature_names: list[str],
) -> None:
    """
    Feature importance:
      linear  → weight vector (interpretable in scaled space, directly comparable)
      rbf     → permutation importance on Set A with cross-validation
                (note: use Set B permutation importance for the final published result)
    """
    print(f"  Feature importance ({kernel}):")
    svm_model = pipeline.named_steps['svm']

    if kernel == 'linear' and hasattr(svm_model, 'coef_'):
        weights    = svm_model.coef_[0]
        importance = np.abs(weights)
        order      = np.argsort(importance)[::-1]
        for rank, idx in enumerate(order, 1):
            direction = '↑' if weights[idx] > 0 else '↓'
            bar = '█' * int(importance[idx] / importance[order[0]] * 20)
            print(f"    {rank:2d}. {feature_names[idx]:<22}  {direction}{importance[idx]:.4f}  {bar}")
        print(f"    (↑ = pushes toward click,  ↓ = pushes toward noise)")

    else:
        # RBF: permutation importance on Set A
        # This is an approximation — use Set B if available for the published result.
        print(f"    Computing permutation importance on Set A (n_repeats=15)...")
        result = permutation_importance(
            pipeline, X, y,
            n_repeats=15, random_state=seed, scoring='recall',
        )
        order = np.argsort(result.importances_mean)[::-1]
        for rank, idx in enumerate(order, 1):
            mean = result.importances_mean[idx]
            std  = result.importances_std[idx]
            bar  = '█' * max(0, int(mean / (result.importances_mean[order[0]] + 1e-9) * 20))
            print(f"    {rank:2d}. {feature_names[idx]:<22}  Δrecall={mean:+.3f} ± {std:.3f}  {bar}")
        print(f"    (Δrecall: drop in recall when feature is shuffled; larger = more important)")
        print(f"    Note: re-run permutation_importance() on Set B for the published result.")


# ── Set B evaluation ──────────────────────────────────────────────────────────

def evaluate_on_set_b(
    df_b:          pd.DataFrame,
    pipeline:      Pipeline,
    thr:           float,
    feature_names: list[str],
) -> None:
    """Evaluate the final model on the held-out Set B test set."""
    print(f"\n{'='*60}")
    print(f"  Set B — held-out test set evaluation")
    print(f"{'='*60}")

    X_b = df_b[feature_names].values.astype(np.float64)
    y_b = df_b['label'].values.astype(int)

    probs_b = pipeline.predict_proba(X_b)[:, 1]
    preds_b = (probs_b >= thr).astype(int)

    print(f"  Threshold used:  {thr:.3f}")
    _print_metrics(y_b, preds_b, probs_b)

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
    parser.add_argument('--kernels',       nargs='+',       default=['linear', 'rbf'],
                        choices=['linear', 'rbf'],
                        help='Kernels to train (default: linear rbf)')
    parser.add_argument('--recall-target', type=float,      default=0.90,
                        help='Target recall for threshold optimisation (default: 0.90)')
    parser.add_argument('--output',        type=Path,       default=Path('plantleaf_svm_v5.pkl'),
                        help='Output model path — must be a .pkl file (default: plantleaf_svm_v5.pkl)')
    parser.add_argument('--no-noise-filter', action='store_true',
                        help='Skip R²>0.1 and SPR<100 noise pre-filter')
    parser.add_argument('--predict-output',  type=Path,       default=None,
                        metavar='PATH',
                        help='If given, write the input CSV with two extra columns '
                             '(svm_probability, svm_prediction) to this path')
    parser.add_argument('--exclude-features', nargs='+', default=[], metavar='FEATURE',
                        choices=FEATURE_NAMES,
                        help='Features to drop from the training vector '
                             '(useful for ablation / diagnostic runs)')
    parser.add_argument('--seed',          type=int,        default=42)
    args = parser.parse_args()

    # Catch the most common mistake: passing a directory instead of a .pkl file
    if args.output.is_dir() or args.output.suffix == '':
        print(f"ERROR: --output must be a file path ending in .pkl, not a directory.")
        print(f"  Got: {args.output}")
        print(f"  Example: --output src/ml/plantleaf_svm_v5.pkl")
        sys.exit(1)

    feature_names = [f for f in FEATURE_NAMES if f not in args.exclude_features]
    if not feature_names:
        print("ERROR: --exclude-features removed every feature. Nothing to train on.")
        sys.exit(1)

    print(f"PlantLeaf v5 — SVM Training")
    print(f"  CSV            :  {args.csv}")
    print(f"  Kernels        :  {args.kernels}")
    print(f"  Recall target  :  {args.recall_target}")
    print(f"  Set B sessions :  {args.set_b or '(none)'}")
    print(f"  Noise filter   :  {'OFF' if args.no_noise_filter else f'R²>{NOISE_FILTER_R2_MIN}, SPR<{NOISE_FILTER_SPR_MAX}'}")
    if args.exclude_features:
        print(f"  Excluded       :  {args.exclude_features}  ({len(feature_names)}/17 features active)")
    print(f"  Seed           :  {args.seed}")
    print()

    df_a, df_b = load_and_prepare(
        args.csv,
        set_b_sessions=args.set_b,
        noise_filter=not args.no_noise_filter,
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

    # Train all requested kernels
    results: dict = {}
    for kernel in args.kernels:
        pipeline, threshold, auc_cv = train_kernel(
            df_a, kernel, args.seed, args.recall_target, feature_names
        )
        results[kernel] = {
            'pipeline' : pipeline,
            'threshold': threshold,
            'auc'      : auc_cv,
        }
        if df_b is not None and len(df_b) > 0:
            evaluate_on_set_b(df_b, pipeline, threshold, feature_names)

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
    save_dict = {
        'pipeline'   : best['pipeline'],
        'threshold'  : best['threshold'],
        'kernel'     : best_kernel,
        'features'   : feature_names,
        'all_results': {k: {'threshold': v['threshold'], 'auc': v['auc']}
                        for k, v in results.items()},
    }
    joblib.dump(save_dict, args.output)
    print(f"\n  Model saved:  {args.output}")

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
    print(f"    # X: (n, 17) array, columns = model['features']")
    print(f"    proba = pipe.predict_proba(X)[:, 1]")
    print(f"    pred  = (proba >= thr).astype(int)  # 1=click, 0=noise")


if __name__ == '__main__':
    main()
