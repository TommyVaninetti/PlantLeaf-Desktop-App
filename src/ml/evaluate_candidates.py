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
evaluate_candidates.py  —  PlantLeaf v5  Stage 2 / 3 / 4 evaluation
=====================================================================

Runs Stages 2, 3, and 4 of the click detection pipeline on one or more
feature CSVs produced by DataCollectionDialogV5 (the "_candidates" files).

Every input row is kept in the output. Three columns are appended:

    svm_probability  —  model confidence score (0.000–1.000).
                        NaN if the row was blocked in Stage 2 before
                        reaching the SVM.

    svm_prediction   —  binary classification at the model threshold.
                        NaN if blocked in Stage 2, 0 if rejected by the
                        SVM, 1 if confirmed as a click by Stage 3.

    stage_blocked    —  which stage first rejected this row, or empty
                        string if the row survived all stages:

        ""             →  final confirmed click detection
        "Stage2_R2"    →  R² < 0.10 (invalid exponential fit; τ and all
                          decay-window features are unreliable)
        "Stage2_SPR"   →  SPR ≥ 100 (extremely tonal; out-of-distribution
                          for the SVM — never seen in training)
        "Stage3_SVM"   →  SVM predicted noise (svm_probability < threshold)
        "Stage4_dedup" →  passed the SVM but removed by deduplication:
                          another detection from the same physical click in
                          the same recording had higher svm_probability

Usage:
    # Two individual files:
    python src/ml/evaluate_candidates.py \\
        --model src/ml/plantleaf_svm_v5.pkl \\
        rec1_candidates.csv rec2_candidates.csv \\
        --output evaluated.csv

    # Whole folder — finds every *.csv inside automatically:
    python src/ml/evaluate_candidates.py \\
        --model src/ml/plantleaf_svm_v5.pkl \\
        path/to/MECHANIC/ \\
        --output evaluated.csv

    # Mix of files and folders:
    python src/ml/evaluate_candidates.py \\
        --model src/ml/plantleaf_svm_v5.pkl \\
        path/to/MECHANIC/ extra_file.csv \\
        --output evaluated.csv

Options:
    --model PATH    Trained SVM model (.pkl from train_svm.py)  [required]
    --output PATH   Output CSV path (default: evaluated_candidates.csv)

Notes:
    Blank rows (Excel artifacts with no frame_idx) and trailing Unnamed columns
    are silently dropped per file, with a warning printed to stdout.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib


# ── Stage constants ──────────────────────────────────────────────────────────
# Imported from the pipeline rather than re-declared, so this script and the
# in-app detector can never drift apart on the thresholds.
#
# src/core/ is put on the path and the module imported *directly*, rather than as
# `core.click_pipeline_v5`: importing it through the package would execute
# src/core/__init__.py, which pulls in the entire PySide6 window stack. This
# script must stay runnable without a GUI. click_pipeline_v5 itself only imports
# numpy, so loading it standalone is safe.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'core')) #KEEP IT THIS WAY, IT'S NEEDED TO IMPORT WITH NO GUI COMPONENTS
from click_pipeline_v5 import (   # noqa: E402
    STAGE2_R2_MIN       as _R2_MIN,        # Minimum R² for a valid exponential fit (Stage 2)
    STAGE2_TAU_MIN      as _TAU_MIN,       # τ must be > 0, else no decay was fitted (Stage 2)
    STAGE2_SPR_MAX      as _SPR_MAX,       # Max SPR before a candidate is out-of-distribution (Stage 2)
    DEDUP_WINDOW_FRAMES as _DEDUP_FRAMES,  # Frame-index proximity window for deduplication (Stage 4)
)

# All 17 feature names in the order computed by compute_features_v5.
# fit_coverage is intentionally excluded from the SVM (see train_svm.py
# --exclude-features). The model's 'features' key is the authoritative list
# used at inference; this constant is here only for column validation.
_ALL_FEATURES = [
    'peak_SNR', 'pre_SNR', 'post_SNR',
    'rise_time_ms', 'fall_time_ms', 'asymmetry_integral',
    'ZCR_pre', 'ZCR_click', 'ZCR_post',
    'kurtosis', 'centroid_shift_hz',
    'tau_ms', 'R2', 'fit_coverage',
    'SPR', 'R_spectral', 'FPE_hz',
]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_csvs(paths: list[Path]) -> pd.DataFrame:
    """
    Load and concatenate one or more candidate CSVs.

    Applies defensive coercion for Italian-locale decimal commas and
    whitespace-padded label values, so the script handles both pipeline-
    generated CSVs (clean floats) and hand-edited Excel exports equally.

    Silently cleans two common Excel artifacts per file:
      • Trailing 'Unnamed: N' columns written by Excel when the sheet has a
        stray non-empty cell beyond the data range.
      • Blank rows (no valid frame_idx) produced when Excel inserts an empty
        row at the end of a saved CSV.
    """
    frames = []
    for p in paths:
        if not p.exists():
            print(f"  Warning: {p} not found — skipping")
            continue
        df = pd.read_csv(p)

        # Drop trailing Unnamed columns (Excel artifact: stray cells beyond
        # the data range cause pandas to create 'Unnamed: 24', etc.)
        unnamed_cols = [c for c in df.columns if str(c).startswith('Unnamed:')]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
            print(f"    Note: dropped {len(unnamed_cols)} Unnamed column(s) from {p.name}")

        # Drop blank rows — rows with no frame_idx are Excel padding rows that
        # carry no data. frame_idx is mandatory for every valid candidate.
        n_before = len(df)
        if 'frame_idx' in df.columns:
            df = df[pd.to_numeric(df['frame_idx'], errors='coerce').notna()].copy()
        n_blank = n_before - len(df)
        if n_blank:
            print(f"    Note: dropped {n_blank} blank row(s) from {p.name}")

        if df.empty:
            print(f"  Warning: {p.name} has no valid rows after cleaning — skipping")
            continue

        df['_source_csv'] = str(p)   # keep provenance; dropped before output

        # Coerce feature columns — handle "12,73" → 12.73 from Italian locale
        for col in _ALL_FEATURES:
            if col not in df.columns:
                continue
            if df[col].dtype == object:
                df[col] = (
                    df[col].astype(str)
                           .str.strip()
                           .str.replace(',', '.', regex=False)
                )
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Coerce other numeric columns with the same Italian-locale fix
        for col in ('timestamp_s', 'frame_idx'):
            if col in df.columns and df[col].dtype == object:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.strip().str.replace(',', '.', regex=False),
                    errors='coerce',
                )

        # Normalise label: strip whitespace, cast to numeric (keeps NaN for
        # rows that have no manual annotation yet)
        if 'label' in df.columns:
            df['label'] = pd.to_numeric(
                df['label'].astype(str).str.strip().str.replace(',', '.', regex=False)
                           .replace('', np.nan),
                errors='coerce',
            )

        frames.append(df)
        print(f"  Loaded: {p.name}  ({len(df)} rows)")

    if not frames:
        print("ERROR: no valid CSV files found.")
        sys.exit(1)

    return pd.concat(frames, ignore_index=True)


# ── Stage 2 — valid-fit and OOD gate ─────────────────────────────────────────

def apply_stage2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tag rows that fail Stage 2 with 'stage_blocked'.

    Gate 1 — R² ≥ _R2_MIN:
        Rows with R² < 0.10 have an invalid exponential fit. τ, decay_start,
        decay_end, and all features that depend on the decay window are
        unreliable artefacts of the fallback placement, not real signal shape.

    Gate 2 — SPR < _SPR_MAX:
        Rows with SPR ≥ 100 are extremely tonal (single dominant component).
        The SVM training distribution never included such samples for either
        class, so a prediction there is out-of-distribution and meaningless.
    """
    # Gate 1: invalid fit — either R² below the minimum, or no decay fitted at all
    # (τ ≤ 0, the sentinel from _fit_decay_segment). Both mean the decay window is
    # unusable, so they share the Stage2_R2 verdict.
    fail_r2  = df['R2'].lt(_R2_MIN) | df['tau_ms'].le(_TAU_MIN)
    df.loc[fail_r2 & (df['stage_blocked'] == ''), 'stage_blocked'] = 'Stage2_R2'

    # Gate 2: out-of-distribution spectrum — checked only if Gate 1 passed
    fail_spr = df['SPR'].ge(_SPR_MAX)
    df.loc[fail_spr & (df['stage_blocked'] == ''), 'stage_blocked'] = 'Stage2_SPR'

    n2 = (df['stage_blocked'].isin(['Stage2_R2', 'Stage2_SPR'])).sum()
    print(f"  Stage 2: {n2} rejected  "
          f"(R²: {fail_r2.sum()}, SPR: {(fail_spr & ~fail_r2).sum()})")

    return df


# ── Stage 3 — SVM classification ─────────────────────────────────────────────

def apply_stage3(df: pd.DataFrame, model: dict) -> pd.DataFrame:
    """
    Run the SVM on all Stage 2 survivors and tag noise predictions.

    The feature vector is built in the exact column order stored in
    model['features']. Any missing or NaN feature value is handled by the
    SimpleImputer embedded in the sklearn Pipeline.
    """
    pipeline   = model['pipeline']
    threshold  = float(model['threshold'])
    feat_names = model['features']

    # Only rows that passed Stage 2 are sent to the SVM
    s2_mask = df['stage_blocked'] == ''

    if s2_mask.sum() == 0:
        print("  Stage 3: 0 candidates reached SVM (all blocked in Stage 2)")
        return df

    # Validate that every required feature column exists in the DataFrame
    missing_feats = [f for f in feat_names if f not in df.columns]
    if missing_feats:
        print(f"ERROR: CSV is missing SVM feature columns: {missing_feats}")
        sys.exit(1)

    # Build feature matrix — float64 required by libsvm
    X = df.loc[s2_mask, feat_names].values.astype(np.float64)

    # predict_proba column 1 = P(click)
    proba = pipeline.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    df.loc[s2_mask, 'svm_probability'] = np.round(proba, 4)
    df.loc[s2_mask, 'svm_prediction']  = preds

    # Tag rows the SVM rejected as noise
    svm_noise_mask = s2_mask & (df['svm_prediction'] == 0)
    df.loc[svm_noise_mask, 'stage_blocked'] = 'Stage3_SVM'

    n3_pass = int((s2_mask & (df['svm_prediction'] == 1)).sum())
    n3_rej  = int(svm_noise_mask.sum())
    print(f"  Stage 3: {n3_pass} click(s) confirmed, {n3_rej} rejected  "
          f"(threshold = {threshold:.3f})")

    return df


# ── Stage 4 — deduplication ──────────────────────────────────────────────────

def apply_stage4(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate Stage 3 survivors within each recording.

    Two detections from the same recording (same 'file' value) are considered
    duplicates of the same physical click when their frame indices are within
    _DEDUP_FRAMES of each other. From each such group, only the detection with
    the highest svm_probability is kept; the others are tagged 'Stage4_dedup'.

    Deduplication is performed per recording (per unique 'file' value) so that
    events at similar frame positions in different files are never merged.
    """
    # Only rows confirmed by Stage 3 (svm_prediction == 1, stage_blocked == '')
    click_mask = (df['stage_blocked'] == '') & (df['svm_prediction'] == 1)
    n_before   = click_mask.sum()

    if n_before == 0:
        print("  Stage 4: no confirmed clicks to deduplicate")
        return df

    n_deduped = 0

    # Process per recording so proximity groups never cross file boundaries
    for file_id, group_df in df[click_mask].groupby('file'):
        sorted_idx = group_df.sort_values('frame_idx').index.tolist()

        if len(sorted_idx) <= 1:
            continue   # single detection in this recording — nothing to merge

        # Build proximity groups with single-linkage chaining:
        # a new group starts whenever the gap to the previous frame exceeds _DEDUP_FRAMES
        groups: list[list] = []
        current_group      = [sorted_idx[0]]

        for idx in sorted_idx[1:]:
            gap = df.at[idx, 'frame_idx'] - df.at[current_group[-1], 'frame_idx']
            if gap <= _DEDUP_FRAMES:
                current_group.append(idx)
            else:
                groups.append(current_group)
                current_group = [idx]
        groups.append(current_group)

        # Within each group, keep the highest-confidence detection
        for grp in groups:
            if len(grp) <= 1:
                continue
            # Find the row with maximum svm_probability in this group
            best_idx = max(grp, key=lambda i: df.at[i, 'svm_probability'])
            # Tag all others as deduplicated duplicates
            for idx in grp:
                if idx != best_idx:
                    df.at[idx, 'stage_blocked'] = 'Stage4_dedup'
                    n_deduped += 1

    n_after = (df['stage_blocked'] == '').sum()
    print(f"  Stage 4: {n_deduped} duplicate(s) removed — "
          f"{n_after} final confirmed click(s)")

    return df


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, model: dict) -> None:
    """Print a per-stage count table and, if 'label' is present, accuracy metrics."""
    threshold = float(model['threshold'])

    total = len(df)
    n_s2_r2  = (df['stage_blocked'] == 'Stage2_R2').sum()
    n_s2_spr = (df['stage_blocked'] == 'Stage2_SPR').sum()
    n_s3     = (df['stage_blocked'] == 'Stage3_SVM').sum()
    n_s4     = (df['stage_blocked'] == 'Stage4_dedup').sum()
    n_final  = (df['stage_blocked'] == '').sum()

    print(f"\n{'='*55}")
    print(f"  Pipeline summary  (threshold = {threshold:.3f})")
    print(f"{'='*55}")
    print(f"  Input rows             : {total}")
    print(f"  Blocked — Stage2_R2    : {n_s2_r2}")
    print(f"  Blocked — Stage2_SPR   : {n_s2_spr}")
    print(f"  Blocked — Stage3_SVM   : {n_s3}")
    print(f"  Blocked — Stage4_dedup : {n_s4}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Final confirmed clicks : {n_final}")

    # If manual labels are present, compare against the pipeline result
    if 'label' not in df.columns:
        return

    labeled = df[df['label'].notna()].copy()
    if labeled.empty:
        return

    labeled['label']          = labeled['label'].astype(int)
    labeled['pipeline_click'] = (labeled['stage_blocked'] == '').astype(int)

    tp = int(((labeled['label'] == 1) & (labeled['pipeline_click'] == 1)).sum())
    fp = int(((labeled['label'] == 0) & (labeled['pipeline_click'] == 1)).sum())
    fn = int(((labeled['label'] == 1) & (labeled['pipeline_click'] == 0)).sum())
    tn = int(((labeled['label'] == 0) & (labeled['pipeline_click'] == 0)).sum())
    n_labeled = len(labeled)

    recall      = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    precision   = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')

    print(f"\n  Comparison with manual labels  ({n_labeled} labeled rows):")
    print(f"  Confusion:  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Recall      : {recall:.3f}   ({tp}/{tp+fn} clicks detected)")
    print(f"  Precision   : {precision:.3f}")
    print(f"  Specificity : {specificity:.3f}")

    # Per-stage breakdown of false negatives (labeled clicks that were blocked)
    if fn > 0:
        print(f"\n  False negatives breakdown (labeled=1 rows blocked by pipeline):")
        missed = labeled[(labeled['label'] == 1) & (labeled['pipeline_click'] == 0)]
        for stage, grp in missed.groupby('stage_blocked'):
            print(f"    {stage}: {len(grp)} row(s)")
            for _, row in grp.iterrows():
                print(f"      frame {int(row['frame_idx'])}  "
                      f"R²={row['R2']:.3f}  SPR={row['SPR']:.1f}  "
                      f"svm_prob={row['svm_probability']:.3f}"
                      if not np.isnan(row['svm_probability'])
                      else f"      frame {int(row['frame_idx'])}  "
                           f"R²={row['R2']:.3f}  SPR={row['SPR']:.1f}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Evaluate candidate CSVs through PlantLeaf v5 Stages 2/3/4',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        'csvs', nargs='+', type=Path,
        help='Candidate CSV files and/or folders containing *.csv files',
    )
    parser.add_argument(
        '--model', required=True, type=Path,
        help='Trained SVM model (.pkl produced by train_svm.py)',
    )
    parser.add_argument(
        '--output', type=Path, default=Path('evaluated_candidates.csv'),
        help='Output CSV path (default: evaluated_candidates.csv)',
    )
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    if not args.model.exists():
        print(f"ERROR: model not found: {args.model}")
        sys.exit(1)

    model = joblib.load(args.model)
    for key in ('pipeline', 'threshold', 'features'):
        if key not in model:
            print(f"ERROR: model is missing key '{key}'. Re-train with train_svm.py.")
            sys.exit(1)

    print(f"PlantLeaf v5 — Candidate Evaluation")
    print(f"  Model    : {args.model.name}  "
          f"(kernel={model.get('kernel','?')}, threshold={model['threshold']:.3f})")
    print(f"  Features : {model['features']}")
    print(f"  Output   : {args.output}")
    print()

    # ── Resolve folders → CSV file lists ─────────────────────────────────────
    resolved_paths: list[Path] = []
    for p in args.csvs:
        if p.is_dir():
            found = sorted(p.glob('*.csv'))
            if not found:
                print(f"  Warning: no *.csv files found in folder {p}")
            else:
                print(f"  Folder: {p.name}  ({len(found)} CSV file(s) found)")
            resolved_paths.extend(found)
        elif p.is_file():
            resolved_paths.append(p)
        else:
            print(f"  Warning: {p} does not exist — skipping")

    if not resolved_paths:
        print("ERROR: no CSV files to process.")
        sys.exit(1)

    # ── Load CSVs ─────────────────────────────────────────────────────────────
    print(f"\nLoading {len(resolved_paths)} CSV file(s):")
    df = load_csvs(resolved_paths)
    print(f"  Total rows: {len(df)}\n")

    # ── Initialise output columns ─────────────────────────────────────────────
    df['svm_probability'] = np.nan
    df['svm_prediction']  = pd.array([pd.NA] * len(df), dtype='Int64')
    df['stage_blocked']   = ''

    # ── Run pipeline stages ───────────────────────────────────────────────────
    df = apply_stage2(df)
    df = apply_stage3(df, model)
    df = apply_stage4(df)

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(df, model)

    # ── Save ─────────────────────────────────────────────────────────────────
    # Drop the internal provenance column before saving
    df = df.drop(columns=['_source_csv'], errors='ignore')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\n  Saved: {args.output}  ({len(df)} rows)")


if __name__ == '__main__':
    main()
