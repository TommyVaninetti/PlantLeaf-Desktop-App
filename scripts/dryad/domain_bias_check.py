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
Does the hybrid pipeline leak, and did colorization actually close the domain gap?

Three checks, run against a features CSV produced by `dryad_build_plots.py` and
PlantLeaf's own `Dataset_20June2026.csv`.


1. peak_SNR label leak
   A fixed target SNR puts injected clicks at one value and injected negatives at
   another, so peak_SNR alone identifies the class. Single-feature AUC on
   PlantLeaf's own 285 real rows is 0.823 — that is the number to land near.
   Materially ABOVE means separation was manufactured; materially BELOW means the
   real physical signal was destroyed by flattening both classes onto one
   distribution.

2. Coverage and overlap
   Synthetic peak_SNR against the real distribution (p10 7.19 / median 12.79 /
   p90 39.11), plus how much the two classes overlap. On real data 63 % of
   negatives fall inside the clicks' range; clean separation would be a red flag,
   not a success.

3. Name-the-dataset (Gorin et al., arXiv:2312.00231)
   Train a classifier to predict the DOMAIN — PlantLeaf-native vs Khait-derived —
   from the 17 features, using domain as the label. If it succeeds, the channel
   model has NOT closed the gap and the features still carry a domain signature
   that any downstream classifier can exploit instead of learning click physics.
   Per-feature importance names exactly which features are responsible.

   Reading the number: 0.5 balanced accuracy = domains indistinguishable (ideal).
   ~1.0 = trivially separable. Some separation is expected and legitimate — the
   species differ (tomato/tobacco vs succulents) — so this is a diagnostic to
   report and interpret, not a pass/fail gate.

Usage:
    python3 scripts/domain_bias_check.py --synthetic out/dryad_render/features_provisional.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# parents[2], not parent.parent: this file lives at scripts/dryad/, so the repo
# root is two levels up. The move from scripts/ left the old expression pointing
# at scripts/src and every one of these scripts died on "No module named hybrid".
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

REAL_CSV = "/Users/tommy/PlantLeaf_dev/Analisi/v5/SVM_Training/Dataset_20June2026.csv"

# The 17 v5 features, in the order the CSV declares them.
FEATURES = ("peak_SNR", "pre_SNR", "post_SNR", "rise_time_ms", "fall_time_ms",
            "asymmetry_integral", "ZCR_pre", "ZCR_click", "ZCR_post", "kurtosis",
            "centroid_shift_hz", "tau_ms", "R2", "fit_coverage", "SPR",
            "R_spectral", "FPE_hz")

# PlantLeaf's own confirmed clicks, Dataset_20June2026.csv.
REAL_P10, REAL_MEDIAN, REAL_P90 = 7.19, 12.79, 39.11


def _f(row: dict, key: str) -> float:
    """
    Parse a CSV cell to float, tolerating comma decimals.

    An Excel-on-Italian-locale re-save silently rewrote 16 rows of one session
    with comma decimal separators (spec section 7). Handling it here means a
    corrupted export degrades to a few NaNs instead of throwing.
    """
    raw = (row.get(key) or "").strip().replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return float("nan")


def load_real(path: str = REAL_CSV) -> tuple[np.ndarray, np.ndarray]:
    """PlantLeaf-native features + binary click label."""
    rows = list(csv.DictReader(open(path)))
    X = np.array([[_f(r, f) for f in FEATURES] for r in rows], dtype=np.float64)
    y = np.array([_f(r, "label") for r in rows], dtype=np.float64)
    keep = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    return X[keep], y[keep]


def load_synthetic(path: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Khait-derived features + binary click label, from a hybrid export."""
    rows = list(csv.DictReader(open(path)))
    X = np.array([[_f(r, f"feat_{f}") for f in FEATURES] for r in rows], dtype=np.float64)
    y = np.array([1.0 if (r.get("is_click") or "").strip() == "True" else 0.0 for r in rows])
    keep = np.all(np.isfinite(X), axis=1)
    return X[keep], y[keep], [r for r, k in zip(rows, keep) if k]


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    ROC AUC via the rank (Mann-Whitney U) identity, ties averaged.

    Hand-rolled rather than pulled from sklearn.metrics so this check runs even
    where scikit-learn is unavailable; the domain classifier below needs sklearn
    and degrades gracefully on its own.
    """
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    values = np.concatenate([pos, neg])
    # Average ranks within tied groups so ties score 0.5, not 0 or 1.
    for value in np.unique(values):
        tied = values == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def check_peak_snr_leak(Xs, ys, Xr, yr):
    print("\n" + "=" * 72)
    print("1. peak_SNR single-feature AUC  -  is the class separation manufactured?")
    print("=" * 72)
    i = FEATURES.index("peak_SNR")
    real = auc(Xr[:, i], yr)
    synth = auc(Xs[:, i], ys)
    print(f"  PlantLeaf real (285 rows) : {real:.3f}")
    print(f"  Khait-derived synthetic   : {synth:.3f}")

    delta = synth - real
    if not np.isfinite(synth):
        verdict = "cannot evaluate (one class missing from the synthetic set)"
    elif abs(delta) <= 0.06:
        verdict = "MATCHES real -- separation inherited, not manufactured"
    elif delta > 0.06:
        verdict = ("HIGHER than real -- suspect a label leak: peak_SNR alone is doing "
                   "more work on synthetic data than it does on real data")
    else:
        verdict = ("LOWER than real -- the real physical signal has been flattened; "
                   "check the amplitude model is not collapsing both classes")
    print(f"  delta {delta:+.3f}  ->  {verdict}")
    return synth


def check_coverage(Xs, ys, Xr, yr):
    print("\n" + "=" * 72)
    print("2. peak_SNR coverage and class overlap")
    print("=" * 72)
    i = FEATURES.index("peak_SNR")

    def describe(values, name):
        if not len(values):
            return
        print(f"  {name:<26} p10 {np.percentile(values, 10):7.2f}  "
              f"median {np.median(values):7.2f}  p90 {np.percentile(values, 90):7.2f}  "
              f"n={len(values)}")

    describe(Xr[yr == 1, i], "real clicks")
    describe(Xr[yr == 0, i], "real negatives")
    describe(Xs[ys == 1, i], "synthetic clicks")
    describe(Xs[ys == 0, i], "synthetic negatives")
    print(f"  {'PlantLeaf reference':<26} p10 {REAL_P10:7.2f}  median {REAL_MEDIAN:7.2f}  "
          f"p90 {REAL_P90:7.2f}")

    for X, y, name in ((Xr, yr, "real"), (Xs, ys, "synthetic")):
        clicks, negs = X[y == 1, i], X[y == 0, i]
        if not len(clicks) or not len(negs):
            continue
        lo, hi = clicks.min(), clicks.max()
        inside = float(np.mean((negs >= lo) & (negs <= hi)))
        sep = np.median(clicks) / max(np.median(negs), 1e-9)
        print(f"  {name:<10} negatives inside the clicks' range: {100 * inside:5.1f} %   "
              f"median separation {sep:.2f}x")
    print("  (real data: 63 % overlap, 2.4x separation. Clean separation on the")
    print("   synthetic set would be a red flag, not a success.)")


def check_name_the_dataset(Xs, Xr):
    print("\n" + "=" * 72)
    print("3. Name-the-dataset  -  can a classifier tell the two domains apart?")
    print("=" * 72)
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.inspection import permutation_importance
    except ImportError:
        print("  SKIPPED: scikit-learn not installed (it is in requirements.txt)")
        return None

    X = np.vstack([Xr, Xs])
    domain = np.concatenate([np.zeros(len(Xr)), np.ones(len(Xs))])

    # Class-balanced so the score is not inflated by the 5477-vs-285 size gap.
    clf = RandomForestClassifier(n_estimators=300, random_state=0,
                                 class_weight="balanced", n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, domain, cv=cv, scoring="balanced_accuracy")
    mean = float(scores.mean())

    print(f"  balanced accuracy: {mean:.3f} +/- {scores.std():.3f}   "
          f"(0.5 = domains indistinguishable, 1.0 = trivially separable)")
    if mean < 0.65:
        print("  -> domains largely overlap. The channel model has closed most of the gap.")
    elif mean < 0.85:
        print("  -> partial separation. Some is legitimate (tomato/tobacco vs succulents),")
        print("     but check the top features below for instrument signatures.")
    else:
        print("  -> domains are trivially separable. A hybrid-trained classifier can key on")
        print("     RECORDING DOMAIN instead of click physics; do not train on this as-is.")

    clf.fit(X, domain)
    imp = permutation_importance(clf, X, domain, n_repeats=10, random_state=0, n_jobs=-1)
    order = np.argsort(imp.importances_mean)[::-1]
    print("\n  Which features carry the domain signature (permutation importance):")
    for rank, idx in enumerate(order[:8], 1):
        print(f"    {rank}. {FEATURES[idx]:<22} {imp.importances_mean[idx]:.4f} "
              f"+/- {imp.importances_std[idx]:.4f}")
    print("\n  Spectral-shape features (FPE, centroid, SPR, R_spectral) near the top means")
    print("  residual coloration error. Noise-normalized features near the top means the")
    print("  injection SNR regime still differs from the real one.")
    return mean


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synthetic", required=True,
                        help="features_provisional.csv from dryad_build_plots.py")
    parser.add_argument("--real", default=REAL_CSV, help="PlantLeaf's own labelled CSV")
    args = parser.parse_args(argv)

    if not Path(args.synthetic).is_file():
        print(f"Synthetic CSV not found: {args.synthetic}", file=sys.stderr)
        return 1

    Xr, yr = load_real(args.real)
    Xs, ys, rows = load_synthetic(args.synthetic)
    print(f"real: {len(Xr)} rows ({int(yr.sum())} clicks)   "
          f"synthetic: {len(Xs)} rows ({int(ys.sum())} clicks)")

    modes = {(r.get("amplitude_mode") or "?") for r in rows}
    print(f"amplitude mode(s) in the synthetic export: {sorted(modes)}")
    if "fixed-snr" in modes:
        print("  WARNING: fixed-snr output is visualization material. Every clip sits at one")
        print("  peak_SNR, which makes check 1 meaningless and check 3 optimistic.")

    check_peak_snr_leak(Xs, ys, Xr, yr)
    check_coverage(Xs, ys, Xr, yr)
    check_name_the_dataset(Xs, Xr)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
