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
feature_distributions.py — click vs ambiguous vs noise, and what a gate would cost

    python3 scripts/v6/feature_distributions.py DATASET_DIR -o out/

Reads labelled v6 `*_candidates.csv` and produces:

    <out>/dist_<feature>.png     overlaid distributions, three classes
    <out>/by_session_<f>.png     per-session panel (the click_feature_by_session layout)
    <out>/gate_table.csv         clicks / ambiguous lost vs noise cut, per threshold
    <out>/k_sweep.csv            what raising Stage 1's k would cost
    <out>/outliers.csv           rows whose values would wreck a scaler
    stdout                       the same tables, readable

WHY THE GATE TABLE IS THE POINT
Stage 2 rejects are HARD — no probability, invisible to the SVM — so a threshold
placed where the click distribution is merely thin costs recall irrecoverably.
Every candidate threshold is therefore reported as *clicks lost*, not as an
accuracy. A gate whose click cost is unknown must not ship.

TWO THINGS THIS IS CAREFUL ABOUT
1. **Exhaustively-labelled recordings only, by default.** A recording where only
   the interesting rows were labelled has a noise class that is not the noise
   population, and every "% of noise removed" computed from it is inflated.
   `--include-partial` opts out, and the header says which mode produced a table.
2. **NaN is a category, never a silent drop.** `harmonic_confinement` is NaN on
   ~23 % of rows BY DESIGN (second harmonic off-band), and every v6 feature is NaN
   when `b3_frames == 0`. A gate must therefore decide what NaN does; here NaN
   always PASSES, because a row whose feature could not be measured cannot be
   judged by it.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

# One colour per class, used everywhere. There is no shared plot style in this
# repo (analyze_dataset defines a dark palette inside its own function,
# v6_ek_plots uses ad-hoc light colours), so consistency here matters more than
# matching either.
C_CLICK, C_AMBIG, C_NOISE = "#1f77b4", "#ff9f1c", "#9aa0a6"
CLASSES = (("1", "click", C_CLICK), ("2", "ambiguous", C_AMBIG), ("0", "noise", C_NOISE))

#: Features worth a distribution panel. Order is deliberate: the strong
#: separators first, so the output reads top-down.
FEATURES = [
    "peak_SNR", "k_ratio", "local_crest", "n_seg",
    "harmonic_confinement", "shape_novelty", "spectral_entropy", "SPR",
    "R2", "tau_ms", "fit_coverage", "kurtosis", "SPR_region",
]

#: Anything beyond this is a reconstruction artefact, not a measurement. Measured
#: maxima on the real corpus: peak_SNR 1.14e40, k_ratio 4.1e81, local_crest
#: 3.8e81 — a single one of these destroys a StandardScaler or an axis range.
#: Clipped for DISPLAY only, and always counted in outliers.csv.
OUTLIER_ABS = 1e6

#: Candidate thresholds per feature, and the direction that KEEPS a row.
#: 'ge' = keep when value >= t (reject small), 'le' = keep when value <= t.
GATES = {
    "peak_SNR":             ("ge", [3.0, 4.0, 4.5, 4.6, 5.0, 6.0, 8.0]),
    "n_seg":                ("ge", [4, 6, 8, 10, 12, 20, 45]),
    "local_crest":          ("ge", [1.0, 1.1, 1.2, 1.25, 1.5, 2.0]),
    "k_ratio":              ("ge", [1.5, 1.6, 1.75, 2.0, 2.5]),
    "harmonic_confinement": ("le", [1.0, 1.6, 2.0, 2.5, 3.0]),
    "shape_novelty":        ("ge", [0.0, 0.01, 0.02, 0.05, 0.1]),
    "SPR":                  ("le", [20.0, 50.0, 100.0]),
}


# ─────────────────────────────────────────────────────────────────────────────

def _f(v):
    """CSV cell -> float, tolerating the Italian decimal comma Excel writes."""
    try:
        return float(str(v).replace(",", "."))
    except (TypeError, ValueError):
        return math.nan


def load(dataset_dir: Path, include_partial: bool):
    """
    Return (rows_by_class, n_files, n_partial_skipped, session_of_row).

    macOS AppleDouble forks ('._name.csv') are skipped. They are metadata, not
    data — 66 of the 132 "CSVs" in the v6 dataset are those, and a naive rglob
    reads them as files and then dies on a UnicodeDecodeError.
    """
    by_class = {k: defaultdict(list) for k, _, _ in CLASSES}
    sessions = {k: [] for k, _, _ in CLASSES}
    n_files = n_skipped = 0
    for f in sorted(dataset_dir.rglob("*_candidates.csv")):
        if f.name.startswith("._"):
            continue
        try:
            with open(f, encoding="utf-8-sig") as fh:
                rows = list(csv.DictReader(fh))
        except (OSError, UnicodeDecodeError) as exc:
            print(f"  ! unreadable: {f.name}  ({type(exc).__name__})", file=sys.stderr)
            continue
        if not rows:
            continue
        labels = [(r.get("label") or "").strip() for r in rows]
        if not include_partial and any(l == "" for l in labels):
            n_skipped += 1
            continue
        n_files += 1
        stem = f.stem.replace("_candidates", "")
        for r, lab in zip(rows, labels):
            if lab not in by_class:
                continue
            for ft in FEATURES:
                by_class[lab][ft].append(_f(r.get(ft)))
            sessions[lab].append(r.get("session_id") or stem)
    return by_class, sessions, n_files, n_skipped


def _arr(by_class, lab, ft):
    return np.asarray(by_class[lab].get(ft, []), dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────

def summarise(by_class, out_dir: Path):
    print("\n── distributions ──")
    hdr = f"{'feature':22s} {'class':10s} {'n':>6} {'NaN%':>6} " \
          f"{'min':>10} {'p5':>10} {'median':>10} {'p95':>10} {'max':>10}"
    print(hdr)
    for ft in FEATURES:
        for lab, name, _ in CLASSES:
            v = _arr(by_class, lab, ft)
            if not v.size:
                continue
            fin = v[np.isfinite(v)]
            nanpct = 100.0 * (1 - fin.size / v.size)
            if not fin.size:
                print(f"{ft:22s} {name:10s} {v.size:>6} {nanpct:>5.1f}%  all NaN")
                continue
            print(f"{ft:22s} {name:10s} {fin.size:>6} {nanpct:>5.1f}% "
                  f"{fin.min():>10.4g} {np.percentile(fin,5):>10.4g} "
                  f"{np.median(fin):>10.4g} {np.percentile(fin,95):>10.4g} "
                  f"{fin.max():>10.4g}")
        print()


def outlier_report(by_class, out_dir: Path):
    rows = []
    for ft in FEATURES:
        for lab, name, _ in CLASSES:
            v = _arr(by_class, lab, ft)
            fin = v[np.isfinite(v)]
            bad = fin[np.abs(fin) > OUTLIER_ABS]
            if bad.size:
                rows.append({"feature": ft, "class": name, "n_outliers": int(bad.size),
                             "n_total": int(fin.size), "max_abs": float(np.abs(bad).max())})
    dest = out_dir / "outliers.csv"
    with open(dest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["feature", "class", "n_outliers",
                                           "n_total", "max_abs"])
        w.writeheader(); w.writerows(rows)
    if rows:
        print(f"\n── outliers (|value| > {OUTLIER_ABS:g}) — reconstruction artefacts, "
              f"not measurements ──")
        for r in rows:
            print(f"   {r['feature']:22s} {r['class']:10s} "
                  f"{r['n_outliers']:>4} of {r['n_total']:<6} max |x| = {r['max_abs']:.3g}")
        print("   these are peak_amp blowups in the iFFT; they will wreck any scaler")
    return rows


def gate_table(by_class, out_dir: Path):
    """clicks / ambiguous lost vs noise cut, for every candidate threshold."""
    print("\n── gate table — NaN always PASSES (an unmeasurable row cannot be judged) ──")
    print(f"{'gate':34s} {'clk lost':>9} {'amb lost':>9} {'noise cut':>10}")
    out = []
    for ft, (direction, thresholds) in GATES.items():
        for t in thresholds:
            cost = {}
            for lab, name, _ in CLASSES:
                v = _arr(by_class, lab, ft)
                if not v.size:
                    cost[name] = math.nan
                    continue
                # NaN passes: ~(v < t) is True for NaN, and so is ~(v > t)
                keep = ~(v < t) if direction == "ge" else ~(v > t)
                cost[name] = 100.0 * (1 - keep.mean())
            sym = ">=" if direction == "ge" else "<="
            label = f"{ft} {sym} {t}"
            print(f"{label:34s} {cost['click']:>8.1f}% {cost['ambiguous']:>8.1f}% "
                  f"{cost['noise']:>9.1f}%")
            out.append({"feature": ft, "direction": direction, "threshold": t,
                        "clicks_lost_pct": round(cost["click"], 3),
                        "ambiguous_lost_pct": round(cost["ambiguous"], 3),
                        "noise_cut_pct": round(cost["noise"], 3)})
        print()
    with open(out_dir / "gate_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)
    return out


def combined_gates(by_class):
    """The proposed Stage 2 tiers, as a single reject rule."""
    def cost(rule):
        r = []
        for lab, name, _ in CLASSES:
            n = len(by_class[lab].get("peak_SNR", []))
            if not n:
                r.append(math.nan); continue
            v = {ft: _arr(by_class, lab, ft) for ft in FEATURES}
            r.append(100.0 * rule(v).mean())
        return r

    print("── combined Stage 2 tiers ──")
    print(f"{'tier':52s} {'clk lost':>9} {'amb lost':>9} {'noise cut':>10}")
    tiers = [
        ("TIER 1  peak_SNR>=4.5 & n_seg>=10 & hc<=1.6",
         lambda v: (v["peak_SNR"] < 4.5) | (v["n_seg"] < 10)
                   | (v["harmonic_confinement"] > 1.6)),
        ("TIER 1  + local_crest>=1.2",
         lambda v: (v["peak_SNR"] < 4.5) | (v["n_seg"] < 10)
                   | (v["harmonic_confinement"] > 1.6) | (v["local_crest"] < 1.2)),
        ("TIER 2  peak_SNR>=5.0 & n_seg>=10 & hc<=1.6 & lc>=1.2",
         lambda v: (v["peak_SNR"] < 5.0) | (v["n_seg"] < 10)
                   | (v["harmonic_confinement"] > 1.6) | (v["local_crest"] < 1.2)),
        ("OLD     the fit gate (fit_valid/R2/tau) — for comparison",
         lambda v: ~np.isfinite(v["R2"]) | ~np.isfinite(v["tau_ms"])
                   | (v["R2"] < 0.10) | (v["tau_ms"] <= 0.0)),
    ]
    rows = []
    for name, rule in tiers:
        c = cost(rule)
        print(f"{name:52s} {c[0]:>8.1f}% {c[1]:>8.1f}% {c[2]:>9.1f}%")
        rows.append({"tier": name, "clicks_lost_pct": round(c[0], 3),
                     "ambiguous_lost_pct": round(c[1], 3),
                     "noise_cut_pct": round(c[2], 3)})
    return rows


def k_sweep(by_class, out_dir: Path):
    """
    What raising Stage 1's k would cost, straight from k_ratio.

    Exact in the sense that Stage 1's rule IS `k_ratio > k`, but a LOWER BOUND on
    survivors: raising k un-flags frames, which can expose a peak that was
    previously suppressed by a neighbour that no longer qualifies.
    """
    print("\n── k sweep (Stage 1 threshold) — a LOWER BOUND on what survives ──")
    print(f"{'k':>6} {'clicks kept':>12} {'ambig kept':>11} {'noise kept':>11} "
          f"{'noise removed':>14}")
    out = []
    for k in (1.5, 1.6, 1.75, 2.0, 2.5, 3.0, 3.46, 4.0, 5.0):
        r = []
        for lab, name, _ in CLASSES:
            v = _arr(by_class, lab, "k_ratio")
            fin = v[np.isfinite(v)]
            r.append(100.0 * np.mean(fin > k) if fin.size else math.nan)
        print(f"{k:>6.2f} {r[0]:>11.1f}% {r[1]:>10.1f}% {r[2]:>10.1f}% {100-r[2]:>13.1f}%")
        out.append({"k": k, "clicks_kept_pct": round(r[0], 3),
                    "ambiguous_kept_pct": round(r[1], 3),
                    "noise_kept_pct": round(r[2], 3),
                    "noise_removed_pct": round(100 - r[2], 3)})
    with open(out_dir / "k_sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)
    kc = _arr(by_class, "1", "k_ratio")
    kc = kc[np.isfinite(kc)]
    if kc.size:
        print(f"\n   lowest labelled click k_ratio = {kc.min():.4f}")
        print("   clicks sit AT the detection threshold, so clicks below k almost"
              " certainly exist and were never exported — they cannot be counted.")
    return out


# ─────────────────────────────────────────────────────────────────────────────

def plot_distribution(by_class, ft: str, dest: Path):
    """Overlaid per-class distribution. Log-x when the feature spans decades."""
    series = []
    for lab, name, colour in CLASSES:
        v = _arr(by_class, lab, ft)
        fin = v[np.isfinite(v)]
        fin = fin[np.abs(fin) <= OUTLIER_ABS]      # clip artefacts for DISPLAY
        if fin.size:
            series.append((name, colour, fin, v.size, v.size - np.isfinite(v).sum()))
    if not series:
        return False

    allv = np.concatenate([s[2] for s in series])
    positive = allv.min() > 0
    span = (allv.max() / allv.min()) if positive else 0
    logx = positive and span > 50

    fig, (ax, axb) = plt.subplots(
        2, 1, figsize=(9, 5.4), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    bins = (np.logspace(np.log10(allv.min()), np.log10(allv.max()), 60)
            if logx else np.linspace(allv.min(), allv.max(), 60))
    for name, colour, fin, n_all, n_nan in series:
        ax.hist(fin, bins=bins, density=True, alpha=0.5, color=colour,
                label=f"{name}  (n={fin.size}, NaN {100*n_nan/max(n_all,1):.0f}%)")
    if logx:
        ax.set_xscale("log")
    ax.set_ylabel("density")
    ax.set_title(f"{ft} — click vs ambiguous vs noise")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # A box row underneath makes the overlap legible where the histograms pile up.
    # 'labels' was renamed 'tick_labels' in Matplotlib 3.9; set them afterwards so
    # the script works on both without a version check.
    axb.boxplot([s[2] for s in series], vert=False, widths=0.6, showfliers=False)
    axb.set_yticklabels([s[0] for s in series])
    axb.grid(alpha=0.25, axis="x")
    axb.set_xlabel(ft + ("  (log scale)" if logx else ""))
    fig.tight_layout()
    fig.savefig(dest, dpi=130)
    plt.close(fig)
    return True


def plot_by_session(by_class, sessions, ft: str, dest: Path, max_sessions: int = 26):
    """
    Per-session strip panel — the click_feature_by_session layout.

    A threshold that works on the pooled distribution can still fail on one
    session; pooling hides exactly that, which is why this panel exists.
    """
    per = defaultdict(lambda: {k: [] for k, _, _ in CLASSES})
    for lab, _, _ in CLASSES:
        vals = _arr(by_class, lab, ft)
        for s, v in zip(sessions[lab], vals):
            if np.isfinite(v) and abs(v) <= OUTLIER_ABS:
                per[s][lab].append(v)
    keys = sorted(per, key=lambda s: -len(per[s]["1"]))[:max_sessions]
    if not keys:
        return False

    fig, ax = plt.subplots(figsize=(10, max(3.2, 0.34 * len(keys))))
    rng = np.random.default_rng(0)
    for row, s in enumerate(keys):
        for lab, name, colour in CLASSES:
            v = np.asarray(per[s][lab])
            if not v.size:
                continue
            y = row + rng.uniform(-0.16, 0.16, v.size)
            ax.scatter(v, y, s=7, alpha=0.55, color=colour, linewidths=0,
                       label=name if row == 0 else None)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([k[:44] for k in keys], fontsize=7)
    ax.set_xlabel(ft)
    ax.set_title(f"{ft} by session — a gate must hold on EVERY row, not on the pool")
    allv = np.concatenate([np.asarray(per[s][l]) for s in keys
                           for l, _, _ in CLASSES if per[s][l]])
    if allv.size and allv.min() > 0 and allv.max() / allv.min() > 50:
        ax.set_xscale("log")
    ax.grid(alpha=0.25, axis="x")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(dest, dpi=130)
    plt.close(fig)
    return True


# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path, help="directory of labelled *_candidates.csv")
    ap.add_argument("-o", "--out", type=Path, default=Path("feature_distributions"))
    ap.add_argument("--include-partial", action="store_true",
                    help="also use recordings that are only partly labelled. OFF by "
                         "default: their noise class is not the noise population, so "
                         "every '%% of noise removed' computed from them is inflated.")
    ap.add_argument("--no-plots", action="store_true", help="tables only")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    by_class, sessions, n_files, n_skipped = load(args.dataset, args.include_partial)
    n = {name: len(by_class[lab].get("peak_SNR", [])) for lab, name, _ in CLASSES}
    mode = "ALL labelled rows" if args.include_partial else "EXHAUSTIVELY-labelled recordings only"

    print(f"\n{mode}")
    print(f"  files used: {n_files}" +
          (f"   (skipped {n_skipped} partly-labelled)" if n_skipped else ""))
    print(f"  clicks={n['click']}  ambiguous={n['ambiguous']}  noise={n['noise']}")
    if not sum(n.values()):
        print("\nno labelled rows found — nothing to do\n", file=sys.stderr)
        return 1

    summarise(by_class, args.out)
    outlier_report(by_class, args.out)
    gate_table(by_class, args.out)
    combined = combined_gates(by_class)
    k_sweep(by_class, args.out)

    with open(args.out / "tiers.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(combined[0]))
        w.writeheader(); w.writerows(combined)

    if not args.no_plots:
        made = 0
        for ft in FEATURES:
            if plot_distribution(by_class, ft, args.out / f"dist_{ft}.png"):
                made += 1
            if ft in GATES:
                plot_by_session(by_class, sessions, ft, args.out / f"by_session_{ft}.png")
        print(f"\nwrote {made} distribution plot(s) + per-session panels to {args.out}")
    print(f"tables: gate_table.csv  k_sweep.csv  tiers.csv  outliers.csv\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
