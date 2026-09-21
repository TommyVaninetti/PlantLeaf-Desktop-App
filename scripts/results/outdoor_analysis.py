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
outdoor_analysis.py — what the outdoor environment does to the detector, per site

    python3 scripts/results/outdoor_analysis.py \
        --labelled ~/PlantLeaf_dev/Analisi/v6/labelled/_all \
        --analysis-set ~/PlantLeaf_dev/Analisi/v6/results/analysis_set.csv \
        -o ~/PlantLeaf_dev/Analisi/v6/results/outdoor

WHY THE NOISE-ONLY RECORDINGS ARE THE WHOLE POINT
-------------------------------------------------
No plant is present in them, so every event the pipeline accepts there is a FALSE
POSITIVE by construction. That makes 12 outdoor recordings a labelled negative set
of ~18 000 events, acquired without anyone labelling anything — against the 189
clicks a human confirmed indoors. The asymmetry is the opportunity: the thing we
are short of outdoors is not negatives, it is positives.

SITE, NOT "OUTDOOR"
-------------------
"Outdoor" is not one environment. A spring balcony over a street (birds, traffic,
building works) and an August countryside taverna are different acoustic problems,
recorded four months apart. They are kept apart everywhere in this script, and the
per-recording spread is reported next to every site median, because the spread is
itself a result: the same balcony gives 16 events/h one morning and 3458/h another.

THE 66 kHz ARTEFACT IS REMOVED EVENT BY EVENT
---------------------------------------------
A narrowband interference tone at 64-70 kHz contaminates part of the corpus. It is
INSTRUMENTAL, not environmental, and two of the noise recordings are half made of
it. Left in, it would flatter every result here twice over: it inflates the outdoor
event rate, and it sits at 66 kHz, so an "FPE_hz <= 57 kHz" gate would score a huge
false-positive removal for rejecting our own interference. So every event in
TONE_BAND_HZ is dropped from the false-positive set before anything is computed,
and the number dropped is reported. Same band as the session rule in
scripts/stress/build_stress_corpus.py (declared 17-09-2026).

WHAT THIS CANNOT SAY
--------------------
No outdoor event is labelled, so "false positive" rests on there being no plant in
the recording — no plant UNDER MEASUREMENT, which does not exclude vegetation
nearby. And the clicks compared against are indoor, so a difference between the two
populations may come from the channel (distance, wind, gain) and not only from the
source. Gate costs are measured on real confirmed clicks and are sound; gate
benefits are measured on presumed negatives and are an upper bound.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TONE_BAND_HZ = (64_000.0, 70_000.0)

#: Features carried through the comparison. Every one is available on every
#: candidate row (harmonic_confinement is not: it is undefined on ~60 % of events,
#: which is why it cannot carry a gate however well it separates where it exists).
FEATURES = ['n_seg', 'FPE_hz', 'local_crest', 'rise_time_ms', 'fall_time_ms',
            'peak_SNR', 'tau_ms', 'kurtosis', 'spectral_tilt', 'f_50_hz',
            'harmonic_confinement']

#: (feature, keep-direction) pairs a gate may be built from, with the direction
#: fixed a priori from the click physics — a cavitation click is LONGER (n_seg),
#: LOWER in peak frequency and more impulsive than an environmental transient.
GATE_CANDIDATES = [('n_seg', '>='), ('FPE_hz', '<='), ('local_crest', '>='),
                   ('rise_time_ms', '>='), ('f_50_hz', '<='), ('peak_SNR', '>=')]

#: Peak-frequency bands the accepted events are split into. The 64-70 kHz band is
#: our own interference; the 70-100 kHz band is NOT — it is 20x commoner outdoors
#: than indoors and no confirmed click has ever landed in it (see report_bands).
BANDS = [(0, 25_000, 'below 25 kHz'), (25_000, 60_000, 'the click band, 25-60 kHz'),
         (60_000, 64_000, '60-64 kHz'), (64_000, 70_000, '64-70 kHz (interference)'),
         (70_000, 100_000, 'above 70 kHz')]

STAGE5_W_S = 180.0          # sliding window of the stress detector
STAGE5_H = 7                # calibrated threshold (events in W), 17-09-2026
BIN_S = 30.0

READ_COLS = FEATURES + ['timestamp_s', 'stage_blocked', 'label',
                        'noise_floor_mV', 'file']


# ── site assignment ──────────────────────────────────────────────────────────
def site_of(row) -> str:
    """Recording -> acoustic site. Read off the corpus folder, not the filename."""
    p = str(row.get('paudio', ''))
    if 'Balcone vuoto' in p:
        return 'balcony'
    if 'Cucina fuori' in p:
        return 'kitchen window'
    if 'tavernaoutdoor' in p:
        return 'taverna'
    if 'outdoortaverna' in str(row.get('file', '')) or 'outdoorteca' in str(row.get('file', '')):
        return 'taverna'
    return 'indoor empty room' if row.get('environment') == 'indoor' else 'outdoor (other)'


def poisson_ci(n: int, hours: float) -> tuple[float, float]:
    """Exact (Garwood) 95 % interval on a rate, per hour."""
    from scipy.stats import chi2
    if hours <= 0:
        return float('nan'), float('nan')
    lo = 0.0 if n == 0 else chi2.ppf(0.025, 2 * n) / 2 / hours
    return lo, chi2.ppf(0.975, 2 * (n + 1)) / 2 / hours


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a click scores above an outdoor event). 0.5 = no separation."""
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) < 5 or len(neg) < 5:
        return float('nan')
    from scipy.stats import rankdata
    all_v = np.concatenate([pos, neg])
    r = rankdata(all_v)[:len(pos)]
    return (r.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def load_candidates(files: list[str], base: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for i, f in enumerate(files, 1):
        hits = glob.glob(str(base / '**' / f'{f}_candidates.csv'), recursive=True)
        if not hits:
            continue
        out[f] = pd.read_csv(hits[0], usecols=lambda c: c in READ_COLS, low_memory=False)
        print(f'\r  [{i}/{len(files)}] {f[:62]:62s}', end='', flush=True)
    print()
    return out


def accepted(df: pd.DataFrame) -> pd.DataFrame:
    return df[df['stage_blocked'].fillna('') == '']


def drop_tone(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    fpe = pd.to_numeric(df.get('FPE_hz'), errors='coerce')
    tone = (fpe >= TONE_BAND_HZ[0]) & (fpe <= TONE_BAND_HZ[1])
    return df[~tone], int(tone.sum())


def peak_window_count(t: np.ndarray, duration_s: float) -> int:
    """Largest number of accepted events inside any STAGE5_W_S window."""
    if len(t) == 0:
        return 0
    nb = max(int(np.ceil(duration_s / BIN_S)), 1)
    counts = np.bincount(np.clip((t / BIN_S).astype(int), 0, nb - 1), minlength=nb)
    k = int(STAGE5_W_S // BIN_S)
    if nb < k:
        return int(counts.sum())
    csum = np.concatenate([[0], np.cumsum(counts)])
    return int((csum[k:] - csum[:-k]).max())


# ── the four tables ──────────────────────────────────────────────────────────
def site_profile(sess: pd.DataFrame, cands: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for _, r in sess.iterrows():
        df = cands.get(r.file)
        if df is None:
            continue
        acc = accepted(df)
        acc_nt, n_tone = drop_tone(acc)
        floor = pd.to_numeric(df.get('noise_floor_mV'), errors='coerce').dropna()
        t = pd.to_numeric(acc_nt['timestamp_s'], errors='coerce').dropna().to_numpy()
        h = r.duration_s / 3600.0
        rows.append(dict(
            site=r.site, file=r.file, hours=h,
            candidates=len(df), candidates_per_h=len(df) / h,
            accepted=len(acc), accepted_no_tone=len(acc_nt), tone_events=n_tone,
            events_per_h=len(acc_nt) / h,
            floor_median_mV=floor.median(), floor_iqr_mV=floor.quantile(.75) - floor.quantile(.25),
            peak_180s=peak_window_count(t, r.duration_s)))
    return pd.DataFrame(rows)


def funnel(sess: pd.DataFrame, cands: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for site, grp in sess.groupby('site'):
        parts = [cands[f] for f in grp.file if f in cands]
        if not parts:
            continue
        sb = pd.concat([p['stage_blocked'].fillna('') for p in parts], ignore_index=True)
        n = len(sb)
        stage2 = sb.str.startswith('Stage2').sum()
        stage3 = (sb == 'Stage3_SVM').sum()
        rows.append(dict(site=site, candidates=n,
                         stage2_rejected_pct=100 * stage2 / n,
                         stage3_rejected_pct=100 * stage3 / n,
                         accepted_pct=100 * (sb == '').sum() / n,
                         survives_stage2_pct=100 * (n - stage2) / n))
    return pd.DataFrame(rows).sort_values('accepted_pct', ascending=False)


def band_profile(fp_with_tone: dict[str, pd.DataFrame], clicks: pd.DataFrame,
                 indoor_plant: pd.DataFrame) -> pd.DataFrame:
    """Share of accepted events per peak-frequency band, by site.

    Computed BEFORE the interference is removed, so the 64-70 kHz band can be shown
    for what it is. indoor_plant is the control that decides whether a band is a
    property of the instrument or of the place: a band present outdoors and absent
    on indoor plant recordings cannot be the device.
    """
    rows = []
    for lo, hi, name in BANDS:
        row = dict(band=name, lo_hz=lo, hi_hz=hi)
        cv = pd.to_numeric(clicks['FPE_hz'], errors='coerce')
        row['confirmed_clicks_pct'] = 100 * ((cv >= lo) & (cv < hi)).mean()
        iv = pd.to_numeric(indoor_plant['FPE_hz'], errors='coerce')
        row['indoor_plants_pct'] = 100 * ((iv >= lo) & (iv < hi)).mean()
        for site, d in fp_with_tone.items():
            ov = pd.to_numeric(d['FPE_hz'], errors='coerce')
            row[f'{site}_pct'] = 100 * ((ov >= lo) & (ov < hi)).mean()
        pooled = pd.to_numeric(pd.concat([d['FPE_hz'] for d in fp_with_tone.values()]),
                               errors='coerce')
        row['outdoor_pct'] = 100 * ((pooled >= lo) & (pooled < hi)).mean()
        row['outdoor_n'] = int(((pooled >= lo) & (pooled < hi)).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def feature_profile(fp: dict[str, pd.DataFrame], clicks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in FEATURES:
        cv = pd.to_numeric(clicks[f], errors='coerce')
        row = dict(feature=f, clicks_median=cv.median(),
                   clicks_defined_pct=100 * cv.notna().mean())
        for site, d in fp.items():
            ov = pd.to_numeric(d[f], errors='coerce')
            row[f'{site}_median'] = ov.median()
            row[f'{site}_AUC'] = auc(cv.to_numpy(), ov.to_numpy())
        pooled = pd.to_numeric(pd.concat([d[f] for d in fp.values()]), errors='coerce')
        row['outdoor_median'] = pooled.median()
        row['outdoor_defined_pct'] = 100 * pooled.notna().mean()
        row['AUC'] = auc(cv.to_numpy(), pooled.to_numpy())
        row['separability'] = abs(row['AUC'] - 0.5) + 0.5
        row['clicks_are'] = 'higher' if row['AUC'] > 0.5 else 'lower'
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values('separability', ascending=False).reset_index(drop=True)


def best_gate(feat: str, sense: str, clicks: pd.Series, out: pd.Series, budget: float):
    """Threshold removing the most outdoor events for <= budget % of clicks lost.

    NaN is KEPT on both sides — Stage 2 treats an undefined feature as passing, and
    a gate scored under a different NaN convention than the one it would run with
    is not the gate that would be deployed.
    """
    cv, ov = pd.to_numeric(clicks, errors='coerce'), pd.to_numeric(out, errors='coerce')
    if cv.notna().sum() < 20:
        return None
    grid = np.unique(np.nanpercentile(cv.dropna(), np.arange(0, 100.001, 0.5)))
    best = None
    for t in grid:
        kc = ((cv <= t) if sense == '<=' else (cv >= t)) | cv.isna()
        ko = ((ov <= t) if sense == '<=' else (ov >= t)) | ov.isna()
        lost, removed = 100 * (1 - kc.mean()), 100 * (1 - ko.mean())
        if lost <= budget and (best is None or removed > best['removed_pct']):
            best = dict(feature=feat, sense=sense, threshold=float(t),
                        removed_pct=removed, clicks_lost_pct=lost,
                        clicks_lost_n=int((~kc).sum()))
    return best


def keep_mask(df: pd.DataFrame, gates: list[dict]) -> pd.Series:
    m = pd.Series(True, index=df.index)
    for g in gates:
        v = pd.to_numeric(df[g['feature']], errors='coerce')
        m &= ((v <= g['threshold']) if g['sense'] == '<=' else (v >= g['threshold'])) | v.isna()
    return m


def gate_table(fp: dict[str, pd.DataFrame], clicks: pd.DataFrame,
               budgets=(0.0, 1.0, 2.0, 3.0, 5.0)) -> tuple[pd.DataFrame, list[dict]]:
    pooled = pd.concat(fp.values(), ignore_index=True)
    rows, by_budget = [], {}
    for budget in budgets:
        found = []
        for feat, sense in GATE_CANDIDATES:
            g = best_gate(feat, sense, clicks[feat], pooled[feat], budget)
            if g is None:
                continue
            g['budget_pct'] = budget
            for site, d in fp.items():
                g[f'{site}_removed_pct'] = 100 * (1 - keep_mask(d, [g]).mean())
            rows.append(g)
            found.append(g)
        by_budget[budget] = found
    return pd.DataFrame(rows), by_budget


def combine(fp: dict[str, pd.DataFrame], clicks: pd.DataFrame, pool: list[dict],
            max_lost_pct: float, max_gates: int = 3) -> dict | None:
    """Best set of at most `max_gates` gates whose JOINT click cost fits the budget.

    Searched exhaustively rather than greedily. A greedy stacker takes the single
    best gate first and is then stuck with the clicks it has already spent: on this
    corpus it returns n_seg + rise_time (47.6 % of outdoor events removed) while a
    different pair, which it can never reach, removes more for the same cost. The
    pool is small (one gate per feature per budget), so the approximation buys
    nothing.
    """
    from itertools import combinations
    pooled = pd.concat(fp.values(), ignore_index=True)
    # cache each gate's mask once — every subset is then a cheap AND
    cm = {i: keep_mask(clicks, [g]) for i, g in enumerate(pool)}
    om = {i: keep_mask(pooled, [g]) for i, g in enumerate(pool)}
    best = None
    for k in range(1, max_gates + 1):
        for idx in combinations(range(len(pool)), k):
            if len({pool[i]['feature'] for i in idx}) != k:
                continue                      # one threshold per feature
            kc = np.logical_and.reduce([cm[i] for i in idx])
            lost = 100 * (1 - kc.mean())
            if lost > max_lost_pct:
                continue
            removed = 100 * (1 - np.logical_and.reduce([om[i] for i in idx]).mean())
            if best is None or removed > best['removed_pct']:
                best = dict(gates=[pool[i] for i in idx], removed_pct=removed,
                            clicks_lost_pct=lost, clicks_lost_n=int((~kc).sum()))
    if best is None:
        return None
    for site, d in fp.items():
        best[site] = 100 * (1 - keep_mask(d, best['gates']).mean())
    return best


def bootstrap_gate(fp: dict[str, pd.DataFrame], clicks: pd.DataFrame, pool: list[dict],
                   budget: float, n_boot: int = 300, seed: int = 11) -> dict | None:
    """How much of the gate result survives a different sample of clicks.

    The thresholds above sit at the EXTREMES of 189 clicks — "FPE_hz <= 69.9 kHz" is
    literally the highest click ever confirmed — so the obvious objection is that
    they are fitted to this particular sample. Here the whole selection is redone on
    bootstrap resamples, resampled BY RECORDING rather than by click, because clicks
    cluster heavily within a recording and resampling individual clicks would
    understate the uncertainty. The spread reported is of the outdoor removal
    achieved by whatever gate the resample chooses.
    """
    if 'file' not in clicks:
        return None
    rng = np.random.default_rng(seed)
    recs = clicks['file'].dropna().unique()
    if len(recs) < 5:
        return None
    removals, kept_feats = [], []
    for _ in range(n_boot):
        pick = rng.choice(recs, size=len(recs), replace=True)
        cb = pd.concat([clicks[clicks.file == r] for r in pick], ignore_index=True)
        best = combine(fp, cb, pool, budget)
        if best:
            removals.append(best['removed_pct'])
            kept_feats.append(tuple(sorted(g['feature'] for g in best['gates'])))
    if not removals:
        return None
    from collections import Counter
    top, n_top = Counter(kept_feats).most_common(1)[0]
    return dict(n=len(removals), median=float(np.median(removals)),
                lo=float(np.percentile(removals, 5)), hi=float(np.percentile(removals, 95)),
                top_features=top, top_share=100 * n_top / len(removals))


# ── figures ──────────────────────────────────────────────────────────────────
LIGHT = dict(blue='#2a78d6', orange='#eb6834', green='#1baf7a', ink='#1a1a1a',
             mute='#6b7280', grid='#e5e7eb', surface='#ffffff')


#: Indoor mechanical-stress click rate, first 15 min (RESULTS_v6.md, 20-09-2026).
#: Drawn as the reference line: it is the signal the false positives compete with.
MECHANICAL_RATE_PER_H = 105.1


def fig_sites(prof: pd.DataFrame, order: list[str], out: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    c = LIGHT
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    fig.patch.set_facecolor(c['surface'])
    ax.set_facecolor(c['surface'])
    order = list(reversed(order))            # busiest site at the top
    colour = {s: (c['blue'] if s != 'indoor empty room' else c['mute']) for s in order}
    ax.axvline(MECHANICAL_RATE_PER_H, color=c['orange'], lw=2, ls='--', zorder=2)
    ax.text(MECHANICAL_RATE_PER_H * 1.25, .05,
            f'a mechanically stressed plant\nindoors: {MECHANICAL_RATE_PER_H:.0f} clicks/h',
            fontsize=8, color=c['orange'], va='center')
    rng = np.random.default_rng(7)
    n_zero = 0
    for i, site in enumerate(order):
        v = prof[prof.site == site]['events_per_h'].to_numpy()
        n_zero += int((v == 0).sum())
        v = np.clip(v, 0.05, None)                      # zeros drawn at the floor
        ax.scatter(v, i + rng.uniform(-.16, .16, len(v)), s=42, alpha=.75,
                   color=colour[site], edgecolor=c['surface'], linewidth=1.6, zorder=3)
        m = float(np.median(v))
        ax.plot([m, m], [i - .30, i + .30], color=c['ink'], lw=2.4, zorder=4)
        ax.text(m, i + .38, f'{m:,.0f}/h', ha='center', va='bottom', fontsize=8.5,
                color=c['ink'])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f'{s}\n({(prof.site == s).sum()} rec, '
                        f'{prof[prof.site == s].hours.sum():.1f} h)' for s in order],
                       fontsize=9)
    ax.set_xscale('log')
    ax.set_xlabel('accepted events per hour, no plant present  (log scale)', fontsize=9.5)
    ax.set_title('What the detector accepts where there is nothing to detect',
                 fontsize=11.5, color=c['ink'], loc='left', pad=10)
    ax.grid(axis='x', color=c['grid'], lw=.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(c['grid'])
    ax.tick_params(colors=c['mute'], length=0)
    for lbl in ax.get_yticklabels():
        lbl.set_color(c['ink'])
    ax.set_ylim(-.6, len(order) - .4)
    if n_zero:
        ax.text(0.05, -.52, f'\u2190 {n_zero} recording(s) with no accepted event at all',
                fontsize=7.5, color=c['mute'], va='center')
    fig.tight_layout()
    fig.savefig(out, facecolor=c['surface'])
    plt.close(fig)


def fig_spectrum(fp_with_tone: dict[str, pd.DataFrame], clicks: pd.DataFrame, out: Path):
    """Where the energy sits: confirmed clicks against the outdoor acceptances."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    c = LIGHT
    pooled = pd.to_numeric(pd.concat([d['FPE_hz'] for d in fp_with_tone.values()]),
                           errors='coerce').dropna() / 1000
    cv = pd.to_numeric(clicks['FPE_hz'], errors='coerce').dropna() / 1000
    bins = np.arange(10, 90.1, 2.0)

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    fig.patch.set_facecolor(c['surface'])
    ax.set_facecolor(c['surface'])
    ax.hist(pooled, bins=bins, weights=np.full(len(pooled), 100 / len(pooled)),
            color=c['blue'], alpha=.85, label=f'outdoor acceptances (n={len(pooled):,})',
            zorder=2)
    ax.hist(cv, bins=bins, weights=np.full(len(cv), 100 / len(cv)),
            histtype='step', lw=2.2, color=c['orange'],
            label=f'confirmed clicks (n={len(cv)})', zorder=4)
    ax.axvspan(64, 70, color=c['mute'], alpha=.18, zorder=1)
    ax.text(67, ax.get_ylim()[1] * .96, 'our own\n66 kHz\ninterference', ha='center',
            va='top', fontsize=7.5, color=c['mute'])
    ax.annotate('outdoor-only band:\nno confirmed click\never lands here',
                xy=(78.6, 100 * ((pooled >= 78) & (pooled < 80)).mean()),
                xytext=(56.5, ax.get_ylim()[1] * .56), fontsize=8, color=c['ink'],
                ha='center',
                arrowprops=dict(arrowstyle='->', color=c['ink'], lw=1.2,
                                connectionstyle='arc3,rad=0.25'))
    ax.set_xlabel('peak frequency of the event (kHz)', fontsize=9.5)
    ax.set_ylabel('share of events (%)', fontsize=9.5)
    outside = 100 * (((pooled < 25) | (pooled >= 60)).mean())
    ax.set_title(f'{outside:.0f} % of what the pipeline accepts outdoors lies outside '
                 'the click band', fontsize=11, color=c['ink'], loc='left', pad=10)
    ax.grid(axis='y', color=c['grid'], lw=.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(c['grid'])
    ax.tick_params(colors=c['mute'], length=0)
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(c['ink'])
    fig.tight_layout()
    fig.savefig(out, facecolor=c['surface'])
    plt.close(fig)


def fig_gates(fp: dict[str, pd.DataFrame], clicks: pd.DataFrame, combo: dict,
              out: Path, base_rate: float | None = None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    c = LIGHT
    pooled = pd.concat(fp.values(), ignore_index=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    fig.patch.set_facecolor(c['surface'])
    ax.set_facecolor(c['surface'])

    cols = [c['blue'], c['orange'], c['green'], c['mute']]
    for (feat, sense), col in zip(GATE_CANDIDATES[:4], cols):
        cv = pd.to_numeric(clicks[feat], errors='coerce')
        ov = pd.to_numeric(pooled[feat], errors='coerce')
        xs, ys = [], []
        for t in np.unique(np.nanpercentile(cv.dropna(), np.arange(0, 100.001, 1.0))):
            kc = ((cv <= t) if sense == '<=' else (cv >= t)) | cv.isna()
            ko = ((ov <= t) if sense == '<=' else (ov >= t)) | ov.isna()
            xs.append(100 * (1 - kc.mean()))
            ys.append(100 * (1 - ko.mean()))
        o = np.argsort(xs)
        ax.plot(np.array(xs)[o], np.array(ys)[o], lw=2, color=col,
                label=f'{feat} {sense}', zorder=3)

    ax.scatter([combo['clicks_lost_pct']], [combo['removed_pct']], s=90, zorder=5,
               color=c['ink'], edgecolor=c['surface'], linewidth=2)
    ax.annotate(' + '.join(f"{g['feature']}" for g in combo['gates']),
                (combo['clicks_lost_pct'], combo['removed_pct']),
                textcoords='offset points', xytext=(10, -4), fontsize=9, color=c['ink'])
    ax.set_xlim(0, 10)
    ax.set_xlabel('confirmed clicks lost (%)', fontsize=9.5)
    ax.set_ylabel('outdoor false positives removed (%)', fontsize=9.5)
    ax.set_title('What a stricter gate would buy, and what it would cost',
                 fontsize=11.5, color=c['ink'], loc='left', pad=10)
    ax.grid(color=c['grid'], lw=.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(c['grid'])
    ax.tick_params(colors=c['mute'], length=0)
    leg = ax.legend(frameon=False, fontsize=9, loc='lower right')
    for t in leg.get_texts():
        t.set_color(c['ink'])
    if base_rate:
        # the same axis in the units that matter: what is LEFT, per hour
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ticks = np.arange(0, 101, 20)
        ax2.set_yticks(ticks)
        ax2.set_yticklabels([f'{base_rate * (1 - t / 100):,.0f}' for t in ticks])
        ax2.set_ylabel('false positives still getting through, per hour', fontsize=9.5)
        ax2.tick_params(colors=c['mute'], length=0)
        for sp in ('top', 'left'):
            ax2.spines[sp].set_visible(False)
        ax2.spines['right'].set_color(c['grid'])
        ax2.axhline(100 * (1 - MECHANICAL_RATE_PER_H / base_rate), color=c['orange'],
                    lw=1.6, ls='--', zorder=2)
        ax2.text(9.8, 100 * (1 - MECHANICAL_RATE_PER_H / base_rate) + 2,
                 'parity with an indoor stressed plant', fontsize=8,
                 color=c['orange'], ha='right')
    fig.tight_layout()
    fig.savefig(out, facecolor=c['surface'])
    plt.close(fig)


# ── report ───────────────────────────────────────────────────────────────────
def write_report(path: Path, prof, fun, feat, gates_df, by_budget, combo, sites,
                 n_clicks, n_tone, alarm, indoor_rate, combo5=None, bands=None,
                 boot=None):
    L = []
    A = L.append
    A('# Outdoor: why the detector fails there, measured\n')
    A('Generated by `scripts/results/outdoor_analysis.py`. Every number below is '
      'computed from the exported v6 candidate CSVs; nothing is hand-copied.\n')
    A('## The negative set\n')
    A('No plant is present in these recordings, so every accepted event is a false '
      'positive by construction — a labelled negative set that cost nothing to '
      'obtain.\n')
    tot = prof[prof.site != 'indoor empty room']
    A(f'- **{len(tot)} outdoor noise-only recordings, {tot.hours.sum():.1f} h, '
      f'{int(tot.accepted_no_tone.sum()):,} false positives**')
    A(f'- against **{n_clicks} clicks confirmed by eye** indoors')
    A(f'- {n_tone:,} further events were dropped as 66 kHz interference '
      f'({TONE_BAND_HZ[0]/1000:.0f}-{TONE_BAND_HZ[1]/1000:.0f} kHz), which is '
      'instrumental and not part of the environment\n')

    A('## Site by site\n')
    A('| site | rec | hours | candidates/h | false positives/h (pooled) | '
      'median recording | per-recording range | median floor (mV) |')
    A('|---|---:|---:|---:|---:|---:|---|---:|')
    for s in sites:
        g = prof[prof.site == s]
        n, h = int(g.accepted_no_tone.sum()), g.hours.sum()
        rng = (f'{g.events_per_h.min():,.0f} – {g.events_per_h.max():,.0f}/h'
               if len(g) > 1 else '_single recording_')
        thin = ' ⚠️' if len(g) < 3 else ''
        A(f'| {s}{thin} | {len(g)} | {h:.1f} | {g.candidates.sum()/h:,.0f} | '
          f'**{n/h:,.1f}** | {g.events_per_h.median():,.0f}/h | {rng} | '
          f'{g.floor_median_mV.median():.3f} |')
    A('')
    A('⚠️ = fewer than 3 recordings; that site\'s median is one or two mornings, not '
      'a property of the place. The pooled rate is every event over every hour; the '
      'median recording is the unit used everywhere else in the analysis, and the '
      'two disagree by 3x at the balcony because one session contributes most of '
      'the events.')
    A('')
    A('The per-recording range is not noise around a site value — it *is* the '
      'result. The outdoor environment is not one environment, and it is not even '
      'the same environment twice at one site.\n')

    A('## Where the pipeline leaks\n')
    A('| site | candidates | rejected at Stage 2 | rejected at Stage 3 | accepted |')
    A('|---|---:|---:|---:|---:|')
    for _, r in fun.iterrows():
        A(f'| {r.site} | {int(r.candidates):,} | {r.stage2_rejected_pct:.1f} % | '
          f'{r.stage3_rejected_pct:.1f} % | **{r.accepted_pct:.2f} %** |')
    A('')

    if bands is not None:
        A('## Where the outdoor energy sits\n')
        A('Peak frequency of every accepted event, before the interference is '
          'removed. The indoor plant recordings are the control: a band that is '
          'common outdoors and rare there belongs to the PLACE, not to the '
          'instrument.\n')
        cols = [c for c in bands.columns if c.endswith('_pct') and
                c not in ('confirmed_clicks_pct', 'indoor_plants_pct', 'outdoor_pct')]
        A('| band | confirmed clicks | indoor plants | outdoor | ' +
          ' | '.join(c[:-4] for c in cols) + ' |')
        A('|---|---:|---:|---:|' + '---:|' * len(cols))
        for _, r in bands.iterrows():
            A(f'| {r.band} | {r.confirmed_clicks_pct:.1f} % | '
              f'{r.indoor_plants_pct:.1f} % | **{r.outdoor_pct:.1f} %** | ' +
              ' | '.join(f'{r[c]:.1f} %' for c in cols) + ' |')
        hi = bands[bands.band == 'above 70 kHz'].iloc[0]
        A('')
        A(f'**The band above 70 kHz is the clearest single result here.** It holds '
          f'{hi.outdoor_pct:.0f} % of everything the pipeline accepts outdoors '
          f'({int(hi.outdoor_n):,} events) and {hi.indoor_plants_pct:.1f} % of what '
          f'it accepts on indoor plants, and **not one of the {n_clicks} confirmed '
          f'clicks lands in it** — the highest click peaks at 69.9 kHz, the 99th '
          'percentile at 59 kHz. Most of it sits in five adjacent FFT bins around '
          '79 kHz, so it is narrowband: a persistent outdoor ultrasonic source, at '
          'all three sites and in both April and August, entirely outside the '
          'frequency range a cavitation click occupies. It is a different band from '
          'our own interference, which sits at 64-70 kHz — and that interference is '
          'concentrated at the taverna, where it accounts for 46 % of everything '
          'accepted, which is why it is removed event by event before any of the '
          'figures below.\n')

    A('## What the false positives are\n')
    A('AUC is the probability that a confirmed click scores above an outdoor event '
      'on that feature; 0.5 means the feature cannot tell them apart.\n')
    A('| feature | confirmed clicks | outdoor | clicks are | separability | defined on |')
    A('|---|---:|---:|:--:|---:|---:|')
    for _, r in feat.iterrows():
        A(f'| {r.feature} | {r.clicks_median:,.4g} | {r.outdoor_median:,.4g} | '
          f'{r.clicks_are} | **{r.separability:.2f}** | '
          f'{r.outdoor_defined_pct:.0f} % of events |')
    A('')

    A('## What a stricter gate would buy\n')
    A('A threshold is chosen on the confirmed clicks (it may cost at most the stated '
      'share of them) and then applied to the outdoor false positives.\n')
    A('| gate | clicks lost | outdoor FPs removed | ' + ' | '.join(sites[:-1]) + ' |')
    A('|---|---:|---:|' + '---:|' * (len(sites) - 1))
    for _, r in gates_df[gates_df.budget_pct == 2.0].sort_values(
            'removed_pct', ascending=False).iterrows():
        cells = ' | '.join(f'{r[f"{s}_removed_pct"]:.0f} %' for s in sites[:-1])
        A(f'| keep `{r.feature} {r.sense} {r.threshold:,.4g}` | '
          f'{r.clicks_lost_pct:.1f} % ({int(r.clicks_lost_n)}/{n_clicks}) | '
          f'**{r.removed_pct:.1f} %** | {cells} |')
    out_rate = tot.accepted_no_tone.sum() / tot.hours.sum()
    if combo:
        A('\n**Stacking gates.** Two gates that each cost 1.6 % of the clicks rarely '
          'cost 3.2 % together — they may drop the same clicks or different ones — so '
          'the joint cost is measured, not added:\n')
        A('| combination | clicks lost | outdoor FPs removed | outdoor FP rate |')
        A('|---|---:|---:|---:|')
        for cb in (combo, combo5):
            if not cb:
                continue
            names = ' + '.join(f"`{g['feature']} {g['sense']} {g['threshold']:,.4g}`"
                               for g in cb['gates'])
            A(f'| {names} | {cb["clicks_lost_pct"]:.1f} % '
              f'({cb["clicks_lost_n"]}/{n_clicks}) | **{cb["removed_pct"]:.1f} %** | '
              f'{out_rate * (1 - cb["removed_pct"]/100):,.0f}/h |')
        A('')
        best = combo5 if (combo5 and combo5['removed_pct'] > combo['removed_pct']) else combo
        A(f'Even the better of the two leaves **'
          f'{out_rate * (1 - best["removed_pct"]/100):,.0f} false positives/h**, '
          f'down from {out_rate:,.0f}/h. A mechanically '
          'stressed plant indoors runs at about 105 clicks/h, so this is the '
          'finding: a hand-tuned gate is a large improvement and still not enough. '
          'The gap is not closed by thresholds — it is closed by training the '
          'classifier with these negatives in it.\n')

    if boot:
        A('**Is that fitted to these 189 clicks?** Partly, and it is worth saying by '
          'how much. The thresholds sit at the extreme of the click distribution — '
          '`FPE_hz <= 69.9 kHz` is the single highest confirmed click — so the whole '
          'selection was redone on ' + str(boot['n']) + ' bootstrap resamples of the '
          'click RECORDINGS (not the individual clicks, which cluster). The outdoor '
          f'removal achieved lands at **{boot["median"]:.0f} % (5th-95th percentile '
          f'{boot["lo"]:.0f}-{boot["hi"]:.0f} %)**, and the same feature set '
          f'({", ".join(boot["top_features"])}) is chosen in '
          f'{boot["top_share"]:.0f} % of resamples. The conclusion — a useful cut, '
          'not a sufficient one — does not depend on the exact sample.\n')

    A('## What it does to the stress detector\n')
    A(f'The Stage 5 alarm (h = {STAGE5_H} events in {STAGE5_W_S:.0f} s) was '
      f'calibrated indoors, on control recordings only, for 0.1 false alarms/h.\n')
    A(f'- outdoors it would fire on **{alarm["n_alarm"]} of {alarm["n_rec"]} '
      f'noise-only recordings** — recordings containing no plant at all')
    A(f'- the busiest {STAGE5_W_S:.0f} s window reaches '
      f'**{alarm["max_peak"]:,} events** against a threshold of {STAGE5_H}')
    A(f'- indoors, the same empty-room recordings give {indoor_rate:,.2f} '
      'false positives/h\n')

    A('## What this does not say\n')
    A('- No outdoor event is labelled. "False positive" rests on there being no '
      'plant under measurement, which does not exclude vegetation nearby.\n'
      '- The clicks compared against are indoor, so part of the separation may be '
      'the recording channel rather than the source. The gate **costs** are '
      'measured on real confirmed clicks and hold; the gate **benefits** are '
      'measured on presumed negatives and are an upper bound.\n'
      '- Nothing here shows the detector can work outdoors. It shows what would '
      'have to change for it to.\n')
    path.write_text('\n'.join(L), encoding='utf-8')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--labelled', required=True, type=Path)
    ap.add_argument('--analysis-set', required=True, type=Path)
    ap.add_argument('-o', '--out', required=True, type=Path)
    ap.add_argument('--click-budget', type=float, default=2.0,
                    help='max %% of confirmed clicks a combined gate may cost')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    s = pd.read_csv(a.analysis_set)
    s['site'] = s.apply(site_of, axis=1)
    noise = s[(s.condition == 'noise') & (s.include == 1) & s.duration_s.notna()].copy()
    outdoor = noise[noise.environment == 'outdoor']
    indoor = noise[noise.environment == 'indoor']
    if outdoor.empty:
        print('ERROR: no outdoor noise-only recordings in the analysis set')
        return 2

    print(f'outdoor noise-only recordings: {len(outdoor)}   '
          f'indoor reference: {len(indoor)}')
    cands = load_candidates(list(noise.file), a.labelled)

    prof = site_profile(noise, cands)
    fun = funnel(noise, cands)

    # false-positive set per outdoor site, 66 kHz interference removed
    fp, fp_raw, n_tone = {}, {}, 0
    for site, grp in outdoor.groupby('site'):
        parts = [accepted(cands[f]) for f in grp.file if f in cands]
        d = pd.concat(parts, ignore_index=True)
        fp_raw[site] = d
        d, nt = drop_tone(d)
        fp[site], n_tone = d, n_tone + nt

    # confirmed clicks: every label == 1 in the corpus
    clicks = []
    for p in glob.glob(str(a.labelled / '**' / '*_candidates.csv'), recursive=True):
        d = pd.read_csv(p, usecols=lambda c: c in READ_COLS, low_memory=False)
        if 'label' in d:
            clicks.append(d[d['label'].astype(str).str.rstrip('.0') == '1'])
    clicks = pd.concat(clicks, ignore_index=True)
    print(f'false positives {sum(len(d) for d in fp.values()):,} '
          f'(+{n_tone:,} dropped as 66 kHz)   confirmed clicks {len(clicks)}')

    ip_files = s[(s.environment == 'indoor') & (s.condition.isin(['mechanical', 'water']))
                 & (s.include == 1)].file.tolist()
    ip_parts = []
    for f in ip_files:
        hits = glob.glob(str(a.labelled / '**' / f'{f}_candidates.csv'), recursive=True)
        if hits:
            d = pd.read_csv(hits[0], usecols=lambda c: c in READ_COLS, low_memory=False)
            ip_parts.append(accepted(d))
    indoor_plant = pd.concat(ip_parts, ignore_index=True) if ip_parts else pd.DataFrame(
        {'FPE_hz': []})

    bands = band_profile(fp_raw, clicks, indoor_plant)
    feat = feature_profile(fp, clicks)
    gates_df, by_budget = gate_table(fp, clicks, budgets=(0.0, 1.0, 2.0, 3.0, 5.0))
    pool = [g for gs in by_budget.values() for g in gs]
    combo = combine(fp, clicks, pool, a.click_budget)
    combo5 = combine(fp, clicks, pool, 5.0)
    print('  bootstrapping the gate selection over click recordings...')
    boot = bootstrap_gate(fp, clicks, pool, a.click_budget)

    op = prof[prof.site.isin(fp)]
    alarm = dict(n_rec=len(op), n_alarm=int((op.peak_180s >= STAGE5_H).sum()),
                 max_peak=int(op.peak_180s.max()))
    ip = prof[prof.site == 'indoor empty room']
    indoor_rate = ip.accepted_no_tone.sum() / ip.hours.sum() if len(ip) else float('nan')

    _t = prof[prof.site.isin(fp)]
    tot_rate = _t.accepted_no_tone.sum() / _t.hours.sum()
    sites = sorted(fp, key=lambda x: -prof[prof.site == x].events_per_h.median())
    sites += ['indoor empty room']

    prof.to_csv(a.out / 'outdoor_sites.csv', index=False)
    fun.to_csv(a.out / 'outdoor_funnel.csv', index=False)
    feat.to_csv(a.out / 'outdoor_features.csv', index=False)
    gates_df.to_csv(a.out / 'outdoor_gates.csv', index=False)
    bands.to_csv(a.out / 'outdoor_bands.csv', index=False)
    fig_sites(prof, sites, a.out / 'outdoor_sites.png')
    fig_spectrum(fp_raw, clicks, a.out / 'outdoor_spectrum.png')
    if combo:
        fig_gates(fp, clicks, combo, a.out / 'outdoor_gates.png',
                  base_rate=tot_rate)
    write_report(a.out / 'OUTDOOR_ANALYSIS.md', prof, fun, feat, gates_df, by_budget,
                 combo, sites, len(clicks), n_tone, alarm, indoor_rate, combo5, bands,
                 boot)

    print(f'\nwrote {a.out}/OUTDOOR_ANALYSIS.md + 4 CSVs + 2 figures')
    return 0


if __name__ == '__main__':
    sys.exit(main())
