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
stress_premises.py — verify the three premises of STRESS_DETECTION_ALGORITHM_v1 §2

    python3 scripts/stress/stress_premises.py --corpus ~/PlantLeaf_dev/Analisi/v6/stress

WHY THIS RUNS BEFORE ANY VALIDATION
-----------------------------------
W and R are "set from burst duration / burst decay (P1)" (§4.2, §6). If they were
chosen by looking at LOSO sensitivity, the validation would be tuning on its own
test set. So this script measures the burst structure — and ONLY this script's
output is used to fix W and R. stress_calibrate_validate.py then sweeps them as a
robustness check around that a-priori point, never to pick it.

WHAT IT MEASURES
----------------
P1  clustering    Fano factor per session at 60 / 300 / 600 s complete bins.
    burst length  gap clustering (events <= tau apart are one burst), tau = 10/30/60 s:
                  span of each session's largest burst; and the window-saturation
                  curve — median max-count in any W window vs W. Where the stimulus
                  curve flattens is the burst duration, independent of tau.
P2  transience    share of each session's events inside its densest W window, and
                  where that window sits in the recording.
P3  quiet control per-condition rates with exact Poisson 95 % CIs. These pool events
                  over sessions and are CONTEXT ONLY (§9.1) — per-session rates
                  are in session_premises.csv.
§2.1 artefacts    noise floor and event rate in the first 5 min vs the rest, per
                  session, with a session-level Wilcoxon signed-rank test.
§13 de-bursting   Fano of the burst-event process (one event per burst, tau = 30 s).

OUTPUTS  (--corpus/premises/)
    premises.md, session_premises.csv, rates_by_condition.csv,
    window_saturation.csv, clusters.csv, raster.png, saturation.png, burst_span.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stress_common import (   # noqa: E402
    BASELINE, GRID, GROUP_COLOR, INK, INK_2, PROBE, SECONDARY, STIMULUS,
    load_corpus, md_table, poisson_ci, style_axes, times_by_file, window_max_count,
)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src' / 'ml'))
from analyze_dataset import bin_counts   # noqa: E402

FANO_BINS_S = (60, 300, 600)
TAUS_S = (10, 30, 60)
TAU_MAIN = 30
SATURATION_W = (10, 20, 30, 60, 90, 120, 180, 240, 300, 450, 600, 900, 1200, 1800)
EARLY_S = 300
GROUPS_SHOWN = (BASELINE, STIMULUS, SECONDARY)


def fano(counts: np.ndarray) -> float:
    if counts.size < 2 or counts.mean() == 0:
        return float('nan')
    return float(counts.var(ddof=1) / counts.mean())


def clusters(t: np.ndarray, tau: float) -> list[tuple[int, float, float]]:
    """(n, t_start, t_end) of each run of events with consecutive gaps <= tau."""
    if t.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(t) > tau) + 1
    return [(int(s.size), float(s[0]), float(s[-1])) for s in np.split(t, cuts)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--corpus', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    out = a.out or a.corpus / 'premises'
    out.mkdir(parents=True, exist_ok=True)

    ses, ev = load_corpus(a.corpus)
    ses = ses[ses['group'].isin((BASELINE, STIMULUS, SECONDARY, PROBE))].copy()
    ses = ses[np.isfinite(ses['duration_s'])]
    tf = times_by_file(ev)
    floor = pd.read_csv(a.corpus / 'floor.csv')
    floor_by = {f: g for f, g in floor.groupby('file')}
    empty = (np.empty(0), np.empty(0))

    rows, cl_rows, sat_rows = [], [], []
    for _, s in ses.iterrows():
        t, _ = tf.get(s['file'], empty)
        T = float(s['duration_s'])
        r = dict(file=s['file'], group=s['group'], environment=s['environment'],
                 condition=s['condition'], needs_review=s['needs_review'],
                 duration_s=T, n=int(t.size), rate_h=t.size / (T / 3600))
        for b in FANO_BINS_S:
            r[f'fano_{b}'] = fano(bin_counts(t, T, b))
        for W in (60, 300, 600):
            m, tp = window_max_count(t, W)
            r[f'max_{W}'] = m
            r[f'share_{W}'] = m / t.size if t.size else float('nan')
            if W == 300:
                r['t_peak_300'] = tp
                r['peak_rel_pos_300'] = tp / T if t.size else float('nan')
        for W in SATURATION_W:
            sat_rows.append(dict(file=s['file'], group=s['group'],
                                 environment=s['environment'], W=W,
                                 max_count=window_max_count(t, W)[0]))
        for tau in TAUS_S:
            cs = clusters(t, tau)
            for n, a0, a1 in cs:
                cl_rows.append(dict(file=s['file'], group=s['group'],
                                    environment=s['environment'], tau=tau,
                                    n=n, t_start=a0, span_s=a1 - a0))
            big = max(cs, key=lambda c: (c[0], -(c[2] - c[1])), default=None)
            r[f'big_n_tau{tau}'] = big[0] if big else 0
            r[f'big_span_tau{tau}'] = big[2] - big[1] if big else float('nan')
            if tau == TAU_MAIN:
                r['big_share_tau30'] = big[0] / t.size if big else float('nan')
                # t90: time from burst start until 90 % of its events have arrived
                if big and big[0] >= 3:
                    inb = t[(t >= big[1]) & (t <= big[2])]
                    r['big_t90_tau30'] = float(np.quantile(inb - inb[0], 0.9))
                else:
                    r['big_t90_tau30'] = float('nan')
                starts = np.array([c[1] for c in cs])
                r['fano_300_debursted'] = fano(bin_counts(starts, T, 300))
        # §2.1 warm-up
        if T >= 3 * EARLY_S:
            r['rate_early_h'] = np.sum(t < EARLY_S) / (EARLY_S / 3600)
            r['rate_late_h'] = np.sum(t >= EARLY_S) / ((T - EARLY_S) / 3600)
        else:
            r['rate_early_h'] = r['rate_late_h'] = float('nan')
        fl = floor_by.get(s['file'])
        if fl is not None and (fl['t'] < EARLY_S).sum() >= 5 and (fl['t'] >= EARLY_S).sum() >= 5:
            r['floor_early_mV'] = float(fl.loc[fl['t'] < EARLY_S, 'noise_floor_mV'].median())
            r['floor_late_mV'] = float(fl.loc[fl['t'] >= EARLY_S, 'noise_floor_mV'].median())
        else:
            r['floor_early_mV'] = r['floor_late_mV'] = float('nan')
        rows.append(r)

    sp = pd.DataFrame(rows)
    cl = pd.DataFrame(cl_rows)
    sat = pd.DataFrame(sat_rows)
    sp.to_csv(out / 'session_premises.csv', index=False)
    cl.to_csv(out / 'clusters.csv', index=False)

    indoor = sp[sp['environment'] == 'indoor']

    # ── P3 rates ─────────────────────────────────────────────────────────────
    rate_rows = []
    for (grp, env, cond), g in sp.groupby(['group', 'environment', 'condition']):
        k, T_h = int(g['n'].sum()), g['duration_s'].sum() / 3600
        lo, hi = poisson_ci(k, T_h)
        rate_rows.append(dict(group=grp, env=env, condition=cond, sessions=len(g),
                              hours=T_h, events=k, rate_h=k / T_h, ci_lo=lo, ci_hi=hi,
                              session_median_rate_h=g['rate_h'].median(),
                              sessions_zero=int((g['n'] == 0).sum())))
    rates = pd.DataFrame(rate_rows).sort_values(['env', 'group', 'condition'])
    rates.to_csv(out / 'rates_by_condition.csv', index=False)

    # ── P1 burst length ──────────────────────────────────────────────────────
    satg = (sat[sat['environment'] == 'indoor']
            .groupby(['group', 'W'])['max_count']
            .agg(median='median', q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75),
                 mean='mean').reset_index())
    satg.to_csv(out / 'window_saturation.csv', index=False)

    def sat_knee(grp: str, frac: float) -> float:
        """Smallest W whose mean max-count reaches `frac` of the mean at the largest W."""
        g = satg[satg['group'] == grp].sort_values('W')
        top = g['mean'].iloc[-1]
        hit = g[g['mean'] >= frac * top]
        return float(hit['W'].iloc[0]) if len(hit) else float('nan')

    per_session_knee = []
    for f, g in sat[(sat['environment'] == 'indoor') & (sat['group'] == STIMULUS)].groupby('file'):
        g = g.sort_values('W')
        top = g['max_count'].iloc[-1]
        if top >= 5:
            per_session_knee.append(dict(file=f, max_at_1800=top,
                                         W50=float(g.loc[g['max_count'] >= .5 * top, 'W'].iloc[0]),
                                         W90=float(g.loc[g['max_count'] >= .9 * top, 'W'].iloc[0])))
    knees = pd.DataFrame(per_session_knee)

    def span_summary(df: pd.DataFrame, col: str) -> dict:
        x = df[col].dropna()
        return dict(n=len(x), median=x.median(), p75=x.quantile(.75),
                    p90=x.quantile(.9), max=x.max())

    burst_tab = []
    for grp in GROUPS_SHOWN:
        g = indoor[(indoor['group'] == grp) & (indoor[f'big_n_tau{TAU_MAIN}'] >= 5)]
        for tau in TAUS_S:
            burst_tab.append(dict(group=grp, tau_s=tau, **span_summary(
                indoor[(indoor['group'] == grp) & (indoor[f'big_n_tau{tau}'] >= 5)],
                f'big_span_tau{tau}')))
        burst_tab.append(dict(group=grp, tau_s='t90@30', **span_summary(g, 'big_t90_tau30')))
    burst_tab = pd.DataFrame(burst_tab)

    # inter-burst gaps inside stimulus sessions: what R has to span
    big_cl = cl[(cl['environment'] == 'indoor') & (cl['tau'] == TAU_MAIN) & (cl['n'] >= 3)]
    gaps = []
    for (f, grp), g in big_cl.groupby(['file', 'group']):
        g = g.sort_values('t_start')
        ends = (g['t_start'] + g['span_s']).to_numpy()
        for k in range(1, len(g)):
            gaps.append(dict(file=f, group=grp, gap_s=g['t_start'].iloc[k] - ends[k - 1]))
    gaps = pd.DataFrame(gaps, columns=['file', 'group', 'gap_s'])

    # ── §2.1 warm-up ─────────────────────────────────────────────────────────
    warm = []
    for grp in GROUPS_SHOWN:
        g = indoor[indoor['group'] == grp].dropna(subset=['rate_early_h', 'rate_late_h'])
        d = g['rate_early_h'] - g['rate_late_h']
        p = wilcoxon(g['rate_early_h'], g['rate_late_h']).pvalue if (d != 0).sum() >= 6 else float('nan')
        gf = indoor[indoor['group'] == grp].dropna(subset=['floor_early_mV', 'floor_late_mV'])
        ratio = gf['floor_early_mV'] / gf['floor_late_mV']
        warm.append(dict(group=grp, sessions=len(g),
                         early_gt_late=int((d > 0).sum()), early_lt_late=int((d < 0).sum()),
                         pooled_early_h=g['rate_early_h'].mean(), pooled_late_h=g['rate_late_h'].mean(),
                         wilcoxon_p=p, floor_sessions=len(gf),
                         floor_early_over_late_median=ratio.median(),
                         floor_early_lower_frac=float((ratio < 1).mean()) if len(gf) else float('nan')))
    warm = pd.DataFrame(warm)

    peak_early = []
    for grp in GROUPS_SHOWN:
        g = indoor[(indoor['group'] == grp) & (indoor['max_300'] >= 3)]
        peak_early.append(dict(group=grp, sessions_with_max300_ge3=len(g),
                               peak_in_first_10min=int((g['t_peak_300'] <= 600).sum()),
                               median_peak_rel_pos=g['peak_rel_pos_300'].median(),
                               median_share_300=g['share_300'].median()))
    peak_early = pd.DataFrame(peak_early)

    fano_tab = (indoor.groupby('group')[['fano_60', 'fano_300', 'fano_600', 'fano_300_debursted']]
                .median().reindex(list(GROUPS_SHOWN)).reset_index())
    fano_n = indoor.groupby('group')['fano_300'].apply(lambda x: int(x.notna().sum()))
    fano_tab.insert(1, 'sessions_with_fano', fano_tab['group'].map(fano_n))
    fano_gt2 = indoor.groupby('group')['fano_300'].apply(lambda x: f"{int((x > 2).sum())}/{int(x.notna().sum())}")
    fano_tab['fano_300_gt_2'] = fano_tab['group'].map(fano_gt2)

    top_base = (indoor[indoor['group'] == BASELINE].sort_values('max_300', ascending=False)
                [['file', 'condition', 'duration_s', 'n', 'max_300', 't_peak_300', 'fano_300',
                  'big_span_tau30', 'needs_review']].head(8))
    stim_tab = (indoor[indoor['group'] == STIMULUS].sort_values('max_300', ascending=False)
                [['file', 'condition', 'duration_s', 'n', 'max_60', 'max_300', 'share_300',
                  't_peak_300', 'big_n_tau30', 'big_span_tau30', 'big_t90_tau30', 'fano_300',
                  'needs_review']])

    # ── plots ────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # raster
    show = indoor[indoor['group'].isin(GROUPS_SHOWN) | (indoor['condition'] == 'insects-only')]
    order = {BASELINE: 0, STIMULUS: 1, SECONDARY: 2, PROBE: 3}
    show = show.assign(_o=show['group'].map(order)).sort_values(['_o', 'max_300'])
    fig, ax = plt.subplots(figsize=(10, max(4, 0.13 * len(show) + 1)), dpi=150)
    style_axes(ax)
    ax.grid(False, axis='y')
    for y, (_, s) in enumerate(show.iterrows()):
        ax.hlines(y, 0, s['duration_s'] / 60, color=GRID, linewidth=0.8, zorder=1)
        t, _ = tf.get(s['file'], empty)
        c = GROUP_COLOR[s['group']]
        ax.scatter(t / 60, np.full(t.size, y), marker='|', s=18, linewidths=0.7, color=c, zorder=2)
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels([f"{f[:48]}" for f in show['file']], fontsize=4.2, color=INK_2)
    ax.set_ylim(-1, len(show))
    ax.set_xlabel('time in recording [min]')
    ax.set_title('Accepted events per indoor recording (sorted by group, then max count in 300 s)',
                 fontsize=9, loc='left')
    handles = [plt.Line2D([], [], color=GROUP_COLOR[g], marker='|', linestyle='', markersize=8,
                          label=g if g != PROBE else 'probe (insects-only)')
               for g in (BASELINE, STIMULUS, SECONDARY, PROBE)]
    ax.legend(handles=handles, frameon=False, fontsize=7, loc='lower right', labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out / 'raster.png')
    plt.close(fig)

    # saturation
    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=150)
    style_axes(ax)
    for grp in GROUPS_SHOWN:
        g = satg[satg['group'] == grp].sort_values('W')
        ax.plot(g['W'], g['mean'], color=GROUP_COLOR[grp], linewidth=2, label=f'{grp} (mean)')
        ax.fill_between(g['W'], g['q25'], g['q75'], color=GROUP_COLOR[grp], alpha=0.15, linewidth=0)
        ax.annotate(grp, (g['W'].iloc[-1], g['mean'].iloc[-1]), xytext=(4, 0),
                    textcoords='offset points', fontsize=7, color=INK_2, va='center')
    ax.set_xscale('log')
    ax.set_xlabel('window W [s]')
    ax.set_ylabel('max events in any W window')
    ax.set_title('Window saturation: where the curve flattens is the burst length',
                 fontsize=9, loc='left')
    ax.legend(frameon=False, fontsize=7, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out / 'saturation.png')
    plt.close(fig)

    # burst span
    fig, ax = plt.subplots(figsize=(6.5, 3.0), dpi=150)
    style_axes(ax)
    rng = np.random.default_rng(0)
    for i, grp in enumerate(GROUPS_SHOWN):
        g = indoor[(indoor['group'] == grp) & (indoor['big_n_tau30'] >= 5)]
        x = g['big_span_tau30'].clip(lower=1)
        ax.scatter(x, i + rng.uniform(-0.18, 0.18, len(x)), s=16, color=GROUP_COLOR[grp],
                   edgecolors='white', linewidths=0.8, zorder=3)
    ax.set_yticks(range(len(GROUPS_SHOWN)))
    ax.set_yticklabels(GROUPS_SHOWN, fontsize=8, color=INK_2)
    ax.set_xscale('log')
    ax.set_xlabel('span of the largest burst (tau = 30 s, >= 5 events) [s]')
    ax.set_title('Burst duration per session', fontsize=9, loc='left')
    fig.tight_layout()
    fig.savefig(out / 'burst_span.png')
    plt.close(fig)

    # ── report ───────────────────────────────────────────────────────────────
    n_ind = indoor.groupby('group').agg(sessions=('file', 'size'),
                                        hours=('duration_s', lambda x: x.sum() / 3600),
                                        events=('n', 'sum'),
                                        needs_review=('needs_review', 'sum'))
    L = []
    L.append('# Stage 5 premises — measured\n')
    L.append(f'Corpus: `{a.corpus}` — {len(sp)} recordings in scope '
             f'({len(indoor)} indoor). Spec: STRESS_DETECTION_ALGORITHM_v1.md §2.\n')
    L.append('> Stage 3 is in-sample on part of the indoor corpus (27-08 training set), '
             'so accepted-event counts there are optimistic. Conditions with '
             '`needs_review = 1` are still guesses (§9.4).\n')
    L.append(md_table(n_ind.reset_index(), '.2f') + '\n')

    L.append('## P3 — rates by condition (context only; pooled events, §9.1)\n')
    L.append(md_table(rates, '.3g') + '\n')

    L.append('## P1 — clustering\n')
    L.append('Median per-session Fano factor over complete bins (1 = Poisson). '
             '`fano_300_debursted` counts one event per burst (tau = 30 s), §13.\n')
    L.append(md_table(fano_tab, '.3g') + '\n')

    L.append('## P1 — burst length (sets W and R)\n')
    L.append('Span of each session\'s largest burst (gap clustering; sessions whose largest '
             'burst has >= 5 events). `t90@30` = time from burst start until 90 % of its '
             'events arrived.\n')
    L.append(md_table(burst_tab, '.3g') + '\n')
    L.append('Window saturation (indoor, mean over sessions of the max count in any W window). '
             f'Stimulus reaches 50 % of its 1800 s value at W = {sat_knee(STIMULUS, .5):.0f} s, '
             f'80 % at W = {sat_knee(STIMULUS, .8):.0f} s, 90 % at W = {sat_knee(STIMULUS, .9):.0f} s. '
             'See `saturation.png`.\n')
    if len(knees):
        L.append(f'Per stimulus session with >= 5 events at W = 1800 s ({len(knees)} sessions): '
                 f'W50 median {knees["W50"].median():.0f} s, W90 median {knees["W90"].median():.0f} s '
                 f'(range {knees["W90"].min():.0f}–{knees["W90"].max():.0f} s).\n')
    if len(gaps):
        L.append('Gaps between successive bursts (>= 3 events, tau = 30 s) inside a session — '
                 'a refractory period R shorter than these re-triggers:\n')
        L.append(md_table(gaps.groupby('group')['gap_s'].describe(percentiles=[.1, .25, .5])
                          .reset_index(), '.3g') + '\n')

    L.append('## P2 — transience\n')
    L.append('Sessions with >= 3 events in some 300 s window: how many have that densest window '
             'in the first 10 minutes, and what share of all events it holds.\n')
    L.append(md_table(peak_early, '.3g') + '\n')

    L.append('## §2.1 — warm-up and handling artefacts\n')
    L.append(f'Per session (>= {3 * EARLY_S // 60} min): accepted-event rate in the first '
             f'{EARLY_S // 60} min vs the rest, Wilcoxon signed-rank across sessions; and the '
             'Buffer 1 noise floor (median `noise_floor_mV` over all candidates) early vs late. '
             'An early floor below the late floor means the estimator starts low, which '
             'inflates SNR at the start.\n')
    L.append(md_table(warm, '.3g') + '\n')

    L.append('## Worst baseline sessions (these set h, §8.5)\n')
    L.append(md_table(top_base, '.4g') + '\n')
    L.append('## Stimulus sessions\n')
    L.append(md_table(stim_tab, '.4g') + '\n')
    L.append('## Figures\n')
    L.append('![raster](raster.png)\n\n![saturation](saturation.png)\n\n![burst span](burst_span.png)\n')
    (out / 'premises.md').write_text('\n'.join(L))
    print(f'wrote {out}/premises.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
