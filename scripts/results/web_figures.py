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
web_figures.py — the three result figures for the website, on the site's dark background

    python3 scripts/results/web_figures.py \
        -d ~/PlantLeaf_dev/Analisi/v6/results \
        --scores "~/PlantLeaf_dev/Analisi/v6/Stress Detector/stress_rerun_20260918/validation/session_scores.csv" \
        -o ~/PlantLeaf_dev/PlantLeaf_Website/HTLM-CSS-JS/static/images

    clicks_rates_v6.png      how many clicks, per species, against both controls
    clicks_timing_v6.png     when they arrive inside a recording
    clicks_detection_v6.png  the automatic detector against its calibrated threshold

Same tables as the report and the poster, so the three can never disagree.

DARK, NOT INVERTED
------------------
The site is black, so these use the palette's DARK steps (#3987e5 / #d95926 /
#199e70) on the #1a1a19 surface the page already uses behind figures — validated
as a set for that surface, which is not the same thing as lightening the light
palette. Every series is also labelled directly, so nothing rests on colour.

Species rows carry their recording count in the label. Several species have one or
two recordings, and a reader is entitled to see that next to the number rather
than in a footnote.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.lines import Line2D                 # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timing_curves import cumulative_curves                     # noqa: E402

SURFACE = '#1a1a19'
INK = '#ffffff'
INK_2 = '#c3c2b7'
GRID = '#333330'
BAND = '#232321'
S1, S2, S3 = '#3987e5', '#d95926', '#199e70'        # validated dark slots 1-3
NEUTRAL = '#8a8880'
FLOOR = 0.55
DPI = 150

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.size': 12, 'text.color': INK, 'axes.labelcolor': INK_2,
    'xtick.color': INK_2, 'ytick.color': INK_2, 'axes.edgecolor': GRID,
})


def style(ax, xlabel=None, ylabel=None):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(length=3, labelsize=11)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12, color=INK_2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, color=INK_2)


def fig_rates(sr: pd.DataFrame, out: Path):
    """One row per species, grouped by stimulus, with both controls at the bottom."""
    def rows_for(cond, grp):
        d = sr[(sr.series == 'algo') & (sr.group == grp) & (sr.condition == cond)]
        order = d.groupby('species')['rate_first15'].median().sort_values(ascending=False)
        return [(sp, cond, grp) for sp in order.index]

    rows = ([('__head__', 'mechanical stress', None)] + rows_for('mechanical', 'stimulus')
            + [('__head__', 'watering', None)] + rows_for('water', 'stimulus')
            + [('__head__', 'controls', None),
               ('unstressed plants', 'plant-only', 'baseline'),
               ('empty room', 'noise', 'baseline')])

    fig, ax = plt.subplots(figsize=(11, 0.52 * len(rows) + 2.2))
    ticks, labels = [], []
    for i, (sp, cond, grp) in enumerate(rows):
        y = len(rows) - 1 - i
        if sp == '__head__':
            ticks.append(y)
            labels.append(cond.upper())
            continue
        d_all = sr[(sr.group == grp) & (sr.condition == cond)]
        if grp == 'baseline':
            ax.axhspan(y - 0.5, y + 0.5, color=BAND, zorder=0)
        n = d_all[(d_all.series == 'algo')].shape[0] if sp in ('unstressed plants', 'empty room') \
            else d_all[(d_all.series == 'algo') & (d_all.species == sp)].shape[0]
        ticks.append(y)
        labels.append(f'{sp}  ·  {n} rec.')
        for k, (series, c) in enumerate((('algo', S1), ('human_click', S2))):
            d = d_all[d_all.series == series]
            if sp not in ('unstressed plants', 'empty room'):
                d = d[d.species == sp]
            v = d['rate_first15'].dropna().to_numpy()
            if not len(v):
                continue
            off = 0.17 if k == 0 else -0.17
            pos, zeros = v[v > 0], v[v == 0]
            ax.scatter(pos, np.full(len(pos), y + off), s=62, facecolor=c,
                       edgecolor=SURFACE, linewidth=1.4, zorder=3)
            if len(zeros):
                ax.scatter([FLOOR], [y + off], s=62, facecolor='none', edgecolor=c,
                           linewidth=1.4, zorder=3)
                if len(zeros) > 1:
                    ax.annotate(f'x{len(zeros)}', (FLOOR, y + off), xytext=(10, 0),
                                textcoords='offset points', va='center', fontsize=10,
                                color=c, zorder=5)
            med = np.median(v)
            if med > 0 and len(v) > 1:
                ax.plot([med, med], [y + off - 0.15, y + off + 0.15], color=c,
                        linewidth=2.4, solid_capstyle='butt', zorder=4)
            if med > 0:
                ax.annotate(f'{med:.0f}', (med, y + off), xytext=(0, 8),
                            textcoords='offset points', ha='center', fontsize=10,
                            color=INK_2, zorder=5)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=11.5, color=INK)
    for t, l in zip(ax.get_yticklabels(), labels):
        if l.isupper():
            t.set_color(INK_2)
            t.set_fontweight('bold')
            t.set_fontsize(10.5)
    ax.set_xscale('log')
    # Upper limit from the data, never a constant: a fixed 1400 silently clipped a
    # recording at 1605 /h off the plot entirely (20-09-2026).
    hi = sr.loc[sr.series.isin(('algo', 'human_click')), 'rate_first15'].max()
    ax.set_xlim(FLOOR * 0.8, max(hi * 1.6, 100))
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(['1', '10', '100', '1000'])
    ax.xaxis.grid(True, color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    style(ax, xlabel='clicks per hour in the first 15 minutes  (log scale)')
    ax.legend(handles=[Line2D([], [], marker='o', linestyle='', markersize=8,
                              markerfacecolor=c, markeredgecolor=SURFACE, label=l)
                       for c, l in ((S1, 'counted automatically'), (S2, 'confirmed by eye'))]
              + [Line2D([], [], marker='o', linestyle='', markersize=8, markerfacecolor='none',
                        markeredgecolor=INK_2, label='recording with no clicks')],
              loc='upper left', bbox_to_anchor=(0, -0.085), frameon=False, fontsize=11, ncol=3)
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f'  {out.name}')


def fig_timing(tc: pd.DataFrame, sr: pd.DataFrame, out: Path, minutes: float = 30.0):
    """Cumulative arrival, as a share of ALL the clicks in each recording.

    See timing_curves.py for why the denominator is the whole recording and not
    the plotted window: normalising inside the window made unstressed plants look
    as concentrated as stressed ones, which is the opposite of what they are.
    """
    cats = [('mechanical stress', 'stimulus', 'mechanical', S1),
            ('watering', 'stimulus', 'water', S2),
            ('unstressed plants', 'baseline', 'plant-only', S3)]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for label, grp, cond, c in cats:
        grid, med, q1, q3, n = cumulative_curves(tc, sr, grp, cond, minutes=minutes)
        if med is None:
            continue
        ax.fill_between(grid, q1, q3, color=c, alpha=0.11, linewidth=0, zorder=2)
        ax.plot(grid, med, color=c, linewidth=2.6, zorder=3, label=f'{label}  ({n} rec.)')
        j = int(np.argmax(grid >= (3.0 if label != 'unstressed plants' else 20.0)))
        ax.annotate(label, (grid[j], med[j]), xytext=(10, -6), textcoords='offset points',
                    fontsize=12, color=c, zorder=5, fontweight='bold')
    ax.axvline(2, color=INK_2, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate('2 minutes', (2, 4), xytext=(6, 0), textcoords='offset points',
                fontsize=11, color=INK_2)
    ax.set_xlim(0, minutes)
    ax.set_ylim(0, 103)
    ax.yaxis.grid(True, color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    style(ax, xlabel='minutes from the start of the recording',
          ylabel="% of ALL the recording's clicks")
    ax.legend(loc='lower right', frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f'  {out.name}')


def fig_detection(sc: pd.DataFrame, out: Path, h: float = 7.0):
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    rng = np.random.default_rng(3)
    for i, (label, grp, c) in enumerate((('stressed plants', 'stimulus', S1),
                                         ('controls', 'baseline', NEUTRAL))):
        v = sc.loc[sc.group == grp, 'max_score_hard'].to_numpy(dtype=float)
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(v)), np.maximum(v, 0.55), s=60,
                   facecolor=c, edgecolor=SURFACE, linewidth=1.4, zorder=3)
        ax.annotate(f'{label}\n{len(v)} recordings', (i, 0.30), ha='center', va='top',
                    fontsize=12, color=INK, annotation_clip=False)
    ax.axhline(h, color=S2, linewidth=2, zorder=4)
    ax.annotate(f'alarm threshold  h = {h:g}', (-0.46, h), xytext=(0, -16),
                textcoords='offset points', ha='left', fontsize=11.5, color=S2,
                fontweight='bold', zorder=5)
    ax.annotate('detected 20 / 22\nfalse alarms 4 / 53', (1.45, 260), ha='right', va='top',
                fontsize=12, color=INK_2, zorder=5)
    ax.annotate('no clicks', (1.47, 0.56), ha='right', va='center',
                fontsize=10, color=INK_2)
    ax.set_yscale('log')
    ax.set_ylim(0.4, 420)
    ax.set_yticks([1, 10, 100])
    ax.set_yticklabels(['1', '10', '100'])
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([])
    ax.yaxis.grid(True, color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    style(ax, ylabel='peak clicks in any 3-minute window  (log scale)')
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f'  {out.name}')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-d', '--dir', type=Path, required=True)
    ap.add_argument('--scores', type=Path, required=True)
    ap.add_argument('-o', '--out', type=Path, required=True)
    a = ap.parse_args()
    sr = pd.read_csv(a.dir / 'session_rates.csv')
    tc = pd.read_csv(a.dir / 'time_course.csv')
    sc = pd.read_csv(a.scores)
    a.out.mkdir(parents=True, exist_ok=True)
    print(f'writing web figures to {a.out}')
    fig_rates(sr, a.out / 'clicks_rates_v6.png')
    fig_timing(tc, sr, a.out / 'clicks_timing_v6.png')
    fig_detection(sc, a.out / 'clicks_detection_v6.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
