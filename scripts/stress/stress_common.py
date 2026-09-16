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
stress_common.py — shared loading, statistics and plot styling for scripts/stress/

Kept deliberately small: the corpus cache written by build_stress_corpus.py, the
two exact-Poisson formulas the spec uses (§2 P3, §8.3), and one figure style.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / 'src' / 'core'))   # the module, not the Qt package

BASELINE, STIMULUS, PROBE, SECONDARY = 'baseline', 'stimulus', 'probe', 'secondary'
REPORTED_GROUPS = (BASELINE, STIMULUS, SECONDARY, PROBE)

# Reference categorical palette (dataviz skill), fixed slot per group.
GROUP_COLOR = {BASELINE: '#2a78d6', STIMULUS: '#eb6834', SECONDARY: '#1baf7a',
               PROBE: '#52514e'}
INK, INK_2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def load_corpus(corpus_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ses = pd.read_csv(corpus_dir / 'sessions.csv', keep_default_na=False,
                      na_values={'duration_s': ['']})
    ev = pd.read_csv(corpus_dir / 'events.csv')
    return ses, ev


def times_by_file(ev: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for f, g in ev.groupby('file', sort=False):
        o = np.argsort(g['t'].to_numpy(), kind='stable')
        out[f] = (g['t'].to_numpy()[o], g['p'].to_numpy()[o])
    return out


def poisson_ci(k: int, T: float, level: float = 0.95) -> tuple[float, float]:
    """Exact (Garwood) CI on a rate, k events in T units."""
    a = 1 - level
    lo = 0.0 if k == 0 else chi2.ppf(a / 2, 2 * k) / (2 * T)
    hi = chi2.ppf(1 - a / 2, 2 * k + 2) / (2 * T)
    return lo, hi


def fa_upper_bound(A: int, T_h: float) -> float:
    """§8.3: ONE-SIDED 95 % upper bound on the false-alarm rate, chi2_{0.95}(2A+2) / 2T.

    SPEC ERRATUM: §8.3 writes chi2_{0.975}, but also says the A = 0 case reduces to
    3/T. Only the 0.95 quantile does that (2.996/T); 0.975 gives 3.689/T, the upper
    end of a TWO-sided 95 % interval. A false-alarm claim is one-sided, and 3/T is
    also what §2 P3 and §8.4 use, so the 0.95 quantile is implemented."""
    return chi2.ppf(0.95, 2 * A + 2) / (2 * T_h)


def window_max_count(t: np.ndarray, W: float) -> tuple[int, float]:
    """Max events in any (t_i - W, t_i], and the t_i where it is first attained."""
    if t.size == 0:
        return 0, float('nan')
    j = np.searchsorted(t, t - W, side='right')
    occ = np.arange(t.size) - j + 1
    k = int(np.argmax(occ))
    return int(occ[k]), float(t[k])


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def md_table(df: pd.DataFrame, floatfmt: str = '.3g') -> str:
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(map(str, cols)) + ' |',
             '|' + '|'.join('---' for _ in cols) + '|']
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append('—' if not np.isfinite(v) else format(v, floatfmt))
            else:
                cells.append(str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)
