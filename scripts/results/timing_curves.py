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
timing_curves.py — "when do the clicks arrive", computed once for every figure

THE DENOMINATOR IS THE WHOLE RECORDING. THIS IS THE POINT.
----------------------------------------------------------
The first version of this figure plotted each recording's clicks as a share of the
clicks IN ITS FIRST 15 MINUTES, and that was wrong in a way that inverted the
conclusion. An unstressed recording with 6 clicks spread over an hour contributes
only the 2 or 3 that land inside the window; its curve then climbs to 100 % inside
those 15 minutes and the plant looks exactly as concentrated as a stressed one.
The normalisation hid the spread it was supposed to show. Tommy caught it on the
website draft, 20-09-2026.

Measured against the whole recording instead (median share arriving in the first
2 minutes): mechanical 73 %, watering 81 %, unstressed plants 17 %, empty room
0 %. Same data, opposite reading.

So: the denominator is every click in the recording, and a curve that plateaus at
60 % is telling the truth — the other 40 % arrive after the window drawn.

WHY RECORDINGS ARE FILTERED
---------------------------
min_clicks   a cumulative curve built from 1-2 events is a step function, and its
             median across recordings is noise dressed as a shape.
min_minutes  a recording shorter than the window would end mid-curve. Rather than
             extend it flat to the right edge (which invents a plateau) or drop it
             from later bins only (which changes the sample halfway across the
             plot), it is excluded outright and the count is reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BIN_S = 30.0


def cumulative_curves(tc: pd.DataFrame, sr: pd.DataFrame, group: str, condition: str,
                      minutes: float = 30.0, series: str = 'algo',
                      min_clicks: int = 3, min_minutes: float = 30.0):
    """(grid_minutes, median_pct, q1_pct, q3_pct, n_recordings).

    One curve per recording: cumulative clicks by time t, divided by that
    recording's TOTAL clicks. The line is the median across recordings — the
    recording is the unit, so one busy recording cannot set the shape.
    """
    sel = sr[(sr.series == series) & (sr.group == group) & (sr.condition == condition)]
    sel = sel[(sel.n_whole >= min_clicks) & (sel.duration_s >= min_minutes * 60)]
    grid = np.arange(0, minutes * 60 + 1, BIN_S) / 60.0
    curves = []
    for _, row in sel.iterrows():
        g = tc[(tc.file == row.file) & (tc.series == series)
               & (tc.bin_start_s < minutes * 60)].sort_values('bin_start_s')
        if g.empty:
            continue
        cum = np.cumsum(g['count'].to_numpy()) / float(row.n_whole)
        curves.append(np.interp(grid, g['bin_start_s'].to_numpy() / 60 + BIN_S / 120, cum,
                                left=0.0, right=cum[-1]))
    if not curves:
        return grid, None, None, None, 0
    M = np.vstack(curves) * 100
    return (grid, np.median(M, axis=0),
            np.percentile(M, 25, axis=0), np.percentile(M, 75, axis=0), len(curves))
