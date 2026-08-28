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
feature_transforms.py — which columns `--scale log10` transforms, and the check
that they may be

TRAINING TIME ONLY. Nothing here ends up inside a saved model, and that is
deliberate.

⚠️ WHY THE TRANSFORM ITSELF IS NOT IN THIS FILE
`FunctionTransformer` pickles a REFERENCE to its callable — module path plus
qualified name — not the code. An earlier version of this module exported a
`log_columns` function used that way, and the saved model then carried the
literal string `feature_transforms.log_columns`. Loading it from any process
without `src/ml` on `sys.path` — the click detection worker, for instance — died
with:

    ModuleNotFoundError: No module named 'feature_transforms'

Verified by loading a real trained model from a foreign process, which is the
only way this class of bug shows up: it trains fine, saves fine, and fails at
deployment. (Defining the function in `train_svm.py` is worse still: that file
runs as a script, so the reference pickles as `__main__.log_columns` and fails
for *every* loader, including itself on a second run.)

So `build_scaler` uses `numpy.log10`, which pickles as `numpy.log10` and resolves
anywhere numpy exists — which is anywhere the model can be loaded at all — with
`ColumnTransformer` doing the subsetting. This module keeps only the things that
are safe to import at training time: the column list, and a validator that is
CALLED rather than stored.
"""

from __future__ import annotations

import numpy as np

#: Ratio-type features on which log10 is defined and meaningful. Membership is a
#: claim that the column is STRICTLY POSITIVE once the NaN policy has been
#: applied — assert_log_safe is what stops that claim from being taken on trust.
#:
#: Deliberately absent, because each can legitimately be zero or negative:
#:   R2, fit_valid              0.0 is a real value, not a failure
#:   kurtosis, spectral_tilt    negative by definition
#:   rise_time_ms, fall_time_ms can be 0.0
#:   asymmetry_integral         measured down to -0.32 on the v6 corpus
#:   harmonic_confinement       a log2 ratio, routinely negative
LOG_COLUMNS = (
    'peak_SNR', 'pre_SNR', 'post_SNR',
    'tau_ms',
    'FPE_hz', 'FPE_hz_region',
    'SPR', 'SPR_region', 'R_spectral',
    'n_seg',
    'local_crest', 'k_ratio',
)


def assert_log_safe(df, feature_names, where='the training data'):
    """
    Raise unless every log-transformed column is strictly positive (NaN allowed).

    Call this BEFORE fitting, so a bad combination fails on a clear message
    instead of on a plausible number. np.log10 of a non-positive value yields
    -inf or NaN plus a RuntimeWarning that is trivially missed in a training log,
    and the result then flows into StandardScaler and looks like data.

    The failure this exists for is `tau_ms = -1.0`: the decay-fit sentinel, on
    46.2 % of rows in the v6 training export. Clipping it to 1e-12 — which an
    earlier draft did — maps it to log10 = -12.0, more than ten decades below the
    real log10(tau_ms) minimum of -1.796, silently dominating the scaled vector.
    train_svm refuses that flag combination up front; this is the second line of
    defence, and the one that would catch a NEW feature added to LOG_COLUMNS that
    turns out to reach zero.

    Parameters
    ----------
    df : pandas.DataFrame
    feature_names : list of str      the active feature set
    where : str                      named in the error, e.g. 'Set A'
    """
    offenders = []
    for col in feature_names:
        if col not in LOG_COLUMNS or col not in df.columns:
            continue
        v = df[col].to_numpy(dtype=np.float64, copy=False)
        bad = ~np.isnan(v) & (v <= 0.0)
        if bad.any():
            offenders.append((col, int(bad.sum()), float(v[bad].min())))
    if not offenders:
        return
    lines = [f"--scale log10 cannot be applied to {where}: "
             f"{len(offenders)} column(s) contain non-positive values."]
    for col, n, worst in offenders:
        lines.append(f"    {col:20s} {n} value(s) <= 0, minimum {worst!r}")
    lines.append("  log10 is undefined there. If the column is tau_ms or R2 this is "
                 "the decay-fit sentinel:")
    lines.append("    use --nan-policy nan to convert it, or --scale standard.")
    raise ValueError("\n".join(lines))
