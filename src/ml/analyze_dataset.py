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
analyze_dataset.py  —  PlantLeaf  corpus click-rate and time-distribution analysis
==================================================================================

Recursively scans one or more input folders for candidate CSVs, runs Stages 2/3/4
of the click detection pipeline on the ones whose schema supports it, and reports
HOW MANY clicks each recording carries and HOW THEY ARE DISTRIBUTED IN TIME —
for the algorithm's predictions AND for the manual labels, side by side.

WHY THE TIME DISTRIBUTION, NOT JUST THE COUNT
---------------------------------------------
A stressed plant and a noisy balcony can emit the same number of events per hour
and be nothing alike: the stressed plant clusters, the balcony does not. A single
clicks/h figure cannot tell them apart, so every rate here is reported alongside
the shape of its arrival process — counts per fixed window at five time scales,
the fraction of empty windows, and the Fano factor (variance / mean of the
per-window counts). Fano = 1 is a Poisson process, the null for "unstructured
background"; Fano > 1 is clustering. That contrast is the material this exists to
produce.

TWO SERIES, ALWAYS SEPARATE
---------------------------
    algo        rows the pipeline confirmed  (stage_blocked == '')
    label_click rows a human marked label == 1
plus label_noise (0) and label_ambiguous (2). They are NEVER merged, and the
labelled series carries `labelled_frac` — the share of that recording's candidates
that have any label at all. Most recordings are labelled only in part, so a
labelled rate is a LOWER BOUND unless labelled_frac == 1.0. `--exhaustive-only`
restricts the labelled statistics to the recordings where it is.

Ambiguous rows (label == 2) are a DECISION, not a missing value, and are their own
series. `--include-ambiguous` adds a fifth series, label_click_incl_amb, that
counts 1 and 2 together — an upper bound to bracket the label_click lower bound.
It never modifies label_click itself.

MIXED SCHEMAS
-------------
v5-era CSVs (24 columns) have no fit_valid / n_seg / local_crest /
harmonic_confinement, so the v6 Stage 2 gates and the v6 SVM cannot be evaluated
on them. They are NOT silently pushed through — a missing column passes every
Stage 2 gate and is imputed by the SVM's SimpleImputer, which would return
confident-looking predictions computed from nothing. Instead each file is checked
against the model's own feature list and the active Stage 2 mode's inputs; files
that fall short get `stage_blocked = 'NoSchema'`, are excluded from every
algorithm statistic, and are listed by name in the report. Their LABEL series is
unaffected and reported in full.

OUTPUTS  (all under --output-dir)
----------------------------------
    report.md              global → per-folder → per-file, with the window tables
    per_file_stats.csv     one row per recording
    per_folder_stats.csv   one row per folder, rolled up at EVERY nesting level
    time_bins.csv          long format: file, series, window_s, bin_idx, t_start_s, count
    window_stats.csv       file, series, window_s, n_bins, mean/median/max/p90,
                           frac_empty, fano, peak_bin_start_s
    evaluated_rows.csv     every input row with svm_probability / svm_prediction /
                           stage_blocked appended
    plots/<file>.png       per recording: multi-series raster over the full
                           recording, per-window count bars, then cluster zooms

Recording durations are read from the .paudio files found under --paudio-dir
(searched recursively).  The 'file' column in each candidate CSV must equal the
.paudio stem (filename without the .paudio extension).  Duration is estimated
from file size: n_frames × FFT_SIZE / fs, where each frame is 154 bins × 5
bytes.  This is accurate to < 0.1 % because the CLCK click-data section at the
end of a .paudio file is tiny relative to the raw FFT stream.

⚠️ A recording with no .paudio has NO duration, hence no rate and no windows. It
still contributes counts. Rates are never computed over a guessed duration.

Usage
-----
    python src/ml/analyze_dataset.py \\
        --model  src/ml/v6/plantleaf_svm_v6_DEPLOYED.pkl \\
        --paudio-dir  /path/to/paudio/recordings/ \\
        /path/to/Dataset/ \\
        --output-dir  /path/to/results/

    # Bracket the labelled rate with the ambiguous rows, and restrict the
    # labelled statistics to exhaustively-labelled recordings:
    python src/ml/analyze_dataset.py \\
        --model model.pkl --paudio-dir /recordings/ \\
        Dataset/PLANTS/ Dataset/NOISE/ \\
        --include-ambiguous --exhaustive-only \\
        --output-dir results/
"""

from __future__ import annotations

import sys
import struct
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

# ── Import stage helpers from sibling module ─────────────────────────────────
# evaluate_candidates.py lives in the same src/ml/ directory.  We insert its
# parent into sys.path so that `import evaluate_candidates` works whether the
# script is launched from the project root or from inside src/ml/.
sys.path.insert(0, str(Path(__file__).parent))
from evaluate_candidates import load_csvs, apply_stage2, apply_stage3, apply_stage4

# src/core/ is added directly rather than importing through the `core` package,
# which would execute src/core/__init__.py and pull in the whole PySide6 window
# stack. This script must stay runnable headless.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'core'))
from click_pipeline_v5 import (   # noqa: E402
    STAGE2_MODE  as _STAGE2_MODE_DEFAULT,
    STAGE2_MODES as _STAGE2_MODES,
    STAGE2_MODE_V5 as _STAGE2_MODE_V5,
)

# ── Series ────────────────────────────────────────────────────────────────────
# Every statistic in this script is computed per SERIES. They are deliberately
# never merged: `algo` is what the pipeline believes, `label_*` is what a human
# saw, and the whole point of the analysis is the gap between them.
SERIES_ALGO      = 'algo'
SERIES_CLICK     = 'label_click'
SERIES_NOISE     = 'label_noise'
SERIES_AMBIG     = 'label_ambiguous'
SERIES_CLICK_AMB = 'label_click_incl_amb'   # only with --include-ambiguous

#: Order used in every table and legend.
SERIES_ORDER = (SERIES_ALGO, SERIES_CLICK, SERIES_CLICK_AMB, SERIES_AMBIG, SERIES_NOISE)

#: Series whose rate is a LOWER BOUND on a partially-labelled recording.
LABEL_SERIES = (SERIES_CLICK, SERIES_NOISE, SERIES_AMBIG, SERIES_CLICK_AMB)

# ── Time windows ──────────────────────────────────────────────────────────────
#: The five scales the report bins at, in seconds. 5 min is short enough to
#: resolve a single stress response; 1 h is long enough that a whole recording is
#: usually one or two bins, which is what makes the Fano factor at 5 min and at
#: 1 h say different things.
WINDOWS_S = (300, 600, 900, 1800, 3600)


def set_windows(minutes: list[float]) -> None:
    """Override the binning scales, in MINUTES. See --windows.

    A module-level rebind rather than a parameter threaded through nine call
    sites: every function here that bins reads WINDOWS_S once, and the set is a
    property of a whole run, never of one recording. Called from main() before
    anything is computed.
    """
    global WINDOWS_S
    secs = tuple(sorted({int(round(m * 60)) for m in minutes if m > 0}))
    if not secs:
        raise ValueError('--windows needs at least one positive value')
    WINDOWS_S = secs

#: A window is only reported for a recording that fits at least this many COMPLETE
#: bins. One bin has no variance, so no Fano factor and no fraction-empty; two is
#: the arithmetic minimum and still nearly meaningless, but it is a number rather
#: than a lie, and the n_bins column is always printed beside it so the reader can
#: discount it. The trailing partial bin is always dropped — counting a 40 s
#: remainder as a 300 s window would report a spurious empty bin at the end of
#: every recording.
MIN_BINS_FOR_STATS = 2

# ── Schema capability ─────────────────────────────────────────────────────────
#: Columns the v6 Stage 2 gates read. A CSV without them does not "fail" the
#: gates — NaN and missing PASS every gate by design (_stage2_reason) — it simply
#: is not being gated, which is worse than a rejection because it is invisible.
_V6_STAGE2_COLS = ('n_seg', 'local_crest', 'harmonic_confinement', 'peak_SNR', 'SPR')

#: Columns the v5 Stage 2 fit gate reads.
_V5_STAGE2_COLS = ('R2', 'tau_ms', 'SPR')

#: Marker written into stage_blocked for rows whose source CSV cannot support the
#: requested model / Stage 2 mode. Distinct from every real stage verdict so it can
#: never be mistaken for one, and so the report can count it separately.
STAGE_NO_SCHEMA = 'NoSchema'

# ── .paudio format constants ──────────────────────────────────────────────────
# Must stay in sync with click_pipeline_v5.py (FS, FFT_SIZE, _K_BINS).
_V5_FS           = 200_000   # Sampling rate [Hz]
_V5_FFT_SIZE     = 512       # FFT window length [samples]
_BINS_PER_FRAME  = 154       # Analysis bins stored per frame (20–80 kHz band)
_BYTES_PER_FRAME = _BINS_PER_FRAME * 5   # 770 bytes  (4-byte float mag + 1-byte int8 phase)
_FRAME_DUR_S     = _V5_FFT_SIZE / _V5_FS  # 2.56 ms per frame


# ── .paudio duration reader ───────────────────────────────────────────────────

def read_paudio_duration(path: Path) -> float | None:
    """
    Return the recording duration in seconds for a .paudio file, or None on
    any error (file missing, wrong magic, truncated header).

    Method: estimate from file size.
        duration = (file_size - 128) / _BYTES_PER_FRAME * _FRAME_DUR_S

    The 128-byte header is excluded.  If the file ends with a CLCK click-data
    section, the estimate is fractionally longer than the true duration, but
    the error is < 0.1 % for any recording longer than a few minutes.

    We do NOT use the header's start_time / end_time timestamps because those
    fields reflect the wall-clock times of the save operation, not the actual
    duration of the captured audio stream.
    """
    try:
        if not path.exists():
            return None

        with open(path, 'rb') as f:
            header = f.read(128)

        if len(header) < 78:
            return None

        magic = header[0:10].rstrip(b'\x00')
        if magic != b'PLANTAUDIO':
            return None

        fs       = struct.unpack('<I', header[34:38])[0]
        fft_size = struct.unpack('<I', header[38:42])[0]

        if fs == 0 or fft_size == 0:
            return None

        file_size  = path.stat().st_size
        data_bytes = file_size - 128
        if data_bytes <= 0:
            return None

        n_frames = data_bytes / _BYTES_PER_FRAME
        return n_frames * fft_size / fs

    except Exception:
        return None


def build_paudio_index(paudio_dir: Path) -> dict[str, Path]:
    """
    Recursively index all *.paudio files under paudio_dir.

    Returns a dict mapping stem (filename without '.paudio') → Path.
    If duplicate stems exist in different subdirectories, the last one wins
    and a warning is printed.
    """
    index: dict[str, Path] = {}
    for p in sorted(paudio_dir.rglob('*.paudio')):
        stem = p.stem
        if stem in index:
            print(f"  Warning: duplicate .paudio stem '{stem}', keeping {p}")
        index[stem] = p
    return index


# ── CSV discovery ─────────────────────────────────────────────────────────────

def _is_real_csv(p: Path) -> bool:
    """Reject the things an rglob('*.csv') over an external drive actually returns.

    macOS writes an AppleDouble sidecar named `._<original>` next to every file on
    a non-HFS volume (exFAT, NTFS — i.e. every external drive this corpus lives
    on). They match *.csv, are a few kB of binary resource fork, and pandas either
    raises on them or, worse, parses a handful of garbage rows that then join the
    corpus as a phantom recording. On the v5 corpus they are 50 % of the rglob hits.

    Zero-byte files are dropped for the same reason: an interrupted export leaves
    one behind and it is not a recording.
    """
    if p.name.startswith('._'):
        return False
    try:
        if p.stat().st_size == 0:
            return False
    except OSError:
        return False
    return True


def resolve_csvs(inputs: list[Path]) -> list[Path]:
    """
    Expand a mixed list of files and/or folders into a sorted list of CSV
    paths.  Folders are searched recursively for *.csv files.

    AppleDouble sidecars and empty files are dropped — see _is_real_csv.
    """
    paths: list[Path] = []
    n_skipped = 0
    for inp in inputs:
        if inp.is_dir():
            raw   = sorted(inp.rglob('*.csv'))
            found = [p for p in raw if _is_real_csv(p)]
            n_skipped += len(raw) - len(found)
            if not found:
                print(f"  Warning: no usable *.csv found recursively in {inp}")
            else:
                print(f"  Folder {inp.name}: {len(found)} CSV file(s) found "
                      f"(recursive)")
            paths.extend(found)
        elif inp.is_file():
            if _is_real_csv(inp):
                paths.append(inp)
            else:
                n_skipped += 1
        else:
            print(f"  Warning: {inp} does not exist — skipping")
    if n_skipped:
        print(f"  Skipped {n_skipped} AppleDouble/empty file(s) "
              f"(._* sidecars, 0-byte exports)")
    return paths


# ── Schema capability ─────────────────────────────────────────────────────────

def read_header_columns(path: Path) -> set[str]:
    """The column names in a CSV's header row, without loading the body.

    Read from the FILE, not from the concatenated DataFrame: pd.concat over a mix
    of v5 and v6 exports produces the union of both schemas, filled with NaN. After
    that there is no way to tell "this recording has no harmonic_confinement" from
    "this recording's harmonic_confinement is NaN on every row" — and the two mean
    opposite things at Stage 2, where NaN passes the gate by design.
    """
    try:
        with open(path, encoding='utf-8-sig', newline='') as fh:
            header = fh.readline()
    except OSError:
        return set()
    return {c.strip().strip('"') for c in header.rstrip('\r\n').split(',')}


def required_columns(model: dict, stage2_mode: str) -> list[str]:
    """Every column that must be PRESENT for Stages 2-3 to mean anything.

    The model's own feature list is authoritative — model['features'], never a
    hard-coded list, because that is what apply_stage3 indexes with. Stage 2's
    inputs depend on the mode: the v5 fit gate reads R2 / tau_ms, the v6 gates read
    n_seg / local_crest / harmonic_confinement.

    fit_valid is added whenever the model declares nan_policy == 'nan'. Without it
    apply_stage3 cannot tell which rows carry the -1 / 0.0 sentinels, so it cannot
    reproduce the encoding the model was fitted on.
    """
    cols = list(model['features'])
    cols += list(_V5_STAGE2_COLS if stage2_mode == _STAGE2_MODE_V5 else _V6_STAGE2_COLS)
    if model.get('nan_policy', 'sentinel') == 'nan':
        cols.append('fit_valid')
    seen: list[str] = []
    for c in cols:
        if c not in seen:
            seen.append(c)
    return seen


def classify_sources(
    csv_paths: list[Path],
    model: dict,
    stage2_mode: str,
) -> dict[str, dict]:
    """{str(path): {schema, missing, can_predict}} for every source CSV.

    `can_predict` is the ONLY thing that decides whether a file's rows are sent
    through Stages 2/3/4. Deciding it per FILE rather than per row is deliberate:
    a schema is a property of the export, and a per-row rule would let one v6 file
    with a NaN column silently take a different path from its neighbours.
    """
    need = required_columns(model, stage2_mode)
    out: dict[str, dict] = {}
    for p in csv_paths:
        cols    = read_header_columns(p)
        missing = [c for c in need if c not in cols]
        out[str(p)] = {
            'schema':      'v6' if 'schema_version' in cols else 'v5',
            'missing':     missing,
            'can_predict': not missing,
        }
    return out


# ── Per-file statistics ───────────────────────────────────────────────────────

def _safe_div(num: float | int, den: float | int) -> float:
    return float(num) / float(den) if den else float('nan')


# ── Series extraction ─────────────────────────────────────────────────────────

def series_masks(df: pd.DataFrame, include_ambiguous: bool) -> dict[str, pd.Series]:
    """Boolean masks over `df`, one per series. See SERIES_ORDER.

    `algo` requires can_predict — a row whose source CSV could not support the
    model is not "predicted noise", it is UNJUDGED, and counting it as a negative
    would understate the algorithm on exactly the recordings it was never run on.

    label == 2 (ambiguous) never enters label_click. It is a reviewer's recorded
    inability to decide, and folding it into either class invents a judgement
    nobody made. label_click_incl_amb exists so the pair can be read as a bound.
    """
    lbl = pd.to_numeric(df.get('label'), errors='coerce') if 'label' in df.columns \
          else pd.Series(np.nan, index=df.index)
    predictable = (df['_can_predict'] if '_can_predict' in df.columns
                   else pd.Series(True, index=df.index))

    masks = {
        SERIES_ALGO:  predictable & (df['stage_blocked'].fillna('') == ''),
        SERIES_CLICK: lbl == 1,
        SERIES_NOISE: lbl == 0,
        SERIES_AMBIG: lbl == 2,
    }
    if include_ambiguous:
        masks[SERIES_CLICK_AMB] = lbl.isin((1, 2))
    return masks


def event_times(df: pd.DataFrame, mask: pd.Series) -> np.ndarray:
    """Sorted timestamps [s] of the masked rows, NaNs dropped."""
    if 'timestamp_s' not in df.columns:
        return np.empty(0, dtype=float)
    ts = pd.to_numeric(df.loc[mask, 'timestamp_s'], errors='coerce').dropna()
    return np.sort(ts.to_numpy(dtype=float))


# ── Time binning ──────────────────────────────────────────────────────────────

def bin_counts(times: np.ndarray, duration_s: float, window_s: float) -> np.ndarray:
    """Counts per COMPLETE non-overlapping window of `window_s`, starting at t = 0.

    The trailing partial window is dropped, and so are any events inside it. A
    recording of 3 h 20 m binned at 1 h yields three bins, not four: the fourth
    would be 20 minutes long and would be compared, as if it were equal, against
    windows three times its length. Every zero-inflation and every deflated max in
    a naive binning comes from that last bin.

    Returns an empty array when the recording is shorter than one window.
    """
    if not np.isfinite(duration_s) or duration_s <= 0 or window_s <= 0:
        return np.empty(0, dtype=int)
    n_bins = int(duration_s // window_s)
    if n_bins < 1:
        return np.empty(0, dtype=int)
    if times.size == 0:
        return np.zeros(n_bins, dtype=int)
    idx = np.floor(times / window_s).astype(np.int64)
    idx = idx[(idx >= 0) & (idx < n_bins)]
    return np.bincount(idx, minlength=n_bins).astype(int)


def window_stats(counts: np.ndarray, window_s: float) -> dict:
    """Shape of the arrival process at one time scale.

    fano — variance / mean of the per-window counts, the index of dispersion.
        It is the number this whole script exists to produce. For a homogeneous
        Poisson process (events independent, constant rate) the count in a fixed
        window is Poisson, whose variance equals its mean, so fano == 1 REGARDLESS
        of the rate. That is what makes it comparable across recordings that emit
        wildly different numbers of events:

            fano ≈ 1   unstructured background — what a noise recording should look like
            fano >> 1  clustered / bursty      — a stress response arrives in bursts
            fano < 1   more regular than chance — periodic interference, a fan, a pump

        Population variance (ddof = 0), not the sample estimator: these bins are
        the entire recording, not a sample drawn from it. NaN when the mean is 0
        (no events — nothing to disperse) or when there are too few bins to have a
        variance at all.

    frac_empty — share of windows with no event. Separates "one big burst" from
        "a steady drizzle" at the same mean, and unlike fano it survives a mean of
        zero, so it is the fallback statistic on near-silent recordings.
    """
    n = int(counts.size)
    base = {
        'window_s': float(window_s), 'n_bins': n,
        'n_events': int(counts.sum()) if n else 0,
        'mean': float('nan'), 'median': float('nan'), 'max': float('nan'),
        'p90': float('nan'), 'frac_empty': float('nan'), 'fano': float('nan'),
        'peak_bin_start_s': float('nan'),
    }
    if n < MIN_BINS_FOR_STATS:
        return base

    mean = float(counts.mean())
    var  = float(counts.var(ddof=0))
    base.update({
        'mean':       mean,
        'median':     float(np.median(counts)),
        'max':        float(counts.max()),
        'p90':        float(np.percentile(counts, 90)),
        'frac_empty': float((counts == 0).mean()),
        'fano':       (var / mean) if mean > 0 else float('nan'),
        'peak_bin_start_s': float(int(np.argmax(counts)) * window_s),
    })
    return base


def compute_time_distribution(
    df: pd.DataFrame,
    per_file: list[dict],
    include_ambiguous: bool,
) -> tuple[list[dict], list[dict]]:
    """(window_rows, bin_rows) for every recording × series × window size.

    window_rows feeds window_stats.csv and the report tables; bin_rows is the raw
    per-bin counts (time_bins.csv), kept because every aggregate here throws away
    WHERE the events were, and that is often the thing worth looking at.
    """
    window_rows: list[dict] = []
    bin_rows:    list[dict] = []
    counts_map:  dict[tuple, np.ndarray] = {}

    for rec in per_file:
        file_id  = rec['file']
        duration = rec['duration_s']
        sub      = df[df['file'] == file_id]
        masks    = series_masks(sub, include_ambiguous)

        for name in SERIES_ORDER:
            if name not in masks:
                continue
            # A file whose schema could not be scored has no algo series at all —
            # zero here would read as "the algorithm found nothing".
            if name == SERIES_ALGO and not rec['can_predict']:
                continue
            times = event_times(sub, masks[name])

            for w in WINDOWS_S:
                counts = bin_counts(times, duration, w)
                counts_map[(file_id, name, w)] = counts
                row = window_stats(counts, w)
                row.update({'file': file_id, 'series': name,
                            'source_folder': rec['source_folder']})
                window_rows.append(row)
                for i, c in enumerate(counts):
                    bin_rows.append({
                        'file': file_id, 'series': name, 'window_s': w,
                        'bin_idx': i, 't_start_s': i * w, 'count': int(c),
                    })
    return window_rows, bin_rows, counts_map


def aggregate_window_stats(
    counts_map: dict[tuple, np.ndarray],
    per_file: list[dict],
    group: str,
    include_ambiguous: bool,
) -> list[dict]:
    """Window statistics for a whole folder, by POOLING its recordings' bins.

    Every complete window in the folder becomes one sample. Pooling the bins is
    the right move and concatenating the event TIMES would be wrong: two
    recordings both start at t = 0, so merging their timelines would stack
    unrelated events into the same window and manufacture clustering that no
    recording contains. Bins are already anonymous with respect to which
    recording they came from, which is exactly what a folder-level question wants.

    A folder mixing 10-minute and 3-hour recordings still pools honestly, because
    every bin is the same length by construction — only the NUMBER of bins each
    recording contributes varies, which is the correct weighting.
    """
    members = [r['file'] for r in per_file if group in r.get('groups', [])]
    if not members:
        return []
    series = list(SERIES_ORDER) if include_ambiguous else \
             [x for x in SERIES_ORDER if x != SERIES_CLICK_AMB]
    out = []
    for name in series:
        for w in WINDOWS_S:
            arrays = [counts_map[(f, name, w)] for f in members
                      if (f, name, w) in counts_map]
            if not arrays:
                continue
            pooled = np.concatenate(arrays) if arrays else np.empty(0, dtype=int)
            row = window_stats(pooled, w)
            row.update({'group': group, 'series': name,
                        'n_recordings': len(arrays)})
            out.append(row)
    return out


def compute_per_file_stats(
    df: pd.DataFrame,
    paudio_index: dict[str, Path],
    include_ambiguous: bool = False,
) -> list[dict]:
    """
    One dict per unique value of the 'file' column.

    Counts, then rates, then the confusion matrix — and the three are computed on
    three different row sets, which is the whole subtlety here:

      * counts      every row of the recording
      * rates       counts / duration, and ONLY when a .paudio gave a duration
      * confusion   labelled rows only, and label == 2 excluded from all four
                    cells (it is not a class), and only when the schema allowed a
                    prediction in the first place

    `labelled_frac` = labelled rows / candidates. It is the number that makes a
    labelled rate readable: at 0.02 the recording's label_click rate is a lower
    bound on a 2 % sample, not a measurement, and comparing it against a fully
    labelled recording is meaningless. Every consumer of a label_* rate must read
    it beside this.
    """
    stats = []

    for file_id, grp in df.groupby('file', sort=True):
        total_cands = len(grp)

        can_predict = bool(grp['_can_predict'].all()) if '_can_predict' in grp.columns else True
        schema      = str(grp['_schema'].iloc[0]) if '_schema' in grp.columns else 'v5'

        # stage_blocked is '' (empty string) for survivors while the DataFrame
        # is in memory; NaN only appears after writing/reading the CSV.
        confirmed = (int((grp['stage_blocked'].fillna('') == '').sum())
                     if can_predict else 0)

        # Source folder: take from _source_csv of the first row
        src_csv = grp['_source_csv'].iloc[0] if '_source_csv' in grp.columns else ''
        source_folder = str(Path(src_csv).parent) if src_csv else ''

        # Stage breakdown for this file — every v6 verdict, not just the two v5 ones
        sb = grp['stage_blocked'].fillna('')
        stage_counts = {k: int(v) for k, v in sb[sb != ''].value_counts().items()}

        # ── Label counts (independent of the schema and of the model) ─────────
        lbl = (pd.to_numeric(grp['label'], errors='coerce')
               if 'label' in grp.columns else pd.Series(np.nan, index=grp.index))
        n_labeled = int(lbl.notna().sum())
        n_click   = int((lbl == 1).sum())
        n_noise   = int((lbl == 0).sum())
        n_ambig   = int((lbl == 2).sum())
        labelled_frac = _safe_div(n_labeled, total_cands)
        # Exhaustive == a human judged EVERY candidate this recording produced.
        exhaustive = total_cands > 0 and n_labeled == total_cands

        # ── Confusion matrix — labelled, non-ambiguous, predictable rows only ──
        tp = fp = fn = tn = 0
        if can_predict and n_labeled:
            binary  = lbl.isin((0, 1))
            pred_1  = (sb == '')
            tp = int((binary & (lbl == 1) & pred_1).sum())
            fp = int((binary & (lbl == 0) & pred_1).sum())
            fn = int((binary & (lbl == 1) & ~pred_1).sum())
            tn = int((binary & (lbl == 0) & ~pred_1).sum())

        recall      = _safe_div(tp, tp + fn)
        precision   = _safe_div(tp, tp + fp)
        specificity = _safe_div(tn, tn + fp)
        f1          = _safe_div(2 * precision * recall, precision + recall)

        # Duration from .paudio
        paudio_path = paudio_index.get(file_id)
        duration_s  = read_paudio_duration(paudio_path) if paudio_path else None
        paudio_found = paudio_path is not None
        has_dur = duration_s is not None and duration_s > 0
        hours   = (duration_s / 3600.0) if has_dur else float('nan')

        # ── Per-series rates ─────────────────────────────────────────────────
        # Built from the same masks the time binning uses, so a rate and its
        # window table can never disagree about what an event is.
        masks   = series_masks(grp, include_ambiguous)
        counts  = {name: int(m.sum()) for name, m in masks.items()}
        if not can_predict:
            counts.pop(SERIES_ALGO, None)
        rates   = {name: (n / hours if has_dur else float('nan'))
                   for name, n in counts.items()}

        rec = {
            'file':            file_id,
            'source_folder':   source_folder,
            'schema':          schema,
            'can_predict':     can_predict,
            'candidates':      total_cands,
            'confirmed':       confirmed,
            'labeled':         n_labeled,
            'n_click':         n_click,
            'n_noise':         n_noise,
            'n_ambiguous':     n_ambig,
            'labelled_frac':   labelled_frac,
            'exhaustive':      exhaustive,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'recall':          recall,
            'precision':       precision,
            'specificity':     specificity,
            'f1':              f1,
            'stage_counts':    stage_counts,
            'duration_s':      duration_s if duration_s is not None else float('nan'),
            'paudio_found':    paudio_found,
        }
        for name in SERIES_ORDER:
            rec[f'n_{name}']    = counts.get(name, float('nan'))
            rec[f'rate_{name}'] = rates.get(name, float('nan'))
        # Back-compat aliases used by the plotting code and the old report rows.
        rec['algo_rate_hr']  = rec.get(f'rate_{SERIES_ALGO}',  float('nan'))
        rec['label_rate_hr'] = rec.get(f'rate_{SERIES_CLICK}', float('nan'))
        stats.append(rec)

    return stats


def aggregate_stats(per_file: list[dict], label: str = 'ALL') -> dict:
    """Roll a set of per-file records into one aggregate.

    Rates are recomputed as total events / total duration — NEVER as the mean of
    the per-file rates. A 6-second recording with one click has a rate of 600/h,
    and averaging it against a 2-hour recording would let it outvote 1200x its own
    evidence. Only recordings with a known duration contribute to the denominator,
    and only their events contribute to the numerator, so the ratio stays a ratio
    of things that were both measured.
    """
    n_files = len(per_file)
    total_candidates = sum(r['candidates'] for r in per_file)
    total_confirmed  = sum(r['confirmed']  for r in per_file)
    total_labeled    = sum(r['labeled']    for r in per_file)
    total_click      = sum(r['n_click']    for r in per_file)
    total_noise      = sum(r['n_noise']    for r in per_file)
    total_ambig      = sum(r['n_ambiguous'] for r in per_file)
    total_tp = sum(r['tp'] for r in per_file)
    total_fp = sum(r['fp'] for r in per_file)
    total_fn = sum(r['fn'] for r in per_file)
    total_tn = sum(r['tn'] for r in per_file)

    stage_counts: dict[str, int] = {}
    for r in per_file:
        for k, v in r['stage_counts'].items():
            stage_counts[k] = stage_counts.get(k, 0) + v

    def _has_dur(r) -> bool:
        return bool(r['paudio_found']) and not np.isnan(r['duration_s'])

    dur_values = [r['duration_s'] for r in per_file if _has_dur(r)]
    total_dur  = sum(dur_values) if dur_values else float('nan')
    n_missing  = sum(1 for r in per_file if not _has_dur(r))

    out = {
        'group':       label,
        'n_files':     n_files,
        'candidates':  total_candidates,
        'confirmed':   total_confirmed,
        'labeled':     total_labeled,
        'n_click':     total_click,
        'n_noise':     total_noise,
        'n_ambiguous': total_ambig,
        'labelled_frac': _safe_div(total_labeled, total_candidates),
        'n_exhaustive':  sum(1 for r in per_file if r['exhaustive']),
        'n_no_schema':   sum(1 for r in per_file if not r['can_predict']),
        'tp': total_tp, 'fp': total_fp, 'fn': total_fn, 'tn': total_tn,
        'recall':      _safe_div(total_tp, total_tp + total_fn),
        'precision':   _safe_div(total_tp, total_tp + total_fp),
        'specificity': _safe_div(total_tn, total_tn + total_fp),
        'stage_counts':     stage_counts,
        'duration_s':       total_dur,
        'n_paudio_missing': n_missing,
    }
    prec, rec = out['precision'], out['recall']
    out['f1'] = _safe_div(2 * prec * rec, prec + rec)

    # ── Per-series aggregate rates ───────────────────────────────────────────
    hours_all = (total_dur / 3600.0) if dur_values else float('nan')
    for name in SERIES_ORDER:
        key = f'n_{name}'
        # The algo series only exists on files the schema could carry, so its
        # denominator is those files' duration, not the whole group's.
        if name == SERIES_ALGO:
            elig = [r for r in per_file if _has_dur(r) and r['can_predict']]
        else:
            elig = [r for r in per_file if _has_dur(r)]
        n_tot = sum(r.get(key, 0) for r in elig
                    if isinstance(r.get(key), (int, float)) and not (
                        isinstance(r.get(key), float) and np.isnan(r[key])))
        h = sum(r['duration_s'] for r in elig) / 3600.0 if elig else float('nan')
        out[key]           = n_tot
        out[f'rate_{name}'] = _safe_div(n_tot, h) if elig else float('nan')

    out['algo_rate_hr']  = out.get(f'rate_{SERIES_ALGO}',  float('nan'))
    out['label_rate_hr'] = out.get(f'rate_{SERIES_CLICK}', float('nan'))
    out['hours'] = hours_all
    return out


# ── Folder roll-up ────────────────────────────────────────────────────────────

def folder_groups(per_file: list[dict], input_roots: list[Path]) -> list[str]:
    """Every folder a recording belongs to, from its own folder up to its root.

    A corpus laid out as  Plants/Aloe Vera/meccanico/  answers three different
    questions at three different depths — "mechanically stressed aloe", "aloe",
    "plants" — and which one matters is not knowable in advance. So every level is
    aggregated, and the report prints them nested. A recording under
    Plants/Aloe Vera/meccanico therefore contributes to Plants, to
    Plants/Aloe Vera, and to Plants/Aloe Vera/meccanico.

    Paths are made relative to whichever input root contains them, so the labels
    read as the corpus tree rather than as absolute paths on someone's drive. The
    LONGEST matching root wins, which matters when one input root is nested inside
    another — otherwise the file would be labelled by the outer root and the inner
    root's group would be silently empty.
    """
    resolved = sorted(
        ((r, r.resolve()) for r in input_roots if r.exists()),
        key=lambda pair: len(str(pair[1])), reverse=True,
    )
    groups: list[str] = []
    for rec in per_file:
        folder = Path(rec['source_folder']) if rec['source_folder'] else None
        rec['groups'] = []
        if folder is None:
            continue
        try:
            fres = folder.resolve()
        except OSError:
            continue
        for raw_root, root in resolved:
            try:
                rel = fres.relative_to(root)
            except ValueError:
                continue
            base  = raw_root.name or str(root)
            chain = [base]
            for part in rel.parts:
                chain.append(f'{chain[-1]}/{part}')
            rec['groups'] = chain
            for g in chain:
                if g not in groups:
                    groups.append(g)
            break
    return sorted(groups)


def aggregate_by_folder(per_file: list[dict], input_roots: list[Path]) -> list[dict]:
    """One aggregate row per folder, at every nesting level. See folder_groups."""
    groups = folder_groups(per_file, input_roots)
    out = []
    for g in groups:
        members = [r for r in per_file if g in r.get('groups', [])]
        if members:
            agg = aggregate_stats(members, label=g)
            agg['depth'] = g.count('/')
            out.append(agg)
    return out


# ── Markdown summary ──────────────────────────────────────────────────────────

def _fmt(val, decimals: int = 3, na: str = 'N/A') -> str:
    if isinstance(val, float) and np.isnan(val):
        return na
    if isinstance(val, float):
        return f'{val:.{decimals}f}'
    return str(val)


def _dur_str(s: float) -> str:
    if np.isnan(s):
        return 'N/A'
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f'{h}h {m:02d}m {sec:02d}s  ({s:.1f} s)'
    if m:
        return f'{m}m {sec:02d}s  ({s:.1f} s)'
    return f'{s:.1f} s'


def _pct(v: float, decimals: int = 1) -> str:
    return 'N/A' if (isinstance(v, float) and np.isnan(v)) else f'{100 * v:.{decimals}f} %'


def _series_label(name: str) -> str:
    return {
        SERIES_ALGO:      'algorithm',
        SERIES_CLICK:     'label = 1 (click)',
        SERIES_CLICK_AMB: 'label = 1 or 2 (click + ambiguous)',
        SERIES_AMBIG:     'label = 2 (ambiguous)',
        SERIES_NOISE:     'label = 0 (noise)',
    }.get(name, name)


def _window_table(rows: list[dict], key: str = 'series') -> list[str]:
    """A markdown table of window_stats rows, sorted by series then window."""
    if not rows:
        return ['*(no window fits this recording — shorter than 5 min, '
                'or no duration available)*', '']
    lines = [
        '| Series | Window | Bins | Events | Mean/bin | Max/bin | Empty bins | Fano |',
        '|---|---|---|---|---|---|---|---|',
    ]
    order = {n: i for i, n in enumerate(SERIES_ORDER)}
    # A series with no events anywhere contributes five rows of zeros per folder
    # and buries the ones that say something. It stays in window_stats.csv, where
    # "this series was computed and was empty" is the useful form of the fact.
    live = {r[key] for r in rows if r['n_events'] > 0}
    for r in sorted(rows, key=lambda x: (order.get(x[key], 99), x['window_s'])):
        if r['n_bins'] < MIN_BINS_FOR_STATS or r[key] not in live:
            continue
        lines.append(
            f'| {_series_label(r[key])} '
            f'| {int(r["window_s"]) // 60} min '
            f'| {r["n_bins"]} '
            f'| {r["n_events"]} '
            f'| {_fmt(r["mean"], 2)} '
            f'| {_fmt(r["max"], 0)} '
            f'| {_pct(r["frac_empty"])} '
            f'| {_fmt(r["fano"], 2)} |'
        )
    lines.append('')
    return lines


def write_report_md(
    per_file:     list[dict],
    per_folder:   list[dict],
    global_stats: dict,
    window_rows:  list[dict],
    counts_map:   dict,
    model:        dict,
    output_path:  Path,
    input_roots:  list[Path],
    opts:         dict,
) -> None:
    g = global_stats
    lines: list[str] = []

    try:
        svc = (model['pipeline'].named_steps.get('svm')
               or model['pipeline'].named_steps.get('svc'))
        hparam_str = '  '.join(f'{k}={v}' for k, v in svc.get_params().items()
                               if k in ('C', 'gamma', 'degree', 'coef0'))
    except Exception:
        hparam_str = ''

    lines += [
        '# PlantLeaf — Click Rate & Time Distribution Report',
        '',
        f'**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M")}  ',
        f'**Model:** `{model.get("kernel","?").upper()}` kernel  '
        + (f'— {hparam_str}  ' if hparam_str else '')
        + f'threshold={model["threshold"]:.3f}  '
        + f'nan_policy={model.get("nan_policy", "sentinel")}  ',
        f'**Features ({len(model["features"])}):** `{", ".join(model["features"])}`  ',
        f'**Stage 2 mode:** `{opts["stage2_mode"]}`  ',
        f'**Options:** include_ambiguous={opts["include_ambiguous"]}  '
        f'exhaustive_only={opts["exhaustive_only"]}  ',
        f'**Input folders:** {", ".join(str(r) for r in input_roots)}  ',
        '',
        '## How to read this',
        '',
        '- **`algorithm`** = rows the pipeline confirmed. **`label = 1`** = rows a',
        '  human marked as a click. They are separate series and are never merged.',
        '- A labelled rate is a **lower bound** unless `labelled_frac` = 100 %. Most',
        '  recordings are labelled only in part.',
        '- **Fano** = variance / mean of the per-window counts. **≈ 1** is a Poisson',
        '  process — unstructured background. **>> 1** is clustered/bursty. **< 1** is',
        '  more regular than chance (periodic interference). It is rate-independent,',
        '  which is what makes it comparable across recordings.',
        '- The trailing partial window of each recording is dropped, so `Bins` ×',
        '  window ≤ duration and `Events` can be lower than the recording total.',
        '',
    ]

    # ── Global ───────────────────────────────────────────────────────────────
    lines += [
        '## Global Summary',
        '',
        '| | Value |',
        '|---|---|',
        f'| Recordings | {g["n_files"]} |',
        f'| Total duration | {_dur_str(g["duration_s"])} |',
        f'| Input candidates | {g["candidates"]} |',
        f'| Labelled rows | {g["labeled"]}  ({_pct(g["labelled_frac"])} of candidates) |',
        f'| Exhaustively labelled recordings | {g["n_exhaustive"]} / {g["n_files"]} |',
        f'| Recordings excluded from prediction (schema) | {g["n_no_schema"]} |',
        '',
        '### Click rate by series',
        '',
        '| Series | Events | Rate (clicks / h) |',
        '|---|---|---|',
    ]
    for name in SERIES_ORDER:
        if f'rate_{name}' not in g:
            continue
        if name == SERIES_CLICK_AMB and not opts['include_ambiguous']:
            continue
        lines.append(f'| {_series_label(name)} | {g.get(f"n_{name}", 0)} '
                     f'| {_fmt(g[f"rate_{name}"], 2)} |')
    lines += ['']

    lines += [
        '### Confusion matrix  *(labelled, non-ambiguous, predictable rows only)*',
        '',
        '| | Predicted **click** | Predicted **noise** |',
        '|---|---|---|',
        f'| Actual **click** | TP = {g["tp"]} | FN = {g["fn"]} |',
        f'| Actual **noise** | FP = {g["fp"]} | TN = {g["tn"]} |',
        '',
        f'Recall = **{_fmt(g["recall"])}**  '
        f'| Precision = {_fmt(g["precision"])}  '
        f'| Specificity = {_fmt(g["specificity"])}  '
        f'| F1 = {_fmt(g["f1"])}',
        '',
    ]

    # ── Stage breakdown ──────────────────────────────────────────────────────
    lines += ['## Stage Breakdown  *(global)*', '',
              '| Stage | Rejected | % of input |', '|---|---|---|']
    cands = g['candidates'] or 1
    for stage, n in sorted(g['stage_counts'].items(), key=lambda kv: -kv[1]):
        lines.append(f'| {stage} | {n} | {100 * n / cands:.1f} % |')
    lines.append(f'| **Final confirmed** | **{g["confirmed"]}** | '
                 f'**{100 * g["confirmed"] / cands:.1f} %** |')
    lines += ['']

    # ── Per-folder ───────────────────────────────────────────────────────────
    lines += [
        '## Per-Folder Summary',
        '',
        '> Every nesting level is rolled up, so a recording under',
        '> `Plants/Aloe/meccanico` is counted in all three. Rates are total events /',
        '> total duration of the folder, never an average of per-file rates.',
        '',
        '| Folder | Files | Duration | Cands | Labelled | Algo /h | Click /h '
        '| Ambig /h | Noise /h |',
        '|---|---|---|---|---|---|---|---|---|',
    ]
    for r in sorted(per_folder, key=lambda x: x['group']):
        indent = '&nbsp;&nbsp;&nbsp;&nbsp;' * r['depth']
        leaf   = r['group'].split('/')[-1]
        lines.append(
            f'| {indent}`{leaf}` '
            f'| {r["n_files"]} '
            f'| {_dur_str(r["duration_s"]).split("  ")[0]} '
            f'| {r["candidates"]} '
            f'| {r["labeled"]} ({_pct(r["labelled_frac"], 1)}) '
            f'| {_fmt(r.get("rate_" + SERIES_ALGO), 2)} '
            f'| {_fmt(r.get("rate_" + SERIES_CLICK), 2)} '
            f'| {_fmt(r.get("rate_" + SERIES_AMBIG), 2)} '
            f'| {_fmt(r.get("rate_" + SERIES_NOISE), 2)} |'
        )
    lines += ['']

    # ── Per-folder time distribution ─────────────────────────────────────────
    lines += [
        '## Per-Folder Time Distribution',
        '',
        '> Bins pooled across the folder\'s recordings — every complete window is one',
        '> sample. Event *times* are never pooled: two recordings both start at t = 0,',
        '> so merging their timelines would manufacture clustering.',
        '',
    ]
    for r in sorted(per_folder, key=lambda x: x['group']):
        rows = aggregate_window_stats(counts_map, per_file, r['group'],
                                      opts['include_ambiguous'])
        # Nothing to say about a folder where no series has a usable window.
        if not any(x['n_bins'] >= MIN_BINS_FOR_STATS and x['n_events'] > 0
                   for x in rows):
            continue
        lines += [f'### `{r["group"]}`  —  {r["n_files"]} recording(s), '
                  f'{_dur_str(r["duration_s"])}', '']
        lines += _window_table(rows)

    # ── Per-file ─────────────────────────────────────────────────────────────
    lines += [
        '## Per-File Summary',
        '',
        '> `Lab%` = share of that recording\'s candidates carrying any label.',
        '> `Fano@5m` / `Fano@1h` are for the algorithm series where available,',
        '> otherwise for `label = 1`; the column says which. Full detail for every',
        '> series and window is in `window_stats.csv`.',
        '',
        '| File | Schema | Dur. | Cands | Lab% | Algo | Click | Ambig '
        '| Algo /h | Click /h | Fano src | Fano@5m | Fano@1h |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|---|',
    ]
    wr_index: dict[tuple, dict] = {(x['file'], x['series'], x['window_s']): x
                                   for x in window_rows}
    for r in sorted(per_file, key=lambda x: (x['source_folder'], x['file'])):
        fano_series = SERIES_ALGO if r['can_predict'] else SERIES_CLICK
        f5  = wr_index.get((r['file'], fano_series, 300.0),  {}).get('fano', float('nan'))
        f1h = wr_index.get((r['file'], fano_series, 3600.0), {}).get('fano', float('nan'))
        lines.append(
            f'| `{r["file"]}` '
            f'| {r["schema"]}{"" if r["can_predict"] else " ⚠"} '
            f'| {_dur_str(r["duration_s"]).split("  ")[0]} '
            f'| {r["candidates"]} '
            f'| {_pct(r["labelled_frac"], 1)}{"*" if r["exhaustive"] else ""} '
            f'| {r["confirmed"] if r["can_predict"] else "—"} '
            f'| {r["n_click"]} '
            f'| {r["n_ambiguous"]} '
            f'| {_fmt(r["algo_rate_hr"], 2) if r["can_predict"] else "—"} '
            f'| {_fmt(r["label_rate_hr"], 2)} '
            f'| {"algo" if r["can_predict"] else "label"} '
            f'| {_fmt(f5, 2)} '
            f'| {_fmt(f1h, 2)} |'
        )
    lines += ['', '`*` = exhaustively labelled (every candidate judged). '
                  '`⚠` = schema too old for this model — no algorithm series.', '']

    # ── Excluded recordings ──────────────────────────────────────────────────
    excluded = [r for r in per_file if not r['can_predict']]
    if excluded:
        lines += [
            '## Recordings Excluded From Prediction',
            '',
            f'{len(excluded)} recording(s) came from a CSV lacking columns this model '
            'or Stage 2 mode requires. They were NOT pushed through with missing '
            'values: a missing column passes every Stage 2 gate and is imputed by '
            'the SVM, which would return confident predictions computed from '
            'nothing. Their label series are reported in full above.',
            '',
            '| File | Schema | Missing columns |',
            '|---|---|---|',
        ]
        for r in excluded:
            miss = ', '.join(r.get('missing_cols', [])) or '—'
            lines.append(f'| `{r["file"]}` | {r["schema"]} | `{miss}` |')
        lines += ['']

    # ── Missing .paudio ──────────────────────────────────────────────────────
    missing = [r['file'] for r in per_file if not r['paudio_found']]
    if missing:
        lines += [
            '## Missing .paudio Files',
            '',
            f'{len(missing)} recording(s) had no matching .paudio file. They have no '
            'duration, so they contribute counts but NO rate and NO time-distribution '
            'windows — a rate over a guessed duration is worse than no rate.',
            '',
        ]
        lines += [f'- `{m}`' for m in missing]
        lines += ['']

    output_path.write_text('\n'.join(lines), encoding='utf-8')


# ── Click-distribution plots ──────────────────────────────────────────────────

def _cluster_clicks(
    click_times: list,
    gap_s: float = 60.0,
) -> list[tuple[float, float, int]]:
    """
    Group sorted click timestamps into time-proximity clusters.

    A new cluster starts whenever two consecutive clicks are more than gap_s
    seconds apart.  Returns a list of (t_first, t_last, n_clicks) tuples,
    one per cluster.
    """
    if not click_times:
        return []
    clusters: list[tuple[float, float, int]] = []
    t0 = t1 = float(click_times[0])
    count = 1
    for t in click_times[1:]:
        t = float(t)
        if t - t1 <= gap_s:
            t1 = t
            count += 1
        else:
            clusters.append((t0, t1, count))
            t0 = t1 = t
            count = 1
    clusters.append((t0, t1, count))
    return clusters


def _build_spikes(click_times: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (x, y) arrays for NaN-separated vertical spike lines in pyqtgraph.

    Each click at time t becomes three points:  (t, 0) → (t, 1) → (NaN, NaN).
    The NaN breaks the polyline so consecutive spikes are not connected.
    """
    if not click_times:
        return np.array([], dtype=float), np.array([], dtype=float)
    xs, ys = [], []
    for t in sorted(click_times):
        xs.extend([t, t, np.nan])
        ys.extend([0.0, 1.0, np.nan])
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def generate_click_plots(
    per_file: list[dict],
    df: pd.DataFrame,
    output_dir: Path,
    counts_map: dict,
    include_ambiguous: bool = False,
) -> None:
    """
    Render one time-distribution PNG per recording.

    Output: plots/{file_id}.png

    Layout:
        Row 0   — Raster of the full recording, ONE TRACK PER SERIES stacked on a
                  shared time axis. Reading the algorithm's track against the
                  human's on the same axis is the point of the figure: agreement,
                  misses and false positives are all vertical relationships.
        Row 1   — Counts per fixed window, as bars, at the smallest window size
                  that fits at least MIN_BINS_FOR_STATS complete bins. This is the
                  time distribution proper; the raster above it shows individual
                  events, this shows the density.
        Rows 2+ — One zoomed panel per cluster, over the union of all series, so a
                  burst the algorithm found and the labels missed still gets a
                  panel (capped at MAX_ZOOM_PANELS).
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError as exc:
        print(f'  Warning: skipping plots — matplotlib not available ({exc})')
        return

    # ── Constants ─────────────────────────────────────────────────────────────
    PLOT_W_IN       = 16      # figure width in inches (≈1600 px at 100 dpi)
    DPI             = 100
    OVERVIEW_BASE_IN = 0.75   # axes furniture: title, tick labels, padding
    LANE_H_IN        = 0.55   # per series lane, so 1 track is not given 4 tracks' room
    BINS_H_IN        = 1.9
    CLUSTER_GAP_S   = 60.0
    CLUSTER_MARGIN  = 15.0
    CLUSTER_PAD_FRAC= 0.20
    MAX_ZOOM_PANELS = 12

    BG           = '#1A1A2E'
    BASE_COLOR   = '#505060'
    BAND_COLOR   = '#E84040'
    TEXT_COLOR   = '#CCCCCC'
    GRID_COLOR   = '#2A2A4A'
    SPINE_COLOR  = '#444466'

    #: One colour per series, fixed across every figure so a colour means the same
    #: thing in any two plots put side by side.
    SERIES_COLOR = {
        SERIES_ALGO:      '#E84040',   # red    — what the pipeline says
        SERIES_CLICK:     '#3FC46B',   # green  — what the human says
        SERIES_CLICK_AMB: '#8FD9A8',   # pale green — the click+ambiguous bound
        SERIES_AMBIG:     '#E8B840',   # amber  — the reviewer could not decide
        SERIES_NOISE:     '#5A7FBF',   # blue   — judged noise
    }
    #: Noise is off the raster by default: it is 90-99 % of every recording and
    #: would render as a solid bar that hides everything else. Its counts are in
    #: the tables, where they are readable.
    RASTER_SERIES = [SERIES_ALGO, SERIES_CLICK, SERIES_CLICK_AMB, SERIES_AMBIG]

    _NICE = [10, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600]

    def _fmt_time(t, _pos=None):
        t = max(0, int(round(t)))
        h, rem = divmod(t, 3600)
        m, s   = divmod(rem, 60)
        return f'{h}:{m:02d}:{s:02d}' if h else f'{m:02d}:{s:02d}'

    def _style_ax(ax, x_lo: float, x_hi: float, title: str) -> None:
        ax.set_facecolor(BG)
        ax.set_xlim(x_lo, x_hi)
        ax.set_title(title, color=TEXT_COLOR, fontsize=8,
                     loc='left', pad=3, fontweight='normal')
        ax.tick_params(axis='x', colors=TEXT_COLOR, labelsize=7, length=3)
        ax.tick_params(axis='y', colors=TEXT_COLOR, labelsize=7, length=3)
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE_COLOR)
        ax.grid(True, which='major', axis='x',
                color=GRID_COLOR, linewidth=0.6, zorder=0)
        span  = max(x_hi - x_lo, 1.0)
        major = min(_NICE, key=lambda v: abs(v - span / 8.0))
        ax.xaxis.set_major_locator(mticker.MultipleLocator(major))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_time))

    def _draw_raster(ax, tracks: list[tuple], x_lo: float, x_hi: float) -> None:
        """tracks = [(series_name, times), ...] — one lane each, top to bottom."""
        n = max(len(tracks), 1)
        ax.set_ylim(-0.5, n - 0.5)
        ax.set_yticks(range(n))
        # Lane 0 at the TOP, which is how a reader scans a stack of tracks.
        ax.set_yticklabels([_series_label(t[0]).split(' (')[0] for t in tracks],
                           fontsize=7, color=TEXT_COLOR)
        ax.invert_yaxis()
        for lane, (name, times) in enumerate(tracks):
            colour = SERIES_COLOR.get(name, BASE_COLOR)
            ax.axhline(lane, color=BASE_COLOR, linewidth=0.6, zorder=1)
            if len(times):
                ax.vlines(times, lane - 0.34, lane + 0.34,
                          colors=colour, linewidth=1.2, zorder=2)

    def _draw_bins(ax, file_id: str, tracks: list[tuple], window_s: float,
                   x_lo: float, x_hi: float) -> None:
        """Grouped bars: per-window counts, one bar group per bin."""
        drawn = [(name, counts_map.get((file_id, name, window_s)))
                 for name, _ in tracks]
        drawn = [(n, c) for n, c in drawn if c is not None and c.size]
        if not drawn:
            ax.text(0.5, 0.5, 'no complete window fits this recording',
                    transform=ax.transAxes, ha='center', va='center',
                    color=TEXT_COLOR, fontsize=8)
            ax.set_yticks([])
            return
        k = len(drawn)
        # The whole group occupies GROUP_FRAC of its window, so consecutive bins
        # stay visually separated. At full width the four bars of one bin span
        # almost the entire window and read as four adjacent bins, which inverts
        # the message of the panel.
        GROUP_FRAC = 0.72
        width = window_s * GROUP_FRAC / k

        n_bins = max(c.size for _, c in drawn)
        for b in range(n_bins + 1):
            ax.axvline(b * window_s, color=SPINE_COLOR, linewidth=0.5,
                       linestyle=':', zorder=1)

        for j, (name, counts) in enumerate(drawn):
            centres = (np.arange(counts.size) + 0.5) * window_s
            offset  = (j - (k - 1) / 2.0) * width
            ax.bar(centres + offset, counts, width=width * 0.9,
                   color=SERIES_COLOR.get(name, BASE_COLOR),
                   label=_series_label(name).split(' (')[0],
                   zorder=2, linewidth=0)
        ax.set_ylabel('events / window', color=TEXT_COLOR, fontsize=7)
        leg = ax.legend(loc='upper right', fontsize=7, framealpha=0.25,
                        facecolor=BG, edgecolor=SPINE_COLOR)
        for txt in leg.get_texts():
            txt.set_color(TEXT_COLOR)

    def _ts(t: float) -> str:
        v = max(0, int(round(t)))
        h, rem = divmod(v, 3600)
        m, sec = divmod(rem, 60)
        return f'{h}:{m:02d}:{sec:02d}' if h else f'{m:02d}:{sec:02d}'

    # ── Per-file plots ────────────────────────────────────────────────────────
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    for r in per_file:
        file_id    = r['file']
        duration_s = r['duration_s']
        has_dur    = not np.isnan(duration_s)

        sub   = df[df['file'] == file_id]
        masks = series_masks(sub, include_ambiguous)

        tracks: list[tuple] = []
        for name in RASTER_SERIES:
            if name not in masks:
                continue
            if name == SERIES_ALGO and not r['can_predict']:
                continue
            # An all-empty lane for a series nobody used is noise on the figure;
            # the algo and label_click lanes are always drawn, because "the
            # algorithm found nothing here" is itself a result worth seeing.
            times = event_times(sub, masks[name])
            if not len(times) and name not in (SERIES_ALGO, SERIES_CLICK):
                continue
            # With no ambiguous rows in this recording the union series IS
            # label_click, and drawing both puts two identical lanes on the figure
            # and two identical bars in every bin. It stays in the tables, where a
            # bound that happens to be tight is still worth stating.
            if name == SERIES_CLICK_AMB and r['n_ambiguous'] == 0:
                continue
            tracks.append((name, times))
        if not tracks:
            continue

        # np.unique, not concatenate: the series OVERLAP by construction — an
        # event the algorithm found and a human labelled is one event in two
        # tracks. Concatenating counts it twice, and the cluster panels then
        # announce twice as many events as the recording contains.
        all_ts  = np.unique(np.concatenate([t for _, t in tracks if len(t)])) \
                  if any(len(t) for _, t in tracks) else np.empty(0)
        x_end   = (duration_s if has_dur
                   else (float(all_ts.max()) * 1.08 if all_ts.size else 60.0))
        dur_lbl = _dur_str(duration_s) if has_dur else 'duration unknown'

        # Smallest window with enough complete bins to be worth drawing.
        bin_window = next((w for w in WINDOWS_S
                           if has_dur and int(duration_s // w) >= MIN_BINS_FOR_STATS),
                          None)

        clusters    = _cluster_clicks(all_ts.tolist(), gap_s=CLUSTER_GAP_S)
        n_clusters  = len(clusters)
        n_zoom_rows = min(n_clusters, MAX_ZOOM_PANELS)
        truncated   = n_clusters > MAX_ZOOM_PANELS

        zoom_windows: list[tuple[float, float, int]] = []
        for t0, t1, k in clusters[:n_zoom_rows]:
            span   = max(t1 - t0, 0.0)
            margin = max(CLUSTER_MARGIN, span * CLUSTER_PAD_FRAC)
            zoom_windows.append((max(0.0, t0 - margin), min(x_end, t1 + margin), k))

        overview_h = OVERVIEW_BASE_IN + len(tracks) * LANE_H_IN
        zoom_h     = OVERVIEW_BASE_IN + len(tracks) * LANE_H_IN * 0.72
        n_rows = 1 + (1 if bin_window else 0) + n_zoom_rows
        fig_h  = (overview_h + (BINS_H_IN if bin_window else 0)
                  + n_zoom_rows * zoom_h)
        fig, axes = plt.subplots(n_rows, 1, figsize=(PLOT_W_IN, fig_h),
                                 constrained_layout=True, squeeze=False)
        axes = [ax for row in axes for ax in row]
        fig.patch.set_facecolor(BG)

        # Row 0: raster overview
        counts_lbl = '  ·  '.join(
            f'{_series_label(n).split(" (")[0]}: {len(t)}' for n, t in tracks)
        ov_title = (f'{file_id}  ·  {dur_lbl}  ·  {counts_lbl}'
                    + (f'  ·  {n_clusters} cluster(s)' if n_clusters > 1 else '')
                    + ('' if r['can_predict']
                       else '  ·  ⚠ schema too old for this model — no algorithm track'))
        _style_ax(axes[0], 0, x_end, ov_title)
        _draw_raster(axes[0], tracks, 0, x_end)
        for x_lo, x_hi, _ in zoom_windows:
            axes[0].axvspan(x_lo, x_hi, alpha=0.15, color=BAND_COLOR,
                            linewidth=0, zorder=0)

        row = 1
        if bin_window:
            n_bins = int(duration_s // bin_window)
            _style_ax(axes[row], 0, x_end,
                      f'Counts per {int(bin_window) // 60}-min window  ·  '
                      f'{n_bins} complete bin(s)  ·  '
                      f'trailing {duration_s - n_bins * bin_window:.0f} s dropped')
            _draw_bins(axes[row], file_id, tracks, float(bin_window), 0, x_end)
            row += 1

        # Zoom rows
        for i, ((t0, t1, k), (x_lo, x_hi, _)) in enumerate(
                zip(clusters[:n_zoom_rows], zoom_windows)):
            suffix = ''
            if truncated and i == n_zoom_rows - 1:
                suffix = f'  (+{n_clusters - MAX_ZOOM_PANELS} more cluster(s) not shown)'
            _style_ax(axes[row + i], x_lo, x_hi,
                      f'Cluster {i + 1}/{n_clusters}  ·  {_ts(t0)} – {_ts(t1)}'
                      f'  ·  {k} event(s)' + suffix)
            _draw_raster(axes[row + i],
                         [(n, t[(t >= x_lo) & (t <= x_hi)]) for n, t in tracks],
                         x_lo, x_hi)

        out_path = plots_dir / f'{file_id}.png'
        fig.savefig(str(out_path), dpi=DPI, facecolor=BG)
        plt.close(fig)

    print(f'    Saved {len(per_file)} plot(s) to {plots_dir}')


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='PlantLeaf — corpus click-rate and time-distribution analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        'inputs', nargs='+', type=Path,
        help='Input folders (searched recursively for *.csv) and/or CSV files',
    )
    parser.add_argument(
        '--model', required=True, type=Path,
        help='Trained SVM model (.pkl from train_svm.py)',
    )
    parser.add_argument(
        '--paudio-dir', required=True, type=Path, dest='paudio_dir',
        help='Root directory to search recursively for .paudio files',
    )
    parser.add_argument(
        '--output-dir', type=Path, default=Path('.'), dest='output_dir',
        help='Directory for output files (default: current directory)',
    )
    parser.add_argument(
        '--stage2-mode', default=_STAGE2_MODE_DEFAULT, choices=list(_STAGE2_MODES),
        dest='stage2_mode',
        help=f'Stage 2 rule (default: {_STAGE2_MODE_DEFAULT})',
    )
    parser.add_argument(
        '--windows', default='5,10,15,30,60', dest='windows',
        help='Comma-separated binning scales in MINUTES (default: 5,10,15,30,60). '
             'A window only appears for a recording long enough to hold at least '
             f'{MIN_BINS_FOR_STATS} complete bins of it.',
    )
    parser.add_argument(
        '--include-ambiguous', action='store_true', dest='include_ambiguous',
        help='Add the label_click_incl_amb series (label 1 or 2) as an UPPER bound '
             'beside label_click. Never modifies label_click itself.',
    )
    parser.add_argument(
        '--exhaustive-only', action='store_true', dest='exhaustive_only',
        help='Restrict the labelled series to recordings where every candidate '
             'carries a label. Algorithm series are unaffected.',
    )
    parser.add_argument(
        '--rows-csv', type=Path, default=None, dest='rows_csv',
        help='Override output path for the rows CSV '
             '(default: <output-dir>/evaluated_rows.csv)',
    )
    parser.add_argument(
        '--report-md', type=Path, default=None, dest='report_md',
        help='Override output path for the report (default: <output-dir>/report.md)',
    )
    parser.add_argument(
        '--no-plots', action='store_true', dest='no_plots',
        help='Skip generating time-distribution PNG plots',
    )
    args = parser.parse_args()

    try:
        set_windows([float(x) for x in args.windows.split(',') if x.strip()])
    except ValueError as exc:
        print(f'ERROR: bad --windows {args.windows!r}: {exc}')
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_csv_path = args.rows_csv  or (args.output_dir / 'evaluated_rows.csv')
    report_path   = args.report_md or (args.output_dir / 'report.md')

    # ── Validate ──────────────────────────────────────────────────────────────
    if not args.model.exists():
        print(f'ERROR: model not found: {args.model}')
        sys.exit(1)
    if not args.paudio_dir.exists():
        print(f'ERROR: --paudio-dir not found: {args.paudio_dir}')
        sys.exit(1)

    model = joblib.load(args.model)
    for key in ('pipeline', 'threshold', 'features'):
        if key not in model:
            print(f"ERROR: model is missing key '{key}'. Re-train with train_svm.py.")
            sys.exit(1)

    print('PlantLeaf — Click Rate & Time Distribution')
    print(f'  Model      : {args.model.name}  '
          f'(kernel={model.get("kernel","?")}, threshold={model["threshold"]:.3f}, '
          f'nan_policy={model.get("nan_policy","sentinel")})')
    print(f'  Stage 2    : {args.stage2_mode}')
    print(f'  Windows    : {", ".join(f"{w // 60} min" for w in WINDOWS_S)}')
    print(f'  .paudio dir: {args.paudio_dir}')
    print(f'  Output dir : {args.output_dir}')
    print()

    # ── Discover CSV files ────────────────────────────────────────────────────
    print('Discovering CSV files:')
    csv_paths = resolve_csvs(args.inputs)
    if not csv_paths:
        print('ERROR: no CSV files found.')
        sys.exit(1)
    print(f'  Total: {len(csv_paths)} CSV file(s)\n')

    # ── Schema capability, per source CSV ─────────────────────────────────────
    print('Checking schemas against the model and Stage 2 mode:')
    caps    = classify_sources(csv_paths, model, args.stage2_mode)
    n_ok    = sum(1 for v in caps.values() if v['can_predict'])
    n_short = len(caps) - n_ok
    print(f'  {n_ok} file(s) can be scored; {n_short} lack required columns')
    if n_short:
        example = next(v['missing'] for v in caps.values() if not v['can_predict'])
        print(f'  Missing (first such file): {", ".join(example)}')
        print('  Those files keep their LABEL series and are excluded from every '
              'algorithm statistic.')
    print()

    # ── Index .paudio files ───────────────────────────────────────────────────
    print(f'Indexing .paudio files under {args.paudio_dir} ...')
    paudio_index = build_paudio_index(args.paudio_dir)
    print(f'  Found {len(paudio_index)} .paudio file(s)\n')

    # ── Load CSVs ─────────────────────────────────────────────────────────────
    print(f'Loading {len(csv_paths)} CSV file(s):')
    df = load_csvs(csv_paths)
    print(f'  Total rows: {len(df)}\n')

    df['_schema']      = df['_source_csv'].map(lambda x: caps.get(x, {}).get('schema', 'v5'))
    df['_can_predict'] = df['_source_csv'].map(
        lambda x: bool(caps.get(x, {}).get('can_predict', False)))

    # ── Initialise pipeline output columns ────────────────────────────────────
    df['svm_probability'] = np.nan
    df['svm_prediction']  = pd.array([pd.NA] * len(df), dtype='Int64')
    df['stage_blocked']   = ''

    # ── Run pipeline stages on the rows whose schema supports them ────────────
    # The unsupported rows are pulled OUT of the frame first rather than masked in
    # place: apply_stage2/3/4 all key off `stage_blocked == ''`, so leaving them in
    # with a marker would work, but Stage 4's groupby and the Stage 3 feature
    # matrix would both still touch them. Splitting is the version that cannot be
    # got subtly wrong later.
    scorable = df[df['_can_predict']].copy()
    skipped  = df[~df['_can_predict']].copy()

    if len(scorable):
        print('Running Stages 2/3/4 on the scorable rows:')
        scorable = apply_stage2(scorable, args.stage2_mode)
        scorable = apply_stage3(scorable, model)
        scorable = apply_stage4(scorable)
    else:
        print('No rows could be scored — every input CSV lacks required columns.')

    if len(skipped):
        skipped['stage_blocked'] = STAGE_NO_SCHEMA
        print(f'  {len(skipped)} row(s) marked {STAGE_NO_SCHEMA!r} '
              f'(schema too old for this model)')

    df = pd.concat([scorable, skipped], ignore_index=True) if len(skipped) else scorable

    # ── Per-file statistics ───────────────────────────────────────────────────
    print('\nComputing per-file statistics ...')
    per_file = compute_per_file_stats(df, paudio_index, args.include_ambiguous)

    # Attach the missing-column list so the report can name what each file lacks.
    src_by_file = (df.groupby('file')['_source_csv'].first().to_dict()
                   if '_source_csv' in df.columns else {})
    for rec in per_file:
        rec['missing_cols'] = caps.get(src_by_file.get(rec['file'], ''), {}).get('missing', [])

    # --exhaustive-only zeroes the LABEL series on partially-labelled recordings.
    # The algorithm series is untouched: the schema, not the labelling, decides
    # whether a prediction exists, and dropping algorithm counts here would make
    # the two series answer different questions on the same table row.
    if args.exhaustive_only:
        n_dropped = 0
        for rec in per_file:
            if rec['exhaustive']:
                continue
            n_dropped += 1
            for name in LABEL_SERIES:
                rec[f'n_{name}']    = float('nan')
                rec[f'rate_{name}'] = float('nan')
            rec['n_click'] = rec['n_noise'] = rec['n_ambiguous'] = 0
            rec['labeled'] = 0
            rec['tp'] = rec['fp'] = rec['fn'] = rec['tn'] = 0
            rec['label_rate_hr'] = float('nan')
        print(f'  --exhaustive-only: label series suppressed on {n_dropped} '
              f'partially-labelled recording(s)')

    global_stats = aggregate_stats(per_file)
    per_folder   = aggregate_by_folder(per_file, args.inputs)   # also sets rec['groups']

    # ── Time distribution ─────────────────────────────────────────────────────
    print('Computing time distributions ...')
    window_rows, bin_rows, counts_map = compute_time_distribution(
        df, per_file, args.include_ambiguous)
    print(f'  {len(window_rows)} (file × series × window) row(s), '
          f'{len(bin_rows)} bin(s)')

    # ── Console summary ───────────────────────────────────────────────────────
    g = global_stats
    print(f'\n{"=" * 62}')
    print(f'  Global summary  (threshold = {model["threshold"]:.3f})')
    print(f'{"=" * 62}')
    print(f'  Recordings             : {g["n_files"]}  '
          f'({g["n_no_schema"]} excluded from prediction)')
    print(f'  Total duration         : {_dur_str(g["duration_s"])}')
    print(f'  Input candidates       : {g["candidates"]}')
    print(f'  Confirmed clicks       : {g["confirmed"]}')
    for stage, n in sorted(g['stage_counts'].items(), key=lambda kv: -kv[1]):
        print(f'  Blocked — {stage:<16}: {n}')
    print('  ' + '-' * 50)
    print(f'  Labelled rows          : {g["labeled"]}  '
          f'({_pct(g["labelled_frac"])} of candidates, '
          f'{g["n_exhaustive"]} recording(s) exhaustive)')
    if g['tp'] + g['fp'] + g['fn'] + g['tn'] > 0:
        print(f'  Confusion  TP={g["tp"]}  FP={g["fp"]}  FN={g["fn"]}  TN={g["tn"]}')
        print(f'  Recall     : {_fmt(g["recall"])}   Precision: {_fmt(g["precision"])}   '
              f'F1: {_fmt(g["f1"])}')
    print('  ' + '-' * 50)
    for name in SERIES_ORDER:
        if f'rate_{name}' not in g:
            continue
        if name == SERIES_CLICK_AMB and not args.include_ambiguous:
            continue
        print(f'  {_series_label(name):<36}: {_fmt(g[f"rate_{name}"], 2):>8} /h  '
              f'({g.get(f"n_{name}", 0)} events)')
    if g['n_paudio_missing']:
        print(f'  Warning: {g["n_paudio_missing"]} file(s) had no matching .paudio '
              f'(no duration → no rate, no windows)')

    # ── Save outputs ──────────────────────────────────────────────────────────
    if not args.no_plots:
        print('\nGenerating time-distribution plots ...')
        generate_click_plots(per_file, df, args.output_dir, counts_map,
                             args.include_ambiguous)

    df_out = df.drop(columns=['_source_csv', '_schema', '_can_predict'], errors='ignore')
    df_out.to_csv(rows_csv_path, index=False)
    print(f'\n  Saved rows CSV     : {rows_csv_path}  ({len(df_out)} rows)')

    # stage_counts / groups are dicts and lists — flattened out of the flat CSVs.
    pf = pd.DataFrame([{k: v for k, v in r.items()
                        if k not in ('stage_counts', 'groups', 'missing_cols')}
                       for r in per_file])
    pf.to_csv(args.output_dir / 'per_file_stats.csv', index=False)
    print(f'  Saved per-file     : {args.output_dir / "per_file_stats.csv"}  '
          f'({len(pf)} rows)')

    folder_rows = [{k: v for k, v in r.items() if k != 'stage_counts'}
                   for r in per_folder]
    pd.DataFrame(folder_rows).to_csv(args.output_dir / 'per_folder_stats.csv', index=False)
    print(f'  Saved per-folder   : {args.output_dir / "per_folder_stats.csv"}  '
          f'({len(folder_rows)} rows)')

    pd.DataFrame(window_rows).to_csv(args.output_dir / 'window_stats.csv', index=False)
    pd.DataFrame(bin_rows).to_csv(args.output_dir / 'time_bins.csv', index=False)
    print(f'  Saved window stats : {args.output_dir / "window_stats.csv"}  '
          f'({len(window_rows)} rows)')
    print(f'  Saved time bins    : {args.output_dir / "time_bins.csv"}  '
          f'({len(bin_rows)} rows)')

    write_report_md(
        per_file, per_folder, global_stats, window_rows, counts_map, model,
        report_path, args.inputs,
        opts={'stage2_mode': args.stage2_mode,
              'include_ambiguous': args.include_ambiguous,
              'exhaustive_only': args.exhaustive_only},
    )
    print(f'  Saved report       : {report_path}')


if __name__ == '__main__':
    main()
