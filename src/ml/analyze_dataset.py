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
analyze_dataset.py  —  PlantLeaf v5  full dataset analysis
===========================================================

Recursively scans one or more input folders for *_candidates.csv files, runs
Stages 2/3/4 of the click detection pipeline on all of them, and writes:

    evaluated_rows.csv  — every input row with svm_probability, svm_prediction,
                          and stage_blocked appended (same column format as
                          evaluate_candidates.py).

    summary.md          — per-file and global statistics table:
                          TP, FP, FN, TN, recall, precision, specificity,
                          recording duration (s), and click rate (clicks / h).
                          Per-file rows are sorted by folder then file name.

    plots/              — one PNG per recording (1600 px wide, dark background):
                            • Row 0: full-recording overview with semi-transparent
                              red bands marking each cluster window.
                            • Rows 1+: one zoomed panel per click cluster.
                          Clusters are groups of clicks with an intra-group gap
                          ≤ CLUSTER_GAP_S (60 s by default); zoom panels are
                          capped at MAX_ZOOM_PANELS (12).  Skipped with --no-plots.

Recording durations are read from the .paudio files found under --paudio-dir
(searched recursively).  The 'file' column in each candidate CSV must equal the
.paudio stem (filename without the .paudio extension).  Duration is estimated
from file size: n_frames × FFT_SIZE / fs, where each frame is 154 bins × 5
bytes.  This is accurate to < 0.1 % because the CLCK click-data section at the
end of a .paudio file is tiny relative to the raw FFT stream.

Usage
-----
    python src/ml/analyze_dataset.py \\
        --model  src/ml/plantleaf_svm_v5_nofitcoverage.pkl \\
        --paudio-dir  /path/to/paudio/recordings/ \\
        /path/to/Dataset/PLANTS/ \\
        --output-dir  /path/to/results/

    # Multiple input folders, mix of files and folders:
    python src/ml/analyze_dataset.py \\
        --model model.pkl --paudio-dir /recordings/ \\
        Dataset/ALOE/ Dataset/CACTUS/ extra_file.csv \\
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

def resolve_csvs(inputs: list[Path]) -> list[Path]:
    """
    Expand a mixed list of files and/or folders into a sorted list of CSV
    paths.  Folders are searched recursively for *.csv files.
    """
    paths: list[Path] = []
    for inp in inputs:
        if inp.is_dir():
            found = sorted(inp.rglob('*.csv'))
            if not found:
                print(f"  Warning: no *.csv found recursively in {inp}")
            else:
                print(f"  Folder {inp.name}: {len(found)} CSV file(s) found "
                      f"(recursive)")
            paths.extend(found)
        elif inp.is_file():
            paths.append(inp)
        else:
            print(f"  Warning: {inp} does not exist — skipping")
    return paths


# ── Per-file statistics ───────────────────────────────────────────────────────

def _safe_div(num: float | int, den: float | int) -> float:
    return float(num) / float(den) if den else float('nan')


def compute_per_file_stats(
    df: pd.DataFrame,
    paudio_index: dict[str, Path],
) -> list[dict]:
    """
    Compute per-recording statistics from the annotated DataFrame.

    One dict per unique value of the 'file' column, containing:
        file            — recording stem name
        source_folder   — parent directory of the source CSV (_source_csv col)
        candidates      — total rows from that recording
        confirmed       — rows that survived all stages (stage_blocked is NaN)
        labeled         — rows with a non-NaN 'label' value
        tp, fp, fn, tn  — confusion matrix (labeled rows only)
        recall, precision, specificity, f1  — derived metrics
        duration_s      — recording duration from .paudio (NaN if not found)
        algo_rate_hr   — algorithm-confirmed clicks / h (NaN if no duration)
        label_rate_hr  — ground-truth clicks (TP+FN) / h (NaN if no labels or no duration)
        paudio_found    — bool
    """
    stats = []

    for file_id, grp in df.groupby('file', sort=True):
        total_cands = len(grp)
        # stage_blocked is '' (empty string) for survivors while the DataFrame
        # is in memory; NaN only appears after writing/reading the CSV.
        confirmed = int((grp['stage_blocked'].fillna('') == '').sum())

        # Source folder: take from _source_csv of the first row
        src_csv = grp['_source_csv'].iloc[0] if '_source_csv' in grp.columns else ''
        source_folder = str(Path(src_csv).parent) if src_csv else ''

        # Stage breakdown for this file
        n_s2_r2  = int((grp['stage_blocked'] == 'Stage2_R2').sum())
        n_s2_spr = int((grp['stage_blocked'] == 'Stage2_SPR').sum())
        n_s3     = int((grp['stage_blocked'] == 'Stage3_SVM').sum())
        n_s4     = int((grp['stage_blocked'] == 'Stage4_dedup').sum())

        # Confusion matrix (only rows with manual labels)
        tp = fp = fn = tn = 0
        n_labeled = 0
        if 'label' in grp.columns:
            labeled = grp[grp['label'].notna()].copy()
            labeled['label']   = labeled['label'].astype(int)
            labeled['pred_1']  = (labeled['stage_blocked'].fillna('') == '').astype(int)
            n_labeled = len(labeled)
            tp = int(((labeled['label'] == 1) & (labeled['pred_1'] == 1)).sum())
            fp = int(((labeled['label'] == 0) & (labeled['pred_1'] == 1)).sum())
            fn = int(((labeled['label'] == 1) & (labeled['pred_1'] == 0)).sum())
            tn = int(((labeled['label'] == 0) & (labeled['pred_1'] == 0)).sum())

        recall      = _safe_div(tp, tp + fn)
        precision   = _safe_div(tp, tp + fp)
        specificity = _safe_div(tn, tn + fp)
        f1          = _safe_div(2 * precision * recall, precision + recall)

        # Duration from .paudio
        paudio_path = paudio_index.get(file_id)
        duration_s  = read_paudio_duration(paudio_path) if paudio_path else None
        paudio_found = paudio_path is not None

        has_dur = duration_s is not None and duration_s > 0
        # Algorithm rate: all confirmed clicks / duration (includes unlabeled rows)
        algo_rate_hr  = (confirmed / (duration_s / 3600.0)
                         if has_dur else float('nan'))
        # Label rate: ground-truth clicks (TP+FN) / duration (only when both
        # manual labels AND a .paudio file are available for this recording)
        label_rate_hr = ((tp + fn) / (duration_s / 3600.0)
                         if has_dur and n_labeled > 0 else float('nan'))

        stats.append({
            'file':            file_id,
            'source_folder':   source_folder,
            'candidates':      total_cands,
            'confirmed':       confirmed,
            'labeled':         n_labeled,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'recall':          recall,
            'precision':       precision,
            'specificity':     specificity,
            'f1':              f1,
            'n_s2_r2':  n_s2_r2,
            'n_s2_spr': n_s2_spr,
            'n_s3':     n_s3,
            'n_s4':     n_s4,
            'duration_s':      duration_s if duration_s is not None else float('nan'),
            'algo_rate_hr':   algo_rate_hr,
            'label_rate_hr':  label_rate_hr,
            'paudio_found':    paudio_found,
        })

    return stats


def aggregate_stats(per_file: list[dict]) -> dict:
    """Sum numeric columns across all per-file records into a global dict."""
    total_candidates = sum(r['candidates']  for r in per_file)
    total_confirmed  = sum(r['confirmed']   for r in per_file)
    total_labeled    = sum(r['labeled']     for r in per_file)
    total_tp = sum(r['tp'] for r in per_file)
    total_fp = sum(r['fp'] for r in per_file)
    total_fn = sum(r['fn'] for r in per_file)
    total_tn = sum(r['tn'] for r in per_file)
    total_s2_r2  = sum(r['n_s2_r2']  for r in per_file)
    total_s2_spr = sum(r['n_s2_spr'] for r in per_file)
    total_s3     = sum(r['n_s3']     for r in per_file)
    total_s4     = sum(r['n_s4']     for r in per_file)

    # Sum durations only for files where the .paudio was found
    dur_values = [r['duration_s'] for r in per_file
                  if r['paudio_found'] and not np.isnan(r['duration_s'])]
    total_dur  = sum(dur_values) if dur_values else float('nan')
    n_missing  = sum(1 for r in per_file if not r['paudio_found'])

    # Algorithm rate: confirmed clicks from files with known duration
    confirmed_with_dur = sum(r['confirmed'] for r in per_file
                             if r['paudio_found'] and not np.isnan(r['duration_s']))
    global_algo_rate = (_safe_div(confirmed_with_dur, total_dur / 3600.0)
                        if total_dur and not np.isnan(total_dur) else float('nan'))

    # Label rate: ground-truth clicks (TP+FN) from files that have BOTH
    # manual labels and a known duration
    label_dur = sum(r['duration_s'] for r in per_file
                    if r['paudio_found'] and not np.isnan(r['duration_s'])
                    and r['labeled'] > 0)
    label_clicks = sum(r['tp'] + r['fn'] for r in per_file
                       if r['paudio_found'] and not np.isnan(r['duration_s'])
                       and r['labeled'] > 0)
    global_label_rate = (_safe_div(label_clicks, label_dur / 3600.0)
                         if label_dur else float('nan'))

    return {
        'candidates':  total_candidates,
        'confirmed':   total_confirmed,
        'labeled':     total_labeled,
        'tp': total_tp, 'fp': total_fp, 'fn': total_fn, 'tn': total_tn,
        'recall':      _safe_div(total_tp, total_tp + total_fn),
        'precision':   _safe_div(total_tp, total_tp + total_fp),
        'specificity': _safe_div(total_tn, total_tn + total_fp),
        'f1':          _safe_div(
                           2 * _safe_div(total_tp, total_tp + total_fp) *
                               _safe_div(total_tp, total_tp + total_fn),
                           _safe_div(total_tp, total_tp + total_fp) +
                               _safe_div(total_tp, total_tp + total_fn)),
        'n_s2_r2':  total_s2_r2,
        'n_s2_spr': total_s2_spr,
        'n_s3':     total_s3,
        'n_s4':     total_s4,
        'duration_s':       total_dur,
        'algo_rate_hr':    global_algo_rate,
        'label_rate_hr':   global_label_rate,
        'n_paudio_missing': n_missing,
    }


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


def write_summary_md(
    per_file: list[dict],
    global_stats: dict,
    model: dict,
    output_path: Path,
    input_roots: list[Path],
) -> None:
    g = global_stats
    lines: list[str] = []

    # ── Header ───────────────────────────────────────────────────────────────
    # Extract fitted hyperparameters from the sklearn Pipeline's SVC step
    try:
        svc = model['pipeline'].named_steps.get('svm') or model['pipeline'].named_steps.get('svc')
        hparam_str = '  '.join(f'{k}={v}' for k, v in svc.get_params().items()
                                if k in ('C', 'gamma', 'degree', 'coef0'))
    except Exception:
        hparam_str = ''

    lines += [
        '# PlantLeaf v5 — Dataset Evaluation Summary',
        '',
        f'**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M")}  ',
        f'**Model:** `{model.get("kernel","?").upper()}` kernel  '
        + (f'— {hparam_str}  ' if hparam_str else '')
        + f'threshold={model["threshold"]:.3f}  ',
        f'**Features ({len(model["features"])}):** '
        f'`{", ".join(model["features"])}`  ',
        f'**Input folders:** {", ".join(str(r) for r in input_roots)}  ',
        '',
    ]

    # ── Global summary ────────────────────────────────────────────────────────
    lines += [
        '## Global Summary',
        '',
        '| | Value |',
        '|---|---|',
        f'| Input candidates | {g["candidates"]} |',
        f'| Confirmed clicks | {g["confirmed"]} |',
        f'| Labeled rows | {g["labeled"]} |',
        f'| Total duration | {_dur_str(g["duration_s"])} |',
        f'| Click rate (algorithm) | {_fmt(g["algo_rate_hr"], 2)} clicks / h |',
        f'| Click rate (manual labels) | {_fmt(g["label_rate_hr"], 2)} clicks / h |',
        '',
        '### Confusion matrix  *(labeled rows only)*',
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

    # ── Stage breakdown ───────────────────────────────────────────────────────
    lines += [
        '## Stage Breakdown  *(global)*',
        '',
        '| Stage | Rejected | % of input |',
        '|---|---|---|',
    ]
    cands = g['candidates'] or 1
    for stage, n in [('Stage2_R2',    g['n_s2_r2']),
                     ('Stage2_SPR',   g['n_s2_spr']),
                     ('Stage3_SVM',   g['n_s3']),
                     ('Stage4_dedup', g['n_s4'])]:
        lines.append(f'| {stage} | {n} | {100*n/cands:.1f} % |')
    lines.append(f'| **Final confirmed** | **{g["confirmed"]}** | '
                 f'**{100*g["confirmed"]/cands:.1f} %** |')
    lines += ['']

    # ── Per-file summary ──────────────────────────────────────────────────────
    lines += [
        '## Per-File Summary',
        '',
        '> Metrics computed only on rows with manual `label` annotation.  '
        '`N/A` = no labeled rows or no .paudio file found.',
        '',
        '| File | Cands | Conf. | Labeled | TP | FP | FN | TN | '
        'Recall | Prec. | Spec. | F1 | Dur. (s) | Rate algo (/h) | Rate label (/h) |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|',
    ]
    for r in per_file:
        lbl = r['labeled']
        lines.append(
            f'| `{r["file"]}` '
            f'| {r["candidates"]} '
            f'| {r["confirmed"]} '
            f'| {lbl} '
            f'| {r["tp"] if lbl else "—"} '
            f'| {r["fp"] if lbl else "—"} '
            f'| {r["fn"] if lbl else "—"} '
            f'| {r["tn"] if lbl else "—"} '
            f'| {_fmt(r["recall"]) if lbl else "—"} '
            f'| {_fmt(r["precision"]) if lbl else "—"} '
            f'| {_fmt(r["specificity"]) if lbl else "—"} '
            f'| {_fmt(r["f1"]) if lbl else "—"} '
            f'| {_fmt(r["duration_s"], 1)} '
            f'| {_fmt(r["algo_rate_hr"], 2)} '
            f'| {_fmt(r["label_rate_hr"], 2) if lbl else "—"} |'
        )

    lines += ['']

    # ── Missing .paudio warnings ──────────────────────────────────────────────
    missing = [r['file'] for r in per_file if not r['paudio_found']]
    if missing:
        lines += [
            '## Missing .paudio Files',
            '',
            f'{len(missing)} recording(s) had no matching .paudio file — '
            'duration and click rate are N/A for these:',
            '',
        ]
        for m in missing:
            lines.append(f'- `{m}`')
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
) -> None:
    """
    Render one click-distribution PNG per recording using matplotlib.

    Output: plots/{file_id}_clicks.png

    Layout:
        Row 0  — Overview: full recording with semi-transparent red bands marking
                 each cluster window.
        Rows 1+ — One zoomed panel per click cluster (gap > CLUSTER_GAP_S starts
                 a new cluster; capped at MAX_ZOOM_PANELS).
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
    OVERVIEW_H_IN   = 1.9
    ZOOM_H_IN       = 1.55
    CLUSTER_GAP_S   = 60.0
    CLUSTER_MARGIN  = 15.0
    CLUSTER_PAD_FRAC= 0.20
    MAX_ZOOM_PANELS = 12

    BG           = '#1A1A2E'
    SPIKE_COLOR  = '#E84040'
    BASE_COLOR   = '#505060'
    BAND_COLOR   = '#E84040'
    TEXT_COLOR   = '#CCCCCC'
    GRID_COLOR   = '#2A2A4A'
    SPINE_COLOR  = '#444466'

    _NICE = [10, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600]

    def _fmt_time(t, _pos=None):
        t = max(0, int(round(t)))
        h, rem = divmod(t, 3600)
        m, s   = divmod(rem, 60)
        return f'{h}:{m:02d}:{s:02d}' if h else f'{m:02d}:{s:02d}'

    def _setup_ax(ax, x_lo: float, x_hi: float, title: str) -> None:
        ax.set_facecolor(BG)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(-0.3, 1.5)
        ax.set_yticks([])
        ax.set_title(title, color=TEXT_COLOR, fontsize=8,
                     loc='left', pad=3, fontweight='normal')
        ax.tick_params(axis='x', colors=TEXT_COLOR, labelsize=7, length=3)
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE_COLOR)
        ax.grid(True, which='major', axis='x',
                color=GRID_COLOR, linewidth=0.6, zorder=0)
        # Clock-friendly major tick interval
        span  = max(x_hi - x_lo, 1.0)
        major = min(_NICE, key=lambda s: abs(s - span / 8.0))
        ax.xaxis.set_major_locator(mticker.MultipleLocator(major))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_time))

    def _draw_content(ax, click_ts: list, x_lo: float, x_hi: float) -> None:
        ax.axhline(0, color=BASE_COLOR, linewidth=0.8, zorder=1)
        if click_ts:
            ax.vlines(click_ts, 0, 1,
                      colors=SPIKE_COLOR, linewidth=1.5, zorder=2)
            ax.plot(click_ts, np.ones(len(click_ts)),
                    'o', color=SPIKE_COLOR, markersize=4, zorder=3)

    def _ts(t: float) -> str:
        v = max(0, int(round(t)))
        h, rem = divmod(v, 3600)
        m, s   = divmod(rem, 60)
        return f'{h}:{m:02d}:{s:02d}' if h else f'{m:02d}:{s:02d}'

    # ── Per-file plots ─────────────────────────────────────────────────────────
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    for r in per_file:
        file_id    = r['file']
        duration_s = r['duration_s']
        has_dur    = not np.isnan(duration_s)

        mask     = (df['file'] == file_id) & (df['stage_blocked'].fillna('') == '')
        click_ts = sorted(df.loc[mask, 'timestamp_s'].dropna().values.tolist())
        n_total  = len(click_ts)

        x_end   = (duration_s if has_dur
                   else (max(click_ts) * 1.08 if click_ts else 60.0))
        dur_lbl = _dur_str(duration_s) if has_dur else 'duration unknown'

        clusters    = _cluster_clicks(click_ts, gap_s=CLUSTER_GAP_S)
        n_clusters  = len(clusters)
        n_zoom_rows = min(n_clusters, MAX_ZOOM_PANELS)
        truncated   = n_clusters > MAX_ZOOM_PANELS

        zoom_windows: list[tuple[float, float, int]] = []
        for t0, t1, k in clusters[:n_zoom_rows]:
            span   = max(t1 - t0, 0.0)
            margin = max(CLUSTER_MARGIN, span * CLUSTER_PAD_FRAC)
            zoom_windows.append((
                max(0.0,   t0 - margin),
                min(x_end, t1 + margin),
                k,
            ))

        n_rows  = 1 + n_zoom_rows
        fig_h   = OVERVIEW_H_IN + n_zoom_rows * ZOOM_H_IN
        fig, axes = plt.subplots(
            n_rows, 1,
            figsize=(PLOT_W_IN, fig_h),
            constrained_layout=True,
            squeeze=False,
        )
        axes = [ax for row in axes for ax in row]
        fig.patch.set_facecolor(BG)

        # Row 0: overview
        ov_title = (
            f'{file_id}  ·  {n_total} confirmed click(s)  ·  {dur_lbl}'
            + (f'  ·  {n_clusters} cluster(s)' if n_clusters > 1 else '')
        )
        _setup_ax(axes[0], 0, x_end, ov_title)
        _draw_content(axes[0], click_ts, 0, x_end)
        for x_lo, x_hi, _ in zoom_windows:
            axes[0].axvspan(x_lo, x_hi, alpha=0.15, color=BAND_COLOR,
                            linewidth=0, zorder=0)

        # Zoom rows
        for i, ((t0, t1, k), (x_lo, x_hi, _)) in enumerate(
                zip(clusters[:n_zoom_rows], zoom_windows)):
            suffix = ''
            if truncated and i == n_zoom_rows - 1:
                suffix = f'  (+{n_clusters - MAX_ZOOM_PANELS} more cluster(s) not shown)'
            zoom_title = (
                f'Cluster {i + 1}/{n_clusters}  ·  {_ts(t0)} – {_ts(t1)}'
                f'  ·  {k} click(s)' + suffix
            )
            zoomed_ts = [t for t in click_ts if x_lo <= t <= x_hi]
            _setup_ax(axes[i + 1], x_lo, x_hi, zoom_title)
            _draw_content(axes[i + 1], zoomed_ts, x_lo, x_hi)

        out_path = plots_dir / f'{file_id}_clicks.png'
        fig.savefig(str(out_path), dpi=DPI, facecolor=BG)
        plt.close(fig)

        n_zoom_lbl = f' + {n_zoom_rows} zoom panel(s)' if n_zoom_rows else ''
        print(f'    Saved: {out_path.name}  [overview{n_zoom_lbl}]')


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='PlantLeaf v5 — full dataset analysis (recursive)',
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
        '--rows-csv', type=Path, default=None, dest='rows_csv',
        help='Override output path for the rows CSV '
             '(default: <output-dir>/evaluated_rows.csv)',
    )
    parser.add_argument(
        '--summary-md', type=Path, default=None, dest='summary_md',
        help='Override output path for the summary Markdown '
             '(default: <output-dir>/summary.md)',
    )
    parser.add_argument(
        '--no-plots', action='store_true', dest='no_plots',
        help='Skip generating click-distribution PNG plots',
    )
    args = parser.parse_args()

    rows_csv_path = args.rows_csv or (args.output_dir / 'evaluated_rows.csv')
    summary_path  = args.summary_md or (args.output_dir / 'summary.md')

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

    print('PlantLeaf v5 — Dataset Analysis')
    print(f'  Model      : {args.model.name}  '
          f'(kernel={model.get("kernel","?")}, threshold={model["threshold"]:.3f})')
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

    # ── Index .paudio files ───────────────────────────────────────────────────
    print(f'Indexing .paudio files under {args.paudio_dir} ...')
    paudio_index = build_paudio_index(args.paudio_dir)
    print(f'  Found {len(paudio_index)} .paudio file(s)\n')

    # ── Load CSVs ─────────────────────────────────────────────────────────────
    print(f'Loading {len(csv_paths)} CSV file(s):')
    df = load_csvs(csv_paths)
    print(f'  Total rows: {len(df)}\n')

    # ── Initialise pipeline output columns ────────────────────────────────────
    df['svm_probability'] = np.nan
    df['svm_prediction']  = pd.array([pd.NA] * len(df), dtype='Int64')
    df['stage_blocked']   = ''

    # ── Run pipeline stages ───────────────────────────────────────────────────
    df = apply_stage2(df)
    df = apply_stage3(df, model)
    df = apply_stage4(df)

    # ── Per-file and global stats ─────────────────────────────────────────────
    print('\nComputing per-file statistics ...')
    per_file     = compute_per_file_stats(df, paudio_index)
    global_stats = aggregate_stats(per_file)

    g = global_stats
    dur_str = _dur_str(g['duration_s'])
    print(f'\n{"="*58}')
    print(f'  Global summary  (threshold = {model["threshold"]:.3f})')
    print(f'{"="*58}')
    print(f'  Input candidates       : {g["candidates"]}')
    print(f'  Confirmed clicks       : {g["confirmed"]}')
    print(f'  Blocked — Stage2_R2    : {g["n_s2_r2"]}')
    print(f'  Blocked — Stage2_SPR   : {g["n_s2_spr"]}')
    print(f'  Blocked — Stage3_SVM   : {g["n_s3"]}')
    print(f'  Blocked — Stage4_dedup : {g["n_s4"]}')
    if g['labeled'] > 0:
        print(f'  ──────────────────────────────────────────────────')
        print(f'  Labeled rows           : {g["labeled"]}')
        print(f'  Confusion  TP={g["tp"]}  FP={g["fp"]}  FN={g["fn"]}  TN={g["tn"]}')
        print(f'  Recall     : {_fmt(g["recall"])}  ({g["tp"]}/{g["tp"]+g["fn"]})')
        print(f'  Precision  : {_fmt(g["precision"])}')
        print(f'  Specificity: {_fmt(g["specificity"])}')
        print(f'  F1         : {_fmt(g["f1"])}')
    if not np.isnan(g['duration_s']):
        print(f'  ──────────────────────────────────────────────────')
        print(f'  Total duration         : {dur_str}')
        print(f'  Click rate (algo)      : {_fmt(g["algo_rate_hr"], 2)} clicks / h')
        print(f'  Click rate (labels)    : {_fmt(g["label_rate_hr"], 2)} clicks / h')
    if g['n_paudio_missing']:
        print(f'  Warning: {g["n_paudio_missing"]} file(s) had no matching .paudio '
              f'(excluded from duration / rate)')

    # ── Save outputs ──────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Click distribution plots
    if not args.no_plots:
        print('\nGenerating click-distribution plots ...')
        generate_click_plots(per_file, df, args.output_dir)

    # Rows CSV — drop internal _source_csv column before saving
    df_out = df.drop(columns=['_source_csv'], errors='ignore')
    df_out.to_csv(rows_csv_path, index=False)
    print(f'\n  Saved rows CSV  : {rows_csv_path}  ({len(df_out)} rows)')

    # Summary Markdown
    write_summary_md(per_file, global_stats, model, summary_path, args.inputs)
    print(f'  Saved summary   : {summary_path}')


if __name__ == '__main__':
    main()
