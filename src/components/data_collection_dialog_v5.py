"""
Data Collection Dialog v5 for PlantLeaf Click Detection

Phase 2: Exports Stage 1 candidates from .paudio files as CSV + screenshots.
- Runs Stage 1 adaptive threshold to find candidates (wide net, k=1.5 default)
- Computes all 17 features for each survivor
- Renders two-panel screenshots (FFT + iFFT + features)
- Exports CSV with schema ready for manual labeling
"""

import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import traceback
from dataclasses import dataclass, asdict, field

import numpy as np
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QSpinBox, QDoubleSpinBox, QLabel, QFileDialog,
    QProgressBar, QPlainTextEdit, QCheckBox, QAbstractItemView,
    QWidget, QSizePolicy, QApplication,
)
from PySide6.QtCore import QThread, Signal, Qt, QPoint, QRect
from PySide6.QtGui import QColor, QPixmap, QPainter, QFont, QImage, QPen

# Import v5 signal processing pipeline (use lazy imports to avoid circular imports)
import importlib.util
import sys


# ── CONFIGURATION ──────────────────────────────────────────────────────────
"""
All tuneable parameters for data collection dialog.
"""

CSV_FILENAME = 'data_collection_export.csv'
SCREENSHOTS_FOLDER = 'screenshots'

# Screenshot rendering (pure QPainter + QImage — no pyqtgraph, no OpenGL, no segfaults)
SCREENSHOT_WIDTH  = 1400
SCREENSHOT_HEIGHT = 800   # taller than 700 to give the feature footer room to breathe
PANEL_HEIGHT      = 480   # plot area height within each panel

# K spinbox constraints
K_MIN = 0.5
K_MAX = 3.0
K_STEP = 0.1
K_DEFAULT = 1.5  # Default Stage 1 multiplier (from click_pipeline_v5.py)


# ── CSV SCHEMA ──────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    'file',
    'frame_idx',
    'timestamp_s',
    'noise_floor_mV',
    'std_noise_mV',
    'E_hat_floor',
    'peak_SNR',
    'pre_SNR',
    'post_SNR',
    'rise_time_ms',
    'fall_time_ms',
    'asymmetry_integral',
    'ZCR_pre',
    'ZCR_click',
    'ZCR_post',
    'kurtosis',
    'centroid_shift_hz',
    'tau_ms',
    'R2',
    'fit_coverage',
    'SPR',
    'R_spectral',
    'FPE_hz',
    'label',  # Empty column for manual labeling (1=click, 0=no click, empty=unknown)
]


# ── CANDIDATE DATA STRUCTURE ────────────────────────────────────────────────

@dataclass
class CandidateData:
    """
    Represents one Stage 1 survivor candidate for export.
    
    Stores metadata, features, and signal data for CSV export + screenshot rendering.
    """
    file: str                           # Filename (.paudio)
    frame_idx: int                      # Frame index
    timestamp_s: float                  # Timestamp in seconds
    
    # Noise estimates (from AdaptiveNoiseEstimatorV5)
    noise_floor: float                  # noise_floor in V
    std_noise: float                    # std_noise in V
    E_hat_floor: float                  # Ê_floor energy threshold in V²
    
    # 17 Features (from compute_features_v5)
    peak_SNR: float
    pre_SNR: float
    post_SNR: float
    rise_time_ms: float
    fall_time_ms: float
    asymmetry_integral: float
    ZCR_pre: float
    ZCR_click: float
    ZCR_post: float
    kurtosis: float
    centroid_shift_hz: float
    tau_ms: float
    R2: float
    fit_coverage: float
    SPR: float
    R_spectral: float
    FPE_hz: float
    
    # Signal data for screenshot rendering
    signal: np.ndarray = field(default_factory=lambda: np.array([]))
    envelope: np.ndarray = field(default_factory=lambda: np.array([]))
    fft_norm: np.ndarray = field(default_factory=lambda: np.array([]))
    freq_axis: np.ndarray = field(default_factory=lambda: np.array([]))

    # Pre-computed exponential fit curve (worker thread) — avoids scipy on main thread
    fit_t_ms: Optional[np.ndarray] = field(default=None)  # time axis [ms] from frame start
    fit_y: Optional[np.ndarray] = field(default=None)     # fit amplitude [V]

    # Label (empty for Phase 2, filled manually later)
    label: str = ''

    def to_csv_dict(self) -> Dict:
        """Convert to dictionary for CSV export (24 columns)."""
        return {
            'file': self.file,
            'frame_idx': self.frame_idx,
            'timestamp_s': round(self.timestamp_s, 6),
            # mV since the iFFT amplitude-scale fix — the reconstructed signal is
            # now in true volts, so these land in the mV range. CSVs exported before
            # that fix carry uV columns 256x smaller; the 17 feature columns are
            # dimensionless ratios and are unaffected, so old and new training sets
            # remain comparable. See docs/fft_and_ifft/IFFT_AMPLITUDE_SCALE_FIX.md
            'noise_floor_mV': round(self.noise_floor * 1e3, 4),
            'std_noise_mV': round(self.std_noise * 1e3, 4),
            'E_hat_floor': round(self.E_hat_floor, 6),
            'peak_SNR': round(self.peak_SNR, 3),
            'pre_SNR': round(self.pre_SNR, 3),
            'post_SNR': round(self.post_SNR, 3),
            'rise_time_ms': round(self.rise_time_ms, 6),
            'fall_time_ms': round(self.fall_time_ms, 6),
            'asymmetry_integral': round(self.asymmetry_integral, 6),
            'ZCR_pre': round(self.ZCR_pre, 3),
            'ZCR_click': round(self.ZCR_click, 3),
            'ZCR_post': round(self.ZCR_post, 3),
            'kurtosis': round(self.kurtosis, 3),
            'centroid_shift_hz': round(self.centroid_shift_hz, 2),
            'tau_ms': round(self.tau_ms, 6),
            'R2': round(self.R2, 4),
            'fit_coverage': round(self.fit_coverage, 4),
            'SPR': round(self.SPR, 3),
            'R_spectral': round(self.R_spectral, 3),
            'FPE_hz': round(self.FPE_hz, 2),
            'label': self.label,
        }


# ── SIGNAL PROCESSING PIPELINE (Stage 2) ────────────────────────────────────

def _get_frame_envelope_pair(
    dm,
    frame_idx: int,
    normalize: bool = True,
) -> Tuple:
    """
    Reconstruct current frame + its neighbours, return envelopes and signals.

    Centralises ALL iFFT/Hilbert work for a candidate so the caller never needs
    to call reconstruct_frame_v5 a second time for the same frame.

    Returns:
        prev_envelope : ndarray or None
        curr_envelope : ndarray
        next_envelope : ndarray or None
        curr_signal   : ndarray
        prev_signal   : ndarray or None
        curr_fft_norm : ndarray  ← mic-corrected FFT magnitudes (full half-spectrum)
        curr_freq_axis: ndarray  ← frequency axis [Hz] for curr_fft_norm
    """
    from core.click_pipeline_v5 import (
        reconstruct_frame_v5,
        compute_hilbert_envelope,
        FS, FFT_SIZE,
    )

    envelopes  = {}
    signals    = {}
    curr_fft_norm  = None
    curr_freq_axis = None

    frames_to_load = [frame_idx]
    if frame_idx > 0:
        frames_to_load.append(frame_idx - 1)
    if frame_idx < len(dm.fft_mags) - 1:
        frames_to_load.append(frame_idx + 1)

    for idx in frames_to_load:
        fd = reconstruct_frame_v5(
            dm.fft_mags[idx], dm.phase_int8[idx],
            FS, FFT_SIZE, normalize=normalize,
        )
        if fd is None:
            continue
        signals[idx]   = fd['signal']
        envelopes[idx] = compute_hilbert_envelope(fd['signal'])
        if idx == frame_idx:          # keep FFT data for the candidate frame
            curr_fft_norm  = fd['fft_norm']
            curr_freq_axis = fd['freq_axis']

    prev_envelope = envelopes.get(frame_idx - 1) if frame_idx > 0 else None
    curr_envelope = envelopes[frame_idx]
    next_envelope = envelopes.get(frame_idx + 1) if frame_idx < len(dm.fft_mags) - 1 else None
    curr_signal   = signals[frame_idx]
    prev_signal   = signals.get(frame_idx - 1) if frame_idx > 0 else None

    return (prev_envelope, curr_envelope, next_envelope,
            curr_signal, prev_signal,
            curr_fft_norm, curr_freq_axis)


def _process_file_for_collection(
    dm,
    k: float = K_DEFAULT,
    normalize: bool = True,
    stop_check=None,
    progress_cb=None,
) -> Tuple[List[CandidateData], List[Dict]]:
    """
    Process single .paudio file: find Stage 1 survivors and compute all features.

    Args (additional):
        stop_check  : Optional callable() → bool. Called before each survivor.
                      Return True to abort early (partial results returned).
        progress_cb : Optional callable(i, total) emitted every 10 survivors so
                      the dialog can show that analysis is progressing.

    Fast path: if dm already carries the pre-computed arrays produced by
    AudioLoadWorker (fft_means, E_hat_floor_arr, noise_floor_arr, std_noise_arr),
    Stage 1 is a pure-numpy threshold + run-length filter — no per-frame iFFT
    or Hilbert transform needed (that was already done during file loading).
    This avoids re-processing every frame a second time, which was the main
    performance bottleneck.

    Slow-path fallback (run_stage1_v5) is used when those arrays are absent.

    Args:
        dm: AudioDataManager with loaded file
        k: Stage 1 multiplier (default 1.5)
        normalize: Use normalized FFT if True

    Returns:
        Tuple of:
        - candidates: List of CandidateData objects (one per Stage 1 survivor)
        - csv_rows: List of CSV-ready dicts (for convenience)
    """
    # Lazy import to avoid circular imports
    from core.click_pipeline_v5 import (
        run_stage1_v5,
        compute_hilbert_envelope,
        find_peak,
        compute_features_v5,
        FS, FFT_SIZE, MAX_RUN,
    )

    candidates = []
    csv_rows   = []

    # ── Stage 1: find above-threshold candidates ──────────────────────────────
    # Fast path: AudioLoadWorker already computed per-frame energy and adaptive
    # noise estimates.  Reuse them — no iFFT/Hilbert needed here.
    _has_precomputed = (
        hasattr(dm, 'fft_means')        and dm.fft_means       is not None and
        hasattr(dm, 'E_hat_floor_arr')  and dm.E_hat_floor_arr is not None and
        hasattr(dm, 'noise_floor_arr')  and dm.noise_floor_arr is not None and
        hasattr(dm, 'std_noise_arr')    and dm.std_noise_arr   is not None and
        len(dm.fft_means) == dm.total_frames
    )

    if _has_precomputed:
        fft_means     = np.asarray(dm.fft_means,       dtype=np.float64)
        E_hat_arr     = np.asarray(dm.E_hat_floor_arr, dtype=np.float64)
        noise_fl_arr  = np.asarray(dm.noise_floor_arr, dtype=np.float64)
        std_noise_arr = np.asarray(dm.std_noise_arr,   dtype=np.float64)

        # Threshold: E_i > k × Ê_floor  (vectorised, no Python loop over frames)
        valid   = E_hat_arr > 0
        flagged = valid & (fft_means > k * E_hat_arr)
        indices = np.where(flagged)[0]

        if len(indices) == 0:
            return candidates, csv_rows

        # Run-length filter: discard runs > MAX_RUN consecutive frames (noise)
        above_threshold = [
            {
                'frame_idx'  : int(i),
                'E_i'        : float(fft_means[i]),
                'E_hat_floor': float(E_hat_arr[i]),
                'noise_floor': float(noise_fl_arr[i]),
                'std_noise'  : float(std_noise_arr[i]),
            }
            for i in indices
        ]

        runs, current = [], [above_threshold[0]]
        for j in range(1, len(above_threshold)):
            if above_threshold[j]['frame_idx'] == above_threshold[j-1]['frame_idx'] + 1:
                current.append(above_threshold[j])
            else:
                runs.append(current)
                current = [above_threshold[j]]
        runs.append(current)

        candidates_raw = []
        for run in runs:
            if len(run) <= MAX_RUN:
                for entry in run:
                    candidates_raw.append({**entry, 'group_size': len(run)})
    else:
        # Slow path: recompute noise estimator from scratch (file was loaded
        # without pre-computed arrays, e.g. older .paudio format).
        candidates_raw = run_stage1_v5(dm, k=k)

    if not candidates_raw:
        return candidates, csv_rows

    n_survivors = len(candidates_raw)

    # Process each survivor
    for i, survivor in enumerate(candidates_raw):
        # Honour cancel request — return partial results already built
        if stop_check is not None and stop_check():
            break

        # Emit periodic progress so the dialog label doesn't appear frozen
        if progress_cb is not None and i % 10 == 0:
            progress_cb(i, n_survivors)

        frame_idx = survivor['frame_idx']

        try:
            # Single iFFT reconstruction pass — _get_frame_envelope_pair now
            # returns curr_fft_norm and curr_freq_axis so we never reconstruct
            # the same frame twice.
            (prev_env, curr_env, next_env,
             curr_sig, prev_sig,
             curr_fft_norm, curr_freq_axis) = _get_frame_envelope_pair(
                dm, frame_idx, normalize=normalize
            )

            # Compute peak index
            peak_idx, peak_amp = find_peak(curr_env)

            # Compute all 17 features (calls run_fit_pipeline_v5 internally)
            features = compute_features_v5(
                signal=curr_sig,
                envelope=curr_env,
                fft_norm=curr_fft_norm,
                freq_axis=curr_freq_axis,
                noise_floor=survivor['noise_floor'],
                std_noise=survivor['std_noise'],
                peak_idx=peak_idx,
                fs=FS,
                next_frame_envelope=next_env,
                prev_frame_envelope=prev_env,
                prev_frame_signal=prev_sig,
            )

            # Calculate timestamp
            timestamp_s = frame_idx / (FS / FFT_SIZE)

            # Pre-compute fit curve for the renderer.
            # Use find_decay_window_v5 directly (already imported inside
            # compute_features_v5) to get the window bounds without calling
            # the full run_fit_pipeline_v5 a second time.
            fit_t_ms_arr = None
            fit_y_arr    = None
            tau_ms_val   = features.get('tau_ms', -1.0)
            R2_val       = features.get('R2', 0.0)
            if tau_ms_val > 0 and R2_val > 0.1:
                try:
                    from core.click_pipeline_v5 import find_decay_window_v5
                    win    = find_decay_window_v5(
                        curr_env, peak_idx,
                        survivor['noise_floor'], survivor['std_noise'],
                        next_env,
                    )
                    d_start = win['decay_start']
                    d_end   = min(win['decay_end'], len(curr_env))
                    if d_end > d_start + 2:
                        n_arr        = np.arange(d_end - d_start, dtype=np.float64)
                        tau_s        = max(tau_ms_val, 0.01) / 1000.0
                        rate         = 1.0 / (tau_s * FS)
                        A0           = float(curr_env[min(d_start, len(curr_env) - 1)])
                        fit_t_ms_arr = np.arange(d_start, d_end, dtype=np.float64) / FS * 1000.0
                        fit_y_arr    = A0 * np.exp(-rate * n_arr)
                except Exception:
                    pass

            ### FILTERS ### (also skips screenshot rendering)

            # Skip candidates with invalid R^2 or tau
            if features.get('R2', 0.00) == 0.00 or features.get('tau_ms', -1.0) <= 0:
                continue

            ### ------- ###

            # Create candidate object
            candidate = CandidateData(
                file=dm.filename,
                frame_idx=frame_idx,
                timestamp_s=timestamp_s,
                noise_floor=survivor['noise_floor'],
                std_noise=survivor['std_noise'],
                E_hat_floor=survivor['E_hat_floor'],
                peak_SNR=features.get('peak_SNR', 0.0),
                pre_SNR=features.get('pre_SNR', 0.0),
                post_SNR=features.get('post_SNR', 0.0),
                rise_time_ms=features.get('rise_time_ms', 0.0),
                fall_time_ms=features.get('fall_time_ms', 0.0),
                asymmetry_integral=features.get('asymmetry_integral', 0.0),
                ZCR_pre=features.get('ZCR_pre', 0.0),
                ZCR_click=features.get('ZCR_click', 0.0),
                ZCR_post=features.get('ZCR_post', 0.0),
                kurtosis=features.get('kurtosis', 0.0),
                centroid_shift_hz=features.get('centroid_shift_hz', 0.0),
                tau_ms=features.get('tau_ms', 0.0),
                R2=features.get('R2', 0.0),
                fit_coverage=features.get('fit_coverage', 0.0),
                SPR=features.get('SPR', 0.0),
                R_spectral=features.get('R_spectral', 0.0),
                FPE_hz=features.get('FPE_hz', 0.0),
                signal=curr_sig,
                envelope=curr_env,
                fft_norm=curr_fft_norm,
                freq_axis=curr_freq_axis,
                fit_t_ms=fit_t_ms_arr,
                fit_y=fit_y_arr,
                label='',
            )
            
            candidates.append(candidate)
            csv_rows.append(candidate.to_csv_dict())
        
        except Exception as e:
            print(f"Warning: Failed to process frame {frame_idx}: {e}")
            traceback.print_exc()
            continue
    
    return candidates, csv_rows


# ── SCREENSHOT RENDERING (Stage 3) ──────────────────────────────────────────
# Pure QPainter + QImage — no pyqtgraph, no OpenGL context, no segfaults.
# All numpy/scipy work is pre-computed in the worker thread; this section
# only does coordinate mapping and QPainter draw calls on the main thread.

# Inner padding (px) reserved inside each panel rect for axes / labels / title
_PL = 62   # left  — y-axis ticks + label
_PR = 8    # right
_PT = 26   # top   — panel title
_PB = 40   # bottom — x-axis ticks + label


def _px_rect(panel: QRect) -> QRect:
    """Return the inner plot area (inside axis padding) for a panel QRect."""
    return QRect(
        panel.left()   + _PL,
        panel.top()    + _PT,
        panel.width()  - _PL - _PR,
        panel.height() - _PT - _PB,
    )


def _draw_axes(
    p: QPainter, panel: QRect,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    title: str, x_label: str, y_label: str,
    c_grid: QColor, c_axis: QColor, c_title: QColor,
    n_x: int = 6, n_y: int = 5,
):
    """Draw grid lines, tick labels, axis labels and panel title."""
    r     = _px_rect(panel)
    x_rng = max(x_max - x_min, 1e-30)
    y_rng = max(y_max - y_min, 1e-30)

    # Grid lines and tick labels
    p.setFont(QFont("Arial", 7))
    for i in range(n_x + 1):
        xv = x_min + i * x_rng / n_x
        px = int(r.left() + i / n_x * r.width())
        p.setPen(QPen(c_grid, 1))
        p.drawLine(px, r.top(), px, r.bottom())
        p.setPen(QPen(c_axis, 1))
        p.drawText(QRect(px - 22, r.bottom() + 3, 44, 15), Qt.AlignCenter, f"{xv:.3g}")

    for i in range(n_y + 1):
        yv = y_min + i * y_rng / n_y
        py = int(r.bottom() - i / n_y * r.height())
        p.setPen(QPen(c_grid, 1))
        p.drawLine(r.left(), py, r.right(), py)
        p.setPen(QPen(c_axis, 1))
        p.drawText(QRect(panel.left() + 2, py - 8, _PL - 5, 16),
                   Qt.AlignRight | Qt.AlignVCenter, f"{yv:.3g}")

    # Axis border
    p.setPen(QPen(c_axis, 1))
    p.drawRect(r)

    # Panel title
    p.setFont(QFont("Arial", 9, QFont.Bold))
    p.setPen(c_title)
    p.drawText(QRect(panel.left(), panel.top() + 3, panel.width(), _PT - 3),
               Qt.AlignCenter, title)

    # X-axis label
    p.setFont(QFont("Arial", 8))
    p.setPen(c_axis)
    p.drawText(QRect(panel.left(), r.bottom() + 21, panel.width(), 16),
               Qt.AlignCenter, x_label)

    # Y-axis label (rotated 90°)
    p.save()
    p.translate(panel.left() + 11, r.top() + r.height() // 2)
    p.rotate(-90)
    p.drawText(QRect(-55, -8, 110, 16), Qt.AlignCenter, y_label)
    p.restore()


def _draw_line(
    p: QPainter, panel: QRect,
    x_arr: np.ndarray, y_arr: np.ndarray,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    color: QColor, lw: int = 1, dashed: bool = False,
):
    """Map x_arr/y_arr from data coords to pixels within panel and draw a polyline.

    Skips any point where x or y is non-finite (inf / nan) — these arise from
    corrupted FFT frames whose energy overflowed. Pixel coordinates are also
    clamped to ±30 000 so Qt's signed-int32 limit is never exceeded.
    """
    if len(x_arr) < 2:
        return
    r   = _px_rect(panel)
    xr  = max(x_max - x_min, 1e-30)
    yr  = max(y_max - y_min, 1e-30)
    rw  = r.width()
    rh  = r.height()
    rl  = r.left()
    rb  = r.bottom()
    # Qt uses signed 32-bit integers for coordinates; clamp well inside that range.
    _MAX_COORD = 30_000
    style = Qt.DashLine if dashed else Qt.SolidLine
    p.setPen(QPen(color, lw, style))

    prev_px: Optional[int] = None
    prev_py: Optional[int] = None

    for i in range(len(x_arr)):
        xv = float(x_arr[i])
        yv = float(y_arr[i])
        if not (np.isfinite(xv) and np.isfinite(yv)):
            # Non-finite point → break the polyline here (don't connect across gap)
            prev_px = prev_py = None
            continue
        cx = int(np.clip(rl + (xv - x_min) / xr * rw, -_MAX_COORD, _MAX_COORD))
        cy = int(np.clip(rb  - (yv - y_min) / yr * rh, -_MAX_COORD, _MAX_COORD))
        if prev_px is not None:
            p.drawLine(prev_px, prev_py, cx, cy)
        prev_px, prev_py = cx, cy


def _draw_hline(
    p: QPainter, panel: QRect,
    y_val: float, y_min: float, y_max: float,
    color: QColor, lw: int = 1,
):
    """Draw a horizontal reference line at y_val across the inner plot area."""
    if not (y_min <= y_val <= y_max):
        return
    r  = _px_rect(panel)
    yr = max(y_max - y_min, 1e-30)
    py = int(r.bottom() - (y_val - y_min) / yr * r.height())
    p.setPen(QPen(color, lw, Qt.DashLine))
    p.drawLine(r.left(), py, r.right(), py)


def _draw_feature_footer(
    p: QPainter, c: 'CandidateData', rect: QRect,
    c_text: QColor, c_key: QColor,
):
    """
    Draw all 17 features + noise info as readable multi-line text below the plots.

    Each group is one line: a bold label on the left, then values separated by │.
    Font is large enough (9 pt) to read without squinting.
    """
    LINE_H = 22

    # (label, [value strings]) — one line per group
    tau_str = f"{c.tau_ms:.4f} ms" if c.tau_ms > 0 else "N/A"
    groups = [
        ("Noise",    [f"floor = {c.noise_floor * 1e3:.4f} mV",
                      f"std = {c.std_noise * 1e3:.4f} mV",
                      f"Ê_floor = {c.E_hat_floor:.3e}"]),
        ("SNR",      [f"peak = {c.peak_SNR:.2f}",
                      f"pre = {c.pre_SNR:.2f}",
                      f"post = {c.post_SNR:.2f}"]),
        ("Shape",    [f"rise = {c.rise_time_ms:.4f} ms",
                      f"fall = {c.fall_time_ms:.4f} ms",
                      f"asymmetry = {c.asymmetry_integral:.4f}",
                      f"kurtosis = {c.kurtosis:.2f}"]),
        ("ZCR",      [f"pre = {c.ZCR_pre:.3f}",
                      f"click = {c.ZCR_click:.3f}",
                      f"post = {c.ZCR_post:.3f}"]),
        ("Spectral", [f"centroid_shift = {c.centroid_shift_hz:.0f} Hz",
                      f"SPR = {c.SPR:.2f}",
                      f"R_spectral = {c.R_spectral:.3f}",
                      f"FPE = {c.FPE_hz:.0f} Hz"]),
        ("Fit",      [f"τ = {tau_str}",
                      f"R² = {c.R2:.4f}",
                      f"coverage = {c.fit_coverage:.3f}"]),
    ]

    f_key = QFont("Arial", 9, QFont.Bold)
    f_val = QFont("Courier New", 9)
    KEY_W = 72  # pixels reserved for group label column

    y = rect.top() + 6
    for label, values in groups:
        if y + LINE_H > rect.bottom():
            break
        p.setFont(f_key)
        p.setPen(c_key)
        p.drawText(QRect(rect.left(), y, KEY_W, LINE_H),
                   Qt.AlignLeft | Qt.AlignVCenter, label + ":")
        p.setFont(f_val)
        p.setPen(c_text)
        p.drawText(QRect(rect.left() + KEY_W, y, rect.width() - KEY_W, LINE_H),
                   Qt.AlignLeft | Qt.AlignVCenter,
                   "   │   ".join(values))
        y += LINE_H


def _render_candidate_screenshot(
    candidate: CandidateData,
    output_path: Path,
) -> bool:
    """
    Render two-panel screenshot using QPainter + QImage.

    Completely avoids pyqtgraph and OpenGL — the previous pyqtgraph approach
    caused segfaults on macOS because container.render() cannot capture
    GL-backed PlotWidget content, and QApplication.processEvents() inside a
    queued signal handler creates dangerous re-entrancy.

    Layout (1400 × 800 px):
      ┌──────────────────────── title bar (36 px) ──────────────────────────┐
      │  Frame XXXXXX  │  t = XX.XXXX s  │  filename                       │
      ├──── FFT panel (40 %) ──────┬─── iFFT panel (60 %) ─────────────────┤
      │  Frequency (kHz)           │  Time (ms)                            │
      │  Amplitude (mV / µV / V)   │  Amplitude (mV / µV / V)             │
      │  linear scale              │  signal (teal) + envelope (red)       │
      │                            │  fit (green dashed, if R²>0.1)        │
      │                            │  noise floor (cyan) + ±std (purple)   │
      ├──────────────── feature footer (6 lines) ───────────────────────────┤
      │  Noise:    floor = … mV  │  std = … mV  │  Ê_floor = …            │
      │  SNR:      peak = …  │  pre = …  │  post = …                       │
      │  Shape:    rise = … ms  │  fall = … ms  │  asymmetry = …           │
      │  ZCR:      pre = …  │  click = …  │  post = …                      │
      │  Spectral: centroid_shift = … Hz  │  SPR = …  │  FPE = … Hz        │
      │  Fit:      τ = … ms  │  R² = …  │  coverage = …                   │
      └─────────────────────────────────────────────────────────────────────┘

    Args:
        candidate: CandidateData with signal, envelope, fft data, pre-computed fit
        output_path: Path to save PNG file

    Returns:
        True if saved successfully, False otherwise
    """
    try:
        from core.click_pipeline_v5 import FS

        # ── Layout constants ──────────────────────────────────────────────────
        W, H       = SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT
        MARGIN     = 14
        HEADER_H   = 36
        PLOT_H     = PANEL_HEIGHT       # height of each plot panel
        GAP        = 8                  # gap between header/panels/footer
        FOOTER_Y   = HEADER_H + MARGIN + PLOT_H + GAP
        FOOTER_H   = H - FOOTER_Y - MARGIN

        FFT_W  = int((W - 3 * MARGIN) * 0.40)
        IFFT_W = W - 3 * MARGIN - FFT_W

        FFT_panel   = QRect(MARGIN,            HEADER_H + MARGIN, FFT_W,  PLOT_H)
        IFFT_panel  = QRect(2*MARGIN + FFT_W,  HEADER_H + MARGIN, IFFT_W, PLOT_H)
        FOOTER_rect = QRect(MARGIN, FOOTER_Y, W - 2*MARGIN, FOOTER_H)

        # ── Colour palette ────────────────────────────────────────────────────
        C_BG     = QColor('#1e1e2e')
        C_PANEL  = QColor('#12121f')
        C_BORDER = QColor('#3a3a55')
        C_TITLE  = QColor('#e8e8e8')
        C_AXIS   = QColor('#8888aa')
        C_GRID   = QColor(36, 36, 56)
        C_FFT    = QColor('#4ea6dc')   # FFT trace — steel blue
        C_SIG    = QColor('#80cbc4')   # raw signal — teal
        C_ENV    = QColor('#ef5350')   # envelope — red
        C_FIT    = QColor('#00E676')   # fit curve — green
        C_FLOOR  = QColor('#00CED1')   # noise floor line — cyan
        C_STD    = QColor('#9370DB')   # noise+std line — purple
        C_TEXT   = QColor('#cccccc')
        C_KEY    = QColor('#7aadcc')

        # ── Create image ──────────────────────────────────────────────────────
        img = QImage(W, H, QImage.Format_RGB32)
        img.fill(C_BG)
        p = QPainter(img)
        # try/finally guarantees p.end() even if any draw call raises — without
        # this Qt prints "Cannot destroy paint device that is being painted".
        try:
            p.setRenderHint(QPainter.Antialiasing, False)
            p.setRenderHint(QPainter.TextAntialiasing, True)

            # ── Header ───────────────────────────────────────────────────────
            p.setFont(QFont("Arial", 11, QFont.Bold))
            p.setPen(C_TITLE)
            hdr = (f"Frame {candidate.frame_idx:06d}   │   "
                   f"t = {candidate.timestamp_s:.4f} s   │   {candidate.file}")
            p.drawText(QRect(0, 4, W, HEADER_H - 4), Qt.AlignCenter, hdr)

            # ── Panel backgrounds + borders ───────────────────────────────────
            for panel in (FFT_panel, IFFT_panel):
                p.fillRect(panel, C_PANEL)
                p.setPen(QPen(C_BORDER, 1))
                p.drawRect(panel)

            # ── FFT Panel ─────────────────────────────────────────────────────
            freq_khz = candidate.freq_axis / 1000.0
            mask     = (freq_khz >= 20) & (freq_khz <= 80)
            fq       = freq_khz[mask]
            fft_vals = candidate.fft_norm[mask].copy()

            if len(fft_vals) == 0:
                fft_vals = np.array([0.0])
                fq       = np.array([20.0])

            # Replace any non-finite values before auto-scaling
            fft_vals = np.where(np.isfinite(fft_vals), fft_vals, 0.0)

            # Auto-scale to most readable SI unit (linear, not dB)
            fft_max = float(np.max(np.abs(fft_vals))) if len(fft_vals) > 0 else 0.0
            if fft_max >= 0.5:
                fft_sc, fft_unit = fft_vals, 'V'
            elif fft_max >= 5e-4:
                fft_sc, fft_unit = fft_vals * 1e3, 'mV'
            else:
                fft_sc, fft_unit = fft_vals * 1e6, 'µV'

            fft_y_min = 0.0
            fft_y_max = float(np.max(fft_sc)) * 1.08 if len(fft_sc) > 0 else 1.0
            if not np.isfinite(fft_y_max) or fft_y_max <= 0:
                fft_y_max = 1.0

            _draw_axes(p, FFT_panel,
                       float(fq[0]), float(fq[-1]), fft_y_min, fft_y_max,
                       title="FFT Spectrum",
                       x_label="Frequency (kHz)", y_label=f"Amplitude ({fft_unit})",
                       c_grid=C_GRID, c_axis=C_AXIS, c_title=C_TITLE)
            _draw_line(p, FFT_panel, fq, fft_sc,
                       float(fq[0]), float(fq[-1]), fft_y_min, fft_y_max,
                       C_FFT, lw=1)

            # ── iFFT Panel ────────────────────────────────────────────────────
            n_samp  = len(candidate.signal)
            time_ms = np.arange(n_samp, dtype=np.float64) / FS * 1000.0

            sig_raw = np.where(np.isfinite(candidate.signal),   candidate.signal,   0.0)
            env_raw = np.where(np.isfinite(candidate.envelope),  candidate.envelope, 0.0)

            # Auto-scale: pick unit from peak envelope amplitude
            env_max = float(np.max(np.abs(env_raw))) if n_samp > 0 else 1e-10
            if env_max >= 0.5:
                i_sc, i_unit = 1.0, 'V'
            elif env_max >= 5e-4:
                i_sc, i_unit = 1e3, 'mV'
            else:
                i_sc, i_unit = 1e6, 'µV'

            sig_sc  = sig_raw * i_sc
            env_sc  = env_raw * i_sc
            floor_v = candidate.noise_floor * i_sc
            std_v   = (candidate.noise_floor + candidate.std_noise) * i_sc

            all_y   = np.concatenate([sig_sc, env_sc])
            i_y_min = float(np.min(all_y)) * 1.05
            i_y_max = float(np.max(all_y)) * 1.10
            if not (np.isfinite(i_y_min) and np.isfinite(i_y_max)) or i_y_max <= i_y_min:
                i_y_min, i_y_max = -1.0, 1.0
            # Always keep noise lines within view
            if np.isfinite(floor_v) and floor_v > 0:
                i_y_min = min(i_y_min, floor_v * 0.9)

            t0, t1 = float(time_ms[0]), float(time_ms[-1])

            _draw_axes(p, IFFT_panel,
                       t0, t1, i_y_min, i_y_max,
                       title="iFFT Signal + Envelope",
                       x_label="Time (ms)", y_label=f"Amplitude ({i_unit})",
                       c_grid=C_GRID, c_axis=C_AXIS, c_title=C_TITLE)
            _draw_line(p, IFFT_panel, time_ms, sig_sc,  t0, t1, i_y_min, i_y_max, C_SIG, lw=1)
            _draw_line(p, IFFT_panel, time_ms, env_sc,  t0, t1, i_y_min, i_y_max, C_ENV, lw=2)
            _draw_hline(p, IFFT_panel, floor_v, i_y_min, i_y_max, C_FLOOR, lw=1)
            _draw_hline(p, IFFT_panel, std_v,   i_y_min, i_y_max, C_STD,   lw=1)

            # Pre-computed fit curve (no scipy on main thread)
            if candidate.fit_t_ms is not None and len(candidate.fit_t_ms) > 1:
                fit_sc = candidate.fit_y * i_sc
                _draw_line(p, IFFT_panel, candidate.fit_t_ms, fit_sc,
                           t0, t1, i_y_min, i_y_max, C_FIT, lw=2, dashed=True)

            # ── Feature footer ────────────────────────────────────────────────
            _draw_feature_footer(p, candidate, FOOTER_rect, C_TEXT, C_KEY)

        finally:
            p.end()   # always release the painter, even on exception

        output_path.parent.mkdir(parents=True, exist_ok=True)
        pixmap = QPixmap.fromImage(img)
        return pixmap.save(str(output_path), 'PNG')

    except Exception as e:
        print(f"Error rendering screenshot: {e}")
        traceback.print_exc()
        return False


# ── CSV EXPORT ──────────────────────────────────────────────────────────────

def _write_csv_for_file(
    candidates: List[CandidateData],
    output_path: Path,
) -> bool:
    """
    Write CSV file with all candidates for a single audio file.
    
    Args:
        candidates: List of CandidateData objects
        output_path: Path to write CSV
    
    Returns:
        True if successful, False otherwise
    """
    try:
        import csv
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            
            for candidate in candidates:
                csv_dict = candidate.to_csv_dict()
                writer.writerow(csv_dict)
        
        return True
    
    except Exception as e:
        print(f"Error writing CSV: {e}")
        traceback.print_exc()
        return False


# ── PLACEHOLDER STUBS (to be filled in next stages) ───────────────────────

class DataCollectionWorkerV5(QThread):
    """
    Non-blocking worker thread for batch data collection processing.
    
    Processes a list of .paudio files:
    1. Loads each file into AudioDataManager
    2. Runs Stage 1 to find candidates
    3. Computes all 17 features for each candidate
    4. Renders screenshots
    5. Accumulates CSV rows
    
    Emits signals to main thread for progress updates and results.
    """
    
    # Signals
    progress_updated = Signal(int, int, int, int)  # (file_idx, total_files, candidates_found, total_candidates)
    load_progress    = Signal(int, int, int)        # (file_idx, total_files, pct_0_100) — fired during file loading
    candidate_ready  = Signal(object)              # (CandidateData, str) — rendered on main thread
    file_complete    = Signal(str, int, str)       # (filename, candidate_count, csv_path)
    error_occurred   = Signal(str)
    finished         = Signal(int, str)            # (total_candidates, output_dir)
    
    def __init__(
        self,
        file_list: List[Path],
        k: float = K_DEFAULT,
        output_dir: Path = None,
        normalize_mode: bool = True,
    ):
        """
        Initialize worker thread.
        
        Args:
            file_list: List of .paudio file paths to process
            k: Stage 1 multiplier (default 1.5)
            output_dir: Directory to save CSVs and screenshots
            normalize_mode: Use normalized FFT if True (default True)
        """
        super().__init__()
        self.file_list = file_list
        self.k = k
        self.output_dir = Path(output_dir) if output_dir else Path.home() / 'plantleaf_data_collection'
        self.normalize_mode = normalize_mode
        self._stop_requested = False
        # Holds the active AudioLoadWorker so request_stop() can cancel it mid-load.
        self._current_audio_worker = None
    
    def run(self):
        """Main worker loop — pure numpy/IO work only. No Qt widgets created here."""
        total_candidates = 0

        try:
            # Use wakepy to prevent sleep during long processing (especially on macOS)
            from core.wake_lock_manager import WakeLockManager
            waker = WakeLockManager()
            waker.acquire()


            self.output_dir.mkdir(parents=True, exist_ok=True)
            screenshots_dir = self.output_dir / SCREENSHOTS_FOLDER
            screenshots_dir.mkdir(parents=True, exist_ok=True)

            for file_idx, paudio_file in enumerate(self.file_list):
                if self._stop_requested:
                    self.error_occurred.emit("Data collection cancelled by user")
                    return

                try:
                    self.error_occurred.emit(f"Loading {paudio_file.name}...")
                    dm = self._load_audio_file(paudio_file, file_idx, len(self.file_list))
                    if dm is None:
                        self.error_occurred.emit(f"Failed to load {paudio_file.name}, skipping")
                        continue

                    # Loading done — tell the user we're now in the analysis phase.
                    # Without this, the label stays frozen at the last "Loading: X%"
                    # value for the entire (potentially long) feature-extraction phase.
                    self.error_occurred.emit(
                        f"Analysing {paudio_file.name} ({dm.total_frames} frames)…"
                    )

                    candidates, csv_rows = _process_file_for_collection(
                        dm, k=self.k, normalize=self.normalize_mode,
                        stop_check=lambda: self._stop_requested,
                        progress_cb=lambda done, total: self.error_occurred.emit(
                            f"  → {paudio_file.name}: {done}/{total} survivors processed…"
                        ),
                    )
                    
                    ## CSV ##
                    # Write CSV immediately — pure file I/O, safe in background thread.
                    # Does NOT depend on screenshots, so we don't need to wait for rendering.
                    csv_filename = f"{paudio_file.stem}_candidates.csv"
                    csv_path = self.output_dir / csv_filename
                    _write_csv_for_file(candidates, csv_path)
                    self.file_complete.emit(paudio_file.name, len(candidates), str(csv_path))

                    ## SCREENSHOTS ##
                    # Emit each candidate to the main thread for screenshot rendering.
                    # Qt.QueuedConnection (set in dialog) guarantees _on_candidate_ready
                    # runs on the main thread — mandatory on macOS for any Qt widget.
                    for candidate in candidates:
                        if self._stop_requested:
                            return

                        screenshot_filename = f"{paudio_file.stem}_{candidate.frame_idx:06d}.png"
                        screenshot_path = screenshots_dir / screenshot_filename

                        # Hand off to main thread — never create widgets here
                        self.candidate_ready.emit((candidate, str(screenshot_path)))

                        total_candidates += 1
                        self.progress_updated.emit(
                            file_idx, len(self.file_list), len(candidates), total_candidates
                        )

                except Exception as e:
                    self.error_occurred.emit(f"Error processing {paudio_file.name}: {str(e)}")
                    traceback.print_exc()
                    continue

            self.finished.emit(total_candidates, str(self.output_dir))

            waker.release() # ensure wake lock is released when done

        except Exception as e:
            self.error_occurred.emit(f"Critical error in worker: {str(e)}")
            traceback.print_exc()
    
    def _load_audio_file(self, paudio_path: Path, file_idx: int = 0, total_files: int = 1):
        """
        Load a .paudio file into AudioDataManager.

        Uses AudioLoadWorker.run() synchronously (no extra QThread needed —
        we are already inside DataCollectionWorkerV5.run()).
        Mirrors file_handler_mixin._on_finished for consistent field population.

        Forwards AudioLoadWorker's internal progress (0-100 %) through the
        load_progress signal so the dialog can animate its progress bar during
        the loading phase — otherwise the UI appears frozen for large files.

        Args:
            paudio_path : Path to .paudio file
            file_idx    : Zero-based index of this file in the batch (for progress scaling)
            total_files : Total number of files in the batch (for progress scaling)

        Returns:
            AudioDataManager if successful, None otherwise
        """
        try:
            from saving.audio_load_progress import AudioLoadWorker
            from windows.replay_window_audio import AudioDataManager

            # Capture worker output via signal lambdas
            result = {'data': None, 'error': None}

            worker = AudioLoadWorker(str(paudio_path))
            worker.finished.connect(lambda d: result.update({'data': d}))
            worker.error.connect(lambda msg: result.update({'error': msg}))

            # Forward AudioLoadWorker's internal progress to the dialog.
            # worker.progress emits 0-100 within this file; load_progress
            # carries (file_idx, total_files, pct) so the dialog can scale it
            # correctly across the full batch.
            worker.progress.connect(
                lambda pct: self.load_progress.emit(file_idx, total_files, pct)
            )

            # Register so request_stop() can cancel this worker immediately.
            self._current_audio_worker = worker

            # Run synchronously — blocks until file is fully loaded.
            # Signals are emitted synchronously (same-thread DirectConnection).
            worker.run()

            # Deregister — the worker has finished (or was cancelled).
            self._current_audio_worker = None

            if self._stop_requested:
                return None   # cancelled during load — skip this file

            if result['error']:
                print(f"Load error for {paudio_path.name}: {result['error']}")
                return None
            if result['data'] is None:
                print(f"No data returned for {paudio_path.name}")
                return None

            data = result['data']

            # Populate AudioDataManager (mirrors file_handler_mixin._on_finished)
            dm = AudioDataManager()
            dm.header_info          = data['header_info']
            dm.fft_data             = data['fft_data']
            dm.phase_data           = data.get('phase_data', [])
            dm.frequency_axis       = np.array(data['frequency_axis'])
            dm.total_frames         = data['total_frames']
            dm.frame_duration_ms    = data['frame_duration_ms']
            dm.total_duration_sec   = data['total_duration_sec']
            dm.click_events         = data['click_events']
            dm.overview_x           = np.array(data['overview_x'])
            dm.overview_y           = np.array(data['overview_y'])
            dm.overview_loaded      = True
            dm.streaming_x          = np.array(data['streaming_x'])
            dm.streaming_y          = np.array(data['streaming_y'])
            dm.streaming_start_time = data['streaming_start_time']
            dm.streaming_end_time   = data['streaming_end_time']
            dm.filename             = paudio_path.stem

            # Aliases expected by click_pipeline_v5 (run_stage1_v5,
            # reconstruct_frame_v5) which use fft_mags/phase_int8 naming.
            dm.fft_mags   = dm.fft_data    # same list, pipeline-compatible alias
            dm.phase_int8 = dm.phase_data  # same list, pipeline-compatible alias

            # Use pre-computed noise arrays from worker if available;
            # otherwise compute them now (required by run_stage1_v5).
            fft_means = data.get('fft_means')
            if fft_means is not None and len(fft_means) > 0:
                dm.fft_means       = data['fft_means']
                dm.fft_timestamps  = data['fft_timestamps']
                dm.E_hat_floor_arr = data['E_hat_floor_arr']
                dm.noise_floor_arr = data['noise_floor_arr']
                dm.std_noise_arr   = data['std_noise_arr']
            else:
                dm.precompute_fft_means()

            return dm

        except Exception as e:
            print(f"Failed to load audio file: {e}")
            traceback.print_exc()
            return None
        

    def request_stop(self):
        """
        Request the worker to stop gracefully.

        Also cancels any AudioLoadWorker that is currently running synchronously
        inside _load_audio_file, so the cancel takes effect immediately instead
        of waiting for the full file-load loop to complete.
        """
        self._stop_requested = True
        if self._current_audio_worker is not None:
            self._current_audio_worker.cancel_load()  # sets AudioLoadWorker._cancelled


class DataCollectionDialogV5(QDialog):
    """
    Dialog for batch data collection from multiple .paudio files.
    
    Provides UI for:
    - Selecting .paudio files to process
    - Configuring Stage 1 multiplier (k)
    - Selecting output folder
    - Running batch export (with progress feedback)
    - Viewing results
    """
    
    def __init__(self, parent=None):
        """Initialize data collection dialog."""
        super().__init__(parent)
        self.setWindowTitle("Data Collection — Stage 1 Batch Export (v5)")
        self.setGeometry(100, 100, 900, 700)
        
        self.file_list = []
        self.output_dir = Path.home()
        self.worker = None
        self.is_processing = False
        
        self._setup_ui()
        self._setup_connections()

        # Put dialog in the middle of the screen
        screen_geometry = self.screen().geometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 4
        self.move(x, y)

        # Correct theme for light mode
        if self.parent() and hasattr(self.parent(), 'theme_manager'):
            try:
                theme = self.parent().theme_manager.load_saved_theme()
                self.parent().theme_manager.apply_theme(self, theme)
                if 'light' in theme.lower():  # known bug fix for light themes
                    self.setStyleSheet("QDialog { background-color: white; }")
                    # Make the loading bar darker for visibility in light mode (gray instead of white)
                    self.progress_bar.setStyleSheet("""
                        QProgressBar {
                            border: 1px solid #bbb;
                            border-radius: 5px;
                            background-color: #eee;
                        }
                        QProgressBar::chunk {
                            background-color: #1f77b4;
                        }
                    """)

            except Exception:
                pass

        # Fai in modo che con ESC si chiuda la finestra
        self.setWindowFlag(Qt.WindowCloseButtonHint, True)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, False)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, False)
        self.setModal(True)

    def _setup_ui(self):
        """Build the dialog UI layout."""
        layout = QVBoxLayout(self)
        
        # ── File List Section ──
        layout.addWidget(self._create_file_list_section())
        
        # ── Parameters Section ──
        layout.addWidget(self._create_parameters_section())
        
        # ── Output Folder Section ──
        layout.addWidget(self._create_output_section())
        
        # ── Progress Section ──
        layout.addWidget(self._create_progress_section())
        
        # ── Button Section ──
        layout.addWidget(self._create_button_section())
        
        # ── Log Area ──
        layout.addWidget(self._create_log_section())
        
        self.setLayout(layout)
    
    def _create_file_list_section(self):
        """Create file selection section (add/remove/clear buttons + list)."""
        from PySide6.QtWidgets import QGroupBox
        
        group = QGroupBox("Audio Files (.paudio)")
        layout = QVBoxLayout()
        
        # File list widget
        self.file_list_widget = QListWidget()
        self.file_list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.file_list_widget.setMaximumHeight(120)
        layout.addWidget(self.file_list_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.btn_add = QPushButton("Add Files")
        self.btn_add.clicked.connect(self._on_add_files)
        button_layout.addWidget(self.btn_add)
        
        self.btn_remove = QPushButton("Remove Selected")
        self.btn_remove.clicked.connect(self._on_remove_file)
        button_layout.addWidget(self.btn_remove)
        
        self.btn_clear = QPushButton("Clear All")
        self.btn_clear.clicked.connect(self._on_clear_files)
        button_layout.addWidget(self.btn_clear)
        
        layout.addLayout(button_layout)
        group.setLayout(layout)
        return group
    
    def _create_parameters_section(self):
        """Create parameters section (k multiplier, normalize checkbox)."""
        from PySide6.QtWidgets import QGroupBox
        
        group = QGroupBox("Parameters")
        layout = QHBoxLayout()
        
        # K multiplier
        layout.addWidget(QLabel(" k (Stage 1 multiplier):"))
        self.spinbox_k = QDoubleSpinBox()
        self.spinbox_k.setMinimum(K_MIN)
        self.spinbox_k.setMaximum(K_MAX)
        self.spinbox_k.setSingleStep(K_STEP)
        self.spinbox_k.setValue(K_DEFAULT)
        self.spinbox_k.setDecimals(1)
        self.spinbox_k.setToolTip("Higher k = stricter threshold (fewer candidates)")
        layout.addWidget(self.spinbox_k)
        
        layout.addSpacing(30)
        
        # Note: data is always normalized (replace checkbox)
        note_lbl = QLabel("Note: data is always normalized (not configurable)")
        note_lbl.setStyleSheet("QLabel { color: gray; font-style: italic; }")
        layout.addWidget(note_lbl)

        layout.addStretch()
        group.setLayout(layout)

        return group
    
    def _create_output_section(self):
        """Create output folder selection section."""
        from PySide6.QtWidgets import QGroupBox
        
        group = QGroupBox("Output Folder")
        layout = QHBoxLayout()
        
        layout.addWidget(QLabel(" Folder:"))
        self.label_output = QLabel(str(self.output_dir))
        self.label_output.setStyleSheet("QLabel { padding: 5px; }")
        layout.addWidget(self.label_output)
        
        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self._on_browse_output)
        layout.addWidget(self.btn_browse)
        
        group.setLayout(layout)
        return group
    
    def _create_progress_section(self):
        """Create progress bar section."""
        from PySide6.QtWidgets import QGroupBox
        
        group = QGroupBox("Progress")
        layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        self.label_progress = QLabel(" Ready")
        layout.addWidget(self.label_progress)
        
        group.setLayout(layout)
        return group
    
    def _create_log_section(self):
        """Create log output section."""
        from PySide6.QtWidgets import QGroupBox
        
        group = QGroupBox("Log")
        layout = QVBoxLayout()
        
        self.log_area = QPlainTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(150)
        layout.addWidget(self.log_area)
        
        group.setLayout(layout)
        return group
    
    def _create_button_section(self):
        """Create action buttons (Export, Cancel, Close)."""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.addStretch()
        
        self.btn_export = QPushButton("Export")
        self.btn_export.setMinimumWidth(100)
        self.btn_export.clicked.connect(self._on_export)
        layout.addWidget(self.btn_export)
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setMinimumWidth(100)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel)
        layout.addWidget(self.btn_cancel)
        
        self.btn_close = QPushButton("Close")
        self.btn_close.setMinimumWidth(100)
        self.btn_close.clicked.connect(self.close)
        layout.addWidget(self.btn_close)
        
        container.setLayout(layout)
        return container
    
    def _setup_connections(self):
        """Connect signals/slots (most connected in _create_* methods)."""
        pass
    
    # ── Event Handlers ──
    
    def _on_add_files(self):
        """Browse and add .paudio files."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select .paudio files",
            str(Path.home()),
            "Audio Files (*.paudio);;All Files (*)",
        )
        
        for filepath in files:
            path = Path(filepath)
            if path not in [Path(self.file_list_widget.item(i).text()) for i in range(self.file_list_widget.count())]:
                self.file_list_widget.addItem(str(path))
                self.file_list.append(path)
        
        self._update_export_button()
    
    def _on_remove_file(self):
        """Remove selected file from list."""
        row = self.file_list_widget.currentRow()
        if row >= 0:
            item = self.file_list_widget.takeItem(row)
            self.file_list = [Path(self.file_list_widget.item(i).text()) for i in range(self.file_list_widget.count())]
        
        self._update_export_button()
    
    def _on_clear_files(self):
        """Clear all files from list."""
        self.file_list_widget.clear()
        self.file_list = []
        self._update_export_button()
    
    def _on_browse_output(self):
        """Browse for output directory."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
            str(self.output_dir),
        )
        
        if folder:
            self.output_dir = Path(folder)
            self.label_output.setText(str(self.output_dir))
    
    def _on_export(self):
        """Start data collection worker."""
        if not self.file_list:
            self._log("No files selected!")
            return
        
        # Update file list from widget
        self.file_list = [Path(self.file_list_widget.item(i).text()) for i in range(self.file_list_widget.count())]
        
        # Disable/enable buttons
        self.btn_export.setEnabled(False)
        self.btn_add.setEnabled(False)
        self.btn_remove.setEnabled(False)
        self.btn_clear.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.is_processing = True
        
        # Clear log
        self.log_area.clear()
        self._log(f"Starting data collection for {len(self.file_list)} file(s)...")
        self._log(f"  k = {self.spinbox_k.value()}")
        self._log(f"  Normalize: True (fixed)")
        self._log(f"  Output: {self.output_dir}")
        self._log("")
        
        # Create and start worker
        self.worker = DataCollectionWorkerV5(
            file_list=self.file_list,
            k=self.spinbox_k.value(),
            output_dir=self.output_dir,
            normalize_mode=True,
        )
        
        # QueuedConnection → slot always executes on the main thread.
        # This is mandatory on macOS: Qt widgets / NSWindow must be created
        # on the main thread. The worker emits the signal from its thread;
        # Qt queues it and delivers it here on the GUI thread.
        self.worker.candidate_ready.connect(
            self._on_candidate_ready, Qt.QueuedConnection
        )
        self.worker.progress_updated.connect(
            self._on_progress_updated, Qt.QueuedConnection
        )
        self.worker.load_progress.connect(
            self._on_load_progress, Qt.QueuedConnection
        )
        self.worker.file_complete.connect(
            self._on_file_complete, Qt.QueuedConnection
        )
        self.worker.error_occurred.connect(
            self._on_error, Qt.QueuedConnection
        )
        self.worker.finished.connect(
            self._on_worker_finished, Qt.QueuedConnection
        )

        self.worker.start()
    
    def _on_candidate_ready(self, args):
        """
        Render screenshot on the main thread.

        Called via Qt.QueuedConnection from DataCollectionWorkerV5 — this
        method ALWAYS runs on the main (GUI) thread, satisfying the macOS
        requirement that NSWindow/QWidget are only instantiated on the main thread.
        """
        candidate, screenshot_path = args
        _render_candidate_screenshot(candidate, Path(screenshot_path))

    def _on_cancel(self):
        """
        Cancel worker thread without blocking the main thread.

        NEVER call self.worker.wait() here — it would block the main thread
        while DataCollectionWorkerV5 is stuck inside AudioLoadWorker.run(),
        which cannot respect _stop_requested until the current file finishes
        loading.  Instead we:
          1. Signal both workers to stop (request_stop also cancels the inner
             AudioLoadWorker via _current_audio_worker.cancel_load())
          2. Disconnect all signals so no stale callbacks reach the UI
          3. Reset the UI immediately
        The background QThread finishes on its own; Python GC cleans it up.
        """
        if self.worker:
            self._log("Cancelling...")
            self.worker.request_stop()
            # Disconnect all signals so in-flight queued events are harmless.
            try:
                self.worker.candidate_ready.disconnect()
                self.worker.progress_updated.disconnect()
                self.worker.load_progress.disconnect()
                self.worker.file_complete.disconnect()
                self.worker.error_occurred.disconnect()
                self.worker.finished.disconnect()
            except RuntimeError:
                pass
            self.worker = None   # allow GC; thread finishes in background
            self._log("Cancelled.")
        self._reset_ui()
    
    def _on_progress_updated(self, file_idx: int, total_files: int, candidates_found: int, total_candidates: int):
        """Update progress bar and label during candidate processing phase."""
        # Each file occupies an equal share of 0-100 %. Within that share,
        # the first half was used for loading (via _on_load_progress) and the
        # second half for candidate processing.  Here we move to the midpoint
        # so the bar never jumps backwards.
        if total_files > 0:
            file_share = 100.0 / total_files
            pct = int(file_idx * file_share + file_share * 0.5)
        else:
            pct = 0
        self.progress_bar.setValue(pct)
        self.label_progress.setText(
            f"Processing file {file_idx + 1}/{total_files}  |  Candidates so far: {total_candidates}"
        )

    def _on_load_progress(self, file_idx: int, total_files: int, pct_in_file: int):
        """
        Update progress bar during file loading phase.

        AudioLoadWorker reports 0-100 % internally as it reads and processes
        the .paudio file. We scale that into each file's share of the total
        progress bar (first half of that share, 0-50 %), so the bar moves
        smoothly during what would otherwise be a frozen UI.
        """
        if total_files > 0:
            file_share = 100.0 / total_files
            # Map pct_in_file (0-100) to the first half of this file's share
            pct = int(file_idx * file_share + (pct_in_file / 100.0) * file_share * 0.5)
        else:
            pct = pct_in_file // 2
       # print(f"Load progress for file {file_idx + 1}/{total_files}: {pct_in_file}% → overall {pct}%")
        self.progress_bar.setValue(pct)
        self.label_progress.setText(
            f"Loading file {file_idx + 1}/{total_files}: {pct_in_file}%"
        )

    def _on_screenshot_saved(self, filepath: str):
        """Log screenshot saved (optional, keep concise)."""
        pass  # Too verbose to log every screenshot
    
    def _on_file_complete(self, filename: str, candidate_count: int, csv_path: str):
        """Log file completion."""
        self._log(f"✓ {filename}: {candidate_count} candidates → {Path(csv_path).name}")
    
    def _on_error(self, error_msg: str):
        """Log error message."""
        self._log(f"⚠ {error_msg}")
    
    def _on_worker_finished(self, total_candidates: int, output_dir: str):
        """Worker finished — reset UI and show results."""
        self._log("")
        self._log(f"✅ Collection complete: {total_candidates} total candidates exported")
        self._log(f"📁 Output: {output_dir}")
        self.progress_bar.setValue(100)
        
        self._reset_ui()
    
    def _reset_ui(self):
        """Reset UI after export completes or cancels."""
        self.btn_export.setEnabled(True)
        self.btn_add.setEnabled(True)
        self.btn_remove.setEnabled(True)
        self.btn_clear.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.is_processing = False
        self._update_export_button()
    
    def _update_export_button(self):
        """Enable/disable Export button based on file list."""
        self.btn_export.setEnabled(self.file_list_widget.count() > 0 and not self.is_processing)
    
    def _log(self, message: str):
        """Append message to log area."""
        self.log_area.appendPlainText(message)
        # Auto-scroll to bottom
        self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())
