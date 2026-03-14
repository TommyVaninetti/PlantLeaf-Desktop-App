"""
Batch Export Screenshots — Pure Qt/pyqtgraph implementation
============================================================
Exports one PNG screenshot for every frame whose mean FFT energy exceeds a
user-chosen threshold (set in the ThresholdConfirmDialog).
The algorithm / click-detector output is NOT involved: this is a pure
energy-threshold scan over all frames of the loaded file.

Architecture:
  - BatchExportWorker(QThread): scans fft_means, reconstructs iFFT,
      computes envelope + decay fit for every above-threshold frame.
      Emits frame_data_ready(dict) → main thread renders.
  - render_click_screenshot(): main-thread only; builds off-screen QWidget,
      calls grab(), saves PNG, destroys widget synchronously.
  - Launched from ReplayWindowAudio "Analysis" menu via launch_batch_export()

Screenshot layout (1400 × 760 px):
  ┌──────────────── title bar ─────────────────────────────────────────────┐
  │  Frame XXXXXX  |  t = XX.XXX s  (absolute start time of frame)        │
  │  ┌────── FFT panel (40%) ───────┐  ┌────── iFFT panel (60%) ──────────┐│
  │  │ raw FFT (accent color)       │  │ raw iFFT (#80cbc4)               ││
  │  │ norm FFT (#ef5350)           │  │ norm iFFT (white dashed)         ││
  │  │                              │  │ norm envelope (#ef5350)          ││
  │  │                              │  │ fit exp (green/orange/red dashed)││
  │  │                              │  │ peak line (#ffd600 dotted)       ││
  │  │                              │  │ skip region (grey α=0.15)        ││
  │  └──────────────────────────────┘  └──────────────────────────────────┘│
  │  ─────────────────── footer (peak, SNR, τ, R², …) ────────────────────│
  └────────────────────────────────────────────────────────────────────────┘

Output: all screenshots saved flat inside the chosen export folder.
"""

import os
import numpy as np

from PySide6.QtCore import QThread, Signal, Qt, QTimer, QPoint
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QDoubleSpinBox, QFileDialog,
    QProgressBar, QGroupBox, QApplication, QWidget,
    QSizePolicy, QMessageBox
)
from PySide6.QtGui import QFont, QColor, QPixmap, QPainter

import pyqtgraph as pg

# ── Re-use existing signal-processing helpers ─────────────────────────────────
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from windows.replay_window_audio import (
    compute_hilbert_envelope, find_peak, check_decay,
    estimate_noise_offline, suppress_edge_artifacts
)
from core.replay_base_window import compute_fft_energy


# ═══════════════════════════════════════════════════════════════════════════════
# WORKER THREAD  — all numpy/scipy work happens here, zero Qt widgets
# ═══════════════════════════════════════════════════════════════════════════════
class BatchExportWorker(QThread):
    """
    Background thread: scans ALL frames whose fft_means exceed threshold_v,
    reconstructs iFFT, computes envelope + decay fit, emits one signal per frame.
    The click-detector algorithm is NOT involved.
    """

    frame_data_ready = Signal(dict)
    progress_updated = Signal(int, str)   # (percent 0-100, message)
    finished_export  = Signal(int)        # total frames exported
    error_occurred   = Signal(str)

    def __init__(self, data_manager, threshold_v,
                 use_normalization, parent=None):
        super().__init__(parent)
        self.data_manager      = data_manager
        self.threshold_v       = threshold_v
        self.use_normalization = use_normalization
        self._cancelled        = False

    def cancel(self):
        self._cancelled = True

    # ── helpers ──────────────────────────────────────────────────────────────
    def _normalize_fft(self, fft_magnitudes):
        datasheet_freq_khz    = np.array([20, 25, 30, 40, 50, 60, 70, 80])
        datasheet_response_db = np.array([8.0, 10.5, 6.0, -2.0, -6.0, -7.0, -6.0, -4.0])
        freq_axis  = self.data_manager.frequency_axis
        valid_mask = (freq_axis >= 20000) & (freq_axis <= 80000)
        freq_range = freq_axis[valid_mask]
        mic_response_db    = np.interp(freq_range, datasheet_freq_khz * 1000, datasheet_response_db)
        correction_gain_50 = 10 ** (-mic_response_db * 0.5 / 20.0)
        normalized = fft_magnitudes.copy()
        normalized[valid_mask] *= correction_gain_50
        return normalized

    def _reconstruct_ifft(self, frame_index, normalized=False):
        dm = self.data_manager
        if frame_index >= len(dm.fft_data):
            return None
        # Always copy raw data first so actual_bins is stable regardless of normalization
        fft_mags_raw = dm.fft_data[frame_index].copy()
        if len(dm.phase_data) == 0:
            return None
        fft_phases_int8 = dm.phase_data[frame_index]
        fs            = dm.header_info.get('fs', 200000)
        fft_size      = dm.header_info.get('fft_size', 512)
        num_bins_full = fft_size // 2
        bin_freq      = fs / fft_size
        bin_start     = int(20000 / bin_freq)
        bin_end       = int(80000 / bin_freq)
        # Compute actual_bins from RAW data before any normalization
        actual_bins   = min(len(fft_mags_raw), bin_end - bin_start + 1, len(fft_phases_int8))
        # Apply normalization AFTER locking in actual_bins
        fft_mags = self._normalize_fft(fft_mags_raw) if normalized else fft_mags_raw
        full_mag      = np.zeros(num_bins_full, dtype=np.float32)
        full_phase    = np.zeros(num_bins_full, dtype=np.int8)
        full_mag  [bin_start:bin_start + actual_bins] = fft_mags[:actual_bins]
        full_phase[bin_start:bin_start + actual_bins] = fft_phases_int8[:actual_bins]
        phases_rad       = (full_phase / 127.0) * np.pi
        complex_spectrum = full_mag * np.exp(1j * phases_rad)
        # Tukey taper — same bins as raw, so taper is identical for both paths
        taper  = max(5, actual_bins // 10)
        window = np.ones(num_bins_full)
        for i in range(taper):
            a = i / taper
            window[bin_start + i]                   = 0.5 * (1 - np.cos(np.pi * a))
            window[bin_start + actual_bins - i - 1] = 0.5 * (1 - np.cos(np.pi * a))
        complex_spectrum *= window
        try:
            sig = np.fft.irfft(complex_spectrum, n=fft_size)
            return suppress_edge_artifacts(sig)
        except Exception:
            return None

    # ── main entry ───────────────────────────────────────────────────────────
    def run(self):
        try:
            self._run_inner()
        except Exception as e:
            self.error_occurred.emit(str(e))

    # ── Criterion helpers (mirrors click_detector_dialog Stage 3 logic) ────────
    @staticmethod
    def _compute_pre_snr(signal, peak_idx, noise_rms):
        """
        Compute pre_snr = RMS(signal before peak) / noise_rms.
        Three cases mirroring click_detector_dialog Criterion 2.
        """
        GUARD = 20
        pre_end = max(0, peak_idx - GUARD)
        if pre_end >= 50:
            pre_window = signal[:pre_end]
        else:
            # Not enough samples: conservative fallback
            pre_window = np.array([noise_rms])
        rms_pre = float(np.sqrt(np.mean(pre_window ** 2))) if len(pre_window) > 0 else noise_rms
        return rms_pre / noise_rms if noise_rms > 0 else 1.0

    @staticmethod
    def _compute_c4(envelope, peak_idx, noise_rms):
        """
        Criterion 4: narrow-spike test.
        Returns (passes: bool, asymmetry_ratio: float).
        """
        ASYM_THRESHOLD    = 0.5
        LEVEL_FRACTION    = 0.10
        FALL_SEARCH       = 40
        SPIKE_HALF_WIN    = 40
        SPIKE_NOISE_FACTOR= 3.0
        peak_amp = envelope[peak_idx]
        level    = peak_amp * LEVEL_FRACTION
        rise_start = peak_idx
        for i in range(peak_idx - 1, -1, -1):
            if envelope[i] < level:
                rise_start = i + 1
                break
        rise_samples = max(1, peak_idx - rise_start)
        fall_end_idx = min(peak_idx + FALL_SEARCH, len(envelope))
        fall_samples = FALL_SEARCH
        fall_cross_idx = peak_idx + FALL_SEARCH
        for i in range(peak_idx + 1, fall_end_idx):
            if envelope[i] < level:
                fall_samples = i - peak_idx
                fall_cross_idx = i
                break
        asym_ratio  = rise_samples / fall_samples if fall_samples > 0 else 1.0
        is_symmetric = (asym_ratio >= ASYM_THRESHOLD)
        left_flank  = envelope[max(0, rise_start - SPIKE_HALF_WIN):rise_start]
        right_flank = envelope[fall_cross_idx:min(fall_cross_idx + SPIKE_HALF_WIN, len(envelope))]
        l_near = (len(left_flank) == 0 or float(np.max(left_flank)) < SPIKE_NOISE_FACTOR * noise_rms)
        r_near = (len(right_flank) == 0 or float(np.max(right_flank)) < SPIKE_NOISE_FACTOR * noise_rms)
        return (not (is_symmetric and l_near and r_near)), asym_ratio

    @staticmethod
    def _compute_c5(envelope, peak_idx):
        """
        Criterion 5: clean tail (no secondary burst > 3× valley).
        Returns passes: bool.
        """
        DECAY_SKIP    = 10
        TAIL_WINDOW   = 300
        REBOUND_FACTOR = 3.0
        LEVEL_FRACTION = 0.10
        peak_amp  = envelope[peak_idx]
        level     = peak_amp * LEVEL_FRACTION
        tail_start = peak_idx + DECAY_SKIP
        tail_end   = min(peak_idx + TAIL_WINDOW, len(envelope))
        if tail_end <= tail_start + 5:
            return True
        tail_env  = envelope[tail_start:tail_end]
        valley_val = float(np.min(tail_env))
        valley_idx = int(np.argmin(tail_env))
        post_valley = tail_env[valley_idx + 1:]
        rebound_thr = valley_val * REBOUND_FACTOR
        has_rebound = (len(post_valley) > 0 and
                       float(np.max(post_valley)) > rebound_thr and
                       rebound_thr > level)
        return not has_rebound

    def _run_inner(self):
        dm            = self.data_manager
        total_frames  = dm.total_frames
        noise_rms     = getattr(dm, '_cached_noise_rms', None) or 1e-6
        has_noise     = getattr(dm, '_cached_noise_rms', None) is not None
        frame_dur_ms  = dm.frame_duration_ms   # ms per frame (e.g. 2.564 ms)

        # ── Pass 1: collect all frame indices above threshold ────────────────
        above_threshold = []
        for i in range(total_frames):
            if self._cancelled:
                self.finished_export.emit(0)
                return
            if i % 5000 == 0:
                pct = int((i / total_frames) * 20)
                self.progress_updated.emit(pct, f"Scanning frames… {i}/{total_frames}")
            if dm.fft_means[i] > self.threshold_v:
                above_threshold.append(i)

        # ── Filter: discard runs of consecutive frames longer than MAX_RUN ────
        # Preserve group structure so folder naming can use first-frame timestamp.
        # Returns a list of (run_list) where each run_list is a list of frame indices.
        MAX_RUN = 5
        filtered_groups = []   # list of lists
        if above_threshold:
            run_start = 0
            for k in range(1, len(above_threshold) + 1):
                at_end  = (k == len(above_threshold))
                new_run = at_end or (above_threshold[k] - above_threshold[k - 1] > 1)
                if new_run:
                    run = above_threshold[run_start:k]
                    run_len = len(run)
                    if run_len <= MAX_RUN:
                        filtered_groups.append(run)
                    else:
                        self.progress_updated.emit(
                            20,
                            f"Discarded run of {run_len} consecutive frames "
                            f"(frame {run[0]}–{run[-1]})"
                        )
                    run_start = k

        # Flatten for total count; keep group metadata per frame
        frame_group_meta = {}   # frame_idx → (group_size, group_first_ts)
        for grp in filtered_groups:
            first_ts = grp[0] * frame_dur_ms / 1000.0
            for fi in grp:
                frame_group_meta[fi] = (len(grp), first_ts)
        above_threshold_flat = [fi for grp in filtered_groups for fi in grp]

        n_total = len(above_threshold_flat)
        if n_total == 0:
            self.progress_updated.emit(100, "No isolated frames above threshold (all runs were too long).")
            self.finished_export.emit(0)
            return

        self.progress_updated.emit(20, f"Found {n_total} isolated frames above threshold. Processing…")

        # ── Pass 2: process each above-threshold frame ───────────────────────
        n_exported = 0
        for seq_idx, frame_idx in enumerate(above_threshold_flat):
            if self._cancelled:
                break

            pct = 20 + int((seq_idx / n_total) * 80)
            self.progress_updated.emit(
                pct, f"Processing frame {seq_idx + 1}/{n_total}  (frame #{frame_idx})"
            )

            # Absolute start time of this frame in the recording
            timestamp_s = frame_idx * frame_dur_ms / 1000.0

            # Group metadata (for subdirectory naming)
            group_size, group_first_ts = frame_group_meta.get(frame_idx, (1, timestamp_s))

            # Reconstruct iFFT
            sig_raw  = self._reconstruct_ifft(frame_idx, normalized=False)
            sig_norm = self._reconstruct_ifft(frame_idx, normalized=True)
            if sig_raw is None or sig_norm is None:
                continue

            # FFT data
            fft_raw   = dm.fft_data[frame_idx].copy()
            fft_norm  = self._normalize_fft(fft_raw)
            freq_axis = dm.frequency_axis

            # Envelopes
            env_raw  = compute_hilbert_envelope(sig_raw)
            env_norm = compute_hilbert_envelope(sig_norm)
            peak_idx, peak_amp = find_peak(env_raw)

            # Decay fit
            fs = dm.header_info.get('fs', 200000)
            next_sig = None
            if peak_idx > 212 and frame_idx + 1 < total_frames:
                next_sig = self._reconstruct_ifft(frame_idx + 1, normalized=False)
            decay = check_decay(env_raw, peak_idx,
                                next_frame_signal=next_sig,
                                noise_rms=noise_rms)

            snr    = peak_amp / noise_rms if noise_rms > 0 else 0.0
            E_W1   = decay['E_W1']
            E_W4   = decay['E_W4']
            slope  = decay['slope_log']
            r2     = decay['r_squared_log']
            tau_ms = decay['tau_ms']
            n_fit  = decay['n_fit_samples']
            fit_skip_total = decay['fit_skip']

            # Fit curve points (for overlay)
            fit_xs = None; fit_ys = None
            if slope < 0 and n_fit >= 5:
                fit_start = peak_idx + fit_skip_total
                fit_end   = min(fit_start + n_fit, len(sig_raw))
                if fit_start < len(env_raw):
                    A0    = float(env_raw[fit_start])
                    n_arr = np.arange(fit_end - fit_start, dtype=float)
                    fit_ys = A0 * np.exp(slope * n_arr)
                    fit_xs = np.arange(fit_start, fit_end) / fs  # seconds

            # Time axis (local within frame, 0 … fft_size/fs seconds)
            time_axis = np.arange(len(sig_raw)) / fs

            # τ annotation
            if tau_ms > 0:
                if tau_ms < 0.15:
                    tau_note = "⚡ short"
                elif tau_ms < 0.5:
                    tau_note = "📊 typical"
                else:
                    tau_note = "🔊 prolonged"
            else:
                tau_note = "— poor fit"

            # ── Extended parameters (only meaningful when noise_rms is real) ─
            # SPR — spectral peak ratio (broadband check)
            fft_power  = fft_norm ** 2
            mean_power = float(np.mean(fft_power))
            max_power  = float(np.max(fft_power))
            spr        = max_power / mean_power if mean_power > 1e-20 else 0.0

            # pre_snr — silence before click (Criterion 2)
            pre_snr = self._compute_pre_snr(sig_raw, peak_idx, noise_rms)

            # C1–C5 (only meaningful when _cached_noise_rms is set)
            c1_pass = snr > 5.0
            c2_pass = pre_snr < 3.0
            c3_pass = E_W1 > E_W4 * 2.0
            c4_pass, asym_ratio = self._compute_c4(env_raw, peak_idx, noise_rms)
            c5_pass = self._compute_c5(env_raw, peak_idx)

            self.frame_data_ready.emit({
                'frame_idx':       frame_idx,
                'seq_idx':         seq_idx,          # position in above-threshold list
                'n_total':         n_total,
                'timestamp':       timestamp_s,      # absolute time in recording (s)
                # Group info (for folder structure)
                'group_size':      group_size,
                'group_first_ts':  group_first_ts,
                # FFT
                'freq_axis':       freq_axis,
                'fft_raw':         fft_raw,
                'fft_norm':        fft_norm,
                # iFFT
                'time_axis':       time_axis,
                'sig_raw':         sig_raw,
                'sig_norm':        sig_norm,
                'env_norm':        env_norm,
                # Fit
                'fit_xs':          fit_xs,
                'fit_ys':          fit_ys,
                'fit_skip':        fit_skip_total,
                'r2':              r2,
                'tau_ms':          tau_ms,
                'tau_note':        tau_note,
                'slope':           slope,
                'n_fit':           n_fit,
                'peak_idx':        peak_idx,
                'peak_amp':        peak_amp,
                # Basic metrics
                'snr':             snr,
                'E_W1':            E_W1,
                'E_W4':            E_W4,
                'noise_rms':       noise_rms,
                # Extended metrics (always computed; show as N/A if !has_noise)
                'spr':             spr,
                'pre_snr':         pre_snr,
                'asym_ratio':      asym_ratio,
                'c1_pass':         c1_pass,
                'c2_pass':         c2_pass,
                'c3_pass':         c3_pass,
                'c4_pass':         c4_pass,
                'c5_pass':         c5_pass,
                'has_noise_ref':   has_noise,   # True only when detector was run
                # Decay details
                'fit_truncated':   decay.get('fit_truncated', False),
                'near_end':        decay.get('near_end', False),
            })
            n_exported += 1

        self.progress_updated.emit(100, "Done.")
        self.finished_export.emit(n_exported)


# ═══════════════════════════════════════════════════════════════════════════════
# OFF-SCREEN RENDERING HELPERS  (main thread only)
# ═══════════════════════════════════════════════════════════════════════════════

def _tau_color(r2: float) -> str:
    if r2 >= 0.70:
        return '#00E676'   # green
    elif r2 >= 0.50:
        return '#FFA726'   # orange
    return '#EF5350'       # red


def _auto_scale(values_v):
    """
    Return (scaled_array, unit_label) choosing the best SI prefix.
    Chooses based on the max absolute value in the array.
    """
    max_abs = float(np.max(np.abs(values_v))) if len(values_v) > 0 else 0.0
    if max_abs == 0:
        return values_v, 'V'
    if max_abs >= 0.5:
        return values_v, 'V'
    elif max_abs >= 5e-4:
        return values_v * 1000.0, 'mV'
    else:
        return values_v * 1e6, 'µV'


def render_click_screenshot(frame_data: dict, out_path: str, accent_color: str):
    """
    Builds an off-screen QWidget (1400×760), renders it, saves PNG.
    Must be called from the main Qt thread.
    """
    W, H = 1400, 800
    PANEL_H = 520   # plot area height

    confirmed = frame_data.get('confirmed', True)   # kept for footer only, not used for color
    frame_idx = frame_data['frame_idx']
    ts        = frame_data['timestamp']   # absolute start time of frame in recording (s)
    seq_idx   = frame_data.get('seq_idx', 0)
    n_total   = frame_data.get('n_total', 1)

    # ── Root container ────────────────────────────────────────────────────────
    # Use Qt.Tool + FramelessWindowHint so the widget is never shown as a
    # normal window in the taskbar / Mission Control on macOS.
    # We move it far off-screen AND keep it at 0×0 size until layout is
    # applied, preventing any visible flash.
    container = QWidget(
        None,
        Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
    )
    container.setFixedSize(W, H)
    container.move(-W - 200, -H - 200)   # well off-screen in all directions
    container.setStyleSheet("background-color: #1e1e2e;")
    container.setAttribute(Qt.WA_DeleteOnClose, False)  # we destroy() manually

    root = QVBoxLayout(container)
    root.setContentsMargins(10, 8, 10, 8)
    root.setSpacing(4)

    # ── Title bar ─────────────────────────────────────────────────────────────
    title_text = (
        f"Frame {frame_idx:06d}  [{seq_idx + 1} / {n_total}]"
        f"  │  t = {ts:.4f} s"
    )
    title_lbl = QLabel(title_text)
    title_lbl.setAlignment(Qt.AlignCenter)
    title_lbl.setStyleSheet(
        "color: #e0e0e0; font-size: 14pt; font-weight: bold; "
        "background: transparent;"
    )
    root.addWidget(title_lbl)

    # ── Two-panel row ─────────────────────────────────────────────────────────
    panels = QHBoxLayout()
    panels.setSpacing(8)

    # ── FFT panel (40%) ───────────────────────────────────────────────────────
    fft_pw = pg.PlotWidget(background='#12121f')
    fft_pw.setFixedHeight(PANEL_H)
    fft_pw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    freq_khz = frame_data['freq_axis'] / 1000.0        # Hz → kHz
    mask = (freq_khz >= 20) & (freq_khz <= 80)
    fq   = freq_khz[mask]

    fft_raw_masked  = frame_data['fft_raw'][mask]
    fft_norm_masked = frame_data['fft_norm'][mask]
    # Auto-scale both curves together so axis label is consistent
    all_fft_vals = np.concatenate([fft_raw_masked, fft_norm_masked])
    fft_scale_factor = 1.0
    max_fft = float(np.max(np.abs(all_fft_vals))) if len(all_fft_vals) > 0 else 0.0
    if max_fft > 0:
        if max_fft < 5e-4:
            fft_scale_factor = 1e6
            fft_unit = 'µV'
        elif max_fft < 0.5:
            fft_scale_factor = 1000.0
            fft_unit = 'mV'
        else:
            fft_scale_factor = 1.0
            fft_unit = 'V'
    else:
        fft_unit = 'V'

    fft_pw.plot(fq, fft_raw_masked * fft_scale_factor,
                pen=pg.mkPen(accent_color, width=1.4),
                name='Raw FFT')
    fft_pw.plot(fq, fft_norm_masked * fft_scale_factor,
                pen=pg.mkPen('#ef5350', width=1.4),
                name='Norm FFT')

    fft_pw.setXRange(20, 80, padding=0.01)
    fft_pw.enableAutoRange('y')
    fft_pw.getAxis('bottom').setLabel('Frequency (kHz)')
    fft_pw.getAxis('left').setLabel(f'Magnitude ({fft_unit})')
    _style_plot_widget(fft_pw)

    panels.addWidget(fft_pw, stretch=40)

    # ── iFFT panel (60%) ──────────────────────────────────────────────────────
    ifft_pw = pg.PlotWidget(background='#12121f')
    ifft_pw.setFixedHeight(PANEL_H)
    ifft_pw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    t_ax   = frame_data['time_axis'] * 1000.0          # s → ms
    sig_r  = frame_data['sig_raw']
    sig_n  = frame_data['sig_norm']
    env_n  = frame_data['env_norm']

    # Auto-scale: choose unit based on peak envelope amplitude
    all_ifft_vals = np.concatenate([sig_r, sig_n, env_n])
    _, ifft_unit = _auto_scale(all_ifft_vals)
    if ifft_unit == 'mV':
        ifft_scale = 1000.0
    elif ifft_unit == 'µV':
        ifft_scale = 1e6
    else:
        ifft_scale = 1.0

    sig_r_sc = sig_r * ifft_scale
    sig_n_sc = sig_n * ifft_scale
    env_n_sc = env_n * ifft_scale

    # raw iFFT — teal, thin
    ifft_pw.plot(t_ax, sig_r_sc,
                 pen=pg.mkPen('#80cbc4', width=0.8, alpha=int(0.85 * 255)),
                 name='Raw iFFT')
    # normalised iFFT — white dashed
    ifft_pw.plot(t_ax, sig_n_sc,
                 pen=pg.mkPen('#ffffff', width=0.7, style=Qt.DashLine,
                              alpha=int(0.5 * 255)),
                 name='Norm iFFT')
    # normalised envelope — red solid, prominent
    ifft_pw.plot(t_ax, env_n_sc,
                 pen=pg.mkPen('#ef5350', width=2.0),
                 name='Norm Envelope')

    # Fit exponential curve
    fit_xs = frame_data['fit_xs']
    fit_ys = frame_data['fit_ys']
    r2     = frame_data['r2']
    tau_ms = frame_data['tau_ms']
    if fit_xs is not None and fit_ys is not None:
        fit_color = _tau_color(r2)
        fit_label = (f"Fit  τ={tau_ms:.3f} ms  R²={r2:.3f}"
                     if tau_ms > 0 else f"Fit  R²={r2:.3f}")
        ifft_pw.plot(fit_xs * 1000.0, fit_ys * ifft_scale,
                     pen=pg.mkPen(fit_color, width=2.2, style=Qt.DashLine),
                     name=fit_label)

    # Peak vertical line
    peak_idx = frame_data['peak_idx']
    fs_val   = 200000
    peak_t_ms = peak_idx / fs_val * 1000.0
    ifft_pw.addLine(x=peak_t_ms,
                    pen=pg.mkPen('#ffd600', width=1.2, style=Qt.DotLine))

    # Skip region (grey, semi-transparent)
    fit_skip = frame_data['fit_skip']
    if fit_skip > 0:
        skip_end_ms = (peak_idx + fit_skip) / fs_val * 1000.0
        skip_region = pg.LinearRegionItem(
            values=[peak_t_ms, skip_end_ms],
            orientation='vertical',
            brush=pg.mkBrush(180, 180, 180, 38),
            movable=False
        )
        skip_region.setZValue(-10)
        ifft_pw.addItem(skip_region)

    ifft_pw.enableAutoRange('xy')
    ifft_pw.getAxis('bottom').setLabel('Time (ms)')
    ifft_pw.getAxis('left').setLabel(f'Amplitude ({ifft_unit})')
    _style_plot_widget(ifft_pw)

    panels.addWidget(ifft_pw, stretch=60)
    root.addLayout(panels)

    # ── Footer (metrics + pass/fail) ─────────────────────────────────────────
    footer_lbl = QLabel(_build_footer(frame_data))
    footer_lbl.setAlignment(Qt.AlignCenter)
    footer_lbl.setWordWrap(False)
    footer_lbl.setTextFormat(Qt.RichText)
    footer_lbl.setStyleSheet(
        "color: #cccccc; font-size: 9.5pt; font-family: monospace; "
        "background: transparent; padding: 4px 0;"
    )
    root.addWidget(footer_lbl)

    # ── Render ────────────────────────────────────────────────────────────────
    # Show the widget (off-screen) so Qt creates native handles and runs
    # the layout engine, but process only posted/paint events — this ensures
    # pyqtgraph PlotWidgets draw their content without the window ever
    # appearing visibly on the user's display.
    container.show()
    QApplication.processEvents()

    # Render directly into a QPixmap.
    # render(QPainter, QPoint, ...) requires a QPoint as second positional arg.
    # The QPainter must be ended BEFORE saving or destroying the pixmap.
    pixmap  = QPixmap(W, H)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    container.render(painter, QPoint(0, 0))
    painter.end()   # ← must end before save() or "Cannot destroy paint device" error
    saved = pixmap.save(out_path, "PNG")

    # Destroy synchronously — do NOT use deleteLater().
    # deleteLater() schedules destruction for the next event loop tick; with
    # 100+ widgets queued they pile up, exhausting native window handles and
    # eventually crashing the app.  close() + destroy() releases the macOS
    # NSWindow handle immediately.
    container.close()
    container.destroy()
    return saved


def _style_plot_widget(pw: pg.PlotWidget):
    """Apply dark-theme axis styling to a PlotWidget."""
    AXIS_COLOR = '#aaaaaa'
    for ax in ('bottom', 'left'):
        pw.getAxis(ax).setTextPen(pg.mkPen(AXIS_COLOR))
        pw.getAxis(ax).setPen(pg.mkPen(AXIS_COLOR))
    pw.showGrid(x=True, y=True, alpha=0.15)


def _pass_fail(passed: bool) -> str:
    return '<span style="color:#00E676;">PASS</span>' if passed else '<span style="color:#EF5350;">FAIL</span>'


def _build_footer(d: dict) -> str:
    """Build an HTML two-line footer with signal metrics and optional C1-C5 pass/fail."""
    peak_mv   = d['peak_amp'] * 1000.0
    snr       = d['snr']
    noise_mv  = d.get('noise_rms', 0) * 1000.0
    E_W1      = d['E_W1']
    E_W4      = d['E_W4']
    dec_ratio = E_W1 / E_W4 if E_W4 > 1e-15 else 999.0
    tau_ms    = d['tau_ms']
    tau_note  = d['tau_note']
    r2        = d['r2']
    n_fit     = d['n_fit']
    trunc     = '✅' if d.get('fit_truncated') else '—'
    near_end  = '⚠️ near-end' if d.get('near_end') else ''
    has_noise = d.get('has_noise_ref', False)

    spr        = d.get('spr', 0.0)
    pre_snr    = d.get('pre_snr', 0.0)
    asym_ratio = d.get('asym_ratio', 0.0)
    c1_pass    = d.get('c1_pass', False)
    c2_pass    = d.get('c2_pass', False)
    c3_pass    = d.get('c3_pass', False)
    c4_pass    = d.get('c4_pass', False)
    c5_pass    = d.get('c5_pass', False)

    tau_str = (f"τ = {tau_ms:.3f} ms  {tau_note}" if tau_ms > 0
               else f"τ = N/A  {tau_note}")

    sep = '&nbsp;&nbsp;│&nbsp;&nbsp;'

    # ── Line 1: basic metrics ─────────────────────────────────────────────────
    line1_parts = [
        f"peak = {peak_mv:.3f} mV",
        f"noise = {noise_mv:.4f} mV",
        f"SNR = {snr:.1f}",
        f"E_W1/E_W4 = {dec_ratio:.1f}",
        tau_str,
        f"R² = {r2:.4f}",
        f"n_fit = {n_fit}",
        f"trunc = {trunc}",
    ]
    if near_end:
        line1_parts.append(near_end)
    line1 = sep.join(line1_parts)

    # ── Line 2: extended metrics (SPR, pre_snr, C1-C5) ───────────────────────
    if has_noise:
        c1_str = _pass_fail(c1_pass)
        c2_str = _pass_fail(c2_pass)
        c3_str = _pass_fail(c3_pass)
        c4_str = _pass_fail(c4_pass)
        c5_str = _pass_fail(c5_pass)
        line2_parts = [
            f"SPR = {spr:.1f}",
            f"pre_snr = {pre_snr:.2f}",
            f"asym = {asym_ratio:.2f}",
            f"C1(SNR>{snr:.0f}) {c1_str}",
            f"C2(pre_snr<3) {c2_str}",
            f"C3(decay) {c3_str}",
            f"C4(spike) {c4_str}",
            f"C5(tail) {c5_str}",
        ]
        line2 = sep.join(line2_parts)
    else:
        line2 = ('<span style="color:#888888; font-style:italic;">'
                 'C1–C5 / SPR / pre_snr: N/A — run Automatic Click Detector first'
                 '</span>')

    return f"{line1}<br/>{line2}"


# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLD CONFIRMATION MINI-DIALOG
# ═══════════════════════════════════════════════════════════════════════════════
class ThresholdConfirmDialog(QDialog):
    """
    Mini-dialog shown before batch export starts.
    Lets the user confirm / adjust the energy threshold and other key params.
    """
    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Export — Confirm Parameters")
        self.setMinimumWidth(480)
        self.setModal(True)

        dm = data_manager
        mean_mv = getattr(dm, 'fft_mean', 0) * 1000
        std_mv  = getattr(dm, 'fft_std',  0) * 1000
        default_thr = mean_mv + 4 * std_mv

        layout = QVBoxLayout(self)

        info_group = QGroupBox("File Statistics")
        info_layout = QFormLayout()
        info_layout.addRow("Mean energy (μ):",   QLabel(f"{mean_mv:.3f} mV"))
        info_layout.addRow("Std deviation (σ):", QLabel(f"{std_mv:.3f} mV"))
        info_layout.addRow("Total frames:",      QLabel(str(dm.total_frames)))
        info_layout.addRow("Duration:",          QLabel(f"{dm.total_duration_sec:.1f} s"))
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        params_group = QGroupBox("Export Parameters")
        params_layout = QFormLayout()

        # Energy threshold — the ONLY parameter: export every frame above this
        thr_row = QHBoxLayout()
        self.thr_spin = QDoubleSpinBox()
        self.thr_spin.setDecimals(3)
        self.thr_spin.setRange(0, 10000)
        self.thr_spin.setSuffix(" mV")
        self.thr_spin.setSingleStep(std_mv if std_mv > 0 else 0.1)
        self.thr_spin.setValue(default_thr)
        thr_row.addWidget(self.thr_spin)
        self.sigma_lbl = QLabel()
        self.sigma_lbl.setStyleSheet("color: gray;")
        thr_row.addWidget(self.sigma_lbl)
        thr_row.addStretch()
        params_layout.addRow("Energy threshold:", thr_row)

        note_lbl = QLabel(
            "Every frame whose mean FFT energy exceeds this threshold\n"
            "will receive a screenshot — no algorithm filtering applied."
        )
        note_lbl.setStyleSheet("color: gray; font-style: italic;")
        params_layout.addRow("", note_lbl)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.ok_btn     = QPushButton("▶  Start Export")
        self.cancel_btn = QPushButton("Cancel")
        self.ok_btn.setMinimumHeight(36)
        self.ok_btn.setDefault(True)
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.ok_btn)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

        # Connect live sigma label update
        self._mean_mv = mean_mv
        self._std_mv  = std_mv
        self.thr_spin.valueChanged.connect(self._update_sigma)
        self._update_sigma()

        # Apply theme if available
        if parent and hasattr(parent, 'theme_manager'):
            theme = parent.theme_manager.load_saved_theme()
            parent.theme_manager.apply_theme(self, theme)
            if 'light' in theme.lower():
                self.setStyleSheet("QDialog { background-color: white; }")

    def _update_sigma(self):
        if self._std_mv > 0:
            mult = (self.thr_spin.value() - self._mean_mv) / self._std_mv
            self.sigma_lbl.setText(f"(μ + {mult:.1f}σ)")
        else:
            self.sigma_lbl.setText("(σ = 0)")

    def get_params(self) -> dict:
        return {
            'threshold_mv': self.thr_spin.value(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PROGRESS DIALOG  (non-blocking — stays open during export)
# ═══════════════════════════════════════════════════════════════════════════════
class BatchExportProgressDialog(QDialog):
    """
    Progress dialog shown during batch export.
    Non-blocking: Cancel button terminates the worker thread.
    """
    cancel_requested = Signal()

    def __init__(self, total: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Export Screenshots")
        self.setMinimumWidth(500)
        self.setModal(False)   # non-blocking

        layout = QVBoxLayout(self)

        self.status_lbl = QLabel("Preparing...")
        self.status_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_lbl)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        layout.addWidget(self.bar)

        self.count_lbl = QLabel("Scanning frames…")
        self.count_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.count_lbl)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

        self._total     = total
        self._processed = 0

        if parent and hasattr(parent, 'theme_manager'):
            theme = parent.theme_manager.load_saved_theme()
            parent.theme_manager.apply_theme(self, theme)

    def _on_cancel(self):
        self.cancel_requested.emit()
        self.cancel_btn.setEnabled(False)
        self.status_lbl.setText("Cancelling...")

    def update_progress(self, pct: int, message: str):
        self.bar.setValue(pct)
        self.status_lbl.setText(message)
        self.count_lbl.setText(message)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT  (called from ReplayWindowAudio)
# ═══════════════════════════════════════════════════════════════════════════════
def launch_batch_export(main_window):
    """
    Full batch-export flow. Called from the main replay window "Analysis" menu.

    Logic:
      - Scans ALL frames of the loaded file
      - Exports a screenshot for every frame whose fft_means > threshold_v
      - No algorithm / click-detector involvement
      - PNGs saved inside  {chosen_dir}/{source_filename}_screenshots/
        · Isolated single frames → directly in the root screenshots folder
        · Consecutive groups (2-5 frames) → in a subdirectory named t=XX.XXXXs
          (timestamp of the first frame of that group)
    """
    dm = main_window.data_manager

    # ── Guard: file must be loaded ────────────────────────────────────────────
    if not hasattr(dm, 'fft_means') or dm.total_frames == 0:
        QMessageBox.warning(
            main_window, "No File Loaded",
            "Please load a .paudio file first."
        )
        return

    # ── Guard: warn if detector has not been run ──────────────────────────────
    has_noise = getattr(dm, '_cached_noise_rms', None) is not None
    if not has_noise:
        result = QMessageBox.question(
            main_window,
            "Automatic Click Detector Not Run",
            "The Automatic Click Detector has not been run yet.\n\n"
            "Extended parameters (SPR, pre_snr, C1–C5 criteria) will appear\n"
            "as N/A in the screenshot footer.\n\n"
            "Run the detector first for full parameter display, or click OK\n"
            "to proceed with basic parameters only.",
            QMessageBox.Ok | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if result != QMessageBox.Ok:
            return

    # ── 1. Choose export base folder ──────────────────────────────────────────
    export_base = QFileDialog.getExistingDirectory(
        main_window, "Select Export Folder for Screenshots", ""
    )
    if not export_base:
        return

    # ── 2. Build root screenshots folder:  {filename}_screenshots/ ───────────
    # main_window.file_path is the canonical attribute (set in ReplayWindowAudio.__init__)
    raw_path = (getattr(main_window, 'file_path', None) or
                getattr(dm, 'file_path', None) or
                getattr(dm, 'filepath', None) or
                getattr(dm, 'filename', None) or '')
    if raw_path:
        source_stem = os.path.splitext(os.path.basename(raw_path))[0]
    else:
        source_stem = "recording"
    root_screenshots_dir = os.path.join(export_base, f"{source_stem}_screenshots")
    os.makedirs(root_screenshots_dir, exist_ok=True)

    # ── 3. Threshold confirmation dialog ─────────────────────────────────────
    confirm_dlg = ThresholdConfirmDialog(dm, parent=main_window)
    if confirm_dlg.exec() != QDialog.Accepted:
        return
    params = confirm_dlg.get_params()

    threshold_v = params['threshold_mv'] / 1000.0
    use_norm    = True   # always normalise for export

    # ── 4. Get accent color from theme ────────────────────────────────────────
    accent_color = '#007bff'
    if hasattr(main_window, 'theme_manager'):
        try:
            accent_color = main_window.theme_manager.get_accent_color()
        except Exception:
            pass

    # ── 5. Progress dialog (non-modal) ────────────────────────────────────────
    prog_dlg = BatchExportProgressDialog(0, parent=main_window)
    prog_dlg.show()
    QApplication.processEvents()

    # ── 6. Shared state ───────────────────────────────────────────────────────
    state = {
        'n_saved': 0, 'n_failed': 0, 'done': False,
        'worker_n_exported': 0,
        'n_emitted': 0, 'n_rendered': 0,
    }

    # ── 7. Slot: render one frame in main thread ──────────────────────────────
    def _on_frame_ready(frame_data: dict):
        frame_idx      = frame_data['frame_idx']
        ts             = frame_data['timestamp']
        group_size     = frame_data.get('group_size', 1)
        group_first_ts = frame_data.get('group_first_ts', ts)

        fname = f"frame_{frame_idx:06d}_t{ts:.4f}s.png"

        if group_size > 1:
            # Consecutive group → subdir named after first frame timestamp
            subdir = os.path.join(root_screenshots_dir, f"t={group_first_ts:.4f}s")
            os.makedirs(subdir, exist_ok=True)
            out_path = os.path.join(subdir, fname)
        else:
            # Isolated frame → directly in root screenshots folder
            out_path = os.path.join(root_screenshots_dir, fname)

        try:
            ok = render_click_screenshot(frame_data, out_path, accent_color)
            if ok:
                state['n_saved'] += 1
            else:
                state['n_failed'] += 1
        except Exception as e:
            print(f"⚠️  render error frame {frame_idx}: {e}")
            state['n_failed'] += 1
        state['n_rendered'] += 1
        if state['done'] and state['n_rendered'] >= state['n_emitted']:
            _show_summary()

    def _show_summary():
        prog_dlg.close()
        QMessageBox.information(
            main_window, "Batch Export Complete",
            f"Export finished!\n\n"
            f"Screenshots exported: {state['n_saved']}\n"
            f"Failed: {state['n_failed']}\n\n"
            f"Folder: {root_screenshots_dir}"
        )

    # ── 8. Worker ─────────────────────────────────────────────────────────────
    worker = BatchExportWorker(
        data_manager=dm,
        threshold_v=threshold_v,
        use_normalization=use_norm,
        parent=main_window,
    )

    # Qt.QueuedConnection: slot always called in main thread (required on macOS)
    worker.frame_data_ready.connect(_on_frame_ready, Qt.QueuedConnection)
    worker.progress_updated.connect(prog_dlg.update_progress, Qt.QueuedConnection)
    prog_dlg.cancel_requested.connect(worker.cancel)

    def _on_finished(n_exported: int):
        state['done'] = True
        state['worker_n_exported'] = n_exported
        state['n_emitted'] = n_exported
        if state['n_rendered'] >= state['n_emitted']:
            _show_summary()

    def _on_error(msg: str):
        state['done'] = True
        prog_dlg.close()
        QMessageBox.critical(main_window, "Export Error", f"Worker error:\n{msg}")

    worker.finished_export.connect(_on_finished, Qt.QueuedConnection)
    worker.error_occurred.connect(_on_error, Qt.QueuedConnection)

    worker.start()
