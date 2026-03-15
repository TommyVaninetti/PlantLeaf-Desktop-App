"""
Batch Click Detection → CSV Export
====================================
Accessible from the Home window File menu.

For each selected .paudio file, runs the full v4.0 detection pipeline
(headless, in a QThread) and appends the results to a single CSV file.

One row per confirmed click, with session-level columns repeated.
Semicolon delimiter, dot decimal separator.

Columns:
  sessione_id ; condizione ; specie ; data ; durata_s ; noise_rms_uV ;
  n_click_totali ; timestamp_s ; peak_iFFT_uV ; pre_snr ; ew1_ew4_ratio ;
  asymmetry_ratio ; tau_ms ; r2_log ; slope_log ; R_spectral ; SPR ; near_end_flag ;
  filename

Architecture:
    MultiFileBatchClickCSVDialog
        │
        ├── ClickDetectionParamsDialog   (standalone params editor)
        │
        └── per-file loop (main thread, step-by-step via QTimer):
              AudioLoadWorker (QThread)
              → AudioDataManager.precompute_fft_means()
              → estimate_noise_offline()
              → _run_full_pipeline()    (pure Python, no Qt)
              → write rows to CSV
"""

import os
import gc
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from PySide6.QtCore import (
    QObject, QThread, Signal, Qt, QTimer
)
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QProgressBar, QListWidget, QListWidgetItem,
    QApplication, QMessageBox, QGroupBox, QWidget,
    QSizePolicy, QPlainTextEdit, QFormLayout,
    QDoubleSpinBox, QSpinBox,
)
from PySide6.QtGui import QFont

from saving.audio_load_progress import AudioLoadWorker
from windows.replay_window_audio import (
    AudioDataManager,
    estimate_noise_offline,
    compute_hilbert_envelope,
    find_peak,
    check_decay,
    suppress_edge_artifacts,
)
from core.replay_base_window import compute_fft_energy
from components.click_detector_dialog import SessionMetadataDialog

# ─────────────────────────────────────────────────────────────────────────────
# STATUS STRINGS
# ─────────────────────────────────────────────────────────────────────────────
STATUS_WAITING   = "⏸ waiting"
STATUS_LOADING   = "⏳ loading…"
STATUS_DETECTING = "🔍 detecting…"
STATUS_DONE      = "✅ done"
STATUS_ERROR     = "❌ error"
STATUS_SKIPPED   = "⏭ skipped"

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT PARAMETERS  (mirrors ClickDetectorDialog defaults)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_PARAMS = {
    'sigma_multiplier': 5.0,   # threshold = μ + N·σ
    'max_spr':          20.0,
    'min_peak_fft_mv':  0.85,  # mV
    'max_pre_snr':      1.8,
    'min_peak_ifft_uv': 130.0, # µV
    'max_asym':         2.5,
    'tau_min_ms':       0.045,
    'tau_max_ms':       1.3,
    'min_r2':           0.45,
}


# =============================================================================
# CLICK DETECTION PARAMETERS DIALOG
# =============================================================================
class ClickDetectionParamsDialog(QDialog):
    """
    Standalone dialog to configure the v4.0 click detection parameters
    before launching the batch export.  Does NOT require a data_manager.
    Returns the chosen values via .get_params().
    """

    def __init__(self, initial_params: dict | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Click Detection Parameters  v4.0")
        self.setMinimumWidth(480)
        self.setWindowModality(Qt.ApplicationModal)

        p = dict(DEFAULT_PARAMS)
        if initial_params:
            p.update(initial_params)

        layout = QVBoxLayout(self)

        # ── Title ─────────────────────────────────────────────────────────────
        title = QLabel("<b style='font-size:12pt;'>Detection Parameters — v4.0</b>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        layout.addWidget(QLabel(
            "<i>These parameters are applied identically to every file in the batch.</i>"
        ))
        layout.addSpacing(4)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # ── Stage 1 ───────────────────────────────────────────────────────────
        grp1 = QGroupBox("Stage 1 — Energy Threshold")
        lay1 = QFormLayout(grp1)

        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setDecimals(1)
        self.sigma_spin.setRange(1.0, 20.0)
        self.sigma_spin.setSingleStep(0.5)
        self.sigma_spin.setValue(p['sigma_multiplier'])
        self.sigma_spin.setToolTip("Threshold = μ + N·σ  (per-file statistics)")
        lay1.addRow("σ multiplier (N):", self.sigma_spin)
        layout.addWidget(grp1)

        # ── Stage 2 ───────────────────────────────────────────────────────────
        grp2 = QGroupBox("Stage 2 — FFT Filters")
        lay2 = QFormLayout(grp2)

        self.max_spr_spin = QDoubleSpinBox()
        self.max_spr_spin.setDecimals(0)
        self.max_spr_spin.setRange(3.0, 200.0)
        self.max_spr_spin.setSingleStep(1.0)
        self.max_spr_spin.setValue(p['max_spr'])
        self.max_spr_spin.setToolTip("SPR = max(|X|²) / mean(|X|²)  — broadband clicks ≤ 20")
        lay2.addRow("Max SPR:", self.max_spr_spin)

        self.min_peak_fft_spin = QDoubleSpinBox()
        self.min_peak_fft_spin.setDecimals(3)
        self.min_peak_fft_spin.setRange(0.001, 100.0)
        self.min_peak_fft_spin.setSingleStep(0.05)
        self.min_peak_fft_spin.setValue(p['min_peak_fft_mv'])
        self.min_peak_fft_spin.setSuffix(" mV")
        lay2.addRow("Min peak FFT norm:", self.min_peak_fft_spin)
        layout.addWidget(grp2)

        # ── Stage 3 ───────────────────────────────────────────────────────────
        grp3 = QGroupBox("Stage 3 — iFFT Validation (6 criteria)")
        lay3 = QFormLayout(grp3)

        self.max_pre_snr_spin = QDoubleSpinBox()
        self.max_pre_snr_spin.setDecimals(2)
        self.max_pre_snr_spin.setRange(1.0, 10.0)
        self.max_pre_snr_spin.setSingleStep(0.1)
        self.max_pre_snr_spin.setValue(p['max_pre_snr'])
        self.max_pre_snr_spin.setToolTip("C2: pre-click noise ratio  (≈1.0 = silence)")
        lay3.addRow("Max pre_snr (C2):", self.max_pre_snr_spin)

        self.min_peak_ifft_spin = QDoubleSpinBox()
        self.min_peak_ifft_spin.setDecimals(1)
        self.min_peak_ifft_spin.setRange(10.0, 10000.0)
        self.min_peak_ifft_spin.setSingleStep(10.0)
        self.min_peak_ifft_spin.setValue(p['min_peak_ifft_uv'])
        self.min_peak_ifft_spin.setSuffix(" µV")
        lay3.addRow("Min peak iFFT:", self.min_peak_ifft_spin)

        self.max_asym_spin = QDoubleSpinBox()
        self.max_asym_spin.setDecimals(2)
        self.max_asym_spin.setRange(0.1, 20.0)
        self.max_asym_spin.setSingleStep(0.1)
        self.max_asym_spin.setValue(p['max_asym'])
        self.max_asym_spin.setToolTip("Max asymmetry ratio (rise / fall)")
        lay3.addRow("Max asym:", self.max_asym_spin)

        tau_row = QHBoxLayout()
        self.tau_min_spin = QDoubleSpinBox()
        self.tau_min_spin.setDecimals(3)
        self.tau_min_spin.setRange(0.001, 5.0)
        self.tau_min_spin.setSingleStep(0.005)
        self.tau_min_spin.setValue(p['tau_min_ms'])
        self.tau_min_spin.setSuffix(" ms")
        tau_row.addWidget(self.tau_min_spin)
        tau_row.addWidget(QLabel("≤  τ  ≤"))
        self.tau_max_spin = QDoubleSpinBox()
        self.tau_max_spin.setDecimals(2)
        self.tau_max_spin.setRange(0.01, 10.0)
        self.tau_max_spin.setSingleStep(0.05)
        self.tau_max_spin.setValue(p['tau_max_ms'])
        self.tau_max_spin.setSuffix(" ms")
        tau_row.addWidget(self.tau_max_spin)
        tau_row.addStretch()
        lay3.addRow("τ range:", tau_row)

        self.min_r2_spin = QDoubleSpinBox()
        self.min_r2_spin.setDecimals(2)
        self.min_r2_spin.setRange(0.0, 1.0)
        self.min_r2_spin.setSingleStep(0.05)
        self.min_r2_spin.setValue(p['min_r2'])
        self.min_r2_spin.setToolTip("Minimum R² of log-linear decay fit")
        lay3.addRow("Min R²:", self.min_r2_spin)
        layout.addWidget(grp3)

        layout.addSpacing(8)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_ok = QPushButton("✔  Confirm")
        btn_ok.setDefault(True)
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_reset = QPushButton("Reset defaults")
        btn_reset.clicked.connect(self._reset_defaults)
        btn_row.addWidget(btn_reset)
        btn_row.addWidget(btn_ok)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

    def _reset_defaults(self):
        self.sigma_spin.setValue(DEFAULT_PARAMS['sigma_multiplier'])
        self.max_spr_spin.setValue(DEFAULT_PARAMS['max_spr'])
        self.min_peak_fft_spin.setValue(DEFAULT_PARAMS['min_peak_fft_mv'])
        self.max_pre_snr_spin.setValue(DEFAULT_PARAMS['max_pre_snr'])
        self.min_peak_ifft_spin.setValue(DEFAULT_PARAMS['min_peak_ifft_uv'])
        self.max_asym_spin.setValue(DEFAULT_PARAMS['max_asym'])
        self.tau_min_spin.setValue(DEFAULT_PARAMS['tau_min_ms'])
        self.tau_max_spin.setValue(DEFAULT_PARAMS['tau_max_ms'])
        self.min_r2_spin.setValue(DEFAULT_PARAMS['min_r2'])

    def get_params(self) -> dict:
        return {
            'sigma_multiplier': self.sigma_spin.value(),
            'max_spr':          self.max_spr_spin.value(),
            'min_peak_fft_mv':  self.min_peak_fft_spin.value(),
            'max_pre_snr':      self.max_pre_snr_spin.value(),
            'min_peak_ifft_uv': self.min_peak_ifft_spin.value(),
            'max_asym':         self.max_asym_spin.value(),
            'tau_min_ms':       self.tau_min_spin.value(),
            'tau_max_ms':       self.tau_max_spin.value(),
            'min_r2':           self.min_r2_spin.value(),
        }


# =============================================================================
# HEADLESS DETECTION PIPELINE  (no Qt widgets, pure Python)
# =============================================================================

def _normalize_fft(fft_magnitudes: np.ndarray, frequency_axis: np.ndarray) -> np.ndarray:
    """50 % conservative mic normalisation (same as ClickDetectorDialog)."""
    datasheet_freq_hz = np.array([20, 25, 30, 40, 50, 60, 70, 80]) * 1000
    datasheet_resp_db = np.array([8.0, 10.5, 6.0, -2.0, -6.0, -7.0, -6.0, -4.0])
    valid_mask = (frequency_axis >= 20000) & (frequency_axis <= 80000)
    mic_db     = np.interp(frequency_axis[valid_mask], datasheet_freq_hz, datasheet_resp_db)
    gain_50    = 10 ** (-mic_db * 0.5 / 20.0)
    normalized = fft_magnitudes.copy()
    normalized[valid_mask] *= gain_50
    return normalized


def _reconstruct_ifft(dm: AudioDataManager, frame_index: int, normalized: bool = True):
    """Reconstruct iFFT for a single frame (Tukey + edge suppression)."""
    if frame_index >= len(dm.fft_data):
        return None
    if len(dm.phase_data) == 0:
        return None

    fft_mags = dm.fft_data[frame_index].copy()
    if normalized:
        fft_mags = _normalize_fft(fft_mags, dm.frequency_axis)

    fft_phases_int8 = dm.phase_data[frame_index]

    fs       = dm.header_info.get('fs', 200000)
    fft_size = dm.header_info.get('fft_size', 512)
    num_bins = fft_size // 2
    bin_freq = fs / fft_size
    bin_start = int(20000 / bin_freq)
    bin_end   = int(80000 / bin_freq)

    full_mag   = np.zeros(num_bins, dtype=np.float32)
    full_phase = np.zeros(num_bins, dtype=np.int8)

    actual_bins = min(len(fft_mags), bin_end - bin_start + 1, len(fft_phases_int8))
    full_mag[bin_start:bin_start + actual_bins]   = fft_mags[:actual_bins]
    full_phase[bin_start:bin_start + actual_bins] = fft_phases_int8[:actual_bins]

    phases_rad       = (full_phase / 127.0) * np.pi
    complex_spectrum = full_mag * np.exp(1j * phases_rad)

    taper  = max(5, actual_bins // 10)
    window = np.ones(num_bins)
    for i in range(taper):
        alpha = i / taper
        window[bin_start + i]                   = 0.5 * (1 - np.cos(np.pi * alpha))
        window[bin_start + actual_bins - i - 1] = 0.5 * (1 - np.cos(np.pi * alpha))
    complex_spectrum *= window

    try:
        sig = np.fft.irfft(complex_spectrum, n=fft_size)
        return suppress_edge_artifacts(sig)
    except Exception:
        return None


def run_full_pipeline(dm: AudioDataManager, params: dict, log_fn=None) -> list[dict]:
    """
    Run the full v4.0 4-stage headless pipeline on a loaded AudioDataManager.

    Parameters
    ----------
    dm      : fully loaded & precomputed AudioDataManager
    params  : dict from ClickDetectionParamsDialog.get_params()
    log_fn  : callable(str) — optional logger

    Returns
    -------
    list of click dicts (same schema used by ClickDetectorDialog)
    """

    def _log(msg):
        if log_fn:
            log_fn(msg)

    # ── Read parameters ───────────────────────────────────────────────────────
    sigma_mult       = params.get('sigma_multiplier', 5.0)
    max_spr          = params.get('max_spr', 20.0)
    min_peak_fft_v   = params.get('min_peak_fft_mv', 0.85) / 1000.0
    max_pre_snr      = params.get('max_pre_snr', 1.8)
    min_peak_ifft_v  = params.get('min_peak_ifft_uv', 130.0) / 1e6
    max_asym         = params.get('max_asym', 2.5)
    tau_min          = params.get('tau_min_ms', 0.045)
    tau_max          = params.get('tau_max_ms', 1.3)
    min_r2           = params.get('min_r2', 0.45)

    threshold_v      = dm.fft_mean + sigma_mult * dm.fft_std
    total_frames     = dm.total_frames
    MAX_RUN          = 4
    MIN_DECAY_RATIO  = 2.0
    MIN_PRE_SAMPLES  = 50
    GUARD            = 20
    LEVEL_FRACTION   = 0.10
    FALL_SEARCH      = 40

    _log(f"  Threshold (μ+{sigma_mult:.1f}σ): {threshold_v*1000:.3f} mV")

    # ── Offline noise ─────────────────────────────────────────────────────────
    noise_info = estimate_noise_offline(dm, energy_threshold_multiplier=4.0, max_samples=500)
    noise_rms  = noise_info['noise_rms']
    dm._cached_noise_rms = noise_rms
    _log(f"  Noise RMS: {noise_rms*1000:.6f} mV")

    # ── Stage 1: threshold + group-size filter ────────────────────────────────
    above = [i for i in range(total_frames) if dm.fft_means[i] > threshold_v]

    filtered_groups = []
    if above:
        run_start = 0
        for k in range(1, len(above) + 1):
            at_end  = (k == len(above))
            new_run = at_end or (above[k] - above[k - 1] > 1)
            if new_run:
                run = above[run_start:k]
                if len(run) <= MAX_RUN:
                    filtered_groups.append(run)
                run_start = k

    frame_group_meta  = {fi: len(grp) for grp in filtered_groups for fi in grp}
    candidates_stage1 = [fi for grp in filtered_groups for fi in grp]
    _log(f"  Stage 1: {len(candidates_stage1)}/{total_frames} passed")

    if not candidates_stage1:
        return []

    # ── Stage 2: SPR + peak amplitude ────────────────────────────────────────
    candidates_stage2 = []
    for frame_idx in candidates_stage1:
        fft_raw  = dm.fft_data[frame_idx]
        fft_norm = _normalize_fft(fft_raw, dm.frequency_axis)

        peak_fft_v = float(np.max(fft_norm))
        if peak_fft_v <= min_peak_fft_v:
            continue

        fft_power  = fft_norm.astype(np.float64) ** 2
        mean_power = float(np.mean(fft_power))
        max_power  = float(np.max(fft_power))
        spr = max_power / mean_power if mean_power > 1e-20 else 0.0
        if spr > max_spr:
            continue

        energies = compute_fft_energy(fft_norm)
        candidates_stage2.append({
            'frame_idx':  frame_idx,
            'group_size': frame_group_meta.get(frame_idx, 1),
            'energies':   energies,
            'spr':        spr,
            'peak_fft_v': peak_fft_v,
        })

    _log(f"  Stage 2: {len(candidates_stage2)}/{len(candidates_stage1)} passed")

    if not candidates_stage2:
        return []

    # ── Stage 3: iFFT validation ──────────────────────────────────────────────
    candidates_stage3 = []
    for candidate in candidates_stage2:
        frame_idx = candidate['frame_idx']

        signal = _reconstruct_ifft(dm, frame_idx, normalized=True)
        if signal is None:
            continue

        envelope = compute_hilbert_envelope(signal)
        peak_idx, peak_amp = find_peak(envelope)

        next_frame_envelope = None
        if peak_idx > 212 and frame_idx + 1 < total_frames:
            next_sig = _reconstruct_ifft(dm, frame_idx + 1, normalized=True)
            if next_sig is not None:
                try:
                    next_frame_envelope = compute_hilbert_envelope(next_sig)
                except Exception:
                    pass

        decay = check_decay(envelope, peak_idx,
                            next_frame_envelope=next_frame_envelope,
                            noise_rms=noise_rms)
        tau_ms = decay['tau_ms']
        r2_log = decay['r_squared_log']
        E_W1   = decay['E_W1']
        E_W4   = decay['E_W4']

        # Criterion: peak_ifft
        peak_pass = (peak_amp > min_peak_ifft_v)

        # C2: pre_snr
        pre_end = max(0, peak_idx - GUARD)
        if pre_end >= MIN_PRE_SAMPLES:
            pre_window = signal[:pre_end]
        elif frame_idx > 0:
            prev_sig = _reconstruct_ifft(dm, frame_idx - 1, normalized=True)
            pre_window = (np.concatenate([prev_sig[-200:],
                                          signal[:pre_end] if pre_end > 0 else np.array([])])
                          if prev_sig is not None
                          else (signal[:pre_end] if pre_end > 0 else np.array([noise_rms])))
        else:
            pre_window = np.array([noise_rms])

        rms_pre = float(np.sqrt(np.mean(pre_window ** 2))) if len(pre_window) > 0 else noise_rms
        pre_snr = rms_pre / noise_rms if noise_rms > 0 else 1.0
        c2_pass = (pre_snr < max_pre_snr)

        # C3: E_W1 > 2× E_W4
        c3_pass = (E_W1 > E_W4 * MIN_DECAY_RATIO)

        # asym
        level = peak_amp * LEVEL_FRACTION
        rise_start = peak_idx
        for i in range(peak_idx - 1, -1, -1):
            if envelope[i] < level:
                rise_start = i + 1
                break
        rise_s = max(1, peak_idx - rise_start)

        fall_end_idx = min(peak_idx + FALL_SEARCH, len(envelope))
        fall_s       = FALL_SEARCH
        for i in range(peak_idx + 1, fall_end_idx):
            if envelope[i] < level:
                fall_s = i - peak_idx
                break
        asym_ratio = rise_s / fall_s if fall_s > 0 else 1.0
        asym_pass  = (asym_ratio < max_asym)

        # tau range
        tau_pass = (tau_min <= tau_ms <= tau_max) if tau_ms > 0 else False

        # R²
        r2_pass = (r2_log > min_r2)

        if not (peak_pass and c2_pass and c3_pass and asym_pass and tau_pass and r2_pass):
            continue

        candidates_stage3.append({
            **candidate,
            'peak_amp':      peak_amp,
            'peak_idx':      peak_idx,
            'decay_results': decay,
            'r2_log':        r2_log,
            'tau_ms':        tau_ms,
            'slope_log':     decay['slope_log'],
            'pre_snr':       pre_snr,
            'rms_pre':       rms_pre,
            'E_W1':          E_W1,
            'E_W4':          E_W4,
            'asym_ratio':    asym_ratio,
            'noise_rms':     noise_rms,
        })

    _log(f"  Stage 3: {len(candidates_stage3)}/{len(candidates_stage2)} passed")

    # ── Stage 4: deduplication ────────────────────────────────────────────────
    if not candidates_stage3:
        return []

    MAX_GAP   = 3
    sorted_c  = sorted(candidates_stage3, key=lambda x: x['frame_idx'])
    groups    = []
    current   = [sorted_c[0]]
    for i in range(1, len(sorted_c)):
        if sorted_c[i]['frame_idx'] - sorted_c[i-1]['frame_idx'] <= MAX_GAP:
            current.append(sorted_c[i])
        else:
            groups.append(current)
            current = [sorted_c[i]]
    groups.append(current)

    final = [max(grp, key=lambda x: x['peak_amp']) for grp in groups]
    _log(f"  Stage 4 (dedup): {len(final)} unique clicks")
    return final


# =============================================================================
# CSV HELPER
# =============================================================================

CSV_HEADER = [
    "sessione_id", "condizione", "specie", "data",
    "durata_s", "noise_rms_uV", "n_click_totali",
    "timestamp_s", "peak_iFFT_uV", "pre_snr", "ew1_ew4_ratio",
    "asymmetry_ratio", "tau_ms", "r2_log", "slope_log",
    "R_spectral", "SPR", "near_end_flag", "filename",
]


def _fmt(v, decimals=6):
    if v is None:
        return "NA"
    if isinstance(v, bool):
        return "1" if v else "0"
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


def _build_csv_rows(clicks: list[dict], dm: AudioDataManager,
                    sessione_id: str, condizione: str,
                    specie: str, data_str: str,
                    filename_stem: str) -> list[str]:
    """Build list of CSV lines (no newline at end) for all clicks of one file."""
    n_total      = len(clicks)
    duration_s   = dm.total_duration_sec
    noise_rms_v  = (clicks[0].get('noise_rms', 0.0)
                    if clicks
                    else getattr(dm, '_cached_noise_rms', 0.0))
    noise_rms_uv = noise_rms_v * 1e6

    rows = []
    for click in clicks:
        frame_idx  = click['frame_idx']
        ts         = (frame_idx * dm.frame_duration_ms) / 1000.0
        peak_iFFT  = click.get('peak_amp', 0.0) * 1e6
        pre_snr    = click.get('pre_snr', 0.0)

        E_W1 = click.get('E_W1', 0.0)
        E_W4 = click.get('E_W4', 1e-30)
        ew_ratio = E_W1 / E_W4 if E_W4 > 1e-30 else 999.0

        asym      = click.get('asym_ratio', 0.0)
        tau_ms    = click.get('tau_ms', -1.0)
        r2_log    = click.get('r2_log', 0.0)
        slope_log = click.get('slope_log', 0.0)

        energies   = click.get('energies', {})
        e_low      = energies.get('low',  0.0)
        e_high     = energies.get('high', 1e-30)
        r_spectral = e_low / e_high if e_high > 1e-30 else 999.0

        spr      = click.get('spr', 0.0)
        near_end = click.get('decay_results', {}).get('near_end', False)

        row = [
            sessione_id,
            condizione,
            specie,
            data_str,
            _fmt(duration_s, 3),
            _fmt(noise_rms_uv, 3),
            str(n_total),
            _fmt(ts, 6),
            _fmt(peak_iFFT, 2),
            _fmt(pre_snr, 4),
            _fmt(ew_ratio, 4),
            _fmt(asym, 4),
            _fmt(tau_ms, 4) if tau_ms > 0 else "NA",
            _fmt(r2_log, 4),
            _fmt(slope_log, 6),
            _fmt(r_spectral, 4),
            _fmt(spr, 2),
            "1" if near_end else "0",
            filename_stem,
        ]
        rows.append(";".join(row))
    return rows


# =============================================================================
# MAIN DIALOG
# =============================================================================

class MultiFileBatchClickCSVDialog(QDialog):
    """
    Modal dialog for multi-file batch click detection → CSV export.

    Layout:
    ┌─ Session metadata ─────────────────────────────────────────────────────┐
    │  [Edit Parameters…]   σ=5  SPR≤20  peak>130µV  τ=[0.045,1.3]ms  R²>0.45│
    ├─ File list ────────────────────────────────────────────────────────────┤
    │  ⏸ file1.paudio  …                                                     │
    ├─ Output CSV ───────────────────────────────────────────────────────────┤
    │  [Browse…]  /path/output.csv                                            │
    ├─ Progress ─────────────────────────────────────────────────────────────┤
    │  ████░░░░  2 / 5 files                                                 │
    ├─ Log ──────────────────────────────────────────────────────────────────┤
    │  …                                                                      │
    ├────────────────────────────────────────────────────────────────────────┤
    │        [Detection Params…]  [Start]  [Cancel]  [Close]                │
    └────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Click Detection → CSV Export")
        self.setMinimumSize(720, 700)
        self.resize(920, 900)

        self._file_paths:       list[str] = []
        self._output_csv:       str = ""
        self._running:          bool = False
        self._cancelled:        bool = False
        self._current_idx:      int = 0
        self._session_meta:     dict | None = None   # filled at Start time
        self._detection_params: dict = dict(DEFAULT_PARAMS)

        # Qt objects for the current file pipeline
        self._load_thread: QThread | None = None
        self._load_worker: AudioLoadWorker | None = None
        self._current_dm:  AudioDataManager | None = None

        self._build_ui()

        if parent and hasattr(parent, 'theme_manager'):
            try:
                theme = parent.theme_manager.load_saved_theme()
                parent.theme_manager.apply_theme(self, theme)
                if 'light' in theme.lower():
                    self.setStyleSheet("QDialog { background-color: white; }")
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────────
    # UI BUILD
    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)

        # Title
        title = QLabel("<b style='font-size:13pt;'>Batch Click Detection → CSV Export</b>")
        title.setAlignment(Qt.AlignCenter)
        root.addWidget(title)

        desc = QLabel(
            "Runs the v4.0 detection pipeline on each .paudio file and writes "
            "all results to a single CSV (one row per click)."
        )
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignCenter)
        root.addWidget(desc)

        # ── Detection parameters summary ──────────────────────────────────────
        grp_params = QGroupBox("Detection Parameters")
        lay_params = QHBoxLayout(grp_params)
        self._lbl_params_summary = QLabel(self._params_summary())
        self._lbl_params_summary.setWordWrap(True)
        lay_params.addWidget(self._lbl_params_summary, stretch=1)
        btn_edit_params = QPushButton("✏  Edit Parameters…")
        btn_edit_params.setFixedWidth(180)
        btn_edit_params.clicked.connect(self._open_params_dialog)
        lay_params.addWidget(btn_edit_params)
        root.addWidget(grp_params)

        # ── File list ─────────────────────────────────────────────────────────
        grp_files = QGroupBox("Files to process")
        lay_files = QVBoxLayout(grp_files)

        btn_row = QHBoxLayout()
        self._btn_add  = QPushButton("Add files…")
        self._btn_add.clicked.connect(self._add_files)
        self._btn_clear = QPushButton("Clear list")
        self._btn_clear.clicked.connect(self._clear_files)
        btn_row.addWidget(self._btn_add)
        btn_row.addWidget(self._btn_clear)
        btn_row.addStretch()
        lay_files.addLayout(btn_row)

        self._file_list = QListWidget()
        self._file_list.setMinimumHeight(80)
        lay_files.addWidget(self._file_list)
        root.addWidget(grp_files)

        # ── Output CSV ────────────────────────────────────────────────────────
        grp_out = QGroupBox("Output CSV file")
        lay_out = QHBoxLayout(grp_out)
        self._lbl_output = QLabel("(not selected)")
        self._lbl_output.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._lbl_output.setWordWrap(True)
        btn_browse = QPushButton("Browse…")
        btn_browse.setFixedWidth(95)
        btn_browse.clicked.connect(self._browse_output)
        lay_out.addWidget(self._lbl_output)
        lay_out.addWidget(btn_browse)
        root.addWidget(grp_out)

        # ── Progress ──────────────────────────────────────────────────────────
        grp_prog = QGroupBox("Progress")
        lay_prog = QVBoxLayout(grp_prog)
        self._lbl_overall  = QLabel("Overall: 0 / 0 files")
        self._prog_overall = QProgressBar()
        self._prog_overall.setRange(0, 100)
        lay_prog.addWidget(self._lbl_overall)
        lay_prog.addWidget(self._prog_overall)
        self._lbl_current  = QLabel("Current file: –")
        self._prog_current = QProgressBar()
        self._prog_current.setRange(0, 100)
        lay_prog.addWidget(self._lbl_current)
        lay_prog.addWidget(self._prog_current)
        root.addWidget(grp_prog)

        # ── Log ───────────────────────────────────────────────────────────────
        grp_log = QGroupBox("Log")
        lay_log = QVBoxLayout(grp_log)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(3000)
        self._log.setMinimumHeight(120)
        lay_log.addWidget(self._log)
        root.addWidget(grp_log)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row2 = QHBoxLayout()
        btn_row2.addStretch()

        self._btn_start = QPushButton("▶  Start")
        self._btn_start.setMinimumWidth(110)
        self._btn_start.clicked.connect(self._start)

        self._btn_cancel = QPushButton("⏹  Cancel")
        self._btn_cancel.setMinimumWidth(110)
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel)

        self._btn_close = QPushButton("Close")
        self._btn_close.setMinimumWidth(110)
        self._btn_close.clicked.connect(self.reject)

        btn_row2.addWidget(self._btn_start)
        btn_row2.addWidget(self._btn_cancel)
        btn_row2.addWidget(self._btn_close)
        root.addLayout(btn_row2)

    # ─────────────────────────────────────────────────────────────────────────
    # PARAMETERS DIALOG
    # ─────────────────────────────────────────────────────────────────────────
    def _open_params_dialog(self):
        dlg = ClickDetectionParamsDialog(
            initial_params=self._detection_params, parent=self
        )
        if self.parent() and hasattr(self.parent(), 'theme_manager'):
            theme = self.parent().theme_manager.load_saved_theme()
            self.parent().theme_manager.apply_theme(dlg, theme)
        if dlg.exec() == QDialog.Accepted:
            self._detection_params = dlg.get_params()
            self._lbl_params_summary.setText(self._params_summary())

    def _params_summary(self) -> str:
        p = self._detection_params
        return (
            f"σ={p['sigma_multiplier']:.1f}  |  "
            f"SPR≤{p['max_spr']:.0f}  |  "
            f"peak_FFT>{p['min_peak_fft_mv']:.3f} mV  |  "
            f"pre_snr<{p['max_pre_snr']:.2f}  |  "
            f"peak_iFFT>{p['min_peak_ifft_uv']:.0f} µV  |  "
            f"asym<{p['max_asym']:.1f}  |  "
            f"τ∈[{p['tau_min_ms']:.3f},{p['tau_max_ms']:.2f}] ms  |  "
            f"R²>{p['min_r2']:.2f}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # FILE / OUTPUT MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────
    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select .paudio files", "",
            "PlantLeaf Audio (*.paudio);;All files (*.*)"
        )
        for p in paths:
            if p not in self._file_paths:
                self._file_paths.append(p)
                item = QListWidgetItem(f"{STATUS_WAITING}  {os.path.basename(p)}")
                item.setToolTip(p)
                self._file_list.addItem(item)
        self._log_msg(f"Added {len(paths)} file(s). Total: {len(self._file_paths)}")

    def _clear_files(self):
        if self._running:
            return
        self._file_paths.clear()
        self._file_list.clear()
        self._log_msg("File list cleared.")

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Choose output CSV file", "",
            "CSV files (*.csv);;All files (*.*)"
        )
        if path:
            if not path.lower().endswith('.csv'):
                path += '.csv'
            self._output_csv = path
            self._lbl_output.setText(path)
            self._log_msg(f"Output CSV: {path}")

    # ─────────────────────────────────────────────────────────────────────────
    # START / CANCEL
    # ─────────────────────────────────────────────────────────────────────────
    def _start(self):
        if not self._file_paths:
            QMessageBox.warning(self, "No files", "Please add at least one .paudio file.")
            return
        if not self._output_csv:
            QMessageBox.warning(self, "No output file", "Please select an output CSV file.")
            return

        # ── Collect session metadata ──────────────────────────────────────────
        meta_dlg = SessionMetadataDialog(self)
        if self.parent() and hasattr(self.parent(), 'theme_manager'):
            theme = self.parent().theme_manager.load_saved_theme()
            self.parent().theme_manager.apply_theme(meta_dlg, theme)
        if meta_dlg.exec() != QDialog.Accepted:
            return
        self._session_meta = {
            'sessione_id': meta_dlg.sessione_id,
            'condizione':  meta_dlg.condizione,
            'specie':      meta_dlg.specie,
            'data':        meta_dlg.data,
        }

        # ── Write CSV header ──────────────────────────────────────────────────
        try:
            with open(self._output_csv, 'w', newline='', encoding='utf-8') as f:
                f.write(";".join(CSV_HEADER) + "\n")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot write to CSV file:\n{e}")
            return

        # ── Reset UI ──────────────────────────────────────────────────────────
        for i in range(self._file_list.count()):
            item = self._file_list.item(i)
            item.setText(f"{STATUS_WAITING}  {os.path.basename(self._file_paths[i])}")

        self._running   = True
        self._cancelled = False
        self._current_idx = 0

        self._btn_start.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._btn_add.setEnabled(False)
        self._btn_clear.setEnabled(False)

        self._prog_overall.setRange(0, len(self._file_paths))
        self._prog_overall.setValue(0)
        self._lbl_overall.setText(f"Overall: 0 / {len(self._file_paths)} files")

        self._log_msg("=" * 60)
        self._log_msg(
            f"Starting batch click detection CSV export\n"
            f"  Files: {len(self._file_paths)}  |  "
            f"Session: {self._session_meta['sessione_id']}  |  "
            f"Output: {self._output_csv}"
        )
        self._log_msg("=" * 60)

        QTimer.singleShot(0, self._process_next_file)

    def _cancel(self):
        self._cancelled = True
        self._log_msg("⚠️  Cancel requested — stopping after current file…")
        if self._load_worker is not None:
            self._load_worker.cancel_load()
        self._btn_cancel.setEnabled(False)

    # ─────────────────────────────────────────────────────────────────────────
    # PER-FILE PIPELINE
    # ─────────────────────────────────────────────────────────────────────────
    def _process_next_file(self):
        if self._cancelled or self._current_idx >= len(self._file_paths):
            self._finish_all()
            return

        idx       = self._current_idx
        file_path = self._file_paths[idx]
        fname     = os.path.basename(file_path)

        self._set_item_status(idx, STATUS_LOADING)
        self._prog_current.setValue(0)
        self._lbl_current.setText(f"[{idx+1}/{len(self._file_paths)}] Loading: {fname}")
        self._log_msg(f"\n▶ File {idx+1}/{len(self._file_paths)}: {fname}")

        self._load_worker = AudioLoadWorker(file_path)
        self._load_thread = QThread(self)
        self._load_worker.moveToThread(self._load_thread)

        self._load_worker.progress.connect(self._on_load_progress, Qt.QueuedConnection)
        self._load_worker.finished.connect(self._on_file_loaded, Qt.QueuedConnection)
        self._load_worker.error.connect(self._on_load_error, Qt.QueuedConnection)
        self._load_thread.started.connect(self._load_worker.run)

        self._load_thread.start()

    def _on_load_progress(self, pct: int):
        self._prog_current.setValue(int(pct * 0.45))
        fname = os.path.basename(self._file_paths[self._current_idx])
        self._lbl_current.setText(
            f"[{self._current_idx+1}/{len(self._file_paths)}] Loading {fname}: {pct}%"
        )

    def _on_load_error(self, msg: str):
        self._set_item_status(self._current_idx, STATUS_ERROR)
        self._log_msg(f"  ❌ Load error: {msg}")
        self._cleanup_load()
        self._advance_to_next_file()

    def _on_file_loaded(self, data_dict: dict):
        idx   = self._current_idx
        fname = os.path.basename(self._file_paths[idx])

        self._log_msg(
            f"  ✅ Loaded: {data_dict.get('total_frames', '?')} frames, "
            f"{data_dict.get('total_duration_sec', 0):.1f} s"
        )
        self._cleanup_load()

        # ── Populate AudioDataManager ─────────────────────────────────────────
        dm = AudioDataManager()
        dm.header_info          = data_dict['header_info']
        dm.fft_data             = data_dict['fft_data']
        dm.phase_data           = data_dict.get('phase_data', [])
        dm.frequency_axis       = np.array(data_dict['frequency_axis'])
        dm.total_frames         = data_dict['total_frames']
        dm.frame_duration_ms    = data_dict['frame_duration_ms']
        dm.total_duration_sec   = data_dict['total_duration_sec']
        dm.click_events         = data_dict['click_events']
        dm.overview_x           = np.array(data_dict['overview_x'])
        dm.overview_y           = np.array(data_dict['overview_y'])
        dm.overview_loaded      = True
        dm.streaming_x          = np.array(data_dict['streaming_x'])
        dm.streaming_y          = np.array(data_dict['streaming_y'])
        dm.streaming_start_time = data_dict['streaming_start_time']
        dm.streaming_end_time   = data_dict['streaming_end_time']
        del data_dict
        gc.collect()

        self._prog_current.setValue(50)
        self._lbl_current.setText(
            f"[{idx+1}/{len(self._file_paths)}] Precomputing stats: {fname}"
        )
        QApplication.processEvents()

        dm.precompute_fft_means()
        self._current_dm = dm

        # ── Run detection pipeline (main thread, but UI stays responsive via
        #    QApplication.processEvents inside _run_detection) ─────────────────
        self._set_item_status(idx, STATUS_DETECTING)
        self._lbl_current.setText(
            f"[{idx+1}/{len(self._file_paths)}] Detecting clicks: {fname}"
        )
        self._prog_current.setValue(55)
        QApplication.processEvents()

        self._log_msg(f"  🔍 Running v4.0 pipeline…")

        def _log(msg):
            self._log_msg(msg)
            QApplication.processEvents()

        try:
            clicks = run_full_pipeline(dm, self._detection_params, log_fn=_log)
        except Exception as e:
            self._set_item_status(idx, STATUS_ERROR)
            self._log_msg(f"  ❌ Pipeline error: {e}")
            self._advance_to_next_file()
            return

        self._prog_current.setValue(90)
        QApplication.processEvents()

        # ── Append rows to CSV ────────────────────────────────────────────────
        stem = os.path.splitext(os.path.basename(self._file_paths[idx]))[0]
        meta = self._session_meta
        rows = _build_csv_rows(
            clicks, dm,
            sessione_id=meta['sessione_id'],
            condizione=meta['condizione'],
            specie=meta['specie'],
            data_str=meta['data'],
            filename_stem=stem,
        )

        try:
            with open(self._output_csv, 'a', newline='', encoding='utf-8') as f:
                for row in rows:
                    f.write(row + "\n")
        except Exception as e:
            self._log_msg(f"  ⚠️  CSV write error: {e}")

        self._log_msg(
            f"  📊 {len(clicks)} click(s) written to CSV  "
            f"(duration={dm.total_duration_sec:.1f} s)"
        )
        self._prog_current.setValue(100)
        self._set_item_status(idx, STATUS_DONE)
        self._advance_to_next_file()

    # ─────────────────────────────────────────────────────────────────────────
    # ADVANCE / FINISH
    # ─────────────────────────────────────────────────────────────────────────
    def _advance_to_next_file(self):
        if self._current_dm is not None:
            self._current_dm.fft_data   = []
            self._current_dm.phase_data = []
            self._current_dm = None
        gc.collect()

        self._current_idx += 1
        n_done  = self._current_idx
        n_total = len(self._file_paths)

        self._prog_overall.setValue(n_done)
        self._lbl_overall.setText(f"Overall: {n_done} / {n_total} files")

        if self._cancelled:
            for i in range(n_done, n_total):
                self._set_item_status(i, STATUS_SKIPPED)
            self._finish_all()
            return

        QTimer.singleShot(0, self._process_next_file)

    def _finish_all(self):
        self._running = False
        self._btn_start.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._btn_add.setEnabled(True)
        self._btn_clear.setEnabled(True)

        n_ok = sum(
            1 for i in range(self._file_list.count())
            if STATUS_DONE in (self._file_list.item(i).text() or "")
        )
        n_err = sum(
            1 for i in range(self._file_list.count())
            if STATUS_ERROR in (self._file_list.item(i).text() or "")
        )

        if self._cancelled:
            msg = (f"Batch detection cancelled.\n\n"
                   f"Completed: {n_ok} file(s)\nErrors: {n_err}\n"
                   f"Skipped: {len(self._file_paths) - self._current_idx}\n\n"
                   f"Output CSV: {self._output_csv}")
        else:
            msg = (f"Batch detection complete!\n\n"
                   f"Files processed: {n_ok}\nErrors: {n_err}\n\n"
                   f"Output CSV: {self._output_csv}")

        self._log_msg("\n" + "=" * 60)
        self._log_msg(msg.replace("\n", "  "))

        QMessageBox.information(self, "Batch Detection Complete", msg)

    # ─────────────────────────────────────────────────────────────────────────
    # CLEANUP
    # ─────────────────────────────────────────────────────────────────────────
    def _cleanup_load(self):
        if self._load_thread is not None:
            try:
                if self._load_thread.isRunning():
                    self._load_thread.quit()
                    self._load_thread.wait(5000)
            except Exception:
                pass
            self._load_thread = None
        self._load_worker = None

    # ─────────────────────────────────────────────────────────────────────────
    # UI HELPERS
    # ─────────────────────────────────────────────────────────────────────────
    def _set_item_status(self, idx: int, status: str):
        if idx < self._file_list.count():
            fname = os.path.basename(self._file_paths[idx])
            self._file_list.item(idx).setText(f"{status}  {fname}")

    def _log_msg(self, text: str):
        self._log.appendPlainText(text)
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())
        QApplication.processEvents()

    def closeEvent(self, event):
        if self._running:
            result = QMessageBox.question(
                self, "Detection in progress",
                "A batch detection is running. Cancel and close?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if result != QMessageBox.Yes:
                event.ignore()
                return
            self._cancel()
        self._cleanup_load()
        super().closeEvent(event)


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

def launch_multi_file_batch_click_csv(parent_window=None):
    """Called from Home window menu."""
    dlg = MultiFileBatchClickCSVDialog(parent=parent_window)
    dlg.exec()
