"""
Click Detector Algorithm Dialog - Rilevamento automatico click ultrasonici

PIPELINE v4.0 (only-normalized):
1. Stage 1: Energy threshold (μ + Nσ) + group-size filter (run ≤ 4 frame)
2. Stage 2: SPR ≤ 20  AND  peak normalized FFT > 0.85 mV
3. Stage 3: iFFT validation (normalized only):
   - C2: pre_snr < 1.7
   - C3: E_W1 > 2× E_W4
   - asym < 2.5
   - peak iFFT normalized > 130 µV
   - 0.045 ms ≤ τ ≤ 1.3 ms   (N/A → fail)
   - R² > 0.45
4. Stage 4: Deduplication
"""

import numpy as np
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QDoubleSpinBox, QCheckBox, QGroupBox, QProgressDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QFormLayout, QSpinBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from windows.replay_window_audio import (
    compute_hilbert_envelope, find_peak, check_decay,
    estimate_noise_offline, suppress_edge_artifacts
)
from core.replay_base_window import compute_fft_energy


class ClickDetectorDialog(QDialog):
    """Dialog per configurazione e esecuzione algoritmo rilevamento click (v4.0)"""

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.parent = parent
        self.detected_clicks = []

        self.setWindowTitle("Automatic Click Detector  v4.0")
        self.setMinimumSize(1000, 750)

        self.setup_ui()
        self.load_default_parameters()

        if parent and hasattr(parent, 'theme_manager'):
            saved_theme = parent.theme_manager.load_saved_theme()
            parent.theme_manager.apply_theme(self, saved_theme)
            if 'light' in saved_theme.lower():
                self.setStyleSheet("QDialog { background-color: white; }")

    # =========================================================================
    # UI SETUP
    # =========================================================================
    def setup_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("<b style='font-size:15pt;'>Automatic Click Detector  v4.0</b>")
        title.setAlignment(Qt.AlignLeft)
        layout.addWidget(title)

        subtitle = QLabel(
            "<i>4-stage pipeline (normalized only): "
            "Energy+Groups → FFT amplitude+SPR → iFFT 6 criteria → Deduplication</i>"
        )
        subtitle.setAlignment(Qt.AlignLeft)
        layout.addWidget(subtitle)

        # ── PARAMETERS ───────────────────────────────────────────────────────
        params_group = QGroupBox("Detection Parameters")
        params_layout = QFormLayout()

        # Stage 1 — threshold
        threshold_layout = QHBoxLayout()
        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setDecimals(3)
        self.threshold_spinbox.setRange(0, 10000)
        self.threshold_spinbox.setSuffix(" mV")
        self.threshold_spinbox.setToolTip("Stage 1: meanFFT energy threshold\nDefault: μ + 5σ")
        threshold_layout.addWidget(self.threshold_spinbox)
        self.threshold_sigma_label = QLabel("(μ + 5.0σ)")
        self.threshold_sigma_label.setStyleSheet("color: gray;")
        threshold_layout.addWidget(self.threshold_sigma_label)
        threshold_layout.addStretch()
        params_layout.addRow("Energy Threshold (Stage 1):", threshold_layout)

        # Stage 2 — Max SPR
        self.max_spr_spinbox = QDoubleSpinBox()
        self.max_spr_spinbox.setDecimals(0)
        self.max_spr_spinbox.setRange(3.0, 200.0)
        self.max_spr_spinbox.setSingleStep(1.0)
        self.max_spr_spinbox.setValue(20.0)
        self.max_spr_spinbox.setToolTip(
            "Stage 2 – Max Spectral Peak Ratio (SPR)\n"
            "SPR = max(|X[k]|²) / mean(|X[k]|²)  over 20-80 kHz bins\n"
            "Click broadband: SPR ≈ 4–15 | Pure tone: SPR ≈ 50–150\n"
            "Default: 20"
        )
        params_layout.addRow("Max SPR (Stage 2):", self.max_spr_spinbox)

        # Stage 2 — Min peak FFT normalized
        self.min_peak_fft_spinbox = QDoubleSpinBox()
        self.min_peak_fft_spinbox.setDecimals(3)
        self.min_peak_fft_spinbox.setRange(0.001, 100.0)
        self.min_peak_fft_spinbox.setSingleStep(0.05)
        self.min_peak_fft_spinbox.setValue(0.85)
        self.min_peak_fft_spinbox.setSuffix(" mV")
        self.min_peak_fft_spinbox.setToolTip(
            "Stage 2 – Minimum peak amplitude of normalized FFT\n"
            "Frames with max(FFT_norm) ≤ threshold are discarded\n"
            "Default: 0.85 mV"
        )
        params_layout.addRow("Min peak FFT norm (Stage 2):", self.min_peak_fft_spinbox)

        # Stage 3 — Max pre_snr
        self.max_pre_snr_spinbox = QDoubleSpinBox()
        self.max_pre_snr_spinbox.setDecimals(2)
        self.max_pre_snr_spinbox.setRange(1.0, 10.0)
        self.max_pre_snr_spinbox.setSingleStep(0.1)
        self.max_pre_snr_spinbox.setValue(1.7)
        self.max_pre_snr_spinbox.setToolTip(
            "Stage 3 – C2: Max pre-click noise level\n"
            "pre_snr = RMS(signal before peak) / noise_rms\n"
            "≈ 1.0 → silence before click  (ideal)\n"
            "Default: 1.7"
        )
        params_layout.addRow("Max pre_snr (C2):", self.max_pre_snr_spinbox)

        # Stage 3 — Min iFFT peak
        self.min_peak_ifft_spinbox = QDoubleSpinBox()
        self.min_peak_ifft_spinbox.setDecimals(1)
        self.min_peak_ifft_spinbox.setRange(10.0, 10000.0)
        self.min_peak_ifft_spinbox.setSingleStep(10.0)
        self.min_peak_ifft_spinbox.setValue(130.0)
        self.min_peak_ifft_spinbox.setSuffix(" µV")
        self.min_peak_ifft_spinbox.setToolTip(
            "Stage 3 – Minimum peak amplitude of normalized iFFT\n"
            "Default: 130 µV"
        )
        params_layout.addRow("Min peak iFFT norm (Stage 3):", self.min_peak_ifft_spinbox)

        # Stage 3 — Max asym
        self.max_asym_spinbox = QDoubleSpinBox()
        self.max_asym_spinbox.setDecimals(2)
        self.max_asym_spinbox.setRange(0.1, 20.0)
        self.max_asym_spinbox.setSingleStep(0.1)
        self.max_asym_spinbox.setValue(2.5)
        self.max_asym_spinbox.setToolTip(
            "Stage 3 – Max asymmetry ratio (rise_samples / fall_samples)\n"
            "Real clicks: very fast rise → ratio typically 0.05–0.5\n"
            "Default: 2.5 (rejects signals with very slow rise relative to fall)"
        )
        params_layout.addRow("Max asym (Stage 3):", self.max_asym_spinbox)

        # Stage 3 — tau range
        tau_layout = QHBoxLayout()
        self.tau_min_spinbox = QDoubleSpinBox()
        self.tau_min_spinbox.setDecimals(3)
        self.tau_min_spinbox.setRange(0.001, 5.0)
        self.tau_min_spinbox.setSingleStep(0.005)
        self.tau_min_spinbox.setValue(0.045)
        self.tau_min_spinbox.setSuffix(" ms")
        tau_layout.addWidget(self.tau_min_spinbox)
        tau_layout.addWidget(QLabel("≤  τ  ≤"))
        self.tau_max_spinbox = QDoubleSpinBox()
        self.tau_max_spinbox.setDecimals(2)
        self.tau_max_spinbox.setRange(0.01, 10.0)
        self.tau_max_spinbox.setSingleStep(0.05)
        self.tau_max_spinbox.setValue(1.3)
        self.tau_max_spinbox.setSuffix(" ms")
        tau_layout.addWidget(self.tau_max_spinbox)
        tau_layout.addStretch()
        params_layout.addRow("τ range (Stage 3):", tau_layout)

        # Stage 3 — min R²
        self.min_r2_spinbox = QDoubleSpinBox()
        self.min_r2_spinbox.setDecimals(2)
        self.min_r2_spinbox.setRange(0.0, 1.0)
        self.min_r2_spinbox.setSingleStep(0.05)
        self.min_r2_spinbox.setValue(0.45)
        self.min_r2_spinbox.setToolTip(
            "Stage 3 – Minimum R² of log-linear decay fit\n"
            "Default: 0.45"
        )
        params_layout.addRow("Min R² (Stage 3):", self.min_r2_spinbox)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # ── FILE INFO ─────────────────────────────────────────────────────────
        info_group = QGroupBox("File Information")
        info_layout = QFormLayout()
        self.info_duration   = QLabel("0.0 s")
        self.info_frames     = QLabel("0")
        self.info_mean_energy = QLabel("0.000 mV")
        self.info_std_energy  = QLabel("0.000 mV")
        info_layout.addRow("Total duration:", self.info_duration)
        info_layout.addRow("Total frames:",   self.info_frames)
        info_layout.addRow("Mean energy (μ):", self.info_mean_energy)
        info_layout.addRow("Std deviation (σ):", self.info_std_energy)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # ── RESULTS TABLE ─────────────────────────────────────────────────────
        results_label = QLabel("<b>Detected Clicks</b>")
        layout.addWidget(results_label)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(11)
        self.results_table.setHorizontalHeaderLabels([
            "Timestamp", "Peak FFT (mV)", "Peak iFFT (µV)", "SPR",
            "pre_snr", "E_W1/E_W4", "asym", "τ (ms)", "R²", "Group sz", "Notes"
        ])
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setAlternatingRowColors(True)

        hdr = self.results_table.horizontalHeader()
        for c in range(10):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(10, QHeaderView.Stretch)
        self.results_table.verticalHeader().setVisible(False)
        layout.addWidget(self.results_table)

        # ── BUTTONS ───────────────────────────────────────────────────────────
        button_layout = QHBoxLayout()
        self.run_button = QPushButton("▶ Run Detection")
        self.run_button.setMinimumHeight(40)
        self.run_button.setStyleSheet("font-size: 12pt; font-weight: bold;")
        self.run_button.clicked.connect(self.run_detection)
        button_layout.addWidget(self.run_button)

        self.export_button = QPushButton("Export Results…")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_results)
        button_layout.addWidget(self.export_button)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)

    # =========================================================================
    # PARAMETER LOADING
    # =========================================================================
    def load_default_parameters(self):
        if not hasattr(self.data_manager, 'fft_mean'):
            QMessageBox.warning(self, "Warning", "File statistics not available.\nPlease load a .paudio file first.")
            return

        mean_mv = self.data_manager.fft_mean * 1000
        std_mv  = self.data_manager.fft_std  * 1000

        default_threshold = mean_mv + 5.0 * std_mv
        self.threshold_spinbox.setValue(default_threshold)
        self.threshold_spinbox.setSingleStep(std_mv)

        self.info_duration.setText(f"{self.data_manager.total_duration_sec:.1f} s")
        self.info_frames.setText(f"{self.data_manager.total_frames}")
        self.info_mean_energy.setText(f"{mean_mv:.3f} mV")
        self.info_std_energy.setText(f"{std_mv:.3f} mV")

        sigma_mult = (default_threshold - mean_mv) / std_mv if std_mv > 0 else 5.0
        self.threshold_sigma_label.setText(f"(μ + {sigma_mult:.1f}σ)")
        self.threshold_spinbox.valueChanged.connect(self.update_sigma_label)

    def update_sigma_label(self):
        if not hasattr(self.data_manager, 'fft_mean'):
            return
        mean_mv = self.data_manager.fft_mean * 1000
        std_mv  = self.data_manager.fft_std  * 1000
        if std_mv > 0:
            sigma_mult = (self.threshold_spinbox.value() - mean_mv) / std_mv
            self.threshold_sigma_label.setText(f"(μ + {sigma_mult:.1f}σ)")

    # =========================================================================
    # HELPERS
    # =========================================================================
    def _normalize_fft(self, fft_magnitudes):
        """Apply 50% conservative mic normalisation."""
        datasheet_freq_hz  = np.array([20, 25, 30, 40, 50, 60, 70, 80]) * 1000
        datasheet_resp_db  = np.array([8.0, 10.5, 6.0, -2.0, -6.0, -7.0, -6.0, -4.0])
        freq_axis  = self.data_manager.frequency_axis
        valid_mask = (freq_axis >= 20000) & (freq_axis <= 80000)
        mic_db     = np.interp(freq_axis[valid_mask], datasheet_freq_hz, datasheet_resp_db)
        gain_50    = 10 ** (-mic_db * 0.5 / 20.0)
        normalized = fft_magnitudes.copy()
        normalized[valid_mask] *= gain_50
        return normalized

    def _reconstruct_ifft(self, frame_index, normalized=False):
        """Reconstruct time-domain signal from FFT+phase data (with Tukey + edge suppression)."""
        if frame_index >= len(self.data_manager.fft_data):
            return None
        if len(self.data_manager.phase_data) == 0:
            return None

        fft_mags = self.data_manager.fft_data[frame_index].copy()
        if normalized:
            fft_mags = self._normalize_fft(fft_mags)

        fft_phases_int8 = self.data_manager.phase_data[frame_index]

        fs       = self.data_manager.header_info.get('fs', 200000)
        fft_size = self.data_manager.header_info.get('fft_size', 512)
        num_bins = fft_size // 2
        bin_freq = fs / fft_size
        bin_start = int(20000 / bin_freq)
        bin_end   = int(80000 / bin_freq)

        full_mag   = np.zeros(num_bins, dtype=np.float32)
        full_phase = np.zeros(num_bins, dtype=np.int8)

        actual_bins = min(len(fft_mags), bin_end - bin_start + 1, len(fft_phases_int8))
        full_mag[bin_start:bin_start + actual_bins]   = fft_mags[:actual_bins]
        full_phase[bin_start:bin_start + actual_bins] = fft_phases_int8[:actual_bins]

        phases_rad      = (full_phase / 127.0) * np.pi
        complex_spectrum = full_mag * np.exp(1j * phases_rad)

        taper = max(5, actual_bins // 10)
        window = np.ones(num_bins)
        for i in range(taper):
            alpha = i / taper
            window[bin_start + i]                      = 0.5 * (1 - np.cos(np.pi * alpha))
            window[bin_start + actual_bins - i - 1]    = 0.5 * (1 - np.cos(np.pi * alpha))
        complex_spectrum *= window

        try:
            sig = np.fft.irfft(complex_spectrum, n=fft_size)
            sig = suppress_edge_artifacts(sig)
            return sig
        except Exception:
            return None

    # =========================================================================
    # MAIN DETECTION PIPELINE
    # =========================================================================
    def run_detection(self):
        """Execute 4-stage detection pipeline (v4.0, normalized only)."""
        print("\n" + "="*80)
        print("🔍 AUTOMATIC CLICK DETECTOR v4.0 — PIPELINE EXECUTION")
        print("="*80)

        self.detected_clicks = []
        self.results_table.setRowCount(0)

        # ── Read parameters ───────────────────────────────────────────────────
        threshold_v      = self.threshold_spinbox.value() / 1000.0
        max_spr          = self.max_spr_spinbox.value()
        min_peak_fft_v   = self.min_peak_fft_spinbox.value() / 1000.0    # mV → V
        max_pre_snr      = self.max_pre_snr_spinbox.value()
        min_peak_ifft_v  = self.min_peak_ifft_spinbox.value() / 1e6      # µV → V
        max_asym         = self.max_asym_spinbox.value()
        tau_min          = self.tau_min_spinbox.value()                   # ms
        tau_max          = self.tau_max_spinbox.value()                   # ms
        min_r2           = self.min_r2_spinbox.value()

        total_frames = self.data_manager.total_frames
        MAX_RUN = 4   # Stage 1 group filter: discard runs of > 4 consecutive frames

        print(f"\n📋 PARAMETERS:")
        print(f"   Threshold:        {threshold_v*1000:.3f} mV")
        print(f"   Max run (Stage 1): {MAX_RUN} frames")
        print(f"   Max SPR (Stage 2): {max_spr:.0f}")
        print(f"   Min peak FFT norm: {min_peak_fft_v*1000:.3f} mV (Stage 2)")
        print(f"   Max pre_snr (C2):  {max_pre_snr:.2f}")
        print(f"   Min peak iFFT (Stage 3): {min_peak_ifft_v*1e6:.1f} µV")
        print(f"   Max asym (Stage 3): {max_asym:.2f}")
        print(f"   τ range:           {tau_min:.2f}–{tau_max:.2f} ms")
        print(f"   Min R²:            {min_r2:.2f}")

        progress = QProgressDialog("Running click detection…", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        # ====================================================================
        # OFFLINE NOISE ESTIMATION
        # ====================================================================
        progress.setLabelText("Estimating noise from empty frames…")
        progress.setValue(3)
        noise_info = estimate_noise_offline(self.data_manager,
                                            energy_threshold_multiplier=4.0,
                                            max_samples=500)
        noise_rms = noise_info['noise_rms']
        self.data_manager._cached_noise_rms = noise_rms
        print(f"✅ Noise RMS: {noise_rms*1000:.6f} mV  ({noise_info['n_samples']} frames sampled)")

        # ====================================================================
        # STAGE 1: ENERGY THRESHOLD + GROUP-SIZE FILTER
        # ====================================================================
        print(f"\n{'='*60}\nSTAGE 1: ENERGY THRESHOLD + GROUP-SIZE FILTER\n{'='*60}")

        above_threshold = []
        for i in range(total_frames):
            if i % 10000 == 0:
                progress.setValue(int((i / total_frames) * 15))
                if progress.wasCanceled(): return
            if self.data_manager.fft_means[i] > threshold_v:
                above_threshold.append(i)

        # Build consecutive runs and keep only those with run_len ≤ MAX_RUN
        filtered_groups = []
        if above_threshold:
            run_start = 0
            for k in range(1, len(above_threshold) + 1):
                at_end  = (k == len(above_threshold))
                new_run = at_end or (above_threshold[k] - above_threshold[k - 1] > 1)
                if new_run:
                    run = above_threshold[run_start:k]
                    if len(run) <= MAX_RUN:
                        filtered_groups.append(run)
                    else:
                        print(f"   ⚠️ Discarded run of {len(run)} frames "
                              f"(#{run[0]}–#{run[-1]}) — group too large")
                    run_start = k

        # Flatten keeping group metadata
        frame_group_meta = {}
        for grp in filtered_groups:
            for fi in grp:
                frame_group_meta[fi] = len(grp)

        candidates_stage1 = [fi for grp in filtered_groups for fi in grp]
        print(f"✅ Stage 1: {len(candidates_stage1)}/{total_frames} frames passed "
              f"({len(above_threshold) - len(candidates_stage1)} rejected by group filter)")

        if not candidates_stage1:
            progress.close()
            QMessageBox.information(self, "No Clicks Found",
                                    "No frames passed Stage 1.\nTry lowering the energy threshold.")
            return

        # ====================================================================
        # STAGE 2: FFT FILTERS (SPR + peak amplitude, normalized)
        # ====================================================================
        print(f"\n{'='*60}\nSTAGE 2: FFT FILTERS (SPR ≤ {max_spr:.0f}, peak_norm > {min_peak_fft_v*1000:.3f} mV)\n{'='*60}")

        candidates_stage2 = []
        rej_spr = rej_amp = 0

        for idx, frame_idx in enumerate(candidates_stage1):
            if idx % 100 == 0:
                progress.setValue(15 + int((idx / len(candidates_stage1)) * 20))
                if progress.wasCanceled(): return

            fft_raw  = self.data_manager.fft_data[frame_idx]
            fft_norm = self._normalize_fft(fft_raw)

            # Peak amplitude check (normalized)
            peak_fft_v = float(np.max(fft_norm))
            if peak_fft_v <= min_peak_fft_v:
                rej_amp += 1
                continue

            # SPR check (normalized)
            fft_power  = fft_norm.astype(np.float64) ** 2
            mean_power = float(np.mean(fft_power))
            max_power  = float(np.max(fft_power))
            spr = max_power / mean_power if mean_power > 1e-20 else 0.0
            if spr > max_spr:
                rej_spr += 1
                continue

            energies = compute_fft_energy(fft_norm)
            ratio    = energies['low'] / energies['high'] if energies['high'] > 0 else 0.0

            candidates_stage2.append({
                'frame_idx':  frame_idx,
                'group_size': frame_group_meta.get(frame_idx, 1),
                'energies':   energies,
                'ratio':      ratio,
                'spr':        spr,
                'peak_fft_v': peak_fft_v,
            })

        print(f"✅ Stage 2: {len(candidates_stage2)}/{len(candidates_stage1)} passed "
              f"({rej_amp} amp, {rej_spr} SPR rejected)")

        if not candidates_stage2:
            progress.close()
            QMessageBox.information(self, "No Clicks Found",
                                    "No frames passed Stage 2 FFT filters.\n"
                                    "Try raising Max SPR or lowering Min peak FFT.")
            return

        # ====================================================================
        # STAGE 3: iFFT VALIDATION (normalized, 6 criteria)
        # ====================================================================
        print(f"\n{'='*60}\nSTAGE 3: iFFT VALIDATION (normalized)\n{'='*60}")
        print(f"   C2: pre_snr < {max_pre_snr:.2f}")
        print(f"   C3: E_W1 > 2× E_W4")
        print(f"   asym < {max_asym:.2f}")
        print(f"   peak_ifft > {min_peak_ifft_v*1e6:.1f} µV")
        print(f"   τ ∈ [{tau_min:.2f}, {tau_max:.2f}] ms")
        print(f"   R² > {min_r2:.2f}")

        MIN_DECAY_RATIO = 2.0
        MIN_PRE_SAMPLES = 50
        GUARD           = 20
        candidates_stage3 = []

        for idx, candidate in enumerate(candidates_stage2):
            if idx % 10 == 0:
                progress.setValue(35 + int((idx / len(candidates_stage2)) * 55))
                if progress.wasCanceled(): return

            frame_idx = candidate['frame_idx']

            # Reconstruct normalized iFFT
            signal = self._reconstruct_ifft(frame_idx, normalized=True)
            if signal is None:
                continue

            envelope = compute_hilbert_envelope(signal)
            peak_idx, peak_amp = find_peak(envelope)

            # ── pre-compute next frame envelope for spill handling ───────────
            next_frame_envelope = None
            if peak_idx > 212 and frame_idx + 1 < total_frames:
                next_sig = self._reconstruct_ifft(frame_idx + 1, normalized=True)
                if next_sig is not None:
                    try:
                        next_frame_envelope = compute_hilbert_envelope(next_sig)
                    except Exception:
                        next_frame_envelope = None

            decay = check_decay(envelope, peak_idx,
                                next_frame_envelope=next_frame_envelope,
                                noise_rms=noise_rms)

            tau_ms   = decay['tau_ms']
            r2_log   = decay['r_squared_log']
            E_W1     = decay['E_W1']
            E_W4     = decay['E_W4']

            # ── Criterion: peak_ifft > threshold ────────────────────────────
            peak_pass = (peak_amp > min_peak_ifft_v)

            # ── C2: pre_snr < max_pre_snr ───────────────────────────────────
            pre_end       = max(0, peak_idx - GUARD)
            n_pre_current = pre_end

            if n_pre_current >= MIN_PRE_SAMPLES:
                pre_window = signal[:pre_end]
                pre_source = "current frame"
            elif frame_idx > 0:
                prev_sig = self._reconstruct_ifft(frame_idx - 1, normalized=True)
                if prev_sig is not None:
                    pre_window = np.concatenate([prev_sig[-200:],
                                                 signal[:pre_end] if pre_end > 0 else np.array([])])
                    pre_source = f"prev[-200:]+current[:{pre_end}]"
                else:
                    pre_window = signal[:pre_end] if pre_end > 0 else np.array([noise_rms])
                    pre_source = "current only"
            else:
                pre_window = np.array([noise_rms])
                pre_source = "first frame fallback"

            rms_pre = float(np.sqrt(np.mean(pre_window ** 2))) if len(pre_window) > 0 else noise_rms
            pre_snr = rms_pre / noise_rms if noise_rms > 0 else 1.0
            c2_pass = (pre_snr < max_pre_snr)

            # ── C3: E_W1 > 2× E_W4 ─────────────────────────────────────────
            c3_pass = (E_W1 > E_W4 * MIN_DECAY_RATIO)

            # ── asym: rise_samples / fall_samples < max_asym ────────────────
            LEVEL_FRACTION = 0.10
            FALL_SEARCH    = 40
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

            # ── tau range ────────────────────────────────────────────────────
            if tau_ms > 0:
                tau_pass = (tau_min <= tau_ms <= tau_max)
            else:
                tau_pass = False   # N/A → fail

            # ── R² ───────────────────────────────────────────────────────────
            r2_pass = (r2_log > min_r2)

            # ── Final verdict ────────────────────────────────────────────────
            all_pass = peak_pass and c2_pass and c3_pass and asym_pass and tau_pass and r2_pass

            print(f"   Frame {frame_idx:6d}: "
                  f"peak={'✅' if peak_pass else '❌'}"
                  f"({peak_amp*1e6:.0f}µV)  "
                  f"C2={'✅' if c2_pass else '❌'}"
                  f"(pre_snr={pre_snr:.2f})  "
                  f"C3={'✅' if c3_pass else '❌'}"
                  f"({E_W1/E_W4 if E_W4>0 else 0:.1f}×)  "
                  f"asym={'✅' if asym_pass else '❌'}"
                  f"({asym_ratio:.2f})  "
                  f"τ={'✅' if tau_pass else '❌'}"
                  f"({tau_ms:.2f}ms)  "
                  f"R²={'✅' if r2_pass else '❌'}"
                  f"({r2_log:.2f})  "
                  f"→ {'PASS' if all_pass else 'FAIL'}")

            if not all_pass:
                continue

            candidates_stage3.append({
                **candidate,
                'peak_amp':    peak_amp,
                'peak_idx':    peak_idx,
                'decay_results': decay,
                'r2_log':      r2_log,
                'tau_ms':      tau_ms,
                'slope_log':   decay['slope_log'],
                'snr':         peak_amp / noise_rms if noise_rms > 0 else 0.0,
                'pre_snr':     pre_snr,
                'rms_pre':     rms_pre,
                'pre_source':  pre_source,
                'E_W1':        E_W1,
                'E_W4':        E_W4,
                'asym_ratio':  asym_ratio,
                'noise_rms':   noise_rms,
                'confirmed':   True,
                'classification': "✅ CONFIRMED",
            })

        pct3 = (len(candidates_stage3) / len(candidates_stage2) * 100) if candidates_stage2 else 0
        print(f"✅ Stage 3: {len(candidates_stage3)}/{len(candidates_stage2)} passed ({pct3:.1f}%)")

        if not candidates_stage3:
            progress.close()
            QMessageBox.information(self, "No Clicks Found",
                                    "No frames passed Stage 3 iFFT validation.\n"
                                    "Try relaxing the criteria parameters.")
            return

        # ====================================================================
        # STAGE 4: DEDUPLICATION
        # ====================================================================
        print(f"\n{'='*60}\nSTAGE 4: DEDUPLICATION\n{'='*60}")
        final_clicks = self._deduplicate_clicks(candidates_stage3)
        progress.setValue(100)
        progress.close()

        print(f"✅ Stage 4: {len(final_clicks)} unique clicks")

        self._populate_results_table(final_clicks)
        self.detected_clicks = final_clicks
        self.export_button.setEnabled(True)

        # Cache on data_manager for batch export
        self.data_manager._detected_clicks    = final_clicks
        self.data_manager._cached_threshold_v = threshold_v

        print(f"\n{'='*80}")
        print(f"🎉 DETECTION COMPLETE: {len(final_clicks)} CLICKS FOUND")
        print(f"{'='*80}\n")

        QMessageBox.information(
            self, "Detection Complete",
            f"Found {len(final_clicks)} ultrasonic clicks!\n\n"
            f"Stage 1 (Energy+Groups): {len(candidates_stage1)} candidates\n"
            f"Stage 2 (FFT filters):   {len(candidates_stage2)} candidates\n"
            f"Stage 3 (iFFT criteria): {len(candidates_stage3)} candidates\n"
            f"Stage 4 (Deduplicated):  {len(final_clicks)} unique clicks"
        )

    # =========================================================================
    # HELPERS
    # =========================================================================
    def _deduplicate_clicks(self, candidates):
        """Keep highest-amplitude frame within each group of consecutive detections."""
        if not candidates:
            return []
        MAX_GAP = 3
        sorted_c = sorted(candidates, key=lambda x: x['frame_idx'])
        groups, current = [], [sorted_c[0]]
        for i in range(1, len(sorted_c)):
            if sorted_c[i]['frame_idx'] - sorted_c[i-1]['frame_idx'] <= MAX_GAP:
                current.append(sorted_c[i])
            else:
                groups.append(current)
                current = [sorted_c[i]]
        groups.append(current)

        unique = []
        for grp in groups:
            best = max(grp, key=lambda x: x['peak_amp'])
            if len(grp) > 1:
                print(f"   🔗 Dedup group: {len(grp)} frames → kept #{best['frame_idx']}")
            unique.append(best)
        return unique

    def _populate_results_table(self, clicks):
        self.results_table.setRowCount(len(clicks))
        for row, click in enumerate(clicks):
            frame_idx = click['frame_idx']
            ts = (frame_idx * self.data_manager.frame_duration_ms) / 1000.0

            E_W4 = click.get('E_W4', 1e-12)
            decay_ratio = click.get('E_W1', 0) / E_W4 if E_W4 > 1e-12 else 999.0
            tau = click.get('tau_ms', -1.0)
            tau_str = f"{tau:.3f}" if tau > 0 else "N/A"

            items = [
                f"{ts:.3f} s",
                f"{click['peak_fft_v']*1000:.3f}",
                f"{click['peak_amp']*1e6:.1f}",
                f"{click['spr']:.1f}",
                f"{click.get('pre_snr', 0.0):.2f}",
                f"{decay_ratio:.1f}",
                f"{click.get('asym_ratio', 0.0):.3f}",
                tau_str,
                f"{click.get('r2_log', 0.0):.3f}",
                f"{click.get('group_size', 1)}",
            ]
            for col, text in enumerate(items):
                self.results_table.setItem(row, col, QTableWidgetItem(text))

            notes = []
            if click.get('decay_results', {}).get('used_next_frame', False):
                notes.append("multi-frame")
            if click.get('decay_results', {}).get('near_end', False):
                notes.append("near-end")
            self.results_table.setItem(row, 10, QTableWidgetItem(", ".join(notes)))

    def export_results(self):
        """Export results to CSV."""
        from PySide6.QtWidgets import QFileDialog
        import csv

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Click Detection Results", "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if not filename:
            return

        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp (s)", "Frame Index", "Group Size",
                    "Peak FFT norm (mV)", "Peak iFFT norm (µV)",
                    "SPR", "pre_snr", "rms_pre (mV)",
                    "E_W1", "E_W4", "Decay ratio",
                    "asym", "τ (ms)", "R²", "SNR", "Notes"
                ])
                for click in self.detected_clicks:
                    frame_idx = click['frame_idx']
                    ts = (frame_idx * self.data_manager.frame_duration_ms) / 1000.0
                    tau = click.get('tau_ms', -1.0)
                    E_W4 = click.get('E_W4', 1e-12)
                    notes = []
                    if click.get('decay_results', {}).get('used_next_frame', False):
                        notes.append("multi-frame")
                    if click.get('decay_results', {}).get('near_end', False):
                        notes.append("near-end")
                    if click.get('pre_source', ''):
                        notes.append(f"pre:{click['pre_source']}")
                    writer.writerow([
                        f"{ts:.3f}", frame_idx,
                        click.get('group_size', 1),
                        f"{click['peak_fft_v']*1000:.4f}",
                        f"{click['peak_amp']*1e6:.2f}",
                        f"{click.get('spr', 0.0):.1f}",
                        f"{click.get('pre_snr', 0.0):.3f}",
                        f"{click.get('rms_pre', 0.0)*1000:.4f}",
                        f"{click.get('E_W1', 0.0):.6e}",
                        f"E_W4:.6e",
                        f"{click.get('E_W1', 0.0)/E_W4 if E_W4>1e-12 else 999:.1f}",
                        f"{click.get('asym_ratio', 0.0):.3f}",
                        f"{tau:.3f}" if tau > 0 else "N/A",
                        f"{click.get('r2_log', 0.0):.4f}",
                        f"{click.get('snr', 0.0):.1f}",
                        "; ".join(notes),
                    ])
            QMessageBox.information(self, "Export Successful", f"Results exported to:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Error writing file:\n{str(e)}")
