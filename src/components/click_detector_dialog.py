"""
Click Detector Algorithm Dialog - Rilevamento automatico click ultrasonici

Implementa la pipeline algoritmica a 4 stadi descritta in:
docs/click_detector_algorithm_strategy.md

PIPELINE (v3.1 - Updated Validation Criteria):
1. Stage 1: Energy threshold (μ + Nσ)
2. Stage 2: SPR broadband filter + Spectral ratio R = E_low/E_high (descriptive)
3. Stage 3: Five-criterion validation:
   - Criterion 1: Temporal SNR > 5.0 (peak_amplitude / noise_rms)
   - Criterion 2: pre_snr < 3.0 (RMS(pre_window) / noise_rms - silence before click)
                  Window: Case A (peak≥60→current frame), Case B (peak<60→prev[-200:]+current),
                          Case C (first frame→fallback pre_snr=1.0)
                  Raised from 2.0: leaf turbulence before impact can legitimately elevate pre-RMS.
   - Criterion 3: Global decay: E_W1 > E_W4 (first > last sub-window)
   - Criterion 4: Asymmetry: rise/fall ratio < 0.5 (rejects symmetric spikes ~0.3 ms)
   - Criterion 5: Clean tail: no secondary burst > 3× valley (rejects ringing/multi-burst)
4. Stage 4: Deduplication

PARAMETRI CONFIGURABILI:
- Threshold energia: μ + Nσ (default: N=4, step=σ)
- Normalizzazione 50%: On/Off
- SNR minimo: 3.0-10.0 (default: 5.0)
- max_pre_snr: 1.0-10.0 (default: 3.0)

REMOVED IN v3.0:
- R² spectral/decay thresholds (now descriptive features only)
- R as validation criterion (now descriptive feature only)

FEATURE UPDATES:
- tau_ms, r2_log, R: saved as DESCRIPTIVE features for statistical analysis
- Offline noise estimation for accurate SNR calculation
- pre_snr replaces PRE_ratio (absolute reference via noise_rms, handles early-peak edge case)
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

# Import funzioni esistenti
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from windows.replay_window_audio import compute_hilbert_envelope, find_peak, check_decay, estimate_noise_offline, suppress_edge_artifacts
from core.replay_base_window import compute_fft_energy


class ClickDetectorDialog(QDialog):
    """Dialog per configurazione e esecuzione algoritmo rilevamento click"""
    
    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        
        self.data_manager = data_manager
        self.parent = parent
        
        # Risultati analisi
        self.detected_clicks = []
        
        # Setup UI
        self.setWindowTitle("Automatic Click Detector")
        self.setMinimumSize(900, 700)
        
        self.setup_ui()
        self.load_default_parameters()
        
        # Applica tema
        if parent and hasattr(parent, 'theme_manager'):
            saved_theme = parent.theme_manager.load_saved_theme()
            parent.theme_manager.apply_theme(self, saved_theme)
            # se il tema è light... allora applica stylesheet per garantire contrasto
            if 'light' in saved_theme.lower():
                self.setStyleSheet("""
                    QDialog { background-color: white;}
                """)
    
    def setup_ui(self):
        """Crea l'interfaccia utente"""
        layout = QVBoxLayout(self)
        
        # TITOLO
        title = QLabel("<b style='font-size:16pt;'>Automatic Click Detector Algorithm</b>")
        title.setAlignment(Qt.AlignLeft)
        layout.addWidget(title)
        
        subtitle = QLabel("<i>4-stage pipeline: Energy → Spectral Ratio → Decay Analysis → Deduplication</i>")
        subtitle.setAlignment(Qt.AlignLeft)
        layout.addWidget(subtitle)
        
        # PARAMETRI CONFIGURABILI
        params_group = QGroupBox("Detection Parameters")
        params_layout = QFormLayout()
        
        # 1. Energy Threshold
        threshold_layout = QHBoxLayout()
        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setDecimals(3)
        self.threshold_spinbox.setRange(0, 10000)
        self.threshold_spinbox.setSuffix(" mV")
        self.threshold_spinbox.setToolTip("Stage 1: Energy threshold for candidate selection\nDefault: μ + 4σ")
        threshold_layout.addWidget(self.threshold_spinbox)
        
        self.threshold_sigma_label = QLabel("(μ + 4.0σ)")
        self.threshold_sigma_label.setStyleSheet("color: gray;")
        threshold_layout.addWidget(self.threshold_sigma_label)
        threshold_layout.addStretch()
        
        params_layout.addRow("Energy Threshold (Stage 1):", threshold_layout)
        
        # 2. Normalizzazione
        self.normalize_checkbox = QCheckBox("Apply 50% microphone correction")
        self.normalize_checkbox.setChecked(True)
        self.normalize_checkbox.setToolTip("Stage 2: Apply frequency response normalization\nRecommended: ON for accurate spectral ratio")
        params_layout.addRow("Use Normalization:", self.normalize_checkbox)
        
        # 3. SNR minimo
        self.snr_min_spinbox = QDoubleSpinBox()
        self.snr_min_spinbox.setDecimals(1)
        self.snr_min_spinbox.setRange(3.0, 10.0)
        self.snr_min_spinbox.setSingleStep(0.5)
        self.snr_min_spinbox.setValue(5.0)
        self.snr_min_spinbox.setToolTip("Stage 3: Minimum SNR for click validation\nRecommended: 5.0")
        params_layout.addRow("Min. SNR (Criterion 1):", self.snr_min_spinbox)
        
        # 4. Max Spectral Peak Ratio (Stage 2 filter)
        self.max_spr_spinbox = QDoubleSpinBox()
        self.max_spr_spinbox.setDecimals(0)
        self.max_spr_spinbox.setRange(5.0, 200.0)
        self.max_spr_spinbox.setSingleStep(5.0)
        self.max_spr_spinbox.setValue(30.0)
        self.max_spr_spinbox.setToolTip(
            "Stage 2 – Spectral Peak Ratio filter (broadband check)\n"
            "SPR = max(|X[k]|²) / mean(|X[k]|²)  over 154 bins (20-80 kHz)\n\n"
            "  Click broadband (≥15 kHz band): SPR ≈ 4–15\n"
            "  Narrow-band tone / sinusoid:    SPR ≈ 50–150\n\n"
            "Candidates with SPR > threshold are rejected as tonal noise.\n"
            "Recommended: 30  (safe margin between clicks and pure tones)"
        )
        params_layout.addRow("Max. SPR (Stage 2 – broadband):", self.max_spr_spinbox)

        # 5. Max pre-click SNR (Criterion 2)
        self.max_pre_snr_spinbox = QDoubleSpinBox()
        self.max_pre_snr_spinbox.setDecimals(1)
        self.max_pre_snr_spinbox.setRange(1.0, 10.0)
        self.max_pre_snr_spinbox.setSingleStep(0.5)
        self.max_pre_snr_spinbox.setValue(3.0)
        self.max_pre_snr_spinbox.setToolTip(
            "Stage 3 – Criterion 2: Max pre-click noise level (relative to noise floor)\n"
            "pre_snr = RMS(signal before peak) / noise_rms\n"
            "  ≈ 1.0 → pure silence before click  (ideal)\n"
            "  ≈ 2.0 → slight background activity\n"
            "  ≈ 3.0 → moderate turbulence before click  (default)\n"
            "  > 5.0 → continuous noise, likely not a click\n"
            "Recommended: 3.0  (raised from 2.0 — leaf turbulence before impact\n"
            "               can legitimately raise pre-click RMS to 2–3×noise)"
        )
        params_layout.addRow("Max. pre-click SNR (Criterion 2):", self.max_pre_snr_spinbox)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # INFO BOX
        info_group = QGroupBox("File Information")
        info_layout = QFormLayout()
        
        self.info_duration = QLabel("0.0 s")
        self.info_frames = QLabel("0")
        self.info_mean_energy = QLabel("0.000 mV")
        self.info_std_energy = QLabel("0.000 mV")
        
        info_layout.addRow("Total duration:", self.info_duration)
        info_layout.addRow("Total frames:", self.info_frames)
        info_layout.addRow("Mean energy (μ):", self.info_mean_energy)
        info_layout.addRow("Std deviation (σ):", self.info_std_energy)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # TABELLA RISULTATI
        results_label = QLabel("<b>Detected Clicks</b>")
        layout.addWidget(results_label)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(12)
        self.results_table.setHorizontalHeaderLabels([
            "Timestamp", "Amplitude", "SNR", "pre_snr", "E_W1/E_W4", "SPR",
            "Energy FFT", "Ratio R", "R² (log)", "τ (ms)", "Classification", "Notes"
        ])
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setAlternatingRowColors(True)
        
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)   # Timestamp
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)   # Amplitude
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)   # SNR
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)   # pre_snr
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)   # E_W1/E_W4
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)   # SPR
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)   # Energy FFT
        header.setSectionResizeMode(7, QHeaderView.ResizeToContents)   # Ratio R
        header.setSectionResizeMode(8, QHeaderView.ResizeToContents)   # R² (log)
        header.setSectionResizeMode(9, QHeaderView.ResizeToContents)   # τ (ms)
        header.setSectionResizeMode(10, QHeaderView.ResizeToContents)  # Classification
        header.setSectionResizeMode(11, QHeaderView.Stretch)           # Notes
        
        self.results_table.verticalHeader().setVisible(False)
        layout.addWidget(self.results_table)
        
        # PULSANTI
        button_layout = QHBoxLayout()
        
        self.run_button = QPushButton("▶ Run Detection")
        self.run_button.setMinimumHeight(40)
        self.run_button.setStyleSheet("font-size: 12pt; font-weight: bold;")
        self.run_button.clicked.connect(self.run_detection)
        button_layout.addWidget(self.run_button)
        
        self.export_button = QPushButton("Export Results...")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_results)
        button_layout.addWidget(self.export_button)
        
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
    
    def load_default_parameters(self):
        """Carica parametri di default dai dati"""
        if not hasattr(self.data_manager, 'fft_mean'):
            QMessageBox.warning(self, "Warning", "File statistics not available.\nPlease load a .paudio file first.")
            return
        
        # Calcola threshold di default: μ + 4σ
        mean_mv = self.data_manager.fft_mean * 1000  # V → mV
        std_mv = self.data_manager.fft_std * 1000
        
        default_threshold = mean_mv + 4 * std_mv
        
        # Imposta valori
        self.threshold_spinbox.setValue(default_threshold)
        self.threshold_spinbox.setSingleStep(std_mv)
        
        # Aggiorna info
        self.info_duration.setText(f"{self.data_manager.total_duration_sec:.1f} s")
        self.info_frames.setText(f"{self.data_manager.total_frames}")
        self.info_mean_energy.setText(f"{mean_mv:.3f} mV")
        self.info_std_energy.setText(f"{std_mv:.3f} mV")
        
        # Aggiorna label sigma
        sigma_multiplier = (default_threshold - mean_mv) / std_mv if std_mv > 0 else 4.0
        self.threshold_sigma_label.setText(f"(μ + {sigma_multiplier:.1f}σ)")
        
        # Connetti spinbox per aggiornamento automatico
        self.threshold_spinbox.valueChanged.connect(self.update_sigma_label)
    
    def update_sigma_label(self):
        """Aggiorna label con moltiplicatore σ"""
        if not hasattr(self.data_manager, 'fft_mean'):
            return
        
        mean_mv = self.data_manager.fft_mean * 1000
        std_mv = self.data_manager.fft_std * 1000
        
        if std_mv > 0:
            current_threshold = self.threshold_spinbox.value()
            sigma_mult = (current_threshold - mean_mv) / std_mv
            self.threshold_sigma_label.setText(f"(μ + {sigma_mult:.1f}σ)")
    
    def run_detection(self):
        """Esegue la pipeline di rilevamento a 4 stadi"""
        print("\n" + "="*80)
        print("🔍 AUTOMATIC CLICK DETECTOR - PIPELINE EXECUTION")
        print("="*80)
        
        # Reset risultati
        self.detected_clicks = []
        self.results_table.setRowCount(0)
        
        # Parametri
        threshold_mv = self.threshold_spinbox.value()
        threshold_v = threshold_mv / 1000.0
        use_normalization = self.normalize_checkbox.isChecked()
        max_spr = self.max_spr_spinbox.value()
        snr_min = self.snr_min_spinbox.value()
        max_pre_snr = self.max_pre_snr_spinbox.value()
        
        print(f"\n📋 PARAMETERS:")
        print(f"   Energy threshold: {threshold_mv:.3f} mV ({threshold_v:.6f} V)")
        print(f"   Normalization: {'ON (50% correction)' if use_normalization else 'OFF'}")
        print(f"   Max. SPR (Stage 2 broadband filter): {max_spr:.0f}")
        print(f"   Min. SNR (Criterion 1): {snr_min:.1f}")
        print(f"   Max. pre-click SNR (Criterion 2): {max_pre_snr:.1f}")
        
        total_frames = self.data_manager.total_frames
        
        # Progress dialog
        progress = QProgressDialog("Running click detection...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        
        # ========================================================================
        # OFFLINE NOISE ESTIMATION (before all stages)
        # ========================================================================
        print(f"\n{'='*80}")
        print("OFFLINE NOISE ESTIMATION")
        print(f"{'='*80}")
        
        progress.setLabelText("Estimating noise from empty frames...")
        progress.setValue(5)
        
        noise_info = estimate_noise_offline(self.data_manager, energy_threshold_multiplier=4.0, max_samples=500)
        noise_rms = noise_info['noise_rms']
        
        # Cache noise_rms on data_manager so IFFTWindow dialog can compute SNR interactively
        self.data_manager._cached_noise_rms = noise_rms
        
        print(f"✅ Noise RMS: {noise_rms*1000:.6f} mV (from {noise_info['n_samples']} empty frames)")
        
        # ========================================================================
        # STAGE 1: ENERGY THRESHOLD
        # ========================================================================
        print(f"\n{'='*80}")
        print("STAGE 1: ENERGY THRESHOLD SELECTION")
        print(f"{'='*80}")
        
        candidates_stage1 = []
        
        for i in range(total_frames):
            if i % 10000 == 0:
                progress.setValue(int((i / total_frames) * 25))
                if progress.wasCanceled():
                    return
            
            energy = self.data_manager.fft_means[i]
            
            if energy > threshold_v:
                candidates_stage1.append(i)
        
        print(f"✅ Stage 1 complete: {len(candidates_stage1)}/{total_frames} frames passed ({len(candidates_stage1)/total_frames*100:.3f}%)")
        
        if len(candidates_stage1) == 0:
            progress.close()
            QMessageBox.information(self, "No Clicks Found", "No frames exceeded the energy threshold.\nTry lowering the threshold.")
            return
        
        # ========================================================================
        # STAGE 2: SPECTRAL PEAK RATIO FILTER (broadband check)
        # ========================================================================
        # SPR = max(|X[k]|²) / mean(|X[k]|²)  over all 154 bins (20-80 kHz)
        #
        # Physical basis:
        #   A broadband click distributes energy across many bins → SPR low (4-15)
        #   A pure tone / narrow-band noise concentrates all energy in 1-3 bins → SPR high (50-150)
        #
        # This filter is FAST (operates on FFT magnitudes, no iFFT needed) and
        # AMPLITUDE-INVARIANT: SPR depends only on spectral shape, not signal level.
        # ========================================================================
        print(f"\n{'='*80}")
        print("STAGE 2: SPECTRAL PEAK RATIO FILTER (broadband check)")
        print(f"{'='*80}")
        print(f"   SPR = max(|X[k]|²) / mean(|X[k]|²)  threshold: {max_spr:.0f}")
        print(f"   Expected: click broadband (≥15 kHz band) SPR≈4–15 | pure tone SPR≈50–150")

        candidates_stage2 = []
        spr_rejected = 0

        for idx, frame_idx in enumerate(candidates_stage1):
            if idx % 100 == 0:
                progress.setValue(25 + int((idx / len(candidates_stage1)) * 15))
                if progress.wasCanceled():
                    return

            # Ottieni FFT magnitudes
            fft_mags = self.data_manager.fft_data[frame_idx]

            # Applica normalizzazione se richiesto
            if use_normalization:
                fft_mags = self._normalize_fft(fft_mags)

            # ── Spectral Peak Ratio (SPR) ──────────────────────────────────────
            fft_power = fft_mags ** 2
            mean_power = np.mean(fft_power)
            max_power  = np.max(fft_power)
            spr = max_power / mean_power if mean_power > 1e-20 else 0.0
            spr_pass = (spr <= max_spr)

            if not spr_pass:
                spr_rejected += 1
                continue   # Reject: tonal / narrowband noise

            # ── Spectral ratio R (descriptive only) ───────────────────────────
            energies = compute_fft_energy(fft_mags)
            ratio = energies['low'] / energies['high'] if energies['high'] > 0 else 0.0

            candidates_stage2.append({
                'frame_idx': frame_idx,
                'energies': energies,
                'ratio': ratio,   # descriptive
                'spr': spr,       # saved for results table
            })

        print(f"✅ Stage 2 complete: {len(candidates_stage2)}/{len(candidates_stage1)} passed "
              f"({spr_rejected} rejected by SPR > {max_spr:.0f})")

        
        # ========================================================================
        # STAGE 3: THREE-CRITERION TEMPORAL VALIDATION (v3.0)
        # ========================================================================
        print(f"\n{'='*80}")
        print("STAGE 3: TEMPORAL VALIDATION (5 CRITERIA)")
        print(f"{'='*80}")
        print("Criterion 1: SNR > {:.1f} (peak_amplitude / noise_rms)".format(snr_min))
        print("Criterion 2: pre_snr < {:.1f} (RMS_pre / noise_rms - silence before click)".format(max_pre_snr))
        print("Criterion 3: Global decay (E_W1 > 2× E_W4, min ratio 2.0)")
        print("Criterion 4: Narrow-spike test – rejects isolated symmetric spikes (EMI/artefacts)")
        print("Criterion 5: Clean tail – no secondary burst > 3× valley (rejects ringing/multi-burst)")

        # Constants for Criteria 3, 4, and 5 (defined here for visibility in summary prints)
        MIN_DECAY_RATIO   = 2.0   # C3: E_W1 must be at least this many times E_W4 (rejects near-flat signals)
        ASYM_THRESHOLD    = 0.5   # C4 step-1: rise/fall ratio – spike must be symmetric at 10%·peak
        LEVEL_FRACTION    = 0.10  # C4: amplitude level for rise/fall measurement
        FALL_SEARCH       = 40    # C4 step-1: max fall-search window (samples, 0.2 ms) — shorter than before
        SPIKE_HALF_WIN    = 40    # C4 step-2: half-window for narrow-spike check (0.2 ms each side)
        SPIKE_NOISE_FACTOR= 3.0   # C4 step-2: flanks near noise if max < SPIKE_NOISE_FACTOR × noise_rms
        DECAY_SKIP        = 10    # C5: skip main lobe before valley search
        TAIL_WINDOW       = 300   # C5: post-peak tail window (samples, 1.5 ms) — matches
                                  #     check_decay() window_samples for consistency
        REBOUND_FACTOR    = 3.0   # C5: secondary peak must be > factor × valley
        
        candidates_stage3 = []
        
        for idx, candidate in enumerate(candidates_stage2):
            if idx % 10 == 0:
                progress.setValue(40 + int((idx / len(candidates_stage2)) * 45))
                if progress.wasCanceled():
                    return
            
            frame_idx = candidate['frame_idx']
            
            # Ricostruisci iFFT
            signal = self._reconstruct_ifft(frame_idx, use_normalization)
            
            if signal is None:
                continue
            
            # Calcola envelope
            envelope = compute_hilbert_envelope(signal)
            
            # ✅ BUG FIX: Trova il picco sull'ENVELOPE (non sul segnale raw).
            # check_decay() riceve l'envelope e usa peak_idx come indice su di esso.
            # Usare il raw potrebbe puntare a un campione di portante, non al vero picco.
            peak_idx, peak_amp = find_peak(envelope)
            
            # Check decay (con gestione spill) - RETURNS DESCRIPTIVE FEATURES ONLY
            next_frame_signal = None
            if peak_idx > 212 and frame_idx + 1 < total_frames:  # matches near_end threshold in check_decay
                next_frame_signal = self._reconstruct_ifft(frame_idx + 1, use_normalization)
            
            decay_results = check_decay(envelope, peak_idx, next_frame_signal=next_frame_signal)
            
            # Extract features for validation
            r2_log = decay_results['r_squared_log']  # Descriptive only
            tau_ms = decay_results['tau_ms']  # Descriptive only
            slope_log = decay_results['slope_log']
            E_W1 = decay_results['E_W1']
            E_W4 = decay_results['E_W4']
            
            # ================================================================
            # CRITERION 1: Temporal SNR > threshold
            # ================================================================
            snr = peak_amp / noise_rms if noise_rms > 0 else 0.0
            criterion_1_pass = (snr > snr_min)
            
            # ================================================================
            # CRITERION 2: pre_snr < max_pre_snr (silence before click)
            # ================================================================
            # Measures whether the signal before the peak is compatible with
            # the noise floor. pre_snr = RMS(pre-window) / noise_rms.
            #   ≈ 1.0 → pure silence (ideal click)
            #   > 2.0 → significant energy before click → likely not a click
            #
            # PRE WINDOW STRATEGY:
            # We need at least MIN_PRE_SAMPLES=50 samples (~250 µs) for a
            # statistically reliable RMS estimate.
            #
            # Case A – peak_idx >= 60: enough room in current frame.
            #   pre_window = signal[0 : peak_idx - 10]
            #
            # Case B – peak_idx < 60 AND frame_idx > 0: extend into previous frame.
            #   Use the last 200 samples of the previous frame (1 ms, certain silence
            #   since that frame didn't pass Stage 1 energy threshold).
            #   pre_window = concat(prev[-200:], signal[0 : peak_idx - 10])
            #
            # Case C – peak_idx < 60 AND frame_idx == 0: no previous frame.
            #   Fallback: assume pre_snr = 1.0 (pass), rely on Criterion 1 SNR.

            MIN_PRE_SAMPLES = 50   # minimum samples for reliable RMS (~250 µs)
            GUARD = 20              # samples before peak to exclude click rise

            pre_end = max(0, peak_idx - GUARD)
            n_pre_current = pre_end  # samples available in current frame before peak

            if n_pre_current >= MIN_PRE_SAMPLES:
                # Case A: enough samples in current frame
                pre_window = signal[:pre_end]
                pre_source = "current frame"

            elif frame_idx > 0:
                # Case B: extend into previous frame (last 200 samples = 1 ms)
                # The previous frame did NOT pass Stage 1 → it is guaranteed
                # to be below the energy threshold → safe noise reference.
                prev_signal = self._reconstruct_ifft(frame_idx - 1, use_normalization)
                if prev_signal is not None:
                    prev_tail = prev_signal[-200:]   # last 1 ms of previous frame
                    current_pre = signal[:pre_end] if pre_end > 0 else np.array([])
                    pre_window = np.concatenate([prev_tail, current_pre])
                    pre_source = f"prev[-200:] + current[:{pre_end}]"
                else:
                    # Reconstruction failed: use whatever we have in current frame
                    pre_window = signal[:pre_end] if pre_end > 0 else np.array([noise_rms])
                    pre_source = "current frame only (prev recon failed)"

            else:
                # Case C: first frame of file — no previous frame available.
                # Conservatively assume silence → pre_snr = 1.0 → PASS.
                # Criterion 1 (SNR) is the primary guard in this edge case.
                pre_window = np.array([noise_rms])   # synthetic: exactly noise level
                pre_source = "first frame fallback"

            rms_pre = np.sqrt(np.mean(pre_window ** 2)) if len(pre_window) > 0 else noise_rms
            pre_snr = rms_pre / noise_rms if noise_rms > 0 else 1.0
            criterion_2_pass = (pre_snr < max_pre_snr)

            print(f"      C2: pre_snr={pre_snr:.2f} (RMS_pre={rms_pre*1000:.4f} mV, "
                  f"n={len(pre_window)} samples, src={pre_source})"
                  f" → {'PASS' if criterion_2_pass else 'FAIL'}")
            
            # ================================================================
            # CRITERION 3: Global energy decay (E_W1 > MIN_DECAY_RATIO × E_W4)
            # ================================================================
            criterion_3_pass = (E_W1 > E_W4 * MIN_DECAY_RATIO)

            # ================================================================
            # CRITERION 4: Narrow-spike test – rejects isolated symmetric spikes
            # ================================================================
            # A real click is highly asymmetric: very fast rise, slow exponential
            # fall. An EMI spike or mechanical artefact looks like a narrow,
            # isolated pulse: symmetric AND both flanks drop to near the noise
            # floor within 0.2 ms.
            #
            # Two-step test (both must be true to REJECT):
            #
            # Step 1 – Symmetry check (same as before, window now 0.2 ms = 40 samp):
            #   rise_samples = samples from first crossing of 10%·peak back to peak
            #   fall_samples = samples from peak until envelope < 10%·peak
            #                  (capped at FALL_SEARCH = 40 samples = 0.2 ms)
            #   → symmetric if rise_samples / fall_samples ≥ ASYM_THRESHOLD (0.5)
            #
            # Step 2 – Flank-to-noise check:
            #   Look at the 0.2 ms window to the LEFT of rise_start and to the
            #   RIGHT of the fall-crossing point (or peak + FALL_SEARCH if no
            #   crossing found). If both flanks stay below SPIKE_NOISE_FACTOR ×
            #   noise_rms, the pulse is narrow and isolated → spike.
            #
            # REJECT if Step1 AND Step2 are both true.
            # PASS   if either fails (signal is asymmetric OR has a broad base).
            #
            # Real click:  fast rise, very slow fall (>0.2 ms) → fall_samples=FALL_SEARCH
            #              → Step 2 right-flank is still above noise → PASS
            # EMI spike:   rise ≈ fall ≈ 5-20 samp, both flanks at noise → FAIL
            level = peak_amp * LEVEL_FRACTION

            # --- Step 1: symmetry ---
            rise_start = peak_idx
            for i in range(peak_idx - 1, -1, -1):
                if envelope[i] < level:
                    rise_start = i + 1
                    break
            rise_samples = max(1, peak_idx - rise_start)

            fall_end_idx = min(peak_idx + FALL_SEARCH, len(envelope))
            fall_samples = FALL_SEARCH      # default: still above level at end of window
            fall_cross_idx = peak_idx + FALL_SEARCH  # where the fall ends (or cap)
            for i in range(peak_idx + 1, fall_end_idx):
                if envelope[i] < level:
                    fall_samples = i - peak_idx
                    fall_cross_idx = i
                    break

            asymmetry_ratio = rise_samples / fall_samples if fall_samples > 0 else 1.0
            is_symmetric = (asymmetry_ratio >= ASYM_THRESHOLD)

            # --- Step 2: flank-to-noise check ---
            # Left flank: SPIKE_HALF_WIN samples before rise_start
            left_start = max(0, rise_start - SPIKE_HALF_WIN)
            left_flank = envelope[left_start:rise_start]
            left_near_noise = (len(left_flank) == 0 or
                               float(np.max(left_flank)) < SPIKE_NOISE_FACTOR * noise_rms)

            # Right flank: SPIKE_HALF_WIN samples after fall crossing
            right_end = min(fall_cross_idx + SPIKE_HALF_WIN, len(envelope))
            right_flank = envelope[fall_cross_idx:right_end]
            right_near_noise = (len(right_flank) == 0 or
                                float(np.max(right_flank)) < SPIKE_NOISE_FACTOR * noise_rms)

            flanks_near_noise = left_near_noise and right_near_noise

            # Reject only if BOTH conditions are met
            criterion_4_pass = not (is_symmetric and flanks_near_noise)

            print(f"      C4: rise={rise_samples} fall={fall_samples} asym={asymmetry_ratio:.3f} "
                  f"sym={is_symmetric} flanks_noise={flanks_near_noise}"
                  f" → {'PASS' if criterion_4_pass else 'FAIL (narrow symmetric spike)'}")

            # ================================================================
            # CRITERION 5: Clean tail – rejects ringing / multi-burst signals
            # ================================================================
            # A real click decays towards the noise floor without significant
            # secondary bursts. Interference type 2 shows one or more secondary
            # peaks after an initial decay.
            #
            # Method (robust to non-perfectly-monotone real clicks):
            #   1. Find the local minimum of the envelope in the window
            #      [peak_idx + DECAY_SKIP : peak_idx + TAIL_WINDOW]
            #   2. Check that no sample after that minimum exceeds
            #      min_local × REBOUND_FACTOR
            #
            # REBOUND_FACTOR = 3.0 → a secondary burst must be at least 3× the
            # local valley to be flagged. This tolerates gentle undulations in
            # real click envelopes while catching clear secondary peaks.
            tail_start = peak_idx + DECAY_SKIP
            tail_end   = min(peak_idx + TAIL_WINDOW, len(envelope))

            if tail_end > tail_start + 5:
                tail_env = envelope[tail_start:tail_end]
                # Find valley (minimum) in the tail
                valley_idx_local = int(np.argmin(tail_env))
                valley_val = float(tail_env[valley_idx_local])

                # Check for rebound after the valley
                post_valley = tail_env[valley_idx_local + 1:]
                rebound_threshold = valley_val * REBOUND_FACTOR
                has_rebound = (len(post_valley) > 0 and
                               float(np.max(post_valley)) > rebound_threshold and
                               rebound_threshold > level)   # only meaningful above noise

                criterion_5_pass = not has_rebound
                rebound_max = float(np.max(post_valley)) if len(post_valley) > 0 else 0.0
                print(f"      C5: valley={valley_val*1000:.4f} mV  rebound_max={rebound_max*1000:.4f} mV"
                      f"  threshold={rebound_threshold*1000:.4f} mV"
                      f" → {'PASS' if criterion_5_pass else 'FAIL (secondary burst)'}")
            else:
                # Tail too short to evaluate (near_end case) → conservative PASS
                criterion_5_pass = True
                print(f"      C5: tail too short ({tail_end - tail_start} samples) → PASS (conservative)")

            # ================================================================
            # FINAL VALIDATION: All 5 criteria must pass
            # ================================================================
            all_criteria_pass = (criterion_1_pass and criterion_2_pass and criterion_3_pass
                                 and criterion_4_pass and criterion_5_pass)
            
            if all_criteria_pass:
                classification = "✅ CONFIRMED"
            else:
                # Determine reason for rejection
                failed_criteria = []
                if not criterion_1_pass:
                    failed_criteria.append(f"SNR={snr:.1f}<{snr_min}")
                if not criterion_2_pass:
                    failed_criteria.append(f"pre_snr={pre_snr:.2f}>{max_pre_snr:.1f}")
                if not criterion_3_pass:
                    failed_criteria.append(f"E_W1/E_W4={E_W1/E_W4 if E_W4>0 else 0:.2f}<{MIN_DECAY_RATIO}")
                if not criterion_4_pass:
                    failed_criteria.append(f"narrow-spike(asym={asymmetry_ratio:.2f}≥{ASYM_THRESHOLD},flanks@noise)")
                if not criterion_5_pass:
                    failed_criteria.append("secondary-burst")
                classification = f"❌ REJECTED (" + ", ".join(failed_criteria) + ")"
                continue  # Skip rejected candidates
            
            candidates_stage3.append({
                **candidate,           # includes 'spr' from Stage 2
                'peak_amp': peak_amp,
                'peak_idx': peak_idx,
                'decay_results': decay_results,
                'r2_log': r2_log,          # Descriptive
                'tau_ms': tau_ms,          # Descriptive
                'slope_log': slope_log,    # Descriptive
                'snr': snr,                # Criterion 1
                'pre_snr': pre_snr,        # Criterion 2
                'rms_pre': rms_pre,        # Criterion 2 diagnostic
                'pre_source': pre_source,  # Criterion 2 diagnostic (which window was used)
                'E_W1': E_W1,              # Criterion 3
                'E_W4': E_W4,              # Criterion 3
                'asymmetry_ratio': asymmetry_ratio,   # Criterion 4
                'rise_samples': rise_samples,          # Criterion 4 diagnostic
                'fall_samples': fall_samples,          # Criterion 4 diagnostic
                'noise_rms': noise_rms,    # Reference
                'criterion_1_pass': criterion_1_pass,
                'criterion_2_pass': criterion_2_pass,
                'criterion_3_pass': criterion_3_pass,
                'criterion_4_pass': criterion_4_pass,
                'criterion_5_pass': criterion_5_pass,
                'confirmed': True,
                'classification': classification
            })
        
        # ✅ STATISTICHE DETTAGLIATE Stage 3
        stage3_pct = (len(candidates_stage3)/len(candidates_stage2)*100) if len(candidates_stage2) > 0 else 0.0
        print(f"✅ Stage 3 complete: {len(candidates_stage3)}/{len(candidates_stage2)} frames passed ({stage3_pct:.1f}%)")
        
        if len(candidates_stage3) == 0:
            progress.close()
            QMessageBox.information(self, "No Clicks Found", "No frames passed the 3-criterion validation.\nAll candidates failed temporal analysis.")
            return
        
        print(f"   📊 All confirmed clicks passed all 5 criteria:")
        print(f"      ✅ Criterion 1 (SNR > {snr_min}): 100%")
        print(f"      ✅ Criterion 2 (pre_snr < {max_pre_snr}): 100%")
        print(f"      ✅ Criterion 3 (E_W1 > E_W4): 100%")
        print(f"      ✅ Criterion 4 (asymmetry ratio < {ASYM_THRESHOLD}): 100%")
        print(f"      ✅ Criterion 5 (clean tail, no secondary burst): 100%")
        
        # ========================================================================
        # STAGE 4: DEDUPLICATION
        # ========================================================================
        print(f"\n{'='*80}")
        print("STAGE 4: DEDUPLICATION")
        print(f"{'='*80}")
        
        final_clicks = self._deduplicate_clicks(candidates_stage3)
        
        progress.setValue(100)
        progress.close()
        
        print(f"✅ Stage 4 complete: {len(final_clicks)} unique clicks identified")
        
        # Popola tabella
        self._populate_results_table(final_clicks)
        
        self.detected_clicks = final_clicks
        self.export_button.setEnabled(True)

        # ✅ Cache clicks on data_manager so batch screenshot export can access them
        self.data_manager._detected_clicks  = final_clicks
        self.data_manager._cached_threshold_v = threshold_v
        self.data_manager._cached_snr_min    = snr_min
        self.data_manager._cached_max_pre_snr = max_pre_snr
        
        print(f"\n{'='*80}")
        print(f"🎉 DETECTION COMPLETE: {len(final_clicks)} CLICKS FOUND")
        print(f"{'='*80}\n")
        
        # Messaggio finale
        QMessageBox.information(
            self, 
            "Detection Complete", 
            f"Found {len(final_clicks)} ultrasonic clicks!\n\n"
            f"Stage 1 (Energy): {len(candidates_stage1)} candidates\n"
            f"Stage 2 (SPR + Spectral): {len(candidates_stage2)} candidates\n"
            f"Stage 3 (5-criterion): {len(candidates_stage3)} candidates\n"
            f"Stage 4 (Final): {len(final_clicks)} unique clicks"
        )
    
    def _normalize_fft(self, fft_magnitudes):
        """Applica normalizzazione 50% a FFT"""
        # Datasheet SPU0410LR5H-QB
        datasheet_freq_khz = np.array([20, 25, 30, 40, 50, 60, 70, 80])
        datasheet_response_db = np.array([8.0, 10.5, 6.0, -2.0, -6.0, -7.0, -6.0, -4.0])
        datasheet_freq_hz = datasheet_freq_khz * 1000
        
        freq_axis = self.data_manager.frequency_axis
        
        # Correzione 50%
        valid_mask = (freq_axis >= 20000) & (freq_axis <= 80000)
        freq_range = freq_axis[valid_mask]
        
        mic_response_db = np.interp(freq_range, datasheet_freq_hz, datasheet_response_db)
        correction_gain_50 = 10 ** (-mic_response_db * 0.5 / 20.0)
        
        # Applica
        normalized_fft = fft_magnitudes.copy()
        normalized_fft[valid_mask] *= correction_gain_50
        
        return normalized_fft
    
    def _reconstruct_ifft(self, frame_index, normalized=False):
        """Ricostruisce segnale temporale da FFT+fasi"""
        if frame_index >= len(self.data_manager.fft_data):
            return None
        
        # FFT
        fft_mags = self.data_manager.fft_data[frame_index]
        
        if normalized:
            fft_mags = self._normalize_fft(fft_mags)
        
        # Fasi
        if len(self.data_manager.phase_data) == 0:
            return None
        
        fft_phases_int8 = self.data_manager.phase_data[frame_index]
        
        # Parametri
        fs = self.data_manager.header_info.get('fs', 200000)
        fft_size = self.data_manager.header_info.get('fft_size', 512)
        num_bins_full = fft_size // 2
        
        bin_freq = fs / fft_size
        bin_start = int(20000 / bin_freq)
        bin_end = int(80000 / bin_freq)
        
        # Spettro completo
        full_spectrum_mag = np.zeros(num_bins_full, dtype=np.float32)
        full_spectrum_phase = np.zeros(num_bins_full, dtype=np.int8)
        
        actual_bins = min(len(fft_mags), bin_end - bin_start + 1, len(fft_phases_int8))
        full_spectrum_mag[bin_start:bin_start + actual_bins] = fft_mags[:actual_bins]
        full_spectrum_phase[bin_start:bin_start + actual_bins] = fft_phases_int8[:actual_bins]
        
        # Complesso
        fft_phases_rad = (full_spectrum_phase / 127.0) * np.pi
        complex_spectrum = full_spectrum_mag * np.exp(1j * fft_phases_rad)
        
        # ✅ APPLICA TUKEY WINDOW ALLO SPETTRO COMPLESSO (FIX GIBBS CORRETTO)
        taper_bins = max(5, actual_bins // 10)
        window_full = np.ones(num_bins_full)
        
        for i in range(taper_bins):
            alpha = i / taper_bins
            window_full[bin_start + i] = 0.5 * (1 - np.cos(np.pi * alpha))
            window_full[bin_start + actual_bins - i - 1] = 0.5 * (1 - np.cos(np.pi * alpha))
        
        complex_spectrum = complex_spectrum * window_full
        
        # iFFT
        try:
            time_domain_signal = np.fft.irfft(complex_spectrum, n=fft_size)
            time_domain_signal = suppress_edge_artifacts(time_domain_signal)
            return time_domain_signal
        except:
            return None
    
    def _deduplicate_clicks(self, candidates):
        """
        Rimuove duplicati: frame consecutivi (gap ≤ max_gap) = stesso click.
        Mantiene il frame con ampiezza massima per ogni gruppo.
        """
        if len(candidates) == 0:
            return []
        
        MAX_GAP = 3  # frame consecutivi entro questo gap → stesso click
        
        # Ordina per frame_idx
        candidates_sorted = sorted(candidates, key=lambda x: x['frame_idx'])
        
        # Raggruppa per prossimità
        groups = []
        current_group = [candidates_sorted[0]]
        
        for i in range(1, len(candidates_sorted)):
            gap = candidates_sorted[i]['frame_idx'] - candidates_sorted[i-1]['frame_idx']
            if gap <= MAX_GAP:
                current_group.append(candidates_sorted[i])
            else:
                groups.append(current_group)
                current_group = [candidates_sorted[i]]
        groups.append(current_group)
        
        # Per ogni gruppo: tieni il frame con peak_amp massima
        unique_clicks = []
        for group in groups:
            best = max(group, key=lambda x: x['peak_amp'])
            if len(group) > 1:
                print(f"   🔗 Dedup group: {len(group)} frames → kept frame {best['frame_idx']} (amp={best['peak_amp']:.6f} V)")
            unique_clicks.append(best)
        
        return unique_clicks
    
    def _populate_results_table(self, clicks):
        """Popola tabella risultati con nuove colonne v3.0"""
        self.results_table.setRowCount(len(clicks))
        
        for row, click in enumerate(clicks):
            frame_idx = click['frame_idx']
            timestamp = (frame_idx * self.data_manager.frame_duration_ms) / 1000.0
            
            # Col 0: Timestamp
            self.results_table.setItem(row, 0, QTableWidgetItem(f"{timestamp:.3f} s"))
            
            # Col 1: Amplitude
            self.results_table.setItem(row, 1, QTableWidgetItem(f"{click['peak_amp']:.6f} V"))
            
            # Col 2: SNR (Criterion 1)
            snr = click.get('snr', 0.0)
            self.results_table.setItem(row, 2, QTableWidgetItem(f"{snr:.1f}"))
            
            # Col 3: pre_snr (Criterion 2)
            pre_snr = click.get('pre_snr', 0.0)
            item_pre = QTableWidgetItem(f"{pre_snr:.2f}")
            self.results_table.setItem(row, 3, item_pre)
            
            # Col 4: E_W1/E_W4 ratio (Criterion 3)
            E_W1 = click.get('E_W1', 0.0)
            E_W4 = click.get('E_W4', 1e-12)
            decay_ratio = E_W1 / E_W4 if E_W4 > 1e-12 else 999.0
            self.results_table.setItem(row, 4, QTableWidgetItem(f"{decay_ratio:.1f}"))

            # Col 5: SPR – Spectral Peak Ratio (Stage 2 filter)
            spr = click.get('spr', 0.0)
            self.results_table.setItem(row, 5, QTableWidgetItem(f"{spr:.1f}"))

            # Col 6: Energy FFT (descriptive)
            energy_mv2 = click['energies']['total'] * 1e6
            self.results_table.setItem(row, 6, QTableWidgetItem(f"{energy_mv2:.3f} mV²"))
            
            # Col 7: Ratio R (descriptive)
            self.results_table.setItem(row, 7, QTableWidgetItem(f"{click['ratio']:.3f}"))
            
            # Col 8: R² (log) - Decay fit quality (descriptive only)
            r2_log = click.get('r2_log', 0.0)
            self.results_table.setItem(row, 8, QTableWidgetItem(f"{r2_log:.4f}"))
            
            # Col 9: τ (tau) - Decay time constant (descriptive)
            tau = click.get('tau_ms', -1.0)
            if tau > 0:
                self.results_table.setItem(row, 9, QTableWidgetItem(f"{tau:.3f}"))
            else:
                self.results_table.setItem(row, 9, QTableWidgetItem("N/A"))
            
            # Col 10: Classification
            classification_item = QTableWidgetItem(click['classification'])
            if "CONFIRMED" in click['classification']:
                classification_item.setForeground(Qt.darkGreen)
            else:
                classification_item.setForeground(Qt.red)
            self.results_table.setItem(row, 10, classification_item)
            
            # Col 11: Notes
            notes = []
            if click.get('decay_results', {}).get('used_next_frame', False):
                notes.append("Multi-frame")
            if click.get('decay_results', {}).get('near_end', False):
                notes.append("Near end")
            self.results_table.setItem(row, 11, QTableWidgetItem(", ".join(notes)))
    
    def export_results(self):
        """Esporta risultati in CSV con nuove colonne v3.0"""
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
                
                # Header (updated for v3.0 + SPR)
                writer.writerow([
                    "Timestamp (s)", "Frame Index", "Amplitude (V)", "SNR", "pre_snr",
                    "rms_pre (mV)", "SPR", "E_W1", "E_W4", "Decay_Ratio",
                    "Energy FFT (mV²)", "Ratio R", "R² (log)", "τ (ms)", "Classification", "Notes"
                ])
                
                # Dati
                for click in self.detected_clicks:
                    frame_idx = click['frame_idx']
                    timestamp = (frame_idx * self.data_manager.frame_duration_ms) / 1000.0
                    energy_mv2 = click['energies']['total'] * 1e6
                    tau = click.get('tau_ms', -1.0)
                    tau_str = f"{tau:.3f}" if tau > 0 else "N/A"
                    
                    E_W1 = click.get('E_W1', 0.0)
                    E_W4 = click.get('E_W4', 1e-12)
                    decay_ratio = E_W1 / E_W4 if E_W4 > 1e-12 else 999.0
                    rms_pre_mv = click.get('rms_pre', 0.0) * 1000.0
                    spr = click.get('spr', 0.0)
                    
                    notes = []
                    if click.get('decay_results', {}).get('used_next_frame', False):
                        notes.append("Multi-frame")
                    if click.get('decay_results', {}).get('near_end', False):
                        notes.append("Near end")
                    if click.get('pre_source', ''):
                        notes.append(f"pre:{click['pre_source']}")
                    
                    writer.writerow([
                        f"{timestamp:.3f}",
                        frame_idx,
                        f"{click['peak_amp']:.6f}",
                        f"{click.get('snr', 0.0):.1f}",
                        f"{click.get('pre_snr', 0.0):.2f}",
                        f"{rms_pre_mv:.4f}",
                        f"{spr:.1f}",
                        f"{E_W1:.6e}",
                        f"{E_W4:.6e}",
                        f"{decay_ratio:.1f}",
                        f"{energy_mv2:.3f}",
                        f"{click['ratio']:.3f}",
                        f"{click.get('r2_log', 0.0):.4f}",
                        tau_str,
                        click['classification'],
                        "; ".join(notes)
                    ])
            
            QMessageBox.information(self, "Export Successful", f"Results exported to:\n{filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Error writing file:\n{str(e)}")
