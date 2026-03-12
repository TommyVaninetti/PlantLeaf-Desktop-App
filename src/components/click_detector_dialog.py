"""
Click Detector Algorithm Dialog - Rilevamento automatico click ultrasonici

Implementa la pipeline algoritmica a 4 stadi descritta in:
docs/click_detector_algorithm_strategy.md

PIPELINE (v3.0 - Updated Validation Criteria):
1. Stage 1: Energy threshold (μ + Nσ)
2. Stage 2: Spectral ratio R = E_low/E_high (DESCRIPTIVE ONLY - not used for validation)
3. Stage 3: Three-criterion validation:
   - Criterion 1: Temporal SNR > 5.0 (peak_amplitude / noise_rms)
   - Criterion 2: PRE_ratio < 0.15 (E_pre / E_post - silence before click)
   - Criterion 3: Global decay: E_W1 > E_W4 (first > last sub-window)
4. Stage 4: Deduplication

PARAMETRI CONFIGURABILI:
- Threshold energia: μ + Nσ (default: N=4, step=σ)
- Normalizzazione 50%: On/Off
- SNR minimo: 3.0-10.0 (default: 5.0)
- PRE_ratio max: 0.0-0.5 (default: 0.15)

REMOVED IN v3.0:
- R² spectral/decay thresholds (now descriptive features only)
- R as validation criterion (now descriptive feature only)

FEATURE UPDATES:
- tau_ms, r2_log, R: saved as DESCRIPTIVE features for statistical analysis
- Offline noise estimation for accurate SNR calculation
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
from windows.replay_window_audio import compute_hilbert_envelope, find_peak, check_decay, estimate_noise_offline
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
        
        # 4. PRE_ratio max
        self.pre_ratio_max_spinbox = QDoubleSpinBox()
        self.pre_ratio_max_spinbox.setDecimals(2)
        self.pre_ratio_max_spinbox.setRange(0.0, 0.5)
        self.pre_ratio_max_spinbox.setSingleStep(0.05)
        self.pre_ratio_max_spinbox.setValue(0.15)
        self.pre_ratio_max_spinbox.setToolTip("Stage 3: Maximum PRE_ratio for click validation\nRecommended: 0.15")
        params_layout.addRow("Max. PRE_ratio (Criterion 2):", self.pre_ratio_max_spinbox)
        
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
        self.results_table.setColumnCount(11)
        self.results_table.setHorizontalHeaderLabels([
            "Timestamp", "Amplitude", "SNR", "PRE_ratio", "E_W1/E_W4",
            "Energy FFT", "Ratio R", "R² (log)", "τ (ms)", "Classification", "Notes"
        ])
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setAlternatingRowColors(True)
        
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Timestamp
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Amplitude
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # SNR
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # PRE_ratio
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # E_W1/E_W4
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # Energy FFT
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)  # Ratio R
        header.setSectionResizeMode(7, QHeaderView.ResizeToContents)  # R² (log)
        header.setSectionResizeMode(8, QHeaderView.ResizeToContents)  # τ (ms)
        header.setSectionResizeMode(9, QHeaderView.ResizeToContents)  # Classification
        header.setSectionResizeMode(10, QHeaderView.Stretch)          # Notes
        
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
        snr_min = self.snr_min_spinbox.value()
        pre_ratio_max = self.pre_ratio_max_spinbox.value()
        
        print(f"\n📋 PARAMETERS:")
        print(f"   Energy threshold: {threshold_mv:.3f} mV ({threshold_v:.6f} V)")
        print(f"   Normalization: {'ON (50% correction)' if use_normalization else 'OFF'}")
        print(f"   Min. SNR (Criterion 1): {snr_min:.1f}")
        print(f"   Max. PRE_ratio (Criterion 2): {pre_ratio_max:.2f}")
        
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
        # STAGE 2: SPECTRAL ANALYSIS (DESCRIPTIVE ONLY - not used for validation)
        # ========================================================================
        print(f"\n{'='*80}")
        print("STAGE 2: SPECTRAL ENERGY ANALYSIS (DESCRIPTIVE FEATURES)")
        print(f"{'='*80}")
        print("⚠️ NOTE: Spectral ratio R is computed but NOT used for validation in v3.0")
        print("         All Stage 1 candidates pass to Stage 3 for temporal analysis")
        
        candidates_stage2 = []
        
        for idx, frame_idx in enumerate(candidates_stage1):
            if idx % 100 == 0:
                progress.setValue(25 + int((idx / len(candidates_stage1)) * 15))
                if progress.wasCanceled():
                    return
            
            # Ottieni FFT
            fft_mags = self.data_manager.fft_data[frame_idx]
            
            # Applica normalizzazione se richiesto
            if use_normalization:
                fft_mags = self._normalize_fft(fft_mags)
            
            # Calcola energie
            energies = compute_fft_energy(fft_mags)
            
            # Calcola ratio R (DESCRIPTIVE ONLY - not used for filtering)
            ratio = energies['low'] / energies['high'] if energies['high'] > 0 else 0.0
            
            candidates_stage2.append({
                'frame_idx': frame_idx,
                'energies': energies,
                'ratio': ratio,  # Saved as descriptive feature
            })
        
        print(f"✅ Stage 2 complete: {len(candidates_stage2)}/{len(candidates_stage1)} candidates analyzed")
        print(f"   All candidates passed to Stage 3 (no filtering in Stage 2)")
        
        # ========================================================================
        # STAGE 3: THREE-CRITERION TEMPORAL VALIDATION (v3.0)
        # ========================================================================
        print(f"\n{'='*80}")
        print("STAGE 3: TEMPORAL VALIDATION (3 CRITERIA)")
        print(f"{'='*80}")
        print("Criterion 1: SNR > {:.1f} (peak_amplitude / noise_rms)".format(snr_min))
        print("Criterion 2: PRE_ratio < {:.2f} (E_pre / E_post - silence before click)".format(pre_ratio_max))
        print("Criterion 3: Global decay (E_W1 > E_W4)")
        
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
            if peak_idx > 380 and frame_idx + 1 < total_frames:
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
            # CRITERION 2: PRE_ratio < threshold (silence before click)
            # ================================================================
            # E_pre = mean energy in a window STRICTLY BEFORE the peak
            # Use [0:peak_idx-10] to avoid including the click rise itself.
            # If peak is very early (< 20 samples), we cannot compute a
            # meaningful pre-window → treat as silence (ratio = 0.0).
            pre_end = max(0, peak_idx - 10)  # 10-sample guard before peak
            if pre_end >= 10:
                E_pre = np.mean(signal[:pre_end] ** 2)
            else:
                E_pre = 0.0  # Peak too early → assume silence before click

            # E_post: mean power in 120-sample post-peak window
            post_start = peak_idx
            post_end = min(peak_idx + 120, len(signal))
            E_post = np.mean(signal[post_start:post_end] ** 2) if post_end > post_start else 1e-9
            
            pre_ratio = E_pre / E_post if E_post > 1e-12 else 999.0
            criterion_2_pass = (pre_ratio < pre_ratio_max)
            
            # ================================================================
            # CRITERION 3: Global energy decay (E_W1 > E_W4)
            # ================================================================
            criterion_3_pass = (E_W1 > E_W4)
            
            # ================================================================
            # FINAL VALIDATION: All 3 criteria must pass
            # ================================================================
            all_criteria_pass = (criterion_1_pass and criterion_2_pass and criterion_3_pass)
            
            if all_criteria_pass:
                classification = "✅ CONFIRMED"
            else:
                # Determine reason for rejection
                failed_criteria = []
                if not criterion_1_pass:
                    failed_criteria.append(f"SNR={snr:.1f}<{snr_min}")
                if not criterion_2_pass:
                    failed_criteria.append(f"PRE={pre_ratio:.2f}>{pre_ratio_max}")
                if not criterion_3_pass:
                    failed_criteria.append(f"E_W1={E_W1:.2e}≤E_W4={E_W4:.2e}")
                classification = f"❌ REJECTED (" + ", ".join(failed_criteria) + ")"
                continue  # Skip rejected candidates
            
            candidates_stage3.append({
                **candidate,
                'peak_amp': peak_amp,
                'peak_idx': peak_idx,
                'decay_results': decay_results,
                'r2_log': r2_log,          # Descriptive
                'tau_ms': tau_ms,          # Descriptive
                'slope_log': slope_log,    # Descriptive
                'snr': snr,                # Criterion 1
                'pre_ratio': pre_ratio,    # Criterion 2
                'E_W1': E_W1,              # Criterion 3
                'E_W4': E_W4,              # Criterion 3
                'E_pre': E_pre,            # Diagnostic
                'E_post': E_post,          # Diagnostic
                'noise_rms': noise_rms,    # Reference
                'criterion_1_pass': criterion_1_pass,
                'criterion_2_pass': criterion_2_pass,
                'criterion_3_pass': criterion_3_pass,
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
        
        print(f"   📊 All confirmed clicks passed all 3 criteria:")
        print(f"      ✅ Criterion 1 (SNR > {snr_min}): 100%")
        print(f"      ✅ Criterion 2 (PRE < {pre_ratio_max}): 100%")
        print(f"      ✅ Criterion 3 (E_W1 > E_W4): 100%")
        
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
        
        print(f"\n{'='*80}")
        print(f"🎉 DETECTION COMPLETE: {len(final_clicks)} CLICKS FOUND")
        print(f"{'='*80}\n")
        
        # Messaggio finale
        QMessageBox.information(
            self, 
            "Detection Complete", 
            f"Found {len(final_clicks)} ultrasonic clicks!\n\n"
            f"Stage 1 (Energy): {len(candidates_stage1)} candidates\n"
            f"Stage 2 (Spectral): {len(candidates_stage2)} candidates\n"
            f"Stage 3 (Decay): {len(candidates_stage3)} candidates\n"
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
            
            # Col 3: PRE_ratio (Criterion 2)
            pre_ratio = click.get('pre_ratio', 0.0)
            self.results_table.setItem(row, 3, QTableWidgetItem(f"{pre_ratio:.3f}"))
            
            # Col 4: E_W1/E_W4 ratio (Criterion 3)
            E_W1 = click.get('E_W1', 0.0)
            E_W4 = click.get('E_W4', 1e-12)
            decay_ratio = E_W1 / E_W4 if E_W4 > 1e-12 else 999.0
            self.results_table.setItem(row, 4, QTableWidgetItem(f"{decay_ratio:.1f}"))
            
            # Col 5: Energy FFT (descriptive)
            energy_mv2 = click['energies']['total'] * 1e6
            self.results_table.setItem(row, 5, QTableWidgetItem(f"{energy_mv2:.3f} mV²"))
            
            # Col 6: Ratio R (descriptive)
            self.results_table.setItem(row, 6, QTableWidgetItem(f"{click['ratio']:.3f}"))
            
            # Col 7: R² (log) - Decay fit quality (descriptive only)
            r2_log = click.get('r2_log', 0.0)
            self.results_table.setItem(row, 7, QTableWidgetItem(f"{r2_log:.4f}"))
            
            # Col 8: τ (tau) - Decay time constant (descriptive)
            tau = click.get('tau_ms', -1.0)
            if tau > 0:
                self.results_table.setItem(row, 8, QTableWidgetItem(f"{tau:.3f}"))
            else:
                self.results_table.setItem(row, 8, QTableWidgetItem("N/A"))
            
            # Col 9: Classification
            classification_item = QTableWidgetItem(click['classification'])
            if "CONFIRMED" in click['classification']:
                classification_item.setForeground(Qt.darkGreen)
            else:
                classification_item.setForeground(Qt.red)
            self.results_table.setItem(row, 9, classification_item)
            
            # Col 10: Notes
            notes = []
            if click.get('decay_results', {}).get('used_next_frame', False):
                notes.append("Multi-frame")
            if click.get('decay_results', {}).get('near_end', False):
                notes.append("Near end")
            self.results_table.setItem(row, 10, QTableWidgetItem(", ".join(notes)))
    
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
                
                # Header (updated for v3.0)
                writer.writerow([
                    "Timestamp (s)", "Frame Index", "Amplitude (V)", "SNR", "PRE_ratio",
                    "E_W1", "E_W4", "Decay_Ratio", "Energy FFT (mV²)", 
                    "Ratio R", "R² (log)", "τ (ms)", "Classification", "Notes"
                ])
                
                # Dati
                for click in self.detected_clicks:
                    frame_idx = click['frame_idx']
                    timestamp = (frame_idx * self.data_manager.frame_duration_ms) / 1000.0
                    energy_mv2 = click['energies']['total'] * 1e6
                    tau = click.get('tau_ms', -1.0)
                    tau_str = f"{tau:.3f}" if tau > 0 else "N/A"
                    
                    # Calculate decay ratio
                    E_W1 = click.get('E_W1', 0.0)
                    E_W4 = click.get('E_W4', 1e-12)
                    decay_ratio = E_W1 / E_W4 if E_W4 > 1e-12 else 999.0
                    
                    notes = []
                    if click.get('decay_results', {}).get('used_next_frame', False):
                        notes.append("Multi-frame")
                    if click.get('decay_results', {}).get('near_end', False):
                        notes.append("Near end")
                    
                    writer.writerow([
                        f"{timestamp:.3f}",
                        frame_idx,
                        f"{click['peak_amp']:.6f}",
                        f"{click.get('snr', 0.0):.1f}",
                        f"{click.get('pre_ratio', 0.0):.3f}",
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
