"""
Click Detector Algorithm Dialog - Rilevamento automatico click ultrasonici

Implementa la pipeline algoritmica a 4 stadi descritta in:
docs/click_detector_algorithm_strategy.md

PIPELINE:
1. Stage 1: Energy threshold (μ + Nσ)
2. Stage 2: Spectral ratio R = E_low/E_high
3. Stage 3: iFFT decay analysis (Hilbert envelope)
4. Stage 4: Deduplication

PARAMETRI CONFIGURABILI:
- Threshold energia: μ + Nσ (default: N=4, step=σ)
- Normalizzazione 50%: On/Off
- R² minimo spectral: 0.0-1.0
- R² minimo decay: 0.0-1.0
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
from windows.replay_window_audio import compute_hilbert_envelope, find_peak, check_decay
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
        
        # 3. R² minimo spectral
        self.r2_spectral_spinbox = QDoubleSpinBox()
        self.r2_spectral_spinbox.setDecimals(2)
        self.r2_spectral_spinbox.setRange(0.0, 1.0)
        self.r2_spectral_spinbox.setSingleStep(0.05)
        self.r2_spectral_spinbox.setValue(0.70)
        self.r2_spectral_spinbox.setToolTip("Stage 2: Minimum R² for spectral ratio quality\n≥0.7 = IDENTIFIED\n≥0.5 = POSSIBLE")
        params_layout.addRow("R² min (spectral ratio):", self.r2_spectral_spinbox)
        
        # 4. R² minimo decay
        self.r2_decay_spinbox = QDoubleSpinBox()
        self.r2_decay_spinbox.setDecimals(2)
        self.r2_decay_spinbox.setRange(0.0, 1.0)
        self.r2_decay_spinbox.setSingleStep(0.05)
        self.r2_decay_spinbox.setValue(0.60)  # Reduced from 0.70 for 120-sample method
        self.r2_decay_spinbox.setToolTip("Stage 3: Minimum R² for decay quality (logarithmic fit on 120 samples)\n≥0.80 = IDENTIFIED (high confidence)\n≥0.60 = POSSIBLE (borderline)\n<0.60 = NOT_CLICK")
        params_layout.addRow("R² min (decay analysis):", self.r2_decay_spinbox)
        
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
        self.results_table.setColumnCount(9)
        self.results_table.setHorizontalHeaderLabels([
            "Timestamp", "Amplitude", "Energy FFT", "Ratio R", 
            "R² Spectral", "R² Decay (log)", "τ (ms)", "Classification", "Notes"
        ])
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setAlternatingRowColors(True)
        
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(8, QHeaderView.Stretch)
        
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
        r2_spectral_min = self.r2_spectral_spinbox.value()
        r2_decay_min = self.r2_decay_spinbox.value()
        
        print(f"\n📋 PARAMETERS:")
        print(f"   Energy threshold: {threshold_mv:.3f} mV ({threshold_v:.6f} V)")
        print(f"   Normalization: {'ON (50% correction)' if use_normalization else 'OFF'}")
        print(f"   R² min (spectral): {r2_spectral_min:.2f}")
        print(f"   R² min (decay): {r2_decay_min:.2f}")
        
        total_frames = self.data_manager.total_frames
        
        # Progress dialog
        progress = QProgressDialog("Running click detection...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        
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
        # STAGE 2: SPECTRAL RATIO CHECK
        # ========================================================================
        print(f"\n{'='*80}")
        print("STAGE 2: SPECTRAL RATIO VERIFICATION (BROADBAND TEST)")
        print(f"{'='*80}")
        
        candidates_stage2 = []
        
        for idx, frame_idx in enumerate(candidates_stage1):
            if idx % 100 == 0:
                progress.setValue(25 + int((idx / len(candidates_stage1)) * 25))
                if progress.wasCanceled():
                    return
            
            # Ottieni FFT
            fft_mags = self.data_manager.fft_data[frame_idx]
            
            # Applica normalizzazione se richiesto
            if use_normalization:
                fft_mags = self._normalize_fft(fft_mags)
            
            # Calcola energie
            energies = compute_fft_energy(fft_mags)
            
            # Calcola ratio R
            ratio = energies['low'] / energies['high'] if energies['high'] > 0 else 0.0
            
            # ✅ RANGE DI ACCETTAZIONE ESTESO (include casi borderline/dubbi)
            # Fisica: Click broadband ha ratio ~0.5-2.0 (energia bilanciata low/high)
            # Artifacts: Toni puri (<0.2) o EMI (>5.0)
            # 
            # STRATEGY: Accetta anche click "dubbi" con distribuzione spettrale anomala
            # → Stage 3 (decay analysis) farà la selezione finale
            if use_normalization:
                # Con normalizzazione: range più ampio (correzione altera distribuzione)
                r_min, r_max = 0.2, 3.0  # Era: 0.3-2.0 (troppo restrittivo)
                classification = "NORMALIZED"
            else:
                # Senza normalizzazione: range molto permissivo
                r_min, r_max = 0.3, 5.0  # Era: 0.5-3.0 (troppo restrittivo)
                classification = "RAW"
            
            # ✅ ACCETTA se ratio in range (anche borderline)
            if r_min <= ratio <= r_max:
                # Classifica qualità ratio per diagnostica
                if use_normalization:
                    if 0.5 <= ratio <= 1.5:
                        ratio_quality = "GOOD"
                    elif 0.3 <= ratio <= 2.0:
                        ratio_quality = "ACCEPTABLE"
                    else:
                        ratio_quality = "BORDERLINE"
                else:
                    if 0.6 <= ratio <= 2.5:
                        ratio_quality = "GOOD"
                    elif 0.4 <= ratio <= 3.5:
                        ratio_quality = "ACCEPTABLE"
                    else:
                        ratio_quality = "BORDERLINE"
                
                candidates_stage2.append({
                    'frame_idx': frame_idx,
                    'energies': energies,
                    'ratio': ratio,
                    'ratio_quality': ratio_quality,
                    'normalization': classification
                })
        
        # ✅ STATISTICHE DETTAGLIATE Stage 2
        num_good = sum(1 for c in candidates_stage2 if c.get('ratio_quality') == 'GOOD')
        num_acceptable = sum(1 for c in candidates_stage2 if c.get('ratio_quality') == 'ACCEPTABLE')
        num_borderline = sum(1 for c in candidates_stage2 if c.get('ratio_quality') == 'BORDERLINE')
        
        print(f"✅ Stage 2 complete: {len(candidates_stage2)}/{len(candidates_stage1)} frames passed ({len(candidates_stage2)/len(candidates_stage1)*100:.1f}%)")
        if len(candidates_stage2) > 0:
            print(f"   📊 Spectral ratio quality breakdown:")
            print(f"      ✅ GOOD: {num_good} ({num_good/len(candidates_stage2)*100:.1f}% of passed)")
            print(f"      ⚠️ ACCEPTABLE: {num_acceptable} ({num_acceptable/len(candidates_stage2)*100:.1f}% of passed)")
            print(f"      🔶 BORDERLINE: {num_borderline} ({num_borderline/len(candidates_stage2)*100:.1f}% of passed - spectrum anomaly)")

        
        if len(candidates_stage2) == 0:
            progress.close()
            QMessageBox.information(self, "No Clicks Found", "No frames passed the spectral ratio test.\nAll candidates are likely narrowband artifacts.")
            return
        
        # ========================================================================
        # STAGE 3: DECAY ANALYSIS (TIME DOMAIN)
        # ========================================================================
        print(f"\n{'='*80}")
        print("STAGE 3: TIME DOMAIN DECAY ANALYSIS")
        print(f"{'='*80}")
        
        candidates_stage3 = []
        
        for idx, candidate in enumerate(candidates_stage2):
            if idx % 10 == 0:
                progress.setValue(50 + int((idx / len(candidates_stage2)) * 40))
                if progress.wasCanceled():
                    return
            
            frame_idx = candidate['frame_idx']
            
            # Ricostruisci iFFT
            signal = self._reconstruct_ifft(frame_idx, use_normalization)
            
            if signal is None:
                continue
            
            # Calcola envelope
            envelope = compute_hilbert_envelope(signal)
            
            # Trova picco
            peak_idx, peak_amp = find_peak(signal)
            
            # Check decay (con gestione spill)
            next_frame_signal = None
            if peak_idx > 380 and frame_idx + 1 < total_frames:
                next_frame_signal = self._reconstruct_ifft(frame_idx + 1, use_normalization)
            
            decay_results = check_decay(envelope, peak_idx, next_frame_signal=next_frame_signal)
            
            # Classifica in base a R²_log (FIT LOGARITMICO su 120 CAMPIONI)
            r2_log = decay_results['r_squared_log']
            tau = decay_results['tau_ms']
            slope_log = decay_results['slope_log']
            
            # CRITERIO PRINCIPALE: Usa solo R²_log e slope (non 'decaying' che ha threshold hardcoded)
            # - slope < 0: OBBLIGATORIO (decay, not growth)
            # - R² >= 0.70: HIGH confidence → IDENTIFIED (abbassato per 120-sample robustness)
            # - R² >= r2_decay_min (default 0.60): borderline → POSSIBLE
            # - R² < r2_decay_min: NOT_CLICK
            
            # Guard: slope deve essere negativo (requisito fisico fondamentale)
            if slope_log >= 0:
                classification = "❌ NOT_CLICK"
                continue  # Growth = physically invalid
            
            # Classifica in base a R² threshold
            if r2_log >= 0.70:  # Abbassato da 0.80 per maggiore sensibilità
                classification = "✅ IDENTIFIED"
            elif r2_log >= r2_decay_min:
                classification = "⚠️ POSSIBLE"
            else:
                classification = "❌ NOT_CLICK"
                continue  # Below user threshold
            
            candidates_stage3.append({
                **candidate,
                'peak_amp': peak_amp,
                'peak_idx': peak_idx,
                'decay_results': decay_results,
                'r2_log': r2_log,
                'tau_ms': tau,
                'classification': classification
            })
        
        # ✅ STATISTICHE DETTAGLIATE
        num_identified = sum(1 for c in candidates_stage3 if "IDENTIFIED" in c['classification'])
        num_possible = sum(1 for c in candidates_stage3 if "POSSIBLE" in c['classification'])
        
        print(f"✅ Stage 3 complete: {len(candidates_stage3)}/{len(candidates_stage2)} frames passed ({len(candidates_stage3)/len(candidates_stage2)*100:.1f}%)")
        print(f"   📊 Classification breakdown:")
        print(f"      ✅ IDENTIFIED: {num_identified} ({num_identified/len(candidates_stage3)*100:.1f}% of passed)")
        print(f"      ⚠️ POSSIBLE: {num_possible} ({num_possible/len(candidates_stage3)*100:.1f}% of passed)")

        
        if len(candidates_stage3) == 0:
            progress.close()
            QMessageBox.information(self, "No Clicks Found", "No frames passed the decay analysis test.\nAll candidates lack exponential decay signature.")
            return
        
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
        
        # iFFT
        try:
            time_domain_signal = np.fft.irfft(complex_spectrum, n=fft_size)
            return time_domain_signal
        except:
            return None
    
    def _deduplicate_clicks(self, candidates):
        """Rimuove duplicati (frame consecutivi = stesso click)"""
        if len(candidates) == 0:
            return []
        
        # Ordina per frame_idx
        candidates_sorted = sorted(candidates, key=lambda x: x['frame_idx'])
        
        unique_clicks = []
        i = 0
        
        while i < len(candidates_sorted):
            current = candidates_sorted[i]
            
            # Controlla se il prossimo frame è consecutivo
            if i + 1 < len(candidates_sorted) and candidates_sorted[i + 1]['frame_idx'] == current['frame_idx'] + 1:
                next_candidate = candidates_sorted[i + 1]
                
                # Mantieni quello con ampiezza maggiore
                if next_candidate['peak_amp'] > current['peak_amp']:
                    unique_clicks.append(next_candidate)
                else:
                    unique_clicks.append(current)
                
                i += 2  # Skip entrambi
            else:
                unique_clicks.append(current)
                i += 1
        
        return unique_clicks
    
    def _populate_results_table(self, clicks):
        """Popola tabella risultati"""
        self.results_table.setRowCount(len(clicks))
        
        for row, click in enumerate(clicks):
            frame_idx = click['frame_idx']
            timestamp = (frame_idx * self.data_manager.frame_duration_ms) / 1000.0
            
            # Timestamp
            self.results_table.setItem(row, 0, QTableWidgetItem(f"{timestamp:.3f} s"))
            
            # Amplitude
            self.results_table.setItem(row, 1, QTableWidgetItem(f"{click['peak_amp']:.6f} V"))
            
            # Energy FFT
            energy_mv2 = click['energies']['total'] * 1e6
            self.results_table.setItem(row, 2, QTableWidgetItem(f"{energy_mv2:.3f} mV²"))
            
            # Ratio R
            self.results_table.setItem(row, 3, QTableWidgetItem(f"{click['ratio']:.3f}"))
            
            # R² Spectral (TODO: implementare calcolo)
            self.results_table.setItem(row, 4, QTableWidgetItem("N/A"))
            
            # R² Decay (LOGARITMICO)
            r2_log = click['r2_log']
            self.results_table.setItem(row, 5, QTableWidgetItem(f"{r2_log:.4f}"))
            
            # τ (tau) - Decay time constant
            tau = click['tau_ms']
            if tau > 0:
                self.results_table.setItem(row, 6, QTableWidgetItem(f"{tau:.3f}"))
            else:
                self.results_table.setItem(row, 6, QTableWidgetItem("N/A"))
            
            # Classification
            classification_item = QTableWidgetItem(click['classification'])
            if "IDENTIFIED" in click['classification']:
                classification_item.setForeground(Qt.darkGreen)
            elif "POSSIBLE" in click['classification']:
                classification_item.setForeground(Qt.darkYellow)
            else:
                classification_item.setForeground(Qt.red)
            self.results_table.setItem(row, 7, classification_item)
            
            # Notes
            notes = []
            if click['decay_results'].get('used_next_frame', False):
                notes.append("Multi-frame")
            if click['decay_results'].get('near_end', False):
                notes.append("Near end")
            self.results_table.setItem(row, 8, QTableWidgetItem(", ".join(notes)))
    
    def export_results(self):
        """Esporta risultati in CSV"""
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
                
                # Header
                writer.writerow([
                    "Timestamp (s)", "Frame Index", "Amplitude (V)", "Energy FFT (mV²)", 
                    "Ratio R", "R² Decay (log)", "τ (ms)", "Classification", "Notes"
                ])
                
                # Dati
                for click in self.detected_clicks:
                    frame_idx = click['frame_idx']
                    timestamp = (frame_idx * self.data_manager.frame_duration_ms) / 1000.0
                    energy_mv2 = click['energies']['total'] * 1e6
                    tau = click['tau_ms'] if click['tau_ms'] > 0 else "N/A"
                    
                    notes = []
                    if click['decay_results'].get('used_next_frame', False):
                        notes.append("Multi-frame")
                    if click['decay_results'].get('near_end', False):
                        notes.append("Near end")
                    
                    writer.writerow([
                        f"{timestamp:.3f}",
                        frame_idx,
                        f"{click['peak_amp']:.6f}",
                        f"{energy_mv2:.3f}",
                        f"{click['ratio']:.3f}",
                        f"{click['r2_log']:.4f}",
                        tau if tau == "N/A" else f"{tau:.3f}",
                        click['classification'],
                        "; ".join(notes)
                    ])
            
            QMessageBox.information(self, "Export Successful", f"Results exported to:\n{filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Error writing file:\n{str(e)}")
