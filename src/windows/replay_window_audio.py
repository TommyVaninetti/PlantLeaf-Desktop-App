"""
Finestra di replay per file audio .paudio - Sistema Multi-Level ottimizzato per Click Detection
Architettura ispirata a replay_window_voltage ma ottimizzata per audio e PC non potenti

ARCHITETTURA MULTI-LEVEL:
- LEVEL 1: Overview (10 FPS, energia media) - SEMPRE CARICATO
- LEVEL 2: Streaming Buffer (100 FPS, energia campionata) - FINESTRA MOBILE 30s
- LEVEL 3: Detail Cache (390 FPS, energia completa) - ON-DEMAND per click detection
"""

import numpy as np
import struct
import json
import zlib
import os
from PySide6.QtWidgets import (QVBoxLayout, QHBoxLayout, QWidget, QTabWidget, 
                              QSplitter, QTableWidget, QTableWidgetItem, 
                              QHeaderView, QLabel, QPushButton, QMessageBox,
                              QSlider, QDoubleSpinBox, QSizePolicy, QDialog)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QFont
from PySide6 import QtCore

from core.replay_base_window import ReplayBaseWindow, compute_fft_energy, SpectralEnergyDialog
from core.replay_base_window import ReplayBaseWindow, compute_fft_energy, SpectralEnergyDialog
from plotting.plot_manager import BasePlotWidget
from core.audio_trim_export import AudioTrimExporter
from scipy.signal import hilbert


class AudioDataManager:
    """Gestisce dati audio con architettura multi-livello per performance ottimali"""
    
    def __init__(self):
        # LEVEL 1: Overview Data (sempre in memoria)
        self.overview_x = np.array([])
        self.overview_y = np.array([])
        self.overview_loaded = False
        
        # LEVEL 2: Streaming Buffer (finestra mobile 20s)
        self.streaming_x = np.array([])
        self.streaming_y = np.array([])
        self.streaming_start_time = 0.0
        self.streaming_end_time = 0.0
        self.streaming_window_size = 20.0  # secondi
        
        # LEVEL 3: Detail Cache (regioni ad alta risoluzione)
        self.detail_cache = {}  # {(start, end): (x_array, y_array, clicks)}
        self.max_detail_regions = 3  # Limite memoria
        
        # Metadati del file
        self.header_info = {}
        self.total_duration_sec = 0.0
        self.total_frames = 0
        self.frame_duration_ms = 2.564  # ~390 FPS
        self.click_events = []
        
        # Dati raw FFT (per calcoli on-demand)
        self.fft_data = []
        self.phase_data = []
        self.frequency_axis = []

        # Array delle medie delle FFT PRECALCOLATE
        self.fft_means = np.array([]) #media delle magnitude delle FFT
        self.fft_timestamps = np.array([]) #timestamps corrispondenti alle FFT
        
        # Performance settings (adattivi)
        self.overview_fps = 10      # FPS per overview
        self.streaming_fps = 100    # FPS per streaming buffer
        self.memory_limit_mb = 200  # Limite memoria totale

    def precompute_fft_means(self):
        """
        Precalcola le medie di tutte le FFT per analisi threshold.
        Chiamato una sola volta all'apertura del file.
        """
        if len(self.fft_data) == 0:
            return
        
        print(f"🔄 Precalcolo medie FFT per {self.total_frames} frame...")
        
        # Calcola medie e timestamp
        means = []
        timestamps = []
        
        for i, fft_frame in enumerate(self.fft_data):
            mean_val = np.mean(np.abs(fft_frame))
            means.append(mean_val)
            timestamps.append((i * self.frame_duration_ms) / 1000.0)
        
        self.fft_means = np.array(means)
        self.fft_timestamps = np.array(timestamps)
        
        memory_mb = (self.fft_means.nbytes + self.fft_timestamps.nbytes) / 1024 / 1024
        print(f"✅ Medie precalcolate: {len(means)} punti, {memory_mb:.1f} MB")

        # ✅ OUTLIER DETECTION: Calcola media e std escludendo SOLO outliers ESTREMI
        # Target: Rimuovere solo frame completamente corrotti (es. 20000V), 
        # NON i click normali che sono parte della distribuzione reale
        if len(self.fft_means) > 0:
            # STRATEGIA: Usa mediana + soglia adattiva basata su MAD (Median Absolute Deviation)
            # Più robusta dell'IQR per outliers estremi isolati
            
            median = np.median(self.fft_means)
            mad = np.median(np.abs(self.fft_means - median))
            
            # Modified Z-score threshold (Iglewicz and Hoaglin, 1993)
            # Soglia 3.5 cattura solo outliers MOLTO estremi (>99.9%)
            # Se MAD è troppo piccolo (dati molto uniformi), usa fallback
            if mad > 0:
                modified_z_scores = 0.6745 * (self.fft_means - median) / mad
                outlier_mask = np.abs(modified_z_scores) > 3.5
            else:
                # Fallback: soglia assoluta basata su multiplo della mediana
                # Rimuovi solo valori >20× la mediana (MOLTO conservativo)
                outlier_mask = self.fft_means > (median * 20)
            
            filtered_means = self.fft_means[~outlier_mask]
            n_outliers = np.sum(outlier_mask)
            
            if n_outliers > 0:
                outlier_values = self.fft_means[outlier_mask]
                outlier_indices = np.where(outlier_mask)[0]
                print(f"⚠️ OUTLIER DETECTION: Rimossi {n_outliers} frame ESTREMI anomali dal calcolo statistico")
                print(f"   Mediana: {median*1000:.3f} mV")
                print(f"   MAD: {mad*1000:.3f} mV")
                print(f"   Frame outlier: {outlier_indices[:10]}" + (" ..." if len(outlier_indices) > 10 else ""))
                print(f"   Valori outlier: min={outlier_values.min()*1000:.1f} mV, max={outlier_values.max()*1000:.1f} mV")
                print(f"   Frame validi: {len(filtered_means)}/{len(self.fft_means)} ({100*len(filtered_means)/len(self.fft_means):.1f}%)")
            else:
                print(f"✅ Nessun outlier estremo rilevato (tutti i frame sono validi)")
            
            # Calcola statistiche sui dati filtrati
            self.fft_mean = np.mean(filtered_means) if len(filtered_means) > 0 else 0
            self.fft_std = np.std(filtered_means) if len(filtered_means) > 0 else 0
            
            print(f"📊 Statistiche FFT (outlier-free):")
            print(f"   Mean: {self.fft_mean*1000:.3f} mV")
            print(f"   Std:  {self.fft_std*1000:.3f} mV")
            print(f"   Threshold μ+4σ: {(self.fft_mean + 4*self.fft_std)*1000:.3f} mV")
        else:
            self.fft_mean = 0
            self.fft_std = 0

    def get_memory_usage_mb(self):
        """Calcola uso memoria corrente in MB"""
        overview_mb = (len(self.overview_x) + len(self.overview_y)) * 8 / 1024 / 1024
        streaming_mb = (len(self.streaming_x) + len(self.streaming_y)) * 8 / 1024 / 1024
        detail_mb = sum((len(x) + len(y)) * 8 for x, y, _ in self.detail_cache.values()) / 1024 / 1024
        return overview_mb + streaming_mb + detail_mb
    
    def contains_streaming_time(self, time_sec):
        """Verifica se il tempo è nel buffer streaming corrente"""
        return (self.streaming_start_time <= time_sec <= self.streaming_end_time and 
                len(self.streaming_x) > 0)
    
    def get_overview_data(self):
        """Restituisce dati overview (sempre disponibili)"""
        return self.overview_x, self.overview_y
    
    def get_streaming_data(self):
        """Restituisce dati streaming correnti"""
        return self.streaming_x, self.streaming_y
    
    def needs_detail_for_time(self, time_sec, window_sec=5.0):
        """Verifica se servono dati detail per un tempo specifico"""
        for (start, end), _ in self.detail_cache.items():
            if start <= time_sec <= end:
                return False  # Già in cache
        return True
    
    def cleanup_detail_cache(self):
        """Pulisce cache detail se supera il limite"""
        if len(self.detail_cache) > self.max_detail_regions:
            # Rimuovi la regione più vecchia (LRU semplice)
            oldest_key = list(self.detail_cache.keys())[0]
            del self.detail_cache[oldest_key]
            print(f"🧹 Rimossa regione detail cache: {oldest_key}")



class IFFTWindow(QDialog):
    """Finestra per mostrare il grafico del segnale iFFT con opzione normalizzazione."""
    def __init__(self, time_data, signal_data, parent=None, frame_index=None, has_real_phases=False):
        super().__init__(parent)
        
        # Salva riferimenti per normalizzazione
        self.parent = parent
        self.frame_index = frame_index
        self.has_real_phases = has_real_phases
        self.time_data = time_data
        self.signal_data_raw = signal_data  # Segnale RAW
        self.signal_data_normalized = None  # Verrà calcolato on-demand
        self.is_normalized = False

        # ✅ TITOLO CON INFO
        title = "Inverse FFT Signal (Reconstructed)"
        if frame_index is not None:
            title += f" - Frame {frame_index}"
        if has_real_phases:
            title += " [Real Phases]"
        else:
            title += " [Zero Phases]"
        
        self.setWindowTitle(title)
        self.setMinimumSize(800, 400)
        
        layout = QVBoxLayout(self)
        
        # Calcola il range corretto dai dati in input
        x_min_val, x_max_val = (0, 1) # Default
        if time_data is not None and len(time_data) > 1:
            x_min_val = time_data[0]
            x_max_val = time_data[-1]

        # Crea il widget del grafico con il range corretto e auto-range per l'asse Y
        self.plot_widget = BasePlotWidget(
            x_label="Time", y_label="Amplitude",
            x_range=(x_min_val, x_max_val), y_range=(-0.002, 0.002),
            x_min=x_min_val, x_max=x_max_val, y_min=-1.7, y_max=1.7,
            unit_x="s", unit_y="V", parent=self
        )
        
        # ✅ COLORE DAL TEMA (accent color)
        # Inizialmente senza colore specifico, verrà applicato dal theme_manager
        self.ifft_curve = self.plot_widget.plot_widget.plot(
            time_data, signal_data, 
            pen={'width': 1.5},
            name='Raw iFFT'
        )
        self.plot_widget.plot_widget.showGrid(x=True, y=True)

        layout.addWidget(self.plot_widget)

        # ✅ AGGIUNGI PULSANTE NORMALIZZAZIONE
        button_layout = QHBoxLayout()
        
        self.normalize_button = QPushButton("Apply 50% Normalization")
        self.normalize_button.setToolTip(
            "Apply conservative 50% frequency response correction\n"
            "Based on SPU0410LR5H-QB datasheet\n"
            "Estimated error: ±2.9 dB (95% confidence)"
        )
        self.normalize_button.clicked.connect(self.toggle_normalization)
        button_layout.addWidget(self.normalize_button)
        
        # ✅ AGGIUNGI PULSANTE ENVELOPE ANALYSIS
        self.envelope_button = QPushButton("Show Hilbert Envelope")
        self.envelope_button.setToolTip(
            "Calculate and display instantaneous amplitude envelope\n"
            "using Hilbert transform (red thick line)"
        )
        self.envelope_button.clicked.connect(self.toggle_envelope)
        button_layout.addWidget(self.envelope_button)
        
        # ✅ AGGIUNGI PULSANTE DECAY ANALYSIS
        self.decay_button = QPushButton("Analyze Decay")
        self.decay_button.setToolTip(
            "Check if signal shows exponential decay typical of ultrasonic clicks\n"
            "Analyzes 0.6 ms post-peak window (120 samples @ 200 ksps)"
        )
        self.decay_button.clicked.connect(self.analyze_decay)
        button_layout.addWidget(self.decay_button)
        
        self.info_label = QLabel("📊 Raw iFFT signal (no correction)")
        self.info_label.setStyleSheet("color: #888; font-size: 10pt;")
        button_layout.addWidget(self.info_label)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        # Variabili per envelope analysis
        self.envelope_data = None
        self.envelope_curve = None
        self.peak_line = None
        self.show_envelope = False

        # Menubar
        from PySide6.QtWidgets import QMenuBar
        menubar = QMenuBar(self)
        file_menu = menubar.addMenu("Analysis")
        action_close = QAction("Close", self)
        action_close.triggered.connect(self.close)
        file_menu.addAction(action_close)

        layout.setMenuBar(menubar)
        self.setLayout(layout)

        # Applica tema (imposta accent_color sulla curva)
        if parent and hasattr(parent, 'theme_manager'):
            saved_theme = parent.theme_manager.load_saved_theme()
            parent.theme_manager.apply_theme(self, saved_theme)
            # Applica accent color del tema alla curva iFFT
            parent.theme_manager.apply_theme_to_plot(
                plot_widget_name=self.plot_widget.plot_widget,
                plot_instance=self.ifft_curve
            )

        # motra in mezzo allo schermo del parent MA CON DIMENSIONE MINORE E TENENDO CONTO DELLE GRAFICHE SOPRATTUTTO PER WINDOWS
        if parent:
            parent_rect = parent.geometry()
            self.resize(parent_rect.width() * 0.8, parent_rect.height() * 0.6)
            self.move(
                parent_rect.x() + (parent_rect.width() - self.width()) // 2,
                parent_rect.y() + (parent_rect.height() - self.height()) // 2
            )
    
    def toggle_normalization(self):
        """Toggle tra iFFT raw e normalizzato (50%)"""
        if not self.is_normalized:
            # APPLICA NORMALIZZAZIONE
            self._compute_normalized_ifft()
            if self.signal_data_normalized is not None:
                self.is_normalized = True
                self._update_display()
                
                # ✅ RICALCOLA ENVELOPE SE GIÀ VISUALIZZATO
                if self.show_envelope:
                    print("🔄 Recalculating envelope for normalized signal...")
                    self._compute_and_show_envelope()
        else:
            # TORNA A RAW
            self.is_normalized = False
            self._update_display()
            
            # ✅ RICALCOLA ENVELOPE SE GIÀ VISUALIZZATO
            if self.show_envelope:
                print("🔄 Recalculating envelope for raw signal...")
                self._compute_and_show_envelope()
    
    def _compute_normalized_ifft(self):
        """Calcola iFFT con correzione 50% dalla FFT normalizzata"""
        if not self.parent or not hasattr(self.parent, 'data_manager'):
            QMessageBox.warning(self, "Error", "Cannot access parent data manager.")
            return
        
        print("🔧 Computing normalized iFFT (50% correction)...")
        
        # === 1. DATI DAL DATASHEET (IDENTICI a normalize_fft_window) ===
        datasheet_freq_khz = np.array([20, 25, 30, 40, 50, 60, 70, 80])
        datasheet_response_db = np.array([8.0, 10.5, 6.0, -2.0, -6.0, -7.0, -6.0, -4.0])
        datasheet_freq_hz = datasheet_freq_khz * 1000
        
        # === 2. RECUPERA FFT ORIGINALE DEL FRAME ===
        if self.frame_index is None or self.frame_index >= len(self.parent.data_manager.fft_data):
            QMessageBox.warning(self, "Error", "Invalid frame index.")
            return
        
        fft_magnitudes = self.parent.data_manager.fft_data[self.frame_index]
        freq_axis = self.parent.data_manager.frequency_axis
        
        # === 3. CALCOLA CORREZIONE ===
        valid_mask = (freq_axis >= 20000) & (freq_axis <= 80000)
        freq_range = freq_axis[valid_mask]
        
        mic_response_db = np.interp(freq_range, datasheet_freq_hz, datasheet_response_db)
        correction_gain_50 = 10 ** (-mic_response_db * 0.5 / 20.0)
        
        full_correction = np.ones(len(freq_axis))
        full_correction[valid_mask] = correction_gain_50
        
        # === 4. VERIFICA COMPATIBILITÀ FASI ===
        if not self.has_real_phases or not hasattr(self.parent.data_manager, 'phase_data'):
            QMessageBox.warning(self, "No Phase Data", 
                              "Normalization requires phase information (file version >= 3.0).")
            return
        
        fft_phases_int8 = self.parent.data_manager.phase_data[self.frame_index]
        
        # Parametri FFT
        fs = self.parent.data_manager.header_info.get('fs', 200000)
        fft_size = self.parent.data_manager.header_info.get('fft_size', 512)
        num_bins_full = fft_size // 2
        
        bin_freq = fs / fft_size
        bin_start = int(20000 / bin_freq)
        bin_end = int(80000 / bin_freq)
        num_received_bins = bin_end - bin_start + 1
        
        # ✅ FIX: Verifica che le dimensioni siano coerenti
        if len(fft_magnitudes) != len(freq_axis):
            print(f"⚠️ WARNING: FFT length mismatch: {len(fft_magnitudes)} vs {len(freq_axis)}")
            QMessageBox.warning(self, "Data Mismatch", "FFT data dimensions inconsistent.")
            return
        
        if len(fft_phases_int8) != num_received_bins:
            print(f"⚠️ WARNING: Phase data length mismatch: {len(fft_phases_int8)} vs {num_received_bins}")
        
        # === 5. APPLICA CORREZIONE SOLO ALLA PARTE 20-80kHz ===
        # Estrai solo la parte 20-80kHz dalla FFT originale (154 bins)
        fft_20_80khz = fft_magnitudes[valid_mask]
        
        # Applica correzione solo a questa parte
        normalized_fft_20_80khz = fft_20_80khz * correction_gain_50
        
        # === 6. RICOSTRUISCI SPETTRO COMPLETO (0-100kHz, 256 bins) ===
        full_spectrum_mag = np.zeros(num_bins_full, dtype=np.float32)
        full_spectrum_phase = np.zeros(num_bins_full, dtype=np.int8)
        
        # Inserisci i dati normalizzati 20-80kHz SENZA windowing (sarà applicata dopo)
        actual_bins_to_copy = min(len(normalized_fft_20_80khz), num_received_bins, len(fft_phases_int8))
        full_spectrum_mag[bin_start:bin_start + actual_bins_to_copy] = normalized_fft_20_80khz[:actual_bins_to_copy]
        full_spectrum_phase[bin_start:bin_start + actual_bins_to_copy] = fft_phases_int8[:actual_bins_to_copy]
        
        # === 7. CONVERTI FASI E CREA SPETTRO COMPLESSO ===
        fft_phases_rad = (full_spectrum_phase / 127.0) * np.pi
        complex_spectrum = full_spectrum_mag * np.exp(1j * fft_phases_rad)
        
        # === 7b. APPLICA TUKEY WINDOW ALLO SPETTRO COMPLESSO (FIX GIBBS) ===
        # ✅ IMPORTANTE: Window applicata allo spettro complesso, non solo alle magnitude
        # Questo elimina discontinuità sia in Re{X[k]} che in Im{X[k]}
        taper_bins = max(5, actual_bins_to_copy // 10)
        
        # Crea finestra Tukey per la regione 20-80kHz
        window_full = np.ones(num_bins_full)
        
        # Left taper (bins 51-66, cosine fade-in)
        for i in range(taper_bins):
            alpha = i / taper_bins
            window_full[bin_start + i] = 0.5 * (1 - np.cos(np.pi * alpha))
        
        # Right taper (bins 189-204, cosine fade-out)
        for i in range(taper_bins):
            alpha = i / taper_bins
            window_full[bin_start + actual_bins_to_copy - i - 1] = 0.5 * (1 - np.cos(np.pi * alpha))
        
        # Applica window allo spettro complesso (attenuazione graduale ai bordi)
        complex_spectrum = complex_spectrum * window_full
        
        # === 8. ESEGUI iFFT ===
        try:
            time_domain_signal = np.fft.irfft(complex_spectrum, n=fft_size)
        except Exception as e:
            QMessageBox.critical(self, "iFFT Error", f"Failed to compute normalized iFFT:\n{str(e)}")
            return
        
        self.signal_data_normalized = time_domain_signal
        
        # Statistiche
        gain_stats = {
            'max_gain_db': 20 * np.log10(np.max(correction_gain_50)),
            'min_gain_db': 20 * np.log10(np.min(correction_gain_50)),
        }
        
        print(f"✅ Normalized iFFT computed:")
        print(f"   Gain range: {gain_stats['min_gain_db']:.2f} to {gain_stats['max_gain_db']:.2f} dB")
        print(f"   Samples: {len(time_domain_signal)}")
    
    def _update_display(self):
        """Aggiorna display con curva corretta"""
        if self.is_normalized and self.signal_data_normalized is not None:
            # Mostra normalizzato (ROSSO) + overlay raw (tema)
            self.ifft_curve.setData(self.time_data, self.signal_data_normalized)
            self.ifft_curve.setPen({'color': 'red', 'width': 2})
            
            # Aggiungi overlay raw (accent_color dal tema, sottile, tratteggiato)
            if not hasattr(self, 'raw_overlay_curve'):
                self.raw_overlay_curve = self.plot_widget.plot_widget.plot(
                    self.time_data, self.signal_data_raw,
                    pen={'width': 1, 'style': QtCore.Qt.DashLine},
                    name='Raw iFFT'
                )
                # Applica accent color del tema alla curva overlay
                if self.parent and hasattr(self.parent, 'theme_manager'):
                    self.parent.theme_manager.apply_theme_to_plot(
                        plot_widget_name=self.plot_widget.plot_widget,
                        plot_instance=self.raw_overlay_curve
                    )
            else:
                self.raw_overlay_curve.setData(self.time_data, self.signal_data_raw)
            
            self.normalize_button.setText("Show Raw iFFT")
            self.info_label.setText("� Normalized iFFT (50% correction, ±2.9 dB error)")
            self.info_label.setStyleSheet("color: red; font-weight: bold; font-size: 10pt;")
            
            # Aggiorna legenda
            try:
                self.plot_widget.plot_widget.addLegend()
            except:
                pass
        else:
            # Mostra raw (ri-applica accent_color del tema)
            self.ifft_curve.setData(self.time_data, self.signal_data_raw)
            
            # Ri-applica il tema per ripristinare l'accent color
            if self.parent and hasattr(self.parent, 'theme_manager'):
                self.parent.theme_manager.apply_theme_to_plot(
                    plot_widget_name=self.plot_widget.plot_widget,
                    plot_instance=self.ifft_curve
                )
            
            # Rimuovi overlay
            if hasattr(self, 'raw_overlay_curve'):
                try:
                    self.plot_widget.plot_widget.removeItem(self.raw_overlay_curve)
                    del self.raw_overlay_curve
                except:
                    pass
            
            self.normalize_button.setText("Apply 50% Normalization")
            self.info_label.setText("📊 Raw iFFT signal (no correction)")
            self.info_label.setStyleSheet("color: #888; font-size: 10pt;")
    
    def toggle_envelope(self):
        """Toggle visualizzazione inviluppo di Hilbert"""
        self.show_envelope = not self.show_envelope
        
        if self.show_envelope:
            # CALCOLA E MOSTRA ENVELOPE
            self._compute_and_show_envelope()
        else:
            # NASCONDI ENVELOPE
            if self.envelope_curve is not None:
                self.plot_widget.plot_widget.removeItem(self.envelope_curve)
                self.envelope_curve = None
            if self.peak_line is not None:
                self.plot_widget.plot_widget.removeItem(self.peak_line)
                self.peak_line = None
            self.envelope_button.setText("Show Hilbert Envelope")
    
    def _compute_and_show_envelope(self):
        """Calcola e visualizza l'inviluppo di Hilbert"""
        # Usa il segnale corrente (raw o normalized)
        current_signal = self.signal_data_normalized if self.is_normalized else self.signal_data_raw
        
        print("🔧 Computing Hilbert envelope...")
        
        # Calcola inviluppo
        self.envelope_data = compute_hilbert_envelope(current_signal)
        
        # Trova picco
        peak_idx, peak_amp = find_peak(current_signal)
        peak_time = self.time_data[peak_idx]
        
        print(f"✅ Envelope computed:")
        print(f"   Peak at t = {peak_time:.6f} s (sample {peak_idx})")
        print(f"   Peak amplitude: {peak_amp:.6f} V")
        
        # Visualizza inviluppo (ROSSO, SPESSO)
        if self.envelope_curve is None:
            self.envelope_curve = self.plot_widget.plot_widget.plot(
                self.time_data, self.envelope_data,
                pen={'color': 'red', 'width': 3},
                name='Hilbert Envelope'
            )
        else:
            self.envelope_curve.setData(self.time_data, self.envelope_data)
        
        # Mostra linea verticale al picco
        if self.peak_line is None:
            self.peak_line = self.plot_widget.plot_widget.addLine(
                x=peak_time,
                pen={'color': 'yellow', 'width': 2, 'style': QtCore.Qt.DashLine},
                label='Peak'
            )
        else:
            self.peak_line.setValue(peak_time)
        
        self.envelope_button.setText("Hide Hilbert Envelope")
    
    def analyze_decay(self):
        """Analizza il decadimento post-picco per rilevare click ultrasonici"""
        # Usa il segnale corrente (raw o normalized)
        current_signal = self.signal_data_normalized if self.is_normalized else self.signal_data_raw
        
        # ✅ RICALCOLA SEMPRE ENVELOPE per assicurare coerenza con segnale corrente
        self.envelope_data = compute_hilbert_envelope(current_signal)
        
        # ✅ BUG FIX: Trova il picco sull'ENVELOPE (non sul segnale raw) per coerenza
        # check_decay() riceve l'envelope e usa peak_idx come indice in esso.
        # Se trovassimo il picco sul raw, potremmo puntare in una posizione diversa
        # dall'envelope a causa delle oscillazioni della portante.
        peak_idx, peak_amp = find_peak(self.envelope_data)
        peak_time = self.time_data[peak_idx]
        
        # ✅ GESTIONE SPILL: Recupera frame successivo se necessario
        next_frame_signal = None
        if peak_idx > 380:  # Picco vicino a fine frame
            # Verifica se esiste frame successivo
            if (self.frame_index is not None and 
                self.parent and hasattr(self.parent, 'data_manager') and
                self.frame_index + 1 < len(self.parent.data_manager.fft_data)):
                
                try:
                    # Recupera FFT e fasi del frame successivo
                    next_fft = self.parent.data_manager.fft_data[self.frame_index + 1]
                    
                    if self.is_normalized and self.has_real_phases:
                        # Se in modalità normalizzata, calcola iFFT normalizzato del prossimo frame
                        print("   🔧 Computing normalized iFFT for next frame (spill handling)...")
                        next_frame_signal = self._compute_ifft_for_frame(self.frame_index + 1, normalized=True)
                    elif self.has_real_phases:
                        # Se in modalità raw con fasi reali
                        next_frame_signal = self._compute_ifft_for_frame(self.frame_index + 1, normalized=False)
                    else:
                        # Fallback: usa solo dati FFT (senza fasi, qualità ridotta)
                        print("   ⚠️ Next frame has no phase data, decay analysis may be inaccurate")
                        next_frame_signal = None
                        
                except Exception as e:
                    print(f"   ⚠️ Failed to load next frame: {e}")
                    next_frame_signal = None
        
        # Analizza decadimento (con o senza frame successivo)
        print(f"\n🔍 Decay Analysis {'(NORMALIZED)' if self.is_normalized else '(RAW)'}:")
        print(f"   Peak at t = {peak_time:.6f} s (sample {peak_idx}/{len(current_signal)})")
        print(f"   Peak amplitude: {peak_amp:.6f} V")
        
        decay_results = check_decay(self.envelope_data, peak_idx, next_frame_signal=next_frame_signal)
        
        # Stampa risultati
        print(f"\n📊 Decay Metrics (120-Sample Logarithmic Fit):")
        print(f"   Window analyzed: {decay_results['n_samples']} samples ({decay_results['n_samples']/200:.3f} ms)")
        if decay_results.get('used_next_frame', False):
            print(f"   ✅ Extended analysis using next frame data")
        
        # Legacy info (4 sub-windows per reference)
        print(f"\n   📋 Legacy Sub-window Energies (reference only):")
        for i, E in enumerate(decay_results['energies'], 1):
            print(f"      W{i}: {E:.9f} V² = {E*1e6:.3f} mV²")
        
        print(f"\n   📈 PRIMARY ANALYSIS: 120-Sample Logarithmic Fit")
        print(f"      Method: Sample-by-sample on Hilbert envelope")
        print(f"      Slope (log): {decay_results['slope_log']:.6f}")
        print(f"      R² (log): {decay_results['r_squared_log']:.4f}  ⭐ DESCRIPTIVE ONLY (v3.0)")
        print(f"      τ (tau): {decay_results['tau_ms']:.3f} ms" if decay_results['tau_ms'] > 0 else "      τ (tau): N/A (invalid slope)")
        
        # v3.0: Show global decay trend (Criterion 3)
        E_W1 = decay_results.get('E_W1', 0)
        E_W4 = decay_results.get('E_W4', 0)
        decay_ratio = E_W1 / E_W4 if E_W4 > 0 else 0
        print(f"\n   ⭐ v3.0 Criterion 3 (Global Decay):")
        print(f"      E_W1 / E_W4 = {decay_ratio:.2f}")
        print(f"      Criterion 3: {'✅ PASS (E_W1 > E_W4)' if E_W1 > E_W4 else '❌ FAIL (E_W1 ≤ E_W4)'}")
        
        print(f"\n   ✅ Quality checks:")
        print(f"      Near frame end: {'⚠️ YES (possible spill)' if decay_results['near_end'] else '✅ NO'}")
        
        # Valutazione finale (v3.0: descriptive only, not validation)
        r2_log = decay_results['r_squared_log']
        if E_W1 > E_W4 and r2_log >= 0.70:
            verdict = "✅ GOOD DECAY TREND (E_W1>E_W4 + R²≥0.70, likely click)"
            color = "green"
        elif E_W1 > E_W4 and r2_log >= 0.50:
            verdict = "⚠️ MODERATE DECAY (E_W1>E_W4 + R²≥0.50, check SNR/PRE_ratio)"
            color = "orange"
        else:
            verdict = "❌ POOR DECAY (E_W1≤E_W4 OR R²<0.50, likely not a click)"
            color = "red"
        
        print(f"\n🎯 Verdict (descriptive): {verdict}")
        print(f"   ⚠️ NOTE: Final validation requires SNR and PRE_ratio tests (Stage 3)")
        
        # Mostra popup con risultati (CUSTOM DIALOG per temi corretti)
        self._show_decay_results_dialog(
            frame_index=self.frame_index,
            is_normalized=self.is_normalized,
            peak_time=peak_time,
            peak_idx=peak_idx,
            peak_amp=peak_amp,
            decay_results=decay_results,
            verdict=verdict,
            verdict_color=color
        )
    
    def _compute_ifft_for_frame(self, frame_index, normalized=False):
        """
        Calcola iFFT per un frame specifico (helper per gestione spill).
        
        Parameters:
        -----------
        frame_index : int
            Indice del frame da processare
        normalized : bool
            Se True, applica correzione 50% microfono
        
        Returns:
        --------
        np.ndarray : Segnale time-domain (512 samples)
        """
        if not self.parent or not hasattr(self.parent, 'data_manager'):
            return None
        
        dm = self.parent.data_manager
        
        if frame_index >= len(dm.fft_data):
            return None
        
        # Recupera dati
        fft_magnitudes = dm.fft_data[frame_index]
        freq_axis = dm.frequency_axis
        
        # Parametri FFT
        fs = dm.header_info.get('fs', 200000)
        fft_size = dm.header_info.get('fft_size', 512)
        num_bins_full = fft_size // 2
        
        bin_freq = fs / fft_size
        bin_start = int(20000 / bin_freq)
        bin_end = int(80000 / bin_freq)
        
        # Inizializza spettro completo
        full_spectrum_mag = np.zeros(num_bins_full, dtype=np.float32)
        full_spectrum_phase = np.zeros(num_bins_full, dtype=np.int8)
        
        # Se normalizzato, applica correzione
        if normalized:
            datasheet_freq_khz = np.array([20, 25, 30, 40, 50, 60, 70, 80])
            datasheet_response_db = np.array([8.0, 10.5, 6.0, -2.0, -6.0, -7.0, -6.0, -4.0])
            datasheet_freq_hz = datasheet_freq_khz * 1000
            
            valid_mask = (freq_axis >= 20000) & (freq_axis <= 80000)
            freq_range = freq_axis[valid_mask]
            
            mic_response_db = np.interp(freq_range, datasheet_freq_hz, datasheet_response_db)
            correction_gain_50 = 10 ** (-mic_response_db * 0.5 / 20.0)
            
            fft_corrected = fft_magnitudes[valid_mask] * correction_gain_50
        else:
            valid_mask = (freq_axis >= 20000) & (freq_axis <= 80000)
            fft_corrected = fft_magnitudes[valid_mask]
        
        # Inserisci magnitude SENZA windowing (sarà applicata dopo)
        actual_bins = min(len(fft_corrected), bin_end - bin_start + 1)
        full_spectrum_mag[bin_start:bin_start + actual_bins] = fft_corrected[:actual_bins]
        
        # Inserisci fasi (se disponibili)
        if len(dm.phase_data) > frame_index:
            phase_data = dm.phase_data[frame_index]
            actual_phase_bins = min(len(phase_data), actual_bins)
            full_spectrum_phase[bin_start:bin_start + actual_phase_bins] = phase_data[:actual_phase_bins]
        
        # Converti fasi e crea spettro complesso
        fft_phases_rad = (full_spectrum_phase / 127.0) * np.pi
        complex_spectrum = full_spectrum_mag * np.exp(1j * fft_phases_rad)
        
        # ✅ APPLICA TUKEY WINDOW ALLO SPETTRO COMPLESSO (FIX GIBBS CORRETTO)
        taper_bins = max(5, actual_bins // 10)
        window_full = np.ones(num_bins_full)
        
        # Left taper (cosine fade-in)
        for i in range(taper_bins):
            alpha = i / taper_bins
            window_full[bin_start + i] = 0.5 * (1 - np.cos(np.pi * alpha))
        
        # Right taper (cosine fade-out)
        for i in range(taper_bins):
            alpha = i / taper_bins
            window_full[bin_start + actual_bins - i - 1] = 0.5 * (1 - np.cos(np.pi * alpha))
        
        # Applica window allo spettro complesso
        complex_spectrum = complex_spectrum * window_full
        
        # iFFT
        try:
            time_domain_signal = np.fft.irfft(complex_spectrum, n=fft_size)
            return time_domain_signal
        except Exception as e:
            print(f"   ❌ iFFT error for frame {frame_index}: {e}")
            return None
    
    def _show_decay_results_dialog(self, frame_index, is_normalized, peak_time, peak_idx, 
                                   peak_amp, decay_results, verdict, verdict_color):
        """Mostra dialog personalizzato con risultati decay analysis (tema-aware, v3.0)."""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QTextEdit, QPushButton, QHBoxLayout
        from PySide6.QtCore import Qt
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Decay Analysis Results (v3.0)")
        dialog.setMinimumSize(640, 820)
        
        layout = QVBoxLayout(dialog)
        
        # TITOLO
        title_label = QLabel(f"<b style='font-size:16pt;'>Frame {frame_index if frame_index is not None else '?'}</b>")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        mode_label = QLabel(f"<i>{'50% Normalized iFFT' if is_normalized else 'Raw iFFT (no correction)'}</i>")
        mode_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(mode_label)
        
        text_widget = QTextEdit()
        text_widget.setReadOnly(True)
        
        # ── Calcola PRE_ratio e SNR per il frame corrente (coerente col detector) ──
        if self.parent and hasattr(self.parent, 'data_manager'):
            dm = self.parent.data_manager
            noise_rms = getattr(dm, '_cached_noise_rms', None)
            snr_str = "N/A (run detector first)"
            snr_pass_str = "⚠️ unknown"
        else:
            noise_rms = None
            snr_str = "N/A"
            snr_pass_str = "⚠️ unknown"

        # Recupera segnale corrente per calcoli
        current_signal = self.signal_data_normalized if self.is_normalized else self.signal_data_raw

        # PRE_ratio: finestra prima del picco (coerente con detector)
        pre_end = max(0, peak_idx - 10)
        if pre_end >= 10:
            E_pre_v = float(np.mean(current_signal[:pre_end] ** 2))
        else:
            E_pre_v = 0.0
        post_end_idx = min(peak_idx + 120, len(current_signal))
        E_post_v = float(np.mean(current_signal[peak_idx:post_end_idx] ** 2)) if post_end_idx > peak_idx else 1e-9
        pre_ratio_v = E_pre_v / E_post_v if E_post_v > 1e-12 else 999.0

        # SNR se disponibile
        if noise_rms is not None and noise_rms > 0:
            snr_v = peak_amp / noise_rms
            snr_str = f"{snr_v:.2f}  ({20*np.log10(snr_v):.1f} dB)"
            snr_pass_str = "✅ PASS (≥5.0)" if snr_v >= 5.0 else "❌ FAIL (<5.0)"
        else:
            snr_v = None
            snr_str = "N/A – run detector to cache noise RMS"
            snr_pass_str = "⚠️ unknown"

        E_W1 = decay_results.get('E_W1', 0)
        E_W4 = decay_results.get('E_W4', 0)
        decay_ratio_v = E_W1 / E_W4 if E_W4 > 1e-15 else 999.0
        pre_pass_str = "✅ PASS (<0.15)" if pre_ratio_v < 0.15 else "❌ FAIL (≥0.15)"
        crit3_pass_str = "✅ PASS (E_W1>E_W4)" if E_W1 > E_W4 else "❌ FAIL (E_W1≤E_W4)"

        html_content = f"""
<div style='font-family: monospace; font-size: 11pt; line-height: 1.6;'>

<!-- ═══ SECTION 1: PEAK ═══ -->
<p><b style='font-size:13pt; text-decoration:underline;'>Peak Information</b></p>
<table style='margin-left:20px; border-collapse:collapse;'>
<tr><td style='padding:4px; width:170px;'>Time in frame:</td>
    <td style='padding:4px;'><b>{peak_time:.6f} s</b>  (sample {peak_idx} / {len(current_signal)})</td></tr>
<tr><td style='padding:4px;'>Amplitude (envelope):</td>
    <td style='padding:4px;'><b>{peak_amp:.6f} V</b>  =  {peak_amp*1000:.4f} mV</td></tr>
</table>

<!-- ═══ SECTION 2: V3.0 CRITERIA ═══ -->
<p style='margin-top:12px;'><b style='font-size:13pt; text-decoration:underline;'>v3.0 Validation Criteria</b>
<span style='font-size:9pt; color:gray;'> (used by Automatic Click Detector)</span></p>
<table style='margin-left:20px; border-collapse:collapse; width:95%;'>
<tr style='background-color:rgba(100,100,255,0.12);'>
    <th style='padding:6px; text-align:left;'>Criterion</th>
    <th style='padding:6px; text-align:right;'>Value</th>
    <th style='padding:6px; text-align:left;'>Default threshold</th>
    <th style='padding:6px; text-align:left;'>Result</th>
</tr>
<tr>
    <td style='padding:5px;'><b>1. SNR</b> = peak / noise_rms</td>
    <td style='padding:5px; text-align:right;'><b>{snr_str}</b></td>
    <td style='padding:5px;'>&gt; 5.0  (14 dB)</td>
    <td style='padding:5px;'>{snr_pass_str}</td>
</tr>
<tr style='background-color:rgba(128,128,128,0.08);'>
    <td style='padding:5px;'><b>2. PRE_ratio</b> = E_pre / E_post</td>
    <td style='padding:5px; text-align:right;'><b>{pre_ratio_v:.4f}</b></td>
    <td style='padding:5px;'>&lt; 0.15  (silence before)</td>
    <td style='padding:5px;'>{pre_pass_str}</td>
</tr>
<tr>
    <td style='padding:5px;'><b>3. Global decay</b> E_W1 / E_W4</td>
    <td style='padding:5px; text-align:right;'><b>{decay_ratio_v:.2f}×</b>  (W1={E_W1:.3e}, W4={E_W4:.3e})</td>
    <td style='padding:5px;'>E_W1 &gt; E_W4</td>
    <td style='padding:5px;'>{crit3_pass_str}</td>
</tr>
</table>
<p style='margin-left:20px; font-size:9pt; color:gray;'>
SNR requires noise_rms cached by the detector (Run Detector once to populate it).
PRE_ratio uses samples [0 : peak-10] as pre-click window.
</p>

<!-- ═══ SECTION 3: SUB-WINDOW ENERGIES ═══ -->
<p style='margin-top:12px;'><b style='font-size:13pt; text-decoration:underline;'>Post-peak Sub-windows  (0.6 ms / 4)</b></p>
<table style='margin-left:20px; border-collapse:collapse; width:80%;'>
<tr style='background-color:rgba(128,128,128,0.2);'>
    <th style='padding:6px; text-align:left;'>Window</th>
    <th style='padding:6px; text-align:right;'>Duration</th>
    <th style='padding:6px; text-align:right;'>Energy (mV²)</th>
    <th style='padding:6px; text-align:left;'>Trend</th>
</tr>
<tr><td style='padding:4px;'>W1</td><td style='padding:4px; text-align:right;'>0 – 150 μs</td>
    <td style='padding:4px; text-align:right;'><b>{decay_results['energies'][0]*1e6:.4f}</b></td>
    <td style='padding:4px;'>⬆ reference</td></tr>
<tr style='background-color:rgba(128,128,128,0.08);'>
<td style='padding:4px;'>W2</td><td style='padding:4px; text-align:right;'>150 – 300 μs</td>
    <td style='padding:4px; text-align:right;'><b>{decay_results['energies'][1]*1e6:.4f}</b></td>
    <td style='padding:4px;'>{'⬇' if decay_results['energies'][1] < decay_results['energies'][0] else '⬆'}</td></tr>
<tr><td style='padding:4px;'>W3</td><td style='padding:4px; text-align:right;'>300 – 450 μs</td>
    <td style='padding:4px; text-align:right;'><b>{decay_results['energies'][2]*1e6:.4f}</b></td>
    <td style='padding:4px;'>{'⬇' if decay_results['energies'][2] < decay_results['energies'][1] else '⬆'}</td></tr>
<tr style='background-color:rgba(128,128,128,0.08);'>
<td style='padding:4px;'>W4</td><td style='padding:4px; text-align:right;'>450 – 600 μs</td>
    <td style='padding:4px; text-align:right;'><b>{decay_results['energies'][3]*1e6:.4f}</b></td>
    <td style='padding:4px;'>{'⬇' if decay_results['energies'][3] < decay_results['energies'][2] else '⬆'}</td></tr>
</table>
<p style='margin-left:20px; font-size:9pt; color:gray;'>Monotone (W1&gt;W2&gt;W3&gt;W4): {'✅ YES' if decay_results['monotone'] else '❌ NO'}</p>

<!-- ═══ SECTION 4: DECAY FIT (DESCRIPTIVE) ═══ -->
<p style='margin-top:12px;'><b style='font-size:13pt; text-decoration:underline;'>Exponential Decay Fit</b>
<span style='font-size:9pt; color:gray;'> (descriptive only – NOT used for validation in v3.0)</span></p>
<table style='margin-left:20px; border-collapse:collapse; width:90%;'>
<tr>
    <td style='padding:4px; width:200px;'>Slope (log-linear):</td>
    <td style='padding:4px;'><b>{decay_results['slope_log']:.6f}</b>
        {'  ↳ ✅ decay' if decay_results['slope_log'] < 0 else '  ↳ ❌ growth'}</td>
</tr>
<tr style='background-color:rgba(128,128,128,0.08);'>
    <td style='padding:4px;'>R² (log-linear):</td>
    <td style='padding:4px;'><b>{decay_results['r_squared_log']:.4f}</b>
        {'  → excellent' if decay_results['r_squared_log'] >= 0.85 else '  → good' if decay_results['r_squared_log'] >= 0.70 else '  → fair' if decay_results['r_squared_log'] >= 0.50 else '  → poor'}</td>
</tr>
<tr>
    <td style='padding:4px;'>τ (decay time constant):</td>
    <td style='padding:4px;'><b>{f"{decay_results['tau_ms']:.3f} ms" if decay_results['tau_ms'] > 0 else "N/A (slope ≥ 0)"}</b>
        {f"  ({'⚡ fast' if decay_results['tau_ms'] < 0.1 else '📊 typical' if decay_results['tau_ms'] < 0.5 else '🔊 slow'} click)" if decay_results['tau_ms'] > 0 else ""}</td>
</tr>
<tr style='background-color:rgba(128,128,128,0.08);'>
    <td style='padding:4px;'>Samples analyzed:</td>
    <td style='padding:4px;'>{decay_results['window_samples']} samples  ({decay_results['window_samples'] / 200:.3f} ms)
        {'  ✅ next-frame extended' if decay_results.get('used_next_frame', False) else ''}</td>
</tr>
</table>

<!-- ═══ SECTION 5: FLAGS ═══ -->
<p style='margin-top:10px; margin-left:20px; font-size:10pt;'>
<b>Near frame end (&gt;74%):</b> {'⚠️ YES – spill possible' if decay_results['near_end'] else '✅ NO'}
</p>

<!-- ═══ VERDICT ═══ -->
<br>
<p style='text-align:center; font-size:14pt; font-weight:bold; padding:10px;
          background-color:rgba({self._verdict_color_to_rgb(verdict_color)},0.3); border-radius:5px;'>
{verdict}
</p>
<p style='text-align:center; font-size:9pt; color:gray;'>
  Verdict is based on decay shape only. Full validation needs Criterion 1 (SNR) + Criterion 2 (PRE_ratio).
</p>

</div>
"""
        
        text_widget.setHtml(html_content)
        layout.addWidget(text_widget)
        
        # PULSANTE CHIUDI
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        close_button = QPushButton("Close")
        close_button.setMinimumWidth(100)
        close_button.clicked.connect(dialog.accept)
        button_layout.addWidget(close_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Applica tema
        if self.parent and hasattr(self.parent, 'theme_manager'):
            saved_theme = self.parent.theme_manager.load_saved_theme()
            self.parent.theme_manager.apply_theme(dialog, saved_theme)
            if 'light' in saved_theme.lower():
                dialog.setStyleSheet("""
                    QDialog { background-color: white; color: black; }
                    QLabel { color: black; }
                    QTextEdit { background-color: white; color: black; }
                    QPushButton { background-color: #f0f0f0; color: black;
                                  border: 1px solid #ccc; padding: 5px; border-radius: 3px; }
                    QPushButton:hover { background-color: #e0e0e0; }
                """)

        dialog.exec()
    
    def _verdict_color_to_rgb(self, color_name):
        """Converti nome colore in RGB per HTML"""
        color_map = {
            'green': '0, 200, 0',
            'orange': '255, 165, 0',
            'red': '255, 0, 0'
        }
        return color_map.get(color_name, '128, 128, 128')
    

# ============================================================================
# HILBERT ENVELOPE ANALYSIS FUNCTIONS (CLICK DETECTION)
# ============================================================================

def estimate_noise_offline(data_manager, energy_threshold_multiplier=4.0, max_samples=500):
    """
    Stima il noise RMS da frame "vuoti" per calcolo SNR offline.
    
    **MOTIVAZIONE**: Per validare click tramite SNR (Criterio 1), serve un riferimento
    di rumore che NON provenga dal frame candidato stesso. In modalità offline (analisi
    completa file), possiamo campionare frame vuoti rappresentativi dell'intero recording.
    
    **STRATEGIA**:
    1. Pass 1: Calcola energia E_frame per tutti i frame
    2. Identifica "empty frames" dove E_frame < μ + multiplier*σ (tipicamente 4σ)
    3. Pass 2: Campiona random ≤max_samples empty frames
    4. Ricostruisci iFFT di ogni frame campionato
    5. Calcola RMS di ogni iFFT
    6. Restituisci noise_rms = mean(tutti i RMS)
    
    **VANTAGGI vs rolling buffer**:
    - Usa campione rappresentativo di tutto il recording (non solo primi 200 frames)
    - Adattivo al livello di rumore specifico di ogni file
    - Statisticamente più robusto (500 campioni vs 200)
    
    Parameters:
    -----------
    data_manager : AudioDataManager
        Data manager con fft_data, phase_data, header_info
    energy_threshold_multiplier : float
        Moltiplicatore per identificare frame vuoti (default: 4.0)
    max_samples : int
        Numero massimo di frame da campionare (default: 500)
    
    Returns:
    --------
    dict : {
        'noise_rms': float,           # RMS medio dei frame vuoti (V)
        'noise_std': float,           # Std dev dei RMS (V)
        'n_samples': int,             # Numero frame campionati
        'n_empty_frames': int,        # Numero totale frame vuoti nel file
        'threshold_used': float,      # Threshold energia usato (V)
    }
    """
    print("\n" + "="*80)
    print("🔍 OFFLINE NOISE ESTIMATION")
    print("="*80)
    
    # Pass 1: Identifica frame vuoti
    print(f"📊 Pass 1: Identifying empty frames...")
    
    # Usa fft_means già precalcolato (con outlier detection)
    energies = data_manager.fft_means
    mu_E = data_manager.fft_mean
    sigma_E = data_manager.fft_std
    
    threshold_energy = mu_E + energy_threshold_multiplier * sigma_E
    
    # Frame vuoti: energia sotto threshold
    empty_mask = energies < threshold_energy
    empty_indices = np.where(empty_mask)[0]
    
    n_empty = len(empty_indices)
    print(f"✅ Found {n_empty}/{len(energies)} empty frames ({n_empty/len(energies)*100:.1f}%)")
    print(f"   Threshold: {threshold_energy*1000:.3f} mV (μ + {energy_threshold_multiplier}σ)")
    
    if n_empty == 0:
        print("⚠️ WARNING: No empty frames found! Using global mean as fallback.")
        return {
            'noise_rms': mu_E,
            'noise_std': sigma_E,
            'n_samples': 0,
            'n_empty_frames': 0,
            'threshold_used': threshold_energy,
        }
    
    # Pass 2: Campiona e ricostruisci
    print(f"\n📊 Pass 2: Sampling {min(max_samples, n_empty)} frames and computing iFFT RMS...")
    
    # Random sampling (riproducibile)
    np.random.seed(42)
    n_to_sample = min(max_samples, n_empty)
    sampled_indices = np.random.choice(empty_indices, size=n_to_sample, replace=False)
    
    rms_values = []
    
    # Parametri FFT per ricostruzione
    fs = data_manager.header_info.get('fs', 200000)
    fft_size = data_manager.header_info.get('fft_size', 512)
    num_bins_full = fft_size // 2
    bin_freq = fs / fft_size
    bin_start = int(20000 / bin_freq)
    bin_end = int(80000 / bin_freq)
    
    for idx in sampled_indices:
        # Ottieni FFT
        fft_mags = data_manager.fft_data[idx]
        
        # Ricostruisci spettro completo
        full_spectrum_mag = np.zeros(num_bins_full, dtype=np.float32)
        full_spectrum_phase = np.zeros(num_bins_full, dtype=np.int8)
        
        actual_bins = min(len(fft_mags), bin_end - bin_start + 1)
        full_spectrum_mag[bin_start:bin_start + actual_bins] = fft_mags[:actual_bins]
        
        # Usa fasi se disponibili
        if len(data_manager.phase_data) > idx:
            phase_data = data_manager.phase_data[idx]
            actual_phase_bins = min(len(phase_data), actual_bins)
            full_spectrum_phase[bin_start:bin_start + actual_phase_bins] = phase_data[:actual_phase_bins]
        
        # Crea spettro complesso
        fft_phases_rad = (full_spectrum_phase / 127.0) * np.pi
        complex_spectrum = full_spectrum_mag * np.exp(1j * fft_phases_rad)
        
        # ✅ APPLICA TUKEY WINDOW ALLO SPETTRO COMPLESSO (noise estimation non critica, ma consistente)
        taper_bins = max(5, actual_bins // 10)
        window_full = np.ones(num_bins_full)
        
        for i in range(taper_bins):
            alpha = i / taper_bins
            window_full[bin_start + i] = 0.5 * (1 - np.cos(np.pi * alpha))
            window_full[bin_start + actual_bins - i - 1] = 0.5 * (1 - np.cos(np.pi * alpha))
        
        complex_spectrum = complex_spectrum * window_full
        
        # iFFT
        try:
            time_signal = np.fft.irfft(complex_spectrum, n=fft_size)
            rms = np.sqrt(np.mean(time_signal ** 2))
            rms_values.append(rms)
        except:
            continue  # Skip frames con errori
    
    if len(rms_values) == 0:
        print("⚠️ ERROR: Failed to reconstruct any frame! Using energy fallback.")
        return {
            'noise_rms': mu_E,
            'noise_std': sigma_E,
            'n_samples': 0,
            'n_empty_frames': n_empty,
            'threshold_used': threshold_energy,
        }
    
    rms_values = np.array(rms_values)
    noise_rms = float(np.mean(rms_values))
    noise_std = float(np.std(rms_values))
    
    print(f"✅ Noise estimation complete:")
    print(f"   RMS (mean): {noise_rms*1000:.6f} mV")
    print(f"   RMS (std):  {noise_std*1000:.6f} mV")
    print(f"   Samples:    {len(rms_values)}/{n_to_sample}")
    print(f"   Min RMS:    {np.min(rms_values)*1000:.6f} mV")
    print(f"   Max RMS:    {np.max(rms_values)*1000:.6f} mV")
    print("="*80 + "\n")
    
    return {
        'noise_rms': noise_rms,
        'noise_std': noise_std,
        'n_samples': len(rms_values),
        'n_empty_frames': n_empty,
        'threshold_used': threshold_energy,
    }


def compute_hilbert_envelope(signal: np.ndarray) -> np.ndarray:
    """
    Calcola l'inviluppo istantaneo del segnale usando la trasformata di Hilbert.
    
    Per un click ultrasonico (sinusoide smorzata), l'inviluppo mostra il decadimento
    esponenziale che lo caratterizza.
    
    Parameters:
    -----------
    signal : np.ndarray
        Segnale time-domain (es: output di iFFT)
    
    Returns:
    --------
    np.ndarray : Instantaneous amplitude envelope A[n] = |analytic_signal[n]|
    """
    # Segnale analitico = signal + j*hilbert(signal)
    analytic_signal = hilbert(signal)
    
    # Inviluppo = modulo del segnale analitico
    envelope = np.abs(analytic_signal)
    
    return envelope


def find_peak(signal: np.ndarray) -> tuple:
    """
    Trova il picco massimo nel segnale (in valore assoluto).
    
    Parameters:
    -----------
    signal : np.ndarray
        Segnale time-domain o inviluppo
    
    Returns:
    --------
    tuple : (peak_index, peak_amplitude)
        - peak_index: Indice del massimo assoluto
        - peak_amplitude: Valore assoluto del picco
    """
    abs_signal = np.abs(signal)
    peak_index = int(np.argmax(abs_signal))
    peak_amplitude = float(abs_signal[peak_index])
    
    return peak_index, peak_amplitude


def compute_decay_r2(post_peak_window: np.ndarray, fs=200000) -> dict:
    """
    Calcola R² del fit logaritmico SAMPLE-BY-SAMPLE sull'envelope post-picco.
    
    ⭐ METODO: Usa 120 campioni individuali per robustezza contro oscillazioni portante.
    
    **Motivazione**: Le oscillazioni della portante del segnale smorzato causano
    fluttuazioni locali nell'envelope. Con 120 campioni sample-by-sample, le 
    oscillazioni si mediano automaticamente nella regressione.
    
    Modello fisico: A(t) = A₀·exp(-t/τ)
    Log-linearizzazione: log(A(t)) = log(A₀) - t/τ
    
    ⚠️ IMPORTANTE: R² NON è usato come criterio di validazione (removed in v3.0).
    R² e tau_ms sono FEATURE DESCRITTIVE per analisi statistica post-rilevamento.
    
    Parameters:
    -----------
    post_peak_window : np.ndarray
        Finestra di envelope POST-PICCO (già estratta dal chiamante)
        Tipicamente 120 campioni (0.6 ms @ 200 ksps)
    fs : int
        Sampling rate (default: 200000 Hz)
    
    Returns:
    --------
    dict : {
        'r2_log': float,          # R² del fit logaritmico (DESCRIPTIVE ONLY)
        'slope_log': float,       # Pendenza fit (negativo = decay)
        'tau_ms': float,          # Costante di tempo in ms (DESCRIPTIVE ONLY)
        'n_samples': int,         # Numero campioni effettivamente usati
    }
    """
    n_samples = len(post_peak_window)
    
    # Guard: campioni insufficienti
    if n_samples < 20:
        return {
            'r2_log': 0.0,
            'slope_log': 0.0,
            'tau_ms': -1.0,
            'n_samples': n_samples,
        }
    
    # Guard: sostituisci valori zero/negativi con epsilon piccolo
    epsilon = 1e-9
    post_peak_safe = np.maximum(post_peak_window, epsilon)
    
    # Log-transform
    log_envelope = np.log(post_peak_safe)
    
    # Array indici campioni: n = [0, 1, 2, ..., n_samples-1]
    n_array = np.arange(n_samples, dtype=float)
    
    try:
        # Fit lineare: log(A[n]) = intercept + slope · n
        slope_log, intercept = np.polyfit(n_array, log_envelope, deg=1)
        
        # R² su scala logaritmica
        log_pred = slope_log * n_array + intercept
        ss_res = np.sum((log_envelope - log_pred) ** 2)
        ss_tot = np.sum((log_envelope - np.mean(log_envelope)) ** 2)
        r2_log = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        
        # Estrazione τ
        if slope_log < 0:
            tau_samples = -1.0 / slope_log
            tau_ms = tau_samples * 1000.0 / fs  # Converti in ms
        else:
            tau_ms = -1.0  # Invalido (crescita invece di decay)
        
    except Exception as e:
        # Fallback in caso di errore numerico
        slope_log = 0.0
        r2_log = 0.0
        tau_ms = -1.0
    
    return {
        'r2_log': float(r2_log),
        'slope_log': float(slope_log),
        'tau_ms': float(tau_ms),
        'n_samples': int(n_samples),
    }


def check_decay(envelope: np.ndarray, peak_idx: int, fs=200000, next_frame_signal=None) -> dict:
    """
    Analizza il decadimento post-picco e calcola feature descrittive.
    
    Click ultrasonici = sinusoidi smorzate con decadimento esponenziale in 0.1-0.6 ms.
    
    ⚠️ IMPORTANTE: Questa funzione calcola SOLO feature descrittive (r2_log, tau_ms, energies).
    NON valida/classifica il click. La validazione avviene nel pipeline di detect_clicks()
    usando i nuovi criteri: SNR, PRE_ratio, E_W1>E_W4.
    
    ✅ GESTIONE SPILL: Se il picco cade vicino alla fine del frame e next_frame_signal
    è fornito, concatena i dati dal frame successivo per analisi completa.
    
    Parameters:
    -----------
    envelope : np.ndarray
        Inviluppo di Hilbert del segnale
    peak_idx : int
        Indice del picco massimo
    fs : int
        Sampling rate (default: 200000 Hz = 200 ksps)
    next_frame_signal : np.ndarray, optional
        Segnale del frame successivo per concatenazione in caso di spill
    
    Returns:
    --------
    dict : {
        # DECAY FIT (DESCRIPTIVE ONLY - not used for validation)
        'r_squared_log': float,              # R² del fit logaritmico
        'slope_log': float,                  # Pendenza fit log
        'tau_ms': float,                     # Costante di decadimento in ms
        'n_samples': int,                    # Numero campioni usati nel fit
        
        # SUB-WINDOW ENERGIES (for validation criteria)
        'energies': [E1, E2, E3, E4],       # Energie medie 4 sub-windows (30 samples each)
        'E_W1': float,                       # First sub-window energy
        'E_W4': float,                       # Last sub-window energy
        
        # METADATA
        'near_end': bool,                    # True se picco vicino a fine frame
        'used_next_frame': bool,             # True se usati dati dal frame successivo
        'window_samples': int,               # Numero campioni analizzati
        
        # LEGACY (backward compatibility)
        'slope': float,                      # Slope fit lineare (reference)
        'r_squared': float,                  # R² fit lineare (reference)
        'monotone': bool,                    # E1>E2>E3>E4 strictly (descriptive only)
    }
    """
    # Parametri finestra: 0.6 ms a 200 ksps = 120 campioni
    window_samples = 120
    
    # ✅ GESTIONE SPILL: Controlla se serve concatenare frame successivo
    near_end = (peak_idx > 380)  # 380/512 samples = 74% del frame
    used_next_frame = False
    
    # Finestra post-picco
    start_idx = peak_idx
    end_idx = peak_idx + window_samples
    
    # ✅ SE PICCO VICINO A FINE FRAME E DATI INSUFFICIENTI
    if end_idx > len(envelope) and near_end and next_frame_signal is not None:
        # Calcola quanti samples mancano
        missing_samples = end_idx - len(envelope)
        
        print(f"   🔗 SPILL DETECTED: Peak at sample {peak_idx}/{len(envelope)}, need {missing_samples} samples from next frame")
        
        # Calcola envelope del frame successivo
        try:
            next_envelope = compute_hilbert_envelope(next_frame_signal)
            
            # Concatena: envelope corrente + primi N samples del prossimo
            extended_envelope = np.concatenate([
                envelope[start_idx:],
                next_envelope[:missing_samples]
            ])
            
            decay_window = extended_envelope
            used_next_frame = True
            
            print(f"   ✅ Extended analysis window: {len(envelope[start_idx:])} + {missing_samples} = {len(decay_window)} samples")
            
        except Exception as e:
            print(f"   ⚠️ Failed to extend window: {e}")
            # Fallback: usa solo dati disponibili
            decay_window = envelope[start_idx:]
            used_next_frame = False
    else:
        # Caso normale: finestra completamente nel frame corrente
        end_idx_clipped = min(end_idx, len(envelope))
        decay_window = envelope[start_idx:end_idx_clipped]
    
    # ========================================================================
    # FIT LOGARITMICO SAMPLE-BY-SAMPLE (120 CAMPIONI)
    # ========================================================================
    # IMPORTANTE: compute_decay_r2 riceve la finestra GIÀ ESTRATTA (post-picco)
    decay_r2_results = compute_decay_r2(decay_window, fs=fs)
    
    # ========================================================================
    # LEGACY: Calcola anche energie sub-windows per compatibilità
    # ========================================================================
    actual_samples = len(decay_window)
    quarter = actual_samples // 4
    
    if quarter >= 5:
        w1 = decay_window[0:quarter]
        w2 = decay_window[quarter:2*quarter]
        w3 = decay_window[2*quarter:3*quarter]
        w4 = decay_window[3*quarter:4*quarter]
        
        E1 = float(np.mean(w1 ** 2))
        E2 = float(np.mean(w2 ** 2))
        E3 = float(np.mean(w3 ** 2))
        E4 = float(np.mean(w4 ** 2))
        
        energies = [E1, E2, E3, E4]
        
        # Fit lineare legacy (solo per reference)
        x_fit = np.array([0, 1, 2, 3])
        y_fit = np.array(energies)
        
        try:
            slope_legacy, _ = np.polyfit(x_fit, y_fit, deg=1)
            y_pred = slope_legacy * x_fit + np.mean(y_fit)
            ss_res = np.sum((y_fit - y_pred) ** 2)
            ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
            r2_legacy = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        except:
            slope_legacy = 0.0
            r2_legacy = 0.0
        
        monotone = (E1 > E2 > E3 > E4)
    else:
        energies = [0.0, 0.0, 0.0, 0.0]
        slope_legacy = 0.0
        r2_legacy = 0.0
        monotone = False
    
    # ========================================================================
    # RISULTATO FINALE: Feature descrittive + energies per validazione
    # ========================================================================
    return {
        # DECAY FIT (DESCRIPTIVE ONLY)
        'r_squared_log': decay_r2_results['r2_log'],
        'slope_log': decay_r2_results['slope_log'],
        'tau_ms': decay_r2_results['tau_ms'],
        'n_samples': decay_r2_results['n_samples'],
        
        # SUB-WINDOW ENERGIES (for validation)
        'energies': energies,
        'E_W1': energies[0],
        'E_W4': energies[3],
        
        # METADATA
        'near_end': near_end,
        'used_next_frame': used_next_frame,
        'window_samples': actual_samples,
        
        # LEGACY (backward compatibility)
        'slope': slope_legacy,
        'r_squared': r2_legacy,
        'monotone': monotone,
    }


class ReplayWindowAudio(ReplayBaseWindow):
    """Finestra replay audio con architettura multi-livello ottimizzata"""
    
    def __init__(self, file_path=None, parent=None):
        super().__init__(parent)
        
        # Data manager
        self.data_manager = AudioDataManager()
        self.file_path = file_path
        
        # Playback state
        self.current_position_ms = 0
        self.playback_rate = 1.0
        self.is_playing = False
        self.last_update_time = 0
        
        # Timer per playback
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._update_playback)
        self.playback_timer.setTimerType(QtCore.Qt.PreciseTimer)
        
        # Setup UI
        self.setWindowTitle(f"Audio Replay - {os.path.basename(file_path)}")
        self.setup_main_layout()
        self.setup_menubar()
        self.setup_toolbar()
        
        # ✅ CONNETTI AZIONI AUDIO-SPECIFIC
        if hasattr(self, 'actionClickDetector'):
            self.actionClickDetector.triggered.connect(self.open_click_detector_dialog)
        
        # Applica tema
        self._load_saved_settings()
        
        # Connetti segnali
        self.playback_speed_changed.connect(self._on_playback_speed_changed)
        self.playback_position_changed.connect(self._on_position_changed)
        
        # Range velocità limitato per audio
        self.velocity.setRange(0.1, 1.0)

        #imposta la vista iniziale dell'asse x del tempo sui primi 20s
        # ✅ IMPOSTA LA VISTA INIZIALE SULLA DIMENSIONE DELLA FINESTRA DI STREAMING
        self.plot_widget_time.set_axis_limits(0, self.data_manager.streaming_window_size)
        self.plot_widget_time.set_x_range(0, self.data_manager.streaming_window_size)

        # mostra a tutto schermo mantenendo le grafiche
        self.showMaximized()

    def show_spectral_energy_analysis(self):
        """Show spectral energy analysis for the current frame (OVERRIDE)"""
        if self.data_manager.total_frames == 0:
            QMessageBox.warning(self, "No Data", "No FFT data available for analysis.")
            return
        
        # Get current frame index (FIX: use round() to match step_frame behavior)
        frame_index = int(round(self.current_position_ms / self.data_manager.frame_duration_ms))
        frame_index = max(0, min(frame_index, self.data_manager.total_frames - 1))
        
        if frame_index >= len(self.data_manager.fft_data):
            QMessageBox.warning(self, "Error", f"Invalid frame index: {frame_index}")
            return
        
        # Get FFT magnitudes for current frame
        fft_magnitudes = self.data_manager.fft_data[frame_index]
        
        # Compute energy
        try:
            energies = compute_fft_energy(fft_magnitudes)
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"Failed to compute energy:\n{str(e)}")
            print(f"❌ Energy computation error: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Show results in dialog
        dialog = SpectralEnergyDialog(energies, frame_index=frame_index, parent=self)
        dialog.exec()
        
        # Print to console for reference (includi ratio)
        ratio_raw = energies['low'] / energies['high'] if energies['high'] > 0 else 0.0
        frame_time_sec = frame_index * self.data_manager.frame_duration_ms / 1000.0
        print(f"\n📊 Spectral Energy Analysis - Frame {frame_index}")
        print(f"   Time: {frame_time_sec:.3f}s")
        print(f"   Total (20-80 kHz):  {energies['total']:.6f} V² = {energies['total']*1e6:.3f} mV²")
        print(f"   Low   (20-40 kHz):  {energies['low']:.6f} V² = {energies['low']*1e6:.3f} mV²")
        print(f"   High  (40-80 kHz):  {energies['high']:.6f} V² = {energies['high']*1e6:.3f} mV²")
        print(f"   Ratio R (E_low/E_high): {ratio_raw:.3f}")
        print(f"   Band 1 (20-30 kHz): {energies['b1']:.6f} V² = {energies['b1']*1e6:.3f} mV²")
        print(f"   Band 2 (30-40 kHz): {energies['b2']:.6f} V² = {energies['b2']*1e6:.3f} mV²")
        print(f"   Band 3 (40-60 kHz): {energies['b3']:.6f} V² = {energies['b3']*1e6:.3f} mV²")
        print(f"   Band 4 (60-80 kHz): {energies['b4']:.6f} V² = {energies['b4']*1e6:.3f} mV²")

    # === PLAYBACK CONTROL METHODS ===
    
    def start_playback(self):
        """Avvia riproduzione ottimizzata"""
        if self.data_manager.total_frames == 0:
            print("⚠️ Nessun dato caricato")
            #mostra messaggio di errore
            QMessageBox.critical(self, "An Error Occurred", "No data loaded for playback.")
            return
        
        self.paused_playing = False
        # Reset se alla fine
        if self.current_position_ms >= self.data_manager.total_duration_sec * 1000:
            self.current_position_ms = 0
            self.time_slider.setValue(0)
        
        # Verifica/carica streaming buffer per posizione corrente
        current_time_sec = self.current_position_ms / 1000.0
        if not self.data_manager.contains_streaming_time(current_time_sec):
            self._load_streaming_buffer_for_time(current_time_sec)
        
        super().start_playback()
        self.is_playing = True
        self._reset_playback_timing()
        
        # Timer a 60 FPS per smoothness
        self.playback_timer.setInterval(16)
        if not self.playback_timer.isActive():
            self.playback_timer.start()

        #self.actionMath.setEnabled(False)  # Disabilita operazioni matematiche durante il playback
        #if hasattr(self, 'region'):
         #   self.voltage_plot.plot_widget.removeItem(self.region)
          #  del self.region  # Rimuovi l'attributo
        print(f"🎬 Playback avviato da {current_time_sec:.2f}s a {self.playback_rate}x")
    
    def pause_playback(self):
        """Mette in pausa"""
        super().pause_playback()
        self.is_playing = False
        self.paused_playing = True
        if self.playback_timer.isActive():
            self.playback_timer.stop()
        self.update_display()
        #self.actionMath.setEnabled(True)  # Abilita operazioni matematiche in pausa
        print("⏸️ Playback in pausa")
    
    def clear_history(self):
        """Stop - reset completo"""
        print("🛑 Stop - Reset completo")
        self.pause_playback()
        
        # Reset position
        self.current_position_ms = 0
        self.time_slider.setValue(0)
        self._reset_playback_timing()
        
        # Mostra overview completa
        overview_x, overview_y = self.data_manager.get_overview_data()
        if len(overview_x) > 0:
            self.time_curve.setData(overview_x, overview_y)
            self.plot_widget_time.set_x_range(0, min(20, overview_x.max()))
        
        # Reset position line
        if hasattr(self, 'time_position_line'):
            self.time_position_line.setPos(0)
        
        # Mostra primo frame FFT
        if len(self.data_manager.fft_data) > 0:
            self.fft_curve.setData(self.data_manager.frequency_axis, self.data_manager.fft_data[0])
        
        self._update_time_labels()
        self.update_display()
        self.plot_widget_time.set_x_range(0, min(20, overview_x.max()))
        self.plot_widget_time.set_axis_limits(0, min(20, overview_x.max()))

        print("✅ Reset completato")
    
    def _update_playback(self):
        """Update durante playback con timing preciso"""
        # Check fine file
        if self.current_position_ms >= self.data_manager.total_duration_sec * 1000:
            print("🏁 Fine riproduzione")
            self.pause_playback()
            return
        
        # Calcola tempo trascorso
        current_time = QtCore.QTime.currentTime().msecsSinceStartOfDay()
        if self.last_update_time == 0:
            elapsed_ms = 0
        else:
            elapsed_ms = current_time - self.last_update_time
        
        self.last_update_time = current_time
        
        # Applica velocità playback
        adjusted_elapsed = elapsed_ms * self.playback_rate
        new_position = self.current_position_ms + adjusted_elapsed
        self.current_position_ms = min(new_position, self.data_manager.total_duration_sec * 1000)
        
        # Update UI
        self.time_slider.setValue(int(self.current_position_ms))
        self.update_display()
        
        # Background tasks
        current_time_sec = self.current_position_ms / 1000.0
        self._check_streaming_buffer_update(current_time_sec)
    
    def _reset_playback_timing(self):
        """Reset timing playback"""
        self.last_update_time = 0
    
    def _on_playback_speed_changed(self, speed):
        """Callback cambio velocità"""
        self.playback_rate = speed
        effective_rate = 390 * speed
        print(f"🚀 Velocità: {speed}x ({effective_rate:.1f} FFT/s)")
        self._reset_playback_timing()
    
    def _on_position_changed(self, position_ms):
        """Callback cambio posizione slider o TEMPO"""
        new_time_sec = position_ms / 1000.0
        
        # Update position
        self.current_position_ms = position_ms
        
        # Verifica se serve nuovo streaming buffer
        if not self.data_manager.contains_streaming_time(new_time_sec):
            self._load_streaming_buffer_for_time(new_time_sec)
        
        self.update_display()
        
        # Update position line
        if hasattr(self, 'time_position_line'):
            self.time_position_line.setPos(new_time_sec)
    
    def update_display(self):
        """Aggiorna visualizzazione basata su posizione corrente"""
        if self.data_manager.total_frames == 0:
            return
        
        current_time_sec = self.current_position_ms / 1000.0
        
        # UPDATE FFT PLOT (FIX: use round() for consistent frame index calculation)
        frame_index = int(round(self.current_position_ms / self.data_manager.frame_duration_ms))
        frame_index = max(0, min(frame_index, self.data_manager.total_frames - 1))
        
        if frame_index < len(self.data_manager.fft_data):
            self.fft_curve.setData(self.data_manager.frequency_axis, 
                                self.data_manager.fft_data[frame_index])
            
            # ✅ RIMUOVI CURVA NORMALIZZATA quando cambi frame
            if hasattr(self, 'normalized_fft_curve'):
                try:
                    self.plot_widget_fft.plot_widget.removeItem(self.normalized_fft_curve)
                    del self.normalized_fft_curve
                except:
                    pass
            
            # ✅ RESET colore curva raw a accent_color del tema
            if hasattr(self, 'theme_manager'):
                self.theme_manager.apply_theme_to_plot(
                    plot_widget_name=self.plot_widget_fft.plot_widget,
                    plot_instance=self.fft_curve
                )
        
        # UPDATE TIME DOMAIN
        if self.data_manager.contains_streaming_time(current_time_sec):
            stream_x, stream_y = self.data_manager.get_streaming_data()
            if len(stream_x) > 0:
                self.time_curve.setData(stream_x, stream_y)
        else:
            overview_x, overview_y = self.data_manager.get_overview_data()
            if len(overview_x) > 0:
                self.time_curve.setData(overview_x, overview_y)
        
        # UPDATE POSITION LINE
        if hasattr(self, 'time_position_line'):
            self.time_position_line.setPos(current_time_sec)
        
        if self.is_playing or self.paused_playing:
            # Centra la vista sulla posizione corrente (PyQtGraph gestisce i limiti automaticamente)
            window_size = 20.0  # Finestra di 20 secondi
            half_window = window_size / 2.0
            x_center = current_time_sec
            x_min = x_center - half_window
            x_max = x_center + half_window

            #fare attenzione a quando current time è minore di window size!
            if x_min < 0:
                x_min = 0
                x_max = window_size
            
            self.plot_widget_time.set_x_limits(x_min, x_max)

        self._update_time_labels()
    

    def _update_time_labels(self):
        """Aggiorna widget tempo con larghezza fissa"""
        current_time_sec = self.current_position_ms / 1000.0
        total_time_sec = self.data_manager.total_duration_sec
        
        # ✅ USA IL NUOVO WIDGET invece di setText()
        if hasattr(self, 'current_time_input'):
            self.current_time_input.set_time(current_time_sec, total_time_sec)
        
        # Tooltip con info dettagliate
        if hasattr(self, 'velocity'):
            self.velocity.setToolTip(f"Playback speed: {self.playback_rate}x")
    
    # === DATA LOADING METHODS ===
    
    def _setup_metadata(self):
        """Setup metadati per playback"""
        estimated_fft_rate = 390.0
        if self.data_manager.frame_duration_ms == 0:
            self.data_manager.frame_duration_ms = 1000.0 / estimated_fft_rate
        if self.data_manager.total_frames > 0:
            self.data_manager.total_duration_sec = (
                self.data_manager.total_frames * self.data_manager.frame_duration_ms / 1000.0
            )
        self.time_slider.setRange(0, int(self.data_manager.total_duration_sec * 1000))
        self.time_slider.setValue(0)
    
        if hasattr(self, 'current_time_input'):
            self.current_time_input.set_time(0.0, self.data_manager.total_duration_sec)

    
    def _generate_overview_data(self):
        """Genera dati overview (10 FPS, energia media)"""
        print("🔄 Generazione overview...")
        
        if self.data_manager.total_frames == 0:
            return
        
        # Calcola step per 10 FPS (manteniamo risoluzione bassa per non sovraccaricare)
        overview_points = int(self.data_manager.total_duration_sec * self.data_manager.overview_fps)
        frame_step = max(1, self.data_manager.total_frames // overview_points)
        
        overview_x = []
        overview_y = []
        
        for i in range(0, self.data_manager.total_frames, frame_step):
            frame_time = (i * self.data_manager.frame_duration_ms) / 1000.0
            fft_data = self.data_manager.fft_data[i]
            
            # Energia media per overview veloce
            energy = np.mean(np.abs(fft_data)) 
            
            overview_x.append(frame_time)
            overview_y.append(energy)
        
        self.data_manager.overview_x = np.array(overview_x)
        self.data_manager.overview_y = np.array(overview_y)
        self.data_manager.overview_loaded = True
        
        print(f"✅ Overview: {len(overview_x)} punti, "
              f"{self.data_manager.get_memory_usage_mb():.1f}MB")
    
    def _load_streaming_buffer_for_time(self, center_time_sec):
        """Carica streaming buffer centrato su un tempo"""
        #print(f"🔄 Caricamento streaming buffer per {center_time_sec:.2f}s...")
        
        # Calcola finestra
        window_size = self.data_manager.streaming_window_size
        start_time = max(0, center_time_sec - window_size/2)
        end_time = min(self.data_manager.total_duration_sec, start_time + window_size)
        
        # Calcola frame range
        start_frame = int((start_time * 1000) / self.data_manager.frame_duration_ms)
        end_frame = int((end_time * 1000) / self.data_manager.frame_duration_ms)
        end_frame = min(end_frame, self.data_manager.total_frames)
        
        # ✅ MODIFICA CRITICA: USA TUTTE LE FFT (390 FPS) per non perdere click
        # Ogni click di 0.1-0.5ms è contenuto in UNA SINGOLA FFT
        # Se skippiamo anche solo 1 FFT, rischiamo di perdere il click!
        frame_step = 1  # NON saltare nessuna FFT
        
        stream_x = []
        stream_y = []
        
        for frame_idx in range(start_frame, end_frame, frame_step):
            if frame_idx >= self.data_manager.total_frames:
                break
            
            frame_time = (frame_idx * self.data_manager.frame_duration_ms) / 1000.0
            fft_data = self.data_manager.fft_data[frame_idx]
            
            # iFFT semplificata per streaming
            # Per performance, usa solo campione centrale
            signal_sample = np.mean(np.abs(fft_data))  # MEDIA DELLE MAGNITUDE NELLA FFT (anche se non c'è la fase, lascio abs)
            
            stream_x.append(frame_time)
            stream_y.append(signal_sample)
        
        # Update streaming buffer
        self.data_manager.streaming_x = np.array(stream_x)
        self.data_manager.streaming_y = np.array(stream_y)
        self.data_manager.streaming_start_time = start_time
        self.data_manager.streaming_end_time = end_time
        

        # ✅ DEBUG: Verifica che stai processando TUTTE le FFT
        #expected_frames = end_frame - start_frame
        #actual_frames = len(stream_x)
        #if actual_frames < expected_frames * 0.95:  # Tolleranza 5%
         #   print(f"⚠️ WARNING: Streaming buffer potrebbe perdere click! "
          #      f"Expected {expected_frames} frames, got {actual_frames}")
        
        #print(f"✅ Streaming buffer: {start_time:.1f}-{end_time:.1f}s, "
         #   f"{len(stream_x)} punti (390 FPS completo)")
    


    def _check_streaming_buffer_update(self, current_time_sec):
        """Verifica se serve aggiornare streaming buffer"""
        # Se ci stiamo avvicinando ai bordi del buffer
        buffer_margin = 5.0  # 5 secondi di margine
        
        needs_update = (current_time_sec < self.data_manager.streaming_start_time + buffer_margin or
                       current_time_sec > self.data_manager.streaming_end_time - buffer_margin)
        
        if needs_update:
            self._load_streaming_buffer_for_time(current_time_sec)
    
    def _setup_ui_with_data(self):
        """Setup UI con dati caricati, gestendo disponibilità fasi"""
        # Info labels
        info_text = (f"File: {os.path.basename(self.file_path)} | "
                    f"Frames: {self.data_manager.total_frames} | "
                    f"Duration: {self.data_manager.total_duration_sec:.1f}s | "
                    f"Clicks: {len(self.data_manager.click_events)}")
        
        self.fft_info_label.setText(f"FFT Spectrum - {info_text}")
        self.time_info_label.setText(f"Time Domain - {info_text}")
        
        # Check fasi
        file_version = self.data_manager.header_info.get('version', 0)
        has_phases = (len(self.data_manager.phase_data) > 0)

        # Aggiorna UI
        if hasattr(self, 'actionIFFTGraph'):
            self.actionIFFTGraph.setEnabled(has_phases)
            if has_phases:
                self.actionIFFTGraph.setToolTip("Show inverse FFT of current frame (using real phases)")
            else:
                self.actionIFFTGraph.setToolTip("iFFT not available (no phase data)")
        
        # Popola tabella click
        self._populate_click_table()
        
        # Mostra lo STREAMING BUFFER iniziale
        stream_x, stream_y = self.data_manager.get_streaming_data()
        if len(stream_x) > 0:
            self.time_curve.setData(stream_x, stream_y)
            max_range = min(self.data_manager.streaming_window_size, self.data_manager.total_duration_sec)
            self.plot_widget_time.set_x_range(0, max_range)
        else:
            overview_x, overview_y = self.data_manager.get_overview_data()
            if len(overview_x) > 0:
                self.time_curve.setData(overview_x, overview_y)
                max_range = min(self.data_manager.streaming_window_size, self.data_manager.total_duration_sec)
                self.plot_widget_time.set_x_range(0, max_range)

        # Mostra primo frame FFT
        if len(self.data_manager.fft_data) > 0:
            self.fft_curve.setData(self.data_manager.frequency_axis, 
                                   self.data_manager.fft_data[0])
        
        # Setup position line
        if hasattr(self, 'time_position_line'):
            self.time_position_line.setPos(0)

        self._update_time_labels()
    
        # Imposta threshold automatico DOPO che i dati sono stati caricati
        if hasattr(self.data_manager, 'fft_mean') and hasattr(self.data_manager, 'fft_std'):
            self.automatic_click_threshold = self.data_manager.fft_mean + 4 * self.data_manager.fft_std
            # Converti V in mV per lo spinbox
            self.PeakThresholdSpinBox.setValue(self.automatic_click_threshold * 1000.0)
            # Applica il filtro automaticamente
            self._apply_threshold_filter()
            # Aggiorna lo step della spinbox per cambiare di deviazioni standard (convertito in mV)
            self.PeakThresholdSpinBox.setSingleStep(self.data_manager.fft_std * 1000.0)
            print(f"✅ Auto-threshold impostato a: {self.automatic_click_threshold*1000:.3f} mV")
        else:
            print("⚠️ WARNING: fft_mean/fft_std non disponibili, usando threshold di default")
            self.PeakThresholdSpinBox.setValue(10.0)  # Default 10 mV
    
    
    def _populate_click_table(self):
        """Popola tabella click events con nuovo formato durata"""
        self.click_table.setRowCount(len(self.data_manager.click_events))
        for row, click in enumerate(self.data_manager.click_events):
            # Timestamp
            timestamp = click.get('timestamp', 0)
            time_str = f"{timestamp:.3f}s"
            self.click_table.setItem(row, 0, QTableWidgetItem(time_str))
            
            # Frequency
            self.click_table.setItem(row, 1, QTableWidgetItem(f"{click.get('frequency', 0):.0f} Hz"))
            
            # Amplitude
            self.click_table.setItem(row, 2, QTableWidgetItem(f"{click.get('amplitude', 0):.4f} V"))
            
            # ✅ DURATION: Converti da μs salvati a numero FFT per visualizzazione
            duration_us = click.get('duration_us', 0)
            if duration_us > 0:
                # ✅ Converti microsecondi → numero FFT (2560 μs per FFT)
                fft_count = int(round(duration_us / 2560))
                duration_str = f"{fft_count} FFT"
            else:
                duration_str = "N/A"
            
            self.click_table.setItem(row, 3, QTableWidgetItem(duration_str))
            
            # Notes
            self.click_table.setItem(row, 4, QTableWidgetItem(click.get('notes', '')))
    
    
    # === UI SETUP METHODS ===
    
    def setup_main_layout(self):
        """Layout principale"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Splitter principale
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setSizes([600, 450])
        main_layout.addWidget(main_splitter)
        
        # Tab widget per grafici
        self.tab_widget = QTabWidget()
        main_splitter.addWidget(self.tab_widget)
        
        # Setup tab
        self.setup_fft_tab()
        self.setup_time_tab()
        
        # Tabella click events
        self.setup_click_table()
        main_splitter.addWidget(self.click_table_widget)
    
    def setup_fft_tab(self):
        """Tab FFT Spectrum"""
        fft_widget = QWidget()
        fft_layout = QVBoxLayout(fft_widget)
        
        self.fft_info_label = QLabel("FFT Spectrum - Frequency Domain")
        self.fft_info_label.setAlignment(Qt.AlignCenter)
        fft_layout.addWidget(self.fft_info_label)
        
        self.plot_widget_fft = BasePlotWidget(
            x_label="Frequency", y_label="Amplitude",
            x_range=(20000, 80000), y_range=(0, 0.02),
            x_min=19000, x_max=81000, y_min=0, y_max=1.7,
            unit_x="Hz", unit_y="V", parent=self
        )
        
        self.fft_curve = self.plot_widget_fft.plot_widget.plot(name="FFT Data")
        self.plot_widget_fft.plot_widget.showGrid(x=True, y=True)

        # ✅ SOLUZIONE: Disabilita l'auto-ranging sull'asse Y dopo il setup iniziale.
        # Questo impedisce al grafico di riscalare l'asse Y automaticamente
        # quando i dati cambiano (es. passando da streaming a overview), 
        # prevenendo il "salto" di scala.
        self.plot_widget_fft.plot_widget.getPlotItem().getViewBox().disableAutoRange(axis='y')
        
        fft_layout.addWidget(self.plot_widget_fft)
        self.tab_widget.addTab(fft_widget, "FFT Spectrum")
    
    def setup_time_tab(self):
        """Tab Time Domain con sistema ottimizzato"""
        time_widget = QWidget()
        time_layout = QVBoxLayout(time_widget)
        
        self.time_info_label = QLabel("Time Domain Signal (Multi-Level)")
        self.time_info_label.setAlignment(Qt.AlignCenter)
        time_layout.addWidget(self.time_info_label)
        
        self.plot_widget_time = BasePlotWidget(
            x_label="Time", y_label="Average Amplitude per FFT",
            x_range=(0, 20), y_range=(0, 0.003),
            x_min=0, x_max=None, y_min=0, y_max=3.3,
            unit_x="s", unit_y="V", parent=self
        )
        
        self.time_curve = self.plot_widget_time.plot_widget.plot(
            name="Average Amplitude Signal", pen={'color': 'blue', 'width': 1}
        )
        
        # Position line
        self.time_position_line = self.plot_widget_time.plot_widget.addLine(
            x=0, pen={'color': 'red', 'width': 2, 'style': QtCore.Qt.DashLine}
        )
        
        self.plot_widget_time.plot_widget.showGrid(x=True, y=True)

        self.plot_widget_time.plot_widget.getPlotItem().getViewBox().disableAutoRange(axis='y')
        
        # ✅ NUOVO: Setup limiti PyQtGraph
        view_box = self.plot_widget_time.plot_widget.getPlotItem().getViewBox()
        view_box.setLimits(
            xMin=0,                    # Non andare prima di t=0
            xMax=None,                 # Impostato dopo il caricamento file
            minXRange=0.3,             # Zoom minimo: 0.3 secondi
            maxXRange=20.0             # Zoom massimo: 20 secondi (streaming buffer)
        )
        
        print("✅ ViewBox limits configured: minXRange=0.5s, maxXRange=20.0s")

        time_layout.addWidget(self.plot_widget_time)
        self.tab_widget.addTab(time_widget, "Time Domain")
    
    def setup_click_table(self):
        """Tabella click events"""
        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)

        table_label = QLabel("Clicks Events")
        table_label.setAlignment(Qt.AlignCenter)
        font = table_label.font()
        font.setBold(True)
        font.setPointSize(12)
        table_label.setFont(font)
        table_layout.addWidget(table_label)

        # CREA 2 TABELLE DIVERSE PER DIVERSI TIPI DI DATI (CLICK EVENTS) USANDO 2 QTABWIDGET

        self.click_tab_widget = QTabWidget()
        table_layout.addWidget(self.click_tab_widget)

        # Tab 1: Click Events
        self.click_table = QTableWidget()
        self.click_table.setColumnCount(5)
        self.click_table.setHorizontalHeaderLabels([
            "Timestamp", "Frequency", "Amplitude", "Duration", "Notes"
        ])
        
        self.click_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.click_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.click_table.setAlternatingRowColors(True)

        self.click_table.itemDoubleClicked.connect(self._on_click_table_double_clicked)
        
        header = self.click_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        
        self.click_table.verticalHeader().setVisible(False)
        
        self.click_table.setToolTip(
            "Double-click on a row to jump to that click event"
        )
        self.click_tab_widget.addTab(self.click_table, "Recorded Events")

        ####

        #SECONDA TABELLA PER MOSTRARE I PICCHI SOPRA UNA SOGLIA IMPOSTATA DALL'UTENTE
        self.peak_table = QTableWidget()
        self.peak_table.setColumnCount(5)
        self.peak_table.setHorizontalHeaderLabels([
            "Timestamp", "Frequency", "Amplitude", "Duration", "Notes"
        ])
        
        self.peak_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.peak_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.peak_table.setAlternatingRowColors(True)

        self.peak_table.itemDoubleClicked.connect(self._on_click_table_double_clicked)
        
        peak_header = self.peak_table.horizontalHeader()
        peak_header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        peak_header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        peak_header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        peak_header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        peak_header.setSectionResizeMode(4, QHeaderView.Stretch)

        self.peak_table.verticalHeader().setVisible(False)

        self.peak_table.setToolTip(
            "Double-click on a row to jump to that peak event"
        )

        self.click_tab_widget.addTab(self.peak_table, "Above Threshold")

       
        self.PeakThresholdLabel = QLabel(table_container)
        self.PeakThresholdLabel.setObjectName(u"PeakThresholdLabel")
        font_peak = QFont()
        font_peak.setPointSize(16)
        font_peak.setBold(True)
        self.PeakThresholdLabel.setFont(font_peak)
        self.PeakThresholdLabel.setText("Threshold on Average Amplitude:")
        self.PeakThresholdLabel.setAlignment(Qt.AlignCenter)
        table_layout.addWidget(self.PeakThresholdLabel)

        # Layout orizzontale per button e spinbox
        threshold_layout = QHBoxLayout()
        table_layout.addLayout(threshold_layout)

        self.PeakThresholdSpinBox = QDoubleSpinBox(table_container)
        self.PeakThresholdSpinBox.setObjectName(u"PeakThresholdSpinBox")
        self.PeakThresholdSpinBox.setDecimals(3)
        self.PeakThresholdSpinBox.setRange(0, 3300) # Valori in mV = 0-3.3V
        self.PeakThresholdSpinBox.setSingleStep(0.001) #dopo viene sovrascritto con valore della deviazione standard
        self.PeakThresholdSpinBox.setValue(10) # Valore di default
        self.PeakThresholdSpinBox.setSuffix(" mV")
        self.PeakThresholdSpinBox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        threshold_layout.addWidget(self.PeakThresholdSpinBox)

        # Crea una linea orizzontale sul grafico del time domain per indicare la soglia
        self.threshold_line = self.plot_widget_time.plot_widget.addLine(
            y=self.PeakThresholdSpinBox.value() / 1000.0,  # Converti mV a V
            pen={'color': 'red', 'width': 2, 'style': QtCore.Qt.DashLine}
        )
        self.PeakThresholdSpinBox.valueChanged.connect(self._update_threshold_line)


        self.PeakThresholdButton = QPushButton("Apply", table_container)
        self.PeakThresholdButton.setObjectName(u"PeakThresholdButton")
        self.PeakThresholdButton.setToolTip("Filter peaks above the set threshold")
        self.PeakThresholdButton.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.PeakThresholdButton.clicked.connect(self._apply_threshold_filter)
        threshold_layout.addWidget(self.PeakThresholdButton)

        ####### AGGIUNTA DELLA LOGICA PER FILTRARE I PICCHI NELLA TABELLA

        self.click_table_widget = table_container


    
    # === FRAME-BY-FRAME NAVIGATION (PRECISION IMPLEMENTATION) ===

    def step_position(self, direction: int):
        """
        SPOSTAMENTO PRECISO PER AUDIO (sovrascrive il metodo base).
        Sposta la posizione di un singolo frame FFT in avanti o indietro.
        """
        if self.data_manager.total_frames == 0:
            return

        # 1. Calcola l'indice del frame corrente in modo preciso
        current_frame_index = int(round(self.current_position_ms / self.data_manager.frame_duration_ms))

        # 2. Calcola il nuovo indice con controllo dei limiti
        new_frame_index = current_frame_index + direction
        new_frame_index = max(0, min(new_frame_index, self.data_manager.total_frames - 1))

        # 3. Calcola la nuova posizione in millisecondi DAL NUOVO INDICE
        new_position_ms = new_frame_index * self.data_manager.frame_duration_ms

        # 4. Aggiorna lo stato e l'interfaccia usando i metodi esistenti
        self._on_position_changed(new_position_ms)
        
        # Assicurati che anche lo slider si aggiorni visivamente al valore esatto
        self.time_slider.setValue(int(new_position_ms))


    def show_ifft_window(self):
        """
        Mostra iFFT del frame FFT CORRENTE usando le fasi reali.
        ✅ RICOSTRUISCE SPETTRO COMPLETO 0-100kHz prima di irfft()
        """
        if self.data_manager.total_frames == 0:
            QMessageBox.warning(self, "No Data", "No data loaded to perform iFFT.")
            return
        
        # ✅ CHECK VERSIONE FILE E FASI
        file_version = self.data_manager.header_info.get('version', 0)
        has_phases = (file_version >= 3.0) and hasattr(self.data_manager, 'phase_data') and len(self.data_manager.phase_data) > 0
        
        if not has_phases:
            QMessageBox.warning(self, "No Phase Data", "iFFT requires phase information (file version >= 3.0).")
            return
        
        # ✅ CALCOLA FRAME CORRENTE (FIX: use round() for consistent frame index)
        current_frame_index = int(round(self.current_position_ms / self.data_manager.frame_duration_ms))
        current_frame_index = max(0, min(current_frame_index, self.data_manager.total_frames - 1))
        
        # ✅ ESTRAI DATI FFT (154 bins, 20-80kHz)
        try:
            fft_magnitudes = self.data_manager.fft_data[current_frame_index]  # 154 bins
            fft_phases_int8 = self.data_manager.phase_data[current_frame_index]  # 154 bins
        except IndexError:
            QMessageBox.warning(self, "Data Error", "Frame index out of range.")
            return
        
        # ✅ PARAMETRI FFT DAL FIRMWARE
        fs = self.data_manager.header_info.get('fs', 200000)
        fft_size = self.data_manager.header_info.get('fft_size', 512)
        num_bins_full = fft_size // 2  # 256 bins (0-100kHz)
        
        # ✅ CALCOLA RANGE BINS (come nel firmware)
        bin_freq = fs / fft_size  # 200000/512 = 390.625 Hz/bin
        bin_start = int(20000 / bin_freq)  # 20000/390.625 = 51
        bin_end = int(80000 / bin_freq)    # 80000/390.625 = 204
        num_received_bins = bin_end - bin_start + 1  # 204-51+1 = 154 ✅
        
        # ✅ VERIFICA COERENZA DATI
        if len(fft_magnitudes) != num_received_bins:
            print(f"⚠️ WARNING: Expected {num_received_bins} bins, got {len(fft_magnitudes)}")
            num_received_bins = len(fft_magnitudes)  # Usa quello che hai
        
        # ✅ RICOSTRUISCI SPETTRO COMPLETO (0-100kHz, 256 bins)
        full_spectrum_mag = np.zeros(num_bins_full, dtype=np.float32)
        full_spectrum_phase = np.zeros(num_bins_full, dtype=np.int8)
        
        # Inserisci i dati 20-80kHz SENZA windowing (sarà applicata dopo)
        full_spectrum_mag[bin_start:bin_start + num_received_bins] = fft_magnitudes
        full_spectrum_phase[bin_start:bin_start + num_received_bins] = fft_phases_int8
        
        # ✅ CONVERTI FASI E CREA SPETTRO COMPLESSO
        fft_phases_rad = (full_spectrum_phase / 127.0) * np.pi
        complex_spectrum = full_spectrum_mag * np.exp(1j * fft_phases_rad)
        
        # ✅ APPLICA TUKEY WINDOW ALLO SPETTRO COMPLESSO (FIX GIBBS CORRETTO)
        # IMPORTANTE: Window applicata allo spettro complesso, non solo alle magnitude
        # Questo elimina discontinuità sia in Re{X[k]} che in Im{X[k]}
        taper_bins = max(5, num_received_bins // 10)
        
        # Crea finestra Tukey per la regione completa (256 bins)
        window_full = np.ones(num_bins_full)
        
        # Left taper (bins 51-66, cosine fade-in)
        for i in range(taper_bins):
            alpha = i / taper_bins
            window_full[bin_start + i] = 0.5 * (1 - np.cos(np.pi * alpha))
        
        # Right taper (bins 189-204, cosine fade-out)
        for i in range(taper_bins):
            alpha = i / taper_bins
            window_full[bin_start + num_received_bins - i - 1] = 0.5 * (1 - np.cos(np.pi * alpha))
        
        # Applica window allo spettro complesso (attenuazione graduale ai bordi)
        complex_spectrum = complex_spectrum * window_full
        
        # ✅ ESEGUI iFFT (ORA con spettro completo 256 bins FORZANDO n=fft_size (512)
        try:
            time_domain_signal = np.fft.irfft(complex_spectrum, n=fft_size)
        except Exception as e:
            QMessageBox.critical(self, "iFFT Error", f"Failed to compute iFFT:\n{str(e)}")
            return
        
        # ✅ CREA ASSE TEMPORALE CORRETTO
        num_samples = len(time_domain_signal)  # Dovrebbe essere 512
        duration_sec = num_samples / fs        # 512/200000 = 2.56ms ✅
        
        # ✅ CALCOLA TEMPO INIZIO FRAME (preciso)
        frame_start_time = current_frame_index * (fft_size / fs)  # t = frame_idx * 2.56ms
        time_axis = np.linspace(frame_start_time, 
                            frame_start_time + duration_sec, 
                            num_samples)
        
        # ✅ DEBUG: Verifica risultato
        actual_duration_ms = duration_sec * 1000
        print(f"📊 iFFT Window Debug:")
        print(f"   Frame index: {current_frame_index}")
        print(f"   Start time: {frame_start_time:.6f}s")
        print(f"   Duration: {actual_duration_ms:.2f}ms (expected: 2.56ms)")
        print(f"   Samples: {num_samples} (expected: 512)")
        print(f"   Bins used: {bin_start}-{bin_start+num_received_bins} ({num_received_bins} bins)")
        print(f"   Frequency range: {bin_start*bin_freq:.0f} Hz - {(bin_start+num_received_bins)*bin_freq:.0f} Hz")
        print(f"   ✅ Tukey window applied ({taper_bins} bins taper per side) to reduce edge artifacts")
        
        if abs(actual_duration_ms - 2.56) > 0.1:
            print(f"⚠️ WARNING: Duration mismatch! Got {actual_duration_ms:.2f}ms, expected 2.56ms")
        
        if num_samples != fft_size:
            print(f"⚠️ WARNING: Sample count mismatch! Got {num_samples}, expected {fft_size}")
        
        # ✅ CHIUDI FINESTRA PRECEDENTE
        if hasattr(self, 'ifft_win') and self.ifft_win is not None:
            try:
                self.ifft_win.close()
                self.ifft_win.deleteLater()
            except RuntimeError:
                pass
        
        # ✅ CREA FINESTRA
        self.ifft_win = IFFTWindow(
            time_axis, 
            time_domain_signal, 
            parent=self,
            frame_index=current_frame_index,
            has_real_phases=True
        )
        self.ifft_win.show()


    # === CLICK EVENTS NAVIGATION METHODS ===
    # DOPPIO CLICK SULLA TABELLA CLICK EVENTS O PEAK TABLE
    def _on_click_table_double_clicked(self, item):
        """
        Gestisce doppio click su ENTRAMBE le tabelle (click_table e peak_table).
        Salta al timestamp dell'evento selezionato.
        """
        # ✅ IDENTIFICA QUALE TABELLA È STATA CLICCATA
        sender_table = self.sender()  # QTableWidget che ha emesso il segnale
        row = item.row()
        
        # ✅ CASO 1: Click su tabella "Recorded Events"
        if sender_table is self.click_table:
            if row < 0 or row >= len(self.data_manager.click_events):
                print(f"⚠️ Invalid click_table row: {row}")
                return
            
            # Estrai timestamp dal click event registrato
            click_event = self.data_manager.click_events[row]
            target_timestamp_sec = click_event.get('timestamp', 0)
            event_freq = click_event.get('frequency', 0)
            event_amp = click_event.get('amplitude', 0)
            event_type = "Recorded Click"
        
        # ✅ CASO 2: Click su tabella "Above Threshold"
        elif sender_table is self.peak_table:
            if row < 0 or row >= self.peak_table.rowCount():
                print(f"⚠️ Invalid peak_table row: {row}")
                return
            
            # ✅ LEGGI IL TIMESTAMP DIRETTAMENTE DALLA CELLA DELLA TABELLA
            timestamp_item = self.peak_table.item(row, 0)  # Colonna 0 = Timestamp
            if not timestamp_item:
                print(f"⚠️ No timestamp in peak_table row {row}")
                return
            
            # Parse timestamp (formato: "123.456s")
            timestamp_str = timestamp_item.text().replace('s', '').strip()
            try:
                target_timestamp_sec = float(timestamp_str)
            except ValueError:
                print(f"⚠️ Invalid timestamp format: {timestamp_str}")
                return
            
            # Leggi anche freq e amplitude per feedback
            freq_item = self.peak_table.item(row, 1)
            amp_item = self.peak_table.item(row, 2)
            event_freq = float(freq_item.text().replace(' Hz', '')) if freq_item else 0
            event_amp = float(amp_item.text().replace(' V', '')) if amp_item else 0
            event_type = "Auto-detected Peak"
        
        else:
            print("⚠️ Unknown table sender")
            return
        
        # ✅ COMUNE: NAVIGAZIONE AL TIMESTAMP
        target_position_ms = target_timestamp_sec * 1000
        
        # Pausa se in playback
        was_playing = self.is_playing
        if was_playing:
            self.pause_playback()
        
        # Salta al timestamp
        print(f"🎯 Jumping to {event_type} at {target_timestamp_sec:.3f}s (row {row})")
        
        self._on_position_changed(target_position_ms)
        self.time_slider.setValue(int(target_position_ms))
        
        # Centra la vista sul click
        window_half = 10.0  # ±10 secondi
        x_min = max(0, target_timestamp_sec - window_half)
        x_max = min(self.data_manager.total_duration_sec, target_timestamp_sec + window_half)
        
        self.plot_widget_time.set_x_range(x_min, x_max)
        self.plot_widget_time.set_axis_limits(x_min, x_max)
        
        # Evidenzia la riga selezionata nella tabella corretta
        sender_table.selectRow(row)
        
        # Feedback visivo
        print(f"✅ Jumped to: {target_timestamp_sec:.3f}s | {event_freq:.0f} Hz | {event_amp:.6f} V")


    # === PEAK THRESHOLD FILTER METHODS ===
    def _apply_threshold_filter(self):
        """
        Filtra e mostra tutti i picchi sopra la soglia impostata.
        """
        threshold = self.PeakThresholdSpinBox.value() * 0.001  # Converti mV in V

        if len(self.data_manager.fft_means) == 0:
            QMessageBox.warning(self, "No Data", "No FFT data available for analysis.")
            return
        
        print(f"🔍 Ricerca picchi sopra {threshold:.6f} V...")
        
        # ✅ Trova tutti i picchi sopra soglia (MOLTO VELOCE con NumPy)
        peak_indices = np.where(self.data_manager.fft_means >= threshold)[0]
        
        if len(peak_indices) == 0:
            #QMessageBox.information(self, "No Peaks Found", 
            #                        f"No peaks found above {threshold:.6f} V")
            self.peak_table.setRowCount(0)
            return
        
        # ✅ OTTIMIZZAZIONE: Raggruppa picchi consecutivi (evita duplicati)
        peaks_grouped = self._group_consecutive_peaks(peak_indices)
        
        # Popola tabella
        self._populate_peak_table(peaks_grouped, threshold)
        
        #Imposta la vista sulla tabella dei picchi
        self.click_tab_widget.setCurrentWidget(self.peak_table)

        # Feedback
        print(f"✅ Trovati {len(peaks_grouped)} picchi sopra soglia")
        #QMessageBox.information(self, "Analysis Complete", 
         #                       f"Found {len(peaks_grouped)} peaks above {threshold:.6f} V")

    def _group_consecutive_peaks(self, indices, max_gap_frames=5):
        """
        Raggruppa picchi consecutivi in eventi singoli.
        max_gap_frames: gap massimo tra picchi per considerarli stesso evento
        """
        if len(indices) == 0:
            return []
        
        groups = []
        current_group = [indices[0]]
        
        for i in range(1, len(indices)):
            if indices[i] - indices[i-1] <= max_gap_frames:
                current_group.append(indices[i])
            else:
                groups.append(current_group)
                current_group = [indices[i]]
        
        groups.append(current_group)  # Aggiungi ultimo gruppo
        
        return groups

    def _populate_peak_table(self, peak_groups, threshold):
        """Popola la tabella peak_table con i risultati"""
        self.peak_table.setRowCount(len(peak_groups))
        
        for row, group in enumerate(peak_groups):
            # Prendi il frame con ampiezza massima nel gruppo
            max_idx_in_group = np.argmax([self.data_manager.fft_means[i] for i in group])
            peak_frame = group[max_idx_in_group]
            
            timestamp = self.data_manager.fft_timestamps[peak_frame]
            amplitude = self.data_manager.fft_means[peak_frame]
            
            # Calcola durata del gruppo
            duration_frames = len(group)
            duration_us = duration_frames * 2560  # 2560 μs per frame
            
            # ✅ Stima frequenza (se disponibile FFT del frame)
            if peak_frame < len(self.data_manager.fft_data):
                fft_frame = self.data_manager.fft_data[peak_frame]
                peak_freq_idx = np.argmax(fft_frame)
                frequency = self.data_manager.frequency_axis[peak_freq_idx] if len(self.data_manager.frequency_axis) > peak_freq_idx else 0
            else:
                frequency = 0
            
            # Popola riga
            self.peak_table.setItem(row, 0, QTableWidgetItem(f"{timestamp:.3f}s"))
            self.peak_table.setItem(row, 1, QTableWidgetItem(f"{frequency:.0f} Hz"))
            self.peak_table.setItem(row, 2, QTableWidgetItem(f"{amplitude:.6f} V"))
            self.peak_table.setItem(row, 3, QTableWidgetItem(f"{duration_frames} FFT"))
            self.peak_table.setItem(row, 4, QTableWidgetItem("Auto-detected"))

    def _update_threshold_line(self):
        """Aggiorna la linea di soglia sul grafico del time domain"""
        threshold_value = self.PeakThresholdSpinBox.value() * 0.001  # Converti mV a V
        self.threshold_line.setValue(threshold_value)

    
    def normalize_fft_window(self):
        """
        Applica normalizzazione CONSERVATIVA (50%) per risposta in frequenza del microfono.
        
        APPROCCIO:
        - Non modifica dati originali (display-only overlay)
        - Correzione al 50% basata su datasheet SPU0410LR5H-QB
        - Mostra entrambe le curve (raw + normalized) per confronto
        
        ERRORE STIMATO: ±2.9 dB (95% confidence)
        ADATTO PER: Analisi qualitative (forma spettro, confronti relativi)
        NON ADATTO PER: Misure assolute di pressione sonora (dB SPL)
        
        DOCUMENTAZIONE: Vedi docs/MICROPHONE_NORMALIZATION_TECHNICAL_REPORT.md
        """
        if self.data_manager.total_frames == 0:
            QMessageBox.warning(self, "No Data", "No FFT data loaded.")
            return
        
        print("🔧 Starting 50% conservative microphone normalization...")
        
        # === 1. DATI DAL DATASHEET (SPU0410LR5H-QB) ===
        # Fonte: Fig. 4 - "Ultrasonic Free Field Response Normalized to 1kHz"
        # Valori letti manualmente dal grafico (±0.5 dB accuracy)
        datasheet_freq_khz = np.array([20, 25, 30, 40, 50, 60, 70, 80])
        datasheet_response_db = np.array([8.0, 10.5, 6.0, -2.0, -6.0, -7.0, -6.0, -4.0])
        
        datasheet_freq_hz = datasheet_freq_khz * 1000
        
        # === 2. OTTIENI SPETTRO FFT CORRENTE (FIX: use round() for consistent frame index) ===
        frame_index = int(round(self.current_position_ms / self.data_manager.frame_duration_ms))
        frame_index = max(0, min(frame_index, self.data_manager.total_frames - 1))
        
        if frame_index >= len(self.data_manager.fft_data):
            QMessageBox.warning(self, "Invalid Frame", "Cannot access FFT data for current frame.")
            return
        
        original_fft = self.data_manager.fft_data[frame_index]
        freq_axis = self.data_manager.frequency_axis
        
        # === 3. FILTRA RANGE 20-80 kHz ===
        valid_mask = (freq_axis >= 20000) & (freq_axis <= 80000)
        freq_range = freq_axis[valid_mask]
        
        if len(freq_range) == 0:
            QMessageBox.warning(self, "Invalid Range", "No frequencies in 20-80 kHz range.")
            return
        
        # === 4. INTERPOLAZIONE RISPOSTA MICROFONO ===
        mic_response_db = np.interp(freq_range, datasheet_freq_hz, datasheet_response_db)
        
        # === 5. CALCOLA CORREZIONE AL 50% (CONSERVATIVA) ===
        # Se mic attenua di -5 dB, amplifico solo +2.5 dB (50%)
        correction_gain_50 = 10 ** (-mic_response_db * 0.5 / 20.0)
        
        # === 6. CREA ARRAY DI CORREZIONE PER INTERO SPETTRO ===
        full_correction = np.ones(len(freq_axis))
        full_correction[valid_mask] = correction_gain_50
        
        # === 7. APPLICA CORREZIONE ===
        normalized_fft = original_fft * full_correction
        
        # === 8. STATISTICHE CORREZIONE ===
        max_gain = np.max(correction_gain_50)
        min_gain = np.min(correction_gain_50)
        max_gain_db = 20 * np.log10(max_gain)
        min_gain_db = 20 * np.log10(min_gain)
        
        print(f"📊 Normalization stats (50% conservative):")
        print(f"   Gain range: {min_gain_db:.2f} dB to {max_gain_db:.2f} dB")
        print(f"   Max gain at: {freq_range[np.argmax(correction_gain_50)]/1000:.1f} kHz")
        print(f"   Min gain at: {freq_range[np.argmin(correction_gain_50)]/1000:.1f} kHz")
        print(f"   Estimated error: ±2.9 dB (95% confidence)")
        
        # === 9. MOSTRA OVERLAY GRAFICO ===
        # Rimuovi curve esistenti (se presenti)
        if hasattr(self, 'normalized_fft_curve'):
            try:
                self.plot_widget_fft.plot_widget.removeItem(self.normalized_fft_curve)
            except:
                pass
        
        # Aggiungi curva normalizzata (overlay ROSSO)
        self.normalized_fft_curve = self.plot_widget_fft.plot_widget.plot(
            freq_axis, normalized_fft,
            pen={'color': 'red', 'width': 2},
            name='Normalized (50%)'
        )
        
        # Assicurati che la curva raw mantenga l'accent_color del tema
        if hasattr(self, 'theme_manager'):
            self.theme_manager.apply_theme_to_plot(
                plot_widget_name=self.plot_widget_fft.plot_widget,
                plot_instance=self.fft_curve
            )
        self.fft_curve.setData(freq_axis, original_fft)
        
        # Aggiorna legenda
        if hasattr(self.plot_widget_fft.plot_widget, 'addLegend'):
            try:
                self.plot_widget_fft.plot_widget.addLegend()
            except:
                pass  # Legenda già presente
        
        # === 10. FEEDBACK UTENTE ===
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Normalization Applied (Display Only)")
        msg.setText(
            f"<b>50% Conservative Normalization Applied</b><br><br>"
            f"<b>Correction Range:</b> {min_gain_db:.1f} to {max_gain_db:.1f} dB<br>"
            f"<b>Estimated Error:</b> ±2.9 dB (95% confidence)<br>"
            f"<b>Peak correction:</b> {freq_range[np.argmax(correction_gain_50)]/1000:.1f} kHz<br><br>"
            f"<b>Theme color curve:</b> Raw data<br>"
            f"<font color='red'><b>Red curve:</b> Normalized (50%)</font><br><br>"
            f"⚠️ <b>Original data unchanged</b> (overlay display only)<br>"
            f"📝 Suitable for <b>qualitative analysis</b> only<br>"
            f"📖 See technical report for details"
        )
        msg.setStyleSheet("""
            QMessageBox {
                background-color: #2b2b2b;
                color: #e0e0e0;
            }
            QLabel {
                color: #e0e0e0;
            }
            QTextEdit {
                background-color: #2b2b2b;
                color: #e0e0e0;
            }
            QPushButton {
                background-color: #3a3a3a;
                color: #e0e0e0;
                border-radius: 6px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
        """)
        msg.exec()
        
        print("✅ Normalization overlay displayed")
        
        # Salva dati normalizzati per iFFT window
        self._normalized_fft_cache = {
            'frame_index': frame_index,
            'normalized_fft': normalized_fft,
            'correction_gain': full_correction
        }
    
    def _execute_trim_export(self, params):
        """
        Esegue l'export trimmed del file audio.
        Chiamato da ReplayBaseWindow.open_trim_dialog()
        
        Args:
            params: dict con parametri da TrimRegionDialog
        """
        # Crea exporter con metadata del file
        exporter = AudioTrimExporter(
            parent=self,
            file_path=self.file_path,
            metadata=self.data_manager.header_info
        )
        
        # Esegui export
        return exporter.execute_trim_export(params)
    
    def open_click_detector_dialog(self):
        """
        Apre il dialog per l'algoritmo di rilevamento automatico dei click.
        
        Pipeline a 4 stadi:
        1. Energy threshold (μ + Nσ)
        2. Spectral ratio check (broadband test)
        3. Decay analysis (Hilbert envelope)
        4. Deduplication (frame consecutivi)
        """
        if self.data_manager.total_frames == 0:
            QMessageBox.warning(self, "No Data", "No audio data loaded.\nPlease open a .paudio file first.")
            return
        
        # Verifica che le fasi siano disponibili (necessarie per iFFT)
        file_version = self.data_manager.header_info.get('version', 0)
        has_phases = (file_version >= 3.0) and hasattr(self.data_manager, 'phase_data') and len(self.data_manager.phase_data) > 0
        
        if not has_phases:
            QMessageBox.warning(
                self, 
                "Phase Data Required", 
                "Automatic click detection requires phase information for iFFT reconstruction.\n\n"
                "File version must be ≥ 3.0 with phase data included."
            )
            return
        
        # Importa dialog
        from components.click_detector_dialog import ClickDetectorDialog
        
        # Crea e mostra dialog
        dialog = ClickDetectorDialog(self.data_manager, parent=self)
        dialog.exec()
        
        print("✅ Click Detector Dialog closed")