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

from core.replay_base_window import ReplayBaseWindow
from plotting.plot_manager import BasePlotWidget
from core.audio_trim_export import AudioTrimExporter



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

        # calcola la media globale e la deviazione standard per impostare threshold automatico
        self.fft_mean = np.mean(self.fft_means) if len(self.fft_means) > 0 else 0
        self.fft_std = np.std(self.fft_means) if len(self.fft_means) > 0 else 0

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
            x_range=(x_min_val, x_max_val), y_range=(-0.03, 0.03),
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
        
        self.info_label = QLabel("📊 Raw iFFT signal (no correction)")
        self.info_label.setStyleSheet("color: #888; font-size: 10pt;")
        button_layout.addWidget(self.info_label)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)

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
        else:
            # TORNA A RAW
            self.is_normalized = False
            self._update_display()
    
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
        
        # Inserisci la parte normalizzata 20-80kHz nelle posizioni corrette
        actual_bins_to_copy = min(len(normalized_fft_20_80khz), num_received_bins, len(fft_phases_int8))
        full_spectrum_mag[bin_start:bin_start + actual_bins_to_copy] = normalized_fft_20_80khz[:actual_bins_to_copy]
        full_spectrum_phase[bin_start:bin_start + actual_bins_to_copy] = fft_phases_int8[:actual_bins_to_copy]
        
        # === 7. CONVERTI FASI E CREA SPETTRO COMPLESSO ===
        fft_phases_rad = (full_spectrum_phase / 127.0) * np.pi
        complex_spectrum = full_spectrum_mag * np.exp(1j * fft_phases_rad)
        
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
        
        # UPDATE FFT PLOT
        frame_index = int(current_time_sec / (self.data_manager.frame_duration_ms / 1000))
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
            x_range=(20000, 80000), y_range=(0, 0.125),
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
            x_range=(0, 20), y_range=(0, 0.03),
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
        
        # ✅ CALCOLA FRAME CORRENTE
        current_time_sec = self.current_position_ms / 1000.0
        current_frame_index = int(current_time_sec / (self.data_manager.frame_duration_ms / 1000.0))
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
        
        # Inserisci i dati 20-80kHz nelle posizioni corrette
        full_spectrum_mag[bin_start:bin_start + num_received_bins] = fft_magnitudes
        full_spectrum_phase[bin_start:bin_start + num_received_bins] = fft_phases_int8
        
        # ✅ CONVERTI FASI E CREA SPETTRO COMPLESSO
        fft_phases_rad = (full_spectrum_phase / 127.0) * np.pi
        complex_spectrum = full_spectrum_mag * np.exp(1j * fft_phases_rad)
        
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
        
        # === 2. OTTIENI SPETTRO FFT CORRENTE ===
        current_time_sec = self.current_position_ms / 1000.0
        frame_index = int(current_time_sec / (self.data_manager.frame_duration_ms / 1000))
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