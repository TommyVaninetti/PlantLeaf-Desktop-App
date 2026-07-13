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
Finestra di replay per file audio .paudio - Sistema Multi-Level ottimizzato per Click Detection
Architettura ispirata a replay_window_voltage ma ottimizzata per audio e PC non potenti

ARCHITETTURA MULTI-LEVEL:
- LEVEL 1: Overview (10 FPS, energia media) - SEMPRE CARICATO
- LEVEL 2: Streaming Buffer (100 FPS, energia campionata) - FINESTRA MOBILE 30s
- LEVEL 3: Detail Cache (390 FPS, energia completa) - ON-DEMAND per click detection
"""

import numpy as np
import os
from PySide6.QtWidgets import (QVBoxLayout, QHBoxLayout, QWidget, QTabWidget, 
                              QSplitter, QTableWidget, QTableWidgetItem, 
                              QHeaderView, QLabel, QPushButton, QMessageBox,
                              QSlider, QDoubleSpinBox, QSizePolicy, QDialog,
                              QProgressDialog)
from PySide6.QtCore import Qt, QTimer, QCoreApplication
from PySide6.QtGui import QAction, QFont
from PySide6 import QtCore

from core.replay_base_window import ReplayBaseWindow
from plotting.plot_manager import BasePlotWidget
from core.audio_trim_export import AudioTrimExporter
from core.click_pipeline_v5 import (
    reconstruct_frame_v5,
    compute_hilbert_envelope,
    find_peak,
    suppress_edge_artifacts,
    compute_fft_energy as compute_fft_energy_v5,
    AdaptiveNoiseEstimatorV5,
    run_fit_pipeline_v5,
    find_decay_window_v5,
    _fit_decay_segment,
    compute_features_v5,
    _normalize_fft,
    _feat_fft_features,
    FS as V5_FS,
    FFT_SIZE as V5_FFT_SIZE,
    _BIN_START as V5_BIN_START,
    _BIN_END   as V5_BIN_END,
    K_STAGE1_DEFAULT,
    LEVEL_STD_FACTOR,
    BIN_START_HZ as V5_BAND_LO_HZ,
    BIN_END_HZ   as V5_BAND_HI_HZ,
)


#: Below this, an onset→decay_end span is a noise excursion rather than a click,
#: and its spectrum would carry a resolution of tens of kHz — i.e. nothing usable.
#: The Region FFT dialog falls back to the whole frame instead.
MIN_CLICK_REGION_SAMPLES = 16


def fmt_volts(value, decimals=3):
    """
    Format a voltage with an automatically chosen SI unit (V / mV / µV).

    Since the iFFT amplitude-scale fix (see docs/fft_and_ifft/IFFT_AMPLITUDE_SCALE_FIX.md)
    the reconstructed signal is in true volts, so clicks land in the mV range while
    noise floors can still be sub-mV. Picking the unit from the magnitude keeps both
    readable instead of hard-coding one scale.
    """
    v = abs(float(value))
    if v >= 0.5:
        return f"{float(value):.{decimals}f} V"
    if v >= 5e-4:
        return f"{float(value) * 1e3:.{decimals}f} mV"
    return f"{float(value) * 1e6:.{decimals}f} µV"


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

        # Array delle medie delle FFT PRECALCOLATE (normalizzate per default)
        self.fft_means = np.array([])      # media delle magnitude FFT normalizzate per frame
        self.fft_timestamps = np.array([]) # timestamps corrispondenti alle FFT

        # Stime adattive del rumore per frame — calcolate da AdaptiveNoiseEstimatorV5
        # durante precompute_fft_means (un valore per frame).
        self.E_hat_floor_arr = np.array([])  # Ê_floor(i) [V²] — per la curva soglia Stage 1
        self.noise_floor_arr = np.array([])  # noise_floor(i) [V] — per le feature v5
        self.std_noise_arr   = np.array([])  # std_noise(i) [V]   — per le feature v5

        # Performance settings (adattivi)
        self.overview_fps = 10      # FPS per overview
        self.streaming_fps = 100    # FPS per streaming buffer
        self.memory_limit_mb = 200  # Limite memoria totale

    def precompute_fft_means(self, progress_callback=None):
        """
        Pre-calcola le medie FFT normalizzate e le stime adattive del rumore per ogni frame.

        Viene chiamato una sola volta dopo il caricamento del file.

        In un unico passaggio sequenziale su tutti i frame:
          1. Calcola le magnitudini FFT normalizzate tramite _normalize_fft() (correzione
             microfono SPU0410LR5H-QB al 50%).
          2. Calcola la media delle magnitudini normalizzate → fft_means[i].
          3. Ricostruisce l'iFFT normalizzato e calcola l'inviluppo di Hilbert per ottenere
             env_mean e env_std → aggiorna AdaptiveNoiseEstimatorV5.
          4. Salva le stime di rumore per frame (E_hat_floor, noise_floor, std_noise).

        Questi array sono poi usati da:
          - La curva soglia adattiva nel grafico principale (k × E_hat_floor_arr)
          - La finestra iFFT / Decay Analysis / Show Fit per il frame corrente
          - Il filtro above-threshold table
        """
        if len(self.fft_data) == 0:
            return

        n = self.total_frames
        fs       = self.header_info.get('fs',       V5_FS)
        fft_size = self.header_info.get('fft_size', V5_FFT_SIZE)

        print(f"🔄 Precomputing normalized FFT means + adaptive noise for {n} frames...")

        # Build the FULL half-spectrum frequency axis once (256 bins, 0–100 kHz).
        # _normalize_fft + compute_fft_energy_v5 must both operate on this axis
        # so that the analysis-band slice [V5_BIN_START:V5_BIN_END+1] is correct.
        _full_freq_ax = np.arange(fft_size // 2, dtype=np.float64) * (fs / fft_size)

        means         = np.empty(n, dtype=np.float32)
        E_hat_floors  = np.empty(n, dtype=np.float32)
        noise_floors  = np.empty(n, dtype=np.float32)
        std_noises    = np.empty(n, dtype=np.float32)
        timestamps    = np.empty(n, dtype=np.float64)

        estimator = AdaptiveNoiseEstimatorV5()

        for i in range(n):
            fft_frame = self.fft_data[i]
            timestamps[i] = i * self.frame_duration_ms / 1000.0

            # 1. Pad analysis-band mags (154 bins) into full half-spectrum (256 bins)
            #    and apply mic normalisation — identical to what reconstruct_frame_v5 does.
            full_mags = np.zeros(fft_size // 2, dtype=np.float64)
            n_bins_i  = min(len(fft_frame), V5_BIN_END - V5_BIN_START + 1)
            full_mags[V5_BIN_START : V5_BIN_START + n_bins_i] = \
                np.asarray(fft_frame, dtype=np.float64)[:n_bins_i]
            fft_norm = _normalize_fft(full_mags, _full_freq_ax)   # 256 bins

            # 2. FFT energy over the analysis band (154 bins) [V²].
            #    FIX: previously used analysis-band freq_axis (154 bins) → the slice
            #    fft_norm[51:205] silently truncated to [51:154] = only 103 bins.
            #    Now fft_norm is 256 bins → slice is exactly 154 bins. ✓
            E_i = compute_fft_energy_v5(fft_norm[V5_BIN_START : V5_BIN_END + 1])

            # FIX (primary): store ENERGY [V²] so that fft_means and E_hat_floor_arr
            # share the same units. Previously mean amplitude [V] was stored here,
            # causing the threshold curve (V²) to appear ~1 000× smaller than the
            # data (V) — effectively zero on the plot.
            means[i] = E_i

            # 3. Reconstruct iFFT for B2 estimator (env_mean, env_std)
            if i < len(self.phase_data):
                frame_data = reconstruct_frame_v5(
                    np.asarray(fft_frame, dtype=np.float64),
                    self.phase_data[i],
                    fs, fft_size, normalize=True
                )
            else:
                frame_data = None

            # 4. Update estimator and store per-frame noise estimates
            if frame_data is not None:
                envelope   = compute_hilbert_envelope(frame_data['signal'])
                env_mean_i = float(np.mean(envelope))
                env_std_i  = float(np.std(envelope))
                noise = estimator.update(E_i, env_mean_i, env_std_i)
            else:
                # No reconstruction → no envelope → no honest env_mean/env_std.
                # The previous code synthesised sqrt(E_i) here, which is a band-RMS
                # of the FFT magnitudes and NOT an envelope mean: it differs from the
                # real thing by ~sqrt(K/2) ≈ 9x. Writing it into B2_mean would bias
                # noise_floor upward for every subsequent frame (the buffer is a
                # rolling median) and silently crush peak_SNR for the rest of the
                # recording. Skip the update and carry the current estimates forward.
                noise = {
                    'E_hat_floor': estimator.E_hat_floor,
                    'noise_floor': estimator.noise_floor,
                    'std_noise':   estimator.std_noise,
                }

            E_hat_floors[i] = noise['E_hat_floor']
            noise_floors[i] = noise['noise_floor']
            std_noises[i]   = noise['std_noise']

            if progress_callback is not None and i % 100 == 0:
                progress_callback(i, n)

        self.fft_means        = means        # [V²] energy per frame
        self.fft_timestamps   = timestamps
        self.E_hat_floor_arr  = E_hat_floors  # [V²]
        self.noise_floor_arr  = noise_floors
        self.std_noise_arr    = std_noises

        memory_mb = (means.nbytes + timestamps.nbytes +
                     E_hat_floors.nbytes + noise_floors.nbytes + std_noises.nbytes) / 1024 / 1024
        print(f"✅ Precomputed: {n} frames, {memory_mb:.1f} MB total")

        if n > 0:
            median = np.median(self.fft_means)
            mad    = np.median(np.abs(self.fft_means - median))
            if mad > 0:
                modified_z = 0.6745 * (self.fft_means - median) / mad
                outlier_mask = np.abs(modified_z) > 3.5
            else:
                outlier_mask = self.fft_means > median * 20
            filtered = self.fft_means[~outlier_mask]
            self.fft_mean = float(np.mean(filtered)) if len(filtered) > 0 else 0.0
            self.fft_std  = float(np.std(filtered))  if len(filtered) > 0 else 0.0
            # fft_mean/std are now in V² — display as sqrt(V²) = V for readability
            print(f"📊 Normalized FFT energy stats — "
                  f"mean: {np.sqrt(self.fft_mean)*1000:.3f} mV-rms  "
                  f"std: {np.sqrt(self.fft_std)*1000:.3f} mV-rms")

        if progress_callback is not None:
            progress_callback(n, n)

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
        self.signal_data_raw = suppress_edge_artifacts(signal_data)
        self.signal_data_normalized = None  # Computed below if phases available
        self.is_normalized = False          # Will be set True after normalization

        # TITOLO CON INFO
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

        # Vista Y derivata dai dati: dopo la correzione di scala dell'iFFT il segnale
        # è in volt veri (click nell'ordine dei mV), quindi un range fisso in µV lo
        # lascerebbe completamente fuori schermo.
        y_peak = float(np.max(np.abs(signal_data))) if (
            signal_data is not None and len(signal_data) > 0) else 1e-3
        if not np.isfinite(y_peak) or y_peak <= 0:
            y_peak = 1e-3
        y_peak *= 1.25   # margine

        # Crea il widget del grafico con il range corretto e auto-range per l'asse Y
        self.plot_widget = BasePlotWidget(
            x_label="Time", y_label="Amplitude",
            x_range=(x_min_val, x_max_val), y_range=(-y_peak, y_peak),
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

        # AGGIUNGI PULSANTE NORMALIZZAZIONE
        button_layout = QHBoxLayout()
        
        self.normalize_button = QPushButton("Apply 50% Normalization")
        self.normalize_button.setToolTip(
            "Apply conservative 50% frequency response correction\n"
            "Based on SPU0410LR5H-QB datasheet\n"
            "Estimated error: ±2.9 dB (95% confidence)"
        )
        self.normalize_button.clicked.connect(self.toggle_normalization)
        button_layout.addWidget(self.normalize_button)
        
        # AGGIUNGI PULSANTE ENVELOPE ANALYSIS
        self.envelope_button = QPushButton("Show Hilbert Envelope")
        self.envelope_button.setToolTip(
            "Calculate and display instantaneous amplitude envelope\n"
            "using Hilbert transform (red thick line)"
        )
        self.envelope_button.clicked.connect(self.toggle_envelope)
        button_layout.addWidget(self.envelope_button)
        
        # AGGIUNGI PULSANTE DECAY ANALYSIS
        self.decay_button = QPushButton("Analyze Decay")
        self.decay_button.setToolTip(
            "Check if signal shows exponential decay typical of ultrasonic clicks\n"
            "Analyzes 0.6 ms post-peak window (120 samples @ 200 ksps)"
        )
        self.decay_button.clicked.connect(self.analyze_decay)
        button_layout.addWidget(self.decay_button)

        # PULSANTE TOGGLE FIT CURVE (disabilitato finché non si esegue Analyze Decay)
        self.fit_curve_button = QPushButton("Show Fit Curve")
        self.fit_curve_button.setToolTip(
            "Show/hide the exponential fit overlay on the plot\n"
        )
        self.fit_curve_button.setEnabled(False)
        self.fit_curve_button.clicked.connect(self.toggle_fit_curve)
        button_layout.addWidget(self.fit_curve_button)

        # PULSANTE TOGGLE SHOW IFFT DATA (to show only Hilbert envelope)
        self.toggle_ifft_button = QPushButton("Show Only Envelope")
        self.toggle_ifft_button.setToolTip(
            "Show/hide iFFT signal to focus on envelope analysis\n"
        )
        self.toggle_ifft_button.clicked.connect(self.toggle_ifft_signal)
        button_layout.addWidget(self.toggle_ifft_button)

        # PULSANTE FFT DELLA REGIONE (spettro della regione fittata / a scelta)
        self.region_fft_button = QPushButton("FFT of Region")
        self.region_fft_button.setToolTip(
            "Spectrum of the fitted click region (Ctrl+F).\n"
            "Opens on the decay window if one exists, and lets you drag the\n"
            "selection to analyse any other part of the signal."
        )
        self.region_fft_button.setShortcut("Ctrl+F")
        self.region_fft_button.clicked.connect(self.open_region_fft)
        button_layout.addWidget(self.region_fft_button)

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

        # Variabili per fit curve overlay (populate da analyze_decay)
        self.show_fit_curve = False
        self._last_decay_peak_idx = None
        self._last_v5_features    = None
        self._last_noise_floor    = 0.0
        self._last_std_noise      = 0.0
        self._last_next_env       = None

        # Menubar
        from PySide6.QtWidgets import QMenuBar
        menubar = QMenuBar(self)
        analysis_menu = menubar.addMenu("Analysis")

        actionNormalize = QAction("Toggle Normalization", self)
        actionNormalize.triggered.connect(self.toggle_normalization)
        analysis_menu.addAction(actionNormalize)

        actionEnvelope = QAction("Show Hilbert Envelope", self)
        actionEnvelope.triggered.connect(self.toggle_envelope)
        analysis_menu.addAction(actionEnvelope)

        actionAnalyseDecay = QAction("Analyse Decay", self)
        actionAnalyseDecay.triggered.connect(self.analyze_decay)
        analysis_menu.addAction(actionAnalyseDecay)

        actionShowFitCurve = QAction("Show Fit Curve", self)
        actionShowFitCurve.triggered.connect(self.toggle_fit_curve)
        analysis_menu.addAction(actionShowFitCurve)

        actionShowOnlyEnvelope = QAction("Show Only Envelope", self)
        actionShowOnlyEnvelope.triggered.connect(self.toggle_ifft_signal)
        analysis_menu.addAction(actionShowOnlyEnvelope)

        analysis_menu.addSeparator()

        actionRegionFFT = QAction("FFT of Region…", self)
        actionRegionFFT.setShortcut("Ctrl+F")
        actionRegionFFT.triggered.connect(self.open_region_fft)
        analysis_menu.addAction(actionRegionFFT)

        analysis_menu.addSeparator()

        actionClose = QAction("Close", self)
        actionClose.triggered.connect(self.close)
        analysis_menu.addAction(actionClose)


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

        # mostra in mezzo allo schermo del parent MA CON DIMENSIONE MINORE E TENENDO CONTO DELLE GRAFICHE SOPRATTUTTO PER WINDOWS
        if parent:
            parent_rect = parent.geometry()
            self.resize(parent_rect.width() * 0.8, parent_rect.height() * 0.6)
            self.move(
                parent_rect.x() + (parent_rect.width() - self.width()) // 2,
                parent_rect.y() + (parent_rect.height() - self.height()) // 2
            )
    
    def toggle_normalization(self):
        """Toggle between normalized (default) and raw iFFT display."""
        if not self.is_normalized:
            self._compute_normalized_ifft()
            if self.signal_data_normalized is not None:
                self.is_normalized = True
                self._update_display()
                if self.show_envelope:
                    self._compute_and_show_envelope()
        else:
            self.is_normalized = False
            self._update_display()
            if self.show_envelope:
                self._compute_and_show_envelope()

    def _compute_normalized_ifft(self):
        """
        Compute the 50%-normalized iFFT for this frame using reconstruct_frame_v5.

        Uses the same reconstruction pipeline as click_pipeline_v5 (Tukey taper,
        Gibbs suppression) for full consistency with the detection algorithm.
        """
        if not self.parent or not hasattr(self.parent, 'data_manager'):
            QMessageBox.warning(self, "Error", "Cannot access parent data manager.")
            return

        if not self.has_real_phases:
            QMessageBox.warning(self, "No Phase Data",
                                "Normalization requires phase information (file version >= 3.0).")
            return

        if self.frame_index is None or self.frame_index >= len(self.parent.data_manager.fft_data):
            QMessageBox.warning(self, "Error", "Invalid frame index.")
            return

        dm       = self.parent.data_manager
        fi       = self.frame_index
        fs       = dm.header_info.get('fs',       V5_FS)
        fft_size = dm.header_info.get('fft_size', V5_FFT_SIZE)

        fft_mags   = np.asarray(dm.fft_data[fi],   dtype=np.float64)
        phase_int8 = np.asarray(dm.phase_data[fi], dtype=np.int8) \
                     if fi < len(dm.phase_data) else np.array([], dtype=np.int8)

        frame_data = reconstruct_frame_v5(fft_mags, phase_int8, fs, fft_size, normalize=True)
        if frame_data is None:
            QMessageBox.warning(self, "Error", "iFFT reconstruction failed.")
            return

        self.signal_data_normalized = frame_data['signal']
        print(f"✅ Normalized iFFT via reconstruct_frame_v5  "
              f"(peak {fmt_volts(np.max(np.abs(self.signal_data_normalized)), 2)})")

    def _rescale_y_to(self, signal):
        """Fit the Y view to the signal currently on screen (raw and normalized
        differ by the mic correction, up to ~10 dB)."""
        if signal is None or len(signal) == 0:
            return
        peak = float(np.max(np.abs(signal)))
        if not np.isfinite(peak) or peak <= 0:
            return
        self.plot_widget.plot_widget.setYRange(-peak * 1.25, peak * 1.25, padding=0)

    def _update_display(self):
        """Show ONLY the selected signal (normalized or raw) — not both overlaid."""
        if self.is_normalized and self.signal_data_normalized is not None:
            self.ifft_curve.setData(self.time_data, self.signal_data_normalized)
            self._rescale_y_to(self.signal_data_normalized)
            #As a color, use a darker accent color = border-color of QPushButton:checked
            self.ifft_curve.setPen({'color': self.parent.theme_manager.get_darker_accent_color(), 'width': 2})
            self.normalize_button.setText("Show Raw iFFT")
            self.info_label.setText("🔧 Normalized iFFT (50% correction, ±2.9 dB)")
            self.info_label.setStyleSheet("color: red; font-weight: bold; font-size: 10pt;")
        else:
            self.ifft_curve.setData(self.time_data, self.signal_data_raw)
            self._rescale_y_to(self.signal_data_raw)
            if self.parent and hasattr(self.parent, 'theme_manager'):
                self.parent.theme_manager.apply_theme_to_plot(
                    plot_widget_name=self.plot_widget.plot_widget,
                    plot_instance=self.ifft_curve
                )
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

    def toggle_ifft_signal(self):
        """Toggle visualizzazione segnale iFFT per focalizzarsi solo sull'envelope"""
        if self.ifft_curve is not None:
            if self.ifft_curve.isVisible():
                self.ifft_curve.hide()
                #mostra hilbert envelope se è nascosto
                if self.envelope_curve is not None and not self.envelope_curve.isVisible():
                    self.envelope_curve.show()
                else:
                    self._compute_and_show_envelope()
                self.toggle_ifft_button.setText("Show iFFT Signal")
                self.envelope_button.setEnabled(False)
            else:
                self.ifft_curve.show()
                self.toggle_ifft_button.setText("Show Only Envelope")
                self.envelope_button.setEnabled(True)
    
    def _compute_and_show_envelope(self):
        """Calcola e visualizza l'inviluppo di Hilbert"""
        # Usa il segnale corrente (raw o normalized)
        current_signal = self.signal_data_normalized if self.is_normalized else self.signal_data_raw
        
        print("🔧 Computing Hilbert envelope...")
        
        # Calcola inviluppo
        self.envelope_data = compute_hilbert_envelope(current_signal)
        
        # ✅ FIX: Trova il picco sull'ENVELOPE (non sul segnale raw).
        # Il segnale raw oscilla attorno alla portante 50 kHz → argmax del raw
        # può cadere in un campione diverso rispetto al massimo fisico dell'ampiezza.
        # L'envelope di Hilbert è il riferimento corretto per il picco di ampiezza.
        peak_idx, peak_amp = find_peak(self.envelope_data)
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
        """
        Compute all 17 v5 features for this frame and display results.

        Uses compute_features_v5() with per-frame noise estimates from
        AudioDataManager (pre-computed by AdaptiveNoiseEstimatorV5 at load time).
        Previous / next frame data is fetched for pre-window and post-SNR accuracy.
        """
        current_signal = self.signal_data_normalized if self.is_normalized else self.signal_data_raw

        # Always recompute envelope from the current signal to stay in sync
        self.envelope_data = compute_hilbert_envelope(current_signal)
        peak_idx, peak_amp = find_peak(self.envelope_data)
        peak_time = self.time_data[peak_idx]

        print(f"\n🔍 Decay Analysis v5 ({'NORMALIZED' if self.is_normalized else 'RAW'}):")
        print(f"   Peak at t = {peak_time:.6f} s (sample {peak_idx})")
        print(f"   Peak amplitude: {fmt_volts(peak_amp, 3)}")

        # ── Retrieve per-frame noise estimates ────────────────────────────────
        dm = self.parent.data_manager if (self.parent and hasattr(self.parent, 'data_manager')) else None
        fi = self.frame_index if self.frame_index is not None else 0

        if dm is not None and fi < len(dm.noise_floor_arr):
            noise_floor = float(dm.noise_floor_arr[fi])
            std_noise   = float(dm.std_noise_arr[fi])
        else:
            noise_floor = 0.0
            std_noise   = 0.0

        # ── Build FFT feature inputs ──────────────────────────────────────────
        if dm is not None and fi < len(dm.fft_data):
            raw_mags  = np.asarray(dm.fft_data[fi], dtype=np.float64)
            freq_axis = np.array(dm.frequency_axis)
            # Pad raw analysis-band magnitudes into full half-spectrum
            fft_size_full = V5_FFT_SIZE
            full_freq  = np.arange(fft_size_full // 2) * (V5_FS / fft_size_full)
            full_mags  = np.zeros(fft_size_full // 2, dtype=np.float64)
            n_b = min(len(raw_mags), V5_BIN_END - V5_BIN_START + 1)
            full_mags[V5_BIN_START : V5_BIN_START + n_b] = raw_mags[:n_b]
            fft_norm_full = _normalize_fft(full_mags, full_freq)
        else:
            full_freq     = np.zeros(V5_FFT_SIZE // 2)
            fft_norm_full = np.zeros(V5_FFT_SIZE // 2)

        # ── Fetch adjacent frame envelopes for pre/post SNR ───────────────────
        prev_env = None; prev_sig = None
        next_env = None

        if dm is not None and fi > 0:
            prev_frame = reconstruct_frame_v5(
                np.asarray(dm.fft_data[fi - 1], dtype=np.float64),
                dm.phase_data[fi - 1] if fi - 1 < len(dm.phase_data) else np.array([], dtype=np.int8),
                normalize=self.is_normalized
            )
            if prev_frame:
                prev_sig = prev_frame['signal']
                prev_env = compute_hilbert_envelope(prev_sig)

        if dm is not None and fi + 1 < dm.total_frames:
            next_frame = reconstruct_frame_v5(
                np.asarray(dm.fft_data[fi + 1], dtype=np.float64),
                dm.phase_data[fi + 1] if fi + 1 < len(dm.phase_data) else np.array([], dtype=np.int8),
                normalize=self.is_normalized
            )
            if next_frame:
                next_env = compute_hilbert_envelope(next_frame['signal'])

        # ── Compute all 17 features ───────────────────────────────────────────
        features = compute_features_v5(
            signal              = current_signal,
            envelope            = self.envelope_data,
            fft_norm            = fft_norm_full,
            freq_axis           = full_freq,
            noise_floor         = noise_floor,
            std_noise           = std_noise,
            peak_idx            = peak_idx,
            next_frame_envelope = next_env,
            prev_frame_envelope = prev_env,
            prev_frame_signal   = prev_sig,
        )

        print(f"   τ = {features['tau_ms']:.4f} ms   R² = {features['R2']:.4f}")

        self._show_decay_results_dialog(
            frame_index = fi,
            peak_time   = peak_time,
            peak_idx    = peak_idx,
            peak_amp    = peak_amp,
            noise_floor = noise_floor,
            std_noise   = std_noise,
            features    = features,
        )

        # Save fit results for the Show Fit overlay
        self._last_decay_peak_idx  = peak_idx
        self._last_v5_features     = features
        self._last_noise_floor     = noise_floor
        self._last_std_noise       = std_noise
        self._last_next_env        = next_env

        self.fit_curve_button.setEnabled(True)
        if self.show_fit_curve:
            self._overlay_fit_curve(peak_idx, features, noise_floor, std_noise, next_env)
        else:
            self._remove_fit_overlay()
            self.fit_curve_button.setText("Show Fit Curve")

    def toggle_fit_curve(self):
        """Toggle: mostra o nasconde la curva esponenziale del fit sul grafico."""
        if not hasattr(self, '_last_v5_features') or self._last_v5_features is None:
            return

        self.show_fit_curve = not self.show_fit_curve

        if self.show_fit_curve:
            self._overlay_fit_curve(
                self._last_decay_peak_idx,
                self._last_v5_features,
                getattr(self, '_last_noise_floor', 0.0),
                getattr(self, '_last_std_noise',   0.0),
                getattr(self, '_last_next_env',    None),
            )
            self.fit_curve_button.setText("Hide Fit Curve")
        else:
            self._remove_fit_overlay()
            self.fit_curve_button.setText("Show Fit Curve")

    def _remove_fit_overlay(self):
        """Remove all fit overlay graphics from the iFFT plot."""
        for attr in ('_fit_curve_item', '_fit_region_item', '_peak_fit_line',
                     '_noise_floor_line', '_noise_std_line'):
            item = getattr(self, attr, None)
            if item is not None:
                try:
                    self.plot_widget.plot_widget.removeItem(item)
                except Exception:
                    pass
                setattr(self, attr, None)

    def _overlay_fit_curve(self, peak_idx: int, features: dict,
                           noise_floor: float, std_noise: float,
                           next_frame_envelope=None):
        """
        Overlay the v5 exponential fit and noise reference lines on the iFFT plot.

        Drawn elements:
          1. Exponential fit curve  A₀·exp(-t/τ)  over [decay_start → decay_end].
             Green / orange / red depending on R².
          2. Vertical dashed line at the peak.
          3. Horizontal dashed line at noise_floor.
          4. Horizontal dotted line at noise_floor + std_noise  (LEVEL).
        """
        self._remove_fit_overlay()

        if len(self.time_data) == 0 or self.envelope_data is None:
            return

        tau_ms  = features.get('tau_ms',  -1.0)
        R2      = features.get('R2',       0.0)

        # Guard: no valid decay
        if tau_ms <= 0:
            #show only noise floor and level lines without fit curve
            fit_color = '#EF5350'  # Red for invalid fit
            self._noise_floor_line = self.plot_widget.plot_widget.addLine(
                y=noise_floor,
                pen={'color': '#80DEEA', 'width': 1.2, 'style': QtCore.Qt.DashLine},
                label=f' noise_floor  {fmt_volts(noise_floor, 3)}',
                labelOpts={'position': 0.05, 'color': '#80DEEA'},
            )
            level = noise_floor + std_noise
            self._noise_std_line = self.plot_widget.plot_widget.addLine(
                y=level,
                pen={'color': '#CE93D8', 'width': 1.2, 'style': QtCore.Qt.DotLine},
                label=f' noise+σ  {fmt_volts(level, 3)}',
                labelOpts={'position': 0.05, 'color': '#CE93D8'},
            )
            return

        # Recompute decay window from the envelope to get decay_start/decay_end
        window = find_decay_window_v5(
            self.envelope_data, peak_idx, noise_floor, std_noise, next_frame_envelope
        )
        decay_start = window['decay_start']
        decay_end   = min(window['decay_end'], len(self.time_data) - 1)

        if decay_start >= decay_end:
            return

        # ── Reconstruct the exponential curve A₀·exp(-t/τ) ───────────────────
        fs = V5_FS
        A0 = float(self.envelope_data[decay_start]) if decay_start < len(self.envelope_data) else 0.0
        tau_s = tau_ms / 1000.0
        n_pts = decay_end - decay_start
        t_arr = np.arange(n_pts) / fs
        fit_env = A0 * np.exp(-t_arr / tau_s)
        fit_time = self.time_data[decay_start : decay_start + n_pts]
        if len(fit_time) != len(fit_env):
            min_len = min(len(fit_time), len(fit_env))
            fit_time = fit_time[:min_len]; fit_env = fit_env[:min_len]

        # ── Colour by R² quality ──────────────────────────────────────────────
        fit_color = '#00E676' if R2 >= 0.70 else ('#FFA726' if R2 >= 0.45 else '#EF5350')

        self._fit_curve_item = self.plot_widget.plot_widget.plot(
            fit_time, fit_env,
            pen={'color': fit_color, 'width': 2.5, 'style': QtCore.Qt.DashLine},
            name=f'Fit  τ={tau_ms:.3f} ms  R²={R2:.3f}'
        )

        # ── Vertical line at peak ─────────────────────────────────────────────
        self._peak_fit_line = self.plot_widget.plot_widget.addLine(
            x=self.time_data[peak_idx],
            pen={'color': '#FFD600', 'width': 1.5, 'style': QtCore.Qt.DotLine},
        )

        # ── Horizontal line: noise_floor ──────────────────────────────────────
        self._noise_floor_line = self.plot_widget.plot_widget.addLine(
            y=noise_floor,
            pen={'color': '#80DEEA', 'width': 1.2, 'style': QtCore.Qt.DashLine},
            label=f'noise_floor  {fmt_volts(noise_floor, 3)}',
            labelOpts={'position': 0.05, 'color': '#80DEEA'},
        )

        # ── Horizontal line: noise_floor + std_noise  (LEVEL) ─────────────────
        level = noise_floor + std_noise
        self._noise_std_line = self.plot_widget.plot_widget.addLine(
            y=level,
            pen={'color': '#CE93D8', 'width': 1.2, 'style': QtCore.Qt.DotLine},
            label=f'noise+σ  {fmt_volts(level, 3)}',
            labelOpts={'position': 0.05, 'color': '#CE93D8'},
        )

        print(f"✅ Fit overlay v5: τ={tau_ms:.3f} ms  R²={R2:.3f}  "
              f"noise={fmt_volts(noise_floor, 3)}  level={fmt_volts(level, 3)}")


    def _show_decay_results_dialog(self, frame_index, peak_time, peak_idx,
                                   peak_amp, noise_floor, std_noise, features):
        """
        Display all 17 v5 features for this frame in a scrollable dialog.

        Shows computed value alongside a physically motivated expected range in
        parentheses (guidance for the SVM, not hard thresholds).
        """
        from PySide6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QTextEdit,
                                       QPushButton, QHBoxLayout)

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Decay Analysis — Frame {frame_index}  (v5)")
        dialog.setMinimumSize(700, 780)

        layout = QVBoxLayout(dialog)

        title = QLabel(f"<b style='font-size:16pt;'>Frame {frame_index}</b>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        mode = QLabel(f"<i>{'50 % Normalized iFFT' if self.is_normalized else 'Raw iFFT'}</i>")
        mode.setAlignment(Qt.AlignCenter)
        layout.addWidget(mode)

        text = QTextEdit()
        text.setReadOnly(True)

        nf_str  = fmt_volts(noise_floor, 3)
        std_str = fmt_volts(std_noise,   3)
        f = features   # shorthand

        def fmt(v, unit='', decimals=4):
            if v is None: return 'N/A'
            return f"{v:.{decimals}f}{unit}"

        html = f"""
<div style='font-family:monospace; font-size:10.5pt; line-height:1.65;'>

<p><b style='font-size:12pt; text-decoration:underline;'>Frame metadata</b></p>
<table style='margin-left:16px; border-collapse:collapse;'>
  <tr><td style='width:220px;padding:3px;'>Peak time in frame:</td>
      <td><b>{peak_time:.6f} s</b>  (sample {peak_idx}/512)</td></tr>
  <tr><td style='padding:3px;'>Peak amplitude:</td>
      <td><b>{fmt_volts(peak_amp, 3)}</b></td></tr>
  <tr><td style='padding:3px;'>Noise floor (adaptive):</td>
      <td>{nf_str}</td></tr>
  <tr><td style='padding:3px;'>Noise std (adaptive):</td>
      <td>{std_str}</td></tr>
</table>

<p style='margin-top:12px;'><b style='font-size:12pt; text-decoration:underline;'>
  v5 Feature Vector  <span style='font-size:9pt;font-weight:normal;'>(expected ranges are guidance — SVM decides)</span>
</b></p>

<table style='margin-left:16px; border-collapse:collapse; width:95%;'>
<tr style='background-color:rgba(100,100,255,0.12);'>
  <th style='padding:5px;text-align:left;width:50%;'>Feature</th>
  <th style='padding:5px;text-align:left;'>Value</th>
  <th style='padding:5px;text-align:left;'>Expected (genuine click)</th>
</tr>

<tr><td style='padding:4px;'><b>1. peak_SNR</b></td>
    <td>{fmt(f['peak_SNR'],decimals=1)}</td>
    <td><i>≫ 1  (typically 10–1000+)</i></td></tr>
<tr style='background-color:rgba(128,128,128,0.07);'>
    <td style='padding:4px;'><b>2. pre_SNR</b></td>
    <td>{fmt(f['pre_SNR'],decimals=3)}</td>
    <td><i>≈ 1.0  (silence before click)</i></td></tr>
<tr><td style='padding:4px;'><b>3. post_SNR</b></td>
    <td>{fmt(f['post_SNR'],decimals=3)}</td>
    <td><i>≈ 1.0  (return to silence)</i></td></tr>
<tr style='background-color:rgba(128,128,128,0.07);'>
    <td style='padding:4px;'><b>4. rise_time_ms</b></td>
    <td>{fmt(f['rise_time_ms'],' ms',4)}</td>
    <td><i>0.025 – 0.3 ms</i></td></tr>
<tr><td style='padding:4px;'><b>5. fall_time_ms</b></td>
    <td>{fmt(f['fall_time_ms'],' ms',4)}</td>
    <td><i>&gt; rise_time  (decay slower than rise)</i></td></tr>
<tr style='background-color:rgba(128,128,128,0.07);'>
    <td style='padding:4px;'><b>6. asymmetry_integral</b></td>
    <td>{fmt(f['asymmetry_integral'],decimals=4)}</td>
    <td><i>positive  (decay &gt; rise)</i></td></tr>
<tr><td style='padding:4px;'><b>7. ZCR_pre</b></td>
    <td>{fmt(f['ZCR_pre'],' crossings/ms',2)}</td>
    <td><i>low  (silence before click)</i></td></tr>
<tr style='background-color:rgba(128,128,128,0.07);'>
    <td style='padding:4px;'><b>8. ZCR_click</b></td>
    <td>{fmt(f['ZCR_click'],' crossings/ms',2)}</td>
    <td><i>oscillation rate during click</i></td></tr>
<tr><td style='padding:4px;'><b>9. ZCR_post</b></td>
    <td>{fmt(f['ZCR_post'],' crossings/ms',2)}</td>
    <td><i>decreasing during decay</i></td></tr>
<tr style='background-color:rgba(128,128,128,0.07);'>
    <td style='padding:4px;'><b>10. kurtosis</b></td>
    <td>{fmt(f['kurtosis'],decimals=2)}</td>
    <td><i>15 – 50  (impulsive event)</i></td></tr>
<tr><td style='padding:4px;'><b>11. centroid_shift_hz</b></td>
    <td>{fmt(f['centroid_shift_hz']/1000,' kHz',2)}</td>
    <td><i>&gt; 2–5 kHz  (high freqs decay first)</i></td></tr>
<tr style='background-color:rgba(128,128,128,0.07);'>
    <td style='padding:4px;'><b>12. τ (tau_ms)</b></td>
    <td>{'N/A (no valid decay)' if f['tau_ms'] < 0 else fmt(f['tau_ms'],' ms',4)}</td>
    <td><i>0.05 – 1.3 ms  (cavitation)</i></td></tr>
<tr><td style='padding:4px;'><b>13. R²</b></td>
    <td>{fmt(f['R2'],decimals=4)}</td>
    <td><i>≥ 0.45  (exponential fit quality)</i></td></tr>
<tr style='background-color:rgba(128,128,128,0.07);'>
    <td style='padding:4px;'><b>14. fit_coverage</b></td>
    <td>{fmt(f['fit_coverage'],decimals=3)}</td>
    <td><i>0.7 – 1.0  (fraction of decay used)</i></td></tr>
<tr><td style='padding:4px;'><b>15. SPR</b></td>
    <td>{fmt(f['SPR'],decimals=2)}</td>
    <td><i>≤ 20  (broadband, not tonal)</i></td></tr>
<tr style='background-color:rgba(128,128,128,0.07);'>
    <td style='padding:4px;'><b>16. R_spectral</b></td>
    <td>{fmt(f['R_spectral'],decimals=3)}</td>
    <td><i>descriptive — E[20-40] / E[40-80]</i></td></tr>
<tr><td style='padding:4px;'><b>17. FPE</b></td>
    <td>{fmt(f['FPE_hz']/1000,' kHz',1)}</td>
    <td><i>dominant frequency in analysis band</i></td></tr>
</table>

<p style='margin-top:10px; margin-left:16px; font-size:9pt; color:gray;'>
  No hard thresholds — all features are fed to the SVM classifier.
  Expected ranges are physically motivated guidance, not pass/fail criteria.
</p>
</div>"""

        text.setHtml(html)
        layout.addWidget(text)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setMinimumWidth(100)
        close_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(close_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

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

    # ── FFT of Region ────────────────────────────────────────────────────────

    def _build_frame_context(self):
        """
        Stitch frames fi-1, fi, fi+1 into one continuous trace.

        A click's decay routinely runs past the end of its own 512-sample frame:
        find_decay_window_v5() already concatenates the NEXT frame's envelope and
        can return decay_end > 511. There is no time-domain equivalent in the
        pipeline, so we build one here. The previous frame is included so the
        pre-click noise window is reachable too.

        Each frame is reconstructed independently (own Tukey taper, own Gibbs
        suppression), so the joins carry a mild seam — we return their positions
        so the dialog can draw them honestly rather than hiding them.

        Returns
        -------
        dict or None:
            'time'     : absolute time axis [s]
            'signal'   : stitched samples [V]
            'envelope' : Hilbert envelope of the stitched signal
            'origin'   : index, within `signal`, of sample 0 of the CURRENT frame
            'seams'    : absolute times of the frame joins [s]
            'fs', 'fft_size'
        """
        dm = getattr(self.parent, 'data_manager', None)
        if dm is None or self.frame_index is None:
            return None

        fi = int(self.frame_index)
        fs = dm.header_info.get('fs', V5_FS)
        fft_size = dm.header_info.get('fft_size', V5_FFT_SIZE)

        def reco(idx):
            if idx < 0 or idx >= dm.total_frames or idx >= len(dm.fft_data):
                return None
            phases = dm.phase_data[idx] if idx < len(dm.phase_data) else np.array([], dtype=np.int8)
            fr = reconstruct_frame_v5(
                np.asarray(dm.fft_data[idx], dtype=np.float64),
                np.asarray(phases, dtype=np.int8),
                fs, fft_size, normalize=self.is_normalized,
            )
            return fr['signal'] if fr is not None else None

        prev_sig, cur_sig, next_sig = reco(fi - 1), reco(fi), reco(fi + 1)
        if cur_sig is None:
            return None

        parts = []
        seam_idx = []
        if prev_sig is not None:
            parts.append(prev_sig)
            seam_idx.append(len(prev_sig))
        origin = sum(len(p) for p in parts)
        parts.append(cur_sig)
        if next_sig is not None:
            seam_idx.append(sum(len(p) for p in parts))
            parts.append(next_sig)

        signal = np.concatenate(parts)

        # Absolute time: sample `origin` is the start of frame fi.
        frame_start = fi * (fft_size / fs)
        time = frame_start + (np.arange(len(signal)) - origin) / fs

        return {
            'time': time,
            'signal': signal,
            'envelope': compute_hilbert_envelope(signal),
            'origin': origin,
            'seams': [float(time[i]) for i in seam_idx if 0 <= i < len(time)],
            'fs': fs,
            'fft_size': fft_size,
        }

    def _find_click_onset(self, ctx, peak_in_ext, level):
        """
        Locate where the click emerged from the noise: scanning BACKWARD from the
        peak, the last envelope sample still below
        LEVEL = noise_floor + LEVEL_STD_FACTOR × std_noise.

        This is exactly the boundary the v5 pipeline itself uses — `rise_start` in
        _feat_rise_fall_time, and the pre-window boundary in _build_pre_window,
        which is why the pre-click noise window ends here.

        The search runs over the STITCHED envelope, not the current frame alone,
        so an onset that falls in the previous frame is found rather than clamped
        away (_build_pre_window extends into the previous frame for the same
        reason). Returns an index into ctx['signal'].
        """
        env_ext = ctx['envelope']
        for n in range(int(peak_in_ext) - 1, -1, -1):
            if env_ext[n] < level:
                return n
        return 0        # never dropped below LEVEL — the click fills the context

    def _resolve_fitted_region(self, ctx):
        """
        Locate the click region, "if it exists".

        The analysed region runs from the click's ONSET (where the envelope first
        rises above LEVEL) to `decay_end`, so the spectrum covers the whole event
        — attack and decay — rather than only its tail. `decay_start → decay_end`
        remains available as a preset.

        Reuses the result of Analyze Decay when it has already been run, and
        otherwise computes the decay window on the fly — so the user is NOT
        forced to press Analyze Decay first.

        Returns (region, markers, banner), where `region` is (t0, t1) in absolute
        seconds and `banner` is None when a genuine fit was found.
        """
        dm = getattr(self.parent, 'data_manager', None)
        fi = int(self.frame_index) if self.frame_index is not None else 0
        fs, origin = ctx['fs'], ctx['origin']
        time = ctx['time']

        def t_of(sample_in_frame):
            """Frame-relative sample index → absolute time, clipped to the trace."""
            return t_ext(origin + sample_in_frame)

        def t_ext(sample_in_ext):
            """Stitched-context sample index → absolute time."""
            i = int(np.clip(sample_in_ext, 0, len(time) - 1))
            return float(time[i])

        whole_frame = (t_of(0), t_of(len(self.signal_data_raw) - 1))

        # Noise estimates — without them there is neither a LEVEL nor a decay window.
        if dm is None or fi >= len(dm.noise_floor_arr) or fi >= len(dm.std_noise_arr):
            return whole_frame, {}, ("No adaptive noise estimate for this frame — "
                                     "showing the whole frame. Drag the region to analyse "
                                     "any part of the signal.")

        noise_floor = float(dm.noise_floor_arr[fi])
        std_noise = float(dm.std_noise_arr[fi])

        signal = self.signal_data_normalized if self.is_normalized else self.signal_data_raw
        envelope = compute_hilbert_envelope(signal)
        peak_idx, _ = find_peak(envelope)

        # Next frame's envelope, so decay_end may legitimately exceed 511.
        next_env = None
        if origin + len(signal) < len(ctx['signal']):
            next_env = compute_hilbert_envelope(ctx['signal'][origin + len(signal):])

        window = find_decay_window_v5(envelope, peak_idx, noise_floor, std_noise, next_env)
        d_start = int(window['decay_start'])
        d_end = int(window['decay_end'])

        # ── Click onset — the same LEVEL crossing the v5 features use ─────────
        level = noise_floor + LEVEL_STD_FACTOR * std_noise
        onset_ext = self._find_click_onset(ctx, origin + peak_idx, level)

        markers = {
            'click start': t_ext(onset_ext),
            'peak': t_of(peak_idx),
            'decay start': t_of(d_start),
            'decay end': t_of(d_end),
        }

        if d_end <= d_start:
            return whole_frame, markers, ("No decay window could be located — "
                                          "showing the whole frame.")

        self._region_peak_idx = peak_idx
        self._region_decay = (d_start, d_end)
        self._region_onset_ext = onset_ext

        # On a frame with no real click the "peak" is just a noise excursion, so
        # onset and decay_end collapse onto each other. A handful of samples has
        # no usable spectrum (Δf would be tens of kHz), so fall back to the whole
        # frame and say so, rather than showing a meaningless curve.
        n_region = (origin + d_end) - onset_ext
        if n_region < MIN_CLICK_REGION_SAMPLES:
            return whole_frame, markers, (
                f"No click-like region in this frame (only {n_region} samples "
                f"between onset and decay end) — showing the whole frame.")

        # Default: the WHOLE click — onset through the end of the decay.
        region = (t_ext(onset_ext), t_of(d_end))

        # Is it a *valid fit*, or just a decay window?
        if self._last_v5_features is not None:
            tau = self._last_v5_features.get('tau_ms')
        else:
            tau = _fit_decay_segment(
                window['extended_envelope'], d_start, d_end, fs
            ).get('tau_ms')

        banner = None
        if tau is None or tau <= 0:
            banner = ("No valid exponential fit for this frame (τ undefined) — "
                      "the region shown is onset → raw decay window.")

        return region, markers, banner

    def open_region_fft(self):
        """Open the Region FFT dialog on the fitted click region (Ctrl+F)."""
        from components.region_fft_dialog import RegionFFTDialog
        from core.click_pipeline_v5 import PRE_WINDOW_SAMPLES

        ctx = self._build_frame_context()
        if ctx is None:
            QMessageBox.warning(self, "Region FFT",
                                "Could not reconstruct this frame for spectral analysis.")
            return

        region, markers, banner = self._resolve_fitted_region(ctx)

        origin, fs, time = ctx['origin'], ctx['fs'], ctx['time']
        n_frame = len(self.signal_data_raw)

        def t_of(s):
            return float(time[int(np.clip(origin + s, 0, len(time) - 1))])

        def t_ext(s):
            return float(time[int(np.clip(s, 0, len(time) - 1))])

        # ── Presets ──────────────────────────────────────────────────────────
        # Default = the whole click: onset → decay end.
        #
        # There is deliberately no "decay start → decay end" preset: decay_start
        # lands on (or within a couple of samples of) the peak in practice, so it
        # was indistinguishable from "Peak → decay end".
        presets = [("Click (onset → decay end)", region)]

        peak_idx = getattr(self, '_region_peak_idx', None)
        decay = getattr(self, '_region_decay', None)
        onset_ext = getattr(self, '_region_onset_ext', None)

        if peak_idx is not None and decay is not None:
            presets.append(("Peak → decay end", (t_of(peak_idx), t_of(decay[1]))))
        presets.append(("Whole frame", (t_of(0), t_of(n_frame - 1))))

        # Pre-click noise: the PRE_WINDOW_SAMPLES immediately BEFORE the onset —
        # the same window _build_pre_window feeds to pre_SNR / ZCR_pre.
        if onset_ext is not None and onset_ext - PRE_WINDOW_SAMPLES >= 0:
            presets.append(("Pre-click noise",
                            (t_ext(onset_ext - PRE_WINDOW_SAMPLES), t_ext(onset_ext))))

        # Reference overlay: the frame's transmitted spectrum. Since the iFFT
        # scale fix both are in true volts, so they share one axis directly.
        reference = None
        dm = self.parent.data_manager
        fi = int(self.frame_index)
        if fi < len(dm.fft_data):
            freqs, mags = self.parent._compute_fft_for_display(fi)
            reference = (np.asarray(freqs), np.asarray(mags), "Frame FFT (transmitted)")

        title = f"FFT of Region — Frame {fi}"
        title += "  [Normalized]" if self.is_normalized else "  [Raw]"

        self._region_fft_win = RegionFFTDialog(
            ctx['time'], ctx['signal'], fs,
            parent=self,
            theme_manager=getattr(self.parent, 'theme_manager', None),
            unit_y='V',
            title=title,
            band=(V5_BAND_LO_HZ, V5_BAND_HI_HZ),      # 20–80 kHz analysis band
            initial_region=region,
            markers=markers,
            seams=ctx['seams'],
            presets=presets,
            reference=reference,
            banner=banner,
        )
        self._region_fft_win.show()


# Signal processing utilities (compute_hilbert_envelope, find_peak,
# suppress_edge_artifacts, run_fit_pipeline_v5, etc.) are now in
# core/click_pipeline_v5.py and imported at the top of this file.

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

        # Flag to indicate if normalized means are being used for FFT analysis
        self._using_normalized_means = True #defualt, always normalized

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
        if hasattr(self, 'actionToggleNormalized'):
            self.actionToggleNormalized.triggered.connect(self.toggle_normalized_data)
        if hasattr(self, 'actionDataCollection'):
            self.actionDataCollection.triggered.connect(self._on_open_data_collection_dialog)

        # Applica tema
        self._load_saved_settings()
        
        # Connetti segnali
        self.playback_speed_changed.connect(self._on_playback_speed_changed)
        self.playback_position_changed.connect(self._on_position_changed)
        
        # Range velocità limitato per audio
        self.velocity.setRange(0.1, 1.0)

        #imposta la vista iniziale dell'asse x del tempo sui primi 20s
        # IMPOSTA LA VISTA INIZIALE SULLA DIMENSIONE DELLA FINESTRA DI STREAMING
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
        
        # UPDATE FFT PLOT (FIX: use round() for consistent frame index calculation)
        frame_index = int(round(self.current_position_ms / self.data_manager.frame_duration_ms))
        frame_index = max(0, min(frame_index, self.data_manager.total_frames - 1))
        
        if frame_index < len(self.data_manager.fft_data):
            # Respect normalization state instead of always showing raw
            freq_axis, display_mags = self._compute_fft_for_display(frame_index)
            self.fft_curve.setData(freq_axis, display_mags)

            # Color: darker accent for normalized, theme accent for raw
            if getattr(self, '_using_normalized_means', True):
                self.fft_curve.setPen({'color': self.theme_manager.get_darker_accent_color(), 'width': 2})
            else:
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

            # Use pre-computed normalized means (fast — already computed at load time)
            if i < len(self.data_manager.fft_means):
                energy = float(self.data_manager.fft_means[i])
            else:
                energy = float(np.mean(np.abs(self.data_manager.fft_data[i])))
            
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

            # Use pre-computed normalized fft_means when available (fast path).
            # fft_means are pre-computed in precompute_fft_means using normalized magnitudes.
            if frame_idx < len(self.data_manager.fft_means):
                signal_sample = float(self.data_manager.fft_means[frame_idx])
            else:
                signal_sample = float(np.mean(np.abs(self.data_manager.fft_data[frame_idx])))
            
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
            freq_axis, display_mags = self._compute_fft_for_display(0)
            self.fft_curve.setData(freq_axis, display_mags)
            if getattr(self, '_using_normalized_means', True):
                self.fft_curve.setPen({'color': self.theme_manager.get_darker_accent_color(), 'width': 2})

        # Setup position line
        if hasattr(self, 'time_position_line'):
            self.time_position_line.setPos(0)

        self._update_time_labels()
    
        # Build the adaptive threshold curve and apply the above-threshold filter
        self._update_threshold_line()
        self._apply_threshold_filter()
    
    
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
            x_range=(20000, 80000), y_range=(0, 0.003),
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
            x_label="Time", y_label="Mean FFT Energy",
            x_range=(0, 20), y_range=(0.00000005, 0.0000002),
            x_min=0, x_max=None, y_min=0, y_max=1e-4,
            unit_x="s", unit_y="V²", parent=self
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
        self.PeakThresholdLabel.setText("Stage 1 threshold multiplier  k :")
        self.PeakThresholdLabel.setAlignment(Qt.AlignCenter)
        table_layout.addWidget(self.PeakThresholdLabel)

        # Layout orizzontale per button e spinbox
        threshold_layout = QHBoxLayout()
        table_layout.addLayout(threshold_layout)

        # SpinBox: controls k — the Stage 1 multiplier (E_i > k × Ê_floor)
        self.PeakThresholdSpinBox = QDoubleSpinBox(table_container)
        self.PeakThresholdSpinBox.setObjectName(u"PeakThresholdSpinBox")
        self.PeakThresholdSpinBox.setDecimals(2)
        self.PeakThresholdSpinBox.setRange(0.5, 20.0)
        self.PeakThresholdSpinBox.setSingleStep(0.5)
        self.PeakThresholdSpinBox.setValue(K_STAGE1_DEFAULT)  # default k = 1.5
        self.PeakThresholdSpinBox.setSuffix("  ×")
        self.PeakThresholdSpinBox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.PeakThresholdSpinBox.setToolTip(
            "Stage 1 threshold multiplier k.\n"
            "A frame is a candidate if  E_i > k × Ê_floor.\n"
            "1.5 = wide net (data collection).  2–3.5 = tighter."
        )
        threshold_layout.addWidget(self.PeakThresholdSpinBox)

        # The adaptive threshold curve — built after data is loaded (see _setup_ui_with_data).
        # Stored as a pyqtgraph PlotDataItem so it can be updated when k changes.
        self.threshold_curve = self.plot_widget_time.plot_widget.plot(
            [], [],
            pen={'color': 'r', 'width': 1.5, 'style': QtCore.Qt.DashLine},
            name='k × Ê_floor'
        )
        self.PeakThresholdSpinBox.valueChanged.connect(self._update_threshold_line)

        # Raw noise floor line — shows Ê_floor(i) directly (without k multiplier).
        self.noise_floor_curve = self.plot_widget_time.plot_widget.plot(
            [], [],
            pen={'color': '#80DEEA', 'width': 2.5},
            name='Ê_floor'
        )

        self.PeakThresholdButton = QPushButton("Apply", table_container)
        self.PeakThresholdButton.setObjectName(u"PeakThresholdButton")
        self.PeakThresholdButton.setToolTip("Show all frames above current k × Ê_floor threshold")
        self.PeakThresholdButton.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.PeakThresholdButton.clicked.connect(self._apply_threshold_filter)
        threshold_layout.addWidget(self.PeakThresholdButton)

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

    def show_fft_parameters(self):
        """Show FFT parameters for the current frame that are fed to the SVM.
        
        NOTE: SPR, R_spectral, FPE are ALWAYS computed on normalized data because
        _feat_fft_features normalizes internally — matching exactly what the SVM sees.
        The user normalization toggle does NOT affect these values by design.
        """
        if self.data_manager.total_frames == 0:
            QMessageBox.warning(self, "No Data", "No FFT data available for analysis.")
            return

        frame_index = int(round(self.current_position_ms / self.data_manager.frame_duration_ms))
        frame_index = max(0, min(frame_index, self.data_manager.total_frames - 1))

        if frame_index >= len(self.data_manager.fft_data):
            QMessageBox.warning(self, "Error", f"Invalid frame index: {frame_index}")
            return

        fs       = self.data_manager.header_info.get('fs',       V5_FS)
        fft_size = self.data_manager.header_info.get('fft_size', V5_FFT_SIZE)
        raw_mags  = np.asarray(self.data_manager.fft_data[frame_index], dtype=np.float64)
        full_freq = np.arange(fft_size // 2) * (fs / fft_size)
        full_mags = np.zeros(fft_size // 2, dtype=np.float64)
        n_bins = min(len(raw_mags), V5_BIN_END - V5_BIN_START + 1)
        full_mags[V5_BIN_START : V5_BIN_START + n_bins] = raw_mags[:n_bins]

        # Always normalize: _feat_fft_features normalizes internally, so these
        # values are invariant to the display toggle — they match what the SVM uses.
        fft_norm = _normalize_fft(full_mags, full_freq)

        try:
            fft_params = _feat_fft_features(fft_norm, full_freq)
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"Failed to compute FFT parameters:\n{str(e)}")
            import traceback; traceback.print_exc()
            return

        params_text = (
            f"Frame index: {frame_index}\n"
            f"(Always normalized — matches SVM input)\n"
            f"SPR: {fft_params['SPR']:.2f}\n"
            f"R_spectral: {fft_params['R_spectral']:.3f}\n"
            f"FPE: {fft_params['FPE_hz'] / 1000:.2f} kHz\n"
        )
        QMessageBox.information(self, "FFT Parameters (SVM)", params_text)
        

    def show_ifft_window(self):
        """
        Show the iFFT of the current frame, follows normalization decided in main window.

        Uses reconstruct_frame_v5 for the raw iFFT. The IFFTWindow then
        auto-applies normalization so the user sees the normalized signal first.
        """
        if self.data_manager.total_frames == 0:
            QMessageBox.warning(self, "No Data", "No data loaded to perform iFFT.")
            return

        file_version = self.data_manager.header_info.get('version', 0)
        has_phases   = (file_version >= 3.0 and
                        hasattr(self.data_manager, 'phase_data') and
                        len(self.data_manager.phase_data) > 0)

        if not has_phases:
            QMessageBox.warning(self, "No Phase Data",
                                "iFFT requires phase information (file version ≥ 3.0).")
            return

        fi = int(round(self.current_position_ms / self.data_manager.frame_duration_ms))
        fi = max(0, min(fi, self.data_manager.total_frames - 1))

        fs       = self.data_manager.header_info.get('fs',       V5_FS)
        fft_size = self.data_manager.header_info.get('fft_size', V5_FFT_SIZE)

        fft_mags   = np.asarray(self.data_manager.fft_data[fi],   dtype=np.float64)
        phase_int8 = np.asarray(self.data_manager.phase_data[fi], dtype=np.int8) \
                     if fi < len(self.data_manager.phase_data) else np.array([], dtype=np.int8)

        # Use reconstruct_frame_v5 for a consistent raw iFFT (Tukey + Gibbs suppression)
        #NOTE: normalize=False to get raw iFFT, IFFTWindow will handle normalization separately
        frame_data = reconstruct_frame_v5(fft_mags, phase_int8, fs, fft_size, normalize=False)
        if frame_data is None:
            QMessageBox.critical(self, "iFFT Error", "Failed to reconstruct iFFT for this frame.")
            return

        raw_signal = frame_data['signal']
        num_samples = len(raw_signal)
        frame_start_time = fi * (fft_size / fs)
        time_axis = np.linspace(frame_start_time,
                                frame_start_time + num_samples / fs,
                                num_samples)

        print(f"📊 iFFT frame {fi}: start={frame_start_time:.6f}s  "
              f"duration={num_samples/fs*1000:.2f}ms  samples={num_samples}")

        # Close previous window if open
        if hasattr(self, 'ifft_win') and self.ifft_win is not None:
            try:
                self.ifft_win.close()
                self.ifft_win.deleteLater()
            except RuntimeError:
                pass

        self.ifft_win = IFFTWindow(
            time_axis, raw_signal, parent=self,
            frame_index=fi, has_real_phases=True
        )

        # Respect main window normalization state instead of always normalizing
        should_normalize = getattr(self, '_using_normalized_means', True)
        #follow _using_normalized_means to decide whether to show normalized or raw iFFT 

        if has_phases and should_normalize:
            self.ifft_win._compute_normalized_ifft()
            if self.ifft_win.signal_data_normalized is not None:
                self.ifft_win.is_normalized = True
                self.ifft_win._update_display()
        else:
            # Open in raw mode — signal_data_raw already set, is_normalized already False
            self.ifft_win._update_display()

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
            event_amp = float(amp_item.text().replace(' mV', '')) if amp_item else 0
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
        Show all frames whose normalized FFT mean exceeds k × Ê_floor.

        k is read from PeakThresholdSpinBox. Ê_floor(i) is the per-frame
        adaptive noise floor estimate from AdaptiveNoiseEstimatorV5.
        Consecutive above-threshold frames are grouped (MAX_GAP = 5 frames).
        """
        k = self.PeakThresholdSpinBox.value()

        if (len(self.data_manager.fft_means) == 0 or
                len(self.data_manager.E_hat_floor_arr) == 0):
            self.peak_table.setRowCount(0)
            return

        threshold_arr = k * self.data_manager.E_hat_floor_arr
        peak_indices  = np.where(self.data_manager.fft_means >= threshold_arr)[0]

        if len(peak_indices) == 0:
            self.peak_table.setRowCount(0)
            return

        peaks_grouped = self._group_consecutive_peaks(peak_indices)
        self._populate_peak_table(peaks_grouped, threshold_arr)
        self.click_tab_widget.setCurrentWidget(self.peak_table)
        print(f"✅ Found {len(peaks_grouped)} candidate groups above k={k:.2f} × Ê_floor")

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

    def _populate_peak_table(self, peak_groups, threshold_arr):
        """Populate the above-threshold table. threshold_arr may be scalar or per-frame array."""
        self.peak_table.setRowCount(len(peak_groups))

        for row, group in enumerate(peak_groups):
            max_idx_in_group = np.argmax([self.data_manager.fft_means[i] for i in group])
            peak_frame = group[max_idx_in_group]

            timestamp       = self.data_manager.fft_timestamps[peak_frame]
            energy          = self.data_manager.fft_means[peak_frame]   # [V²]
            duration_frames = len(group)

            # Convert energy [V²] → RMS amplitude [mV] for display
            amp_rms_mv = float(np.sqrt(max(energy, 0.0))) * 1000.0

            if peak_frame < len(self.data_manager.fft_data):
                fft_frame = self.data_manager.fft_data[peak_frame]
                freq_axis = np.array(self.data_manager.frequency_axis)
                peak_freq_idx = int(np.argmax(np.abs(fft_frame)))
                frequency = float(freq_axis[peak_freq_idx]) if peak_freq_idx < len(freq_axis) else 0.0
            else:
                frequency = 0.0

            if hasattr(threshold_arr, '__len__') and peak_frame < len(threshold_arr):
                thr_at_frame = float(threshold_arr[peak_frame])
            else:
                thr_at_frame = float(threshold_arr)

            self.peak_table.setItem(row, 0, QTableWidgetItem(f"{timestamp:.3f}s"))
            self.peak_table.setItem(row, 1, QTableWidgetItem(f"{frequency:.0f} Hz"))
            self.peak_table.setItem(row, 2, QTableWidgetItem(f"{amp_rms_mv:.4f} mV"))   # FIX: sqrt(E)
            self.peak_table.setItem(row, 3, QTableWidgetItem(f"{duration_frames} FFT"))
            self.peak_table.setItem(row, 4, QTableWidgetItem(
                f"SNR {energy/thr_at_frame:.1f}×" if thr_at_frame > 0 else ""))

    def _update_threshold_line(self):
        """
        Rebuild the adaptive threshold curve on the time-domain plot.

        The curve shows  k × Ê_floor(i)  over all frames, where k is set by
        the SpinBox and Ê_floor(i) is the per-frame noise floor estimate from
        AdaptiveNoiseEstimatorV5 (pre-computed at load time).
        """
        k = self.PeakThresholdSpinBox.value()

        if (len(self.data_manager.E_hat_floor_arr) == 0 or
                len(self.data_manager.fft_timestamps) == 0):
            return

        threshold_y = k * self.data_manager.E_hat_floor_arr
        self.threshold_curve.setData(self.data_manager.fft_timestamps, threshold_y)

        # Update the raw noise floor line (Ê_floor without k).
        # This is the same array the threshold is derived from — showing it
        # directly lets you verify that the estimator tracks the background correctly.
        self.noise_floor_curve.setData(
            self.data_manager.fft_timestamps,
            self.data_manager.E_hat_floor_arr   # [V²], no k multiplier
        )

    
    def _compute_fft_for_display(self, frame_index: int) -> tuple:
        """
        Return (freq_axis, magnitudes) for the FFT plot, normalized or raw
        depending on _using_normalized_means.
        Always returns data in the analysis-band slice (len = freq_axis).
        """
        raw_mags  = np.asarray(self.data_manager.fft_data[frame_index], dtype=np.float64)
        freq_axis = np.array(self.data_manager.frequency_axis)

        if not getattr(self, '_using_normalized_means', True):
            # Raw — show as-is
            return freq_axis, raw_mags

        # Normalized — pad into full half-spectrum then apply mic correction
        fs       = self.data_manager.header_info.get('fs',       V5_FS)
        fft_size = self.data_manager.header_info.get('fft_size', V5_FFT_SIZE)
        full_freq = np.arange(fft_size // 2) * (fs / fft_size)
        full_mags = np.zeros(fft_size // 2, dtype=np.float64)
        n_bins = min(len(raw_mags), V5_BIN_END - V5_BIN_START + 1)
        full_mags[V5_BIN_START : V5_BIN_START + n_bins] = raw_mags[:n_bins]
        norm_mags = _normalize_fft(full_mags, full_freq)
        return freq_axis, norm_mags[V5_BIN_START : V5_BIN_START + len(freq_axis)]

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
    
    def toggle_normalized_data(self):
        """
        Toggle between normalized and raw FFT means for the main energy timeline PLUS FFT window.

        Recomputes fft_means using normalized (default) or raw magnitudes,
        rebuilds the adaptive threshold curve, and refreshes the above-threshold table.
        This affects: the time-domain energy plot, the threshold curve, and the table.
        Also affects the FFT window.
        The iFFT window has its own independent toggle.
        """
        # Determine new mode (if E_hat_floor_arr exists we're post-load)
        if self.data_manager.total_frames == 0:
            return

        # Toggle the fft_means array between normalized and raw
        if not hasattr(self, '_using_normalized_means'):
            self._using_normalized_means = True   # we start in normalized mode

        self._using_normalized_means = not self._using_normalized_means

        n = self.data_manager.total_frames
        # ── Progress dialog ───────────────────────────────────────────────────
        label = ("Recomputing normalized FFT means…"
                 if self._using_normalized_means
                 else "Switching to raw FFT means…")

        progress = QProgressDialog(label, None, 0, n, self)  # no Cancel button
        progress.setWindowTitle("Please wait")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)   # show immediately, no delay
        progress.setValue(0)
        progress.show()
        QCoreApplication.processEvents()

        def _cb(current, total):
            progress.setValue(current)
            QCoreApplication.processEvents()

        # ── Recompute ─────────────────────────────────────────────────────────
        if self._using_normalized_means:
            self.data_manager.precompute_fft_means(progress_callback=_cb)
            print("✅ Switched to normalized FFT means")
        else:
            # FIX: store energy [V²] (not mean amplitude [V]) to stay consistent
            # with E_hat_floor_arr units and Stage 1 criterion.
            raw_means = np.empty(n, dtype=np.float32)
            for i in range(n):
                raw_means[i] = float(compute_fft_energy_v5(
                    np.asarray(self.data_manager.fft_data[i], dtype=np.float64)
                ))
                if i % 50 == 0:
                    _cb(i, n)
            self.data_manager.fft_means = raw_means
            _cb(n, n)
            print("✅ Switched to raw FFT means")

        progress.close()

        # Refresh the energy plot
        stream_x, stream_y = self.data_manager.get_streaming_data()
        # Re-map streaming Y to the current fft_means values
        if len(self.data_manager.fft_timestamps) > 0:
            # Rebuild streaming Y from fft_means (which are already per-frame)
            start_frame = int(self.data_manager.streaming_start_time * 1000
                              / self.data_manager.frame_duration_ms)
            end_frame   = int(self.data_manager.streaming_end_time   * 1000
                              / self.data_manager.frame_duration_ms)
            end_frame   = min(end_frame, self.data_manager.total_frames)
            new_y = self.data_manager.fft_means[start_frame:end_frame]
            new_x = self.data_manager.fft_timestamps[start_frame:end_frame]
            self.time_curve.setData(new_x, new_y)

        # Rebuild threshold curve and table
        self._update_threshold_line()
        self._apply_threshold_filter()

        # Update FFT window
        self.update_display()
    
    def _on_open_data_collection_dialog(self):
        """Open data collection dialog for Phase 2."""
        try:
            from components.data_collection_dialog_v5 import DataCollectionDialogV5
            dialog = DataCollectionDialogV5(parent=self)
            dialog.exec()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open data collection dialog: {e}")