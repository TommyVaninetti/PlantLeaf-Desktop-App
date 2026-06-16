import sys
import os
sys.path.insert(0, "/Users/fridatirari/PlantLeaf development /PlantLeaf-Desktop-App/src/chemical_simulators")
sys.path.insert(0, "/Users/fridatirari/PlantLeaf development /PlantLeaf-Desktop-App/src")

import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QDoubleSpinBox, QSlider, QGroupBox,
    QFileDialog, QMessageBox, QProgressDialog, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy, QFrame,
    QApplication
)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QIcon, QAction, QFont

from core.settings_manager import SettingsManager
from core.font_manager import FontManager
from core.layout_manager import LayoutManager
from core.theme_manager import ThemeManager
from config.app_config import AppConfig
from plotting.plot_manager import BasePlotWidget

SLIDER_CSS = (
    "QSlider::groove:horizontal { background: #a5d6a7; height: 6px; border-radius: 3px; }"
    "QSlider::handle:horizontal { background: #5a7559; border: 2px solid #5a7559;"
    " width: 16px; height: 16px; margin: -5px 0; border-radius: 8px; }"
    "QSlider::sub-page:horizontal { background: #689f67; border-radius: 3px; }"
    "QSlider::add-page:horizontal { background: #c8e6c9; border-radius: 3px; }"
)


class SimulationWorker(QObject):
    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(int)

    def __init__(self, R0, P_inf, distance_m, tau_target_ms=None):
        super().__init__()
        self.R0 = R0
        self.P_inf = P_inf
        self.distance_m = distance_m
        self.tau_target_ms = tau_target_ms

    def run(self):
        try:
            import sys
            sys.path.insert(0, "/Users/fridatirari/PlantLeaf development /PlantLeaf-Desktop-App/src/chemical_simulators")
            from run_acoustic_simulation import run_simulation
            self.progress.emit(30)
            result = run_simulation(
                R0=self.R0,
                P_inf=self.P_inf,
                distance_m=self.distance_m,
                tau_target_ms=self.tau_target_ms
            )
            self.progress.emit(100)
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")


class MainWindowChemicalSimulator(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings_manager = SettingsManager()
        self.font_manager = FontManager(self.settings_manager.settings)
        self.layout_manager = LayoutManager(self.font_manager)
        self.theme_manager = ThemeManager(self.settings_manager.settings, self.font_manager)

        self.setWindowTitle("Audio Chemical Simulator")
        self.setWindowIcon(QIcon(AppConfig.LOGO_DIR))
        self.setMinimumSize(1100, 650)

        self.sim_result = None
        self.real_clicks = []
        self.sim_thread = None
        self.sim_worker = None
        self.paudio_data = None

        self._setup_ui()
        self._setup_menubar()
        self._setup_toolbar()
        self._load_saved_settings()
        self.setStatusBar(None)
        self._apply_plot_themes()
        self.r0_slider.setStyleSheet(SLIDER_CSS)
        self.pinf_slider.setStyleSheet(SLIDER_CSS)
        self.dist_slider.setStyleSheet(SLIDER_CSS)
        self.showMaximized()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        splitter.addWidget(self._build_controls_panel())
        splitter.addWidget(self._build_plots_panel())
        splitter.addWidget(self._build_results_panel())
        splitter.setSizes([280, 620, 260])

    def _build_controls_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        layout.setContentsMargins(4, 4, 4, 4)

        title = QLabel("Simulation Parameters")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2d4a2b; padding: 8px;")
        layout.addWidget(title)

        phys_group = QGroupBox("Physical Parameters")
        phys_layout = QVBoxLayout(phys_group)
        phys_layout.setSpacing(8)

        phys_layout.addWidget(QLabel("Bubble radius R0 [µm]:"))
        self.r0_slider = QSlider(Qt.Horizontal)
        self.r0_slider.setRange(20, 100)
        self.r0_slider.setValue(50)
        self.r0_spinbox = QDoubleSpinBox()
        self.r0_spinbox.setRange(20.0, 100.0)
        self.r0_spinbox.setValue(50.0)
        self.r0_spinbox.setSuffix(" µm")
        self.r0_spinbox.setDecimals(1)
        self.r0_spinbox.setSingleStep(1.0)
        r0_row = QHBoxLayout()
        r0_row.addWidget(self.r0_slider)
        r0_row.addWidget(self.r0_spinbox)
        phys_layout.addLayout(r0_row)
        self.r0_slider.valueChanged.connect(lambda v: self.r0_spinbox.setValue(float(v)))
        self.r0_spinbox.valueChanged.connect(lambda v: self.r0_slider.setValue(int(v)))

        phys_layout.addWidget(QLabel("Xylem pressure P∞ [MPa]:"))
        self.pinf_slider = QSlider(Qt.Horizontal)
        self.pinf_slider.setRange(-150, -30)
        self.pinf_slider.setValue(-30)
        self.pinf_spinbox = QDoubleSpinBox()
        self.pinf_spinbox.setRange(-1.5, -0.3)
        self.pinf_spinbox.setValue(-0.3)
        self.pinf_spinbox.setSuffix(" MPa")
        self.pinf_spinbox.setDecimals(2)
        self.pinf_spinbox.setSingleStep(0.05)
        pinf_row = QHBoxLayout()
        pinf_row.addWidget(self.pinf_slider)
        pinf_row.addWidget(self.pinf_spinbox)
        phys_layout.addLayout(pinf_row)
        self.pinf_slider.valueChanged.connect(lambda v: self.pinf_spinbox.setValue(v / 100.0))
        self.pinf_spinbox.valueChanged.connect(lambda v: self.pinf_slider.setValue(int(v * 100)))

        phys_layout.addWidget(QLabel("Distance bubble → mic [cm]:"))
        self.dist_slider = QSlider(Qt.Horizontal)
        self.dist_slider.setRange(5, 50)
        self.dist_slider.setValue(10)
        self.dist_spinbox = QDoubleSpinBox()
        self.dist_spinbox.setRange(0.5, 5.0)
        self.dist_spinbox.setValue(1.0)
        self.dist_spinbox.setSuffix(" cm")
        self.dist_spinbox.setDecimals(1)
        self.dist_spinbox.setSingleStep(0.1)
        dist_row = QHBoxLayout()
        dist_row.addWidget(self.dist_slider)
        dist_row.addWidget(self.dist_spinbox)
        phys_layout.addLayout(dist_row)
        self.dist_slider.valueChanged.connect(lambda v: self.dist_spinbox.setValue(v / 10.0))
        self.dist_spinbox.valueChanged.connect(lambda v: self.dist_slider.setValue(int(v * 10)))

        layout.addWidget(phys_group)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)

        self.btn_load = QPushButton("📂  Load .paudio File")
        self.btn_load.setMinimumHeight(42)
        self.btn_load.setObjectName("mainButton")
        self.btn_load.clicked.connect(self._load_paudio)
        layout.addWidget(self.btn_load)

        self.click_selector_label = QLabel("Select click to analyze:")
        layout.addWidget(self.click_selector_label)

        self.click_table = QTableWidget(0, 4)
        self.click_table.setHorizontalHeaderLabels(["Time (s)", "τ (ms)", "Peak (µV)", "R²"])
        self.click_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.click_table.verticalHeader().setVisible(False)
        self.click_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.click_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.click_table.setMaximumHeight(200)
        self.click_table.itemSelectionChanged.connect(self._on_click_selected)
        layout.addWidget(self.click_table)

        self.btn_run = QPushButton("▶  Run Simulation")
        self.btn_run.setMinimumHeight(42)
        self.btn_run.setObjectName("mainButton")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._run_simulation)
        layout.addWidget(self.btn_run)

        self.btn_pdf = QPushButton("📄  Generate PDF Report")
        self.btn_pdf.setMinimumHeight(38)
        self.btn_pdf.setObjectName("mainButton")
        self.btn_pdf.setEnabled(False)
        self.btn_pdf.clicked.connect(self._generate_report)
        layout.addWidget(self.btn_pdf)

        self.file_label = QLabel("No .paudio file loaded")
        self.file_label.setAlignment(Qt.AlignCenter)
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self.file_label)

        layout.addStretch(1)
        return panel

    def _build_plots_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        tab = QTabWidget()
        tab.setStyleSheet("QTabBar::tab { font-size: 13px; font-weight: 600; padding: 10px 20px; }")
        layout.addWidget(tab)

        time_widget = QWidget()
        time_layout = QVBoxLayout(time_widget)
        self.time_info_label = QLabel("Time Domain — Real click vs Simulated")
        self.time_info_label.setAlignment(Qt.AlignCenter)
        time_layout.addWidget(self.time_info_label)
        self.plot_time = BasePlotWidget(
            x_label="Time", y_label="Amplitude",
            x_range=(0, 1e-4), y_range=(-1, 1),
            x_min=0, x_max=1e-3, y_min=-10, y_max=10,
            unit_x="s", unit_y="", parent=self
        )
        self.curve_sim_time = self.plot_time.plot_widget.plot(name="Simulated", pen={'color': '#689f67', 'width': 2})
        self.curve_real_time = self.plot_time.plot_widget.plot(
            name="Real click", pen={'color': 'r', 'width': 1.5}
        )
        self.plot_time.plot_widget.showGrid(x=True, y=True)
        time_layout.addWidget(self.plot_time)
        tab.addTab(time_widget, "Time Domain")

        freq_widget = QWidget()
        freq_layout = QVBoxLayout(freq_widget)
        self.freq_info_label = QLabel("Frequency Domain 20–80 kHz")
        self.freq_info_label.setAlignment(Qt.AlignCenter)
        freq_layout.addWidget(self.freq_info_label)
        self.plot_freq = BasePlotWidget(
            x_label="Frequency", y_label="Amplitude",
            x_range=(20000, 80000), y_range=(0, 1),
            x_min=19000, x_max=81000, y_min=0, y_max=10,
            unit_x="Hz", unit_y="", parent=self
        )
        self.curve_sim_freq = self.plot_freq.plot_widget.plot(name="Simulated", pen={'color': '#689f67', 'width': 2})
        self.curve_real_freq = self.plot_freq.plot_widget.plot(
            name="Real", pen={'color': 'r', 'width': 1.5}
        )
        self.plot_freq.plot_widget.showGrid(x=True, y=True)
        freq_layout.addWidget(self.plot_freq)
        tab.addTab(freq_widget, "Frequency Domain")

        bubble_widget = QWidget()
        bubble_layout = QVBoxLayout(bubble_widget)
        bubble_layout.addWidget(QLabel("Bubble radius R(t) during collapse"))
        self.plot_bubble = BasePlotWidget(
            x_label="Time", y_label="Radius",
            x_range=(0, 1e-4), y_range=(0, 100),
            x_min=0, x_max=1e-3, y_min=0, y_max=200,
            unit_x="s", unit_y="µm", parent=self
        )
        self.curve_bubble = self.plot_bubble.plot_widget.plot(
            name="R(t)", pen={'color': '#2196F3', 'width': 2}
        )
        self.plot_bubble.plot_widget.showGrid(x=True, y=True)
        bubble_layout.addWidget(self.plot_bubble)
        tab.addTab(bubble_widget, "Bubble Dynamics")

        return panel

    def _build_results_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)
        layout.setContentsMargins(4, 4, 4, 4)

        title = QLabel("Diagnostics")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2d4a2b; padding: 8px;")
        layout.addWidget(title)

        sim_group = QGroupBox("Simulated Parameters")
        sim_layout = QVBoxLayout(sim_group)
        self.table_sim = QTableWidget(0, 2)
        self.table_sim.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.table_sim.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_sim.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table_sim.verticalHeader().setVisible(False)
        self.table_sim.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_sim.setAlternatingRowColors(True)
        sim_layout.addWidget(self.table_sim)
        layout.addWidget(sim_group)

        compare_group = QGroupBox("Comparison Real vs Simulated")
        compare_layout = QVBoxLayout(compare_group)
        self.table_compare = QTableWidget(0, 2)
        self.table_compare.setHorizontalHeaderLabels(["Metric", "Value"])
        self.table_compare.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_compare.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table_compare.verticalHeader().setVisible(False)
        self.table_compare.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_compare.setAlternatingRowColors(True)
        compare_layout.addWidget(self.table_compare)
        layout.addWidget(compare_group)

        self.correlation_label = QLabel("Correlation: —")
        self.correlation_label.setAlignment(Qt.AlignCenter)
        self.correlation_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 6px;")
        layout.addWidget(self.correlation_label)

        layout.addStretch(1)
        return panel

    def _setup_menubar(self):
        menubar = self.menuBar()
        font = QFont()
        font.setPointSize(12)
        menubar.setFont(font)

        file_menu = menubar.addMenu("File")
        self.actionHome = QAction("Home", self)
        self.actionHome.triggered.connect(self._go_home)
        file_menu.addAction(self.actionHome)
        file_menu.addSeparator()
        action_load = QAction("Load .paudio File...", self)
        action_load.setShortcut("Ctrl+O")
        action_load.triggered.connect(self._load_paudio)
        file_menu.addAction(action_load)

        sim_menu = menubar.addMenu("Simulation")
        action_run = QAction("Run Simulation", self)
        action_run.setShortcut("Ctrl+R")
        action_run.triggered.connect(self._run_simulation)
        sim_menu.addAction(action_run)

        settings_menu = menubar.addMenu("Settings")
        theme_menu = settings_menu.addMenu("Theme")
        for name, f in [("Dark","dark.css"),("Dark Green","dark_green.css"),
                        ("Light","light.css"),("Light Green","light_green.css"),
                        ("Light Blue","light_blue.css")]:
            a = QAction(name, self)
            a.triggered.connect(lambda checked, fn=f: self._update_style(theme_name=fn))
            theme_menu.addAction(a)

        about_menu = menubar.addMenu("About")
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        about_menu.addAction(quit_action)

    def _setup_toolbar(self):
        from PySide6.QtCore import QSize
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(28, 28))
        home_action = QAction("🏠 Home", self)
        home_action.triggered.connect(self._go_home)
        toolbar.addAction(home_action)
        toolbar.addSeparator()
        run_action = QAction("▶ Run", self)
        run_action.triggered.connect(self._run_simulation)
        toolbar.addAction(run_action)
        load_action = QAction("📂 Load .paudio", self)
        load_action.triggered.connect(self._load_paudio)
        toolbar.addAction(load_action)

    def _load_paudio(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open .paudio File", "",
            "PlantLeaf Audio (*.paudio);;All Files (*)"
        )
        if not file_path:
            return

        try:
            import struct, zlib, json as _json

            with open(file_path, 'rb') as f:
                header_bytes = f.read(128)
                remaining_data = f.read()

            magic = header_bytes[:10].rstrip(b'\x00')
            if magic != b'PLANTAUDIO':
                QMessageBox.warning(self, "Error", "Not a valid .paudio file.")
                return

            version = struct.unpack('<f', header_bytes[10:14])[0]
            fs       = struct.unpack('<I', header_bytes[34:38])[0]
            fft_size = struct.unpack('<I', header_bytes[38:42])[0]
            freq_min = struct.unpack('<I', header_bytes[42:46])[0]
            freq_max = struct.unpack('<I', header_bytes[46:50])[0]

            bin_freq  = fs / fft_size
            bin_start = int(freq_min / bin_freq)
            bin_end   = int(freq_max / bin_freq)
            num_bins  = bin_end - bin_start + 1
            frame_duration_ms = (fft_size / fs) * 1000.0

            click_start_pos = remaining_data.find(b'CLCK')
            if click_start_pos >= 0:
                fft_bytes = remaining_data[:click_start_pos]
                click_section = remaining_data[click_start_pos:]
            else:
                fft_bytes = remaining_data
                click_section = None

            fft_data = []
            phase_data = []
            bytes_per_sample = 5 if version >= 3.0 else 4
            offset = 0
            while offset + num_bins * bytes_per_sample <= len(fft_bytes):
                frame_mags = []
                frame_phases = []
                for b in range(num_bins):
                    pos = offset + b * bytes_per_sample
                    mag = struct.unpack('<f', fft_bytes[pos:pos+4])[0]
                    frame_mags.append(mag)
                    if version >= 3.0:
                        ph = struct.unpack('<b', fft_bytes[pos+4:pos+5])[0]
                        frame_phases.append(ph)
                fft_data.append(np.array(frame_mags, dtype=np.float32))
                if version >= 3.0:
                    phase_data.append(np.array(frame_phases, dtype=np.int8))
                offset += num_bins * bytes_per_sample

            freq_axis = np.linspace(freq_min, freq_max, num_bins)

            self.paudio_data = {
                'fft_data':   fft_data,
                'phase_data': phase_data,
                'freq_axis':  freq_axis,
                'fs':         fs,
                'fft_size':   fft_size,
                'freq_min':   freq_min,
                'freq_max':   freq_max,
                'bin_start':  bin_start,
                'num_bins':   num_bins,
                'version':    version,
                'frame_duration_ms': frame_duration_ms,
            }

            clicks_raw = []
            if click_section and len(click_section) >= 8:
                marker = click_section[0:4]
                if marker == b'CLCK':
                    click_length = struct.unpack('<I', click_section[4:8])[0]
                    if len(click_section) >= 8 + click_length:
                        compressed = click_section[8:8+click_length]
                        try:
                            clicks_raw = _json.loads(zlib.decompress(compressed).decode('utf-8'))
                        except:
                            clicks_raw = []

            if not clicks_raw:
                self.file_label.setText("🔍 Running click detector...")
                QApplication.processEvents()
                clicks_raw = self._run_click_detector(
                    fft_data, phase_data, freq_axis, fs, fft_size, frame_duration_ms
                )

            if not clicks_raw:
                QMessageBox.warning(self, "No Clicks",
                    "No ultrasonic clicks found in this file.")
                self.file_label.setText("No clicks found")
                return

            self.real_clicks = []
            for click in clicks_raw:
                ts_str = str(click.get('timestamp', '0'))
                try:
                    ts = float(ts_str.replace('s', '').strip())
                except:
                    ts = 0.0

                tau_ms = float(click.get('tau_ms', -1.0))
                if tau_ms <= 0:
                    duration_str = str(click.get('duration', ''))
                    if 'FFT' in duration_str:
                        try:
                            fft_count = int(duration_str.replace(' FFT', '').strip())
                            tau_ms = fft_count * 2.56 / 3.0
                        except:
                            pass

                frame_idx = int(round(ts * 1000.0 / frame_duration_ms))
                frame_idx = max(0, min(frame_idx, len(fft_data) - 1))

                self.real_clicks.append({
                    'timestamp': ts,
                    'frame_idx': frame_idx,
                    'tau_ms':    tau_ms,
                    'frequency': click.get('frequency', ''),
                    'amplitude': click.get('amplitude', ''),
                    'r2':        float(click.get('r2_log', click.get('r2', 0.0))),
                    'peak_amp':  float(click.get('peak_amp', 0.0)),
                })

            self._populate_click_table()
            self.file_label.setText(
                f"{os.path.basename(file_path)}\n{len(self.real_clicks)} clicks found"
            )
            if self.real_clicks:
                self.btn_run.setEnabled(True)

        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Load Error", f"Could not load file:\n{str(e)}\n{traceback.format_exc()}")

    def _run_click_detector(self, fft_data, phase_data, freq_axis, fs, fft_size, frame_duration_ms):
        from windows.replay_window_audio import (
            compute_hilbert_envelope, find_peak, check_decay, suppress_edge_artifacts
        )

        total_frames = len(fft_data)
        if total_frames == 0:
            return []

        fft_means = np.array([np.mean(np.abs(f)) for f in fft_data])
        fft_mean  = np.mean(fft_means)
        fft_std   = np.std(fft_means)

        datasheet_freq_hz = np.array([20, 25, 30, 40, 50, 60, 70, 80]) * 1000
        datasheet_resp_db = np.array([8.0, 10.5, 6.0, -2.0, -6.0, -7.0, -6.0, -4.0])
        valid_mask = (freq_axis >= 20000) & (freq_axis <= 80000)
        mic_db = np.interp(freq_axis[valid_mask], datasheet_freq_hz, datasheet_resp_db)
        gain_50 = 10 ** (-mic_db * 0.5 / 20.0)

        def normalize_fft(mags):
            n = mags.copy()
            n[valid_mask] *= gain_50
            return n

        def reconstruct_ifft(frame_idx):
            if not phase_data or frame_idx >= len(phase_data):
                return None
            mags = normalize_fft(fft_data[frame_idx].copy())
            phases_int8 = phase_data[frame_idx]
            num_bins_full = fft_size // 2
            bin_freq = fs / fft_size
            bin_s = int(20000 / bin_freq)
            bin_e = int(80000 / bin_freq)
            actual = min(len(mags), bin_e - bin_s + 1, len(phases_int8))
            full_mag = np.zeros(num_bins_full, dtype=np.float32)
            full_phase = np.zeros(num_bins_full, dtype=np.int8)
            full_mag[bin_s:bin_s+actual] = mags[:actual]
            full_phase[bin_s:bin_s+actual] = phases_int8[:actual]
            phases_rad = (full_phase / 127.0) * np.pi
            cs = full_mag * np.exp(1j * phases_rad)
            taper = max(5, actual // 10)
            window = np.ones(num_bins_full)
            for i in range(taper):
                alpha = i / taper
                window[bin_s + i] = 0.5 * (1 - np.cos(np.pi * alpha))
                window[bin_s + actual - i - 1] = 0.5 * (1 - np.cos(np.pi * alpha))
            cs *= window
            try:
                sig = np.fft.irfft(cs, n=fft_size)
                return suppress_edge_artifacts(sig)
            except:
                return None

        threshold_noise = fft_mean + 4 * fft_std
        empty_indices = np.where(fft_means < threshold_noise)[0]
        noise_rms = fft_mean
        if len(empty_indices) > 0:
            np.random.seed(42)
            sampled = np.random.choice(empty_indices, size=min(200, len(empty_indices)), replace=False)
            rms_vals = []
            for idx in sampled:
                sig = reconstruct_ifft(idx)
                if sig is not None:
                    rms_vals.append(np.sqrt(np.mean(sig**2)))
            if rms_vals:
                noise_rms = float(np.mean(rms_vals))

        threshold_v = fft_mean + 5 * fft_std
        above = [i for i in range(total_frames) if fft_means[i] > threshold_v]
        MAX_RUN = 4
        filtered = []
        if above:
            run_start = 0
            for k in range(1, len(above) + 1):
                at_end = (k == len(above))
                new_run = at_end or (above[k] - above[k-1] > 1)
                if new_run:
                    run = above[run_start:k]
                    if len(run) <= MAX_RUN:
                        filtered.extend(run)
                    run_start = k

        candidates2 = []
        for fi in filtered:
            fft_norm = normalize_fft(fft_data[fi].copy())
            peak_v = float(np.max(fft_norm))
            if peak_v <= 0.00085:
                continue
            power = fft_norm.astype(np.float64) ** 2
            mean_p = float(np.mean(power))
            max_p  = float(np.max(power))
            spr = max_p / mean_p if mean_p > 1e-20 else 0.0
            if spr > 20:
                continue
            candidates2.append({'frame_idx': fi, 'peak_fft_v': peak_v, 'spr': spr})

        candidates3 = []
        for c in candidates2:
            fi = c['frame_idx']
            sig = reconstruct_ifft(fi)
            if sig is None:
                continue
            env = compute_hilbert_envelope(sig)
            peak_idx, peak_amp = find_peak(env)
            if peak_amp <= 130e-6:
                continue
            next_env = None
            if peak_idx > 212 and fi + 1 < total_frames:
                ns = reconstruct_ifft(fi + 1)
                if ns is not None:
                    next_env = compute_hilbert_envelope(ns)
            decay = check_decay(env, peak_idx, next_frame_envelope=next_env, noise_rms=noise_rms)
            tau_ms = decay['tau_ms']
            r2 = decay['r_squared_log']
            E_W1 = decay['E_W1']
            E_W4 = decay['E_W4']
            if tau_ms <= 0 or not (0.045 <= tau_ms <= 1.3):
                continue
            if r2 <= 0.45:
                continue
            if E_W1 <= E_W4 * 2.0:
                continue
            GUARD = 20
            pre_end = max(0, peak_idx - GUARD)
            pre_w = sig[:pre_end] if pre_end >= 50 else np.array([noise_rms])
            rms_pre = float(np.sqrt(np.mean(pre_w**2)))
            pre_snr = rms_pre / noise_rms if noise_rms > 0 else 1.0
            if pre_snr >= 1.8:
                continue
            level = peak_amp * 0.1
            rise_start = peak_idx
            for i in range(peak_idx - 1, -1, -1):
                if env[i] < level:
                    rise_start = i + 1
                    break
            rise_s = max(1, peak_idx - rise_start)
            fall_end = min(peak_idx + 40, len(env))
            fall_s = 40
            for i in range(peak_idx + 1, fall_end):
                if env[i] < level:
                    fall_s = i - peak_idx
                    break
            asym = rise_s / fall_s if fall_s > 0 else 1.0
            if asym >= 2.5:
                continue
            candidates3.append({**c, 'peak_amp': peak_amp, 'tau_ms': tau_ms, 'r2_log': r2})

        if not candidates3:
            return []

        sorted_c = sorted(candidates3, key=lambda x: x['frame_idx'])
        groups, current = [], [sorted_c[0]]
        for i in range(1, len(sorted_c)):
            if sorted_c[i]['frame_idx'] - sorted_c[i-1]['frame_idx'] <= 3:
                current.append(sorted_c[i])
            else:
                groups.append(current)
                current = [sorted_c[i]]
        groups.append(current)

        result = []
        for grp in groups:
            best = max(grp, key=lambda x: x['peak_amp'])
            ts = best['frame_idx'] * frame_duration_ms / 1000.0
            result.append({
                'timestamp': ts,
                'tau_ms':    best['tau_ms'],
                'peak_amp':  best['peak_amp'],
                'r2_log':    best['r2_log'],
                'frequency': '',
                'amplitude': str(best['peak_fft_v']),
            })

        print(f"✅ Click detector: found {len(result)} clicks")
        return result

    def _populate_click_table(self):
        self.click_table.setRowCount(len(self.real_clicks))
        for i, click in enumerate(self.real_clicks):
            self.click_table.setItem(i, 0, QTableWidgetItem(f"{click['timestamp']:.3f}"))
            tau_str = f"{click['tau_ms']:.3f}" if click['tau_ms'] > 0 else "N/A"
            self.click_table.setItem(i, 1, QTableWidgetItem(tau_str))
            peak_uv = click.get('peak_amp', 0.0) * 1e6
            self.click_table.setItem(i, 2, QTableWidgetItem(f"{peak_uv:.1f}"))
            r2_val = click.get('r2', click.get('r2_log', 0.0))
            self.click_table.setItem(i, 3, QTableWidgetItem(f"{r2_val:.3f}"))

    def _on_click_selected(self):
        rows = self.click_table.selectedItems()
        if not rows:
            return
        row = self.click_table.currentRow()
        if row < 0 or row >= len(self.real_clicks):
            return
        click = self.real_clicks[row]
        self._show_real_click(click)

    def _show_real_click(self, click):
        if not self.paudio_data:
            return
        frame_idx = click['frame_idx']
        fft_data   = self.paudio_data['fft_data']
        phase_data = self.paudio_data['phase_data']
        freq_axis  = self.paudio_data['freq_axis']

        if frame_idx >= len(fft_data):
            return

        self.curve_real_freq.setData(freq_axis, fft_data[frame_idx])

        if phase_data and frame_idx < len(phase_data):
            signal = self._reconstruct_ifft(frame_idx)
            if signal is not None:
                fs = self.paudio_data['fs']
                fft_size = self.paudio_data['fft_size']
                t = np.linspace(0, fft_size/fs, fft_size)
                self.curve_real_time.setData(t, signal)

    def _reconstruct_ifft(self, frame_idx):
        pd = self.paudio_data
        fft_mags = pd['fft_data'][frame_idx].copy()
        if not pd['phase_data'] or frame_idx >= len(pd['phase_data']):
            return None
        phases_int8 = pd['phase_data'][frame_idx]
        fs       = pd['fs']
        fft_size = pd['fft_size']
        num_bins_full = fft_size // 2
        bin_start = pd['bin_start']
        num_bins  = pd['num_bins']
        full_mag   = np.zeros(num_bins_full, dtype=np.float32)
        full_phase = np.zeros(num_bins_full, dtype=np.int8)
        actual = min(len(fft_mags), num_bins, len(phases_int8))
        full_mag[bin_start:bin_start+actual]   = fft_mags[:actual]
        full_phase[bin_start:bin_start+actual] = phases_int8[:actual]
        phases_rad = (full_phase / 127.0) * np.pi
        complex_spectrum = full_mag * np.exp(1j * phases_rad)
        taper = max(5, actual // 10)
        window = np.ones(num_bins_full)
        for i in range(taper):
            alpha = i / taper
            window[bin_start + i] = 0.5 * (1 - np.cos(np.pi * alpha))
            window[bin_start + actual - i - 1] = 0.5 * (1 - np.cos(np.pi * alpha))
        complex_spectrum *= window
        try:
            sig = np.fft.irfft(complex_spectrum, n=fft_size)
            return sig
        except Exception:
            return None

    def _run_simulation(self):
        R0 = self.r0_spinbox.value() * 1e-6
        P_inf = self.pinf_spinbox.value() * 1e6
        distance_m = self.dist_spinbox.value() * 0.01

        rows_sel = self.click_table.selectedItems()
        tau_target = None
        if rows_sel:
            row = self.click_table.currentRow()
            if row >= 0 and row < len(self.real_clicks):
                tau_target = self.real_clicks[row].get('tau_ms', None)

        self.progress_dialog = QProgressDialog("Running simulation...", None, 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(10)
        self.progress_dialog.show()

        self.sim_thread = QThread(self)
        self.sim_worker = SimulationWorker(R0, P_inf, distance_m, tau_target)
        self.sim_worker.moveToThread(self.sim_thread)
        self.sim_thread.started.connect(self.sim_worker.run)
        self.sim_worker.finished.connect(self._on_simulation_finished)
        self.sim_worker.error.connect(self._on_simulation_error)
        self.sim_worker.progress.connect(self.progress_dialog.setValue)
        self.sim_worker.finished.connect(self.sim_thread.quit)
        self.sim_thread.finished.connect(self.sim_thread.deleteLater)
        self.sim_thread.start()

    def _on_simulation_finished(self, result):
        self.progress_dialog.close()
        self.sim_result = result
        self._update_plots(result)
        self._update_diagnostics(result)
        self.btn_pdf.setEnabled(True)
        print("Simulation completed")

    def _on_simulation_error(self, error_msg):
        self.progress_dialog.close()
        QMessageBox.critical(self, "Simulation Error", f"An error occurred:\n{error_msg}")

    def _update_plots(self, result):
        bubble      = result['bubble']
        propagation = result['propagation']
        plantleaf   = result['plantleaf']

        t      = bubble['t']
        signal = propagation['signal']
        self.curve_sim_time.setData(t, signal)

        freq = plantleaf['freq']
        spec = plantleaf['spectrum']
        self.curve_sim_freq.setData(freq, spec)

        R_um = bubble['R'] * 1e6
        t_us = t * 1e6
        self.curve_bubble.setData(t_us, R_um)

    def _update_diagnostics(self, result):
        diag   = result['diagnostics']
        bubble = result['bubble']

        rows = [
            ("R₀",         f"{bubble['R0']*1e6:.1f} µm"),
            ("P∞",         f"{bubble['P_inf']/1e6:.2f} MPa"),
            ("Collapsed",  "Yes" if bubble['collapsed'] else "No"),
            ("τ simulated", f"{diag['tau']*1000:.3f} ms" if diag['tau'] else "N/A"),
            ("SPR",        f"{diag['SPR']:.2f}" if diag['SPR'] else "N/A"),
            ("Asymmetry",  f"{diag['asymmetry']:.3f}" if diag['asymmetry'] else "N/A"),
            ("R spectral", f"{diag['R_spectral']:.3f}" if diag['R_spectral'] else "N/A"),
        ]
        self.table_sim.setRowCount(len(rows))
        for i, (p, v) in enumerate(rows):
            self.table_sim.setItem(i, 0, QTableWidgetItem(p))
            self.table_sim.setItem(i, 1, QTableWidgetItem(v))

        rows_sel = self.click_table.selectedItems()
        if rows_sel and diag['tau']:
            row = self.click_table.currentRow()
            click = self.real_clicks[row]
            tau_real = click.get('tau_ms', -1.0)
            tau_sim  = diag['tau'] * 1000.0

            real_signal = None
            if self.paudio_data:
                real_signal = self._reconstruct_ifft(click['frame_idx'])

            sim_signal = result['propagation']['signal']
            corr = 0.0
            if real_signal is not None and len(real_signal) > 0 and len(sim_signal) > 0:
                n = min(len(real_signal), len(sim_signal))
                r_norm = real_signal[:n] / (np.max(np.abs(real_signal[:n])) + 1e-30)
                s_norm = sim_signal[:n]  / (np.max(np.abs(sim_signal[:n]))  + 1e-30)
                corr = float(np.corrcoef(r_norm, s_norm)[0, 1])

            self.correlation_label.setText(f"Correlation: {corr:.4f}")

            compare_rows = [
                ("τ real (ms)",  f"{tau_real:.3f}" if tau_real > 0 else "N/A"),
                ("τ sim (ms)",   f"{tau_sim:.3f}"),
                ("Correlation",  f"{corr:.4f}"),
                ("Match τ",      "Yes" if tau_real > 0 and abs(tau_sim - tau_real) / tau_real < 0.2 else "No"),
            ]
            self.table_compare.setRowCount(len(compare_rows))
            for i, (m, v) in enumerate(compare_rows):
                self.table_compare.setItem(i, 0, QTableWidgetItem(m))
                self.table_compare.setItem(i, 1, QTableWidgetItem(v))

    def _generate_report(self):
        if not self.sim_result:
            QMessageBox.warning(self, "No Data", "Run a simulation first.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF Report", "report_acoustic.pdf", "PDF Files (*.pdf)"
        )
        if not file_path:
            return
        try:
            from report_acoustic import generate_report
            generate_report(simulation_result=self.sim_result, output_path=file_path)
            QMessageBox.information(self, "Done", f"PDF saved to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not generate report:\n{str(e)}")

    def _apply_plot_themes(self):
        bg = '#fafcfa'
        fg = '#2d4a2b'
        for plot in [self.plot_time, self.plot_freq, self.plot_bubble]:
            plot.plot_widget.setBackground(bg)
            plot.plot_widget.getAxis("bottom").setTextPen(fg)
            plot.plot_widget.getAxis("left").setTextPen(fg)

    def _load_saved_settings(self):
        saved_font_scale = self.font_manager.load_font_scale()
        self.font_manager.current_font_scale = saved_font_scale
        saved_theme = self.theme_manager.load_saved_theme()
        self.theme_manager.apply_theme(self, saved_theme)

    def _update_style(self, theme_name=None, font_scale=None):
        if font_scale is not None:
            self.font_manager.save_font_scale(font_scale)
            self.font_manager.current_font_scale = font_scale
        if theme_name is not None:
            self.theme_manager.apply_theme(self, theme_name)
        else:
            self.theme_manager.apply_theme(self, self.theme_manager.current_theme)
        self.setStatusBar(None)
        self._apply_plot_themes()
        self.r0_slider.setStyleSheet(SLIDER_CSS)
        self.pinf_slider.setStyleSheet(SLIDER_CSS)
        self.dist_slider.setStyleSheet(SLIDER_CSS)

    def _go_home(self):
        from windows.main_window_home import MainWindowHome
        home = MainWindowHome()
        self.layout_manager.center_window_on_screen(home)
        home.show()
        self.close()

    def _save_current_settings(self):
        current_theme = getattr(self.theme_manager, 'current_theme', None)
        if current_theme:
            self.theme_manager.save_theme(current_theme)
        font_scale = getattr(self.font_manager, 'current_font_scale', None)
        if font_scale:
            self.font_manager.save_font_scale(font_scale)
        self.settings_manager.save_window_geometry(self)

    def closeEvent(self, event):
        self._save_current_settings()
        event.accept()