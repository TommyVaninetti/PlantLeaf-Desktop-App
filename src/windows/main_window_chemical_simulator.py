import sys
import os

# Add src/chemical_simulators/ to the path so modules there can import each
# other with bare names (e.g. "from acoustic_parameters import ...").
# src/ itself is already on the path (inserted by main.py at startup).
_SIM_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chemical_simulators')
)
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)

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


class _PaudioDM:
    """Minimal adapter so run_stage1_v5 can consume raw paudio arrays."""
    def __init__(self, fft_data, phase_data, fs, fft_size):
        self.fft_data     = fft_data
        self.phase_data   = phase_data
        self.header_info  = {'fs': fs, 'fft_size': fft_size}
        self.total_frames = len(fft_data)


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
            from chemical_simulators.run_acoustic_simulation import run_simulation
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


class ClickDetectorWorker(QObject):
    finished = Signal(list)   # emits the raw click list on success
    error    = Signal(str)

    def __init__(self, fft_data, phase_data, fs, fft_size, frame_duration_ms):
        super().__init__()
        self.fft_data          = fft_data
        self.phase_data        = phase_data
        self.fs                = fs
        self.fft_size          = fft_size
        self.frame_duration_ms = frame_duration_ms

    def run(self):
        try:
            from pathlib import Path
            from core.click_pipeline_v5 import (
                run_stage1_v5, run_stage2_v5, run_stage3_v5, run_stage4_v5,
                reconstruct_frame_v5,
                build_click_context, resolve_click, click_event_key,
                compute_features_v5, load_svm_model,
            )
            from config.app_config import AppConfig

            fft_data          = self.fft_data
            phase_data        = self.phase_data
            fs                = self.fs
            fft_size          = self.fft_size
            frame_duration_ms = self.frame_duration_ms

            model_path = Path(AppConfig.BASE_DIR) / 'src' / 'ml' / 'plantleaf_svm_v5.pkl'
            svm_model = load_svm_model(model_path)

            dm = _PaudioDM(fft_data, phase_data, fs, fft_size)
            stage1_candidates = run_stage1_v5(dm, k=1.5)
            if not stage1_candidates:
                self.finished.emit([])
                return

            candidates_with_features = []
            for cand in stage1_candidates:
                fi = cand['frame_idx']
                try:
                    fd = reconstruct_frame_v5(fft_data[fi], phase_data[fi], fs, fft_size, normalize=True)
                    if fd is None:
                        continue
                    curr_sig = fd['signal']

                    prev_sig = None
                    if fi > 0 and fi - 1 < len(phase_data):
                        pf = reconstruct_frame_v5(fft_data[fi-1], phase_data[fi-1], fs, fft_size, normalize=True)
                        if pf is not None:
                            prev_sig = pf['signal']

                    next_sig = None
                    if fi + 1 < len(fft_data) and fi + 1 < len(phase_data):
                        nf = reconstruct_frame_v5(fft_data[fi+1], phase_data[fi+1], fs, fft_size, normalize=True)
                        if nf is not None:
                            next_sig = nf['signal']

                    ctx      = build_click_context(prev_sig, curr_sig, next_sig)
                    resolved = resolve_click(ctx, cand['noise_floor'], cand['std_noise'])
                    features = compute_features_v5(
                        ctx, resolved,
                        fd['fft_norm'], fd['freq_axis'],
                        cand['noise_floor'], cand['std_noise'], fs,
                    )
                    peak_abs, canonical_frame_idx = click_event_key(ctx, resolved, fi)
                    candidates_with_features.append({
                        **cand, **features,
                        'peak_amp': resolved['peak_amp'],
                        'peak_abs': peak_abs,
                        'canonical_frame_idx': canonical_frame_idx,
                    })
                except Exception as e:
                    print(f"Warning: frame {fi} skipped: {e}")
                    continue

            stage2_survivors, _ = run_stage2_v5(candidates_with_features)
            if not stage2_survivors:
                self.finished.emit([])
                return

            detections, _ = run_stage3_v5(stage2_survivors, svm_model)
            if not detections:
                self.finished.emit([])
                return

            final = run_stage4_v5(detections)

            result = []
            for det in final:
                ts = det['frame_idx'] * frame_duration_ms / 1000.0
                result.append({
                    'timestamp': ts,
                    'tau_ms':   det.get('tau_ms', -1.0),
                    'peak_amp': det.get('peak_amp', 0.0),
                    'R2':       det.get('R2', 0.0),
                    'frequency': '',
                    'amplitude': str(det.get('SPR', '')),
                    'svm_probability': det.get('svm_probability', 0.0),
                })

            print(f"✅ Click detector v5: {len(result)} clicks")
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
        self._detect_thread = None
        self._detect_worker = None
        self._detect_progress = None

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

        phys_label = QLabel("Physical Parameters")
        phys_label.setAlignment(Qt.AlignCenter)
        phys_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2d4a2b; padding: 8px;")
        layout.addWidget(phys_label)

        phys_widget = QWidget()
        phys_layout = QVBoxLayout(phys_widget)
        phys_layout.setSpacing(8)
        phys_layout.setContentsMargins(0, 0, 0, 0)

        # R0
        r0_row = QHBoxLayout()
        r0_lbl = QLabel("R0:")
        r0_lbl.setFixedWidth(35)
        self.r0_slider = QSlider(Qt.Horizontal)
        self.r0_slider.setRange(20, 100)
        self.r0_slider.setValue(50)
        self.r0_spinbox = QDoubleSpinBox()
        self.r0_spinbox.setRange(20.0, 100.0)
        self.r0_spinbox.setValue(50.0)
        self.r0_spinbox.setSuffix(" µm")
        self.r0_spinbox.setDecimals(1)
        self.r0_spinbox.setSingleStep(1.0)
        self.r0_spinbox.setFixedWidth(90)
        r0_row.addWidget(r0_lbl)
        r0_row.addWidget(self.r0_slider)
        r0_row.addWidget(self.r0_spinbox)
        phys_layout.addLayout(r0_row)
        self.r0_slider.valueChanged.connect(lambda v: self.r0_spinbox.setValue(float(v)))
        self.r0_spinbox.valueChanged.connect(lambda v: self.r0_slider.setValue(int(v)))

        # P_inf
        pinf_row = QHBoxLayout()
        pinf_lbl = QLabel("P∞:")
        pinf_lbl.setFixedWidth(35)
        self.pinf_slider = QSlider(Qt.Horizontal)
        self.pinf_slider.setRange(-150, -30)
        self.pinf_slider.setValue(-30)
        self.pinf_spinbox = QDoubleSpinBox()
        self.pinf_spinbox.setRange(-1.5, -0.3)
        self.pinf_spinbox.setValue(-0.3)
        self.pinf_spinbox.setSuffix(" MPa")
        self.pinf_spinbox.setDecimals(2)
        self.pinf_spinbox.setSingleStep(0.05)
        self.pinf_spinbox.setFixedWidth(90)
        pinf_row.addWidget(pinf_lbl)
        pinf_row.addWidget(self.pinf_slider)
        pinf_row.addWidget(self.pinf_spinbox)
        phys_layout.addLayout(pinf_row)
        self.pinf_slider.valueChanged.connect(lambda v: self.pinf_spinbox.setValue(v / 100.0))
        self.pinf_spinbox.valueChanged.connect(lambda v: self.pinf_slider.setValue(int(v * 100)))

        # Distance
        dist_row = QHBoxLayout()
        dist_lbl = QLabel("Dist:")
        dist_lbl.setFixedWidth(35)
        self.dist_slider = QSlider(Qt.Horizontal)
        self.dist_slider.setRange(5, 50)
        self.dist_slider.setValue(10)
        self.dist_spinbox = QDoubleSpinBox()
        self.dist_spinbox.setRange(0.5, 5.0)
        self.dist_spinbox.setValue(1.0)
        self.dist_spinbox.setSuffix(" cm")
        self.dist_spinbox.setDecimals(1)
        self.dist_spinbox.setSingleStep(0.1)
        self.dist_spinbox.setFixedWidth(90)
        dist_row.addWidget(dist_lbl)
        dist_row.addWidget(self.dist_slider)
        dist_row.addWidget(self.dist_spinbox)
        phys_layout.addLayout(dist_row)
        self.dist_slider.valueChanged.connect(lambda v: self.dist_spinbox.setValue(v / 10.0))
        self.dist_spinbox.valueChanged.connect(lambda v: self.dist_slider.setValue(int(v * 10)))

        layout.addWidget(phys_widget)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)

        self.btn_load = QPushButton("Load .paudio File")
        self.btn_load.setMinimumHeight(42)
        self.btn_load.setObjectName("mainButton")
        self.btn_load.clicked.connect(self._load_paudio)
        layout.addWidget(self.btn_load)

        self.click_selector_label = QLabel("Select click to analyze:")
        layout.addWidget(self.click_selector_label)

        self.click_table = QTableWidget(0, 4)
        self.click_table.setHorizontalHeaderLabels(["Time (s)", "τ (ms)", "Peak", "R²"])
        self.click_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.click_table.verticalHeader().setVisible(False)
        self.click_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.click_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.click_table.setMaximumHeight(200)
        self.click_table.itemSelectionChanged.connect(self._on_click_selected)
        layout.addWidget(self.click_table)

        self.btn_run = QPushButton("Run Simulation")
        self.btn_run.setMinimumHeight(42)
        self.btn_run.setObjectName("mainButton")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._run_simulation)
        layout.addWidget(self.btn_run)

        self.btn_pdf = QPushButton("Generate PDF Report")
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
            x_label="Time", y_label="Norm. Amplitude",
            x_range=(0, 0.00768), y_range=(-1.2, 1.2),
            x_min=0, x_max=0.00768, y_min=-2, y_max=2,
            unit_x="s", unit_y="", parent=self
        )
        self.curve_sim_time  = self.plot_time.plot_widget.plot(name="Simulated",  pen={'color': '#689f67', 'width': 3.5})
        self.curve_real_time = self.plot_time.plot_widget.plot(name="Real click", pen={'color': 'r',       'width': 2})
        self.plot_time.plot_widget.showGrid(x=True, y=True)
        time_layout.addWidget(self.plot_time)
        tab.addTab(time_widget, "Time Domain")

        freq_widget = QWidget()
        freq_layout = QVBoxLayout(freq_widget)
        self.freq_info_label = QLabel("Frequency Domain 20–80 kHz")
        self.freq_info_label.setAlignment(Qt.AlignCenter)
        freq_layout.addWidget(self.freq_info_label)
        self.plot_freq = BasePlotWidget(
            x_label="Frequency", y_label="Norm. Amplitude",
            x_range=(20000, 80000), y_range=(0, 1.1),
            x_min=19000, x_max=81000, y_min=0, y_max=1.5,
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
            x_range=(0, 0.00256), y_range=(0, 100),
            x_min=0, x_max=0.003, y_min=0, y_max=200,
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
        title.setStyleSheet("font-size: 26px; font-weight: bold; color: #2d4a2b; padding: 8px;")
        layout.addWidget(title)

        layout.addSpacing(48)

        sim_title = QLabel("Simulated\nParameters")
        sim_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2d4a2b; padding: 4px;")
        sim_title.setAlignment(Qt.AlignLeft)
        layout.addWidget(sim_title)

        self.table_sim = QTableWidget(0, 2)
        self.table_sim.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.table_sim.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_sim.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table_sim.verticalHeader().setVisible(False)
        self.table_sim.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_sim.setAlternatingRowColors(True)
        self.table_sim.setStyleSheet("QTableWidget { border: none; }")
        layout.addWidget(self.table_sim)

        compare_title = QLabel("Comparison Real\nvs Simulated")
        compare_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2d4a2b; padding: 4px;")
        compare_title.setAlignment(Qt.AlignLeft)
        layout.addWidget(compare_title)

        self.table_compare = QTableWidget(0, 2)
        self.table_compare.setHorizontalHeaderLabels(["Metric", "Value"])
        self.table_compare.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_compare.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table_compare.verticalHeader().setVisible(False)
        self.table_compare.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_compare.setAlternatingRowColors(True)
        self.table_compare.setStyleSheet("QTableWidget { border: none; }")
        layout.addWidget(self.table_compare)

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
        home_action = QAction("Home", self)
        home_action.triggered.connect(self._go_home)
        toolbar.addAction(home_action)
        toolbar.addSeparator()
        run_action = QAction("Run", self)
        run_action.triggered.connect(self._run_simulation)
        toolbar.addAction(run_action)
        load_action = QAction("Load .paudio", self)
        load_action.triggered.connect(self._load_paudio)
        toolbar.addAction(load_action)

    def _load_paudio(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open .paudio File", self.settings_manager.get_last_directory("chem_sim_paudio"),
            "PlantLeaf Audio (*.paudio);;All Files (*)"
        )
        if not file_path:
            return
        self.settings_manager.set_last_directory("chem_sim_paudio", file_path)

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

            bytes_per_sample = 5 if version >= 3.0 else 4
            bytes_per_frame  = num_bins * bytes_per_sample
            n_frames = len(fft_bytes) // bytes_per_frame
            raw = np.frombuffer(
                fft_bytes[:n_frames * bytes_per_frame], dtype=np.uint8
            ).reshape(n_frames, num_bins, bytes_per_sample)
            # Magnitudes: first 4 bytes of each bin → float32
            mag_raw  = np.ascontiguousarray(raw[:, :, :4])
            fft_2d   = mag_raw.view(np.float32).reshape(n_frames, num_bins)
            fft_data = list(fft_2d)   # list of 1-D float32 arrays
            if version >= 3.0:
                # Phase: 5th byte of each bin → int8
                phase_2d   = np.ascontiguousarray(raw[:, :, 4]).view(np.int8).reshape(n_frames, num_bins)
                phase_data = list(phase_2d)
            else:
                phase_data = []

            # True STFT bin-center frequencies: f[k] = k * (fs/fft_size), k = bin_start..bin_end.
            # (linspace(freq_min, freq_max, num_bins) would mislabel the top bin by ~1 bin.)
            freq_axis = np.arange(bin_start, bin_start + num_bins) * bin_freq

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

            # Use embedded clicks only if they carry v5 fields (tau_ms + peak_amp).
            # Older CLCK sections (format: timestamp/frequency/amplitude/duration_us)
            # predate these fields — re-run the v5 detector instead.
            has_v5_fields = bool(clicks_raw) and clicks_raw[0].get('tau_ms') is not None
            if not has_v5_fields:
                # Launch detection in a background thread so the UI stays responsive.
                self._detect_file_path = file_path
                self._launch_click_detector(fft_data, phase_data, fs, fft_size, frame_duration_ms)
                return   # UI update happens via _on_detection_finished

            self._process_click_results(clicks_raw, frame_duration_ms, file_path)

        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Load Error", f"Could not load file:\n{str(e)}\n{traceback.format_exc()}")

    def _launch_click_detector(self, fft_data, phase_data, fs, fft_size, frame_duration_ms):
        self.file_label.setText("Running click detector v5…")
        self.btn_run.setEnabled(False)

        dlg = QProgressDialog(
            "Detecting clicks with v5 pipeline…\n(this may take a moment for long recordings)",
            None, 0, 0, self,
        )
        dlg.setWindowTitle("Click Detection")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setValue(0)
        dlg.show()
        QApplication.processEvents()
        self._detect_progress = dlg

        # Store frame_duration_ms so the finished callback can read it without
        # needing a lambda — lambdas have no thread affinity in Qt and cause
        # DirectConnection (= callback on worker thread = UI calls on wrong thread).
        self._pending_frame_duration_ms = frame_duration_ms

        worker = ClickDetectorWorker(fft_data, phase_data, fs, fft_size, frame_duration_ms)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        # Both connections below are QObject→QObject across threads: Qt automatically
        # uses QueuedConnection, ensuring the slots run on the main thread.
        worker.finished.connect(self._on_detection_finished)
        worker.error.connect(self._on_detection_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._detect_thread = thread
        self._detect_worker = worker
        thread.start()

    def _on_detection_finished(self, clicks_raw):
        if self._detect_progress:
            self._detect_progress.close()
            self._detect_progress = None
        frame_duration_ms = getattr(self, '_pending_frame_duration_ms', 2.56)
        file_path         = getattr(self, '_detect_file_path', '')
        self._process_click_results(clicks_raw, frame_duration_ms, file_path)

    def _on_detection_error(self, msg):
        if self._detect_progress:
            self._detect_progress.close()
            self._detect_progress = None
        if 'SVM model not found' in msg or 'FileNotFoundError' in msg:
            QMessageBox.warning(self, "SVM Model Missing",
                "Could not find plantleaf_svm_v5.pkl.\n"
                "Train the model first (src/ml/train_svm.py) and save it to src/ml/.")
        else:
            QMessageBox.critical(self, "Click Detector Error",
                f"An error occurred during click detection:\n\n{msg}")
        self.file_label.setText("Detection failed")

    def _process_click_results(self, clicks_raw, frame_duration_ms, file_path):
        fft_data = self.paudio_data['fft_data'] if self.paudio_data else []

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
            except Exception:
                ts = 0.0

            tau_ms = float(click.get('tau_ms', -1.0))
            if tau_ms <= 0:
                duration_str = str(click.get('duration', ''))
                if 'FFT' in duration_str:
                    try:
                        fft_count = int(duration_str.replace(' FFT', '').strip())
                        tau_ms = fft_count * 2.56 / 3.0
                    except Exception:
                        pass

            frame_idx = int(round(ts * 1000.0 / frame_duration_ms))
            frame_idx = max(0, min(frame_idx, max(0, len(fft_data) - 1)))

            self.real_clicks.append({
                'timestamp': ts,
                'frame_idx': frame_idx,
                'tau_ms':    tau_ms,
                'frequency': click.get('frequency', ''),
                'amplitude': click.get('amplitude', ''),
                'r2':        float(click.get('R2', click.get('r2', click.get('r2_log', 0.0)))),
                'peak_amp':  float(click.get('peak_amp', 0.0)),
            })

        self._populate_click_table()
        self.file_label.setText(
            f"{os.path.basename(file_path)}\n{len(self.real_clicks)} clicks found"
        )
        if self.real_clicks:
            self.btn_run.setEnabled(True)

    def _populate_click_table(self):
        self.click_table.setRowCount(len(self.real_clicks))
        for i, click in enumerate(self.real_clicks):
            self.click_table.setItem(i, 0, QTableWidgetItem(f"{click['timestamp']:.3f}"))
            tau_str = f"{click['tau_ms']:.3f}" if click['tau_ms'] > 0 else "N/A"
            self.click_table.setItem(i, 1, QTableWidgetItem(tau_str))
            peak_uv = click.get('peak_amp', 0.0) * 1e6
            self.click_table.setItem(i, 2, QTableWidgetItem(f"{peak_uv:.0f}"))
            r2_val = click.get('r2', click.get('r2_log', 0.0))
            self.click_table.setItem(i, 3, QTableWidgetItem(f"{r2_val:.3f}"))

    def _on_click_selected(self):
        rows = self.click_table.selectedItems()
        if not rows:
            return
        row = self.click_table.currentRow()
        if row < 0 or row >= len(self.real_clicks):
            return
        # Selecting a different click invalidates any existing simulation
        self._discard_simulation()
        click = self.real_clicks[row]
        self._show_real_click(click)

    def _discard_simulation(self):
        """Clear the simulated (green) overlay and its result tables.

        Called whenever the selected click changes: the previous simulation no
        longer corresponds to the newly selected click, so it must be discarded.
        """
        self.sim_result = None
        self.curve_sim_time.setData([], [])
        self.curve_sim_freq.setData([], [])
        self.curve_bubble.setData([], [])
        self.table_sim.setRowCount(0)
        self.table_compare.setRowCount(0)
        self.correlation_label.setText("Correlation: —")
        self.btn_pdf.setEnabled(False)

    def _show_real_click(self, click):
        if not self.paudio_data:
            return
        pd        = self.paudio_data
        frame_idx = click['frame_idx']
        fft_data  = pd['fft_data']
        phase_data = pd['phase_data']

        if frame_idx >= len(fft_data):
            return

        from core.click_pipeline_v5 import reconstruct_frame_v5

        fs       = pd['fs']
        fft_size = pd['fft_size']
        phases = (
            phase_data[frame_idx]
            if phase_data and frame_idx < len(phase_data)
            else np.array([], dtype=np.int8)
        )
        fd = reconstruct_frame_v5(fft_data[frame_idx], phases, fs, fft_size, normalize=True)
        if fd is None:
            return

        # FFT plot: mic-corrected, Tukey-tapered magnitudes (analysis band only), normalized to [0,1]
        bin_start     = pd['bin_start']
        num_bins      = pd['num_bins']
        freq_axis     = pd['freq_axis']
        fft_norm_band = fd['fft_norm'][bin_start : bin_start + num_bins]
        peak = np.max(fft_norm_band)
        if peak > 0:
            fft_norm_band = fft_norm_band / peak
        self.curve_real_freq.setData(freq_axis, fft_norm_band)

        # Time-domain: [frame−1 | frame | frame+1] concatenated into one continuous signal
        frame_dur  = fft_size / fs                           # seconds per frame

        def _recon(fi):
            if fi < 0 or fi >= len(fft_data):
                return np.zeros(fft_size, dtype=np.float32)  # silent padding at file edges
            ph = (phase_data[fi]
                  if phase_data and fi < len(phase_data)
                  else np.array([], dtype=np.int8))
            fd_r = reconstruct_frame_v5(fft_data[fi], ph, fs, fft_size, normalize=True)
            return fd_r['signal'] if fd_r is not None else np.zeros(fft_size, dtype=np.float32)

        sig_prev   = _recon(frame_idx - 1)
        sig_center = fd['signal']
        sig_next   = _recon(frame_idx + 1)

        sig_full    = np.concatenate([sig_prev, sig_center, sig_next])
        center_peak = np.max(np.abs(sig_center)) + 1e-30
        t_full      = np.linspace(0, 3 * frame_dur, 3 * fft_size)

        # Peak position inside the full 3-frame axis (for sim alignment)
        self._real_click_peak_t = frame_dur + np.argmax(np.abs(sig_center)) / fs

        self.curve_real_time.setData(t_full, sig_full / center_peak)

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

        t_sim    = bubble['t']                                      # seconds
        signal   = propagation['signal']
        sim_norm = signal / (np.max(np.abs(signal)) + 1e-30)
        t_peak_sim  = t_sim[np.argmax(np.abs(signal))]
        t_peak_real = getattr(self, '_real_click_peak_t', 0.0)
        self.curve_sim_time.setData(t_sim - t_peak_sim + t_peak_real, sim_norm)

        freq      = plantleaf['freq']
        spec      = plantleaf['spectrum']
        spec_peak = np.max(spec)
        spec_norm = spec / spec_peak if spec_peak > 0 else spec
        self.curve_sim_freq.setData(freq, spec_norm)

        R_um = bubble['R'] * 1e6
        self.curve_bubble.setData(t_sim, R_um)

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
                from core.click_pipeline_v5 import reconstruct_frame_v5
                pd  = self.paudio_data
                fi  = click['frame_idx']
                ph  = (pd['phase_data'][fi]
                       if pd['phase_data'] and fi < len(pd['phase_data'])
                       else np.array([], dtype=np.int8))
                fd2 = reconstruct_frame_v5(pd['fft_data'][fi], ph,
                                           pd['fs'], pd['fft_size'], normalize=True)
                if fd2 is not None:
                    real_signal = fd2['signal']

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
        start_dir = self.settings_manager.get_last_directory("chem_sim_report")
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF Report", os.path.join(start_dir, "report_acoustic.pdf"), "PDF Files (*.pdf)"
        )
        if not file_path:
            return
        try:
            from chemical_simulators.report_acoustic import generate_report
            generate_report(simulation_result=self.sim_result, output_path=file_path)
            self.settings_manager.set_last_directory("chem_sim_report", file_path)
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