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
from scipy.signal import hilbert
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

    def __init__(self, R0, P_inf, distance_m, tau_target_ms=None, freq_target_hz=None, real_signal_for_fit=None):
        super().__init__()
        self.R0 = R0
        self.P_inf = P_inf
        self.distance_m = distance_m
        self.tau_target_ms = tau_target_ms
        self.freq_target_hz = freq_target_hz
        self.real_signal_for_fit = real_signal_for_fit

    def run(self):
        try:
            from chemical_simulators.run_acoustic_simulation import run_simulation
            self.progress.emit(30)
            result = run_simulation(
                R0=self.R0,
                P_inf=self.P_inf,
                distance_m=self.distance_m,
                tau_target_ms=self.tau_target_ms,
                freq_target_hz=self.freq_target_hz,
                real_signal_for_fit=self.real_signal_for_fit
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
        self._start_load(file_path)

    def _start_load(self, file_path):
        """
        Read the recording with the app's shared loader (AudioLoadWorker).

        It reads both .paudio formats (v3 continuous, v4 event) and computes, at
        load time, the Stage-1 arrays and the Buffer-3 noise snapshots that the v6
        detector needs. The replay window and the Data Collection export use the
        same loader, so the clicks listed here are the clicks in the dataset.
        """
        from saving.audio_load_progress import AudioLoadWorker

        self._discard_simulation()
        self.real_clicks = []
        self.click_table.setRowCount(0)
        self.btn_run.setEnabled(False)
        self._load_file_path = file_path
        self.file_label.setText("Loading…")

        dlg = QProgressDialog("Loading .paudio file…", None, 0, 100, self)
        dlg.setWindowTitle("Load")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setValue(0)
        dlg.show()
        self._load_progress = dlg

        worker = AudioLoadWorker(file_path)
        thread = QThread(self)
        worker.moveToThread(thread)
        # Bound methods, never lambdas: see _launch_click_detector.
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_load_progress)
        worker.finished.connect(self._on_load_finished)
        worker.error.connect(self._on_load_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._load_thread = thread
        self._load_worker = worker
        thread.start()

    def _on_load_progress(self, pct):
        if getattr(self, '_load_progress', None) is not None:
            self._load_progress.setValue(int(pct))

    def _close_load_progress(self):
        if getattr(self, '_load_progress', None) is not None:
            self._load_progress.close()
            self._load_progress = None

    def _on_load_error(self, msg):
        self._close_load_progress()
        self.file_label.setText("Load failed")
        QMessageBox.critical(self, "Load Error", f"Could not load file:\n{msg}")

    def _on_load_finished(self, data):
        self._close_load_progress()
        try:
            from types import SimpleNamespace

            hi       = data['header_info']
            fs       = int(hi['fs'])
            fft_size = int(hi['fft_size'])
            bin_freq  = fs / fft_size
            bin_start = int(hi['freq_min'] / bin_freq)
            bin_end   = int(hi['freq_max'] / bin_freq)
            num_bins  = bin_end - bin_start + 1

            # The v6 pipeline reads the loader's output as attributes of a data
            # manager (fft_means, E_hat_floor_arr, p_noise_snapshots, event_*…).
            self._dm = SimpleNamespace(**data)

            self.paudio_data = {
                'fft_data':   data['fft_data'],
                'phase_data': data['phase_data'],
                # True STFT bin-center frequencies: f[k] = k * (fs/fft_size), k = bin_start..bin_end.
                'freq_axis':  np.arange(bin_start, bin_start + num_bins) * bin_freq,
                'fs':         fs,
                'fft_size':   fft_size,
                'freq_min':   hi['freq_min'],
                'freq_max':   hi['freq_max'],
                'bin_start':  bin_start,
                'num_bins':   num_bins,
                'version':    hi['version'],
                'frame_duration_ms': data['frame_duration_ms'],
                'is_event_recording': bool(data.get('is_event_recording', False)),
            }
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Load Error", f"Could not load file:\n{str(e)}\n{traceback.format_exc()}")
            return

        # Clicks embedded in the file (CLCK / EVTR) are not reused: they may come
        # from an older detector. Re-running v6 keeps this list identical to the
        # replay window's and to the exported dataset.
        self._launch_click_detector(self._dm)

    def _launch_click_detector(self, dm):
        from core.click_detection_worker import ClickDetectionWorker

        pd = self.paudio_data
        self.file_label.setText("Running click detector v6…")
        self.btn_run.setEnabled(False)

        dlg = QProgressDialog(
            "Detecting clicks with the v6 pipeline…\n(this may take a moment for long recordings)",
            None, 0, 100, self,
        )
        dlg.setWindowTitle("Click Detection")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setValue(0)
        dlg.show()
        QApplication.processEvents()
        self._detect_progress = dlg

        worker = ClickDetectionWorker(
            fft_data=dm.fft_data,
            phase_data=dm.phase_data,
            fs=pd['fs'],
            fft_size=pd['fft_size'],
            frame_duration_ms=pd['frame_duration_ms'],
            dm=dm,   # Stage-1 arrays + Buffer 3 from the loader (v6 features)
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        # Both connections below are QObject→QObject across threads: Qt automatically
        # uses QueuedConnection, ensuring the slots run on the main thread. Lambdas
        # have no thread affinity and would run the slots on the worker thread.
        worker.progress.connect(self._on_detection_progress)
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

    def _on_detection_progress(self, done, total):
        if self._detect_progress is not None and total > 0:
            self._detect_progress.setValue(int(100 * done / total))

    def _on_detection_finished(self, rows):
        if self._detect_progress:
            self._detect_progress.close()
            self._detect_progress = None
        from core.click_pipeline_v5 import STAGE_OK
        # The worker returns every Stage-1 candidate with its verdict; only the
        # ones that survived all four stages are clicks.
        confirmed = [r for r in (rows or []) if r.get('stage_blocked', STAGE_OK) == STAGE_OK]
        self._process_click_results(confirmed)

    def _on_detection_error(self, msg):
        if self._detect_progress:
            self._detect_progress.close()
            self._detect_progress = None
        QMessageBox.critical(self, "Click Detector Error",
            f"An error occurred during click detection:\n\n{msg}")
        self.file_label.setText("Detection failed")

    def _process_click_results(self, detections):
        file_path = getattr(self, '_load_file_path', '')

        if not detections:
            QMessageBox.warning(self, "No Clicks",
                "The v6 detector confirmed no ultrasonic clicks in this file.")
            self.file_label.setText("No clicks found")
            return

        is_event = self.paudio_data.get('is_event_recording', False)
        self.real_clicks = []
        for det in sorted(detections, key=lambda d: d.get('frame_idx', 0)):
            fi = int(det['frame_idx'])
            if is_event:
                # Array rows are not frame indices in an event recording, and a
                # neighbour the board never sent is None.
                row, prev_row, next_row = det['row_idx'], det.get('prev_row'), det.get('next_row')
            else:
                row, prev_row, next_row = fi, fi - 1, fi + 1
            fpe = det.get('FPE_hz_region')
            self.real_clicks.append({
                'timestamp': float(det.get('timestamp_s', 0.0)),
                'frame_idx': fi,
                'row_idx':   row,
                'prev_row':  prev_row,
                'next_row':  next_row,
                'tau_ms':    float(det.get('tau_ms', -1.0)),
                'r2':        float(det.get('R2', 0.0)),
                'peak_amp':  float(det.get('peak_amp', 0.0)),
                'FPE_hz_region':   float(fpe) if fpe is not None else float('nan'),
                'svm_probability': det.get('svm_probability'),
            })

        self._populate_click_table()
        self.file_label.setText(
            f"{os.path.basename(file_path)}\n{len(self.real_clicks)} clicks found (v6)"
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

    def _reconstruct_row(self, row):
        """
        One recorded frame as reconstruct_frame_v5 returns it (mic-corrected,
        Tukey-tapered iFFT, Gibbs-suppressed), or None when the row does not
        exist. `row` is an ARRAY row: in a v4 event recording it differs from
        the frame index, and a neighbour the board never sent is None.
        """
        from core.click_pipeline_v5 import reconstruct_frame_v5

        pd = self.paudio_data
        fft_data, phase_data = pd['fft_data'], pd['phase_data']
        n_rows = min(len(fft_data), len(phase_data))
        if row is None or not (0 <= row < n_rows):
            return None
        return reconstruct_frame_v5(fft_data[row], phase_data[row],
                                    pd['fs'], pd['fft_size'], normalize=True)

    def _click_frame_signals(self, click):
        """[prev | current | next] frame signals of a click; silence where absent."""
        silence = np.zeros(self.paudio_data['fft_size'], dtype=np.float32)
        out = []
        for row in (click['prev_row'], click['row_idx'], click['next_row']):
            fd = self._reconstruct_row(row)
            out.append(fd['signal'] if fd is not None else silence)
        return out

    def _real_click_signal(self, click):
        """
        The real click as ONE signal: the three frames stitched, exactly as the
        Time Domain plot shows it. Every place that measures the real click
        (frequency, fit, correlation) uses this, so they all see the same signal.
        """
        if not self.paudio_data:
            return None
        return np.concatenate(self._click_frame_signals(click))

    def _show_real_click(self, click):
        if not self.paudio_data:
            return
        pd = self.paudio_data
        fs       = pd['fs']
        fft_size = pd['fft_size']

        fd = self._reconstruct_row(click['row_idx'])
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

        sig_prev, sig_center, sig_next = self._click_frame_signals(click)

        sig_full    = np.concatenate([sig_prev, sig_center, sig_next])
        center_peak = np.max(np.abs(sig_center)) + 1e-30
        t_full      = np.linspace(0, 3 * frame_dur, 3 * fft_size)

        # Peak position inside the full 3-frame axis (for sim alignment)
        self._real_click_peak_t = frame_dur + np.argmax(np.abs(sig_center)) / fs

        self.curve_real_time.setData(t_full, sig_full / center_peak)

    def _find_click_envelope_bounds(self, signal, level_fraction=0.1, guard=5, max_search=300):
        """
        Trova inizio, picco e fine di un click reale dentro un segnale
        grezzo, usando una soglia sull'inviluppo di Hilbert (10% del
        picco). Restituisce (start_idx, peak_idx, end_idx), o None.
        """
        if signal is None or len(signal) < 10:
            return None
        try:
            envelope = np.abs(hilbert(signal))
            peak_idx = int(np.argmax(envelope))
            peak_amp = envelope[peak_idx]
            if peak_amp <= 0:
                return None
            level = peak_amp * level_fraction

            start_idx = max(peak_idx - max_search, 0)
            for i in range(peak_idx - 1, max(peak_idx - max_search, -1), -1):
                if envelope[i] < level:
                    start_idx = i + 1
                    break

            end_idx = min(peak_idx + max_search, len(envelope) - 1)
            for i in range(peak_idx + 1, end_idx + 1):
                if envelope[i] < level:
                    end_idx = i
                    break

            return start_idx, peak_idx, end_idx
        except Exception:
            return None

    def _extract_click_dominant_frequency(self, click):
        """
        Estrae la frequenza dominante dal click reale selezionato tramite
        conteggio degli attraversamenti dello zero (zero-crossing) sulla
        porzione isolata del click. Per segmenti così brevi la FFT
        soffre di spectral leakage; lo zero-crossing è più diretto e
        affidabile.
        """
        if not self.paudio_data:
            return None
        try:
            signal = self._real_click_signal(click)
            if signal is None or len(signal) < 10:
                return None

            bounds = self._find_click_envelope_bounds(signal)
            if bounds is None:
                return None
            start_idx, peak_idx, end_idx = bounds

            segment = signal[start_idx:end_idx + 1]
            if len(segment) < 4:
                return None

            signs = np.sign(segment)
            signs[signs == 0] = 1
            crossings = np.where(np.diff(signs) != 0)[0]
            if len(crossings) < 2:
                return None

            n_cycles = len(crossings) / 2.0
            fs = self.paudio_data['fs']
            duration_s = len(segment) / fs
            if duration_s <= 0:
                return None

            freq = n_cycles / duration_s
            if freq < 20000 or freq > 80000:
                return None
            return float(freq)
        except Exception:
            return None

    def _run_simulation(self):
        R0 = self.r0_spinbox.value() * 1e-6
        P_inf = self.pinf_spinbox.value() * 1e6
        distance_m = self.dist_spinbox.value() * 0.01

        rows_sel = self.click_table.selectedItems()
        tau_target = None
        freq_target = None
        real_signal_for_fit = None
        if rows_sel:
            row = self.click_table.currentRow()
            if row >= 0 and row < len(self.real_clicks):
                click = self.real_clicks[row]
                tau_target = click.get('tau_ms', None)
                freq_target = self._extract_click_dominant_frequency(click)
                real_signal_for_fit = self._real_click_signal(click)

        self.progress_dialog = QProgressDialog("Running simulation...", None, 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(10)
        self.progress_dialog.show()

        self.sim_thread = QThread(self)
        self.sim_worker = SimulationWorker(R0, P_inf, distance_m, tau_target, freq_target, real_signal_for_fit)
        self.sim_worker.moveToThread(self.sim_thread)
        self.sim_thread.started.connect(self.sim_worker.run)
        self.sim_worker.finished.connect(self._on_simulation_finished)
        self.sim_worker.error.connect(self._on_simulation_error)
        self.sim_worker.progress.connect(self.progress_dialog.setValue)
        self.sim_worker.finished.connect(self.sim_thread.quit)
        self.sim_worker.error.connect(self.sim_thread.quit)
        self.sim_thread.finished.connect(self.sim_thread.deleteLater)
        self.sim_thread.start()

    def _on_simulation_finished(self, result):
        self.progress_dialog.close()
        self.sim_result = result
        self._update_plots(result)
        self._update_diagnostics(result)
        self._handle_calibration_feedback(result)
        self.btn_pdf.setEnabled(True)
        print("Simulation completed")

    def _handle_calibration_feedback(self, result):
        calibration = result.get('calibration')
        if calibration is None:
            return

        R0_used_um = result['bubble']['R0'] * 1e6

        self.r0_spinbox.blockSignals(True)
        self.r0_slider.blockSignals(True)
        self.r0_spinbox.setValue(R0_used_um)
        self.r0_slider.setValue(int(round(R0_used_um)))
        self.r0_spinbox.blockSignals(False)
        self.r0_slider.blockSignals(False)

        note = calibration.get('note', '')
        msg = f"R0 calibrato automaticamente a {R0_used_um:.1f} µm.\n\n{note}"
        QMessageBox.information(self, "Calibrazione completata", msg)

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

    def _compute_hilbert_envelope(self, signal):
        """
        Calcola l'inviluppo istantaneo del segnale usando la trasformata
        di Hilbert. Implementazione identica a quella usata dall'algoritmo
        di rilevamento click di PlantLeaf (replay_window_audio.py), in
        numpy puro (no scipy) per coerenza e thread-safety.
        """
        N = len(signal)
        Xf = np.fft.rfft(signal, n=N)
        h = np.zeros(N // 2 + 1, dtype=np.float64)
        h[0] = 1.0
        if N % 2 == 0:
            h[1:-1] = 2.0
            h[-1] = 1.0
        else:
            h[1:] = 2.0
        hilbert_part = np.fft.irfft(Xf * h, n=N)
        return np.sqrt(signal ** 2 + hilbert_part ** 2)

    def _compute_envelope_correlation(self, real_signal, sim_signal):
        """
        Confronta la FORMA DEL DECADIMENTO (inviluppo) tra click reale e
        simulato, con una finestra di confronto proporzionale alla vera
        durata di ciascun click (non una lunghezza fissa), individuata
        tramite soglia sull'inviluppo — la stessa identica logica usata
        da PlantLeaf per isolare un click dal resto del segnale. Una
        finestra fissa troppo lunga rispetto a un click molto rapido
        (es. τ=0.05ms) trascinerebbe dentro rumore/silenzio, diluendo
        artificialmente la correlazione anche quando il decadimento
        combacia bene.

        Returns:
            float: correlazione di Pearson tra i due inviluppi (-1..1)
        """
        try:
            real_bounds = self._find_click_envelope_bounds(real_signal, max_search=500)
            sim_bounds = self._find_click_envelope_bounds(sim_signal, max_search=500)
            if real_bounds is None or sim_bounds is None:
                return 0.0

            real_start, real_peak, real_end = real_bounds
            sim_start, sim_peak, sim_end = sim_bounds

            pre = min(real_peak - real_start, sim_peak - sim_start)
            post = min(real_end - real_peak, sim_end - sim_peak)
            if post <= 3:
                return 0.0

            real_env = self._compute_hilbert_envelope(real_signal)
            sim_env = self._compute_hilbert_envelope(sim_signal)

            r_win = real_env[real_peak - pre: real_peak + post + 1]
            s_win = sim_env[sim_peak - pre: sim_peak + post + 1]
            n = min(len(r_win), len(s_win))
            if n <= 6:
                return 0.0
            r_win = r_win[:n]
            s_win = s_win[:n]

            r_norm = r_win / (np.max(r_win) + 1e-30)
            s_norm = s_win / (np.max(s_win) + 1e-30)

            corr = np.corrcoef(r_norm, s_norm)[0, 1]
            return float(corr) if np.isfinite(corr) else 0.0
        except Exception: 
            return 0.0


    def _compute_best_correlation(self, real_signal, sim_signal, max_lag=60):
        """
        Calcola la correlazione tra il click reale e quello simulato
        cercando lo sfasamento (lag) che la massimizza, invece di un
        confronto a fase fissa allineata solo sui picchi.

        Due oscillazioni con la stessa frequenza e lo stesso decadimento
        possono avere una correlazione di Pearson vicina a zero se sono
        sfasate (es. un coseno contro un seno) — la fase assoluta del
        click reale dipende dall'istante esatto di nucleazione, che non
        conosciamo né modelliamo. Cercare il miglior allineamento
        temporale è la pratica standard per confrontare forme d'onda
        oscillatorie di fase relativa sconosciuta, e riflette meglio se
        la FORMA del click (non la fase arbitraria) combacia.

        Returns:
            float: la massima correlazione di Pearson trovata (-1..1)
        """
        try:
            real_bounds = self._find_click_envelope_bounds(real_signal)
            sim_bounds = self._find_click_envelope_bounds(sim_signal)
            if real_bounds is None or sim_bounds is None:
                return 0.0

            _, real_peak, _ = real_bounds
            _, sim_peak, _ = sim_bounds

            pre = min(real_peak, sim_peak, 20)
            post = min(len(real_signal) - real_peak, len(sim_signal) - sim_peak, 200) - 1
            if post <= 5:
                return 0.0

            r_win = real_signal[real_peak - pre: real_peak + post]
            s_win = sim_signal[sim_peak - pre: sim_peak + post]
            n = min(len(r_win), len(s_win))
            if n <= 10:
                return 0.0
            r_win = r_win[:n]
            s_win = s_win[:n]

            r_norm = r_win / (np.max(np.abs(r_win)) + 1e-30)
            s_norm = s_win / (np.max(np.abs(s_win)) + 1e-30)

            best_corr = 0.0
            max_shift = min(max_lag, n // 2)
            for shift in range(-max_shift, max_shift + 1):
                if shift >= 0:
                    a = r_norm[shift:]
                    b = s_norm[:len(a)]
                else:
                    b = s_norm[-shift:]
                    a = r_norm[:len(b)]
                if len(a) < 10:
                    continue
                c = np.corrcoef(a, b)[0, 1]
                if np.isfinite(c) and abs(c) > abs(best_corr):
                    best_corr = c

            return float(best_corr)
        except Exception:
            return 0.0

    def _update_diagnostics(self, result):
        diag   = result['diagnostics']
        bubble = result['bubble']

        f0_val = bubble.get('f0', None)
        extra_damping = bubble.get('extra_damping_rate', 0.0)

        rows = [
            ("R₀",         f"{bubble['R0']*1e6:.1f} µm"),
            ("P∞",         f"{bubble['P_inf']/1e6:.2f} MPa"),
            ("Freq. naturale (f₀)", f"{f0_val/1000:.1f} kHz" if f0_val else "N/A"),
            ("Smorz. extra vaso (fit)", f"{extra_damping:.0f} 1/s" if extra_damping else "0 (nessun fitting)"),
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

            real_signal = self._real_click_signal(click)

            sim_signal = result['propagation']['signal']
            corr = 0.0
            env_corr = 0.0
            if real_signal is not None and len(real_signal) > 0 and len(sim_signal) > 0:
                corr = self._compute_best_correlation(real_signal, sim_signal)
                env_corr = self._compute_envelope_correlation(real_signal, sim_signal)

            self.correlation_label.setText(f"Envelope Correlation: {env_corr:.4f}")

            compare_rows = [
                ("τ real (ms)",  f"{tau_real:.3f}" if tau_real > 0 else "N/A"),
                ("τ sim (ms)",   f"{tau_sim:.3f}"),
                ("Correlation (onda)",  f"{corr:.4f}"),
                ("Correlation (busta)", f"{env_corr:.4f}"),
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