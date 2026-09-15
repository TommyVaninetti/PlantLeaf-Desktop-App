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
    QApplication, QComboBox, QScrollArea
)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QIcon, QAction, QFont

from core.settings_manager import SettingsManager
from core.font_manager import FontManager
from core.layout_manager import LayoutManager
from core.theme_manager import ThemeManager
from config.app_config import AppConfig
from plotting.plot_manager import BasePlotWidget
import pyqtgraph as pg


SLIDER_CSS = (
    "QSlider::groove:horizontal { background: #a5d6a7; height: 6px; border-radius: 3px; }"
    "QSlider::handle:horizontal { background: #5a7559; border: 2px solid #5a7559;"
    " width: 16px; height: 16px; margin: -5px 0; border-radius: 8px; }"
    "QSlider::sub-page:horizontal { background: #689f67; border-radius: 3px; }"
    "QSlider::add-page:horizontal { background: #c8e6c9; border-radius: 3px; }"
)


class ModelComparisonWorker(QObject):
    """
    Runs click_model_comparison.analyse_click on one or more clicks, off the
    GUI thread. Each job is (index, frames, frame_idx, noise_floor, std_noise).
    """
    result   = Signal(int, object)   # (click index, analyse_click result)
    progress = Signal(int, int)      # (done, total)
    finished = Signal()
    error    = Signal(str)

    def __init__(self, jobs):
        super().__init__()
        self.jobs = jobs
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        try:
            import click_model_comparison as cmc
            total = len(self.jobs)
            for n, (idx, frames, frame_idx, noise_floor, std_noise) in enumerate(self.jobs):
                if self._stop_requested:
                    break
                self.result.emit(idx, cmc.analyse_click(frames, frame_idx, noise_floor, std_noise))
                self.progress.emit(n + 1, total)
            self.finished.emit()
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
        self.model_results = {}     # click index -> analyse_click result
        self._compare_thread = None
        self._compare_worker = None
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
        # Scrollable: with the model selector and the batch buttons the controls
        # no longer fit a laptop-height window, and a squeezed QVBoxLayout
        # draws its rows on top of each other instead of shrinking them.
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QFrame.NoFrame)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        controls_panel = self._build_controls_panel()
        controls_panel.setObjectName("chemSimControls")
        controls_scroll.setObjectName("chemSimControlsScroll")
        # The scroll area's viewport paints the SYSTEM palette (black in macOS dark
        # mode), which the app themes do not style. Make both see-through so the
        # themed window background shows, as it did before the scroll area existed.
        controls_scroll.setStyleSheet(
            "QScrollArea#chemSimControlsScroll { background: transparent; border: none; }"
            "QWidget#chemSimControls { background: transparent; }")
        controls_scroll.viewport().setAutoFillBackground(False)
        controls_panel.setAutoFillBackground(False)
        controls_scroll.setWidget(controls_panel)
        splitter.addWidget(controls_scroll)
        splitter.addWidget(self._build_plots_panel())
        splitter.addWidget(self._build_results_panel())
        splitter.setSizes([340, 620, 280])

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
        self.r0_slider.setRange(1, 500)
        self.r0_slider.setValue(50)
        self.r0_spinbox = QDoubleSpinBox()
        self.r0_spinbox.setRange(1.0, 500.0)
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

        # Physical model the real click is compared with. Both are computed on
        # every run; the selector only chooses which one the plots and tables show.
        model_row = QHBoxLayout()
        model_lbl = QLabel("Model:")
        model_lbl.setFixedWidth(50)
        self.model_combo = QComboBox()
        self.model_combo.addItem("Xylem vessel — Dutta 2022", 'vessel')
        self.model_combo.addItem("Free bubble — Minnaert (+thermal)", 'bubble')
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        model_row.addWidget(model_lbl)
        model_row.addWidget(self.model_combo)
        phys_layout.addLayout(model_row)

        # Vessel wall (Dutta 2022, Eq. 2): only the vessel LENGTH depends on them.
        wall_row = QHBoxLayout()
        wall_row.addWidget(QLabel("E:"))
        self.young_spinbox = QDoubleSpinBox()
        self.young_spinbox.setRange(0.05, 2.0)
        self.young_spinbox.setDecimals(2)
        self.young_spinbox.setSingleStep(0.05)
        self.young_spinbox.setValue(0.20)
        self.young_spinbox.setSuffix(" GPa")
        self.young_spinbox.setToolTip("Young's modulus of the vessel wall (Dutta 2022: 0.2 ± 0.1 GPa, fresh stems)")
        wall_row.addWidget(self.young_spinbox)
        wall_row.addWidget(QLabel("h:"))
        self.wall_spinbox = QDoubleSpinBox()
        self.wall_spinbox.setRange(0.1, 10.0)
        self.wall_spinbox.setDecimals(1)
        self.wall_spinbox.setSingleStep(0.1)
        self.wall_spinbox.setValue(1.0)
        self.wall_spinbox.setSuffix(" µm")
        self.wall_spinbox.setToolTip("Vessel wall thickness (Dutta 2022: ~1 µm)")
        wall_row.addWidget(self.wall_spinbox)
        phys_layout.addLayout(wall_row)
        self.young_spinbox.valueChanged.connect(self._on_model_changed)
        self.wall_spinbox.valueChanged.connect(self._on_model_changed)

        # R0 is no longer an input: each model derives its geometry from the
        # selected click (bubble: R0 from f; vessel: R from τ, L from f).
        for w in (self.r0_slider, self.r0_spinbox):
            w.setEnabled(False)
            w.setToolTip("Free-bubble R₀, derived from the selected click's frequency (read-only)")
        for w in (self.pinf_slider, self.pinf_spinbox, self.dist_slider, self.dist_spinbox):
            w.setToolTip("Used only by the PDF report; not part of the model comparison")

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

        self.click_table = QTableWidget(0, 7)
        self.click_table.setHorizontalHeaderLabels(
            ["Time (s)", "τ (ms)", "Peak", "R²", "R (µm)", "L (mm)", "τ/τ bubble"])
        self.click_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
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

        batch_row = QHBoxLayout()
        self.btn_run_all = QPushButton("Analyze All")
        self.btn_run_all.setMinimumHeight(38)
        self.btn_run_all.setObjectName("mainButton")
        self.btn_run_all.setEnabled(False)
        self.btn_run_all.setToolTip("Compare every click in this recording with both models")
        self.btn_run_all.clicked.connect(self._analyze_all_clicks)
        batch_row.addWidget(self.btn_run_all)

        self.btn_export = QPushButton("Export CSV")
        self.btn_export.setMinimumHeight(38)
        self.btn_export.setObjectName("mainButton")
        self.btn_export.setEnabled(False)
        self.btn_export.setToolTip("One row per analysed click: real τ/f, vessel R/L, bubble τ ratio, R²")
        self.btn_export.clicked.connect(self._export_results_csv)
        batch_row.addWidget(self.btn_export)
        layout.addLayout(batch_row)

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
        self.region_time = pg.LinearRegionItem(brush=(104, 159, 103, 40), movable=False)
        self.region_time.setZValue(-10)
        self.region_time.hide()
        self.plot_time.plot_widget.addItem(self.region_time)
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
        bubble_layout.addWidget(QLabel("Bubble radius R(t) of the free-bubble model at the R₀ that rings at the click frequency\n(10 % initial perturbation; the amplitude is arbitrary, the ring-down time is the prediction)"))
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

        # τ–f map: every analysed click (chain-calibrated f, τ) against the two
        # models — the free bubble fixes τ for each f, the vessel model maps τ to a
        # vessel radius (horizontal lines).
        map_widget = QWidget()
        map_layout = QVBoxLayout(map_widget)
        map_caption = QLabel(
            "Each red dot is one analysed click: its frequency and its decay time τ, both "
            "corrected for the measurement chain.\n"
            "Blue curve — the τ a FREE air bubble ringing at that frequency must have "
            "(Minnaert + radiation/viscous/thermal damping): a click on the curve is "
            "consistent with a free bubble, far above it rings too long for one.\n"
            "Dashed lines — Dutta 2022 vessel model: a click on the line 'R = 30 µm' would "
            "come from a xylem vessel of 30 µm radius (compare with the plant's anatomy).")
        map_caption.setWordWrap(True)
        map_layout.addWidget(map_caption)
        self.plot_map = pg.PlotWidget()
        self.plot_map.setLogMode(x=False, y=True)
        self.plot_map.setLabel('bottom', 'Frequency', units='Hz')
        self.plot_map.setLabel('left', 'τ (ms)')
        self.plot_map.showGrid(x=True, y=True)
        self.plot_map.addLegend()
        map_layout.addWidget(self.plot_map)
        tab.addTab(map_widget, "τ–f Map")
        self._draw_map_models()

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

        self.correlation_label = QLabel("Waveform R²: —")
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
                'noise_floor': float(det['noise_floor']),
                'std_noise':   float(det['std_noise']),
                'FPE_hz_region':   float(fpe) if fpe is not None else float('nan'),
                'svm_probability': det.get('svm_probability'),
            })

        self.model_results = {}
        self.btn_export.setEnabled(False)
        self._populate_click_table()
        self._refresh_map()
        self.file_label.setText(
            f"{os.path.basename(file_path)}\n{len(self.real_clicks)} clicks found (v6)"
        )
        if self.real_clicks:
            self.btn_run.setEnabled(True)
            self.btn_run_all.setEnabled(True)

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
            self._update_click_row(i)

    def _update_click_row(self, i):
        """Model columns of one click-table row (empty until the click is analysed)."""
        res = self.model_results.get(i) or {}
        v = res.get('vessel') or {}
        b = res.get('bubble') or {}
        vals = ["", "", ""]
        if v.get('ok'):
            vals[0] = f"{v['R_um']:.1f}"
            vals[1] = f"{self._vessel_length_mm(v):.2f}"
        if b.get('ok') and np.isfinite(b.get('tau_ratio_true_over_pred', np.nan)):
            vals[2] = f"{b['tau_ratio_true_over_pred']:.2f}"
        for k, text in enumerate(vals):
            self.click_table.setItem(i, 4 + k, QTableWidgetItem(text))

    def _selected_click_index(self):
        row = self.click_table.currentRow()
        if not self.click_table.selectedItems() or row < 0 or row >= len(self.real_clicks):
            return None
        return row

    def _on_click_selected(self):
        idx = self._selected_click_index()
        if idx is None:
            return
        self._discard_simulation()
        self._show_real_click(self.real_clicks[idx])
        if idx in self.model_results:
            self._display_result(idx)
        self._refresh_map()

    def _discard_simulation(self):
        """Clear the simulated (green) overlay and its result tables.

        Called whenever the selected click changes: the overlay belongs to the
        previous click. Results already computed stay cached in model_results
        and are shown again when their click is selected.
        """
        self.sim_result = None
        self.region_time.hide()
        self.curve_sim_time.setData([], [])
        self.curve_sim_freq.setData([], [])
        self.curve_bubble.setData([], [])
        self.table_sim.setRowCount(0)
        self.table_compare.setRowCount(0)
        self.correlation_label.setText("Waveform R²: —")
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
        t_full      = np.arange(3 * fft_size) / fs

        # Normalised to the click's envelope peak as v6 measures it (peak_amp), so
        # the click reads 1. The simulated click is drawn with the amplitude the
        # waveform fit gives it (least squares in the v6 region), so a click whose
        # peak is an initial impulse above its ring-down shows the model BELOW the
        # peak — that gap is the 'Impulse factor'. Absolute pressure is not
        # comparable (unknown source-to-mic coupling), so only shape is plotted.
        # (Before: the centre frame's |max|, which is not the click when the click
        # straddles two frames, and made the two curves' peaks disagree.)
        norm = click.get('peak_amp', 0.0)
        if not norm or norm <= 0:
            norm = np.max(np.abs(sig_full)) + 1e-30
        self._real_norm = norm
        self.curve_real_time.setData(t_full, sig_full / norm)
        self.plot_time.plot_widget.setXRange(0, 3 * frame_dur, padding=0)

    # =========================================================================
    # Model comparison (click_model_comparison) — single click and batch
    # =========================================================================

    def _click_job(self, idx):
        """(index, frames, frame_idx, noise_floor, std_noise) for one click."""
        click = self.real_clicks[idx]
        pd = self.paudio_data
        n_rows = min(len(pd['fft_data']), len(pd['phase_data']))

        def frame(row):
            if row is None or not (0 <= row < n_rows):
                return None
            return (pd['fft_data'][row], pd['phase_data'][row])

        frames = (frame(click['prev_row']), frame(click['row_idx']), frame(click['next_row']))
        return (idx, frames, click['frame_idx'], click['noise_floor'], click['std_noise'])

    def _run_simulation(self):
        idx = self._selected_click_index()
        if idx is None or not self.paudio_data:
            QMessageBox.information(self, "Select a click",
                                    "Load a recording and select a click to compare with the models.")
            return
        self._launch_comparison([self._click_job(idx)], batch=False)

    def _analyze_all_clicks(self):
        if not self.real_clicks:
            return
        self._launch_comparison([self._click_job(i) for i in range(len(self.real_clicks))], batch=True)

    def _launch_comparison(self, jobs, batch):
        if self._compare_thread is not None:
            return
        self.btn_run.setEnabled(False)
        self.btn_run_all.setEnabled(False)

        label = (f"Comparing {len(jobs)} clicks with the physical models…" if batch
                 else "Comparing the click with the physical models…")
        self.progress_dialog = QProgressDialog(label, "Cancel" if batch else None, 0, len(jobs), self)
        self.progress_dialog.setWindowTitle("Model comparison")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)
        self.progress_dialog.show()

        worker = ModelComparisonWorker(jobs)
        thread = QThread(self)
        worker.moveToThread(thread)
        # Bound methods only (see _launch_click_detector): slots must run on the GUI thread.
        thread.started.connect(worker.run)
        worker.result.connect(self._on_compare_result)
        worker.progress.connect(self._on_compare_progress)
        worker.finished.connect(self._on_compare_finished)
        worker.error.connect(self._on_compare_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        if batch:
            self.progress_dialog.canceled.connect(worker.request_stop)
        self._compare_thread = thread
        self._compare_worker = worker
        thread.start()

    def _on_compare_result(self, idx, res):
        self.model_results[idx] = res
        self._update_click_row(idx)
        if idx == self._selected_click_index():
            self._display_result(idx)

    def _on_compare_progress(self, done, total):
        if getattr(self, 'progress_dialog', None) is not None:
            self.progress_dialog.setValue(done)

    def _end_comparison(self):
        if getattr(self, 'progress_dialog', None) is not None:
            self.progress_dialog.close()
        self._compare_thread = None
        self._compare_worker = None
        self.btn_run.setEnabled(bool(self.real_clicks))
        self.btn_run_all.setEnabled(bool(self.real_clicks))
        self.btn_export.setEnabled(bool(self.model_results))
        self._refresh_map()

    def _on_compare_finished(self):
        self._end_comparison()

    def _on_compare_error(self, error_msg):
        self._end_comparison()
        QMessageBox.critical(self, "Model comparison error", f"An error occurred:\n{error_msg}")

    def _on_model_changed(self, *_):
        for i in range(len(self.real_clicks)):
            self._update_click_row(i)
        idx = self._selected_click_index()
        if idx is not None and idx in self.model_results:
            self._display_result(idx)
        self._refresh_map()

    def _vessel_length_mm(self, v):
        """L for the current wall parameters (E, h) — the only model output they change."""
        import vessel_resonance as vr
        return vr.length_from_frequency(v['f_true_hz'], v['R_um'] * 1e-6,
                                        h=self.wall_spinbox.value() * 1e-6,
                                        E=self.young_spinbox.value() * 1e9) * 1e3

    @staticmethod
    def _fill_table(table, rows):
        table.setRowCount(len(rows))
        for i, (name, value) in enumerate(rows):
            table.setItem(i, 0, QTableWidgetItem(name))
            table.setItem(i, 1, QTableWidgetItem(value))

    def _display_result(self, idx):
        """Plots and tables for one analysed click, for the selected model."""
        res = self.model_results.get(idx) or {}
        model = self.model_combo.currentData()
        real = res.get('real')
        r = res.get(model) or {}
        if real is None or not r.get('ok'):
            reason = r.get('reason', 'the click could not be reconstructed') if real else \
                'the click could not be reconstructed'
            self._fill_table(self.table_sim, [("Model", self.model_combo.currentText()),
                                              ("Not available", reason)])
            self.table_compare.setRowCount(0)
            self.curve_sim_time.setData([], [])
            self.curve_sim_freq.setData([], [])
            self.correlation_label.setText("Waveform R²: —")
            return

        pd = self.paudio_data
        fs = pd['fs']

        # Time domain: the simulated click after the SAME chain, on the real click's axis.
        sim = r['sim_signal']
        t = (np.arange(len(sim)) + real['layout_offset']) / fs
        norm = getattr(self, '_real_norm', None) or real['peak_amp']
        self.curve_sim_time.setData(t, sim / norm)
        # Zoom on the click: it lasts ~0.1–1 ms inside a 7.68 ms, 3-frame axis.
        # The shaded band is the v6 click region, where the waveform R² is computed.
        lo, hi = r['window']
        self.region_time.setRegion(((lo + real['layout_offset']) / fs,
                                    (hi - 1 + real['layout_offset']) / fs))
        self.region_time.show()
        self.plot_time.plot_widget.setXRange((lo + real['layout_offset']) / fs - 0.3e-3,
                                             (hi + real['layout_offset']) / fs + 0.6e-3, padding=0)

        band = r['sim_fft_norm'][pd['bin_start']:pd['bin_start'] + pd['num_bins']]
        peak = np.max(band) if len(band) else 0.0
        self.curve_sim_freq.setData(pd['freq_axis'], band / peak if peak > 0 else band)

        b = res.get('bubble') or {}
        if b.get('ok'):
            R_um = b['bubble_R'] * 1e6
            self.curve_bubble.setData(b['bubble_t'], R_um)
            # R0 is ~40–165 µm and the ring-down lasts ~6τ ≈ 0.2–1.5 ms: fixed axes
            # (0–100 µm, 0–2.56 ms) put the curve off-screen or squash it flat.
            pw = self.plot_bubble.plot_widget
            span = max(np.max(np.abs(R_um - b['R0_um'])), 1e-3) * 1.3
            pw.setLimits(xMin=0, xMax=float(b['bubble_t'][-1]),
                         yMin=b['R0_um'] - 5 * span, yMax=b['R0_um'] + 5 * span)
            pw.setXRange(0, float(b['bubble_t'][-1]), padding=0)
            pw.setYRange(b['R0_um'] - span, b['R0_um'] + span, padding=0)
            # Show the derived R0 in the (read-only) R0 control.
            for w_ in (self.r0_spinbox, self.r0_slider):
                w_.blockSignals(True)
            self.r0_spinbox.setValue(b['R0_um'])
            self.r0_slider.setValue(int(round(b['R0_um'])))
            for w_ in (self.r0_spinbox, self.r0_slider):
                w_.blockSignals(False)
        else:
            self.curve_bubble.setData([], [])

        def ms(x):
            return f"{x:.3f} ms" if x is not None and np.isfinite(x) and x > 0 else "N/A"

        def khz(x):
            return f"{x / 1000:.1f} kHz" if x is not None and np.isfinite(x) else "N/A"

        conv = (f"converged ({r['iterations']} it.)" if r['converged']
                else f"not converged — best of {r['iterations']} it.")
        if model == 'vessel':
            sim_rows = [
                ("Model", "Xylem vessel (Dutta 2022)"),
                ("f (calibrated)", khz(r['f_true_hz'])),
                ("τ (calibrated)", ms(r['tau_true_ms'])),
                ("Q = π·f·τ", f"{r['Q']:.1f}"),
                ("Vessel radius R", f"{r['R_um']:.1f} µm"),
                ("Element length L", f"{self._vessel_length_mm(r):.2f} mm"),
                ("Calibration", conv),
            ]
        else:
            sim_rows = [
                ("Model", "Free bubble (Minnaert)"),
                ("R₀ (from f)", f"{r['R0_um']:.1f} µm"),
                ("f₀", khz(r['f_true_hz'])),
                ("κ (polytropic)", f"{r['kappa']:.2f}"),
                ("δ rad / vis / th", f"{r['delta_rad']:.3f} / {r['delta_vis']:.3f} / {r['delta_th']:.3f}"),
                ("τ predicted", ms(r['tau_pred_ms'])),
                ("Q predicted", f"{r['Q_pred']:.1f}"),
                ("Calibration (f)", conv),
            ]
        if r.get('below_resolution'):
            sim_rows.append(("⚠ Resolution", "real τ below ~0.06 ms: τ-based results unreliable"))
        self._fill_table(self.table_sim, sim_rows)

        compare_rows = [
            ("τ real (v6)", ms(real['tau_ms'])),
            ("τ simulated (v6)", ms(r['sim_tau_ms'])),
            ("f real (click region)", khz(real['f_region_hz'])),
            ("f simulated (click region)", khz(r['sim_f_hz'])),
            ("Waveform R² (v6 region)", f"{r['r2_wave']:.3f}"),
            ("Shape correlation r", f"{r['r_wave']:.3f}"),
            ("Impulse factor", f"{r['impulse_factor']:.2f}"),
            ("Envelope corr.", f"{r['env_corr']:.3f}"),
        ]
        if model == 'bubble':
            ratio = r.get('tau_ratio_real_over_sim', float('nan'))
            compare_rows.append(("τ real / τ bubble", f"{ratio:.2f}" if np.isfinite(ratio) else "N/A"))
            if 'Q_real' in r:
                compare_rows.append(("τ calibrated / τ predicted", f"{r['tau_ratio_true_over_pred']:.2f}"))
                compare_rows.append(("Q real / Q bubble", f"{r['Q_real']:.1f} / {r['Q_pred']:.1f}"))
        self._fill_table(self.table_compare, compare_rows)
        self.correlation_label.setText(f"Waveform R²: {r['r2_wave']:.3f}   ·   r: {r['r_wave']:.3f}")

        self.sim_result = res
        self.btn_pdf.setEnabled(bool(b.get('ok')))

    # ── τ–f map ────────────────────────────────────────────────────────────

    def _draw_map_models(self):
        """Static model curves: bubble τ(f) and vessel iso-radius lines."""
        import click_model_comparison as cmc
        import vessel_resonance as vr
        import rayleigh_plesset as rp

        f_grid = np.linspace(18_000, 90_000, 60)
        tau_bubble = []
        for f in f_grid:
            R0 = cmc._bubble_R0_for_frequency(f)
            tau_bubble.append(rp.bubble_linear_properties(R0)['tau'] * 1e3)
        self.plot_map.plot(f_grid, np.array(tau_bubble), name="Free bubble τ(f)",
                           pen={'color': '#2196F3', 'width': 2})
        for R_um in (10, 20, 30, 40, 50):
            tau_ms = vr.settling_time(R_um * 1e-6) * 1e3
            self.plot_map.plot([18_000, 90_000], [tau_ms, tau_ms],
                               pen=pg.mkPen('#8d6e63', width=1, style=Qt.DashLine))
            label = pg.TextItem(f"vessel R={R_um} µm", color='#6d4c41', anchor=(0, 1))
            label.setPos(18_500, np.log10(tau_ms))
            self.plot_map.addItem(label)
        self._map_scatter = self.plot_map.plot([], [], pen=None, symbol='o', symbolSize=7,
                                               symbolBrush='#e53935', name="Clicks (calibrated)")
        self._map_selected = self.plot_map.plot([], [], pen=None, symbol='o', symbolSize=13,
                                                symbolBrush=None, symbolPen=pg.mkPen('k', width=2))

    def _refresh_map(self):
        xs, ys, sel = [], [], None
        chosen = self._selected_click_index()
        for i, res in self.model_results.items():
            v = (res or {}).get('vessel') or {}
            if v.get('ok') and v['tau_true_ms'] > 0:
                xs.append(v['f_true_hz'])
                ys.append(v['tau_true_ms'])
                if i == chosen:
                    sel = (v['f_true_hz'], v['tau_true_ms'])
        self._map_scatter.setData(xs, ys)
        self._map_selected.setData([sel[0]] if sel else [], [sel[1]] if sel else [])

    # ── export ─────────────────────────────────────────────────────────────

    def _export_results_csv(self):
        if not self.model_results:
            return
        import csv
        import click_model_comparison as cmc

        stem = os.path.splitext(os.path.basename(getattr(self, '_load_file_path', 'recording')))[0]
        start_dir = self.settings_manager.get_last_directory("chem_sim_export")
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export model comparison", os.path.join(start_dir, f"{stem}_model_comparison.csv"),
            "CSV Files (*.csv)")
        if not file_path:
            return
        columns = cmc.CSV_COLUMNS + ['svm_probability', 'wall_E_GPa', 'wall_h_um']
        try:
            with open(file_path, 'w', newline='') as fh:
                writer = csv.DictWriter(fh, fieldnames=columns)
                writer.writeheader()
                for i in sorted(self.model_results):
                    click = self.real_clicks[i]
                    res = self.model_results[i]
                    row = cmc.result_to_row(res, file=stem, frame_idx=click['frame_idx'],
                                            timestamp_s=click['timestamp'])
                    v = res.get('vessel') or {}
                    if v.get('ok'):
                        row['vessel_L_mm'] = self._vessel_length_mm(v)
                    row['svm_probability'] = click.get('svm_probability')
                    row['wall_E_GPa'] = self.young_spinbox.value()
                    row['wall_h_um'] = self.wall_spinbox.value()
                    writer.writerow(row)
            self.settings_manager.set_last_directory("chem_sim_export", file_path)
            QMessageBox.information(self, "Done", f"{len(self.model_results)} clicks exported to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export error", f"Could not write the CSV:\n{e}")

    def _generate_report(self):
        """
        PDF from report_acoustic, for the free-bubble model at the R0 calibrated
        on the selected click (the report's content is still the bubble-only one).
        """
        b = (self.sim_result or {}).get('bubble') or {}
        if not b.get('ok'):
            QMessageBox.warning(self, "No Data", "Analyse a click first.")
            return
        start_dir = self.settings_manager.get_last_directory("chem_sim_report")
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF Report", os.path.join(start_dir, "report_acoustic.pdf"), "PDF Files (*.pdf)"
        )
        if not file_path:
            return
        try:
            from chemical_simulators.run_acoustic_simulation import run_simulation
            from chemical_simulators.report_acoustic import generate_report
            sim = run_simulation(R0=b['R0_um'] * 1e-6, P_inf=self.pinf_spinbox.value() * 1e6,
                                 distance_m=self.dist_spinbox.value() * 0.01)
            generate_report(simulation_result=sim, output_path=file_path)
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
        self.plot_map.setBackground(bg)
        self.plot_map.getAxis("bottom").setTextPen(fg)
        self.plot_map.getAxis("left").setTextPen(fg)

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