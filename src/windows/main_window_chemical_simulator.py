"""
Finestra principale del simulatore chimico acustico PlantLeaf.
Modella la cavitazione xilematica e genera click ultrasonici sintetici.
"""

import numpy as np
import os
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QDoubleSpinBox, QSlider, QGroupBox,
    QFileDialog, QMessageBox, QProgressDialog, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy, QFrame
)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QIcon, QAction, QFont

from core.settings_manager import SettingsManager
from core.font_manager import FontManager
from core.layout_manager import LayoutManager
from core.theme_manager import ThemeManager
from config.app_config import AppConfig
from plotting.plot_manager import BasePlotWidget


# =============================================================================
# WORKER THREAD PER LA SIMULAZIONE
# =============================================================================

class SimulationWorker(QObject):
    """Worker per eseguire la simulazione in un thread separato."""
    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(int)

    def __init__(self, R0, P_inf, distance_m):
        super().__init__()
        self.R0 = R0
        self.P_inf = P_inf
        self.distance_m = distance_m

    def run(self):
        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'chemical_simulators'))
            from run_acoustic_simulation import run_simulation
            self.progress.emit(30)
            result = run_simulation(R0=self.R0, P_inf=self.P_inf, distance_m=self.distance_m)
            self.progress.emit(100)
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")


# =============================================================================
# FINESTRA PRINCIPALE
# =============================================================================

class MainWindowChemicalSimulator(QMainWindow):
    """Finestra del simulatore chimico acustico PlantLeaf."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Manager
        self.settings_manager = SettingsManager()
        self.font_manager = FontManager(self.settings_manager.settings)
        self.layout_manager = LayoutManager(self.font_manager)
        self.theme_manager = ThemeManager(self.settings_manager.settings, self.font_manager)

        self.setWindowTitle("🧪 Audio Chemical Simulator")
        self.setWindowIcon(QIcon(AppConfig.LOGO_DIR))
        self.setMinimumSize(1100, 650)

        # Stato simulazione
        self.sim_result = None
        self.real_clicks_tau = []
        self.sim_thread = None
        self.sim_worker = None

        # Costruisci UI
        self._setup_ui()
        self._setup_menubar()
        self._setup_toolbar()

        # Applica tema salvato
        self._load_saved_settings()
        self.setStatusBar(None)

        self.showMaximized()

    # =========================================================================
    # SETUP UI
    # =========================================================================

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # --- Pannello sinistra: controlli ---
        splitter.addWidget(self._build_controls_panel())

        # --- Centro: grafici ---
        splitter.addWidget(self._build_plots_panel())

        # --- Destra: risultati ---
        splitter.addWidget(self._build_results_panel())

        splitter.setSizes([260, 620, 260])

    def _build_controls_panel(self):
        """Pannello sinistra con i controlli di tuning."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        layout.setContentsMargins(4, 4, 4, 4)

        # --- Titolo ---
        title = QLabel("Simulation Parameters")
        title.setAlignment(Qt.AlignCenter)
        title.setObjectName("titleLabel")
        layout.addWidget(title)

        # --- Gruppo parametri fisici ---
        phys_group = QGroupBox("Physical Parameters")
        phys_layout = QVBoxLayout(phys_group)
        phys_layout.setSpacing(6)

        # R0
        phys_layout.addWidget(QLabel("Bubble radius R₀ [µm]:"))
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

        # P_inf
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

        # Distanza
        phys_layout.addWidget(QLabel("Distance bubble→mic [cm]:"))
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

        # --- Label stress idrico ---
        self.stress_label = QLabel("Stress: Well hydrated")
        self.stress_label.setAlignment(Qt.AlignCenter)
        self.stress_label.setStyleSheet("font-weight: bold; padding: 4px;")
        layout.addWidget(self.stress_label)
        self.pinf_spinbox.valueChanged.connect(self._update_stress_label)

        # --- Separatore ---
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)

        # --- Pulsante Run Simulation ---
        self.btn_run = QPushButton("▶  Run Simulation")
        self.btn_run.setMinimumHeight(40)
        self.btn_run.setObjectName("mainButton")
        self.btn_run.clicked.connect(self._run_simulation)
        layout.addWidget(self.btn_run)

        # --- Pulsante Load .paudio ---
        self.btn_load = QPushButton("📂  Load .paudio File")
        self.btn_load.setMinimumHeight(35)
        self.btn_load.setObjectName("mainButton")
        self.btn_load.clicked.connect(self._load_paudio)
        layout.addWidget(self.btn_load)

        # --- Pulsante Generate PDF ---
        self.btn_pdf = QPushButton("📄  Generate PDF Report")
        self.btn_pdf.setMinimumHeight(35)
        self.btn_pdf.setObjectName("mainButton")
        self.btn_pdf.setEnabled(False)
        self.btn_pdf.clicked.connect(self._generate_report)
        layout.addWidget(self.btn_pdf)

        # --- Info file caricato ---
        self.file_label = QLabel("No .paudio file loaded")
        self.file_label.setAlignment(Qt.AlignCenter)
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self.file_label)

        layout.addStretch(1)
        return panel

    def _build_plots_panel(self):
        """Pannello centrale con i grafici."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        tab = QTabWidget()
        layout.addWidget(tab)

        # --- Tab 1: Time domain ---
        time_widget = QWidget()
        time_layout = QVBoxLayout(time_widget)
        self.time_info_label = QLabel("Time Domain — Simulated Click")
        self.time_info_label.setAlignment(Qt.AlignCenter)
        time_layout.addWidget(self.time_info_label)
        self.plot_time = BasePlotWidget(
            x_label="Time", y_label="Pressure",
            x_range=(0, 1e-4), y_range=(-1, 1),
            x_min=0, x_max=1e-3, y_min=-10, y_max=10,
            unit_x="s", unit_y="Pa", parent=self
        )
        self.curve_sim_time = self.plot_time.plot_widget.plot(name="Simulated click")
        self.curve_real_time = self.plot_time.plot_widget.plot(
            name="Real click", pen={'color': 'r', 'width': 1.5}
        )
        self.plot_time.plot_widget.showGrid(x=True, y=True)
        time_layout.addWidget(self.plot_time)
        tab.addTab(time_widget, "Time Domain")

        # --- Tab 2: Frequency domain ---
        freq_widget = QWidget()
        freq_layout = QVBoxLayout(freq_widget)
        self.freq_info_label = QLabel("Frequency Domain — Spectrum 20–80 kHz")
        self.freq_info_label.setAlignment(Qt.AlignCenter)
        freq_layout.addWidget(self.freq_info_label)
        self.plot_freq = BasePlotWidget(
            x_label="Frequency", y_label="Amplitude",
            x_range=(20000, 80000), y_range=(0, 1),
            x_min=19000, x_max=81000, y_min=0, y_max=10,
            unit_x="Hz", unit_y="Pa", parent=self
        )
        self.curve_sim_freq = self.plot_freq.plot_widget.plot(name="Simulated spectrum")
        self.curve_real_freq = self.plot_freq.plot_widget.plot(
            name="Real spectrum", pen={'color': 'r', 'width': 1.5}
        )
        self.plot_freq.plot_widget.showGrid(x=True, y=True)
        freq_layout.addWidget(self.plot_freq)
        tab.addTab(freq_widget, "Frequency Domain")

        # --- Tab 3: Bubble dynamics ---
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

        # --- Tab 4: τ comparison ---
        tau_widget = QWidget()
        tau_layout = QVBoxLayout(tau_widget)
        tau_layout.addWidget(QLabel("τ distribution: simulated vs measured"))
        self.plot_tau = BasePlotWidget(
            x_label="τ", y_label="Count",
            x_range=(0, 2), y_range=(0, 10),
            x_min=0, x_max=5, y_min=0, y_max=50,
            unit_x="ms", unit_y="", parent=self
        )
        self.plot_tau.plot_widget.showGrid(x=True, y=True)
        tau_layout.addWidget(self.plot_tau)
        tab.addTab(tau_widget, "τ Comparison")

        return panel

    def _build_results_panel(self):
        """Pannello destra con i risultati diagnostici."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)
        layout.setContentsMargins(4, 4, 4, 4)

        title = QLabel("Diagnostics")
        title.setAlignment(Qt.AlignCenter)
        title.setObjectName("titleLabel")
        layout.addWidget(title)

        # --- Tabella parametri simulati ---
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

        # --- Tabella confronto con reale ---
        real_group = QGroupBox("Comparison with Real Data")
        real_layout = QVBoxLayout(real_group)
        self.table_compare = QTableWidget(0, 2)
        self.table_compare.setHorizontalHeaderLabels(["Metric", "Value"])
        self.table_compare.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_compare.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table_compare.verticalHeader().setVisible(False)
        self.table_compare.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_compare.setAlternatingRowColors(True)
        real_layout.addWidget(self.table_compare)
        layout.addWidget(real_group)

        # --- Label P∞ stimato ---
        self.pinf_estimated_label = QLabel("Estimated P∞: —")
        self.pinf_estimated_label.setAlignment(Qt.AlignCenter)
        self.pinf_estimated_label.setWordWrap(True)
        self.pinf_estimated_label.setStyleSheet("font-weight: bold; padding: 6px;")
        layout.addWidget(self.pinf_estimated_label)

        layout.addStretch(1)
        return panel

    # =========================================================================
    # MENUBAR E TOOLBAR
    # =========================================================================

    def _setup_menubar(self):
        menubar = self.menuBar()
        font = QFont()
        font.setPointSize(12)
        menubar.setFont(font)

        # File
        file_menu = menubar.addMenu("File")
        self.actionHome = QAction("Home", self)
        self.actionHome.triggered.connect(self._go_home)
        file_menu.addAction(self.actionHome)
        file_menu.addSeparator()
        self.actionLoadPaudio = QAction("Load .paudio File...", self)
        self.actionLoadPaudio.setShortcut("Ctrl+O")
        self.actionLoadPaudio.triggered.connect(self._load_paudio)
        file_menu.addAction(self.actionLoadPaudio)
        self.actionGeneratePDF = QAction("Generate PDF Report...", self)
        self.actionGeneratePDF.setShortcut("Ctrl+P")
        self.actionGeneratePDF.triggered.connect(self._generate_report)
        file_menu.addAction(self.actionGeneratePDF)

        # Simulation
        sim_menu = menubar.addMenu("Simulation")
        self.actionRunSim = QAction("Run Simulation", self)
        self.actionRunSim.setShortcut("Ctrl+R")
        self.actionRunSim.triggered.connect(self._run_simulation)
        sim_menu.addAction(self.actionRunSim)

        # Settings
        settings_menu = menubar.addMenu("Settings")
        theme_menu = settings_menu.addMenu("Theme")
        themes = [
            ("Dark", "dark.css"), ("Dark Green", "dark_green.css"),
            ("Dark Blue", "dark_blue.css"), ("Dark Amber", "dark_amber.css"),
            ("Light", "light.css"), ("Light Green", "light_green.css"),
            ("Light Blue", "light_blue.css"), ("Light Amber", "light_amber.css"),
        ]
        for name, file in themes:
            action = QAction(name, self)
            action.triggered.connect(lambda checked, f=file: self._update_style(theme_name=f))
            theme_menu.addAction(action)

        font_menu = settings_menu.addMenu("Font Scale")
        for label, scale in [("Very Small", 1.15), ("Small", 1.25), ("Medium", 1.35), ("Large", 1.4)]:
            action = QAction(label, self)
            action.triggered.connect(lambda checked, s=scale: self._update_style(font_scale=s))
            font_menu.addAction(action)

        # About
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

        pdf_action = QAction("📄 PDF Report", self)
        pdf_action.triggered.connect(self._generate_report)
        toolbar.addAction(pdf_action)

    # =========================================================================
    # LOGICA
    # =========================================================================

    def _update_stress_label(self, p_mpa):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'chemical_simulators'))
        try:
            from acoustic_parameters import XylemPressure
            label = XylemPressure.get_stress_label(p_mpa * 1e6)
            self.stress_label.setText(f"Stress: {label}")
        except Exception:
            self.stress_label.setText("")

    def _run_simulation(self):
        """Lancia la simulazione nel thread separato."""
        R0 = self.r0_spinbox.value() * 1e-6
        P_inf = self.pinf_spinbox.value() * 1e6
        distance_m = self.dist_spinbox.value() * 0.01

        self.progress_dialog = QProgressDialog("Running simulation...", None, 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(10)
        self.progress_dialog.show()

        self.sim_thread = QThread()
        self.sim_worker = SimulationWorker(R0, P_inf, distance_m)
        self.sim_worker.moveToThread(self.sim_thread)
        self.sim_thread.started.connect(self.sim_worker.run)
        self.sim_worker.finished.connect(self._on_simulation_finished)
        self.sim_worker.error.connect(self._on_simulation_error)
        self.sim_worker.progress.connect(self.progress_dialog.setValue)
        self.sim_worker.finished.connect(self.sim_thread.quit)
        self.sim_worker.finished.connect(self.sim_worker.deleteLater)
        self.sim_thread.finished.connect(self.sim_thread.deleteLater)
        self.sim_thread.start()

    def _on_simulation_finished(self, result):
        self.progress_dialog.close()
        self.sim_result = result
        self._update_plots(result)
        self._update_diagnostics(result)
        self.btn_pdf.setEnabled(True)
        print("✅ Simulation completed")

    def _on_simulation_error(self, error_msg):
        self.progress_dialog.close()
        QMessageBox.critical(self, "Simulation Error", f"An error occurred:\n{error_msg}")
        print(f"❌ Simulation error: {error_msg}")

    def _update_plots(self, result):
        """Aggiorna tutti i grafici con i risultati della simulazione."""
        bubble = result['bubble']
        propagation = result['propagation']
        plantleaf = result['plantleaf']

        # Time domain
        t = bubble['t']
        signal = propagation['signal']
        self.curve_sim_time.setData(t, signal)
        self.time_info_label.setText(
            f"Time Domain — R₀={bubble['R0']*1e6:.0f} µm, "
            f"P∞={bubble['P_inf']/1e6:.2f} MPa"
        )

        # Frequency domain
        freq = plantleaf['freq']
        spec = plantleaf['spectrum']
        self.curve_sim_freq.setData(freq, spec)

        # Bubble dynamics
        R_um = bubble['R'] * 1e6
        t_us = t * 1e6
        self.curve_bubble.setData(t_us, R_um)

        # Applica tema ai plot
        for plot, curve in [
            (self.plot_time, self.curve_sim_time),
            (self.plot_freq, self.curve_sim_freq),
            (self.plot_bubble, self.curve_bubble),
        ]:
            self.theme_manager.apply_theme_to_plot(plot.plot_widget, curve)

    def _update_diagnostics(self, result):
        """Aggiorna la tabella dei parametri diagnostici."""
        diag = result['diagnostics']
        bubble = result['bubble']

        rows = [
            ("R₀", f"{bubble['R0']*1e6:.1f} µm"),
            ("P∞", f"{bubble['P_inf']/1e6:.2f} MPa"),
            ("Collapsed", "✅ Yes" if bubble['collapsed'] else "❌ No"),
            ("τ (simulated)", f"{diag['tau']*1000:.3f} ms" if diag['tau'] else "N/A"),
            ("SPR", f"{diag['SPR']:.2f}" if diag['SPR'] else "N/A"),
            ("Asymmetry", f"{diag['asymmetry']:.3f}" if diag['asymmetry'] else "N/A"),
            ("R_spectral", f"{diag['R_spectral']:.3f}" if diag['R_spectral'] else "N/A"),
            ("Peak amplitude", f"{diag['peak_amplitude']:.4f} Pa" if diag['peak_amplitude'] else "N/A"),
        ]

        self.table_sim.setRowCount(len(rows))
        for i, (param, value) in enumerate(rows):
            self.table_sim.setItem(i, 0, QTableWidgetItem(param))
            self.table_sim.setItem(i, 1, QTableWidgetItem(value))

        # Confronto con dati reali se disponibili
        if self.real_clicks_tau and diag['tau']:
            tau_sim_ms = diag['tau'] * 1000
            tau_meas_ms = np.array(self.real_clicks_tau) * 1000
            mean_meas = np.mean(tau_meas_ms)
            diff = abs(tau_sim_ms - mean_meas)
            rel_err = diff / mean_meas * 100 if mean_meas > 0 else 0

            compare_rows = [
                ("τ sim [ms]", f"{tau_sim_ms:.3f}"),
                ("τ meas mean [ms]", f"{mean_meas:.3f}"),
                ("Difference [ms]", f"{diff:.3f}"),
                ("Relative error", f"{rel_err:.1f}%"),
                ("Match (< 20%)", "✅ Yes" if rel_err < 20 else "❌ No"),
                ("N real clicks", str(len(self.real_clicks_tau))),
            ]
            self.table_compare.setRowCount(len(compare_rows))
            for i, (metric, value) in enumerate(compare_rows):
                self.table_compare.setItem(i, 0, QTableWidgetItem(metric))
                self.table_compare.setItem(i, 1, QTableWidgetItem(value))

            # Aggiorna grafico τ
            self._update_tau_plot(tau_sim_ms, tau_meas_ms)

    def _update_tau_plot(self, tau_sim_ms, tau_meas_ms):
        """Aggiorna il grafico di confronto τ."""
        import pyqtgraph as pg

        self.plot_tau.plot_widget.clear()

        all_tau = np.concatenate([[tau_sim_ms], tau_meas_ms])
        bins = np.linspace(np.min(all_tau) * 0.8, np.max(all_tau) * 1.2, 20)
        bin_width = bins[1] - bins[0]

        # Istogramma misurati (rosso)
        counts_meas, _ = np.histogram(tau_meas_ms, bins=bins)
        for i, count in enumerate(counts_meas):
            if count > 0:
                bar = pg.BarGraphItem(
                    x=[bins[i] + bin_width / 2], height=[count],
                    width=bin_width * 0.4, brush='#E53935', pen='#E53935'
                )
                self.plot_tau.plot_widget.addItem(bar)

        # Linea verticale τ simulato (blu)
        self.plot_tau.plot_widget.addLine(
            x=tau_sim_ms,
            pen={'color': '#2196F3', 'width': 2, 'style': Qt.DashLine},
            label=f"τ sim={tau_sim_ms:.3f} ms"
        )

    def _load_paudio(self):
        """Carica un file .paudio ed estrae i τ dai click rilevati."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open .paudio File", "",
            "PlantLeaf Audio (*.paudio);;All Files (*)"
        )
        if not file_path:
            return

        try:
            import struct, json, zlib
            with open(file_path, 'rb') as f:
                # Salta header 128 byte
                f.seek(128)
                data = f.read()

            # Cerca marker click data
            click_start = data.find(b'CLCK')
            if click_start < 0:
                self.file_label.setText(f"Loaded: {os.path.basename(file_path)}\n(no click data found)")
                self.real_clicks_tau = []
                return

            # Leggi click JSON
            size_bytes = data[click_start + 4: click_start + 8]
            compressed_size = struct.unpack('<I', size_bytes)[0]
            compressed_data = data[click_start + 8: click_start + 8 + compressed_size]
            click_json = zlib.decompress(compressed_data).decode('utf-8')
            clicks = json.loads(click_json)

            # Estrai durate come proxy di τ (in secondi)
            # PlantLeaf salva durata in FFT count; ogni FFT = 2.56 ms
            tau_list = []
            for click in clicks:
                duration_str = str(click.get('duration', ''))
                if 'FFT' in duration_str:
                    fft_count = int(duration_str.replace(' FFT', '').strip())
                    tau_ms = fft_count * 2.56 / 3  # stima τ come 1/3 della durata
                    tau_list.append(tau_ms / 1000.0)

            self.real_clicks_tau = tau_list
            self.file_label.setText(
                f"✅ {os.path.basename(file_path)}\n{len(clicks)} clicks, {len(tau_list)} τ values"
            )

            if self.sim_result:
                self._update_diagnostics(self.sim_result)

        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Could not load file:\n{str(e)}")
            self.file_label.setText("❌ Load error")

    def _generate_report(self):
        """Genera il PDF del report scientifico."""
        if not self.sim_result:
            QMessageBox.warning(self, "No Data", "Run a simulation first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF Report", "report_acoustic.pdf",
            "PDF Files (*.pdf);;All Files (*)"
        )
        if not file_path:
            return

        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'chemical_simulators'))
            from report_acoustic import generate_report

            tau_meas = self.real_clicks_tau if self.real_clicks_tau else None
            generate_report(
                simulation_result=self.sim_result,
                tau_measured=tau_meas,
                output_path=file_path
            )
            QMessageBox.information(self, "Report Generated", f"PDF saved to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Report Error", f"Could not generate report:\n{str(e)}")

    # =========================================================================
    # TEMA E IMPOSTAZIONI
    # =========================================================================

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