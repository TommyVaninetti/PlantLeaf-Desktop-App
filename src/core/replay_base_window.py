from PySide6.QtWidgets import QMainWindow, QMenu, QPushButton, QMessageBox, QSlider, QDoubleSpinBox, QLabel, QSizePolicy, QWidget, QHBoxLayout, QProgressDialog, QDialog, QVBoxLayout, QHBoxLayout, QMenuBar, QComboBox, QLayout, QFrame, QTableWidget, QTableWidgetItem, QHeaderView
from PySide6.QtGui import QIcon, QAction, QFont
from PySide6.QtCore import Qt, QSize, Signal


import os
import numpy as np
import pyqtgraph as pg
from scipy.optimize import curve_fit


from core.settings_manager import SettingsManager
from config.app_config import AppConfig
from core.font_manager import FontManager
from core.layout_manager import LayoutManager
from core.theme_manager import ThemeManager
from core.file_handler_mixin import FileHandlerMixin
from windows.ui.ui_MathDialog import Ui_QDialogMath
from components.time_input_widget import TimeInputWidget


class ReplayBaseWindow(FileHandlerMixin, QMainWindow):

    playback_speed_changed = Signal(float)  # Segnale per la velocità di riproduzione
    playback_position_changed = Signal(int)  # Segnale per la posizione di riproduzione
    started_playing = Signal()  # Segnale per l'inizio della riproduzione
    paused_playing = Signal()  # Segnale per la pausa della riproduzione

    def __init__(self, parent=None):
        self.settings_manager = SettingsManager()
        super().__init__(parent)

        # Inizializza manager base
        self.font_manager = FontManager(self.settings_manager.settings)
        self.layout_manager = LayoutManager(self.font_manager)
        self.theme_manager = ThemeManager(self.settings_manager.settings, self.font_manager)

        # Imposta icona della finestra
        self.setWindowIcon(QIcon(AppConfig.LOGO_DIR))

        # Carica impostazioni salvate per tema etc.
        #self._load_saved_settings() da chiamare dopo la creazione della finestra

        #imposta dimensioni iniziali della finestra e minime
        self.setMinimumSize(800, 600)  # Dimensioni minime della finestra
        self.resize(1200, 800)  # Dimensioni iniziali della finestra

        self.is_running = False  # Stato di riproduzione
        self.text_start = "Play"
        self.text_stop = "Pause"

        # Definisce la dimensione di uno step per la navigazione (in ms)
        self.time_step_ms = 100.0 # Default a 100ms (utile per voltage)


    def _load_saved_settings(self):
        """Carica tutte le impostazioni salvate e applica tema ai plot specifici"""
        saved_font_scale = self.font_manager.load_font_scale()
        self.font_manager.current_font_scale = saved_font_scale
        saved_theme = self.theme_manager.load_saved_theme()
        self.theme_manager.apply_theme(self, saved_theme)
        # Sovrascrive la grandezza del font dei pulsanti SOLO in questa finestra
        for widget in self.findChildren(QPushButton):
            font = widget.font()
            font.setPointSize(int(14 * self.font_manager.current_font_scale))  # Scegli la dimensione base che preferisci
            widget.setFont(font)

        # Applica tema ai plot appropriati basato sul tipo di finestra
        self._apply_theme_to_plots()
    
    def _apply_theme_to_plots(self):
        """Applica il tema a tutti i plot appropriati (voltage o audio)"""
        if not hasattr(self, 'theme_manager'):
            return
        
        # === VOLTAGE PLOTS ===
        if hasattr(self, 'voltage_plot') and hasattr(self, 'voltage_curve'):
            self.theme_manager.apply_theme_to_plot(
                plot_widget_name=self.voltage_plot.plot_widget, 
                plot_instance=self.voltage_curve
            )
        
        # === AUDIO PLOTS ===
        # Applica tema al plot FFT
        if hasattr(self, 'plot_widget_fft') and hasattr(self.plot_widget_fft, 'plot_widget'):
            self.theme_manager.apply_theme_to_plot(
                plot_widget_name=self.plot_widget_fft.plot_widget,
                plot_instance=self.fft_curve
            )
        else:
            print("⚠️ plot_widget_fft o fft_curve non trovati, salto applicazione tema FFT")
        
        # Applica tema al plot Time Domain
        if hasattr(self, 'plot_widget_time') and hasattr(self.plot_widget_time, 'plot_widget'):
            self.theme_manager.apply_theme_to_plot(
                plot_widget_name=self.plot_widget_time.plot_widget,
                plot_instance=self.time_curve
            )

    def update_style(self, theme_name=None, font_scale=None):
        """Aggiorna tema e/o scala font in modo centralizzato per tutti i tipi di plot"""
        if font_scale is not None:
            self.font_manager.save_font_scale(font_scale)
            self.font_manager.current_font_scale = font_scale
        if theme_name is not None:
            self.theme_manager.apply_theme(self, theme_name) #APPLICA FONT (con scala dinamica, uso questa funzione perché è collegata)
        else:
            self.theme_manager.apply_theme(self, self.theme_manager.current_theme) #APPLICA SIA FONT CHE TEMA
            # Sovrascrive la grandezza del font dei pulsanti SOLO in questa finestra
            for widget in self.findChildren(QPushButton):
                font = widget.font()
                font.setPointSize(int(14 * self.font_manager.current_font_scale))  # Scegli la dimensione base che preferisci
                widget.setFont(font)
        # Aggiorna toolbar
        #elimina toolbar esistente se presente
        self.setup_toolbar_style()  # Aggiorna la toolbar con il nuovo colore tema

        # Applica tema a tutti i plot appropriati (voltage o audio)
        self._apply_theme_to_plots()

        self.setStatusBar(None)



    def setup_menubar(self):
        menubar = self.menuBar()

        # Impostazioni grafiche
        from PySide6.QtGui import QFont
        font = QFont()
        font.setPointSize(12)
        font.setBold(False)
        menubar.setFont(font)

        # === FILE ===
        file_menu = menubar.addMenu("File")
        self.actionNewFile = QAction("New", self)
        self.actionNewFile.setShortcut("Ctrl+N")
        self.actionOpenFile = QAction("Open...", self)
        self.actionOpenFile.setShortcut("Ctrl+O")
        
        # ✅ NUOVO: Export Trimmed Region
        self.actionExportTrimmed = QAction("Export Trimmed Region...", self)
        self.actionExportTrimmed.setShortcut("Ctrl+T")
        
        file_menu.addActions([self.actionNewFile, self.actionOpenFile])
        file_menu.addSeparator()
        file_menu.addAction(self.actionExportTrimmed)

        # === ANALYSIS ===
        analysis_menu = menubar.addMenu("Analysis")
        self.actionStart = QAction("Start/Pause", self)
        self.actionStop = QAction("Stop", self)
        #self.actionAddBookmark = QAction("Add Bookmark", self)
        analysis_menu.addActions([self.actionStart, self.actionStop, 
                                #self.actionAddBookmark
                                ])
        # SOLO AUDIO
        if hasattr(self, 'plot_widget_fft'):
            analysis_menu.addSeparator()
            self.actionNormalizeFFT = QAction("Normalize FFT window", self)
            analysis_menu.addAction(self.actionNormalizeFFT)
            self.actionSpectralEnergy = QAction("Spectral Energy Analysis", self)
            self.actionSpectralEnergy.setToolTip("Analyze energy distribution across frequency bands (20-80 kHz)")
            analysis_menu.addAction(self.actionSpectralEnergy)
            
            # ✅ CLICK DETECTOR
            analysis_menu.addSeparator()
            self.actionClickDetector = QAction("🔍 Automatic Click Detector...", self)
            self.actionClickDetector.setShortcut("Ctrl+D")
            self.actionClickDetector.setToolTip(
                "Run automatic ultrasonic click detection algorithm\n"
                "4-stage pipeline: Energy → Spectral → Decay → Deduplication"
            )
            analysis_menu.addAction(self.actionClickDetector)

        self.actionMath = QAction("Select Region for Analysis", self)
        self.actionMath.setToolTip("Select a region in the plot to perform mathematical analysis")
        # Azione per aprire popup per aprire analisi salvate
        self.actionMathDialog = QAction("Analyze Region", self)
        self.actionOpenSavedAnalysis = QAction("Open Saved Analysis", self)

        #PER ORA SOLO VOLTAGE
        if hasattr(self, 'voltage_plot'):
            #aggiungi separatore
            analysis_menu.addSeparator()
            analysis_menu.addAction(self.actionMath)
            analysis_menu.addAction(self.actionMathDialog)
            analysis_menu.addAction(self.actionOpenSavedAnalysis)

        self.actionMathDialog.setEnabled(False)  # Disabilitato finché non si seleziona una regione

        # === SETTINGS ===
        settings_menu = menubar.addMenu("Settings")
        # --- Theme submenu ---
        theme_menu = QMenu("Theme", self)
        self.actionDark = QAction("Dark", self)
        self.actionDark_Green = QAction("Dark Green", self)
        self.actionDark_Blue = QAction("Dark Blue", self)
        self.actionDark_Amber = QAction("Dark Amber", self)
        self.actionLight = QAction("Light", self)
        self.actionLight_Green = QAction("Light Green", self)
        self.actionLight_Blue = QAction("Light Blue", self)
        self.actionLight_Amber = QAction("Light Amber", self)
        theme_menu.addActions([
            self.actionDark, self.actionDark_Green, self.actionDark_Blue, self.actionDark_Amber,
            self.actionLight, self.actionLight_Green, self.actionLight_Blue, self.actionLight_Amber
        ])
        # --- Font submenu ---
        font_menu = QMenu("Font Size", self)
        self.actionVery_Small = QAction("Very Small", self)
        self.actionSmall = QAction("Small", self)
        self.actionMedium = QAction("Medium", self)
        self.actionLarge = QAction("Large", self)
        font_menu.addActions([self.actionVery_Small, self.actionSmall, self.actionMedium, self.actionLarge])
        # --- Add submenus to settings ---
        settings_menu.addMenu(theme_menu)
        settings_menu.addMenu(font_menu)

        # === HELP ===
        help_menu = menubar.addMenu("Help")
        self.actionAbout = QAction("About", self)
        self.actionDocumentation = QAction("Documentation", self)
        help_menu.addActions([self.actionAbout, self.actionDocumentation])

         # === INFO ===
        info_menu = menubar.addMenu("About")
        self.actionHome = QAction("Home", self)
        menubar.addSeparator()
        self.actionInfo = QAction("Info", self)
        self.actionVersion = QAction("Version", self)
        self.actionWebSite = QAction("Website", self)
        self.actionLicense = QAction("License", self)
        self.actionQuit = QAction("Quit", self)
        self.actionQuit.triggered.connect(self.close)  # Chiude la finestra
        info_menu.addActions([
            self.actionHome, self.actionInfo, self.actionVersion, self.actionWebSite, self.actionLicense, self.actionQuit
        ])

        # === COLLEGA LE AZIONI AGLI SLOT ===
        self.actionHome.triggered.connect(self.go_to_home)
        self.actionNewFile.triggered.connect(self.new_file_action)
        self.actionOpenFile.triggered.connect(self.open_file_action)
        self.actionExportTrimmed.triggered.connect(self.open_trim_dialog)  # ✅ NUOVO
        #self.actionSave.triggered.connect(self.save_file_action)
        self.actionStart.triggered.connect(self.toggle_state_playing)
        self.actionStop.triggered.connect(self.clear_history)
        #self.actionAddBookmark.triggered.connect(self.add_bookmark_action)
        self.actionDark.triggered.connect(lambda: self.update_style(theme_name="dark.css"))
        self.actionDark_Green.triggered.connect(lambda: self.update_style(theme_name="dark_green.css"))
        self.actionDark_Blue.triggered.connect(lambda: self.update_style(theme_name="dark_blue.css"))
        self.actionDark_Amber.triggered.connect(lambda: self.update_style(theme_name="dark_amber.css"))
        self.actionLight.triggered.connect(lambda: self.update_style(theme_name="light.css"))
        self.actionLight_Green.triggered.connect(lambda: self.update_style(theme_name="light_green.css"))
        self.actionLight_Blue.triggered.connect(lambda: self.update_style(theme_name="light_blue.css"))
        self.actionLight_Amber.triggered.connect(lambda: self.update_style(theme_name="light_amber.css"))
        self.actionVery_Small.triggered.connect(lambda: self.update_style(font_scale=1.15))
        self.actionSmall.triggered.connect(lambda: self.update_style(font_scale=1.25))
        self.actionMedium.triggered.connect(lambda: self.update_style(font_scale=1.35))
        self.actionLarge.triggered.connect(lambda: self.update_style(font_scale=1.4))
        self.actionAbout.triggered.connect(self.about_action)
        self.actionDocumentation.triggered.connect(self.documentation_action)
        self.actionInfo.triggered.connect(self.info_action)
        self.actionVersion.triggered.connect(self.version_action)
        self.actionWebSite.triggered.connect(self.website_action)
        self.actionLicense.triggered.connect(self.license_action)
        self.actionMath.triggered.connect(self.prepare_for_math_dialog)
        self.actionMathDialog.triggered.connect(self.open_math_dialog)
        self.actionOpenSavedAnalysis.triggered.connect(self.on_open_saved_analysis_triggered)

        #SOLO AUDIO
        if hasattr(self, 'plot_widget_fft'):
            self.actionNormalizeFFT.triggered.connect(self.normalize_fft_window)
            self.actionSpectralEnergy.triggered.connect(self.show_spectral_energy_analysis)


    ###### AZIONI MENUBAR ######

    def go_to_home(self):
        """Versione sicura per tornare alla home"""   
        # Ferma la riproduzione in corso
        self.pause_playback()
        
        # Salva le impostazioni correnti
        self._save_current_settings()    
                #se si è in schermo intero, torna a finestra normale
        if self.isFullScreen():
            self.showNormal()
            
        from windows.main_window_home import MainWindowHome
        self.home_window = MainWindowHome()
        self.layout_manager.center_window_on_screen(self.home_window)
        self.home_window.show()
        self.close()


    def new_file_action(self):
        """Azione per creare un nuovo file basata sul tipo di finestra"""
        if hasattr(self, 'voltage_plot'):
            # Finestra voltage - apri main window voltage
            from windows.main_window_voltage import MainWindowVoltage
            self.voltage_window = MainWindowVoltage()
            self.layout_manager.center_window_on_screen(self.voltage_window)
            self.voltage_window.show()
            self.close()
        elif hasattr(self, 'plot_widget_fft'):
            # Finestra audio - apri main window audio
            from windows.main_window_audio import MainWindowAudio
            self.audio_window = MainWindowAudio()
            self.layout_manager.center_window_on_screen(self.audio_window)
            self.audio_window.show()
            self.close()
        else: 
            QMessageBox.warning(self, "Warning", "Cannot determine window type. Unable to create new file.")
    

    def add_bookmark_action(self):
        """Azione per aggiungere un segnalibro"""
        print("Azione 'Aggiungi Segnalibro' non implementata")

    def open_file_with_default_app(self,filepath):
        import platform
        import subprocess

        if platform.system() == "Darwin":
            # Forza l'apertura con TextEdit (o altra app di testo)
            subprocess.call(["open", "-a", "TextEdit", filepath])
        elif platform.system() == "Windows":
            os.startfile(filepath)
        else:  # Linux
            subprocess.call(["xdg-open", filepath])

    # AZIONI MENU (SOLAMENTE DEL MENU, NON DELLA TOOLBAR)
    def info_action(self):
        self.open_file_with_default_app(AppConfig.README_PATH)
    
    def about_action(self):
        """Azione per mostrare informazioni sull'applicazione"""
        #stessa di documentazione per ora
        self.documentation_action()
    
    def documentation_action(self):
        """Azione per mostrare la documentazione"""
        import webbrowser
        webbrowser.open(AppConfig.DOCUMENTATION_URL)

    def version_action(self):
        """Azione per mostrare la versione dell'applicazione"""
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Version")
        msg.setText(
            "The current version is 1.0.0 (Beta)."
        )
        msg.setStyleSheet("""
            QMessageBox {
                background-color: #fafafa;
                color: #222;
            }
            QLabel {
                color: #222;
            }
            QPushButton {
                background-color: #e0e0e0;
                color: #222;
                border-radius: 6px;
                padding: 6px 12px;
            }
        """)
        msg.exec()
    
    def website_action(self):
        """Azione per aprire il sito web dell'applicazione"""
        import webbrowser
        webbrowser.open(AppConfig.WEBSITE_URL)
    
    def license_action(self):
        self.open_file_with_default_app(AppConfig.LICENSES_PATH)  # o il file scelto dall'utente




    def setup_toolbar(self):
        toolbar = self.addToolBar("Replay Settings")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(32, 32))
        self.toolbar = toolbar  # salva riferimento per lo stile

        # Play/Pause
        self.actionPlayPause = QAction("Play", self)
        toolbar.addAction(self.actionPlayPause)
        self.actionPlayPause.triggered.connect(self.toggle_state_playing)

        # Stop
        self.actionStop = QAction("Stop", self)
        toolbar.addAction(self.actionStop)
        self.actionStop.triggered.connect(self.clear_history)

        toolbar.addSeparator()

        # Aggiungi un pulsante per aggiornare finestra mobile
        self.actionTimeWindow = QAction("Time Update", self)
        #aggiungi solo a finestra voltage
        if hasattr(self, 'voltage_plot'):
            toolbar.addAction(self.actionTimeWindow)
            self.actionTimeWindow.triggered.connect(self.toggle_time_window)
            self.time_window_enabled = True  # Variabile per tenere traccia dello stato della finestra mobile
            self.actionTimeWindow.setToolTip("Turn off time window update")

        # Aggiungi un pulsante per svolgere le operazioni matematiche
        #self.actionMath = QAction("Math Operations", self) DEFINITA IN MENUBAR
        #PER ORA DISPONIBILE SOLO SU VOLTAGE
        if hasattr(self, 'voltage_plot'):
            toolbar.addAction(self.actionMath)
            self.actionMath.setToolTip("Select region for analysis")

        # aggiungi azione per iFFT solo ad audio
        self.actionIFFTGraph = QAction("iFFT Graph", self)
        if hasattr(self, 'plot_widget_fft'):
            toolbar.addAction(self.actionIFFTGraph)
            self.actionIFFTGraph.triggered.connect(self.show_ifft_window)
            self.actionIFFTGraph.setToolTip("Open iFFT graph")
            #scorciatoia da tastiera
            self.actionIFFTGraph.setShortcut("Ctrl+I")

        toolbar.addSeparator()

        # Slider tempo in un contenitore espandibile
        slider_container = QWidget()
        slider_layout = QHBoxLayout(slider_container)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.setSpacing(6)
        slider_label = QLabel("Set Time:")
        slider_layout.addWidget(slider_label)
        self.time_slider = QSlider(Qt.Horizontal)  # ✅ Qt è già importato in cima al file
        self.time_slider.setRange(0, 120)
        self.time_slider.setValue(0)
        self.time_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        slider_layout.addWidget(self.time_slider)
        self.time_slider.valueChanged.connect(self.get_update_time_position)
        self.time_slider.sliderPressed.connect(self.pause_playback)
        slider_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(slider_container)

        toolbar.addSeparator()

        # ✅ SOSTITUISCI QLabel con TimeInputWidget
        self.current_time_input = TimeInputWidget()
        self.current_time_input.setToolTip(
            "Current playback position\n"
            "Click to edit and jump to specific time\n"
            "Formats: '123.45', '1:30', '90s'"
        )
        toolbar.addWidget(self.current_time_input)
        self.current_time_input.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        
        # ✅ CONNETTI al sistema di navigazione
        self.current_time_input.timeChanged.connect(self._on_time_input_changed)

        toolbar.addSeparator()
        self.velocity = QDoubleSpinBox()
        self.velocity.setRange(0.1, 1.0)
        self.velocity.setValue(1)
        self.velocity.setSuffix("x")
        self.velocity.setSingleStep(0.1)
        self.velocity.setFixedWidth(75)
        self.velocity.setDecimals(1)
        #applica solo a audio
        if hasattr(self, 'plot_widget_fft'):
            toolbar.addWidget(QLabel("Replay Velocity:"))
            toolbar.addWidget(self.velocity)
            self.velocity.valueChanged.connect(self.get_update_time_speed)

        # ✅ FIX WINDOWS: Crea icone con versione disabilitata
        from PySide6.QtGui import QPixmap, QPainter, QIcon
        # ✅ RIMUOVI l'import di Qt qui - è già importato in cima al file
        
        def create_icon_with_disabled_state(icon_path):
            """Helper per creare un'icona con stato disabilitato"""
            icon = QIcon(icon_path)
            pixmap = QPixmap(icon_path)
            disabled_pixmap = QPixmap(pixmap.size())
            disabled_pixmap.fill(Qt.transparent)  # ✅ Usa il Qt globale
            
            painter = QPainter(disabled_pixmap)
            painter.setOpacity(0.4)
            painter.drawPixmap(0, 0, pixmap)
            painter.end()
            
            icon.addPixmap(disabled_pixmap, QIcon.Disabled)
            return icon
        
        # Applica le icone a start e stop e update time
        play_pause_path = os.path.join(AppConfig.ICON_DIR, "actionPlayPause.png")
        stop_path = os.path.join(AppConfig.ICON_DIR, "actionStop.png")
        
        self.actionPlayPause.setIcon(create_icon_with_disabled_state(play_pause_path))
        self.actionStop.setIcon(create_icon_with_disabled_state(stop_path))
        
        # ✅ Applica SOLO se le azioni esistono (per evitare errori)
        if hasattr(self, 'actionTimeWindow'):
            time_update_path = os.path.join(AppConfig.ICON_DIR, "actionUpdatedTimeWindow.png")
            self.actionTimeWindow.setIcon(create_icon_with_disabled_state(time_update_path))
        
        if hasattr(self, 'actionMath'):
            math_path = os.path.join(AppConfig.ICON_DIR, "actionMath.png")
            self.actionMath.setIcon(create_icon_with_disabled_state(math_path))
        
        if hasattr(self, 'actionIFFTGraph'):
            iFF_path = os.path.join(AppConfig.ICON_DIR, "actionIFFTGraph.png")
            self.actionIFFTGraph.setIcon(create_icon_with_disabled_state(iFF_path))

        # Applica lo stile iniziale
        self.setup_toolbar_style()
        

    def setup_toolbar_style(self):
        """Applica lo stile della toolbar e dei suoi widget in base al tema/font correnti"""
        # Prendi i colori dal tema corrente
        toolbar_bg = self.theme_manager.get_toolbar_bg()
        theme_colors = self.theme_manager.get_theme_specific_colors()
        spinbox_bg = theme_colors.get('inactive_bg', '#222')
        spinbox_fg = theme_colors.get('inactive_text', '#fff')

        if hasattr(self, "toolbar"):
            self.toolbar.setStyleSheet(f"""
                QToolBar {{
                    min-height: 48px;
                    max-height: 56px;
                    background: {toolbar_bg};
                }}
                QToolButton, QLabel {{
                    color: white;
                    font-size: 16px;
                }}
            """)

        # Aggiorna lo stile dei widget della toolbar
        if hasattr(self, "velocity"):
            self.velocity.setStyleSheet(f"""
                QDoubleSpinBox {{
                    background: {spinbox_bg};
                    color: {spinbox_fg};
                    font-size: 16px;
                    border: 1px solid #555;
                }}
                QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
                    background: transparent;
                }}
            """)      

        # ✅ FIX: USA IL FONT DAL FONT MANAGER invece di hardcoded 16px
        if hasattr(self, "current_time_input"):
            # ✅ Ottieni il font size dinamico dalla toolbar
            toolbar_font_size = 16  # Default fallback
            
            # ✅ Prendi il font size REALE dal tema corrente
            if hasattr(self, 'toolbar') and self.toolbar:
                # Estrai font size dal CSS della toolbar
                import re
                toolbar_css = self.toolbar.styleSheet()
                font_match = re.search(r'font-size:\s*(\d+)px', toolbar_css)
                if font_match:
                    toolbar_font_size = int(font_match.group(1))
            
            # ✅ APPLICA IL FONT CORRETTO
            self.current_time_input.setStyleSheet(f"""
                QLineEdit {{
                    background-color: transparent;
                    border: 1px solid transparent;
                    color: white;
                    font-size: {toolbar_font_size}px;
                    min-height: 28px;
                    max-height: 28px;
                    padding: 2px 8px;
                }}
            """)
            
            print(f"✅ TimeInputWidget font size updated to: {toolbar_font_size}px")


    def _save_current_settings(self):
        """Salva le impostazioni correnti (tema e font)"""
        if hasattr(self, 'theme_manager'):
            # Salva il tema corrente
            current_theme = getattr(self.theme_manager, 'current_theme', None)
            if current_theme:
                self.theme_manager.save_theme(current_theme)
                print(f"🎨 Tema salvato: {current_theme}")
            
            # Salva la scala del font
            font_scale = getattr(self.font_manager, 'current_font_scale', None)
            if font_scale:
                self.font_manager.save_font_scale(font_scale)
                print(f"🔠 Scala font salvata: {font_scale}")
        
        # Salva altre impostazioni della finestra (posizione, dimensione)
        self.settings_manager.save_window_geometry(self)
        print("💾 Impostazioni finestra salvate")

    
    def closeEvent(self, event):
        """Gestisce l'evento di chiusura della finestra"""
        print("🔒 Chiusura della finestra...")
        # Salva le impostazioni correnti prima di chiudere
        self._save_current_settings()
        super().closeEvent(event)

        print("✅ Finestra chiusa correttamente")

    
    def toggle_time_window(self):
        self.time_window_enabled = not self.time_window_enabled
        stato = "ATTIVA" if self.time_window_enabled else "DISATTIVA"
        print(f"⏳ Finestra mobile {stato}")
        # (Opzionale) Cambia il testo o lo stato del pulsante se vuoi
        self.actionTimeWindow.setToolTip("Turn on time window update" if not self.time_window_enabled else "Turn off time window update")

    def update_display(self):
        pass #da implementare nelle finestre derivate

    def toggle_state_playing(self):
        """Cambia lo stato di riproduzione e avvia/ferma il playback"""
        self.is_running = not self.is_running
        
        # Aggiorna il testo del pulsante
        if self.is_running:
            print("toggle start")
            self.actionPlayPause.setText(self.text_stop)
            self.start_playback()  # Chiama il metodo per avviare la riproduzione
        else:
            print("toggle pause")
            self.actionPlayPause.setText(self.text_start)
            self.pause_playback()  # Chiama il metodo per mettere in pausa

    def get_update_time_speed(self):
        """Aggiorna la velocità di riproduzione in base al valore del QDoubleSpinBox"""
        speed = self.velocity.value()
        print(f"Velocità di riproduzione aggiornata a: {speed:.1f}x")
        # Qui puoi implementare la logica per aggiornare la velocità di riproduzione
        self.playback_speed_changed.emit(speed)

    def get_update_time_position(self):
        """Aggiorna la posizione di riproduzione in base al valore dello slider"""
        position = self.time_slider.value()
        self.current_position = position
        self.update_display()  # Aggiorna la visualizzazione in base alla nuova posizione
        self.playback_position_changed.emit(position)

    def clear_history(self):
        """Pulisce la cronologia della riproduzione"""
        print("🔄 Cronologia della riproduzione cancellata")
        # Implementa la logica per cancellare la cronologia
        self.is_running = False


    def prepare_for_math_dialog(self):
        import pyqtgraph as pg
        """Prepara i dati e apre la finestra di operazioni matematiche"""
        # Rimuovi la regione esistente se presente
        if hasattr(self, 'region'):
            if hasattr(self, 'plot_widget_time') and hasattr(self.plot_widget_time, 'plot_widget'):
                self.plot_widget_time.plot_widget.removeItem(self.region)
            elif hasattr(self, 'voltage_plot') and hasattr(self.voltage_plot, 'plot_widget'):
                self.voltage_plot.plot_widget.removeItem(self.region)
            del self.region # Rimuovi l'attributo
            return
        
        # ✅ Ottieni il plot widget corretto
        if hasattr(self, 'plot_widget_time') and hasattr(self.plot_widget_time, 'plot_widget'):
            plot_widget = self.plot_widget_time.plot_widget
        elif hasattr(self, 'voltage_plot') and hasattr(self.voltage_plot, 'plot_widget'):
            plot_widget = self.voltage_plot.plot_widget
        else:
            print("⚠️ No plot widget found")
            return
        
        # ✅ Ottieni il range visibile corrente
        view_range = plot_widget.viewRange()
        visible_x_min, visible_x_max = view_range[0]
        
        # ✅ Calcola il centro della vista corrente
        view_center = (visible_x_min + visible_x_max) / 2
        
        # ✅ Crea una regione di 5 secondi centrata sulla vista
        region_width = 5.0  # 5 secondi
        half_width = region_width / 2
        
        start_pos = view_center - half_width
        end_pos = view_center + half_width
        
        # ✅ Verifica limiti dei dati
        if hasattr(self, 'data_manager') and hasattr(self.data_manager, 'total_duration_sec'):
            max_limit = self.data_manager.total_duration_sec
        elif hasattr(self, 'total_duration_ms'):
            max_limit = self.total_duration_ms / 1000.0
        else:
            max_limit = None
        
        min_limit = 0.0
        
        # ✅ Aggiusta i limiti se necessario
        if start_pos < min_limit:
            start_pos = min_limit
            end_pos = min(start_pos + region_width, max_limit if max_limit else start_pos + region_width)
        
        if max_limit and end_pos > max_limit:
            end_pos = max_limit
            start_pos = max(end_pos - region_width, min_limit)
        
        # ✅ Se la regione è troppo piccola (vicino ai bordi), usa valori di default
        if end_pos - start_pos < 1.0:  # Meno di 1 secondo
            if hasattr(self, 'current_position'):
                current_time = self.current_position / 1000.0
            elif hasattr(self, 'current_position_ms'):
                current_time = self.current_position_ms / 1000.0
            else:
                current_time = 5.0
            
            start_pos = max(0, current_time - 2.5)
            end_pos = start_pos + 5.0
            
            if max_limit and end_pos > max_limit:
                end_pos = max_limit
                start_pos = max(0, end_pos - 5.0)
        
        print(f"📍 Creating region: {start_pos:.2f}s to {end_pos:.2f}s (center: {view_center:.2f}s)")
        
        # Crea una nuova regione selezionabile nella posizione calcolata
        self.region = ClickableLinearRegionItem([start_pos, end_pos], movable=True, brush=(0, 0, 255, 50))
        self.actionMath.setToolTip("Double-click the region to analyze")
        self.actionMathDialog.setEnabled(True)  # Abilita l'azione di dialogo matematico

        plot_widget.addItem(self.region)
        
        # Connetti il segnale di modifica della regione a una funzione di callback
        self.region.sigRegionChanged.connect(self._region_check)
        self.region.doubleClicked.connect(self.open_math_dialog)

        #chiama subito per assicurarsi che sia nei limiti
        self._region_check()


    def _region_check(self):
        """Controlla i limiti e la larghezza massima della regione selezionata"""
        min_limit = 0.0

        # Calcola limite massimo (in secondi)
        if hasattr(self, 'data_manager') and hasattr(self.data_manager, 'total_duration_sec'):
            max_limit = self.data_manager.total_duration_sec
        elif hasattr(self, 'total_duration_ms'):
            max_limit = self.total_duration_ms / 1000.0
        else:
            max_limit = None

        start, end = self.region.getRegion()
        width = end - start
        MAX_WIDTH = 30.0  # secondi massimi

        # ---- 1️⃣ Limiti inferiori/superiori ----
        if start < min_limit:
            start = min_limit
            end = start + width

        if max_limit is not None and end > max_limit:
            end = max_limit
            start = end - width
            # ✅ Se lo start va sotto zero, aggiusta anche l'end
            if start < min_limit:
                start = min_limit
                end = min(start + width, max_limit)

        # ---- 2️⃣ Limite di larghezza ----
        if width > MAX_WIDTH:
            center = (start + end) / 2
            start = center - MAX_WIDTH / 2
            end = center + MAX_WIDTH / 2
            
            # ✅ Ricontrolla i limiti dopo aver ridimensionato
            if start < min_limit:
                start = min_limit
                end = start + MAX_WIDTH
            if max_limit is not None and end > max_limit:
                end = max_limit
                start = end - MAX_WIDTH
                if start < min_limit:
                    start = min_limit

        # ---- 3️⃣ Non permettere di andare oltre la fine dell'esperimento ----
        if max_limit is not None:
            if end > max_limit:
                end = max_limit
                start = max(min_limit, end - width)
                print(f"⚠️ Region adjusted: cannot exceed experiment end ({max_limit:.3f}s)")

        # ---- 4️⃣ Se è in pausa, non permettere di spostare la regione oltre la posizione corrente ----
        if not self.is_running:
            # ✅ INIZIALIZZA current_time SEMPRE
            current_time = None
            
            # Ottieni la posizione corrente in secondi
            if hasattr(self, 'current_position'):
                if self.current_position == 0:
                    # Se siamo all'inizio, permetti qualsiasi regione valida
                    current_time = None
                else:
                    current_time = self.current_position / 1000.0
            elif hasattr(self, 'current_position_ms'):
                if self.current_position_ms == 0:
                    # Se siamo all'inizio, permetti qualsiasi regione valida
                    current_time = None
                else:
                    current_time = self.current_position_ms / 1000.0

            # ✅ Controlla solo se current_time è valido
            if current_time is not None and current_time > 0:
                if end > current_time:
                    end = current_time
                    start = max(min_limit, end - width)
                    print(f"⚠️ Region adjusted: cannot exceed current position ({current_time:.3f}s) while paused")

        # ---- 5️⃣ Evita loop di aggiornamento ----
        current_region = self.region.getRegion()
        if abs(current_region[0] - start) > 1e-6 or abs(current_region[1] - end) > 1e-6:
            self.region.blockSignals(True)
            self.region.setRegion([start, end])
            self.region.blockSignals(False)



    def open_math_dialog(self):
        start, end = self.region.getRegion()

        ### VOLTAGE ###
        if hasattr(self, 'voltage_plot'):
            print(f"Opening math dialog for region: {start:.3f}s to {end:.3f}s")
            x_data = np.concatenate([c[0] for c in self._data_chunks if c is not None])
            y_data = np.concatenate([c[1] for c in self._data_chunks if c is not None])
            
            # ✅ FIX: Calcola start_with_pre in modo sicuro (massimo 1s prima, minimo inizio file)
            time_before = min(1.0, start)  # Non andare mai sotto 0
            start_with_pre = max(0, start - time_before)
            
            mask = (x_data >= start_with_pre) & (x_data <= end)
            x_region = x_data[mask] 
            y_region = y_data[mask]
            if len(x_region) == 0 or len(y_region) == 0:
                self.show_error_dialog("No Data", "No data found in the selected region.")
                return
            
            print(f"   Data range: {start_with_pre:.3f}s to {end:.3f}s ({time_before:.3f}s before selection)")
            
            self.math_dialog = MathOperations(
                x_region, 
                y_region, 
                parent=self, 
                amplified_data=self.amplified_data, 
                sampling_rate=self.sampling_rate,
                file_path=self.file_path
            )
            self.math_dialog.show()


    def on_open_saved_analysis_triggered(self):
        """Legge le analisi dal file e apre un dialogo per la selezione."""
        if not self.file_path:
            self.show_error_dialog("No File", "Please open a .pvolt or .paudio file first.")
            return

        analyses = self._read_analyses_from_file()
        if not analyses:
            QMessageBox.information(self, "No Analyses", "No saved analyses found in this file.")
            return
        
        dialog = SelectAnalysisDialog(analyses, self)
        # ✅ Connetti il segnale di eliminazione
        dialog.analysis_deleted.connect(self._delete_analysis_from_file)
        if dialog.exec():
            selected_id = dialog.get_selected_analysis_id()
            if selected_id:
                print(f"🚀 Loading analysis with ID: {selected_id}")
                # Apri il dialogo matematico passando i dati dell'analisi selezionata
                self.open_math_dialog_with_data(analyses[selected_id])

    def _read_analyses_from_file(self):
        """
        Legge tutte le analisi salvate da un file .pvolt/.paudio.
        Restituisce un dizionario di analisi o un dizionario vuoto.
        """
        import json
        import struct
        try:
            with open(self.file_path, "rb") as f:
                # Controlla che il file sia abbastanza grande
                file_size = f.seek(0, 2)
                if file_size <= 8:
                    print("📖 File too small for a footer. No analyses found.")
                    return {}

                # 1. Vai agli ultimi 8 byte per leggere il puntatore
                f.seek(-8, 2)
                footer_offset_bytes = f.read(8)
                footer_offset = struct.unpack('<Q', footer_offset_bytes)[0]

                # 2. Se il puntatore è valido, leggi il JSON
                if footer_offset > 0 and footer_offset < file_size:
                    f.seek(footer_offset)
                    json_data_len = file_size - footer_offset - 8
                    json_str = f.read(json_data_len).decode('utf-8')
                    data = json.loads(json_str)
                    print(f"📖 Found {len(data.get('analyses', {}))} saved analyses.")
                    return data.get('analyses', {})
                else:
                    print("📖 No valid footer found. No analyses saved.")
                    return {}
        except (IOError, struct.error, json.JSONDecodeError) as e:
            print(f"⚠️ Could not read analyses: {e}. File might be old or corrupted.")
            return {}
        
    def _delete_analysis_from_file(self, analysis_id_to_delete):
        """Rimuove un'analisi specifica dal footer JSON del file."""
        import json
        import struct
        print(f"🔥 Attempting to physically delete analysis ID: {analysis_id_to_delete}")

        # 1. Leggi tutte le analisi esistenti
        analyses = self._read_analyses_from_file()
        if analysis_id_to_delete not in analyses:
            print(f"⚠️ Analysis ID {analysis_id_to_delete} not found in file. Nothing to delete.")
            return

        # 2. Rimuovi l'analisi dal dizionario
        del analyses[analysis_id_to_delete]
        print(f"   Removed from dictionary. {len(analyses)} analyses remaining.")

        # 3. Riscrivi l'intero footer JSON aggiornato
        try:
            with open(self.file_path, "r+b") as f:
                file_size = f.seek(0, 2)
                footer_offset = 0
                if file_size > 8:
                    f.seek(-8, 2)
                    footer_offset = struct.unpack('<Q', f.read(8))[0]

                # Tronca il file per rimuovere il vecchio footer
                if footer_offset > 0 and footer_offset < file_size:
                    f.seek(footer_offset)
                    f.truncate()
                    print(f"   ✂️ Old footer truncated at offset {footer_offset}.")
                
                # Se non ci sono più analisi, scrivi solo un puntatore nullo
                if not analyses:
                    print("   🗑️ No analyses left. Writing null pointer.")
                    f.write(struct.pack('<Q', 0))
                    return

                # Scrivi il nuovo footer JSON
                full_footer_dict = {"file_format_version": "1.0", "analyses": analyses}
                new_json_str = json.dumps(full_footer_dict, indent=2)
                new_json_bytes = new_json_str.encode('utf-8')
                
                new_footer_offset = f.tell()
                f.write(new_json_bytes)
                f.write(struct.pack('<Q', new_footer_offset))
                print(f"   ✅ New footer written successfully at offset {new_footer_offset}.")

        except Exception as e:
            self.show_error_dialog("Deletion Error", f"An error occurred while deleting the analysis from the file:\n{e}")
            print(f"❌ Error during physical deletion: {e}")




    def open_math_dialog_with_data(self, analysis_data):
        """Apre il dialogo MathOperations pre-compilato con i dati di un'analisi."""
        # Dobbiamo comunque fornire i dati del segnale grezzo
        start, end = analysis_data['parameters']['general']['start_time'], analysis_data['parameters']['general']['end_time']
        
        # Estrai i dati del segnale per la regione dell'analisi
        x_data = np.concatenate([c[0] for c in self._data_chunks if c is not None])
        y_data = np.concatenate([c[1] for c in self._data_chunks if c is not None])
        
        # ✅ FIX: Usa un buffer più ampio per la baseline, ma gestisci start < 1s
        time_before = min(1.0, start)
        start_with_pre = max(0, start - time_before)
        
        mask = (x_data >= start_with_pre) & (x_data <= end)
        x_region = x_data[mask]
        y_region = y_data[mask]

        if len(x_region) == 0:
            self.show_error_dialog("Data Error", "Could not find the signal data for the selected analysis region.")
            return
        
        print(f"   Loading saved analysis: {start_with_pre:.3f}s to {end:.3f}s ({time_before:.3f}s before)")

        self.math_dialog = MathOperations(
            x_region, 
            y_region, 
            parent=self, 
            amplified_data=self.amplified_data, 
            sampling_rate=self.sampling_rate,
            file_path=self.file_path,
            analysis_data=analysis_data
        )
        self.math_dialog.show()



    ####  ####
    def start_playback(self):
        """Avvia la riproduzione dall'inizio o dalla posizione corrente"""
        if not self.is_running:
            self.is_running = True
            self.actionPlayPause.setText(self.text_stop)
            self.started_playing.emit()  # Emetti il segnale

    def pause_playback(self):
        """Mette in pausa la riproduzione"""
        if self.is_running:
            self.is_running = False
            self.actionPlayPause.setText(self.text_start)
            #self.paused_playing.emit()  # Emetti il segnale

    def stop_playback(self):
        """Ferma la riproduzione e resetta alla posizione iniziale"""
        self.is_running = False
        self.actionPlayPause.setText(self.text_start)
        self.time_slider.setValue(0)
        self.playback_position_changed.emit(0)
        self.update_display()


    def keyPressEvent(self, event):
        """Gestisce la pressione dei tasti per la navigazione e il playback."""
        
        # ✅ SOLUZIONE: Controlla prima lo spazio, che deve funzionare sempre.
        if event.key() == Qt.Key_Space:
            self.toggle_state_playing()
            event.accept()
            return # Abbiamo gestito l'evento, usciamo.

        # La logica seguente (frecce) funziona solo se il playback è in pausa.
        if self.is_running:
            super().keyPressEvent(event)
            return

        if event.key() == Qt.Key_Right:
            self.step_position(1)  # Vai avanti
            event.accept()
        elif event.key() == Qt.Key_Left:
            self.step_position(-1) # Vai indietro
            event.accept()
        else:
            super().keyPressEvent(event)

    def step_position(self, direction: int):
        """
        Implementazione di default per lo spostamento temporale.
        Le classi figlie possono sovrascriverla per un comportamento specifico.
        """
        current_pos = getattr(self, 'current_position', 0)
        total_duration = getattr(self, 'total_duration_ms', 0)

        if total_duration == 0:
            return

        # Calcola la nuova posizione
        new_position_ms = current_pos + (direction * self.time_step_ms)
        
        # Controlla i limiti
        new_position_ms = max(0, min(new_position_ms, total_duration))

        # Aggiorna lo slider (che a sua volta aggiornerà il resto)
        self.time_slider.setValue(int(new_position_ms))


    def _on_time_input_changed(self, target_time_sec: float):
        """
        Gestisce input manuale di tempo dall'utente.
        Naviga alla posizione specificata.
        """
        print(f"⏱️ Time input: jumping to {target_time_sec:.2f}s")
        
        # Converti in millisecondi per compatibilità
        target_position_ms = target_time_sec * 1000.0
        
        # Pausa se in playback
        was_playing = self.is_running
        if was_playing:
            self.pause_playback()
        
        # ✅ USA IL MECCANISMO ESISTENTE
        self.get_update_time_position()  # Aggiorna posizione
        self.time_slider.setValue(int(target_position_ms))
        
        # ✅ Trigger display update (implementato nelle sottoclassi)
        self.update_display()
    
    def open_trim_dialog(self):
        """Apre il dialog per l'export di una regione trimmed"""
        from components.trim_region_dialog import TrimRegionDialog
        
        # Determina il tipo di file
        if hasattr(self, 'voltage_plot'):
            file_type = 'voltage'
        elif hasattr(self, 'plot_widget_fft'):
            file_type = 'audio'
        else:
            QMessageBox.warning(self, "Error", "Cannot determine file type")
            return
        
        # Ottieni durata totale
        if hasattr(self, 'data_manager') and hasattr(self.data_manager, 'total_duration_sec'):
            total_duration = self.data_manager.total_duration_sec
        elif hasattr(self, 'total_duration_ms'):
            total_duration = self.total_duration_ms / 1000.0
        else:
            QMessageBox.warning(self, "Error", "Cannot determine file duration")
            return
        
        # Verifica che ci sia un file caricato
        if not hasattr(self, 'file_path') or not self.file_path:
            QMessageBox.warning(self, "No File", "Please open a file first")
            return
        
        # Apri il dialog
        dialog = TrimRegionDialog(
            parent=self,
            file_path=self.file_path,
            file_type=file_type,
            total_duration_sec=total_duration
        )
        
        if dialog.exec():
            # Utente ha confermato → avvia export
            export_params = dialog.get_export_parameters()
            print(f"✅ Export parameters: {export_params}")
            self._execute_trim_export(export_params)
    
    def _execute_trim_export(self, params):
        """
        Esegue l'export della regione trimmed.
        Chiamato dalle sottoclassi (voltage/audio specific).
        """
        # Implementazione di default (override nelle sottoclassi)
        QMessageBox.information(
            self,
            "Export",
            f"Export not yet implemented for this file type.\n"
            f"Parameters: {params}"
        )
    
    def show_spectral_energy_analysis(self):
        """
        Show spectral energy analysis for the current frame (AUDIO ONLY).
        This method should be overridden in audio subclass.
        """
        QMessageBox.warning(
            self, 
            "Not Available", 
            "Spectral energy analysis is only available for audio files."
        )


# Personalized LinearRegionItem with double-click signal
class ClickableLinearRegionItem(pg.LinearRegionItem):
    doubleClicked = Signal()  # Custom signal

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def mouseDoubleClickEvent(self, ev):
        self.doubleClicked.emit()
        ev.accept()



class SelectAnalysisDialog(QDialog):
    """Dialogo per selezionare un'analisi salvata da una lista."""
    # ✅ SEGNALE per notificare l'eliminazione di un'analisi
    analysis_deleted = Signal(str)

    def __init__(self, analyses_dict, parent=None):
        from PySide6.QtWidgets import QListWidget, QListWidgetItem, QDialogButtonBox
        super().__init__(parent)
        self.setWindowTitle("Select a Saved Analysis")
        self.setMinimumSize(450, 300)

        self.layout = QVBoxLayout(self)
        self.list_widget = QListWidget()
        self.layout.addWidget(self.list_widget)

        # Popola la lista
        if not analyses_dict:
            self.list_widget.addItem("No analyses found.")
            self.list_widget.setEnabled(False)
        else:
            for analysis_id, data in sorted(analyses_dict.items(), key=lambda item: item[1]['metadata']['saved_at'], reverse=True):
                name = data.get("metadata", {}).get("name", "Unnamed Analysis")
                timestamp_str = data.get("metadata", {}).get("saved_at", "")
                
                # Formatta il timestamp per una migliore leggibilità
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    readable_time = dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    readable_time = "Invalid date"

                item_text = f"{name} (saved at: {readable_time})"
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, analysis_id)  # Salva l'UUID nell'item
                self.list_widget.addItem(item)
        
        # Connetti doppio click
        self.list_widget.itemDoubleClicked.connect(self.accept)

        # --- ✅ NUOVO LAYOUT PER I PULSANTI ---
        # Crea un layout orizzontale per i pulsanti
        button_layout = QHBoxLayout()
        
        # 1. Pulsante Delete (a sinistra)
        self.delete_button = QPushButton("Delete")
        self.delete_button.setEnabled(False) # Disabilitato di default
        self.delete_button.setStyleSheet("color: #d32f2f;") # Stile per renderlo "distruttivo"
        self.delete_button.clicked.connect(self.on_delete_triggered)
        button_layout.addWidget(self.delete_button)

        # 2. Spaziatore per spingere i pulsanti a destra
        button_layout.addStretch(1)

        # 3. Pulsante Annulla
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        # 4. Pulsante OK (pulsante di default)
        self.ok_button = QPushButton("OK")
        self.ok_button.setDefault(True)
        self.ok_button.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_button)

        # Aggiungi il layout dei pulsanti al layout principale
        self.layout.addLayout(button_layout)
        
        # Connetti la selezione della lista allo stato del pulsante Delete
        self.list_widget.itemSelectionChanged.connect(self.update_delete_button_state)

    def get_selected_analysis_id(self):
        selected_items = self.list_widget.selectedItems()
        if selected_items:
            return selected_items[0].data(Qt.UserRole)
        return None
    
    def update_delete_button_state(self):
        """Abilita il pulsante Delete solo se un item è selezionato."""
        self.delete_button.setEnabled(len(self.list_widget.selectedItems()) > 0)

    def on_delete_triggered(self):
        """Gestisce il click sul pulsante Delete."""
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            return

        item = selected_items[0]
        analysis_id = item.data(Qt.UserRole)
        analysis_name = item.text().split(' (saved at:')[0]

        # ✅ Finestra di conferma
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to permanently delete the analysis:\n\n'{analysis_name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            print(f"🔥 Deleting analysis with ID: {analysis_id}")
            # Rimuovi l'item dalla lista nella UI
            self.list_widget.takeItem(self.list_widget.row(item))
            # Emetti il segnale per l'eliminazione fisica dal file
            self.analysis_deleted.emit(analysis_id)





#classe per visualizzazione analisi matematica
class MathOperations(QDialog, Ui_QDialogMath):
    def __init__(self, time_data, signal_data, parent=None, amplified_data=True, sampling_rate=500, file_path=None, analysis_data=None):
        self.settings_manager = SettingsManager()

        QDialog.__init__(self, parent)
        self.setupUi(self)

        self.source_file_path = file_path
        self.current_analysis_id = None
        self.name_of_analysis = None

        self._updating_ui = False
        self.analysis_items_visible = True

        self.font_manager = FontManager(self.settings_manager.settings)
        self.theme_manager = ThemeManager(self.settings_manager.settings, self.font_manager)
        self.layout_manager = LayoutManager(self.font_manager)

        self.sampling_rate = sampling_rate
        self.total_time_data = time_data
        self.total_signal_data = signal_data
        self.amplified_data = amplified_data

        # Imposta i dati temporanei per la creazione iniziale del plot
        # Sono time data e signal data togliendo un secondo dall'inizio che è passato solo per fare la media
        # Sfrutta sampling rate per calcolare quanti punti togliere
        points_to_remove = int(sampling_rate * 1.0)  # punti da togliere (1 secondo)
        self.time_data = self.total_time_data[points_to_remove:]
        self.signal_data = self.total_signal_data[points_to_remove:]

        self.recreate_plot()
        self._setup_specific_ui()

        # === LOGICA DI INIZIALIZZAZIONE ===
        if analysis_data:
            # CARICA DA ANALISI ESISTENTE
            print("🚀 Initializing dialog from saved analysis...")
            self._load_from_analysis_data(analysis_data)
        else:
            # NUOVA ANALISI DA ZERO
            print("✨ Initializing new analysis from scratch...")
            self.get_arrays()
            self.initialize_default_variables()        
            self.get_V_baseline()
            fitting_success = self.perform_curve_fitting()
            if not fitting_success:
                # Gestione fallback se il fitting automatico fallisce
                print("⚠️ Automatic fitting skipped. Manual fitting available.")
                self.setup_general_variables_ui()


        #Connessione pulsanti
        self.moreInfoButton.clicked.connect(self.on_more_info_button_triggered)
        self.saveButton.clicked.connect(self.on_save_button_triggered)
        self.cancelButton.clicked.connect(self.on_cancel_button_triggered)
        self.exportButton.clicked.connect(self.on_export_csv_button_triggered)
        self.autoFitButton.clicked.connect(self.on_autofitbutton_triggered)
        self.analysisLinesButton.clicked.connect(self.toggle_analysis_items_visibility)

        # ✅ ALLA FINE, dopo aver creato tutta la UI
        print("🔧 Configuring parameter callbacks...")
        self.setup_parameter_callbacks()

        # Se stiamo caricando, aggiorna la curva e le linee una volta che tutto è pronto
        if analysis_data:
            # Forza l'aggiornamento dello stato e della UI basandosi sul modello caricato.
            self._on_signal_type_changed(self.signalTypeComboBox.currentText())

            # Ora che lo stato è corretto, aggiorna anche tutte le linee di riferimento.
            self._update_reference_lines()
            self._update_fit_region_display()
            self._update_peak_line()

        # Mostra la finestra in mezzo allo schermo del parent mantenendo le grafiche
        if parent:
            parent_rect = parent.geometry()
            self.resize(parent_rect.width() * 0.8, parent_rect.height() * 0.6)
            self.move(
                parent_rect.x() + (parent_rect.width() - self.width()) // 2,
                parent_rect.y() + (parent_rect.height() - self.height()) // 2 - 150 #correzione verticale verso l'alto per windows
            )

    #### ALGORITMO PER MODIFICHE MANUALI DEI PARAMETRI ####

    def setup_parameter_callbacks(self):
        """Configura i callback per aggiornare parametri correlati e ricalcolare la curva"""
        print("🔧 Setting up parameter callbacks...")
        
        # ===== PARAMETRI GENERALI =====
        if hasattr(self, 'direction_combo'):
            self.direction_combo.currentTextChanged.connect(self._on_direction_changed)
        
        # ===== EXPONENTIAL RETURN =====
        if hasattr(self, 'A_spin'):
            self.A_spin.valueChanged.connect(self._on_exponential_param_changed)
        if hasattr(self, 'tau_spin'):
            self.tau_spin.valueChanged.connect(self._on_exponential_param_changed)
        
        # ===== ACTION POTENTIAL =====
        if hasattr(self, 'A_sin_spin'):
            self.A_sin_spin.valueChanged.connect(self._on_action_potential_param_changed)
        if hasattr(self, 'f_spin'):
            self.f_spin.valueChanged.connect(self._on_action_potential_param_changed)
        if hasattr(self, 'phi_spin'):
            self.phi_spin.valueChanged.connect(self._on_action_potential_param_changed)
        if hasattr(self, 'A_exp_spin'):
            self.A_exp_spin.valueChanged.connect(self._on_action_potential_param_changed)
        if hasattr(self, 'tau_exp_spin'):
            self.tau_exp_spin.valueChanged.connect(self._on_action_potential_param_changed)
        
        # ===== CAMBIO TIPO DI SEGNALE =====
        self.signalTypeComboBox.currentTextChanged.connect(self._on_signal_type_changed)
        
        print("✅ Parameter callbacks configured")


    def _on_direction_changed(self, new_direction):
        """Gestisce il cambio di direzione del segnale"""

        if self._updating_ui:
            return

        print(f"📊 Direction changed to: {new_direction}")
        
        old_direction = self.direction
        self.direction = new_direction.lower()
        
        # ✅ Se cambio direzione, inverto i segni delle ampiezze
        if old_direction != self.direction:
            if self.is_exponential_return:
                self.amplitude_exponential = -self.amplitude_exponential
                if hasattr(self, 'A_spin'):
                    self.A_spin.blockSignals(True)
                    self.A_spin.setValue(self.amplitude_exponential)
                    self.A_spin.blockSignals(False)
            
            elif self.is_action_potential:
                self.A_sin = -self.A_sin
                self.A_exp = -self.A_exp
                
                if hasattr(self, 'A_sin_spin'):
                    self.A_sin_spin.blockSignals(True)
                    self.A_sin_spin.setValue(self.A_sin)
                    self.A_sin_spin.blockSignals(False)
                
                if hasattr(self, 'A_exp_spin'):
                    self.A_exp_spin.blockSignals(True)
                    self.A_exp_spin.setValue(self.A_exp)
                    self.A_exp_spin.blockSignals(False)
        
        # Ricalcola e aggiorna il plot
        self._update_curve_from_params()


    def _on_exponential_param_changed(self, value):
        """Gestisce il cambio di parametri esponenziali"""
        print(f"🔄 Exponential parameter changed")
        if self._updating_ui:
            return

        # Aggiorna le variabili interne
        if hasattr(self, 'A_spin'):
            self.amplitude_exponential = self.A_spin.value()
        if hasattr(self, 'tau_spin'):
            self.tau_exponential = self.tau_spin.value()
        
        # Ricalcola la curva con i nuovi parametri
        self._update_curve_from_params()


    def _on_action_potential_param_changed(self, value):
        """Gestisce il cambio di parametri del potenziale d'azione"""

        if self._updating_ui:
            return

        print(f"🔄 Action potential parameter changed")
        
        # Aggiorna le variabili interne
        if hasattr(self, 'A_sin_spin'):
            self.A_sin = self.A_sin_spin.value()
        if hasattr(self, 'f_spin'):
            self.f_osc = self.f_spin.value()
        if hasattr(self, 'phi_spin'):
            self.phi = self.phi_spin.value()
        if hasattr(self, 'A_exp_spin'):
            self.A_exp = self.A_exp_spin.value()
        if hasattr(self, 'tau_exp_spin'):
            self.tau_action_potential = self.tau_exp_spin.value()
        
        # Ricalcola la curva con i nuovi parametri
        self._update_curve_from_params()


    def _on_signal_type_changed(self, new_type):
        """Gestisce il cambio di tipo di segnale"""

        if self._updating_ui:
            return
        
        print(f"🔀 Signal type changed to: {new_type}")
        
        # ✅ Blocca temporaneamente per evitare loop durante setup UI
        self._updating_ui = True
        
        try:
            if new_type == "Exponential Return":
                self.is_exponential_return = True
                self.is_action_potential = False
                
                if not hasattr(self, 'amplitude_exponential') or self.amplitude_exponential == 0:
                    self.setup_exponential_variables()
                
                if not hasattr(self, 'params_dict_exp'):
                    self.params_dict_exp = {
                        'A': self.amplitude_exponential,
                        't0': self.time_peak,
                        'tau': self.tau_exponential,
                        'Vbaseline': self.V_baseline
                    }
                
                self.setup_exponential_return_ui()
                
                # ✅ Riconfigura callbacks e aggiorna curva
                self._update_curve_from_params()
                
            elif new_type == "Action Potential":
                self.is_action_potential = True
                self.is_exponential_return = False
                
                if not hasattr(self, 'A_sin') or self.A_sin == 0:
                    self.setup_action_potential_variables()
                
                if not hasattr(self, 'params_dict_sin_action_potential'):
                    self.params_dict_sin_action_potential = {
                        'A_sin': self.A_sin,
                        'f': self.f_osc,
                        'phi': self.phi
                    }
                
                if not hasattr(self, 'params_dict_exp_action_potential'):
                    self.params_dict_exp_action_potential = {
                        'A_exp': self.A_exp,
                        't_peak': self.time_peak,
                        'tau': self.tau_action_potential,
                        'V_baseline': self.V_baseline
                    }
                
                self.setup_action_potential_ui()
                
                # ✅ Riconfigura callbacks e aggiorna curva
                self._update_curve_from_params()
                
            elif new_type == "General Variables":
                # ✅ RESET dei flag PRIMA di setup
                self.is_exponential_return = False
                self.is_action_potential = False
                
                # ✅ Setup UI per General Variables
                self.setup_general_variables_ui()
                
                # ✅ Mostra le linee di riferimento e la regione di fitting
                self._update_reference_lines()
                self._update_fit_region_display()
                self._update_peak_line()
                
                # ⚠️ NON chiamare _update_curve_from_params() perché non c'è modello attivo
                return  # ← USCITA ANTICIPATA
            
        finally:
            # ✅ Rilascia il flag
            self._updating_ui = False


    #FUNZIONE CENTRALE DI AGGIORNAMENTO CURVA
    def _update_curve_from_params(self):
        """
        Ricalcola la curva usando i parametri CORRENTI (dalle spinbox)
        SENZA eseguire curve_fit. Aggiorna anche R² e formula.
        """
        print("🔄 Updating curve from current parameters...")
        
        try:
            # ✅ Verifica che ci siano dati da fittare
            if not hasattr(self, 'fit_time_data') or len(self.fit_time_data) == 0:
                print("⚠️ No data available for fitting")
                return
            
            # ===== EXPONENTIAL RETURN =====
            if self.is_exponential_return:
            # Usa i parametri correnti dalle spinbox (o variabili)
                A = self.A_spin.value() if hasattr(self, 'A_spin') else self.amplitude_exponential
                
                # ✅ FIX: Usa t0 fittato se disponibile, altrimenti time_peak
                t0 = self.t0_exponential if hasattr(self, 't0_exponential') else self.time_peak
                
                tau = self.tau_spin.value() if hasattr(self, 'tau_spin') else self.tau_exponential
                Vb = self.V_baseline
                
                # ✅ DEBUG: Stampa TUTTI i parametri incluso t0
                print(f"   📐 Exponential params: A={A:.4f}, t0={t0:.4f}, tau={tau:.4f}, Vb={Vb:.4f}")

                # Genera la curva con la funzione matematica
                # La funzione exp_return(t, A, t0, tau, Vb) si applica correttamente
                # sull'intero array 't' perché il decadimento inizia da 't0'.
                self.fitted_curve = self.exp_return(
                    self.fit_time_data,
                    A, t0, tau, Vb
                )
                
                # Aggiorna i parametri per la formula
                self.fitted_params = [A, t0, tau, Vb]
            
            # ===== ACTION POTENTIAL =====
            elif self.is_action_potential:
                # Usa i parametri correnti dalle spinbox
                A_sin = self.A_sin_spin.value() if hasattr(self, 'A_sin_spin') else self.A_sin
                f = self.f_spin.value() if hasattr(self, 'f_spin') else self.f_osc
                phi = self.phi_spin.value() if hasattr(self, 'phi_spin') else self.phi
                A_exp = self.A_exp_spin.value() if hasattr(self, 'A_exp_spin') else self.A_exp
                t_peak = self.t_peak_action_potential if hasattr(self, 't_peak_action_potential') else self.time_peak
                tau = self.tau_exp_spin.value() if hasattr(self, 'tau_exp_spin') else self.tau_action_potential
                Vb = self.V_baseline
                
                print(f"   📐 Action potential params: A_sin={A_sin:.3f}, f={f:.2f}, phi={phi:.2f}")
                print(f"      A_exp={A_exp:.3f}, t_peak={t_peak:.3f}, tau={tau:.3f}, Vb={Vb:.3f}")
                
                # Genera la curva con la funzione matematica
                self.fitted_curve = self.action_potential(
                    self.fit_time_data,
                    A_sin, f, phi, A_exp, t_peak, tau, Vb
                )
                
                # Aggiorna i parametri per la formula
                self.fitted_params = [A_sin, f, phi, A_exp, t_peak, tau, Vb]
            
            else:
                print("⚠️ No valid signal type selected")
                return
            
            # ✅ VALIDAZIONE: Controlla che la curva sia valida
            if np.any(np.isnan(self.fitted_curve)) or np.any(np.isinf(self.fitted_curve)):
                print("⚠️ Generated curve contains invalid values")
                return
            
            # ===== AGGIORNA VISUALIZZAZIONE =====
            # 1. Calcola R² con i parametri correnti
            self.calculate_r_squared()
            
            # 2. Aggiorna il plot
            self.plot_fitted_curve()
            
            # 3. Calcola energia
            self.compute_signal_energy(self.fit_signal_data, self.fit_time_data, self.V_baseline)
            
            # 4. Aggiorna formula
            self.update_formula_display()
            
            print(f"✅ Curve updated. R² = {self.r_squared:.4f}")
            
        except Exception as e:
            import traceback
            print(f"❌ Error updating curve: {str(e)}")
            print(traceback.format_exc())




    # ALGORITMO DI AUTODETECT E GESTIONE SEGNALI #

    #0. prendi gli array di cui hai bisogno
    def get_arrays(self):
        # ✅ FIX: Calcola dinamicamente quanti campioni "pre" abbiamo
        # Invece di assumere sempre 1 secondo, usa quello che effettivamente c'è
        
        first_time = self.total_time_data[0]
        # Trova l'indice dove iniziano i "veri" dati selezionati
        # (tutto prima è considerato "pre" per la baseline)
        
        # Calcola quanti punti equivalgono a 1 secondo (per riferimento)
        n_samples_per_second = int(self.sampling_rate)
        
        # Trova il primo punto che è >= al tempo di inizio della selezione vera
        # Assumiamo che i primi ~1s (o meno) siano "pre"
        # Cerca il punto dove il tempo fa un "salto" maggiore (transizione pre->main)
        
        # Strategia semplice: prendi gli ultimi 50 campioni prima del punto di 1s (se esiste)
        # altrimenti prendi i primi 50 campioni disponibili come "pre"
        
        if len(self.total_time_data) < 50:
            self.show_error_dialog("Data Error", "Not enough data for analysis (minimum 50 samples required).")
            self.close()
            return
        
        # Trova dove iniziano i dati "principali" (dopo ~1s dall'inizio dati)
        # Se abbiamo meno di n_samples_per_second campioni, usa tutti come "main"
        if len(self.total_time_data) >= n_samples_per_second + 50:
            # Caso normale: abbastanza dati per 1s pre + 50 campioni baseline
            split_idx = n_samples_per_second
            self.time_data_pre = self.total_time_data[split_idx-50:split_idx]
            self.signal_data_pre = self.total_signal_data[split_idx-50:split_idx]
            self.time_data = self.total_time_data[split_idx:]
            self.signal_data = self.total_signal_data[split_idx:]
        else:
            # Caso bordo: pochi dati (es. regione 0-0.5s)
            # Usa i primi 50 campioni come "pre", il resto come "main"
            if len(self.total_time_data) > 50:
                self.time_data_pre = self.total_time_data[:50]
                self.signal_data_pre = self.total_signal_data[:50]
                self.time_data = self.total_time_data[50:]
                self.signal_data = self.total_signal_data[50:]
            else:
                # Caso estremo: meno di 50 campioni totali
                # Usa i primi 10 come "pre" (se possibile), il resto come "main"
                pre_samples = min(10, len(self.total_time_data) // 3)
                self.time_data_pre = self.total_time_data[:pre_samples]
                self.signal_data_pre = self.total_signal_data[:pre_samples]
                self.time_data = self.total_time_data[pre_samples:]
                self.signal_data = self.total_signal_data[pre_samples:]
        
        print(f"   Array split: {len(self.time_data_pre)} pre-samples, {len(self.time_data)} main samples")
        print(f"   Time range: pre={self.time_data_pre[0]:.3f}-{self.time_data_pre[-1]:.3f}s, "
              f"main={self.time_data[0]:.3f}-{self.time_data[-1]:.3f}s")

    #1. inizializzo variabili
    def initialize_default_variables(self):
        self.amplitude_exponential = 0.00
        self.tau_exponential = 0.00
        self.t0_exponential = 0.00 
        
        self.A_sin = 0.00
        self.f_osc = 0.00
        self.phi = 0.0872 # 5 gradi in radianti default

        self.A_exp = 0.00
        self.tau_action_potential = 0.00
        self.t_peak_action_potential = 0.00

        #generali
        self.V_baseline = 0.00
        self.baseline_std = 0.00
        self.direction = "unknown"
        self.start_time = 0.00
        self.start_index = 0
        self.is_action_potential = False # se falso
        self.is_exponential_return = False # se falso
        self.end_time = 0.00
        self.end_index = 0
        self.V_max = 0.00
        self.V_min = 0.00
        self.time_peak = 0.00


    #2. calcola Vbaseline
    def get_V_baseline(self):
        #calcola la media dei valori
        self.V_baseline = np.mean(self.signal_data_pre)
        self.baseline_std = np.std(self.signal_data_pre)

    #3. calcolo direction
    def get_signal_direction(self):
        # Trova gli indici dei valori massimo e minimo
        self.idx_max = np.argmax(self.signal_data)
        self.idx_min = np.argmin(self.signal_data)
        
        # Ottieni i valori effettivi usando gli indici
        self.V_max = self.signal_data[self.idx_max]
        self.V_min = self.signal_data[self.idx_min]
        
        # calcola differenze
        self.diff_max = self.V_max - self.V_baseline  # entrambe potenzialmente positive
        self.diff_min = self.V_baseline - self.V_min

        if self.diff_max > self.diff_min and self.diff_max > 2*self.baseline_std:
            self.direction = "upward"
        elif self.diff_min > self.diff_max and self.diff_min > 2*self.baseline_std:
            self.direction = "downward"
        else:
            print("direction not found")
            return

    #4. calcolo tipo di segnale
    def get_signal_type(self):
        """Determina il tipo di segnale con validazioni migliorate"""
        
        print(f"\n=== Signal Type Detection ===")
        print(f"V_max: {self.V_max:.4f}V at index {self.idx_max}")
        print(f"V_min: {self.V_min:.4f}V at index {self.idx_min}")
        print(f"V_baseline: {self.V_baseline:.4f}V")
        print(f"diff_max: {self.diff_max:.4f}V (max - baseline)")
        print(f"diff_min: {self.diff_min:.4f}V (baseline - min)")
        print(f"baseline_std: {self.baseline_std:.4f}V")
        
        # ✅ VALIDAZIONE 1: Controlla se c'è davvero un segnale significativo
        MIN_SIGNAL_AMPLITUDE = 3 * self.baseline_std  # Minimo 3σ
        
        if self.diff_max < MIN_SIGNAL_AMPLITUDE and self.diff_min < MIN_SIGNAL_AMPLITUDE:
            print(f"⚠️ No significant signal detected (both deviations < {MIN_SIGNAL_AMPLITUDE:.4f}V)")
            return False
        
        # ✅ VALIDAZIONE 2: Determina la direzione principale
        if self.direction == "upward":
            main_deviation = self.diff_max
            secondary_deviation = self.diff_min
            main_idx = self.idx_max
            secondary_idx = self.idx_min
            
        elif self.direction == "downward":
            main_deviation = self.diff_min
            secondary_deviation = self.diff_max
            main_idx = self.idx_min
            secondary_idx = self.idx_max
        else:
            print("⚠️ Invalid direction")
            return False
        
        # ✅ VALIDAZIONE 3: Controlla l'ordine temporale dei picchi
        # Per un action potential, il rimbalzo deve venire DOPO il picco principale
        time_between_peaks = abs(self.time_data[secondary_idx] - self.time_data[main_idx])
        peaks_in_order = secondary_idx > main_idx  # Il secondo picco deve essere dopo
        
        print(f"Main peak at t={self.time_data[main_idx]:.3f}s")
        print(f"Secondary peak at t={self.time_data[secondary_idx]:.3f}s")
        print(f"Time between peaks: {time_between_peaks:.3f}s")
        print(f"Peaks in correct order: {peaks_in_order}")
        
        # ✅ VALIDAZIONE 4: Soglia per considerarlo Action Potential
        # Il rimbalzo deve essere significativo (almeno 30% del picco principale)
        REBOUND_RATIO_MIN = 0.3
        rebound_ratio = secondary_deviation / main_deviation if main_deviation > 0 else 0
        
        print(f"Rebound ratio: {rebound_ratio:.2f} (secondary/main)")
        
        # ✅ DECISIONE FINALE
        is_action_potential = (
            secondary_deviation > MIN_SIGNAL_AMPLITUDE and  # Rimbalzo significativo
            rebound_ratio > REBOUND_RATIO_MIN and          # Rimbalzo abbastanza grande
            peaks_in_order and                              # Ordine temporale corretto
            time_between_peaks < 2.0                        # Picchi non troppo distanti
        )
        
        if is_action_potential:
            print("✅ Detected: ACTION POTENTIAL")
            self.is_action_potential = True
            self.is_exponential_return = False
            
            # Il picco principale è quello del secondo indice (rimbalzo)
            self.idx_peak = secondary_idx
            self.time_peak = self.time_data[secondary_idx]
            
        else:
            print("✅ Detected: EXPONENTIAL RETURN")
            self.is_exponential_return = True
            self.is_action_potential = False
            
            # Il picco è quello principale
            self.idx_peak = main_idx
            self.time_peak = self.time_data[main_idx]
        
        # Imposta il valore nella combo box
        self.signalTypeComboBox.setCurrentText(
            "Action Potential" if self.is_action_potential else "Exponential Return"
        )
        
        return True

    #5. calcolo inizio e fine
    def get_start_end_times(self):
        """Calcola inizio e fine del segnale in modo robusto"""
        
        # Inizializzazione con valori di default sicuri
        self.start_index = 0
        self.end_index = len(self.signal_data) - 1
        self.start_time = self.time_data[0]
        self.end_time = self.time_data[-1]
        
        # ✅ VALIDAZIONE: Assicurati che ci siano i dati necessari
        if not hasattr(self, 'direction') or not hasattr(self, 'V_baseline'):
            print("⚠️ Direction or baseline not set")
            return
        
        if not hasattr(self, 'idx_max') or not hasattr(self, 'idx_min'):
            print("⚠️ Max/min indices not found")
            return
        
        # ✅ Determina il picco principale in base alla direzione
        if self.direction == "upward":
            peak_index = self.idx_max
            thresh_start = self.V_baseline + self.baseline_std
            thresh_end = self.V_baseline + self.baseline_std
            
            if self.is_action_potential:
                # Per action potential, cerca anche il secondo picco (rimbalzo negativo)
                search_rebound_from = peak_index
                for i in range(search_rebound_from, len(self.signal_data)):
                    if self.signal_data[i] < (self.V_baseline - self.baseline_std):
                        # Trovato il punto più basso del rimbalzo
                        break
                thresh_end = self.V_baseline - self.baseline_std
                
        elif self.direction == "downward":
            peak_index = self.idx_min
            thresh_start = self.V_baseline - self.baseline_std
            thresh_end = self.V_baseline - self.baseline_std
            
            if self.is_action_potential:
                # Per action potential, cerca anche il secondo picco (rimbalzo positivo)
                search_rebound_from = peak_index
                for i in range(search_rebound_from, len(self.signal_data)):
                    if self.signal_data[i] > (self.V_baseline + self.baseline_std):
                        # Trovato il punto più alto del rimbalzo
                        break
                thresh_end = self.V_baseline + self.baseline_std
        else:
            print("⚠️ Invalid direction")
            return
        
        # ✅ STEP 1: Trova l'INIZIO (backward dal picco)
        self.start_index = 0  # fallback
        for i in range(peak_index, -1, -1):
            if self.direction == "upward":
                if self.signal_data[i] <= thresh_start:
                    self.start_index = i
                    break
            elif self.direction == "downward":
                if self.signal_data[i] >= thresh_start:
                    self.start_index = i
                    break
        
        self.start_time = self.time_data[self.start_index]
        print(f"[DEBUG] Found start: index={self.start_index}, time={self.start_time:.3f}s, value={self.signal_data[self.start_index]:.4f}V")
        
        # ✅ STEP 2: Trova la FINE (backward dalla fine dell'array)
        # QUESTA È LA MODIFICA PRINCIPALE!
        self.end_index = len(self.signal_data) - 1  # fallback
        
        # Inizia dalla fine e vai indietro fino a trovare il punto di ritorno alla baseline
        for i in range(len(self.signal_data) - 1, peak_index, -1):
            if self.direction == "upward":
                if self.is_action_potential:
                    # Per action potential upward, cerca dove torna sopra la baseline negativa
                    if self.signal_data[i] >= thresh_end:
                        self.end_index = i
                        break
                else:  # exponential return
                    # Per exponential return upward, cerca dove torna alla baseline
                    if self.signal_data[i] <= thresh_end:
                        self.end_index = i
                        break
                        
            elif self.direction == "downward":
                if self.is_action_potential:
                    # Per action potential downward, cerca dove torna sotto la baseline positiva
                    if self.signal_data[i] <= thresh_end:
                        self.end_index = i
                        break
                else:  # exponential return
                    # Per exponential return downward, cerca dove torna alla baseline
                    if self.signal_data[i] >= thresh_end:
                        self.end_index = i
                        break
        
        self.end_time = self.time_data[self.end_index]
        print(f"[DEBUG] Found end: index={self.end_index}, time={self.end_time:.3f}s, value={self.signal_data[self.end_index]:.4f}V")
        
        # ✅ VALIDAZIONE FINALE: Verifica che start < end
        if self.start_index >= self.end_index:
            print(f"⚠️ Invalid indices after search: start={self.start_index}, end={self.end_index}")
            print(f"   This should NOT happen with backward search!")
            
            # Strategia di recupero: usa regione centrata sul picco
            span_points = min(200, len(self.signal_data) // 4)  # 200 punti o 25% del segnale
            self.start_index = max(0, peak_index - span_points)
            self.end_index = min(len(self.signal_data) - 1, peak_index + span_points)
            self.start_time = self.time_data[self.start_index]
            self.end_time = self.time_data[self.end_index]
            
            print(f"   Using fallback: ±{span_points} points around peak")
        
        # ✅ VALIDAZIONE: Verifica durata minima
        duration = self.end_time - self.start_time
        MIN_DURATION = 0.05  # 50ms minimo
        
        if duration < MIN_DURATION:
            print(f"⚠️ Duration too short: {duration*1000:.1f}ms")
            
            # Espandi simmetricamente intorno al picco
            samples_needed = int(MIN_DURATION * self.sampling_rate)
            half_samples = samples_needed // 2
            
            self.start_index = max(0, peak_index - half_samples)
            self.end_index = min(len(self.signal_data) - 1, peak_index + half_samples)
            self.start_time = self.time_data[self.start_index]
            self.end_time = self.time_data[self.end_index]
            
            print(f"   Expanded to {(self.end_time - self.start_time)*1000:.1f}ms")
        
        # ✅ VALIDAZIONE: Verifica durata massima ragionevole
        MAX_DURATION = 10.0  # 10 secondi massimo
        if duration > MAX_DURATION:
            print(f"⚠️ Duration too long: {duration:.1f}s, capping to {MAX_DURATION}s")
            
            # Mantieni start, limita end
            max_end_index = self.start_index + int(MAX_DURATION * self.sampling_rate)
            self.end_index = min(self.end_index, max_end_index)
            self.end_time = self.time_data[self.end_index]
        
        print(f"✅ Final range: Start: {self.start_time:.3f}s (idx={self.start_index}), "
            f"End: {self.end_time:.3f}s (idx={self.end_index}), "
            f"Duration: {(self.end_time - self.start_time)*1000:.1f}ms, "
            f"Points: {self.end_index - self.start_index + 1}")

    #6. determina gli array effettivi da usare per i fit
    def get_actual_arrays(self):
        """Prepara gli array per il fitting con validazioni multiple"""
        
        # ✅ VALIDAZIONE 1: Verifica ordine degli indici
        if self.start_index >= self.end_index:
            print(f"❌ CRITICAL: start_index ({self.start_index}) >= end_index ({self.end_index})")
            print(f"   This should have been caught earlier!")
            
            # Ultimo tentativo di recupero
            mid = len(self.signal_data) // 2
            span = min(100, len(self.signal_data) // 4)
            self.start_index = max(0, mid - span)
            self.end_index = min(len(self.signal_data) - 1, mid + span)
            self.start_time = self.time_data[self.start_index]
            self.end_time = self.time_data[self.end_index]
        
        # ✅ VALIDAZIONE 2: Numero minimo di punti
        MIN_POINTS = 50
        num_points = self.end_index - self.start_index + 1
        
        if num_points < MIN_POINTS:
            print(f"⚠️ Too few points: {num_points}, expanding to {MIN_POINTS}")
            
            # Espandi mantenendo il centro
            mid = (self.start_index + self.end_index) // 2
            half_span = MIN_POINTS // 2
            
            self.start_index = max(0, mid - half_span)
            self.end_index = min(len(self.signal_data) - 1, mid + half_span)
            self.start_time = self.time_data[self.start_index]
            self.end_time = self.time_data[self.end_index]
            
            num_points = self.end_index - self.start_index + 1
        
        # ✅ VALIDAZIONE 3: Durata temporale minima
        time_span = self.end_time - self.start_time
        MIN_TIME_SPAN = 0.05  # 50ms
        
        if time_span < MIN_TIME_SPAN:
            print(f"⚠️ Time span too short: {time_span*1000:.1f}ms")
            return False
        
        # Estrai gli array
        self.fit_time_data = self.time_data[self.start_index:self.end_index+1]
        self.fit_signal_data = self.signal_data[self.start_index:self.end_index+1]
        
        print(f"✅ Arrays prepared: {len(self.fit_time_data)} points, "
            f"{time_span*1000:.1f}ms span, "
            f"range: [{self.start_time:.3f}s, {self.end_time:.3f}s]")
        
        return True


    ###### VARIABILI PER IL FIT ######

    #### RITORNO ESPONENZIALE ####
    def setup_exponential_variables(self):
        self.amplitude_exponential = self.signal_data[self.idx_peak] - self.V_baseline
        print(f"Amplitude exponential: {self.amplitude_exponential:.3f}V")

        # Calcola il tempo in cui il segnale scende al 36.8% del valore massimo
        target_value = self.V_baseline + 0.632 * abs(self.amplitude_exponential) if self.direction == "upward" else self.V_baseline - 0.632 * abs(self.amplitude_exponential)
        if self.direction == "upward":
            for i in range(self.idx_peak, len(self.signal_data)):
                if self.signal_data[i] <= target_value:
                    self.time_to_target = self.time_data[i]
                    break
        elif self.direction == "downward":
            for i in range(self.idx_peak, len(self.signal_data)):
                if self.signal_data[i] >= target_value:
                    self.time_to_target = self.time_data[i]
                    break
        if hasattr(self, 'time_to_target'):
            self.tau_exponential = self.time_to_target - self.time_data[self.idx_peak]
        else:
            self.tau_exponential = 0.1  # default se non trovato


    ### POTENZIALE D'AZIONE ####
    def setup_action_potential_variables(self):

        #SINE
        if self.direction == "upward":
            v_first = self.signal_data[self.idx_max]  # picco massimo
            v_second = self.signal_data[self.idx_min]  # picco minimo successivo
            self.A_sin = (v_first - v_second) / 2
        elif self.direction == "downward":
            v_first = self.signal_data[self.idx_min]  # picco minimo
            v_second = self.signal_data[self.idx_max]  # picco massimo successivo
            self.A_sin = (v_first - v_second) / 2

        
        # Frequenza approssimativa
        if self.direction == "upward":
            for i in range(self.idx_max, len(self.signal_data)-1):
                if self.signal_data[i] <= self.V_baseline:
                    self.f_osc = 1 / (2*self.time_data[i])
                    break
        elif self.direction == "downward":
            for i in range(self.idx_min, len(self.signal_data)-1):
                if self.signal_data[i] >= self.V_baseline:
                    self.f_osc = 1 / (2*self.time_data[i])
                    break

        #phi è fisso

        #EXP
        self.A_exp = self.signal_data[self.idx_peak] - self.V_baseline

        # Calcola il tempo in cui il segnale scende al 36.8% del valore massimo
        target_value = self.V_baseline - 0.632 * abs(self.A_exp) if self.direction == "upward" else self.V_baseline + 0.632 * abs(self.A_exp)
        if self.direction == "upward":
            for i in range(self.idx_peak, len(self.signal_data)):
                if self.signal_data[i] >= target_value:
                    self.time_to_target = self.time_data[i]
                    break
        elif self.direction == "downward":
            for i in range(self.idx_peak, len(self.signal_data)):
                if self.signal_data[i] <= target_value:
                    self.time_to_target = self.time_data[i]
                    break
        if hasattr(self, 'time_to_target'):
            self.tau_action_potential = self.time_to_target - self.time_data[self.idx_peak]
        else:
            self.show_warning_dialog("Warning", "Could not determine tau for action potential. Default value used.")
            self.tau_action_potential = 0.1  # default se non trovato

    ############CALCOLI MATEMATICI #############

    #funzione semplice ritorno esponenziale
    @staticmethod
    def exp_return(t, A, t0, tau, Vb):
        return A * np.exp(-(t - t0)/tau) + Vb

    @staticmethod
    def action_potential(t, A_sin, f, phi, A_exp, t_peak, tau, Vb):
        # esempio base di funzione doppia
        res = np.zeros_like(t)
        for i in range(len(t)):
            if t[i] < t_peak:
                res[i] = A_sin * np.sin(2*np.pi*f*(t[i]-t[0]) + phi)
            else:
                res[i] = A_exp * np.exp(-(t[i]-t_peak)/tau) + Vb
        return res
    

    def fit_exponential_with_current_params(self):
        """Esegue il fitting usando i parametri correnti dalle spinbox (se esistono) o autodetect"""
        
        # Se le spinbox esistono, usa quei valori; altrimenti usa i valori autodetect
        if hasattr(self, 'A_spin'):
            A_init = self.A_spin.value()
            tau_init = self.tau_spin.value()
        else:
            A_init = self.amplitude_exponential
            tau_init = self.tau_exponential
        
        # Parametri iniziali per curve_fit
        p0 = [
            A_init,              # A
            self.time_peak,      # t0
            tau_init,            # tau
            self.V_baseline      # Vb
        ]

        print(f"[DEBUG] Fitting exponential with initial params: "
            f"A={p0[0]:.3f}, t0={p0[1]:.3f}, tau={p0[2]:.3f}, Vb={p0[3]:.3f}")

        # ✅ FIX: Seleziona solo i dati DAL PICCO IN POI per il fitting esponenziale
        try:
            # Trova l'indice di partenza che corrisponde al tempo del picco
            peak_start_index = np.argmin(np.abs(self.fit_time_data - self.time_peak))
            
            # Crea array locali per il fitting che partono dal picco
            exp_fit_time = self.fit_time_data[peak_start_index:]
            exp_fit_signal = self.fit_signal_data[peak_start_index:]

            if len(exp_fit_time) < 10: # Assicurati che ci siano abbastanza punti dopo il picco
                print("⚠️ Not enough data points after the peak to perform a reliable fit.")
                self.show_warning_dialog("Fitting Error", "Not enough data points after the peak for a reliable fit.")
                return False

            print(f"   🔬 Fitting exponential model on {len(exp_fit_time)} data points (from t_peak onwards).")

        except Exception as e:
            print(f"❌ Error slicing data for exponential fit: {e}")
            return False

        # 3. Range del segnale corrente (fisico, non normalizzato)
        V_min, V_max = np.min(self.total_signal_data), np.max(self.total_signal_data)
        V_span = V_max - V_min
        if V_span == 0:
            print("[WARN] Flat signal: skipping fit")
            return

        # 4. Limiti dinamici
        # A può essere negativo o positivo ma non superare ±1.2× l’escursione effettiva
        A_limit = 1.2 * V_span
        tau_max = max(0.05, (self.time_data[-1] - self.time_data[0]))  # non più lungo del segnale
        
        bounds = (
            [-A_limit,            # A_min
            self.time_peak - 0.5,  # t0_min (0.5s prima del picco)
            0.0001,              # tau_min
            V_min - V_span],     # Vb_min
            [A_limit,             # A_max
            self.time_peak + 0.5,  # t0_max
            tau_max,             # tau_max
            V_max + V_span]      # Vb_max
        )

        # 5. Pulizia iniziali (evita valori iniziali assurdi)
        p0[0] = np.clip(p0[0], -A_limit, A_limit)
        p0[2] = max(0.0001, min(p0[2], tau_max))
        
        # Esegui il fitting
        # ✅ USA GLI ARRAY SLICED (exp_fit_time, exp_fit_signal)
        self.fitted_params, pcov = curve_fit(
            self.exp_return, 
            exp_fit_time, 
            exp_fit_signal,
            p0=p0,
            bounds=bounds,
            maxfev=10000
        )
        
        # Genera la curva fittata
        self.fitted_curve = self.exp_return(self.fit_time_data, *self.fitted_params)
        
        # ✅ VALIDAZIONE: Controlla che la curva fittata non sia NaN/Inf
        if np.any(np.isnan(self.fitted_curve)) or np.any(np.isinf(self.fitted_curve)):
            print("❌ Fitted curve contains NaN or Inf values!")
            print(f"   Parameters: A={self.fitted_params[0]:.4f}, tau={self.fitted_params[2]:.4f}")
            # Usa parametri iniziali come fallback
            self.fitted_params = p0
            self.fitted_curve = self.exp_return(self.fit_time_data, *self.fitted_params)
        
        # Calcola R²
        self.calculate_r_squared()
        
        # ✅ VALIDAZIONE R²
        if self.r_squared == -999.0:
            print("❌ R² calculation failed!")
            return False
        
        if self.r_squared < -0.5:
            print(f"❌ R² too negative ({self.r_squared:.4f}) - model completely wrong!")
            self.show_warning_dialog(
                "Fitting Failed",
                f"The model fits worse than a flat line (R²={self.r_squared:.4f}).\n"
                "Please check:\n"
                "• Signal type selection (Exponential vs Action Potential)\n"
                "• Time range selection\n"
                "• Initial parameter values"
            )
            return False
        
        # Aggiorna le variabili interne
        self.amplitude_exponential = self.fitted_params[0]
        self.tau_exponential = self.fitted_params[2]
        self.t0_exponential = self.fitted_params[1]
        self.V_baseline = self.fitted_params[3]
        
        # Aggiorna il dizionario
        self.params_dict_exp = {
            'A': self.fitted_params[0],
            't0': self.fitted_params[1],
            'tau': self.fitted_params[2],
            'Vbaseline': self.fitted_params[3]
        }
        
        # Aggiorna le spinbox SENZA triggare i segnali
        if hasattr(self, 'A_spin'):
            self.A_spin.blockSignals(True)
            self.A_spin.setValue(self.fitted_params[0])
            self.A_spin.blockSignals(False)
            
            self.tau_spin.blockSignals(True)
            self.tau_spin.setValue(self.fitted_params[2])
            self.tau_spin.blockSignals(False)
        
        print(f"Exponential fit: A={self.fitted_params[0]:.4f}, tau={self.fitted_params[2]:.4f}")
        return True


    def fit_action_potential_with_current_params(self):
        """Esegue il fitting del potenziale d'azione usando i parametri correnti con bounds robusti"""
        
        # Se le spinbox esistono, usa quei valori; altrimenti usa i valori autodetect
        if hasattr(self, 'A_sin_spin'):
            A_sin_init = self.A_sin_spin.value()
            f_init = self.f_spin.value()
            phi_init = self.phi_spin.value()
            A_exp_init = self.A_exp_spin.value()
            tau_init = self.tau_exp_spin.value()
        else:
            A_sin_init = self.A_sin
            f_init = self.f_osc
            phi_init = self.phi
            A_exp_init = self.A_exp
            tau_init = self.tau_action_potential
        
        # Parametri iniziali per curve_fit
        p0 = [
            A_sin_init,          # A_sin
            f_init,              # f
            phi_init,            # phi
            A_exp_init,          # A_exp
            self.time_peak,      # t_peak
            tau_init,            # tau
            self.V_baseline      # Vb
        ]
        
        print(f"[DEBUG] Fitting action potential with initial params:")
        print(f"        A_sin={p0[0]:.3f}, f={p0[1]:.3f}, phi={p0[2]:.3f}")
        print(f"        A_exp={p0[3]:.3f}, t_peak={p0[4]:.3f}, tau={p0[5]:.3f}, Vb={p0[6]:.3f}")
        
        # ✅ Range del segnale corrente (fisico)
        V_min, V_max = np.min(self.total_signal_data), np.max(self.total_signal_data)
        V_span = V_max - V_min
        
        if V_span == 0:
            print("[WARN] Flat signal: skipping fit")
            self.fitted_params = p0
            return
        
        # ✅ Limiti dinamici per ciascun parametro
        # A_sin: ampiezza della componente sinusoidale (±1.2× l'escursione del segnale)
        A_sin_limit = 1.2 * V_span
        
        # f: frequenza (ragionevole tra 0.1 Hz e 50 Hz per segnali biologici)
        f_min = 0.1
        f_max = 50.0
        
        # phi: fase (tra -π e +π)
        phi_min = -np.pi
        phi_max = np.pi
        
        # A_exp: ampiezza della componente esponenziale (±1.2× l'escursione del segnale)
        A_exp_limit = 1.2 * V_span
        
        # t_peak: tempo del picco (con margine di ±1s dal valore rilevato)
        t_peak_min = max(self.time_data[0], self.time_peak - 1.0)
        t_peak_max = min(self.time_data[-1], self.time_peak + 1.0)
        
        # tau: costante di tempo (tra 0.1ms e durata totale del segnale)
        tau_min = 0.0001  # 0.1ms
        tau_max = max(0.05, (self.time_data[-1] - self.time_data[0]))
        
        # Vb: baseline (può variare nell'intero range del segnale ± span)
        Vb_min = V_min - V_span
        Vb_max = V_max + V_span
        
        # ✅ Definizione dei bounds
        bounds = (
            [                    # lower bounds
                -A_sin_limit,    # A_sin_min
                f_min,           # f_min
                phi_min,         # phi_min
                -A_exp_limit,    # A_exp_min
                t_peak_min,      # t_peak_min
                tau_min,         # tau_min
                Vb_min           # Vb_min
            ],
            [                    # upper bounds
                A_sin_limit,     # A_sin_max
                f_max,           # f_max
                phi_max,         # phi_max
                A_exp_limit,     # A_exp_max
                t_peak_max,      # t_peak_max
                tau_max,         # tau_max
                Vb_max           # Vb_max
            ]
        )
        
        # ✅ Pulizia parametri iniziali (garantisce che siano dentro i bounds)
        p0[0] = np.clip(p0[0], -A_sin_limit, A_sin_limit)      # A_sin
        p0[1] = np.clip(p0[1], f_min, f_max)                   # f
        p0[2] = np.clip(p0[2], phi_min, phi_max)               # phi
        p0[3] = np.clip(p0[3], -A_exp_limit, A_exp_limit)      # A_exp
        p0[4] = np.clip(p0[4], t_peak_min, t_peak_max)         # t_peak
        p0[5] = np.clip(p0[5], tau_min, tau_max)               # tau
        p0[6] = np.clip(p0[6], Vb_min, Vb_max)                 # Vb
        
        print(f"[DEBUG] Clipped initial params:")
        print(f"        A_sin={p0[0]:.3f}, f={p0[1]:.3f}, phi={p0[2]:.3f}")
        print(f"        A_exp={p0[3]:.3f}, t_peak={p0[4]:.3f}, tau={p0[5]:.3f}, Vb={p0[6]:.3f}")
        
        # ✅ Esegui il fitting con bounds
        try:
            self.fitted_params, pcov = curve_fit(
                self.action_potential, 
                self.fit_time_data, 
                self.fit_signal_data,
                p0=p0,
                bounds=bounds,
                maxfev=15000  # Aumentato per fitting più complesso
            )
            
            # ✅ Verifica se la covarianza è valida
            if np.any(np.isinf(pcov)) or np.any(np.isnan(pcov)):
                print("⚠️ Covariance estimation failed, using initial parameters")
                self.fitted_params = p0
                
        except RuntimeError as e:
            print(f"⚠️ Curve fitting failed: {str(e)}, using initial parameters")
            self.fitted_params = p0
        except ValueError as e:
            print(f"⚠️ Invalid parameters: {str(e)}, using initial parameters")
            self.fitted_params = p0
        
        # Genera la curva fittata
        self.fitted_curve = self.action_potential(self.fit_time_data, *self.fitted_params)
        
        # Calcola R²
        self.calculate_r_squared()
        
        print(f"[DEBUG] Fitted params: A_sin={self.fitted_params[0]:.3f}, f={self.fitted_params[1]:.3f}, "
            f"A_exp={self.fitted_params[3]:.3f}, tau={self.fitted_params[5]:.3f}")
        print(f"[DEBUG] R² = {self.r_squared:.4f}")
        
        # ✅ SEMPRE crea i dizionari, anche se il fit è pessimo
        self.A_sin = self.fitted_params[0]
        self.f_osc = self.fitted_params[1]
        self.phi = self.fitted_params[2]
        self.A_exp = self.fitted_params[3]
        self.t_peak_action_potential = self.fitted_params[4]
        self.tau_action_potential = self.fitted_params[5]
        self.V_baseline = self.fitted_params[6]
        
        self.params_dict_sin_action_potential = {
            'A_sin': self.fitted_params[0],
            'f': self.fitted_params[1],
            'phi': self.fitted_params[2]
        }
        
        self.params_dict_exp_action_potential = {
            'A_exp': self.fitted_params[3],
            't_peak': self.fitted_params[4],
            'tau': self.fitted_params[5],
            'V_baseline': self.fitted_params[6]
        }
        
        # ✅ Se R² è basso, SEGNALA il problema ma NON chiudere
        if self.r_squared < 0.5:
            print(f"⚠️ WARNING: Low R² ({self.r_squared:.4f})")
            # NON chiudere, lascia che perform_curve_fitting gestisca il problema
            return False  # ✅ Indica che il fit è fallito
        
        # Aggiorna le spinbox SENZA triggare i segnali
        if hasattr(self, 'A_sin_spin'):
            self.A_sin_spin.blockSignals(True)
            self.A_sin_spin.setValue(self.fitted_params[0])
            self.A_sin_spin.blockSignals(False)
            
            self.f_spin.blockSignals(True)
            self.f_spin.setValue(self.fitted_params[1])
            self.f_spin.blockSignals(False)
            
            self.phi_spin.blockSignals(True)
            self.phi_spin.setValue(self.fitted_params[2])
            self.phi_spin.blockSignals(False)
            
            self.A_exp_spin.blockSignals(True)
            self.A_exp_spin.setValue(self.fitted_params[3])
            self.A_exp_spin.blockSignals(False)
            
            self.tau_exp_spin.blockSignals(True)
            self.tau_exp_spin.setValue(self.fitted_params[5])
            self.tau_exp_spin.blockSignals(False)
        
        print(f"✅ Action potential fit completed")
        return True  # ✅ Indica che il fit è riuscito

    def perform_curve_fitting(self):
        """Esegue il fitting automatico completo (solo al primo avvio)"""
        try:
            # Analisi completa del segnale
            self.get_signal_direction()
            print("1. Signal direction:", self.direction)
            
            # ✅ Verifica che il tipo sia stato rilevato correttamente
            signal_type_detected = self.get_signal_type()
            
            if not signal_type_detected:
                print("⚠️ Could not determine signal type reliably")
                self.show_warning_dialog(
                    "Signal Type Unknown",
                    "Could not reliably detect if this is an Action Potential or Exponential Return.\n"
                    "The automatic fitting will be skipped.\n"
                    "You can manually select the signal type and click 'AutoFit'."
                )
                return False
            
            print("2. Signal type - Is Action Potential:", self.is_action_potential, 
                "Is Exponential Return:", self.is_exponential_return)
            
            self.get_start_end_times()
            print(f"3. Start time: {self.start_time:.3f}s, End time: {self.end_time:.3f}s")
            
            self.get_actual_arrays()
            
            # ✅ VALIDAZIONE: Verifica che ci siano dati sufficienti
            MIN_POINTS_FOR_FIT = 50  # Minimo 50 punti per un fitting affidabile
            time_span = self.end_time - self.start_time
            MIN_TIME_SPAN = 0.1  # Minimo 100ms di durata
            
            if len(self.fit_time_data) < MIN_POINTS_FOR_FIT:
                self.show_warning_dialog(
                    "Insufficient Data", 
                    f"Only {len(self.fit_time_data)} points available for fitting (minimum {MIN_POINTS_FOR_FIT} required).\n"
                    "The automatic fitting will be skipped. You can manually adjust parameters and click 'AutoFit'."
                )
                print(f"⚠️ Skipping automatic fit: only {len(self.fit_time_data)} points available")
                return False
            
            if time_span < MIN_TIME_SPAN:
                self.show_warning_dialog(
                    "Time Range Too Small",
                    f"Time span is only {time_span*1000:.1f}ms (minimum {MIN_TIME_SPAN*1000:.0f}ms required).\n"
                    "The automatic fitting will be skipped. You can manually adjust parameters and click 'AutoFit'."
                )
                print(f"⚠️ Skipping automatic fit: time span too small ({time_span:.4f}s)")
                return False
            
            # Setup e fitting
            if self.is_exponential_return:
                print("Fitting exponential return model...")
                self.setup_exponential_variables()
                
                # ✅ Validazione dei parametri iniziali
                if not self._validate_exponential_params():
                    return False
                
                self.fit_exponential_with_current_params()
                
                # ✅ Verifica R² dopo il fit
                if hasattr(self, 'r_squared') and (self.r_squared < 0 or self.r_squared > 1.1 or np.isnan(self.r_squared) or np.isinf(self.r_squared)):
                    print(f"⚠️ Invalid R² value: {self.r_squared}")
                    self.show_warning_dialog(
                        "Fitting Failed",
                        f"The fitting produced an invalid R² value ({self.r_squared:.4f}).\n"
                        "Please manually adjust the parameters and try again."
                    )
                    return False
                
                self.signalTypeComboBox.setCurrentText("Exponential Return") 
                self.setup_exponential_return_ui()
                
            elif self.is_action_potential:
                print("Fitting action potential model...")
                self.setup_action_potential_variables()
                
                # ✅ Validazione dei parametri iniziali
                if not self._validate_action_potential_params():
                    return False
                
                self.fit_action_potential_with_current_params()
                
                # ✅ Verifica R² dopo il fit
                if hasattr(self, 'r_squared') and (self.r_squared < 0 or self.r_squared > 1.1 or np.isnan(self.r_squared) or np.isinf(self.r_squared)):
                    print(f"⚠️ Invalid R² value: {self.r_squared}")
                    self.show_warning_dialog(
                        "Fitting Failed",
                        f"The fitting produced an invalid R² value ({self.r_squared:.4f}).\n"
                        "Please manually adjust the parameters and try again."
                    )
                    return False
                
                self.signalTypeComboBox.setCurrentText("Action Potential")
                self.setup_action_potential_ui()
            else:
                print("Signal type not recognized.")
                return False
            
            # Aggiorna il grafico
            self.plot_fitted_curve()

            # Calcola l'energia
            self.compute_signal_energy(self.fit_signal_data, self.fit_time_data, self.V_baseline)

            # Aggiorna la visualizzazione della formula
            self.update_formula_display()

            # ✅ AGGIUNGI: Mostra le linee descrittive subito dopo il fit
            self._update_reference_lines()
            self._update_fit_region_display()
            self._update_peak_line()
            
            print(f"✅ Fit completed. R² = {self.r_squared:.4f}")
            return True
            
        except Exception as e:
            import traceback
            print(f"❌ Error during curve fitting: {str(e)}")
            print(traceback.format_exc())
            self.show_warning_dialog(
                "Fitting Error",
                f"An error occurred during automatic fitting:\n{str(e)}\n\n"
                "The dialog will open without automatic fitting. "
                "You can manually adjust parameters and click 'AutoFit'."
            )
            return False

    def _validate_exponential_params(self):
        """Valida i parametri iniziali per il fitting esponenziale"""
        MAX_AMPLITUDE = 3
        
        # Controlla ampiezza
        if abs(self.amplitude_exponential) > MAX_AMPLITUDE:
            print(f"⚠️ WARNING: Amplitude {self.amplitude_exponential:.4f}V exceeds system limits (±{MAX_AMPLITUDE}V)")
            self.show_warning_dialog(
                "Invalid Amplitude",
                f"Detected amplitude ({self.amplitude_exponential:.4f}V) exceeds system limits (±{MAX_AMPLITUDE}V).\n"
                "This indicates the signal may not be suitable for exponential fitting.\n"
                "Automatic fitting will be skipped."
            )
            return False
        
        # Controlla tau
        if self.tau_exponential <= 0 or self.tau_exponential > 10:
            print(f"⚠️ WARNING: Invalid tau value: {self.tau_exponential:.4f}s")
            self.show_warning_dialog(
                "Invalid Time Constant",
                f"Detected time constant (τ = {self.tau_exponential:.4f}s) is invalid.\n"
                "Automatic fitting will be skipped."
            )
            return False
        
        # Controlla baseline
        if abs(self.V_baseline) > MAX_AMPLITUDE:
            print(f"⚠️ WARNING: Baseline {self.V_baseline:.4f}V exceeds system limits")
            return False
        
        return True

    def _validate_action_potential_params(self):
        """Valida i parametri iniziali per il fitting del potenziale d'azione"""
        MAX_AMPLITUDE = 3        
        # Controlla ampiezze
        if abs(self.A_sin) > MAX_AMPLITUDE or abs(self.A_exp) > MAX_AMPLITUDE:
            print(f"⚠️ WARNING: Amplitudes exceed system limits")
            self.show_warning_dialog(
                "Invalid Amplitudes",
                f"Detected amplitudes (A_sin={self.A_sin:.4f}V, A_exp={self.A_exp:.4f}V) exceed system limits.\n"
                "Automatic fitting will be skipped."
            )
            return False
        
        # Controlla frequenza
        if self.f_osc <= 0 or self.f_osc > 100:
            print(f"⚠️ WARNING: Invalid frequency: {self.f_osc:.4f}Hz")
            self.show_warning_dialog(
                "Invalid Frequency",
                f"Detected frequency ({self.f_osc:.4f}Hz) is invalid.\n"
                "Automatic fitting will be skipped."
            )
            return False
        
        # Controlla tau
        if self.tau_action_potential <= 0 or self.tau_action_potential > 10:
            print(f"⚠️ WARNING: Invalid tau: {self.tau_action_potential:.4f}s")
            return False
        
        return True


    def calculate_r_squared(self):
        """Calcola il coefficiente di determinazione R² con validazione robusta"""
        
        # ✅ SELEZIONA I DATI CORRETTI PER IL CALCOLO
        if self.is_exponential_return:
            # Per l'esponenziale, R² ha senso solo DOPO il picco
            try:
                peak_start_index = np.argmin(np.abs(self.fit_time_data - self.time_peak))
                y_observed = self.fit_signal_data[peak_start_index:]
                y_predicted = self.fitted_curve[peak_start_index:]
                print("   🔬 Calculating R² for exponential model (post-peak data only).")
            except (ValueError, IndexError):
                print("⚠️ Could not slice data for post-peak R² calculation.")
                self.r_squared = -999.0
                return self.r_squared
        else:
            # Per altri modelli (es. Action Potential), usa l'intero range
            y_observed = self.fit_signal_data
            y_predicted = self.fitted_curve
            print("   🔬 Calculating R² for the full selected range.")

        # ✅ Validazione preliminare
        if len(y_observed) < 10:
            print("⚠️ Too few data points for R² calculation")
            self.r_squared = -999.0  # Valore sentinella
            return self.r_squared
        
        if y_predicted is None or len(y_observed) != len(y_predicted):
            print("⚠️ Mismatch between observed and predicted data for R²")
            self.r_squared = -999.0
            return self.r_squared
        
        # Calcola la media dei dati osservati
        y_mean = np.mean(y_observed)
        
        # ✅ Controlla che ci sia varianza nei dati
        if np.std(y_observed) < 1e-10:
            print("⚠️ Signal has no variance (flat signal)")
            self.r_squared = 1.0 # Se il segnale è piatto e il modello lo fitta, R² è 1
            return self.r_squared
        
        # Calcola la somma totale dei quadrati (SST)
        sst = np.sum((y_observed - y_mean) ** 2)
        
        # ✅ Verifica che SST non sia zero
        if sst < 1e-10:
            print("⚠️ SST is zero or too small")
            self.r_squared = -999.0
            return self.r_squared
        
        # Calcola la somma dei quadrati dei residui (SSR)
        residuals = y_observed - y_predicted
        ssr = np.sum(residuals ** 2)
        
        # ✅ Controlla overflow/underflow
        if np.isnan(ssr) or np.isinf(ssr):
            print("⚠️ SSR is NaN or Inf")
            self.r_squared = -999.0
            return self.r_squared
        
        # Calcola R²
        self.r_squared = 1 - (ssr / sst)
        
        # ✅ Analisi dettagliata per debug
        print(f"\n=== R² Calculation Debug ===")
        print(f"Data points: {len(self.fit_signal_data)}")
        print(f"Signal mean: {y_mean:.4f}")
        print(f"Signal std: {np.std(self.fit_signal_data):.4f}")
        print(f"SST (total variance): {sst:.6f}")
        print(f"SSR (residual error): {ssr:.6f}")
        print(f"Mean squared error: {ssr/len(self.fit_signal_data):.6f}")
        print(f"Root mean squared error: {np.sqrt(ssr/len(self.fit_signal_data)):.6f}")
        print(f"Max residual: {np.max(np.abs(residuals)):.4f}")
        print(f"R² = {self.r_squared:.6f}")
        
        # ✅ Interpretazione del risultato
        if self.r_squared < -1.0:
            print("❌ CRITICAL: R² < -1 indicates severe model failure")
        elif self.r_squared < 0:
            print("⚠️ WARNING: Negative R² - model worse than mean baseline")
        elif self.r_squared < 0.5:
            print("⚠️ WARNING: Low R² - poor fit quality")
        elif self.r_squared < 0.8:
            print("✓ Acceptable fit")
        elif self.r_squared < 0.95:
            print("✓ Good fit")
        else:
            print("✓ Excellent fit")
        
        return self.r_squared

    def plot_fitted_curve(self):
        """Aggiorna il grafico con la curva fittata"""
        # Rimuovi la curva fittata se già presente
        if hasattr(self, 'fit_curve') and self.fit_curve is not None:
            self.voltage_plot.plot_widget.removeItem(self.fit_curve)
        
        # Dati di default da plottare
        plot_time = self.fit_time_data
        plot_curve = self.fitted_curve

        # ✅ Se è un ritorno esponenziale, taglia la curva a t_peak
        if self.is_exponential_return:
            try:
                # Trova l'indice di partenza che corrisponde al tempo del picco
                peak_start_index = np.argmin(np.abs(self.fit_time_data - self.time_peak))
                
                # Seleziona solo i dati dal picco in poi per il plot
                plot_time = self.fit_time_data[peak_start_index:]
                plot_curve = self.fitted_curve[peak_start_index:]
                print("   🎨 Plotting exponential curve from t_peak onwards.")
            except (ValueError, IndexError):
                print("⚠️ Could not slice curve for plotting. Plotting full curve as fallback.")
                pass # In caso di errore, plotta comunque l'intera curva

        # Crea una nuova curva per il fit con i dati (potenzialmente tagliati)
        pen = pg.mkPen(color='r', width=2)
        self.fit_curve = self.voltage_plot.plot_widget.plot(
            plot_time, 
            plot_curve, 
            pen=pen, 
            name="Fitted Curve"
        )
        
        # Aggiorna il titolo con informazioni sul fit
        model_type = "Exponential Return" if self.is_exponential_return else "Action Potential"
        title = f"{model_type} Fit"
        self.voltage_plot.plot_widget.setTitle(title)


    def update_formula_display(self):
        """Aggiorna il widget della formula con la formula matematica corrente e l'R²"""
        
        if not hasattr(self, 'r_squared'):
            self.r_squared = 0.0
        
        # Imposta il formato HTML per il QLabel
        self.formulaWidget.setTextFormat(Qt.RichText)
        self.formulaWidget.setAlignment(Qt.AlignCenter)
        
        # Ottieni il colore del testo dal tema corrente
        theme_colors = self.theme_manager.get_theme_colors()
        text_color = theme_colors.get('foreground', '#ffffff')
        
        # Stile CSS base
        style = f"""
        <div style='color: {text_color}; font-size: 12pt; padding: 10px;'>
        """
        
        if self.is_exponential_return:
            # Formula ritorno esponenziale: V(t) = A·e^(-(t-t₀)/τ) + V_b
            A = self.fitted_params[0] if hasattr(self, 'fitted_params') else self.amplitude_exponential
            t0 = self.fitted_params[1] if hasattr(self, 'fitted_params') else self.time_peak
            tau = self.fitted_params[2] if hasattr(self, 'fitted_params') else self.tau_exponential
            Vb = self.fitted_params[3] if hasattr(self, 'fitted_params') else self.V_baseline
            
            # ✅ Converti Vb in float PRIMA di formattare
            Vb = float(Vb)
            # ✅ Crea la stringa formattata con segno
            Vb_str = f"+ {Vb:.4f}" if Vb >= 0 else f"- {abs(Vb):.4f}"

            formula = f"""
            <span style='font-size: 16pt; font-weight: 500;'>Exponential Return Model:</span><br/>
            <span style='font-size: 14pt;'>
                V(t) = {A:.4f} · e<sup>-(t-{t0:.3f})/{tau:.3f}</sup> {Vb_str}
            </span>
            <br/>
            <span style='font-size: 14pt; font-weight: bold;'>R² = {self.r_squared:.4f}</span>
            """
            
        elif self.is_action_potential:
            # Formula potenziale d'azione: parte sinusoidale + parte esponenziale
            if hasattr(self, 'fitted_params'):
                A_sin = self.fitted_params[0]
                f = self.fitted_params[1]
                phi = self.fitted_params[2]
                A_exp = self.fitted_params[3]
                t_peak = self.fitted_params[4]
                tau = self.fitted_params[5]
                Vb = float(self.fitted_params[6])
                # ✅ Crea la stringa formattata con segno
                Vb_str = f"+ {Vb:.3f}" if Vb >= 0 else f"- {abs(Vb):.3f}"
            else:
                A_sin = self.A_sin
                f = self.f_osc
                phi = self.phi
                A_exp = self.A_exp
                t_peak = self.time_peak
                tau = self.tau_action_potential
                Vb = float(self.V_baseline)
                # ✅ Crea la stringa formattata con segno
                Vb_str = f"+ {Vb:.3f}" if Vb >= 0 else f"- {abs(Vb):.3f}"
            
            formula = f"""
            <span style='font-size: 16pt; font-weight: 500;'>Action Potential Model:</span><br/>
            <span style='font-size: 14pt;'>
                V(t) = {A_sin:.3f} · sin(2π·{f:.2f}·t + {phi:.2f})&nbsp;&nbsp;for t &lt; {t_peak:.3f}s
                &nbsp;&nbsp;|&nbsp;&nbsp;
                V(t) = {A_exp:.3f} · e<sup>-(t-{t_peak:.3f})/{tau:.3f}</sup> {Vb_str}&nbsp;&nbsp;for t ≥ {t_peak:.3f}s
            </span>
            <br/>
            <span style='font-size: 14pt; font-weight: bold;'>R² = {self.r_squared:.4f}</span>
            """
        else:
            formula = "<i>No model selected</i>"
        
        # Combina stile e formula
        html_content = style + formula + "</div>"
        
        # Imposta il contenuto HTML
        self.formulaWidget.setText(html_content)


    ###### INTEGRALE NUMERICA ######
    def compute_signal_energy(self, voltage_array, time_array, V_baseline):
        """
        Calcola l'energia di un segnale come integrale numerico della deviazione
        quadrata rispetto a una baseline, gestendo correttamente i casi limite.

        Args:
            voltage_array (np.ndarray): Array delle tensioni.
            time_array (np.ndarray): Array dei tempi corrispondenti.
            V_baseline (float): Il valore della baseline da considerare come zero.

        Returns:
            tuple: L'energia calcolata per la prima e la seconda fase (unità V²·s).
        """
        # 1. Centra il segnale sulla baseline e calcola la deviazione quadrata
        squared_deviation = np.square(voltage_array - V_baseline)

        # 2. Dividi il segnale in due fasi rispetto al picco
        t_peak = self.time_peak
        first_phase_mask = time_array < t_peak
        second_phase_mask = time_array >= t_peak

        # ✅ FIX: Calcola l'energia per ogni fase solo se ci sono dati
        
        # --- Prima Fase (prima del picco) ---
        time_first_phase = time_array[first_phase_mask]
        if len(time_first_phase) > 1:
            self.first_phase_energy = np.trapezoid(squared_deviation[first_phase_mask], time_first_phase)
        else:
            self.first_phase_energy = 0.0  # Se non ci sono punti o ce n'è uno solo, l'area è zero

        # --- Seconda Fase (dal picco in poi) ---
        time_second_phase = time_array[second_phase_mask]
        if len(time_second_phase) > 1:
            self.second_phase_energy = np.trapezoid(squared_deviation[second_phase_mask], time_second_phase)
        else:
            self.second_phase_energy = 0.0 # Se non ci sono punti o ce n'è uno solo, l'area è zero

        # 3. Aggiorna le etichette nella UI
        try:
            if self.is_exponential_return:
                self.spike_energy_label.setText(f"Spike phase: {self.first_phase_energy:.4f} V²·s")
                self.exp_energy_label.setText(f"Exponential phase: {self.second_phase_energy:.4f} V²·s")
            elif self.is_action_potential:
                self.sine_action_potential_energy_label.setText(f"Sine phase: {self.first_phase_energy:.4f} V²·s")
                self.exp_action_potential_energy_label.setText(f"Exponential phase: {self.second_phase_energy:.4f} V²·s")
        except AttributeError:
            # Questo è normale se la UI non è ancora stata creata per il modello specifico
            pass

        return self.first_phase_energy, self.second_phase_energy





    #UI per le variabili generali e specifiche

    ############# VARIABILI GENERALI #############
    def setup_general_variables_ui(self):
        """Mostra variabili generali EDITABILI con callbacks per aggiornare parametri dipendenti"""
        
        # ✅ BLOCCA segnali durante setup
        self._updating_ui = True

        # Disattiva il pulsante AutoFit e Save
        self.autoFitButton.setEnabled(False)
        self.saveButton.setEnabled(False)
        
        try:
            layout = self.resultsWidget.layout()
            if layout is None:
                layout = QVBoxLayout(self.resultsWidget)
                self.resultsWidget.setLayout(layout)

            # ✅ Rimuovi widget esistenti
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.setParent(None)

            general_label = QLabel("<b>General Variables:</b>")
            layout.addWidget(general_label)

            # === MAX VOLTAGE ===
            max_voltage_label = QLabel("Max Voltage (V<sub>max</sub>):")
            max_voltage_label.setTextFormat(Qt.RichText)
            layout.addWidget(max_voltage_label)
            self.max_voltage_spin = QDoubleSpinBox()
            self.max_voltage_spin.setDecimals(4)
            self.max_voltage_spin.setRange(-3, 3)
            self.max_voltage_spin.setValue(self.V_max)
            self.max_voltage_spin.setSingleStep(0.01)
            self.max_voltage_spin.setReadOnly(False)
            layout.addWidget(self.max_voltage_spin)

            # === MIN VOLTAGE ===
            min_voltage_label = QLabel("Min Voltage (V<sub>min</sub>):")
            min_voltage_label.setTextFormat(Qt.RichText)
            layout.addWidget(min_voltage_label)
            self.min_voltage_spin = QDoubleSpinBox()
            self.min_voltage_spin.setDecimals(4)
            self.min_voltage_spin.setRange(-1.65, 1.65)
            self.min_voltage_spin.setValue(self.V_min)
            self.min_voltage_spin.setSingleStep(0.01)
            self.min_voltage_spin.setReadOnly(False)  
            layout.addWidget(self.min_voltage_spin)

            # === BASELINE ===
            baseline_label = QLabel("Baseline (V<sub>b</sub>):")
            baseline_label.setTextFormat(Qt.RichText)
            layout.addWidget(baseline_label)
            self.baseline_spin = QDoubleSpinBox()
            self.baseline_spin.setDecimals(4)
            self.baseline_spin.setRange(-1.65, 1.65)
            self.baseline_spin.setValue(self.V_baseline)
            self.baseline_spin.setSingleStep(0.01)
            self.baseline_spin.setReadOnly(False)  
            layout.addWidget(self.baseline_spin)

            # === START TIME ===
            start_time_label = QLabel("Start Time (t<sub>start</sub>):")
            start_time_label.setTextFormat(Qt.RichText)
            layout.addWidget(start_time_label)
            self.start_time_spin = QDoubleSpinBox()
            self.start_time_spin.setRange(self.time_data[0], self.time_data[-1])
            self.start_time_spin.setDecimals(4)
            self.start_time_spin.setValue(self.start_time)
            self.start_time_spin.setSingleStep(0.01)
            self.start_time_spin.setReadOnly(False) 
            layout.addWidget(self.start_time_spin)

            # === END TIME ===
            stop_time_label = QLabel("End Time (t<sub>end</sub>):")
            stop_time_label.setTextFormat(Qt.RichText)
            layout.addWidget(stop_time_label)
            self.stop_time_spin = QDoubleSpinBox()
            self.stop_time_spin.setRange(0, self.time_data[-1])
            self.stop_time_spin.setDecimals(4)
            self.stop_time_spin.setValue(self.end_time)
            self.stop_time_spin.setSingleStep(0.01)
            self.stop_time_spin.setReadOnly(False) 
            layout.addWidget(self.stop_time_spin)

            # === PEAK TIME ===
            peak_time_label = QLabel("Peak Time (t<sub>peak</sub>):")
            peak_time_label.setTextFormat(Qt.RichText)
            layout.addWidget(peak_time_label)
            self.peak_time_spin = QDoubleSpinBox()
            self.peak_time_spin.setRange(0, self.time_data[-1])
            self.peak_time_spin.setDecimals(4)
            self.peak_time_spin.setValue(self.time_peak)
            self.peak_time_spin.setSingleStep(0.01)
            self.peak_time_spin.setReadOnly(False)
            layout.addWidget(self.peak_time_spin)

            # === DIRECTION ===
            direction_label = QLabel("Signal Direction:")
            layout.addWidget(direction_label)
            self.direction_combo = QComboBox()
            self.direction_combo.addItems(["Upward", "Downward"])
            # ✅ Imposta il valore corrente
            self.direction_combo.setCurrentText("Upward" if self.direction == "upward" else "Downward")
            self.direction_combo.setEnabled(True)  # ✅ Editabile
            layout.addWidget(self.direction_combo)
            
            layout.addStretch(1)

            # ✅ BLOCCA ANCHE la ComboBox del signal type
            if hasattr(self, 'signalTypeComboBox'):
                self.signalTypeComboBox.blockSignals(True)
                self.signalTypeComboBox.setCurrentText("General Variables")
                self.signalTypeComboBox.blockSignals(False)
            
            # ✅ CONNETTI i callbacks DOPO aver creato tutti i widget
            self._setup_general_variables_callbacks()
            
            print("✅ General Variables UI created (editable)")
                        
        finally:
            # ✅ Rilascia flag
            self._updating_ui = False


    ############# ESPONENZIALE #############

    def setup_exponential_return_ui(self):
        from PySide6.QtWidgets import QVBoxLayout, QLabel, QDoubleSpinBox

        # ✅ IMPOSTA flag PRIMA di modificare la UI
        self._updating_ui = True

        # Attiva il pulsante di save e autoFit
        self.autoFitButton.setEnabled(True)
        self.saveButton.setEnabled(True)

        try:
            layout = self.resultsWidget.layout()
            if layout is None:
                layout = QVBoxLayout(self.resultsWidget)
                self.resultsWidget.setLayout(layout)

            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.setParent(None)

            layout.addStretch()
            layout.setSpacing(5)
            layout.setContentsMargins(5, 5, 5, 5)

            # --- Parameters ---
            params_label = QLabel("<b>Parameters:</b>")
            layout.addWidget(params_label)

            #--- Amplitude ---
            A_label = QLabel("Amplitude:")
            layout.addWidget(A_label)
            self.A_spin = QDoubleSpinBox()
            self.A_spin.setDecimals(4)
            self.A_spin.setRange(-3, 3)
            self.A_spin.setSingleStep(0.01)
            self.A_spin.setValue(self.params_dict_exp['A'])
            layout.addWidget(self.A_spin)

            #--- Tau ---
            tau_label = QLabel("&tau; (time constant):")
            tau_label.setTextFormat(Qt.RichText)
            layout.addWidget(tau_label)
            self.tau_spin = QDoubleSpinBox()
            self.tau_spin.setDecimals(4)
            self.tau_spin.setRange(0.01, 10)
            self.tau_spin.setSingleStep(0.01)
            self.tau_spin.setValue(self.params_dict_exp['tau'])
            layout.addWidget(self.tau_spin)

            # --- Energy ---
            energy_label = QLabel("<b>Energy:</b>")
            layout.addWidget(energy_label)

            self.spike_energy_label = QLabel("Spike phase: ...")
            layout.addWidget(self.spike_energy_label)
            self.exp_energy_label = QLabel("Exponential phase: ...")
            layout.addWidget(self.exp_energy_label)

            layout.addStretch(1)
            
            # ✅ Riconnetti i callbacks DOPO aver creato tutti i widget
            self.setup_parameter_callbacks()
            
        finally:
            # ✅ RILASCIA flag SEMPRE, anche in caso di errore
            self._updating_ui = False
        
        print(f"Initial A for exponential: {self.params_dict_exp['A']:.4f}")




    ##### ACTION POTENTIAL ######
    def setup_action_potential_ui(self):
        from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, QPushButton

        # ✅ IMPOSTA flag PRIMA di modificare la UI
        self._updating_ui = True

        # Attiva il pulsante di save e autoFit
        self.autoFitButton.setEnabled(True)
        self.saveButton.setEnabled(True)

        try:
            layout = self.resultsWidget.layout()
            if layout is None:
                layout = QVBoxLayout(self.resultsWidget)
                self.resultsWidget.setLayout(layout)

            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.setParent(None)

            layout.addStretch()
            layout.setSpacing(5)
            layout.setContentsMargins(5, 5, 5, 5)

            # --- Parameters ---
            params_label_sin = QLabel("<b>Sin Parameters:</b>")
            layout.addWidget(params_label_sin)

            #sine part
            A_sin_label = QLabel("A<sub>sin</sub>: Sine amplitude")
            A_sin_label.setTextFormat(Qt.RichText)
            layout.addWidget(A_sin_label)
            self.A_sin_spin = QDoubleSpinBox()
            self.A_sin_spin.setDecimals(4)
            self.A_sin_spin.setRange(-1.65, 1.65)
            self.A_sin_spin.setSingleStep(0.01)
            self.A_sin_spin.setValue(self.params_dict_sin_action_potential['A_sin'])
            layout.addWidget(self.A_sin_spin)

            phi_label = QLabel("&phi;: Phase shift")
            phi_label.setTextFormat(Qt.RichText)
            layout.addWidget(phi_label)
            self.phi_spin = QDoubleSpinBox()
            self.phi_spin.setDecimals(4)
            self.phi_spin.setRange(-3.14, 3.14)
            self.phi_spin.setSingleStep(0.01)
            self.phi_spin.setValue(self.params_dict_sin_action_potential['phi'])
            layout.addWidget(self.phi_spin)

            f_label = QLabel("f: Oscillation frequency")
            layout.addWidget(f_label)
            self.f_spin = QDoubleSpinBox()
            self.f_spin.setDecimals(4)
            self.f_spin.setRange(0.01, 10)
            self.f_spin.setSingleStep(0.01)
            self.f_spin.setValue(self.params_dict_sin_action_potential['f'])
            layout.addWidget(self.f_spin)

            #frame
            frame = QFrame()
            frame.setFrameShape(QFrame.HLine)
            frame.setFrameShadow(QFrame.Sunken)
            layout.addWidget(frame)

            #exponential part
            params_label_exp = QLabel("<b>Exp Parameters:</b>")
            layout.addWidget(params_label_exp)

            A_exp_label = QLabel("A<sub>exp</sub>: Decay amplitude")
            A_exp_label.setTextFormat(Qt.RichText)
            layout.addWidget(A_exp_label)
            self.A_exp_spin = QDoubleSpinBox()
            self.A_exp_spin.setDecimals(4)
            self.A_exp_spin.setRange(-3, 3)
            self.A_exp_spin.setSingleStep(0.01)
            self.A_exp_spin.setValue(self.params_dict_exp_action_potential['A_exp'])
            layout.addWidget(self.A_exp_spin)

            tau_label = QLabel("&tau;: Time constant ")
            tau_label.setTextFormat(Qt.RichText)
            layout.addWidget(tau_label)
            self.tau_exp_spin = QDoubleSpinBox()
            self.tau_exp_spin.setDecimals(4)
            self.tau_exp_spin.setRange(0.01, 10)
            self.tau_exp_spin.setSingleStep(0.01)
            self.tau_exp_spin.setValue(self.params_dict_exp_action_potential['tau'])
            layout.addWidget(self.tau_exp_spin)

            # --- Energy ---
            energy_label = QLabel("<b>Energy:</b>")
            layout.addWidget(energy_label)

            self.sine_action_potential_energy_label = QLabel("Spike phase: ...")
            layout.addWidget(self.sine_action_potential_energy_label)
            self.exp_action_potential_energy_label = QLabel("Exponential phase: ...")
            layout.addWidget(self.exp_action_potential_energy_label)

            layout.addStretch(1)
            
            # ✅ Riconnetti i callbacks DOPO aver creato tutti i widget
            self.setup_parameter_callbacks()
            
        finally:
            # ✅ RILASCIA flag SEMPRE, anche in caso di errore
            self._updating_ui = False


    #UI per la formula con i relativi parametri


    

    #PULSANTI#
    def on_more_info_button_triggered(self):
        #per ora apri solo dialogo di info che dice che la documentazione sarà implementata a breve
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(
            self,
            "More Info",
            "Documentation for the fitting models will be implemented soon.\n"
            "Stay tuned!"
        )



    def on_export_csv_button_triggered(self):
        """
        Esporta i dati dell'analisi corrente (parametri e segnale) in un file CSV.
        """
        import csv
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        # 1. Chiedi all'utente dove salvare il file
        default_filename = f"analysis_{self.name_of_analysis.replace(' ', '_')}.csv" if self.name_of_analysis else "analysis_export.csv"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Analysis to CSV",
            default_filename,
            "CSV Files (*.csv)"
        )

        if not file_path:
            print("⚠️ CSV export cancelled.")
            return

        print(f"📦 Exporting analysis to: {file_path}")

        try:
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # --- SEZIONE 1: METADATI E PARAMETRI ---
                writer.writerow(['# METADATA & PARAMETERS'])
                
                analysis_data = self._gather_analysis_data()

                # Scrivi metadati
                writer.writerow(['Analysis Name', analysis_data['metadata']['name']])
                writer.writerow(['Saved At', analysis_data['metadata']['saved_at']])
                writer.writerow(['Notes', analysis_data['metadata']['notes']])
                writer.writerow(['Active Model', analysis_data['model']['active_model']])
                
                writer.writerow([]) # Riga vuota per separare
                writer.writerow(['# GENERAL PARAMETERS'])
                for key, value in analysis_data['parameters']['general'].items():
                    # ✅ FIX: Scrivi il valore come stringa, senza forzare la formattazione float
                    writer.writerow([key, str(value)])

                writer.writerow([]) # Riga vuota
                
                # Scrivi parametri specifici del modello attivo
                if self.is_exponential_return:
                    writer.writerow(['# EXPONENTIAL RETURN PARAMETERS'])
                    for key, value in analysis_data['parameters']['exponential_return'].items():
                        writer.writerow([key, str(value)])
                elif self.is_action_potential:
                    writer.writerow(['# ACTION POTENTIAL PARAMETERS'])
                    for key, value in analysis_data['parameters']['action_potential'].items():
                        writer.writerow([key, str(value)])

                writer.writerow([]) # Riga vuota
                writer.writerow(['# RESULTS'])
                for key, value in analysis_data['results'].items():
                    writer.writerow([key, str(value)])

                # --- SEZIONE 2: DATI DEL SEGNALE ---
                writer.writerow([]) # Riga vuota per separare
                writer.writerow(['# TIME SERIES DATA'])
                writer.writerow(['Time (s)', 'Original Signal (V)', 'Fitted Curve (V)'])

                # Assicurati che tutti gli array abbiano la stessa lunghezza
                if len(self.fit_time_data) != len(self.fit_signal_data) or len(self.fit_time_data) != len(self.fitted_curve):
                    self.show_error_dialog("Export Error", "Data arrays have mismatched lengths. Cannot export.")
                    return

                # Scrivi i dati riga per riga
                for i in range(len(self.fit_time_data)):
                    writer.writerow([
                        f"{self.fit_time_data[i]:.6f}",
                        f"{self.fit_signal_data[i]:.6f}",
                        f"{self.fitted_curve[i]:.6f}"
                    ])

            print("✅ CSV export completed successfully.")
            QMessageBox.information(self, "Export Successful", f"Analysis data successfully exported to:\n{file_path}")

        except Exception as e:
            import traceback
            print(f"❌ Error during CSV export: {e}")
            print(traceback.format_exc())
            self.show_error_dialog("Export Error", f"An error occurred while exporting to CSV:\n{e}")





    def on_save_button_triggered(self):
        """
        Salva l'analisi corrente nel file .pvolt/.paudio originale.
        Gestisce sia la creazione del footer che l'aggiornamento.
        """
        import json
        import struct


        #apri dialog per il nome dell'analisi
        from PySide6.QtWidgets import QInputDialog, QLineEdit, QMessageBox
        # Chiedi il nome dell'analisi
        text, ok = QInputDialog.getText(self, "Save Analysis", "Enter a name for this analysis:", QLineEdit.Normal, self.name_of_analysis if self.name_of_analysis is not None else "")
        if ok and text.strip():
            self.name_of_analysis = text.strip()
        else:
            print("⚠️ Save cancelled or invalid name.")
            return  # L'utente ha annullato o inserito un nome vuoto


        if not self.source_file_path:
            self.show_error_dialog("Save Error", "Source file path is not available. Cannot save analysis.")
            return

        print(f"💾 Saving analysis to file: {self.source_file_path}")
        analysis_data = self._gather_analysis_data()

        try:
            with open(self.source_file_path, "r+b") as f:
                # 1. Controlla se il file è abbastanza lungo per contenere un puntatore
                file_size = f.seek(0, 2)
                footer_offset = 0
                analyses_dict = {"file_format_version": "1.0", "analyses": {}}

                if file_size > 8:
                    # Leggi il puntatore esistente (ultimi 8 byte)
                    f.seek(-8, 2)
                    footer_offset_bytes = f.read(8)
                    footer_offset = struct.unpack('<Q', footer_offset_bytes)[0]

                # 2. Se esiste un footer valido, leggilo
                if footer_offset > 0 and footer_offset < file_size:
                    f.seek(footer_offset)
                    # Leggi fino alla fine del JSON (prima del puntatore)
                    json_data_len = file_size - footer_offset - 8
                    existing_json_str = f.read(json_data_len).decode('utf-8')
                    try:
                        analyses_dict = json.loads(existing_json_str)
                        print(f"📖 Read existing footer with {len(analyses_dict.get('analyses', {}))} analyses.")
                    except json.JSONDecodeError:
                        print("⚠️ Could not decode existing JSON footer. Overwriting.")
                    
                    # Troncai il file per rimuovere il vecchio footer e puntatore
                    f.seek(footer_offset)
                    f.truncate()
                    print(f"✂️ Old footer truncated. File size is now {f.tell()} bytes.")
                else:
                    print("📖 No valid footer found. Creating a new one.")
                    # Se non c'era un footer, il file è già alla posizione giusta (dopo i dati)

                # 3. Aggiungi o aggiorna la nuova analisi
                analysis_id = analysis_data["id"]
                analyses_dict["analyses"][analysis_id] = analysis_data
                print(f"➕ Added/Updated analysis with ID: {analysis_id}")

                # 4. Scrivi il nuovo footer JSON
                new_json_str = json.dumps(analyses_dict, indent=2)
                new_json_bytes = new_json_str.encode('utf-8')
                
                # La nuova posizione del footer è la posizione corrente
                new_footer_offset = f.tell()
                f.write(new_json_bytes)
                print(f"✍️ New JSON footer written at offset {new_footer_offset}.")

                # 5. Scrivi il nuovo puntatore di 8 byte alla fine
                f.write(struct.pack('<Q', new_footer_offset))
                print(f"✍️ New 8-byte offset pointer written at the end of the file.")
                
                # 6. Conferma il salvataggio e chiudi la finestra
                print(f"✅ Analysis '{analysis_data['metadata']['name']}' saved successfully.")
                self.current_analysis_id = analysis_id  # Aggiorna l'ID corrente
                QMessageBox.information(self, "Success", f"Analysis '{analysis_data['metadata']['name']}' saved successfully.")
                self.close()

        except Exception as e:
            import traceback
            print(f"❌ Error during analysis saving: {e}")
            print(traceback.format_exc())
            self.show_error_dialog("Save Error", f"An error occurred while saving the analysis:\n{e}")




    def _gather_analysis_data(self):
        """Raccoglie tutti i dati dell'analisi corrente in un dizionario."""
        import datetime
        import uuid

        if not hasattr(self, 'r_squared'):
            self.calculate_r_squared()
        
        # Se non abbiamo un ID, ne creiamo uno nuovo. Altrimenti usiamo quello esistente.
        if self.current_analysis_id is None:
            self.current_analysis_id = str(uuid.uuid4())

        analysis_data = {
            "id": self.current_analysis_id,
            "metadata": {
                "name": self.name_of_analysis if self.name_of_analysis else "Unnamed Analysis",
                "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "notes": self.notesLineEdit.text()
            },
            "model": {
                "active_model": self.signalTypeComboBox.currentText()
            },
            "parameters": {
                "general": {
                    "v_max": self.V_max,
                    "v_min": self.V_min,
                    "v_baseline": self.V_baseline,
                    "time_peak": self.time_peak,
                    "start_time": self.start_time,
                    "end_time": self.end_time,
                    "direction": self.direction
                },
                "exponential_return": {
                    "amplitude": self.amplitude_exponential,
                    "tau": self.tau_exponential,
                    "t0": self.t0_exponential
                },
                "action_potential": {
                    "a_sin": self.A_sin,
                    "f_osc": self.f_osc,
                    "phi": self.phi,
                    "a_exp": self.A_exp,
                    "tau": self.tau_action_potential,
                    "t_peak": self.t_peak_action_potential
                }
            },
            "results": {
                "r_squared": self.r_squared,
                "first_phase_energy": self.first_phase_energy,
                "second_phase_energy": self.second_phase_energy
            }
        }
        return analysis_data

    def on_cancel_button_triggered(self):
        self.close()

    def on_autofitbutton_triggered(self):
        """Esegue il curve_fit ottimizzato con i parametri correnti come iniziali"""
        print("🎯 Performing optimized curve fitting...")
        
        self._updating_ui = True
        
        try:
            signal_type = self.signalTypeComboBox.currentText()
            
            if signal_type == "Exponential Return":
                self.is_exponential_return = True
                self.is_action_potential = False
                
                success = self.fit_exponential_with_current_params()
                
                if success:
                    # ✅ Aggiorna parametri generali con i nuovi valori fittati
                    self.V_baseline = self.fitted_params[3]
                    self.t0_exponential = self.fitted_params[1]
                    
                    # ✅ Ricalcola diff_max e diff_min con la nuova baseline
                    self.diff_max = self.V_max - self.V_baseline
                    self.diff_min = self.V_baseline - self.V_min
                    
                    self.setup_exponential_return_ui()
                else:
                    print("⚠️ Fitting failed, keeping manual parameters")
                    return
                    
            elif signal_type == "Action Potential":
                self.is_action_potential = True
                self.is_exponential_return = False
                
                success = self.fit_action_potential_with_current_params()
                
                if success:
                    # ✅ Aggiorna parametri generali
                    self.V_baseline = self.fitted_params[6]
                    self.t_peak_action_potential = self.fitted_params[4]
                    
                    # ✅ Ricalcola diff_max e diff_min
                    self.diff_max = self.V_max - self.V_baseline
                    self.diff_min = self.V_baseline - self.V_min
                    
                    self.setup_action_potential_ui()
                else:
                    print("⚠️ Fitting failed, keeping manual parameters")
                    return
            
            else:
                print("⚠️ Please select a valid signal type (not General Variables)")
                self.show_warning_dialog(
                    "Invalid Selection",
                    "Please select either 'Exponential Return' or 'Action Potential' before using AutoFit."
                )
                return
            
            # ✅ Aggiorna le linee descrittive con i nuovi parametri
            self._update_reference_lines()
            self._update_fit_region_display()
            self._update_peak_line()
            
            print(f"✅ Optimized fit completed. R² = {self.r_squared:.4f}")
            
        except Exception as e:
            import traceback
            print(f"❌ Error during optimized fitting: {str(e)}")
            print(traceback.format_exc())
            self.show_warning_dialog("Fitting Error", f"Could not optimize fit: {str(e)}")
            
        finally:
            self._updating_ui = False
            self._update_curve_from_params()


    ###### funzioni per aggiornare parametri generali ######
    def _setup_general_variables_callbacks(self):
        """Configura i callback specifici per le variabili generali"""
        
        print("🔧 Setting up general variables callbacks...")

        # ✅ CONNETTI i callback per le variabili generali
        if hasattr(self, 'max_voltage_spin'):
            try:
                self.max_voltage_spin.valueChanged.disconnect()
            except:
                pass
            self.max_voltage_spin.valueChanged.connect(self._on_vmax_changed)
        
        if hasattr(self, 'min_voltage_spin'):
            try:
                self.min_voltage_spin.valueChanged.disconnect()
            except:
                pass
            self.min_voltage_spin.valueChanged.connect(self._on_vmin_changed)
        
        if hasattr(self, 'baseline_spin'):
            try:
                self.baseline_spin.valueChanged.disconnect()
            except:
                pass
            self.baseline_spin.valueChanged.connect(self._on_baseline_changed)
        
        if hasattr(self, 'start_time_spin'):
            try:
                self.start_time_spin.valueChanged.disconnect()
            except:
                pass
            self.start_time_spin.valueChanged.connect(self._on_start_time_changed)
        
        if hasattr(self, 'stop_time_spin'):
            try:
                self.stop_time_spin.valueChanged.disconnect()
            except:
                pass
            self.stop_time_spin.valueChanged.connect(self._on_end_time_changed)
        
        if hasattr(self, 'peak_time_spin'):
            try:
                self.peak_time_spin.valueChanged.disconnect()
            except:
                pass
            self.peak_time_spin.valueChanged.connect(self._on_peak_time_changed)
        
        if hasattr(self, 'direction_combo'):
            try:
                self.direction_combo.currentTextChanged.disconnect()
            except:
                pass
            self.direction_combo.currentTextChanged.connect(self._on_direction_changed_general)
        
        print("✅ General variables callbacks configured")


    def _on_vmax_changed(self, new_vmax):
        """Gestisce cambio di V_max con propagazione alle ampiezze"""
        if self._updating_ui:
            return
        
        print(f"🔄 V_max changed to {new_vmax:.4f}V")
        
        self.V_max = new_vmax
        self.diff_max = self.V_max - self.V_baseline
        
        # ✅ Aggiorna ampiezze SOLO se un modello è attivo
        if self.direction == "upward":
            if self.is_exponential_return:
                new_amplitude = self.V_max - self.V_baseline
                self.amplitude_exponential = new_amplitude
                
                if hasattr(self, 'A_spin'):
                    self._updating_ui = True
                    self.A_spin.setValue(new_amplitude)
                    self._updating_ui = False
                
                print(f"   📐 Updated A_exponential = {new_amplitude:.4f}V")
                
            elif self.is_action_potential:
                new_A_sin = (self.V_max - self.V_min) / 2
                self.A_sin = new_A_sin
                new_A_exp = self.V_min - self.V_baseline
                self.A_exp = new_A_exp
                
                if hasattr(self, 'A_sin_spin'):
                    self._updating_ui = True
                    self.A_sin_spin.setValue(new_A_sin)
                    self.A_exp_spin.setValue(new_A_exp)
                    self._updating_ui = False
                
                print(f"   📐 Updated A_sin = {new_A_sin:.4f}V, A_exp = {new_A_exp:.4f}V")
        
        # ✅ Aggiorna linee e ricalcola curva
        self._update_reference_lines()
        
        # ✅ Ricalcola la curva con i nuovi parametri
        if self.is_exponential_return or self.is_action_potential:
            self._update_curve_from_params()


    def _on_vmin_changed(self, new_vmin):
        """Gestisce cambio di V_min con propagazione alle ampiezze"""
        if self._updating_ui:
            return
        
        print(f"🔄 V_min changed to {new_vmin:.4f}V")
        
        self.V_min = new_vmin
        self.diff_min = self.V_baseline - self.V_min
        
        # ✅ Aggiorna ampiezze
        if self.direction == "downward":
            if self.is_exponential_return:
                new_amplitude = self.V_min - self.V_baseline
                self.amplitude_exponential = new_amplitude
                
                # ✅ Aggiorna spinbox
                if hasattr(self, 'A_spin'):
                    self._updating_ui = True
                    self.A_spin.setValue(new_amplitude)
                    self._updating_ui = False
                
                print(f"   📐 Updated A_exponential = {new_amplitude:.4f}V")
                
            elif self.is_action_potential:
                new_A_sin = (self.V_min - self.V_max) / 2
                self.A_sin = new_A_sin
                new_A_exp = self.V_max - self.V_baseline
                self.A_exp = new_A_exp
                
                # ✅ Aggiorna spinbox
                if hasattr(self, 'A_sin_spin'):
                    self._updating_ui = True
                    self.A_sin_spin.setValue(new_A_sin)
                    self.A_exp_spin.setValue(new_A_exp)
                    self._updating_ui = False
                
                print(f"   📐 Updated A_sin = {new_A_sin:.4f}V, A_exp = {new_A_exp:.4f}V")
        
        # ✅ Aggiorna linee e ricalcola curva
        self._update_reference_lines()
        
        # ✅ Ricalcola la curva con i nuovi parametri
        if self.is_exponential_return or self.is_action_potential:
            self._update_curve_from_params()


    def _on_baseline_changed(self, new_baseline):
        """Gestisce cambio di V_baseline con propagazione a tutti i parametri dipendenti"""
        if self._updating_ui:
            return
        
        print(f"🔄 V_baseline changed to {new_baseline:.4f}V")
        
        old_baseline = self.V_baseline
        self.V_baseline = new_baseline
        
        # ✅ Aggiorna diff_max e diff_min
        self.diff_max = self.V_max - self.V_baseline
        self.diff_min = self.V_baseline - self.V_min
        
        # ✅ Aggiorna tutte le ampiezze che dipendono dalla baseline
        if self.direction == "upward":
            if self.is_exponential_return:
                self.amplitude_exponential = self.V_max - self.V_baseline
                print(f"   📐 Updated A_exponential = {self.amplitude_exponential:.4f}V")
                
            elif self.is_action_potential:
                # A_exp per upward è negativo rispetto alla baseline
                self.A_exp = self.V_min - self.V_baseline
                print(f"   📐 Updated A_exp = {self.A_exp:.4f}V")
                
        elif self.direction == "downward":
            if self.is_exponential_return:
                self.amplitude_exponential = self.V_min - self.V_baseline
                print(f"   📐 Updated A_exponential = {self.amplitude_exponential:.4f}V")
                
            elif self.is_action_potential:
                # A_exp per downward è positivo rispetto alla baseline
                self.A_exp = self.V_max - self.V_baseline
                print(f"   📐 Updated A_exp = {self.A_exp:.4f}V")
        
        # ✅ Aggiorna il plot mostrando la linea della baseline
        self._update_reference_lines()

        # ✅ FIX: Ricalcola la curva anche quando cambia la baseline
        if self.is_exponential_return or self.is_action_potential:
            self._update_curve_from_params()

    def _on_start_time_changed(self, new_start_time):
        """Gestisce cambio di start_time con ricalcolo degli array"""
        if self._updating_ui:
            return
        
        print(f"🔄 Start time changed to {new_start_time:.4f}s")
        
        # ✅ Validazione: start_time deve essere < end_time
        if new_start_time >= self.end_time:
            print(f"⚠️ Invalid: start_time >= end_time, reverting")
            self.start_time_spin.blockSignals(True)
            self.start_time_spin.setValue(self.start_time)
            self.start_time_spin.blockSignals(False)
            return
        
        self.start_time = new_start_time
        
        # ✅ Trova il nuovo indice di start
        self.start_index = np.argmin(np.abs(self.time_data - new_start_time))
        
        # ✅ Ricalcola gli array per il fitting
        self.fit_time_data = self.time_data[self.start_index:self.end_index+1]
        self.fit_signal_data = self.signal_data[self.start_index:self.end_index+1]
        
        print(f"   📊 New fitting range: {len(self.fit_time_data)} points")
        
        # ✅ Aggiorna visualizzazione della regione
        self._update_fit_region_display()

        # ✅ CONTROLLO T_PEAK: Assicura che t_peak sia dentro il nuovo range
        if self.time_peak < self.start_time:
            print(f"   ⚠️ Peak time ({self.time_peak:.4f}s) is now outside the new start time. Adjusting...")
            # Sposta t_peak all'inizio del nuovo range e triggera il suo aggiornamento
            if hasattr(self, 'peak_time_spin'):
                # Modificare la spinbox triggererà automaticamente _on_peak_time_changed
                self.peak_time_spin.setValue(self.start_time)
            else:
                # Se la UI non è visibile, aggiorna manualmente
                self._on_peak_time_changed(self.start_time)


    def _on_end_time_changed(self, new_end_time):
        """Gestisce cambio di end_time con ricalcolo degli array"""
        if self._updating_ui:
            return
        
        print(f"🔄 End time changed to {new_end_time:.4f}s")
        
        # ✅ Validazione: end_time deve essere > start_time
        if new_end_time <= self.start_time:
            print(f"⚠️ Invalid: end_time <= start_time, reverting")
            self.stop_time_spin.blockSignals(True)
            self.stop_time_spin.setValue(self.end_time)
            self.stop_time_spin.blockSignals(False)
            return
        
        self.end_time = new_end_time
        
        # ✅ Trova il nuovo indice di end
        self.end_index = np.argmin(np.abs(self.time_data - new_end_time))
        
        # ✅ Ricalcola gli array per il fitting
        self.fit_time_data = self.time_data[self.start_index:self.end_index+1]
        self.fit_signal_data = self.signal_data[self.start_index:self.end_index+1]
        
        print(f"   📊 New fitting range: {len(self.fit_time_data)} points")
        
        # ✅ Aggiorna visualizzazione della regione
        self._update_fit_region_display()

            # ✅ CONTROLLO T_PEAK: Assicura che t_peak sia dentro il nuovo range
        if self.time_peak > self.end_time:
            print(f"   ⚠️ Peak time ({self.time_peak:.4f}s) is now outside the new end time. Adjusting...")
            # Sposta t_peak alla fine del nuovo range e triggera il suo aggiornamento
            if hasattr(self, 'peak_time_spin'):
                # Modificare la spinbox triggererà automaticamente _on_peak_time_changed
                self.peak_time_spin.setValue(self.end_time)
            else:
                # Se la UI non è visibile, aggiorna manualmente
                self._on_peak_time_changed(self.end_time)


    def _on_peak_time_changed(self, new_peak_time):
        """Gestisce cambio di peak_time (t0 per exponential, t_peak per action potential)"""
        if self._updating_ui:
            return
        
        print(f"🔄 Peak time changed to {new_peak_time:.4f}s")
        
        # ✅ Validazione: deve essere tra start e end
        if new_peak_time < self.start_time or new_peak_time > self.end_time:
            print(f"⚠️ Invalid: peak_time must be between start and end, reverting")
            self.peak_time_spin.blockSignals(True)
            self.peak_time_spin.setValue(self.time_peak)
            self.peak_time_spin.blockSignals(False)
            return
        
        old_peak = self.time_peak
        self.time_peak = new_peak_time
        
        # ✅ Aggiorna i parametri specifici del modello
        if self.is_exponential_return:
            self.t0_exponential = new_peak_time
            print(f"   📐 Updated t0_exponential = {new_peak_time:.4f}s")
            
        elif self.is_action_potential:
            self.t_peak_action_potential = new_peak_time
            print(f"   📐 Updated t_peak_action_potential = {new_peak_time:.4f}s")
        
        # ✅ Ricalcola energia (dipende dal peak time per dividere le fasi)
        self.compute_signal_energy(self.fit_signal_data, self.fit_time_data, self.V_baseline)
        
        # ✅ Aggiorna visualizzazione della linea verticale del picco
        self._update_peak_line()


    def _on_direction_changed_general(self, new_direction):
        """Gestisce cambio di direzione dalle General Variables (CRITICO)"""
        if self._updating_ui:
            return
        
        print(f"🔀 Direction changed to: {new_direction} (from General Variables)")
        
        old_direction = self.direction
        self.direction = new_direction.lower()
        
        # ✅ Se la direzione è cambiata, INVERTI TUTTI i segni delle ampiezze
        if old_direction != self.direction:
            print("   ⚠️ Direction reversal detected - inverting all amplitudes")
            
            # ✅ Scambia V_max e V_min
            temp_max = self.V_max
            self.V_max = self.V_min
            self.V_min = temp_max
            
            # ✅ Aggiorna anche gli indici
            temp_idx = self.idx_max
            self.idx_max = self.idx_min
            self.idx_min = temp_idx
            
            # ✅ Ricalcola diff_max e diff_min
            self.diff_max = self.V_max - self.V_baseline
            self.diff_min = self.V_baseline - self.V_min
            
            # ✅ Inverti le ampiezze del modello attivo
            if self.is_exponential_return:
                self.amplitude_exponential = -self.amplitude_exponential
                print(f"   📐 Inverted A_exponential = {self.amplitude_exponential:.4f}V")
                
            elif self.is_action_potential:
                self.A_sin = -self.A_sin
                self.A_exp = -self.A_exp
                print(f"   📐 Inverted A_sin = {self.A_sin:.4f}V")
                print(f"   📐 Inverted A_exp = {self.A_exp:.4f}V")
            
            # ✅ Aggiorna le spinbox delle General Variables con i nuovi valori
            self._updating_ui = True
            try:
                if hasattr(self, 'max_voltage_spin'):
                    self.max_voltage_spin.setValue(self.V_max)
                if hasattr(self, 'min_voltage_spin'):
                    self.min_voltage_spin.setValue(self.V_min)
            finally:
                self._updating_ui = False
            
            # ✅ Aggiorna il plot
            self._update_reference_lines()



    def _update_reference_lines(self):
        """Aggiorna le linee di riferimento sul plot (V_max, V_min, V_baseline)"""
        try:
            # ✅ Rimuovi vecchie linee se esistono
            if hasattr(self, '_vmax_line'):
                self.voltage_plot.plot_widget.removeItem(self._vmax_line)
            if hasattr(self, '_vmin_line'):
                self.voltage_plot.plot_widget.removeItem(self._vmin_line)
            if hasattr(self, '_vbaseline_line'):
                self.voltage_plot.plot_widget.removeItem(self._vbaseline_line)
            
            # ✅ Crea nuove linee
            # V_max (rosso)
            self._vmax_line = pg.InfiniteLine(
                pos=self.V_max,
                angle=0,
                pen=pg.mkPen(color='r', width=1, style=Qt.DashLine),
                label='V_max'
            )
            self.voltage_plot.plot_widget.addItem(self._vmax_line)
            
            # V_min (magenta)
            self._vmin_line = pg.InfiniteLine(
                pos=self.V_min,
                angle=0,
                pen=pg.mkPen(color='m', width=1, style=Qt.DashLine),
                label='V_min'
            )
            self.voltage_plot.plot_widget.addItem(self._vmin_line)

            # V_baseline (ciano)
            self._vbaseline_line = pg.InfiniteLine(
                pos=self.V_baseline,
                angle=0,
                pen=pg.mkPen(color='c', width=1, style=Qt.DashLine),
                label='V_baseline'
            )
            self.voltage_plot.plot_widget.addItem(self._vbaseline_line)
            
            print("✅ Reference lines updated")
            
        except Exception as e:
            print(f"⚠️ Could not update reference lines: {e}")


    def _update_fit_region_display(self):
        """Aggiorna la visualizzazione della regione di fitting sul plot"""
        try:
            # ✅ Rimuovi vecchia regione se esiste
            if hasattr(self, '_fit_region'):
                self.voltage_plot.plot_widget.removeItem(self._fit_region)
            
            # ✅ Crea nuova regione visuale ma non farla troppo scura
            self._fit_region = pg.LinearRegionItem(
                values=[self.start_time, self.end_time],
                brush=(100, 100, 150, 50),  # Colore con trasparenza
                movable=False
            )
            self.voltage_plot.plot_widget.addItem(self._fit_region)
            
            print(f"✅ Fit region updated: [{self.start_time:.3f}s, {self.end_time:.3f}s]")
            
        except Exception as e:
            print(f"⚠️ Could not update fit region: {e}")


    def _update_peak_line(self):
        """Aggiorna la linea verticale del picco"""
        try:
            # ✅ Rimuovi vecchia linea se esiste
            if hasattr(self, '_peak_line'):
                self.voltage_plot.plot_widget.removeItem(self._peak_line)

            # ✅ Crea nuova linea verticale al picco di colore verde
            self._peak_line = pg.InfiniteLine(
                pos=self.time_peak,
                angle=90,
                pen=pg.mkPen(color='g', width=1, style=Qt.DashLine),
                label=f't_peak={self.time_peak:.3f}s'
            )
            self.voltage_plot.plot_widget.addItem(self._peak_line)
            
            print(f"✅ Peak line updated at t={self.time_peak:.3f}s")
            
        except Exception as e:
            print(f"⚠️ Could not update peak line: {e}")




    # FUNZIONI PER IL PLOT
    def recreate_plot(self):
        from plotting.plot_manager import BasePlotWidget

        # ORA SOLO PER VOLTAGE

        # Rimuovi il plot esistente se presente
        start_sec = self.time_data[0]
        end_sec = self.time_data[-1]

        if self.amplified_data:
            y_max = 1.7
            y_min = -1.7
            unit_y = "V"
        else:
            y_max = 130
            y_min = 130
            unit_y = "mV"

        custom_plot = BasePlotWidget(
                    x_label="Time",
                    y_label="Voltage",
                    x_range=(start_sec, end_sec),
                    y_range=(y_min, y_max),
                    x_min=start_sec, x_max=end_sec, y_min=y_min, y_max=y_max,
                    unit_x="s", unit_y=unit_y,
                    parent=self
                )
        self._replace_plot_widget_in_math_dialog(custom_plot)
        self.voltage_plot = custom_plot

        # Crea la curva principale con la penna desiderata
        self.voltage_curve = self.voltage_plot.plot_widget.plot(name="Amplitude Data")

        # Imposta i dati
        self.voltage_curve.setData(self.time_data, self.signal_data)

        # Applica il tema
        self.apply_theme(self.voltage_plot, self.voltage_curve)

        # Configura il plot (etichette, griglia, ecc.)
        self.voltage_plot.plot_widget.showGrid(x=True, y=True)

    def _replace_plot_widget_in_math_dialog(self, custom_plot):
        """
        Sostituisce il widget 'plotWidget' con 'custom_plot' nel dialog MathOperations.
        """
        # Trova il vecchio widget
        old_widget = self.plotWidget  # attributo generato da Qt Designer
        if old_widget is None:
            print("❌ plotWidget non trovato!")
            return

        # Trova il layout che contiene plotWidget (plotSectionVLayout)
        parent_layout = self.plotSectionVLayout

        # Trova l'indice di plotWidget nel layout
        for i in range(parent_layout.count()):
            if parent_layout.itemAt(i).widget() == old_widget:
                parent_layout.removeWidget(old_widget)
                old_widget.setParent(None)
                parent_layout.insertWidget(i, custom_plot)
                self.plotWidget = custom_plot  # aggiorna l'attributo
                print("✅ plotWidget sostituito con custom_plot.")
                return

        print("❌ plotWidget non trovato nel plotZoneVLayout!")



    def type_of_function_changed(self):
        type_of_function = self.signalTypeComboBox.currentText()
        if type_of_function == "Exponential return":
            self.setup_exponential_return_ui()

        elif type_of_function == "Action Potential":
            self.setup_action_potential_ui()

        elif type_of_function == "General Variables":
            self.setup_general_variables_ui()



    #FUNZIONI GENERALI #
    def _setup_specific_ui(self):

        self.setWindowTitle("Mathematical Analysis")

        # Imposta icona della finestra
        self.setWindowIcon(QIcon(AppConfig.LOGO_DIR))

         # --- Menubar personalizzata ---
        menubar = QMenuBar(self)
        file_menu = QMenu("About", self)
        action_close = QAction("Close", self)
        action_close.triggered.connect(self.close)
        file_menu.addAction(action_close)
        menubar.addMenu(file_menu)

        # Puoi aggiungere altre azioni/menu qui

        # Inserisci la menubar nel layout principale
        layout = self.layout() or QVBoxLayout(self)
        layout.setMenuBar(menubar)
        self.setLayout(layout)

        self.apply_theme()

        #centra la finestra
        self.layout_manager.center_window_on_screen(self)

        #imposta placeholder per le notes
        self.notesLineEdit.setPlaceholderText("Enter your notes here...")

        #aggiungi i modelli alla combobox
        self.signalTypeComboBox.addItems(["General Variables", "Exponential Return", "Action Potential"])
        self.signalTypeComboBox.setFixedWidth(200)  # modifica la grandezza per farci stare tutto


    def closeEvent(self, event):
        """Gestisce l'evento di chiusura della finestra"""
        print("🔒 Chiusura della finestra di analisi matematica...")
        super().closeEvent(event)
        print("✅ Finestra di analisi matematica chiusa correttamente")

    def apply_theme(self, plot_widget_name = None, plot_instance = None):
        theme_colors = self.theme_manager.get_theme_colors()
        theme_fg = theme_colors['foreground']
        theme_bg = theme_colors['background']

        # Cambia sfondo
        if plot_widget_name is not None and plot_instance is not None:
            self.theme_manager.apply_theme_to_plot(
                plot_widget_name=plot_widget_name.plot_widget, 
                plot_instance=plot_instance
            )

        # Imposta lo sfondo del QDialog
        self.setStyleSheet(f"background-color: {theme_bg}; color: {theme_fg};")

        # ✅ USA LO STILE DINAMICO DAL THEME MANAGER
        # Applichiamo lo stile "attivo" a tutti i pulsanti standard.
        # Lo stato :disabled verrà gestito automaticamente.
        button_style = self.theme_manager.get_toggle_button_style(is_active=True)
        
        self.analysisLinesButton.setStyleSheet(button_style)
        self.autoFitButton.setStyleSheet(button_style)
        self.saveButton.setStyleSheet(button_style)
        self.cancelButton.setStyleSheet(button_style)
        self.exportButton.setStyleSheet(button_style)
        self.moreInfoButton.setStyleSheet(button_style)

        # Imposta lo stile delle QLineEdit
        lineedit_style = f"""
            QLineEdit {{
                background-color: {theme_bg};
                color: {theme_fg};
                border: 1px solid {theme_fg};
                padding: 3px;
            }}
        """
        self.notesLineEdit.setStyleSheet(lineedit_style)

        # Imposta lo stile delle QComboBox
        combobox_style = f"""
            QComboBox {{
                background-color: {theme_bg};
                color: {theme_fg};
                border: 1px solid {theme_fg};
                padding: 3px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {theme_bg};
                color: {theme_fg};
                selection-background-color: {theme_fg};
                selection-color: {theme_bg};
            }}
        """
        self.signalTypeComboBox.setStyleSheet(combobox_style)

        print("🎨 Theme applied to ReplayBaseWindow and its components")


    #funzione per mostrare un messaggio di avvertenza dato un messaggio e un titolo chiedendo se continuare o no
    def show_warning_dialog(self, title, message):
        from PySide6.QtWidgets import QMessageBox
        """Show a warning dialog to the user and return True if they choose to continue"""
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setIcon(QMessageBox.Warning)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def show_error_dialog(self, title, message):
        from PySide6.QtWidgets import QErrorMessage
        error_dialog = QErrorMessage(self)
        error_dialog.setWindowTitle(title)
        error_dialog.showMessage(message)
        error_dialog.exec_()

    def toggle_analysis_items_visibility(self):
        """Mostra o nasconde tutte le linee di analisi e la regione di fitting."""
        self.analysis_items_visible = not self.analysis_items_visible
        
        state_str = "Visible" if self.analysis_items_visible else "Hidden"
        print(f"📊 Toggling analysis items to: {state_str}")

        # Imposta la visibilità per ogni elemento
        if hasattr(self, '_vmax_line'):
            self._vmax_line.setVisible(self.analysis_items_visible)
        if hasattr(self, '_vmin_line'):
            self._vmin_line.setVisible(self.analysis_items_visible)
        if hasattr(self, '_vbaseline_line'):
            self._vbaseline_line.setVisible(self.analysis_items_visible)
        if hasattr(self, '_peak_line'):
            self._peak_line.setVisible(self.analysis_items_visible)
        if hasattr(self, '_fit_region'):
            self._fit_region.setVisible(self.analysis_items_visible)
        
        # Aggiorna il testo del pulsante
        self.analysisLinesButton.setText("Hide Lines" if self.analysis_items_visible else "Show Lines")



    def _load_from_analysis_data(self, data):
        """Popola tutte le variabili e la UI dai dati di un'analisi salvata."""
        self._updating_ui = True
        try:
            # Metadata
            self.current_analysis_id = data.get('id')
            self.name_of_analysis = data.get('metadata', {}).get('name')
            self.notesLineEdit.setText(data.get('metadata', {}).get('notes', ''))
            print(f"   Loading analysis: '{self.name_of_analysis}' (ID: {self.current_analysis_id})")

            # Modello
            active_model = data.get('model', {}).get('active_model', 'General Variables')

            # Parametri Generali
            p_gen = data.get('parameters', {}).get('general', {})
            self.V_max = p_gen.get('v_max', 0)
            self.V_min = p_gen.get('v_min', 0)
            self.V_baseline = p_gen.get('v_baseline', 0)
            self.time_peak = p_gen.get('time_peak', 0)
            self.start_time = p_gen.get('start_time', self.total_time_data[0])
            self.end_time = p_gen.get('end_time', self.total_time_data[-1])
            self.direction = p_gen.get('direction', 'upward')

            # Parametri Esponenziali
            p_exp = data.get('parameters', {}).get('exponential_return', {})
            self.amplitude_exponential = p_exp.get('amplitude', 0)
            self.tau_exponential = p_exp.get('tau', 0.1)
            self.t0_exponential = p_exp.get('t0', self.time_peak)

            # Parametri Potenziale d'Azione
            p_ap = data.get('parameters', {}).get('action_potential', {})
            self.A_sin = p_ap.get('a_sin', 0)
            self.f_osc = p_ap.get('f_osc', 1)
            self.phi = p_ap.get('phi', 0)
            self.A_exp = p_ap.get('a_exp', 0)
            self.tau_action_potential = p_ap.get('tau', 0.1)
            self.t_peak_action_potential = p_ap.get('t_peak', self.time_peak)
            
            # Risultati
            results = data.get('results', {})
            self.r_squared = results.get('r_squared', 0)
            self.first_phase_energy = results.get('first_phase_energy', 0)
            self.second_phase_energy = results.get('second_phase_energy', 0)



            # Imposta gli array di fit basati su start/end time caricati
            self.start_index = np.argmin(np.abs(self.time_data - self.start_time))
            self.end_index = np.argmin(np.abs(self.time_data - self.end_time))
            self.fit_time_data = self.time_data[self.start_index:self.end_index+1]
            self.fit_signal_data = self.signal_data[self.start_index:self.end_index+1]
            
            # Imposta anche gli array principali usati da recreate_plot
            #self.time_data = self.time_data
            #self.signal_data = self.signal_data

            # Imposta la UI
            self.signalTypeComboBox.setCurrentText(active_model)
            if active_model == "Exponential Return":
                self.is_exponential_return = True
                self.is_action_potential = False
                self.params_dict_exp = {'A': self.amplitude_exponential, 'tau': self.tau_exponential}
                self.setup_exponential_return_ui()
            elif active_model == "Action Potential":
                self.is_action_potential = True
                self.is_exponential_return = False
                self.params_dict_sin_action_potential = {'A_sin': self.A_sin, 'f': self.f_osc, 'phi': self.phi}
                self.params_dict_exp_action_potential = {'A_exp': self.A_exp, 'tau': self.tau_action_potential}
                self.setup_action_potential_ui()
            else:
                self.setup_general_variables_ui()

            print("✅ Dialog populated with saved data.")

        finally:
            self._updating_ui = False


# ============================================================================
# SPECTRAL ENERGY ANALYSIS FUNCTIONS
# ============================================================================

def get_bin_indices(f_start_khz, f_end_khz, fs=200000, n_fft=512, array_offset_khz=20):
    """
    Convert frequency range (in kHz) to bin indices in the 154-bin array.
    
    Parameters:
    -----------
    f_start_khz : float
        Start frequency in kHz (e.g., 20 for 20 kHz)
    f_end_khz : float
        End frequency in kHz (e.g., 40 for 40 kHz)
    fs : int
        Sampling rate in Hz (default: 200000)
    n_fft : int
        FFT size (default: 512)
    array_offset_khz : float
        Frequency of bin 0 in the array (default: 20 kHz)
    
    Returns:
    --------
    tuple : (start_idx, end_idx)
        Indices for slicing the 154-bin array (inclusive range)
    """
    # Bin spacing in Hz
    bin_spacing_hz = fs / n_fft  # 390.625 Hz
    
    # Convert kHz to Hz
    f_start_hz = f_start_khz * 1000
    f_end_hz = f_end_khz * 1000
    array_offset_hz = array_offset_khz * 1000
    
    # Calculate indices relative to array start (20 kHz)
    start_idx = int(np.round((f_start_hz - array_offset_hz) / bin_spacing_hz))
    end_idx = int(np.round((f_end_hz - array_offset_hz) / bin_spacing_hz))
    
    # Clamp to valid range [0, 153]
    start_idx = max(0, start_idx)
    end_idx = min(153, end_idx)
    
    return start_idx, end_idx


def compute_fft_energy(bins):
    """
    Compute spectral energy over multiple frequency sub-bands.
    
    Energy is calculated as E = mean(magnitude²) for each band,
    representing average power in that frequency range.
    
    Parameters:
    -----------
    bins : np.ndarray
        Array of 154 FFT magnitude values covering 20-80 kHz
    
    Returns:
    --------
    dict : Energy values for each sub-band with keys:
        - 'total': 20-80 kHz (all 154 bins)
        - 'low': 20-40 kHz
        - 'high': 40-80 kHz
        - 'b1': 20-30 kHz
        - 'b2': 30-40 kHz
        - 'b3': 40-60 kHz
        - 'b4': 60-80 kHz
    """
    if len(bins) != 154:
        raise ValueError(f"Expected 154 bins, got {len(bins)}")
    
    # Define sub-bands (in kHz)
    bands = {
        'total': (20, 80),
        'low': (20, 40),
        'high': (40, 80),
        'b1': (20, 30),
        'b2': (30, 40),
        'b3': (40, 60),
        'b4': (60, 80),
    }
    
    # Compute energy for each band
    energies = {}
    bins_squared = bins ** 2
    
    for band_name, (f_start, f_end) in bands.items():
        start_idx, end_idx = get_bin_indices(f_start, f_end)
        # Include end_idx (Python slicing is exclusive at end)
        band_energy = np.mean(bins_squared[start_idx:end_idx + 1])
        energies[band_name] = band_energy
    
    return energies


class SpectralEnergyDialog(QDialog):
    """Finestra per mostrare l'analisi dell'energia spettrale di un frame FFT"""
    def __init__(self, energies, frame_index=None, parent=None):
        super().__init__(parent)
        
        # Salva riferimenti per normalizzazione
        self.parent = parent
        self.frame_index = frame_index
        self.energies_raw = energies
        self.energies_normalized = None
        self.is_normalized = False
        
        title = "Spectral Energy Analysis"
        if frame_index is not None:
            title += f" - Frame {frame_index}"
        self.setWindowTitle(title)
        self.setMinimumSize(500, 400)
        
        layout = QVBoxLayout(self)
        
        # Title label
        self.title_label = QLabel("<h2>FFT Spectral Energy Analysis</h2>")
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)
        
        # Info label
        self.info_label = QLabel(
            "Energy calculated as <b>E = mean(magnitude²)</b> over each frequency band.<br>"
            "Represents average power in that frequency range."
        )
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("color: #888; font-size: 10pt; margin: 10px;")
        layout.addWidget(self.info_label)
        
        # ✅ CALCOLA RATIO R (broadband click detection)
        self.ratio_raw = self._compute_energy_ratio(energies)
        self.ratio_normalized = None  # Verrà calcolato on-demand
        
        # Ratio info label
        ratio_color = self._get_ratio_color(self.ratio_raw)
        ratio_status = self._get_ratio_status(self.ratio_raw)
        self.ratio_label = QLabel(
            f"<b>Energy Ratio R (E_low/E_high):</b> <span style='color:{ratio_color}; font-size:14pt;'>{self.ratio_raw:.3f}</span> — {ratio_status}"
        )
        self.ratio_label.setAlignment(Qt.AlignCenter)
        self.ratio_label.setStyleSheet("margin: 5px;")
        layout.addWidget(self.ratio_label)
        
        # Create table for results
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Band", "Frequency Range", "Energy (V²)", "Energy (mV²)"])

        # Band definitions with pretty names
        self.band_info = [
            ('total', 'Total', '20-80 kHz'),
            ('low', 'Low', '20-40 kHz'),
            ('high', 'High', '40-80 kHz'),
            ('b1', 'Band 1', '20-30 kHz'),
            ('b2', 'Band 2', '30-40 kHz'),
            ('b3', 'Band 3', '40-60 kHz'),
            ('b4', 'Band 4', '60-80 kHz'),
        ]
        
        self._populate_table(energies)
        
        # Style table
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        
        # ✅ RIMUOVE I NUMERI DI RIGA
        self.table.verticalHeader().setVisible(False)
        
        layout.addWidget(self.table)
        
        # Buttons layout
        button_layout = QHBoxLayout()
        
        # Normalize button
        self.normalize_button = QPushButton("Apply 50% Normalization")
        self.normalize_button.clicked.connect(self._toggle_normalization)
        button_layout.addWidget(self.normalize_button)
        
        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        
        # Apply theme if available
        if parent and hasattr(parent, 'theme_manager'):
            saved_theme = parent.theme_manager.load_saved_theme()
            parent.theme_manager.apply_theme(self, saved_theme)
            
            # Apply font manager fonts
            if hasattr(parent, 'font_manager'):
                parent.font_manager.apply_fonts_to_widgets(self)

            # Se tema contiene light, imposta sfondo bianco con setStyleSheet
            current_theme = getattr(parent.theme_manager, 'current_theme', 'dark.css')
            is_light_theme = 'light' in current_theme.lower()
            if is_light_theme:
                self.setStyleSheet("background-color: white;")

        # Center on parent
        if parent:
            parent_rect = parent.geometry()
            self.resize(parent_rect.width() * 0.5, parent_rect.height() * 0.6)
            self.move(
                parent_rect.x() + (parent_rect.width() - self.width()) // 2,
                parent_rect.y() + (parent_rect.height() - self.height()) // 2
            )
    
    def _compute_energy_ratio(self, energies):
        """
        Calcola il ratio R = E_low / E_high per rilevare click broadband.
        
        Per un click acustico vero (broadband):
        - R ≈ 0.5–3.0 (energia distribuita su tutto lo spettro)
        
        Per artefatti narrowband (es. ronzio 40 kHz):
        - R >> 1 (energia concentrata in low) o R << 1 (energia concentrata in high)
        
        Returns:
            float: Ratio R, o 0.0 se E_high è zero
        """
        e_low = energies.get('low', 0.0)
        e_high = energies.get('high', 0.0)
        
        if e_high == 0.0:
            return 0.0  # Evita divisione per zero
        
        ratio = e_low / e_high
        return ratio
    
    def _get_ratio_color(self, ratio, is_normalized=False):
        """
        Ritorna il colore per visualizzare R in base alla validità.
        
        Soglie diverse per raw vs normalized:
        - RAW: [0.5, 3.0] (microfono bias verso low freq)
        - NORMALIZED: [0.3, 2.0] (correzione applicata, più accurato)
        """
        if is_normalized:
            # Soglie per R_normalized (più strict in alto, più permissive in basso)
            if 0.3 <= ratio <= 2.0:
                return "green"  # Click broadband VALIDO
            elif 0.2 <= ratio < 0.3 or 2.0 < ratio <= 2.5:
                return "orange"  # Borderline
            else:
                return "red"  # Artifact
        else:
            # Soglie per R_raw (originali, più permissive)
            if 0.5 <= ratio <= 3.0:
                return "green"  # Click broadband VALIDO
            elif 0.3 <= ratio < 0.5 or 3.0 < ratio <= 5.0:
                return "orange"  # Borderline
            else:
                return "red"  # Artifact
    
    def _get_ratio_status(self, ratio, is_normalized=False):
        """
        Ritorna lo stato testuale per R.
        
        Soglie adattive basate su normalizzazione applicata.
        """
        if is_normalized:
            # Soglie per R_normalized
            if 0.3 <= ratio <= 2.0:
                return "✅ Broadband Click (Valid)"
            elif 0.2 <= ratio < 0.3 or 2.0 < ratio <= 2.5:
                return "⚠️ Borderline (Check manually)"
            elif ratio > 2.5:
                return "❌ Low-frequency artifact"
            elif ratio < 0.2:
                return "❌ High-frequency artifact"
            else:
                return "⚠️ Unknown (E_high = 0?)"
        else:
            # Soglie per R_raw
            if 0.5 <= ratio <= 3.0:
                return "✅ Broadband Click (Valid)"
            elif 0.3 <= ratio < 0.5 or 3.0 < ratio <= 5.0:
                return "⚠️ Borderline (Check manually)"
            elif ratio > 5.0:
                return "❌ Low-frequency artifact"
            elif ratio < 0.3:
                return "❌ High-frequency artifact"
            else:
                return "⚠️ Unknown (E_high = 0?)"
    
    def _populate_table(self, energies):
        """Popola la tabella con i valori di energia"""
        self.table.setRowCount(len(self.band_info))
        
        for row, (key, name, freq_range) in enumerate(self.band_info):
            energy_v2 = energies[key]
            energy_mv2 = energy_v2 * 1e6  # Convert V² to mV²
            
            # Band name
            name_item = QTableWidgetItem(name)
            name_item.setFont(QFont("Arial", 11, QFont.Bold))
            self.table.setItem(row, 0, name_item)
            
            # Frequency range
            self.table.setItem(row, 1, QTableWidgetItem(freq_range))
            
            # Energy in V²
            self.table.setItem(row, 2, QTableWidgetItem(f"{energy_v2:.6f}"))
            
            # Energy in mV²
            self.table.setItem(row, 3, QTableWidgetItem(f"{energy_mv2:.3f}"))
    
    def _toggle_normalization(self):
        """Toggle tra energia raw e normalizzata (50%)"""
        if not self.is_normalized:
            # APPLICA NORMALIZZAZIONE
            self._compute_normalized_energy()
            if self.energies_normalized is not None:
                self.is_normalized = True
                self._update_display()
        else:
            # TORNA A RAW
            self.is_normalized = False
            self._update_display()
    
    def _compute_normalized_energy(self):
        """Calcola energia con correzione 50% dalla FFT normalizzata"""
        if not self.parent or not hasattr(self.parent, 'data_manager'):
            QMessageBox.warning(self, "Error", "Cannot access parent data manager.")
            return
        
        if self.frame_index is None or self.frame_index >= len(self.parent.data_manager.fft_data):
            QMessageBox.warning(self, "Error", "Invalid frame index.")
            return
        
        print("🔧 Computing normalized spectral energy (50% correction)...")
        
        # === 1. DATI DAL DATASHEET (SPU0410LR5H-QB) ===
        datasheet_freq_khz = np.array([20, 25, 30, 40, 50, 60, 70, 80])
        datasheet_response_db = np.array([8.0, 10.5, 6.0, -2.0, -6.0, -7.0, -6.0, -4.0])
        datasheet_freq_hz = datasheet_freq_khz * 1000
        
        # === 2. RECUPERA FFT ORIGINALE DEL FRAME ===
        fft_magnitudes = self.parent.data_manager.fft_data[self.frame_index]
        freq_axis = self.parent.data_manager.frequency_axis
        
        # === 3. CALCOLA CORREZIONE 50% ===
        valid_mask = (freq_axis >= 20000) & (freq_axis <= 80000)
        freq_range = freq_axis[valid_mask]
        
        mic_response_db = np.interp(freq_range, datasheet_freq_hz, datasheet_response_db)
        correction_gain_50 = 10 ** (-mic_response_db * 0.5 / 20.0)
        
        # === 4. APPLICA CORREZIONE ===
        normalized_fft = fft_magnitudes.copy()
        normalized_fft[valid_mask] = fft_magnitudes[valid_mask] * correction_gain_50
        
        # === 5. RICALCOLA ENERGIE E RATIO ===
        try:
            self.energies_normalized = compute_fft_energy(normalized_fft)
            self.ratio_normalized = self._compute_energy_ratio(self.energies_normalized)
            
            # Statistiche
            gain_stats = {
                'max_gain_db': 20 * np.log10(np.max(correction_gain_50)),
                'min_gain_db': 20 * np.log10(np.min(correction_gain_50)),
            }
            
            print(f"✅ Normalized energies computed:")
            print(f"   Gain range: {gain_stats['min_gain_db']:.2f} to {gain_stats['max_gain_db']:.2f} dB")
            print(f"   Total energy: {self.energies_normalized['total']:.6f} V² (raw: {self.energies_raw['total']:.6f})")
            print(f"   Ratio R: {self.ratio_normalized:.3f} (raw: {self.ratio_raw:.3f})")
            
        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", 
                               f"Failed to compute normalized energies:\n{str(e)}")
            print(f"❌ Error: {e}")
    
    def _update_display(self):
        """Aggiorna display con energia corretta"""
        if self.is_normalized and self.energies_normalized is not None:
            # Mostra energia normalizzata
            self._populate_table(self.energies_normalized)
            
            # Aggiorna ratio label con valore normalizzato (SOGLIE ADAPTIVE)
            ratio_color = self._get_ratio_color(self.ratio_normalized, is_normalized=True)
            ratio_status = self._get_ratio_status(self.ratio_normalized, is_normalized=True)
            self.ratio_label.setText(
                f"<b>Energy Ratio R (E_low/E_high) - NORMALIZED:</b> <span style='color:{ratio_color}; font-size:14pt;'>{self.ratio_normalized:.3f}</span> — {ratio_status}<br>"
                f"<span style='color:#888; font-size:9pt;'>(Raw R = {self.ratio_raw:.3f} | Valid range: 0.3-2.0 for normalized)</span>"
            )
            
            # Aggiorna titolo e info
            self.info_label.setText(
                "Energy calculated with <b>50% frequency response correction</b>.<br>"
                "Estimated error: <b>±2.9 dB</b> (95% confidence)."
            )
            self.info_label.setStyleSheet("color: red; font-weight: bold; font-size: 10pt; margin: 10px;")
            
            self.normalize_button.setText("Show Raw Energy")
            
        else:
            # Mostra energia raw
            self._populate_table(self.energies_raw)
            
            # Aggiorna ratio label con valore raw (SOGLIE ORIGINALI)
            ratio_color = self._get_ratio_color(self.ratio_raw, is_normalized=False)
            ratio_status = self._get_ratio_status(self.ratio_raw, is_normalized=False)
            self.ratio_label.setText(
                f"<b>Energy Ratio R (E_low/E_high):</b> <span style='color:{ratio_color}; font-size:14pt;'>{self.ratio_raw:.3f}</span> — {ratio_status}<br>"
                f"<span style='color:#888; font-size:9pt;'>(Valid range: 0.5-3.0 for raw)</span>"
            )
            
            # Ripristina titolo e info
            self.title_label.setText("<h2>FFT Spectral Energy Analysis</h2>")
            self.info_label.setText(
                "Energy calculated as <b>E = mean(magnitude²)</b> over each frequency band.<br>"
                "Represents average power in that frequency range."
            )
            self.info_label.setStyleSheet("color: #888; font-size: 10pt; margin: 10px;")
            
            self.normalize_button.setText("Apply 50% Normalization")
