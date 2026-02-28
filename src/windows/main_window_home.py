from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QSpacerItem, QSizePolicy, QFrame, QMenuBar, QMenu, QMessageBox
)
from PySide6.QtGui import QPixmap, QFont, QIcon, QAction, QActionGroup
from PySide6.QtCore import Qt, QSettings
import sys
import os
import platform
import json

from config.app_config import AppConfig
from core.font_manager import FontManager
from core.theme_manager import ThemeManager
from core.base_window import BaseWindow


class MainWindowHome(BaseWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Home")
        self.setWindowIcon(QIcon(AppConfig.LOGO_DIR))
        self.setFixedSize(500, 650)

        # CREA QUI TUTTE LE AZIONI DEL MENU
        self.actionQuit = QAction("Quit", self)
        self.actionInfo = QAction("Info", self)
        self.actionVersion = QAction("Version", self)
        self.actionWebSite = QAction("Website", self)
        self.actionLicense = QAction("License", self)
        self.actionOpenFile = QAction("Open File", self)
        self.actionDark = QAction("Dark", self)
        self.actionDark_Green = QAction("Dark Green", self)
        self.actionDark_Blue = QAction("Dark Blue", self)
        self.actionDark_Amber = QAction("Dark Amber", self)
        self.actionLight = QAction("Light", self)
        self.actionLight_Green = QAction("Light Green", self)
        self.actionLight_Blue = QAction("Light Blue", self)
        self.actionLight_Amber = QAction("Light Amber", self)
        self.actionVery_Small = QAction("Very Small", self)
        self.actionSmall = QAction("Small", self)
        self.actionMedium = QAction("Medium", self)
        self.actionLarge = QAction("Large", self)
        self.actionAbout = QAction("About", self)
        self.actionDocumentation = QAction("Documentation", self)

        self.voltage_window = None
        self.audio_window = None

        # 2. Setup UI
        self.setup_ui()
 
        # 3. Togli menuBar e statusBar
        self._setup_home_menu_bar()
        self.setStatusBar(None)

        self._is_closing = False



    def _setup_home_menu_bar(self):
        """Setup a minimal menu bar for the home window con le azioni principali"""
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)

        # Impostazioni grafiche
        font = QFont()
        font.setPointSize(12)
        font.setBold(False)
        menu_bar.setFont(font)

        # Menu File
        menu_file = QMenu("File", self)
        menu_bar.addMenu(menu_file)
        menu_file.addAction(self.actionOpenFile)
        self.actionOpenFile.setShortcut("Ctrl+O")

        # Menu Settings (solo temi e font)
        menu_settings = QMenu("Settings", self)
        menu_bar.addMenu(menu_settings)
        menu_theme = QMenu("Choose Theme", self)
        menu_settings.addMenu(menu_theme)
        menu_theme.addAction(self.actionDark)
        menu_theme.addAction(self.actionDark_Green)
        menu_theme.addAction(self.actionDark_Blue)
        menu_theme.addAction(self.actionDark_Amber)
        menu_theme.addSeparator()
        menu_theme.addAction(self.actionLight)
        menu_theme.addAction(self.actionLight_Green)
        menu_theme.addAction(self.actionLight_Blue)
        menu_theme.addAction(self.actionLight_Amber)

        menu_font = QMenu("Choose Font Scale", self)
        menu_settings.addMenu(menu_font)
        menu_font.addAction(self.actionVery_Small)
        menu_font.addAction(self.actionSmall)
        menu_font.addAction(self.actionMedium)
        menu_font.addAction(self.actionLarge)

        # Menu Help
        menu_help = QMenu("Help", self)
        menu_bar.addMenu(menu_help)
        menu_help.addAction(self.actionAbout)
        menu_help.addAction(self.actionDocumentation)

        # Menu PlantLeaf (cambiato in about)
        menu_plantleaf = QMenu("About", self)
        menu_bar.addMenu(menu_plantleaf)
        menu_plantleaf.addAction(self.actionQuit)
        menu_plantleaf.addSeparator()
        menu_plantleaf.addAction(self.actionInfo)
        menu_plantleaf.addAction(self.actionVersion)
        menu_plantleaf.addAction(self.actionWebSite)
        menu_plantleaf.addAction(self.actionLicense)

        self._setup_menubar_actions()


    def _setup_menubar_actions(self):
        """Collega le azioni della menuBar ai rispettivi metodi"""
        actions = [
            (self.actionQuit, self.close),
            (self.actionInfo, self.info_action),
            (self.actionVersion, self.version_action),
            (self.actionWebSite, self.website_action),
            (self.actionLicense, self.license_action),
            (self.actionOpenFile, self.open_file_action),
            (self.actionDark, lambda: self.update_style(theme_name="dark.css")),
            (self.actionDark_Green, lambda: self.update_style(theme_name="dark_green.css")),
            (self.actionDark_Blue, lambda: self.update_style(theme_name="dark_blue.css")),
            (self.actionDark_Amber, lambda: self.update_style(theme_name="dark_amber.css")),
            (self.actionLight, lambda: self.update_style(theme_name="light.css")),
            (self.actionLight_Green, lambda: self.update_style(theme_name="light_green.css")),
            (self.actionLight_Blue, lambda: self.update_style(theme_name="light_blue.css")),
            (self.actionLight_Amber, lambda: self.update_style(theme_name="light_amber.css")),
            (self.actionVery_Small, lambda: self.update_style(font_scale=1.15)),
            (self.actionSmall, lambda: self.update_style(font_scale=1.25)),
            (self.actionMedium, lambda: self.update_style(font_scale=1.35)),
            (self.actionLarge, lambda: self.update_style(font_scale=1.4)),
            (self.actionAbout, self.about_action),
            (self.actionDocumentation, self.documentation_action),
        ]
        for action, slot in actions:
            try:
                action.triggered.disconnect()
            except Exception:
                print(f"azione {action} non trovata")
                pass
            action.triggered.connect(slot)



    def setup_ui(self):
        """Setup the main user interface"""
        # Central widget setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout with adjusted margins
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(40, 15, 40, 15)
        main_layout.setSpacing(12)
        central_widget.setLayout(main_layout)

        # Header section with logo and title
        header_layout = QVBoxLayout()
        header_layout.setSpacing(8)
        
        # Logo - LOGO PIÙ GRANDE
        logo_label = QLabel()
        AppConfig.load_application_logo(logo_label, target_width=215)
        logo_label.setContentsMargins(0, 10, 0, 5)
        header_layout.addWidget(logo_label)


        # Welcome title
        title_label = QLabel("Welcome to PlantLeaf")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setObjectName("titleLabel")
        header_layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel("Choose an option to get started")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setObjectName("subtitleLabel")
        header_layout.addWidget(subtitle_label)
        
        main_layout.addLayout(header_layout)

        # Spacer
        main_layout.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Buttons section
        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(10)
        
        # Create buttons
        btn_voltage = self.create_main_button("⚡ New Voltage Analysis", "Start a new voltage analysis")
        btn_audio = self.create_main_button("🔊 New Audio Analysis", "Start a new audio analysis")
        btn_open = self.create_main_button("📁 Open Existing File", "Open a previously saved analysis file")
        
        # Add buttons to layout
        for btn in (btn_voltage, btn_audio, btn_open):
            buttons_layout.addWidget(btn)
        
        # Center the buttons
        buttons_container = QWidget()
        buttons_container.setLayout(buttons_layout)
        buttons_container.setMaximumWidth(400)
        
        buttons_h_layout = QHBoxLayout()
        buttons_h_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        buttons_h_layout.addWidget(buttons_container)
        buttons_h_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        main_layout.addLayout(buttons_h_layout)

        # Connect signals
        btn_open.clicked.connect(self.open_file_action)
        btn_voltage.clicked.connect(self.new_voltage_analysis)
        btn_audio.clicked.connect(self.new_audio_analysis)

        # Bottom spacer
        main_layout.addSpacerItem(QSpacerItem(20, 15, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        # Footer
        footer_label = QLabel("PlantLeaf v1.0 - Tommaso Vaninetti")
        footer_label.setAlignment(Qt.AlignCenter)
        footer_label.setObjectName("footerLabel")
        main_layout.addWidget(footer_label)

        # Apply styles
        #self.()


    def create_main_button(self, text, tooltip=""):
        """Create a styled main button - ALTEZZA RIDOTTA"""
        button = QPushButton(text)
        button.setToolTip(tooltip)
        # PULSANTI MENO ALTI - da 45px a 35px
        button.setMinimumHeight(35)
        button.setMaximumHeight(35)
        button.setObjectName("mainButton")
        return button


    #AZIONI PULSANTI PRINCIPALI


    def new_voltage_analysis(self):
        """Start a new voltage analysis"""
        print("🔋 Starting new voltage analysis...")
        try:
            from windows.main_window_voltage import MainWindowVoltage
            # Chiudi PRIMA la home
            self.hide()  # Nascondi invece di chiudere
            voltage_window = MainWindowVoltage()
            self.layout_manager.center_window_on_screen(voltage_window)
            voltage_window.show()
            print("✅ Voltage analysis window opened")
        except Exception as e:
            self.show()  # Mostra di nuovo la home se c'è un errore
            print(f"❌ Error opening voltage window: {e}")
            self.show_error_dialog("Error", f"Could not open voltage analysis:\n{str(e)}")



    def new_audio_analysis(self):
        """Start a new audio analysis"""
        print("🎵 Starting new audio analysis...")
        try:
            from windows.main_window_audio import MainWindowAudio
            # Chiudi PRIMA la home
            self.hide()  # Nascondi invece di chiudere
            audio_window = MainWindowAudio()
            self.layout_manager.center_window_on_screen(audio_window)
            audio_window.show()
            print("✅ Audio analysis window opened")
        except Exception as e:
            self.show()  # Mostra di nuovo la home se c'è un errore
            print(f"❌ Error opening audio window: {e}")
            self.show_error_dialog("Error", f"Could not open audio analysis:\n{str(e)}")

    
    def closeEvent(self, event):
        """Override del closeEvent per la finestra home"""
        # Chiudi direttamente senza passare per BaseWindow
        event.accept()
        super().closeEvent(event)


    def show_error_dialog(self, title, message):
        """Show an error dialog to the user"""
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setIcon(QMessageBox.Critical)
        msg.addButton("OK", QMessageBox.AcceptRole)
        msg.exec()


