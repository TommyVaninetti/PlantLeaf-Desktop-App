from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QHBoxLayout, QSpinBox, QPushButton, QLineEdit, QLabel
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPixmap
from config.app_config import AppConfig


class VoltageSamplingSettingsPopup(QDialog):
    def __init__(self, theme_manager, parent=None):
        super().__init__(parent)
        self.theme_manager = theme_manager
        self.setWindowTitle("Voltage Sampling Settings")
        self.setModal(True)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setFixedSize(400, 250)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        title = QLabel("Settings for voltage sampling:")
        title.setObjectName("titleLabel")
        title_font = QFont()
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Sampling Rate
        rate_layout = QHBoxLayout()
        rate_label = QLabel("Sampling Rate (Hz):")
        self.rate_spin = QSpinBox()
        self.rate_spin.setRange(50, 1000)
        self.rate_spin.setSingleStep(50)
        if hasattr(parent, 'sampling_rate'):
            self.rate_spin.setValue(parent.sampling_rate)

        self.rate_spin.setFixedSize(100, 25)

        rate_layout.addWidget(rate_label)
        rate_layout.addWidget(self.rate_spin)
        layout.addLayout(rate_layout)

        # Type of Experiment (QLineEdit, max 20 caratteri)
        type_layout = QHBoxLayout()
        type_label = QLabel("Type of Experiment:")
        self.type_edit = QLineEdit()
        self.type_edit.setMaxLength(20)
        if hasattr(parent, 'type_of_experiment'):
            self.type_edit.setText(parent.type_of_experiment)
        type_layout.addWidget(type_label)
        type_layout.addWidget(self.type_edit)
        layout.addLayout(type_layout)
        

        rate_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        type_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.type_edit.setFixedSize(200, 40)


        # Pulsante OK
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(self.ok_btn_clicked)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)


        # Applica il tema accedendo a theme_manager di BaseWindow
        # Applica font coerenti
        self.theme_manager.apply_theme_to_widgets(self)

        ok_btn.setStyleSheet(self.theme_manager.get_toggle_button_style(is_active=True))
        cancel_btn.setStyleSheet(self.theme_manager.get_toggle_button_style(is_active=False))


    def get_settings(self):
        return {
            "sampling_rate": self.rate_spin.value(),
            "experiment_type": self.type_edit.text()
        }
    
    def ok_btn_clicked(self): #da sovrascivere con logica personalizzata
        """Logica da eseguire quando si clicca OK"""
        print(f"impostazioni voltage: {self.get_settings()}")
        self.accept()  # Chiude il dialogo






class AudioSamplingSettingsPopup(QDialog):
    def __init__(self, theme_manager, parent=None):
        super().__init__(parent)
        self.theme_manager = theme_manager
        self.setWindowTitle("Audio Sampling Settings")
        self.setModal(True)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setFixedSize(400, 175)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        title = QLabel("Type of test for audio:")
        title.setObjectName("titleLabel")
        title_font = QFont()
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)


        # Type of Experiment (QLineEdit, max 25 caratteri)
        type_layout = QHBoxLayout()
        type_label = QLabel("Type of Experiment:")
        self.type_edit = QLineEdit()
        self.type_edit.setMaxLength(25)
        type_layout.addWidget(type_label)
        type_layout.addWidget(self.type_edit)
        layout.addLayout(type_layout)
        

        type_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.type_edit.setMinimumWidth(150)


        # Pulsante OK
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(self.ok_btn_clicked)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)


        # Applica il tema accedendo a theme_manager di BaseWindow
        # Applica font coerenti
        self.theme_manager.apply_theme_to_widgets(self)

        ok_btn.setStyleSheet(self.theme_manager.get_toggle_button_style(is_active=True))
        cancel_btn.setStyleSheet(self.theme_manager.get_toggle_button_style(is_active=False))


    def get_settings(self):
        return {
            "experiment_type": self.type_edit.text()
        }
    
    def ok_btn_clicked(self): #da sovrascivere con logica personalizzata
        """Logica da eseguire quando si clicca OK"""
        print(f"impostazioni audio: {self.get_settings()}")
        self.accept()  # Chiude il dialogo

    def set_existing_settings(self, experiment_type):
        """Imposta i valori esistenti nel popup"""
        self.type_edit.setText(experiment_type)