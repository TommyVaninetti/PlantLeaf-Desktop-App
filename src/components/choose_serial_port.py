from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QHBoxLayout, QComboBox, QPushButton, QLabel
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPixmap
from config.app_config import AppConfig
import serial.tools.list_ports


class ChooseSerialPort(QDialog):
    #segnale per la selezione della porta seriale
    serial_port_selected = Signal(str)

    def __init__(self, theme_manager, parent=None):
        super().__init__(parent)
        self.theme_manager = theme_manager
        self.setWindowTitle("Choose Serial Port")
        self.setModal(True)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setFixedSize(450, 175)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Titolo
        title = QLabel("Select Serial Port from the list below:")
        title.setObjectName("titleLabel")
        title_font = QFont()
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Porta seriale
        port_layout = QHBoxLayout()
        port_label = QLabel("Serial Port:")
        self.port_combo = QComboBox()
        self.port_combo.setFixedWidth(300)  # Lunghezza fissa
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_ports)
        port_layout.addWidget(port_label)
        port_layout.addWidget(self.port_combo)
        port_layout.addWidget(refresh_button)
        layout.addLayout(port_layout)

        # Pulsanti OK/Cancel
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(self.ok_btn_clicked)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(refresh_button)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Applica font coerenti
        ok_btn.setStyleSheet(self.theme_manager.get_toggle_button_style(is_active=True))
        cancel_btn.setStyleSheet(self.theme_manager.get_toggle_button_style(is_active=False))
        refresh_button.setStyleSheet(self.theme_manager.get_toggle_button_style(is_active=False))
        self.theme_manager.apply_theme_to_widgets(self)

        self.refresh_ports()


    def refresh_ports(self):
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        keywords = [
    "ch340", "usb serial", "esp32", "stm32", "STM32",
    "silicon labs", "ftdi", "microcontroller", 
    "virtual com port", "vcp", "stmicroelectronics",  
    "usb composite device", "communication device",
    "vid_0483&pid_5740", #STM32
    "vid_10c4&pid_ea60", #Silicon Labs
    "vid_1a86&pid_7523", #CH340
    "usb",
]
        found = False
        for port in ports:
            desc = (port.description or "").lower()
            manuf = (getattr(port, "manufacturer", "") or "").lower()
            if any(k in desc or k in manuf for k in keywords):
                label = f"{port.description} ({port.device})"
                self.port_combo.addItem(label, port.device)
                found = True
        if not found:
            self.port_combo.addItem("No ports found", "NO_PORT")
    

    def ok_btn_clicked(self): #sovrascive il metodo per gestire il click su OK
        """Logica da eseguire quando si clicca OK"""
        selected_port = self.port_combo.currentData()
        if selected_port and selected_port != "NO_PORT":
            print(f"🔌 Porta seriale selezionata: {selected_port}")
            self.accept()  # Chiude il dialogo e ritorna i dati selezionati
            #emette segnale a funzione per creare la connessione seriale
            self.serial_port_selected.emit(selected_port)

            #esempio di handle del segnale
            # self.serial_port_selected.connect(self.create_serial_connection)

        else:
            print("❌ Nessuna porta seriale selezionata o disponibile.")
            self.reject()  # Chiude il dialogo senza ritorno dati

    
