"""
Componente pulsante Start/Stop riutilizzabile
"""

from PySide6.QtWidgets import QPushButton
from PySide6.QtCore import Signal
from PySide6.QtGui import QFont
import platform
from core.special_component import SpecialComponent

class StartStopButton(QPushButton, SpecialComponent):
    started = Signal()
    stopped = Signal()
    
    def __init__(self, theme_manager, text_start="START", text_stop="STOP", parent=None,
                 replace_widget_name=None, replace_with_widget=None):
        super().__init__(parent)
        self.setText(text_start)
        
        # Configurazione iniziale
        self.text_start = text_start
        self.text_stop = text_stop
        self.is_running = False
        self.theme_manager = theme_manager
        
        # Setup iniziale
        self.setText(self.text_start)
        self.setCheckable(False)
        
        # Connetti il click
        self.clicked.connect(self.toggle_state)
        
        # Applica stile iniziale
        self.set_initial_style()


    def set_initial_style(self):
        """Applica lo stile iniziale"""
        self.setStyleSheet(self.theme_manager.get_start_stop_css(is_running=False))

    def start(self):
        if not self.is_running:
            self.toggle_state()

    def stop(self):
        if self.is_running:
            self.toggle_state()
    
    #Aggiornamento dello stato e dello stile del pulsante
    def toggle_state(self):
        print(f"Toggling state from {self.is_running} to {not self.is_running}")
        self.is_running = not self.is_running
        
        # Aggiorna il testo e il tema (vedi theme_manager.py)
        if self.is_running:
            self.setText(self.text_stop)
            self.setStyleSheet(self.theme_manager.get_start_stop_css(is_running=True))
        else:
            self.setText(self.text_start)
            self.setStyleSheet(self.theme_manager.get_start_stop_css(is_running=False))
        
        # Emetti segnali
        if self.is_running:
            self.started.emit()
        else:
            self.stopped.emit()
        # Nella classe specifica implementare la funzione che gestisce il segnale
        #eg. self.start_stop_button.started.connect(self.on_started)
        #.   self.start_stop_button.stopped.connect(self.on_stopped)

