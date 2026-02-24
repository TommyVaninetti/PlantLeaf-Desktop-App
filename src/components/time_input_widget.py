from PySide6.QtWidgets import QLineEdit
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QValidator, QDoubleValidator
import re


class TimeInputWidget(QLineEdit):
    """
    Widget per input/display del tempo corrente.
    Supporta formati: "123.45s", "1:30", "90" (secondi)
    Emette timeChanged quando l'utente conferma un nuovo valore valido.
    """
    timeChanged = Signal(float)  # Emette il tempo in secondi
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.total_duration_sec = 0.0
        self.current_time_sec = 0.0
        self.is_editing = False
        
        # ✅ FIX: Dimensione ridotta e allineamento
        self.setFixedWidth(180)
        self.setAlignment(Qt.AlignCenter)
        self.setPlaceholderText("0.00/0.00s")
        
        # Validatore permissivo (accetta numeri, ":", ".")
        self.setMaxLength(20)
        
        # Connessioni
        self.editingFinished.connect(self._on_editing_finished)
        self.textChanged.connect(self._on_text_changed)
        
        # ✅ SALVA IL FONT INIZIALE (ereditato dalla toolbar)
        self._saved_font = None
        
        # ✅ Stile iniziale con font system
        self._set_normal_style()
    
    def set_time(self, current_sec: float, total_sec: float = None):
        """
        Aggiorna il display del tempo (chiamato dal sistema).
        Non triggera timeChanged.
        """
        if self.is_editing:
            return  # Non aggiornare durante editing utente
        
        self.current_time_sec = current_sec
        
        if total_sec is not None:
            self.total_duration_sec = total_sec
        
        display_text = f"{current_sec:.2f}/{self.total_duration_sec:.2f}s"
        self.blockSignals(True)
        self.setText(display_text)
        self.blockSignals(False)
        
        # ✅ FIX CRITICO: Ripristina il font dopo setText()
        # setText() può resettare il font, quindi lo riapplichiamo
        if self._saved_font is not None:
            self.setFont(self._saved_font)
    
    def focusInEvent(self, event):
        """Quando l'utente clicca, mostra solo il tempo corrente editabile"""
        super().focusInEvent(event)
        self.is_editing = True
        
        # ✅ SALVA IL FONT PRIMA DI EDITARE
        self._saved_font = self.font()
        
        # Mostra solo la parte editabile (prima della /)
        self.blockSignals(True)
        self.setText(f"{self.current_time_sec:.2f}")
        self.selectAll()  # Seleziona tutto per facilitare sovrascrittura
        self.blockSignals(False)
        
        self._set_editing_style()
    
    def focusOutEvent(self, event):
        """Quando perde focus, ripristina display completo"""
        super().focusOutEvent(event)
        self.is_editing = False
        
        # ✅ NON chiamare _set_normal_style() qui
        # Il font è già corretto, basta aggiornare il testo
        self.set_time(self.current_time_sec, self.total_duration_sec)
        
        # ✅ Rimuovi il CSS editing e ripristina trasparenza
        self.setStyleSheet("""
            QLineEdit {
                background-color: transparent;
                border: 1px solid transparent;
                color: white;
            }
        """)
        
        # ✅ Riapplica il font salvato (garantito)
        if self._saved_font is not None:
            self.setFont(self._saved_font)
    
    def keyPressEvent(self, event):
        """Gestisce tasti speciali"""
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self._on_editing_finished()
            self.clearFocus()  # Esci dalla modalità editing
            event.accept()
            return
        
        if event.key() == Qt.Key_Escape:
            # Annulla modifiche
            self.set_time(self.current_time_sec, self.total_duration_sec)
            self.clearFocus()
            event.accept()
            return
        
        super().keyPressEvent(event)
    
    def _on_text_changed(self, text):
        """Validazione in tempo reale con feedback visivo"""
        if not self.is_editing:
            return
        
        parsed_time = self._parse_time_input(text)
        
        if parsed_time is None:
            self._set_invalid_style()
        elif parsed_time > self.total_duration_sec:
            self._set_warning_style()  # Tempo oltre durata
        else:
            self._set_valid_style()
    
    def _on_editing_finished(self):
        """Chiamato quando l'utente conferma (Enter o perde focus)"""
        if not self.is_editing:
            return
        
        input_text = self.text().strip()
        parsed_time = self._parse_time_input(input_text)
        
        if parsed_time is None:
            print(f"⚠️ Input tempo non valido: '{input_text}'")
            # Ripristina valore precedente
            self.set_time(self.current_time_sec, self.total_duration_sec)
            return
        
        # Clamp al range valido
        parsed_time = max(0.0, min(parsed_time, self.total_duration_sec))
        
        if abs(parsed_time - self.current_time_sec) > 0.01:  # Tolleranza 10ms
            print(f"⏱️ Time input: {input_text} → {parsed_time:.2f}s")
            self.current_time_sec = parsed_time
            self.timeChanged.emit(parsed_time)  # ✅ EMETTI SEGNALE
        
        # ✅ FIX: Non aggiornare display qui, sarà fatto in focusOutEvent
        # Questo evita chiamate duplicate a set_time()
    
    def _parse_time_input(self, text: str) -> float:
        """
        Parse intelligente di input tempo.
        Supporta:
        - "123.45" → 123.45s
        - "1:30" → 90s
        - "1:30.5" → 90.5s
        - "90s" → 90s
        """
        text = text.strip().lower().replace('s', '')
        
        try:
            # Formato MM:SS o MM:SS.ms
            if ':' in text:
                parts = text.split(':')
                if len(parts) != 2:
                    return None
                
                minutes = float(parts[0])
                seconds = float(parts[1])
                
                return minutes * 60 + seconds
            
            # Formato semplice: secondi
            return float(text)
        
        except ValueError:
            return None
    
    def _set_normal_style(self):
        """Stile normale (solo display) - ✅ INHERIT FONT DA TOOLBAR"""
        self.setReadOnly(False)
        
        # ✅ SALVA IL FONT DAL PARENT
        if self.parent() and self._saved_font is None:
            parent_font = self.parent().font()
            self.setFont(parent_font)
            self._saved_font = parent_font
        
        self.setStyleSheet("""
            QLineEdit {
                background-color: transparent;
                border: 1px solid transparent;
                color: white;
            }
        """)
    
    def _set_editing_style(self):
        """Stile durante editing - MANTIENI FONT CORRENTE"""
        # ✅ NON salvare il font qui, è già salvato in focusInEvent
        
        self.setStyleSheet("""
            QLineEdit {
                background-color: #2a2a2a;
                border: 2px solid #4a9eff;
                border-radius: 3px;
                color: white;
                padding: 4px;
            }
        """)
        
        # ✅ Riapplica il font salvato
        if self._saved_font is not None:
            self.setFont(self._saved_font)
    
    def _set_valid_style(self):
        """Stile per input valido"""
        self.setStyleSheet("""
            QLineEdit {
                background-color: #2a2a2a;
                border: 2px solid #4ade80;
                border-radius: 3px;
                color: white;
                padding: 4px;
            }
        """)
        if self._saved_font is not None:
            self.setFont(self._saved_font)
    
    def _set_invalid_style(self):
        """Stile per input non valido"""
        self.setStyleSheet("""
            QLineEdit {
                background-color: #2a2a2a;
                border: 2px solid #ef4444;
                border-radius: 3px;
                color: #ef4444;
                padding: 4px;
            }
        """)
        if self._saved_font is not None:
            self.setFont(self._saved_font)
    
    def _set_warning_style(self):
        """Stile per input oltre durata"""
        self.setStyleSheet("""
            QLineEdit {
                background-color: #2a2a2a;
                border: 2px solid #f59e0b;
                border-radius: 3px;
                color: #f59e0b;
                padding: 4px;
            }
        """)
        if self._saved_font is not None:
            self.setFont(self._saved_font)