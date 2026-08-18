# Copyright (C) 2026 Tommaso Vaninetti
#
# This file is part of PlantLeaf.
#
# PlantLeaf is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# PlantLeaf is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with PlantLeaf. If not, see <https://www.gnu.org/licenses/>.

from PySide6.QtWidgets import QLineEdit
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QValidator, QDoubleValidator
import re


def format_hms(seconds: float) -> str:
    """
    Seconds -> "H:MM:SS.ss".

    Hours are always shown so the field never changes width mid-playback, which
    would make the label jitter. Two decimals keep sub-second position visible:
    a frame is 2.56 ms, and clicks are located to the frame.
    """
    try:
        total = float(seconds)
    except (TypeError, ValueError):
        total = 0.0
    if total < 0 or total != total:          # negative or NaN
        total = 0.0
    hours, rem = divmod(total, 3600.0)
    minutes, secs = divmod(rem, 60.0)
    # Guard the 59.995 -> "60.00" carry that would print e.g. 0:01:60.00
    if round(secs, 2) >= 60.0:
        secs = 0.0
        minutes += 1
        if minutes >= 60:
            minutes = 0
            hours += 1
    return f"{int(hours)}:{int(minutes):02d}:{secs:05.2f}"


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
        # Width and max length sized for the H:MM:SS.ss/H:MM:SS.ss readout, which is
        # 21 characters ("0:20:34.56/1:50:00.00"). The previous 20-character cap was
        # set for the old "1234.56/6600.00s" form and silently TRUNCATED the new one,
        # dropping the last digit of the total duration.
        self.setFixedWidth(210)
        self.setAlignment(Qt.AlignCenter)
        self.setPlaceholderText("0:00:00.00/0:00:00.00")

        # Validatore permissivo (accetta numeri, ":", ".")
        # Generous enough for the display string plus a hand-typed value.
        self.setMaxLength(32)
        
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
        
        display_text = (f"{format_hms(current_sec)}"
                        f"/{format_hms(self.total_duration_sec)}")
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
        self.setText(format_hms(self.current_time_sec))
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
        Parse a time input into seconds, accepting every reasonable spelling.

            "90"          -> 90.0        plain seconds
            "90.5"        -> 90.5
            "1:30"        -> 90.0        M:SS
            "1:30.5"      -> 90.5
            "0:01:30"     -> 90.0        H:MM:SS
            "1:49:36.42"  -> 6576.42     H:MM:SS.ss  (the display format)

        A single dot is always a decimal fraction, never a field separator, so
        "1:30.5" is unambiguously 90.5 s. That is why H:MM.SS is NOT accepted:
        it cannot be told apart from M:SS.s without guessing.

        The display uses H:MM:SS.ss, but typing plain seconds has always worked and
        that habit is preserved deliberately. Returns None on anything unparseable,
        which the caller renders as the invalid style.
        """
        text = text.strip().lower().replace('s', '')
        if not text:
            return None

        try:
            if ':' in text:
                parts = text.split(':')
                if len(parts) == 2:
                    minutes, seconds = float(parts[0]), float(parts[1])
                    if seconds >= 60:
                        return None
                    return minutes * 60.0 + seconds
                if len(parts) == 3:
                    hours, minutes = float(parts[0]), float(parts[1])
                    seconds = float(parts[2])
                    if minutes >= 60 or seconds >= 60:
                        return None
                    return hours * 3600.0 + minutes * 60.0 + seconds
                return None

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