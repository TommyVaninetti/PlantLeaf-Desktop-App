"DATA TABLE"

import time
from PySide6.QtWidgets import (QTableWidget, QTableWidgetItem, QLineEdit, 
                              QHeaderView, QStyledItemDelegate)
from PySide6.QtCore import Qt

class NotesDelegate(QStyledItemDelegate):
    """Delegate personalizzato per editing delle note con campo più grande"""
    
    def createEditor(self, parent, option, index):
        if index.column() == 4:  # Colonna Notes
            editor = QLineEdit(parent)
            editor.setMaxLength(20)
            editor.setStyleSheet("""
                QLineEdit {
                    padding: 8px;
                    font-size: 16px;
                    border: 2px solid #4CAF50;
                    border-radius: 4px;
                    background-color: white;
                    color: black;
                }
            """)
            return editor
        return super().createEditor(parent, option, index)
    
    def setEditorData(self, editor, index):
        if index.column() == 4:
            text = index.model().data(index, Qt.ItemDataRole.EditRole) or ""
            editor.setText(text)
        else:
            super().setEditorData(editor, index)
    
    def setModelData(self, editor, model, index):
        if index.column() == 4:
            text = editor.text()[:20]
            model.setData(index, text, Qt.ItemDataRole.EditRole)
        else:
            super().setModelData(editor, model, index)

    def updateEditorGeometry(self, editor, option, index):
        if index.column() == 4:
            # Editor più alto e largo, centrato sulla cella
            rect = option.rect
            rect.setHeight(40)
            rect.setWidth(max(rect.width(), 220))
            editor.setGeometry(rect)
        else:
            super().updateEditorGeometry(editor, option, index)

class DataTable(QTableWidget):
    def __init__(self, theme_manager, parent=None):
        super().__init__(parent)
        self.theme_manager = theme_manager
        self.setup_table()
    
    def setup_table(self):
        """Configura la tabella con colonne appropriate"""
        # Imposta 5 colonne: Timestamp, Frequency, Amplitude, Duration, Notes
        self.setColumnCount(5)
        self.setHorizontalHeaderLabels(["Timestamp", "Frequency", "Amplitude", "Duration", "Notes"])
        
        # NUOVO: Imposta delegate personalizzato per editing
        self.notes_delegate = NotesDelegate()
        self.setItemDelegate(self.notes_delegate)
        
        # Configura il comportamento delle colonne
        header = self.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # Timestamp
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)  # Frequency
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)  # Amplitude  
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)  # Duration
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)           # Notes

        # Imposta larghezza minima per la colonna Notes
        self.setColumnWidth(4, 180)  # <-- AGGIUNGI QUESTA RIGA

        # Imposta altezza minima per le righe
        self.verticalHeader().setDefaultSectionSize(25)
        self.verticalHeader().setVisible(False)
        
        # Configura comportamento generale
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(False)
        
        self.setToolTip(
            "Double click on 'Notes' to edit (max 20 characters).\n"
        )
        
    def add_data_row(self, data):
        """Aggiunge una riga di dati alla tabella"""
        expected_cols = 5  # Timestamp, Frequency, Amplitude, Duration, Notes
        if len(data) != expected_cols:
            print(f"⚠️ Dati incompleti per tabella: {len(data)} colonne (attese {expected_cols})")
            return
            
        row_position = self.rowCount()
        self.insertRow(row_position)

        # Timestamp (colonna 0)
        timestamp_item = QTableWidgetItem(f"{data[0]:.2f} s")
        timestamp_item.setFlags(timestamp_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.setItem(row_position, 0, timestamp_item)

        # Frequency (colonna 1)
        freq_item = QTableWidgetItem(str(data[1]))
        freq_item.setFlags(freq_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.setItem(row_position, 1, freq_item)

        # Amplitude (colonna 2)
        amp_item = QTableWidgetItem(str(data[2]))
        amp_item.setFlags(amp_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.setItem(row_position, 2, amp_item)

        # Duration (colonna 3)
        duration_item = QTableWidgetItem(str(data[3]))
        duration_item.setFlags(duration_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.setItem(row_position, 3, duration_item)

        # Notes (colonna 4)
        notes_item = NotesTableItem(str(data[4]))
        self.setItem(row_position, 4, notes_item)

        self.scrollToBottom()
    
    def export_click_data(self):
        click_data = []
        for row in range(self.rowCount()):
            timestamp_item = self.item(row, 0)
            freq_item = self.item(row, 1)
            amp_item = self.item(row, 2)
            dur_item = self.item(row, 3)
            notes_item = self.item(row, 4)

            try:
                timestamp = float(timestamp_item.text().replace('s', '').strip())
                frequency = float(freq_item.text().replace('Hz', '').strip())
                amplitude = float(amp_item.text().replace('V', '').strip())
                
                # ✅ NUOVO: Gestisci sia formato vecchio (μs/ms) che nuovo (FFT)
                duration_text = dur_item.text().strip()
                
                if "FFT" in duration_text:
                    # ✅ NUOVO FORMATO: "3 FFT" → salva numero FFT
                    fft_count = int(duration_text.replace('FFT', '').strip())
                    # Converti in microsecondi per retrocompatibilità (2.56ms per FFT)
                    duration_us = int(fft_count * 2560)  # 1 FFT = 2560 μs
                elif "ms" in duration_text:
                    # VECCHIO FORMATO: mantieni per file legacy
                    duration_us = int(float(duration_text.replace('ms', '').strip()) * 1000)
                elif "μs" in duration_text:
                    duration_us = int(duration_text.replace('μs', '').strip())
                else:
                    duration_us = 0
                
                notes = notes_item.text()[:20] if notes_item else ""
                
                click_data.append({
                    "timestamp": timestamp,
                    "frequency": frequency,
                    "amplitude": amplitude,
                    "duration_us": duration_us,  # ✅ Sempre salvato in μs per compatibilità
                    "notes": notes
                })
            except Exception as e:
                print(f"⚠️ Errore export click row {row}: {e}")
                
        return click_data

class NotesTableItem(QTableWidgetItem):
    """Item personalizzato per le note con limite di caratteri"""
    
    def __init__(self, text=""):
        super().__init__(text[:20])  # Limita a 20 caratteri
        self.setFlags(self.flags() | Qt.ItemFlag.ItemIsEditable)
        # Aggiungi tooltip
        self.setToolTip("Doppio click per modificare (max 20 caratteri)")
        
    def setData(self, role, value):
        """Override per limitare i caratteri inseriti"""
        if role == Qt.ItemDataRole.EditRole and isinstance(value, str):
            # Limita a 20 caratteri
            limited_value = value[:20]
            super().setData(role, limited_value)
            # Aggiorna tooltip con caratteri rimanenti
            remaining = 20 - len(limited_value)
            self.setToolTip(f"Caratteri rimanenti: {remaining}/20")
        else:
            super().setData(role, value)