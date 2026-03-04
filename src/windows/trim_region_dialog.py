"""
Dialog per selezione e export di regioni trimmed da file .pvolt/.paudio
"""
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                              QPushButton, QLineEdit, QCheckBox, QDoubleSpinBox,
                              QFileDialog, QGroupBox, QFormLayout, QRadioButton,
                              QButtonGroup, QSlider, QFrame, QMessageBox)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
import os


class TrimRegionDialog(QDialog):
    """
    Dialog per configurare l'export di una regione trimmed.
    Supporta sia file voltage che audio con opzioni specifiche.
    """
    
    def __init__(self, parent, file_path, file_type, total_duration_sec):
        """
        Args:
            parent: ReplayBaseWindow instance
            file_path: Path al file corrente
            file_type: 'voltage' o 'audio'
            total_duration_sec: Durata totale del file in secondi
        """
        super().__init__(parent)
        
        self.parent_window = parent
        self.file_path = file_path
        self.file_type = file_type
        self.total_duration_sec = total_duration_sec
        
        # ✅ Eredita tema dal parent
        if hasattr(parent, 'theme_manager'):
            self.theme_manager = parent.theme_manager
        
        self.setWindowTitle("Export Trimmed Region")
        self.setFixedSize(600, 805)
        
        self._setup_ui()
        self._apply_theme()
        
        # Inizializza con range di default (centro del file, 30s)
        self._set_default_range()
    
    def _setup_ui(self):
        """Costruisce l'interfaccia completa del dialog"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Ottieni font size standard dal FontManager del parent
        if hasattr(self.parent_window, 'font_manager'):
            font_sizes = self.parent_window.font_manager.get_scaled_font_sizes()
            self.standard_font_size = font_sizes.get('info_label', 14)  # 14px è il default
        else:
            self.standard_font_size = 14
        
        # === TITOLO ===
        self.title_label = QLabel("Export Trimmed Region")
        self.title_font = QFont()
        self.title_font.setPointSize(21)  # Dimensione maggiore per il titolo VIENE SOVRASCRITTA DAL TEMA, quindi la aggiungo a fine __init__
        self.title_font.setBold(True)
        self.title_label.setFont(self.title_font)
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)

        layout.addSpacing(5)
        
        # === SEZIONE 1: RANGE SELECTION ===
        range_group = self._create_range_selection_group()
        layout.addWidget(range_group)
        
        # === SEZIONE 2: OUTPUT FILE ===
        output_group = self._create_output_file_group()
        layout.addWidget(output_group)
        
        # === SEZIONE 3: TYPE-SPECIFIC OPTIONS ===
        options_group = self._create_options_group()
        layout.addWidget(options_group)
        
        # === SEZIONE 4: PREVIEW ===
        preview_group = self._create_preview_group()
        layout.addWidget(preview_group)
        
        layout.addSpacing(5)
        
        # === PULSANTI FINALI ===
        button_layout = self._create_button_layout()
        layout.addLayout(button_layout)
    
    def _create_range_selection_group(self):
        """Crea il gruppo per la selezione del range temporale"""
        group = QGroupBox("Select Time Range")
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(10, 20, 10, 10)  # Top margin maggiore per non tagliare il titolo
        
        # === INPUT MANUALI - START ED END SULLA STESSA RIGA (label + spinbox affiancati) ===
        time_inputs_layout = QHBoxLayout()
        time_inputs_layout.setSpacing(10)
        
        # Start time: label + spinbox orizzontali
        start_label = QLabel("Start Time:")
        start_label.setStyleSheet(f"font-size: {self.standard_font_size}px;")
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(0, self.total_duration_sec)
        self.start_spin.setValue(0)
        self.start_spin.setDecimals(2)
        self.start_spin.setSuffix(" s")
        self.start_spin.setSingleStep(1)
        self.start_spin.setFixedWidth(120)
        # Valida solo quando l'editing è completato, non ad ogni cambio
        self.start_spin.editingFinished.connect(self._validate_and_update)
        time_inputs_layout.addWidget(start_label)
        time_inputs_layout.addWidget(self.start_spin)
        
        # Separator frame
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setLineWidth(1)
        time_inputs_layout.addWidget(separator)
        
        # End time: label + spinbox orizzontali
        end_label = QLabel("End Time:")
        end_label.setStyleSheet(f"font-size: {self.standard_font_size}px;")
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(0, self.total_duration_sec)
        self.end_spin.setValue(self.total_duration_sec)
        self.end_spin.setDecimals(2)
        self.end_spin.setSuffix(" s")
        self.end_spin.setSingleStep(1)
        self.end_spin.setFixedWidth(120)
        # Valida solo quando l'editing è completato, non ad ogni cambio
        self.end_spin.editingFinished.connect(self._validate_and_update)
        time_inputs_layout.addWidget(end_label)
        time_inputs_layout.addWidget(self.end_spin)
        
        time_inputs_layout.addStretch()
        layout.addLayout(time_inputs_layout)
        
        layout.addSpacing(10)
        
        group.setLayout(layout)
        return group
    
    def _create_output_file_group(self):
        """Crea il gruppo per la selezione del file di output"""
        group = QGroupBox("Output Mode")
        layout = QVBoxLayout()
        layout.setSpacing(6)
        layout.setContentsMargins(10, 20, 10, 10)  # Top margin maggiore per non tagliare il titolo
        
        # === RADIO BUTTONS ===
        self.output_mode_group = QButtonGroup(self)
        
        # Opzione 1: Nuovo file
        self.new_file_radio = QRadioButton("Create new file (recommended)")
        self.new_file_radio.setStyleSheet(f"font-size: {self.standard_font_size}px;")
        self.new_file_radio.setChecked(True)
        self.output_mode_group.addButton(self.new_file_radio, 0)
        layout.addWidget(self.new_file_radio)
        
        # File path edit + browse (mostra solo il nome file)
        file_layout = QHBoxLayout()
        file_layout.setContentsMargins(20, 0, 0, 0)  # Indent
        file_layout.setSpacing(5)
        
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("Choose output filename...")
        self.output_path_edit.setStyleSheet(f"font-size: {self.standard_font_size}px;")
        self.output_path_edit.setReadOnly(True)  # Solo tramite browse button
        
        browse_btn = QPushButton("Browse...")
        browse_btn.setFixedWidth(95)  # Allargato per evitare il taglio della B
        browse_btn.setAutoDefault(False)  # Evita che diventi default quando si preme Enter
        browse_btn.clicked.connect(self._browse_output_file)
        
        file_layout.addWidget(self.output_path_edit)
        file_layout.addWidget(browse_btn)
        layout.addLayout(file_layout)
        
        # Opzione 2: Overwrite
        layout.addSpacing(5)
        self.overwrite_radio = QRadioButton("Overwrite current file (backup created)")
        self.overwrite_radio.setStyleSheet(f"font-size: {self.standard_font_size}px;")
        self.output_mode_group.addButton(self.overwrite_radio, 1)
        layout.addWidget(self.overwrite_radio)
        
        # Connetti radio buttons
        self.new_file_radio.toggled.connect(self._on_output_mode_changed)
        
        group.setLayout(layout)
        return group
    
    def _create_options_group(self):
        """Crea il gruppo per le opzioni specifiche del tipo di file"""
        group = QGroupBox("Export Options")
        layout = QVBoxLayout()
        layout.setSpacing(6)
        layout.setContentsMargins(10, 20, 10, 10)  # Top margin maggiore per non tagliare il titolo
        
        if self.file_type == 'audio':
            # === AUDIO-SPECIFIC OPTIONS ===
            self.include_clicks_check = QCheckBox("Export click events in selected range")
            self.include_clicks_check.setStyleSheet(f"font-size: {self.standard_font_size}px;")
            self.include_clicks_check.setChecked(True)
            self.include_clicks_check.setToolTip(
                "Include only click events that are COMPLETELY within the selected range"
            )
            layout.addWidget(self.include_clicks_check)
            
            # Label info click
            self.click_count_label = QLabel("📊 0 of 0 clicks in range")
            self.click_count_label.setStyleSheet(f"font-size: {self.standard_font_size}px; font-style: italic;")
            layout.addWidget(self.click_count_label)
            
        elif self.file_type == 'voltage':
            # === VOLTAGE-SPECIFIC OPTIONS ===
            self.include_analyses_check = QCheckBox("Export saved analyses in selected range")
            self.include_analyses_check.setStyleSheet(f"font-size: {self.standard_font_size}px;")
            self.include_analyses_check.setChecked(True)
            self.include_analyses_check.setToolTip(
                "Include only analyses that are COMPLETELY within the selected range"
            )
            layout.addWidget(self.include_analyses_check)
            
            # Label info analyses
            self.analyses_count_label = QLabel("📊 0 of 0 analyses in range")
            self.analyses_count_label.setStyleSheet(f"font-size: {self.standard_font_size}px; font-style: italic;")
            layout.addWidget(self.analyses_count_label)
        
        group.setLayout(layout)
        return group
    
    def _create_preview_group(self):
        """Crea il gruppo per il preview dell'export"""
        group = QGroupBox("Export Preview")
        layout = QVBoxLayout()  # Cambiato da QFormLayout a QVBoxLayout per controllo allineamento
        layout.setSpacing(6)
        layout.setContentsMargins(10, 20, 10, 10)  # Top margin maggiore per non tagliare il titolo
        
        # Source file info
        filename = os.path.basename(self.file_path)
        filesize_mb = os.path.getsize(self.file_path) / (1024 * 1024)
        
        source_layout = QHBoxLayout()
        source_layout.setSpacing(8)
        source_key = QLabel("Source:")
        source_key.setStyleSheet(f"font-size: {self.standard_font_size}px; font-weight: bold;")
        source_key.setFixedWidth(120)  # Larghezza fissa per allineamento
        source_label = QLabel(f"{filename}")
        source_label.setStyleSheet(f"font-size: {self.standard_font_size}px;")
        source_layout.addWidget(source_key)
        source_layout.addWidget(source_label)
        source_layout.addStretch()
        layout.addLayout(source_layout)
        
        # Output Duration
        duration_layout = QHBoxLayout()
        duration_layout.setSpacing(8)
        duration_key = QLabel("Duration:")
        duration_key.setStyleSheet(f"font-size: {self.standard_font_size}px; font-weight: bold;")
        duration_key.setFixedWidth(120)
        self.output_duration_label = QLabel("0.00s")
        self.output_duration_label.setStyleSheet(f"font-size: {self.standard_font_size}px;")
        duration_layout.addWidget(duration_key)
        duration_layout.addWidget(self.output_duration_label)
        duration_layout.addStretch()
        layout.addLayout(duration_layout)
        
        # Data Points
        points_layout = QHBoxLayout()
        points_layout.setSpacing(8)
        points_key = QLabel("Data Points:")
        points_key.setStyleSheet(f"font-size: {self.standard_font_size}px; font-weight: bold;")
        points_key.setFixedWidth(120)
        self.data_points_label = QLabel("0 points")
        self.data_points_label.setStyleSheet(f"font-size: {self.standard_font_size}px;")
        points_layout.addWidget(points_key)
        points_layout.addWidget(self.data_points_label)
        points_layout.addStretch()
        layout.addLayout(points_layout)
        
        # Estimated Size
        size_layout = QHBoxLayout()
        size_layout.setSpacing(8)
        size_key = QLabel("Size:")
        size_key.setStyleSheet(f"font-size: {self.standard_font_size}px; font-weight: bold;")
        size_key.setFixedWidth(120)
        self.filesize_label = QLabel("0.0 MB")
        self.filesize_label.setStyleSheet(f"font-size: {self.standard_font_size}px;")
        size_layout.addWidget(size_key)
        size_layout.addWidget(self.filesize_label)
        size_layout.addStretch()
        layout.addLayout(size_layout)
        
        # Size Reduction
        reduction_layout = QHBoxLayout()
        reduction_layout.setSpacing(8)
        reduction_key = QLabel("Size Reduction:")
        reduction_key.setStyleSheet(f"font-size: {self.standard_font_size}px; font-weight: bold;")
        reduction_key.setFixedWidth(120)
        self.reduction_label = QLabel("0%")
        self.reduction_label.setStyleSheet(f"font-size: {self.standard_font_size}px;")
        reduction_layout.addWidget(reduction_key)
        reduction_layout.addWidget(self.reduction_label)
        reduction_layout.addStretch()
        layout.addLayout(reduction_layout)
        
        group.setLayout(layout)
        return group
    
    def _create_button_layout(self):
        """Crea il layout dei pulsanti finali"""
        layout = QHBoxLayout()
        layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedWidth(100)
        cancel_btn.setAutoDefault(False)  # Evita che diventi default quando si preme Enter
        cancel_btn.clicked.connect(self.reject)
        
        self.export_btn = QPushButton("Export")
        self.export_btn.setFixedWidth(100)
        self.export_btn.setAutoDefault(False)  # Evita che diventi default quando si preme Enter
        self.export_btn.clicked.connect(self._on_export_clicked)
        
        layout.addWidget(cancel_btn)
        layout.addWidget(self.export_btn)
        
        return layout
    
    # === EVENT HANDLERS ===
    
    def _validate_and_update(self):
        """Valida il range selezionato e aggiorna il preview (chiamato solo quando editing è completato)"""
        start_val = self.start_spin.value()
        end_val = self.end_spin.value()
        
        # Validazione: start deve essere < end con almeno 0.1s di differenza
        if start_val >= end_val:
            # Correggi automaticamente mantenendo almeno 1s di distanza
            if self.sender() == self.start_spin:
                # L'utente ha modificato start, aggiusta start
                new_start = max(0, end_val - 1.0)
                self.start_spin.blockSignals(True)
                self.start_spin.setValue(new_start)
                self.start_spin.blockSignals(False)
            else:
                # L'utente ha modificato end, aggiusta end
                new_end = min(self.total_duration_sec, start_val + 1.0)
                self.end_spin.blockSignals(True)
                self.end_spin.setValue(new_end)
                self.end_spin.blockSignals(False)
        
        # Aggiorna il preview
        self._update_preview()
    
    def _on_output_mode_changed(self, checked):
        """Handler per cambio modalità output"""
        # Abilita/disabilita path edit e browse button
        is_new_file = self.new_file_radio.isChecked()
        self.output_path_edit.setEnabled(is_new_file)
        
        # Se passa a "new file" e il path è vuoto, genera default
        if is_new_file and not self.output_path_edit.text():
            self._generate_default_filename()
    
    def _browse_output_file(self):
        """Apre dialog per selezione file output"""
        ext = "*.pvolt" if self.file_type == 'voltage' else "*.paudio"
        filter_str = f"{self.file_type.capitalize()} Files ({ext})"
        
        # Usa il full path salvato se esiste
        default_path = getattr(self, '_full_output_path', '')
        if not default_path:
            self._generate_default_filename()
            default_path = getattr(self, '_full_output_path', '')
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Trimmed File",
            default_path,
            filter_str
        )
        
        if filename:
            # Salva il path completo internamente
            self._full_output_path = filename
            # Mostra solo il nome del file
            self.output_path_edit.setText(os.path.basename(filename))
    
    def _on_export_clicked(self):
        """Handler per click sul pulsante Export"""
        # Validazione finale
        if not self._validate_export():
            return
        
        # Se overwrite, mostra warning finale
        if self.overwrite_radio.isChecked():
            reply = QMessageBox.warning(
                self,
                "Confirm Overwrite",
                f"Are you sure you want to overwrite the current file?\n\n"
                f"Original: {os.path.basename(self.file_path)}\n"
                f"A backup will be created automatically.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
        
        self.accept()
    
    # === UTILITY METHODS ===
    
    def _set_default_range(self):
        """Imposta un range di default intelligente"""
        # Default: centro del file, 30s (o 50% se file < 60s)
        if self.total_duration_sec > 60:
            center = self.total_duration_sec / 2
            start = max(0, center - 15)
            end = min(self.total_duration_sec, center + 15)
        else:
            # File corto: usa 25%-75%
            start = self.total_duration_sec * 0.25
            end = self.total_duration_sec * 0.75
        
        self.start_spin.setValue(start)
        self.end_spin.setValue(end)
        
        self._generate_default_filename()
        self._update_preview()
    
    def _generate_default_filename(self):
        """Genera il filename di default basato sul range selezionato"""
        start = self.start_spin.value()
        end = self.end_spin.value()
        
        original_name = os.path.basename(self.file_path)
        name_no_ext = os.path.splitext(original_name)[0]
        ext = os.path.splitext(original_name)[1]
        
        default_name = f"{name_no_ext}_trimmed_{start:.1f}-{end:.1f}s{ext}"
        default_path = os.path.join(os.path.dirname(self.file_path), default_name)
        
        # Salva il path completo internamente
        self._full_output_path = default_path
        # Mostra solo il nome del file
        self.output_path_edit.setText(default_name)
    
    def _update_preview(self):
        """Aggiorna tutte le statistiche di preview"""
        start = self.start_spin.value()
        end = self.end_spin.value()
        duration = end - start
        
        # Calcola riduzione percentuale
        reduction_pct = (duration / self.total_duration_sec) * 100
        kept_pct = 100 - reduction_pct
        
        # Stima dimensione file
        orig_size_mb = os.path.getsize(self.file_path) / (1024 * 1024)
        est_size_mb = orig_size_mb * (duration / self.total_duration_sec)
        
        # Stima data points
        if self.file_type == 'voltage':
            if hasattr(self.parent_window, 'sampling_rate'):
                orig_points = int(self.parent_window.sampling_rate * self.total_duration_sec)
                new_points = int(self.parent_window.sampling_rate * duration)
            else:
                orig_points = int(500 * self.total_duration_sec)
                new_points = int(500 * duration)
        else:  # audio
            if hasattr(self.parent_window, 'data_manager'):
                orig_points = self.parent_window.data_manager.total_frames
                new_points = int(orig_points * (duration / self.total_duration_sec))
            else:
                orig_points = int(390 * self.total_duration_sec)
                new_points = int(390 * duration)
        
        # Aggiorna labels
        self.output_duration_label.setText(f"{duration:.2f}s ({reduction_pct:.1f}% of original)")
        self.data_points_label.setText(f"{new_points:,} ({kept_pct:.1f}% reduction)")
        self.filesize_label.setText(f"~{est_size_mb:.1f} MB")
        self.reduction_label.setText(f"{kept_pct:.1f}% smaller")
        
        # === COUNT CLICKS/ANALYSES ===
        if self.file_type == 'audio':
            self._count_clicks_in_range(start, end)
        elif self.file_type == 'voltage':
            self._count_analyses_in_range(start, end)
    
    def _count_clicks_in_range(self, start, end):
        """Conta i click COMPLETAMENTE nel range"""
        if not hasattr(self.parent_window, 'data_manager'):
            self.click_count_label.setText("No click data available")
            return
        
        click_events = self.parent_window.data_manager.click_events
        total_clicks = len(click_events)
        
        # Filtra click COMPLETAMENTE nel range
        clicks_in_range = [
            c for c in click_events
            if start <= c['time'] <= end and start <= (c['time'] + c['duration']) <= end
        ]
        
        count = len(clicks_in_range)
        self.click_count_label.setText(f"📊 {count} of {total_clicks} clicks in range")
    
    def _count_analyses_in_range(self, start, end):
        """Conta le analisi COMPLETAMENTE nel range"""
        # Leggi analisi dal file
        analyses = self.parent_window._read_analyses_from_file() if hasattr(self.parent_window, '_read_analyses_from_file') else {}
        total_analyses = len(analyses)
        
        if total_analyses == 0:
            self.analyses_count_label.setText("No saved analyses")
            return
        
        # Filtra analisi COMPLETAMENTE nel range
        analyses_in_range = [
            a for a_id, a in analyses.items()
            if (start <= a['parameters']['general']['start_time'] and
                a['parameters']['general']['end_time'] <= end)
        ]
        
        count = len(analyses_in_range)
        self.analyses_count_label.setText(f"📊 {count} of {total_analyses} analyses in range")
    
    def _validate_export(self):
        """Validazione finale prima dell'export"""
        # Check 1: Range valido
        start = self.start_spin.value()
        end = self.end_spin.value()
        duration = end - start
        
        if duration < 0.1:
            QMessageBox.critical(self, "Invalid Range", "Duration must be at least 0.1 seconds")
            return False
        
        # Check 2: Se new file, verifica path
        if self.new_file_radio.isChecked():
            # Usa il full path salvato
            output_path = getattr(self, '_full_output_path', '')
            
            if not output_path:
                QMessageBox.critical(self, "No Output File", "Please specify an output filename")
                return False
            
            # Check se il file esiste già
            if os.path.exists(output_path):
                reply = QMessageBox.question(
                    self,
                    "File Exists",
                    f"File already exists:\n{os.path.basename(output_path)}\n\nOverwrite?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if reply != QMessageBox.Yes:
                    return False
        
        return True
    
    def _apply_theme(self):
        """Applica il tema corrente al dialog"""
        if not hasattr(self, 'theme_manager'):
            return
        
        # Determina se è un tema chiaro o scuro
        current_theme = getattr(self.theme_manager, 'current_theme', 'dark.css')
        is_light_theme = 'light' in current_theme.lower()
        
        if is_light_theme:
            # ✅ TEMA CHIARO: Usa colori appropriati (sfondo chiaro, testo scuro)
            self.setStyleSheet("""
                QDialog {
                    background-color: #f5f5f5;
                    color: #222222;
                }
                QLabel, QRadioButton, QCheckBox {
                    background-color: #f5f5f5;
                    color: #222222;
                }
                               
            """)
        self.title_label.setStyleSheet("font-size: 21pt; font-weight: bold;")  # Applica dimensione titolo dal tema (sovrascrive quella impostata in __init__)
        
    # === PUBLIC INTERFACE ===
    
    def get_export_parameters(self):
        """Restituisce tutti i parametri per l'export"""
        params = {
            'start_time': self.start_spin.value(),
            'end_time': self.end_spin.value(),
            'output_path': getattr(self, '_full_output_path', None) if self.new_file_radio.isChecked() else None,
            'overwrite_mode': self.overwrite_radio.isChecked()
        }
        
        # Type-specific options
        if self.file_type == 'audio':
            params['include_clicks'] = self.include_clicks_check.isChecked()
        elif self.file_type == 'voltage':
            params['include_analyses'] = self.include_analyses_check.isChecked()
        
        return params
