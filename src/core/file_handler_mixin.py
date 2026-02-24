import os
import numpy as np
from PySide6.QtWidgets import QProgressDialog, QMessageBox
from PySide6.QtCore import Qt, QThread


class FileHandlerMixin:
    """
    Mixin che fornisce metodi comuni per:
    - Apertura file .pvolt/.paudio
    - Rilevamento tipo file
    - Gestione progress dialog
    - Callback di caricamento (progress, finished, error, cancel)
    """
    
    # Variabile di classe per memorizzare l'ultima directory usata
    _last_used_directory = None
    
    def open_file_action(self, file_path=None):
        """
        Apre un file di analisi PlantLeaf (.pvolt o .paudio).
        Può essere chiamato con un path specifico o mostrare un dialog.
        """
        from PySide6.QtWidgets import QFileDialog
        from saving.audio_load_progress import AudioLoadWorker
        
        # Se non è stato passato un file path, mostra il dialog
        if not file_path:
            # Gestisci eventuali dati non salvati (se la classe che usa il mixin lo supporta)
            if hasattr(self, 'handle_unsaved_data') and not self.handle_unsaved_data():
                if getattr(self, '_pending_close_event', False):
                    self.opening_new_file = True
                print("❌ Apertura file annullata dall'utente.")
                return
            
            # Se non è replay, pulisci esperimento (se supportato)
            if hasattr(self, 'clear_experiment_action') and not getattr(self, 'replay_requested', False):
                self.clear_experiment_action()
            
            # Determina la directory di partenza
            if FileHandlerMixin._last_used_directory and os.path.exists(FileHandlerMixin._last_used_directory):
                start_directory = FileHandlerMixin._last_used_directory
            else:
                start_directory = os.path.expanduser("~")  # Home dell'utente
            
            print(f"📁 Opening file dialog from: {start_directory}")
            file_dialog = QFileDialog()
            self.file_path, _ = file_dialog.getOpenFileName(
                self,
                "Open PlantLeaf Analysis File",
                start_directory,  # ← Qui impostiamo la directory di partenza
                "PlantLeaf Files (*.pvolt *.paudio);;PlantLeaf Voltage (*.pvolt);;PlantLeaf Audio (*.paudio);;All Files (*)"
            )
            
            # Memorizza la directory per la prossima volta
            if self.file_path:
                FileHandlerMixin._last_used_directory = os.path.dirname(self.file_path)
        
        else:
            self.file_path = file_path
        
        # Procedi solo se c'è un file selezionato
        if not self.file_path:
            print("❌ No file selected")
            return
        
        print(f"📂 Selected file: {self.file_path}")
        
        try:
            self.is_closing = True
            
            # Salvataggio automatico se necessario (BaseWindow only)
            if hasattr(self, 'save_file_action') and not getattr(self, 'opening_new_file', False):
                if not getattr(self, 'replay_requested', False):
                    if getattr(self, '_last_saved_file', None):
                        self.save_file_action(ask_filename=False)
            
            # Rileva il tipo di file
            analysis_type = self.detect_file_type(self.file_path)
            
            # Reset flag replay
            if hasattr(self, 'replay_requested'):
                self.replay_requested = False
            
            # === AUDIO REPLAY ===
            if analysis_type == "audio":
                print("🔊 Opening audio analysis file (threaded)...")
                self.progress_dialog = self.get_progress_widget("Loading Audio File...")
                self.progress_dialog.setValue(0)
                self.progress_dialog.show()
                
                # Setup worker thread
                self.load_thread = QThread()
                self.load_worker = AudioLoadWorker(self.file_path)
                self.load_worker.moveToThread(self.load_thread)
                
                # Connetti segnali
                self.load_worker.progress.connect(self._on_progress)
                self.load_worker.finished.connect(self._on_finished)
                self.load_worker.error.connect(self._on_error)
                self.progress_dialog.canceled.connect(self._on_cancel)
                
                self.load_thread.started.connect(self.load_worker.run)
                self.load_thread.start()
                return
            
            # === VOLTAGE REPLAY ===
            elif analysis_type == "voltage":
                from windows.replay_window_voltage import ReplayVoltageWindow
                print("⚡ Opening voltage analysis file...")
                self.replay_window = ReplayVoltageWindow(self.file_path)
                self.replay_window.show()
                self.close()
            
            elif analysis_type == "unknown":
                print("❓ Unknown file type")
                self.show_error_dialog("Unknown File Type", "The selected file type is not recognized.")
                self.is_closing = False
                if hasattr(self, 'opening_new_file'):
                    self.opening_new_file = False
        
        except Exception as e:
            print(f"❌ Error opening file: {e}")
            self.show_error_dialog("File Error", f"Could not open file:\n{str(e)}")
            self.is_closing = False
            if hasattr(self, 'opening_new_file'):
                self.opening_new_file = False


    def detect_file_type(self, file_path):
        """Rileva il tipo di file PlantLeaf tramite magic number ed estensione"""
        try:
            file_extension = file_path.lower().split('.')[-1]
            print(f"🔍 File extension: .{file_extension}")
            
            with open(file_path, 'rb') as f:
                magic = f.read(9)
                
                if magic == b'PLANTVOLT':
                    if file_extension == 'pvolt':
                        print("🔋 Detected PlantLeaf Voltage file (.pvolt)")
                        return "voltage"
                    else:
                        print(f"⚠️ Voltage magic number but extension .{file_extension} - assuming .pvolt")
                        return "voltage"
                
                elif magic.startswith(b'PLANTAUD'):
                    if file_extension == 'paudio':
                        print("🔊 Detected PlantLeaf Audio file (.paudio)")
                        return "audio"
                    else:
                        print(f"⚠️ Audio magic number but extension .{file_extension} - assuming .paudio")
                        return "audio"
                
                else:
                    print(f"❌ Unrecognized magic number: {magic}")
                    
                    # Fallback su estensione
                    if file_extension in ['pvolt', 'plantvolt']:
                        print("🔄 Fallback: assuming voltage from extension")
                        return "voltage"
                    elif file_extension in ['paudio', 'plantaudio']:
                        print("🔄 Fallback: assuming audio from extension")
                        return "audio"
                    else:
                        return "unknown"
        
        except Exception as e:
            print(f"❌ Error detecting file type: {e}")
            return "unknown"


    def _on_progress(self, percent):
        """Callback per aggiornamento progress bar"""
        self.progress_dialog.setValue(percent)
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()


    def _on_finished(self, data):
        """
        Callback per completamento caricamento audio.
        DEVE essere sovrascritto dalle classi che usano il mixin se necessario.
        """
        print("✅ File loading finished (base implementation)")
        self.progress_dialog.close()
        self.load_thread.quit()
        self.load_thread.wait()
        self.is_closing = False
        
        # Crea finestra replay audio
        from windows.replay_window_audio import ReplayWindowAudio
        replay_window = ReplayWindowAudio(self.file_path)
        dm = replay_window.data_manager
        
        # Popola data manager
        dm.header_info = data['header_info']
        dm.fft_data = data['fft_data']
        dm.phase_data = data.get('phase_data', [])
        dm.frequency_axis = np.array(data['frequency_axis'])
        dm.total_frames = data['total_frames']
        dm.frame_duration_ms = data['frame_duration_ms']
        dm.total_duration_sec = data['total_duration_sec']
        dm.click_events = data['click_events']
        dm.overview_x = np.array(data['overview_x'])
        dm.overview_y = np.array(data['overview_y'])
        dm.overview_loaded = True
        dm.streaming_x = np.array(data['streaming_x'])
        dm.streaming_y = np.array(data['streaming_y'])
        dm.streaming_start_time = data['streaming_start_time']
        dm.streaming_end_time = data['streaming_end_time']
        
        # ✅ PRECALCOLA LE MEDIE FFT
        print("🔄 Precalcolo medie FFT...")
        dm.precompute_fft_means()
        
        # Setup UI
        replay_window._setup_metadata()
        replay_window._setup_ui_with_data()
        replay_window.setWindowTitle(f"Replay Audio - {os.path.basename(self.file_path)}")
        replay_window.show()
        
        print("✅ Audio analysis file opened successfully")
        self.file_path = None
        self.close()


    def _on_error(self, msg):
        """Callback per errore durante caricamento"""
        print(f"❌ Loading error: {msg}")
        self.progress_dialog.close()
        self.load_thread.quit()
        self.load_thread.wait()
        self.show_error_dialog("Error", f"An error occurred while loading the file:\n{msg}")
        self.is_closing = False


    def _on_cancel(self):
        """Callback per cancellazione caricamento"""
        print("🚫 Loading cancelled")
        self.load_worker.cancel_load()
        self.progress_dialog.close()
        self.load_thread.quit()
        self.load_thread.wait()
        self.is_closing = False


    def get_progress_widget(self, title: str):
        """Crea e configura un progress dialog con stile tematizzato"""
        progress = QProgressDialog(title, "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        
        # Centra il dialog
        screen = self.screen()
        screen_geometry = screen.geometry()
        dialog_size = progress.sizeHint()
        x = (screen_geometry.width() - dialog_size.width()) // 2
        y = (screen_geometry.height() - dialog_size.height()) // 2
        progress.move(x, y)
        progress.setFixedSize(400, 120)
        
        # Applica tema (se theme_manager è disponibile)
        if hasattr(self, 'theme_manager'):
            theme_colors = self.theme_manager.get_theme_specific_colors()
            progress_bg = theme_colors.get('bg', '#2b2b2b')
            progress_text = theme_colors.get('text', '#ffffff')
            button_bg = theme_colors.get('inactive_bg', '#444')
            button_text = theme_colors.get('inactive_text', '#fff')
            
            progress.setStyleSheet(f"""
                QProgressDialog {{
                    background-color: {progress_bg};
                    color: {progress_text};
                }}
                QProgressBar {{
                    background-color: {button_bg};
                    color: #000000;
                    border: 1px solid #555;
                    text-align: center;
                    font-weight: bold;
                }}
                QProgressBar::chunk {{
                    background-color: #4CAF50;
                }}
                QPushButton {{
                    background-color: {button_bg};
                    color: {button_text};
                    border: 1px solid #555;
                    padding: 5px 15px;
                    border-radius: 3px;
                }}
                QPushButton:hover {{
                    background-color: #555;
                }}
                QLabel {{
                    color: {progress_text};
                }}
            """)
        
        return progress


    def show_error_dialog(self, title, message):
        """Mostra un dialog di errore"""
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setIcon(QMessageBox.Critical)
        msg.addButton("OK", QMessageBox.AcceptRole)
        msg.exec()