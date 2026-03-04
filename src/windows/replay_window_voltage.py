import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QIcon, QAction
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSizePolicy, QProgressDialog, QMessageBox
import sys
import os
import struct
from PySide6.QtCore import QTimer

from plotting.plot_manager import BasePlotWidget
from core.replay_base_window import ReplayBaseWindow
from core.voltage_trim_export import VoltageTrimExporter

class ReplayVoltageWindow(ReplayBaseWindow):
    def __init__(self, file_path):
        super().__init__()

        # Gestione memoria
        self._chunk_size = 100000  # Punti per blocco
        self._data_chunks = []     # Memorizza i dati a blocchi
        self._loaded_chunks = set() # Tiene traccia dei blocchi caricati
        self._current_chunk = 0     # Blocco corrente
        self.current_position = 0    # Posizione corrente nel file (in millisecondi)
        self.playback_speed = 1.0   # Velocità di riproduzione
        self.playback_timer = None  # Timer per la riproduzione
        self.total_duration_ms = 0  # Durata totale in millisecondi
        self.sampling_rate = 1.0    # Frequenza di campionamento (Hz)

        # Specific ui
        self._setup_specific_ui()

        # Menubar
        self.setup_menubar()

        # tema
        self._load_saved_settings()

        # Toolbar
        self.setup_toolbar()

        # Carica il file
        self.load_file(file_path=file_path)

        # Inizializza timer di riproduzione
        self.last_update_time = 0  # Timestamp dell'ultimo aggiornamento
        self.accumulated_time = 0  # Tempo accumulato per la riproduzione
        
        # Rimuovi o modifica l'inizializzazione del timer
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._update_playback)
        self.playback_timer.setTimerType(QtCore.Qt.PreciseTimer)  # Timer più preciso

        self.clear_history()  # Inizia in stato "fermo"

        # mostra a tutto schermo mantenendo le grafiche
        self.showMaximized()


    def start_playback(self):
        """Avvia la riproduzione - sovrascrive il metodo della classe base"""
        super().start_playback()
        self._reset_playback_timing()
        self.last_update_time = QtCore.QTime.currentTime().msecsSinceStartOfDay()
        
        # Usa un timer fisso invece di calcolare l'intervallo
        self.playback_timer.setInterval(16)  # ~60 FPS
        if not self.playback_timer.isActive():
            self.playback_timer.start()
        self.actionMath.setEnabled(False)  # Disabilita operazioni matematiche durante il playback
        self.actionMathDialog.setEnabled(False)  # Disabilita operazioni matematiche durante il playback
        # Se c'è una regione di selezione, rimuovila
        if hasattr(self, 'region'):
            self.voltage_plot.plot_widget.removeItem(self.region)
            self.actionMath.setToolTip("Select a region in the plot to perform mathematical analysis")
            del self.region # Rimuovi l'attributo
        print("Riproduzione avviata")

    def pause_playback(self):
        """Mette in pausa la riproduzione - sovrascrive il metodo della classe base"""
        super().pause_playback()  # Chiama il metodo della classe base
        if self.playback_timer.isActive():
            self.playback_timer.stop()
        self.actionMath.setEnabled(True)  # Abilita operazioni matematiche in pausa
        print("Riproduzione in pausa")

    def _on_playback_speed_changed(self, speed=1):
        """Aggiorna la velocità di riproduzione"""
        self.playback_speed = speed
        print(f"Velocità cambiata a: {speed}x")
        self._reset_playback_timing()  # Resetta il timing quando cambia la velocità
    
    def _reset_playback_timing(self):
        """Resetta il timing della riproduzione"""
        self.last_update_time = 0
        self.accumulated_time = 0

    def _update_timer_interval(self):
        """Aggiorna l'intervallo del timer in base alla velocità di riproduzione"""
        if self.sampling_rate > 0:
            # Calcola l'intervallo in base alla velocità e al sampling rate
            base_interval = 1000 / self.sampling_rate  # ms per campione a velocità 1x
            adjusted_interval = max(1, int(base_interval / self.playback_speed))  # Assicura almeno 1ms
            self.playback_timer.setInterval(adjusted_interval)
            print(f"Timer interval: {adjusted_interval}ms, Sampling rate: {self.sampling_rate}Hz, Speed: {self.playback_speed}x")

    def _update_playback(self):
        """Aggiorna la posizione durante la riproduzione usando timing reale"""
        if self.current_position >= self.total_duration_ms:
            self.pause_playback()
            print("Riproduzione completata")
            return

        try:
            # Calcola il tempo trascorso dall'ultimo update
            current_time = QtCore.QTime.currentTime().msecsSinceStartOfDay()
            if self.last_update_time == 0:
                elapsed_ms = 0
            else:
                elapsed_ms = current_time - self.last_update_time
            
            self.last_update_time = current_time
            
            # Applica la velocità di riproduzione al tempo trascorso
            adjusted_elapsed = elapsed_ms * self.playback_speed
            
            # Aggiorna la posizione
            new_position = self.current_position + adjusted_elapsed
            self.current_position = min(new_position, self.total_duration_ms)
            
            # Aggiorna lo slider e la visualizzazione
            self.time_slider.setValue(int(self.current_position))
            self.update_display()
            
        except Exception as e:
            print(f"Errore in _update_playback: {str(e)}")
            self.pause_playback()


    def update_display(self):
        """Aggiorna la visualizzazione basata sulla posizione corrente"""
        try:
            if not self._data_chunks or self.current_position <= 0:
                return
                
            # Converti millisecondi in secondi
            current_time_sec = self.current_position / 1000.0
            
            # Calcola l'indice del campione corrente
            current_sample = int(current_time_sec * self.sampling_rate)
            total_samples = self.metadata['data_points']
            
            # Trova il blocco corrispondente
            points_per_chunk = self._chunk_size
            chunk_idx = current_sample // points_per_chunk
            
            if chunk_idx < len(self._data_chunks) and self._data_chunks[chunk_idx] is not None:
                x_chunk, y_chunk = self._data_chunks[chunk_idx]
                
                # Calcola la posizione nel blocco
                pos_in_chunk = current_sample % points_per_chunk
                pos_in_chunk = min(pos_in_chunk, len(x_chunk) - 1)
                
                # Mostra tutti i dati fino alla posizione corrente
                x_to_show = x_chunk[:pos_in_chunk+1]
                y_to_show = y_chunk[:pos_in_chunk+1]
                
                # Filtra valori NaN
                valid = ~np.isnan(x_to_show) & ~np.isnan(y_to_show)
                if np.any(valid):
                    self.voltage_curve.setData(x_to_show[valid], y_to_show[valid])
                    
                    # Aggiorna finestra temporale
                    if self.time_window_enabled and len(x_to_show[valid]) > 0:
                        current_display_time = x_to_show[valid][-1]
                        window_size = 15
                        xmin = max(0, current_display_time - window_size/2)
                        xmax = xmin + window_size
                        self.voltage_plot.set_x_range(xmin, xmax)
            
            self._update_time_labels()
            
        except Exception as e:
            print(f"Errore in update_display: {str(e)}")

    def _update_time_labels(self):
        """Aggiorna le etichette del tempo corrente con larghezza fissa"""
        current_time_sec = self.current_position / 1000.0
        total_time_sec = self.total_duration_ms / 1000.0
        
        # ✅ USA IL NUOVO WIDGET invece di setText()
        if hasattr(self, 'current_time_input'):
            self.current_time_input.set_time(current_time_sec, total_time_sec)
        
        # Tooltip con info dettagliate
        if hasattr(self, 'velocity'):
            self.velocity.setToolTip(f"Sampling rate: {self.sampling_rate}Hz\nSpeed: {self.playback_speed}x")


    def load_file(self, file_path):
        """Carica il file nel thread principale mostrando un popup di caricamento"""
        try:
            #progress widget
            progress = self.get_progress_widget("Loading Voltage File...")
            progress.setValue(0)
            progress.show()

            self.file_path = file_path

            # Leggi solo l'header e chiudi subito il file
            with open(file_path, 'rb') as f:
                # STEP 1: Load header
                progress.setValue(10)
                progress.setLabelText("Loading header...")
                header = f.read(128)
            self.metadata = self._parse_header(header)

            # Imposta la frequenza di campionamento
            self.sampling_rate = self.metadata.get('sampling_rate', 1.0)
            print(f"Sampling rate: {self.sampling_rate}Hz")

            self.setWindowTitle(f"Replay Voltage - {os.path.basename(file_path)}")

            # Aggiorna impostazioni in base a header
            if self.metadata['amplified']:
                self.voltage_plot.set_y_range(-1.7, 1.7)
                self.voltage_plot.set_y_limits(-1.7, 1.7)
                #cambia titolo dell'asse y
                self.voltage_plot.set_y_label('Voltage', 'V')
                self.amplified_data = True
            else:
                self.voltage_plot.set_y_range(-130, 130)
                self.voltage_plot.set_y_limits(-130, 130)
                #cambia titolo dell'asse y
                self.voltage_plot.set_y_label('Voltage', 'mV')
                self.amplified_data = False

            # Imposta la durata totale in millisecondi
            self.total_duration_ms = int(self.metadata['duration'] * 1000)
            self.time_slider.setRange(0, self.total_duration_ms)

            # Calcola chunk info
            total_points = self.metadata['data_points']
            chunk_size = self._chunk_size
            num_chunks = (total_points + chunk_size - 1) // chunk_size


            with open(file_path, 'rb') as f:
                for chunk_idx in range(num_chunks):
                    if progress.wasCanceled():
                        break
                    chunk_points = min(chunk_size, total_points - chunk_idx * chunk_size)
                    start = 128 + chunk_idx * chunk_size * 8
                    f.seek(start)
                    raw_data = np.fromfile(f, dtype=np.float32, count=chunk_points*2)
                    if raw_data.size != chunk_points*2:
                        print(f"⚠️ Dati incompleti nel blocco {chunk_idx}, atteso {chunk_points*2} punti, ottenuti {raw_data.size}")
                        self.show_error_dialog("Error", f"Incompleted data: {chunk_idx}.")
                        break  # dati incompleti
                    data = raw_data.reshape(-1, 2)
                    x = data[:, 0].astype(np.float64)
                    y = data[:, 1].astype(np.float64)
                    x[np.isinf(x)] = np.nan
                    y[np.isinf(y)] = np.nan
                    self._process_data_chunk(chunk_idx, x, y, progress)
                    progress.setValue(10 + int(80 * (chunk_idx + 1) / num_chunks))

            # Dopo aver caricato i dati, aggiorna la durata reale
            if self._data_chunks and any(c is not None for c in self._data_chunks):
                x_all = np.concatenate([c[0] for c in self._data_chunks if c is not None])
                y_all = np.concatenate([c[1] for c in self._data_chunks if c is not None])
                valid = ~np.isnan(x_all) & ~np.isnan(y_all)
                if np.any(valid):
                    # Usa l'ultimo timestamp per la durata reale
                    real_duration = x_all[valid][-1]
                    self.total_duration_ms = int(real_duration * 1000)
                    self.time_slider.setRange(0, self.total_duration_ms)
                    print(f"Durata reale aggiornata: {self.total_duration_ms/1000:.2f} s")

            print("File caricato con successo")
            progress.close()
            return True
        except Exception as e:
            print(f"Errore nel caricamento del file: {str(e)}")
            if 'progress' in locals():
                progress.close()
            QMessageBox.critical(self, "Errore", f"Errore nel caricamento del file:\n{str(e)}")
            return False
            

    def _parse_header(self, header_bytes):
        """Parser più robusto per l'header con validazione"""
        if len(header_bytes) < 128:
            raise ValueError("Header troppo corto (min 128 byte richiesti)")

        try:
            # Verifica il magic number
            magic = header_bytes[:9]
            if magic != b'PLANTVOLT':
                raise ValueError("Formato file non valido (magic number mancante)")

            # Parsing con controllo degli errori
            version = struct.unpack('<f', header_bytes[9:13])[0]
            if version not in [1.0, 2.0]:
                raise ValueError(f"Versione non supportata: {version}")

            experiment_type = header_bytes[13:33].decode('ascii', errors='ignore').strip('\x00')

            # Leggi tutti i campi con validazione
            metadata = {
                'version': version,
                'experiment_type': experiment_type if experiment_type else 'Unknown',
                'sampling_rate': max(0, struct.unpack('<f', header_bytes[33:37])[0]),  # Evita valori negativi
                'duration': max(0, struct.unpack('<f', header_bytes[37:41])[0]),
                'amplified': bool(struct.unpack('<?', header_bytes[41:42])[0]),
                'data_points': struct.unpack('<I', header_bytes[58:62])[0],
                'acquisition_count': struct.unpack('<I', header_bytes[62:66])[0]
            }

            # Gestione timestamp per versione 2.0
            if version >= 2.0:
                metadata.update({
                    'start_time': max(0, struct.unpack('<d', header_bytes[41:49])[0]),
                    'end_time': max(0, struct.unpack('<d', header_bytes[49:57])[0])
                })
            else:
                metadata.update({
                    'start_time': 0.0,
                    'end_time': metadata['duration']
                })

            # Validazione aggiuntiva
            if metadata['sampling_rate'] <= 0 or metadata['sampling_rate'] > 100000:
                raise ValueError(f"Frequenza di campionamento non valida: {metadata['sampling_rate']}")

            return metadata

        except struct.error as e:
            raise ValueError(f"Errore nel parsing dell'header: {str(e)}")
        except Exception as e:
            raise ValueError(f"Errore sconosciuto nel parsing: {str(e)}")

    def _process_data_chunk(self, chunk_idx, x, y, progress_dialog=None):
        """Processa un blocco di dati caricato e aggiorna il progresso se richiesto"""
        try:
            # Memorizza il blocco
            if chunk_idx >= len(self._data_chunks):
                self._data_chunks.extend([None] * (chunk_idx - len(self._data_chunks) + 1))
            self._data_chunks[chunk_idx] = (x, y)
            self._loaded_chunks.add(chunk_idx)
            
            # Aggiorna progresso
            if progress_dialog is not None:
                progress_dialog.setValue(chunk_idx + 1)
                progress_dialog.setLabelText(f"Caricamento dati... ({chunk_idx + 1}/{len(self._data_chunks)})")
                QtWidgets.QApplication.processEvents()  # Forza aggiornamento UI

            # Se è il primo blocco, mostra subito i dati
            if chunk_idx == 0:
                self._update_plot_from_chunks()

            if np.any(np.isnan(x)) or np.any(np.isnan(y)):
                print(f"Chunk {chunk_idx}: dati non validi trovati")
            if x.shape != y.shape:
                print(f"Chunk {chunk_idx}: shape mismatch {x.shape} vs {y.shape}")

        except Exception as e:
            print(f"Errore nell'elaborazione blocco: {str(e)}")


    def _update_plot_from_chunks(self):
        """Aggiorna il plot con i dati caricati finora"""
        try:
            if not self._data_chunks or all(c is None for c in self._data_chunks):
                return
                
            # Unisci i blocchi caricati
            x_all = np.concatenate([c[0] for c in self._data_chunks if c is not None])
            y_all = np.concatenate([c[1] for c in self._data_chunks if c is not None])

            print(f"Plotting {len(x_all)} points from {len(self._loaded_chunks)} chunks")
            
            # Filtra valori non validi
            valid = ~np.isnan(x_all) & ~np.isnan(y_all)
            if np.any(valid):
                self.voltage_curve.setData(x_all[valid], y_all[valid])
                
                # Aggiorna slider e durata reale se tutti i chunk sono caricati
                if len(self._loaded_chunks) == len(self._data_chunks):
                    duration = x_all[valid][-1]
                    self.total_duration_ms = int(duration * 1000)
                    self.time_slider.setRange(0, self.total_duration_ms)
                    print(f"Durata reale aggiornata: {self.total_duration_ms/1000:.2f} s")
        except Exception as e:
            print(f"Errore nell'aggiornamento plot: {str(e)}")







    def clear_history(self):
        """Pulisce la cronologia della riproduzione e resetta alla visualizzazione completa"""
        print("🔄 Cronologia della riproduzione cancellata")
        self.pause_playback()
        self.current_position = 0
        self.time_slider.setValue(0)

        # Mostra tutti i dati caricati (modalità stop)
        if self._data_chunks and any(c is not None for c in self._data_chunks):
            x_all = np.concatenate([c[0] for c in self._data_chunks if c is not None])
            y_all = np.concatenate([c[1] for c in self._data_chunks if c is not None])
            valid = ~np.isnan(x_all) & ~np.isnan(y_all)
            if np.any(valid):
                self.voltage_curve.setData(x_all[valid], y_all[valid])
        #reimposta vista
        if self.metadata['amplified']:
            self.voltage_plot.set_y_range(-1.7, 1.7)
            self.voltage_plot.set_y_limits(-1.7, 1.7)
            #cambia titolo dell'asse y
            self.voltage_plot.set_y_label('Voltage', 'V')
        else:
            self.voltage_plot.set_y_range(-130, 130)
            self.voltage_plot.set_y_limits(-130, 130)
            #cambia titolo dell'asse y
            self.voltage_plot.set_y_label('Voltage', 'mV')
        self.voltage_plot.set_x_range(0, 15)
        self._update_time_labels()
        

    def _setup_specific_ui(self):
        # Widget centrale
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principale
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Layout per i grafici
        graph_layout = QtWidgets.QHBoxLayout()
        graph_layout.setContentsMargins(10, 20, 10, 0)
        self.voltage_plot = BasePlotWidget(
            x_label="Time",
            y_label="Voltage",
            x_range=(0, 15),
            y_range=(-1.7, 1.7),
            x_min=0, x_max=None, y_min=-1.7, y_max=1.7,
            unit_x="s", unit_y="V",
            parent=self
        )
        self.voltage_curve = self.voltage_plot.plot_widget.plot(name="Voltage Data")
        self.voltage_plot.plot_widget.showGrid(x=True, y=True)
        graph_layout.addWidget(self.voltage_plot)
        layout.addLayout(graph_layout)

    def _execute_trim_export(self, params):
        """
        Override del metodo base per eseguire l'export trimmed di file voltage.
        Utilizza VoltageTrimExporter per la logica di export.
        """
        exporter = VoltageTrimExporter(self)
        return exporter.execute_trim_export(params)

