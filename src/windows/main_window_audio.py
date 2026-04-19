"""
Finestra principale per il monitoraggio Audio
"""

import os
import tempfile
import struct

from PySide6.QtWidgets import QSplitter, QMessageBox, QFileDialog, QProgressDialog, QApplication
from PySide6.QtCore import Signal, QTimer, Qt, QThread

from core import BaseWindow
from core.wake_lock_manager import WakeLockManager
from .ui.ui_MainWindowAudio import Ui_MainWindowAudio
from core.special_component import replace_widget
from components.start_stop_button import StartStopButton
from components.data_table import DataTable
from plotting.plot_manager import BasePlotWidget
from components.sampling_settings import AudioSamplingSettingsPopup
from serial_communication.audio_reader import AudioSerialWorker
from saving.audio_save_worker import AudioSaveWorker

import numpy as np
import time
from datetime import datetime 


class MainWindowAudio(BaseWindow, Ui_MainWindowAudio):
    """Finestra principale per il monitoraggio audio delle piante"""
    clicks_detector_toggled = Signal(bool)
    filter_button_toggled = Signal(bool)

    def __init__(self, parent=None):
        Ui_MainWindowAudio.__init__(self)
        BaseWindow.__init__(self, parent)
        self.setupUi(self)

        # Inizializza array per i dati (salvataggio)
        #data x alla fine del __init__
        self.data_y_buffer = np.array([]) #magnitudini
        self.data_phase_buffer = np.array([]) #fasi
        #INIZIALIZZA ARRAY PER VALORI DA PLOTTARE
        #per data x vedi fine del __init__
        self.data_y_plot = np.array([], dtype=float)
        self.plot_needs_update = False  # Flag per aggiornamento plot


        # Inizializza timer con tempo assoluto
        self.total_elapsed_time = 0
        self.chrono_start_time = 0


        # Sostituzione pulsanti custom START/STOP
        customFFT = StartStopButton(self.theme_manager, parent=self)
        replace_widget(self, "FFTStartStopButton", customFFT)
        self.FFTStartStopButton = customFFT

        self.FFTStartStopButton.started.connect(self.on_start) 
        self.FFTStartStopButton.stopped.connect(self.on_stop)

        self.actionStart.setEnabled(False)

        #setup splitter
        self.setup_splitter()

        self._setup_ui()

        # Sostituzione tabelle
        customTableFFT = DataTable(self.theme_manager, parent=self)

        replace_widget(self, "FFTClicksDetectedTableWidget", customTableFFT)

        self.FFTClicksDetectedTableWidget = customTableFFT

        self._setup_table_fonts()  # Imposta i font per le tabelle

        # Sostituzione plotWidgets
        #FFT
        custom_plot_fft = BasePlotWidget(
            x_label="Frequency",
            y_label="Amplitude",
            x_range=(20000, 80000),
            y_range=(0, 0.035),
            x_min=19000, x_max=81000, y_min=0, y_max=1.7,
            unit_x="Hz", unit_y="V",
            parent=self
        )
        replace_widget(self, "FFTPotWidget", custom_plot_fft)
        self.plot_widget_fft = custom_plot_fft

        # Crea la curva principale con la penna desiderata
        self.plot_widget_fft.plot = self.plot_widget_fft.plot_widget.plot(name="Amplitude Data")

        self.setWindowTitle("Audio Monitor")

        self.layout_manager.center_window_on_screen(self)

        self.setup_toolbar_actions()
        self.setup_menubar_actions()

        self.theme_manager.apply_theme_to_plot(self.plot_widget_fft.plot_widget, self.plot_widget_fft.plot)

        #riapplica modifiche tema, font, layout
        self.layout_manager.adjust_window_size_for_content(self)

        self.type_of_experiment = "Test"  # Default, può essere modificato in sampling_settings

        #richiamo il sistemafont
        self._load_saved_settings()

        self.setStatusBar(None)  # Disabilita la status bar predefinita

        # Imposta l'azione di avvio dell'esperimento come disattivata e anche i pulsanti di start/stop
        self.FFTStartStopButton.setEnabled(False)
        self.FFTThresholdSpinBox.setEnabled(True)
        self.FFTClicksDetectorButton.setEnabled(True)
        
        
        #### CREA I VALORI FISSI DELL'ASSE X (FREQUENZA) ####
        # Calcola il range corretto basato sul firmware
        self.fs = 200000  # 200kHz
        self.fft_size = 512
        self.freq_min = 20000  # 20kHz
        self.freq_max = 80000  # 80kHz
        
        # Calcola bin frequency
        bin_freq = self.fs / self.fft_size
        bin_start = int(self.freq_min / bin_freq)
        bin_end = int(self.freq_max / bin_freq)
        num_bins = bin_end - bin_start + 1
        
        # Crea array frequenze corretto
        self.data_x = np.linspace(self.freq_min, self.freq_max, num_bins)

        # Variabili per click detection OTTIMIZZATE
        self.click_active = False
        self.click_start_time = 0
        self.click_peak_frequency = 0
        self.click_peak_amplitude = 0
        self.click_fft_count = 0
        self.last_fft_time = 0
        self._last_table_update = 0  # Throttling tabella
        
        # Calcola intervallo FFT in microsecondi per precisione temporale
        self.fft_interval_us = (self.fft_size / self.fs) * 1_000_000  # microseconds

        self.create_initial_threshold()
        self._last_user_threshold_change = 0 # NUOVO: Timestamp dell'ultima modifica utente

        ## variabili per salvataggio
        self._last_temp_file = None
        self._last_saved_file = None
        self._acquisition_count = 0  #non credo serva a qualcosa ma lasciamolo per ora

        self.save_thread = None
        self.save_worker = None
        self.definetly_saved = False
        self._pending_close_event = False

        self.is_acquiring = False
        self._wake_lock = WakeLockManager()

        # mostra a tutto schermo mantenendo le grafiche
        self.showMaximized()




    #### GESTIONE START E STOP #####
    
    def on_start(self):
        # ✅ CONTROLLO THREAD PRECEDENTE ANCORA ATTIVO
        if (hasattr(self, 'serial_worker') and 
            self.serial_worker is not None and 
            self.serial_worker.isRunning()):
            print("⚠️ Thread precedente ancora attivo, attendere...")
            return
        
        # ✅ CONTROLLO SICUREZZA COMPLETO
        if (not hasattr(self, 'serial_worker') or 
            self.serial_worker is None or 
            not getattr(self.serial_worker, 'is_connected', False)):            
            print("❌ Impossibile avviare: porta seriale non connessa")
            
            # ✅ RIABILITA azione SerialPort
            if hasattr(self, 'actionSerialPort'):
                self.actionSerialPort.setEnabled(True)
                
            return  # ✅ ESCI SUBITO
        
        # ✅ Solo se tutto OK, procedi
        self.is_acquiring = True
        self._wake_lock.acquire()  # ☀️ Previeni sleep durante acquisizione
        self.start_chronometer()
        self.serial_worker.start(self.threshold_value)  # PASSA LA SOGLIA CORRENTE AL METODO START

        # Disabilita azioni ma verifica se dopo serial_worker.start() è andcora tutto attivo:
        try:
            if self.serial_worker.is_connected:
                self.actionClear.setEnabled(False)
                self.actionSamplingSettings.setEnabled(False)
                self.actionSerialPort.setEnabled(False)
                self.actionSave.setEnabled(False)
                self.actionOpenFile.setEnabled(False)
                self.actionNewFile.setEnabled(False)
        except Exception as e:
            print(e)
            return

        # Inizializza variabile ora di inizio
        if not hasattr(self, 'start_datetime'):
            self.start_datetime = datetime.now().timestamp()


    def on_stop(self):
        if not self.isVisible():  # Se la finestra sta chiudendosi, non salvare
            return
        self.is_acquiring = False
        self._wake_lock.release()  # 🌙 Rilascia wake lock
        
        # Chiama il metodo sicuro centralizzato in BaseWindow
        self._safe_stop_serial_worker()
        
        # ✅ CONTROLLO SICUREZZA per riabilitazione porta
        if (hasattr(self, 'serial_worker') and 
            self.serial_worker is not None and 
            not getattr(self.serial_worker, 'is_connected', False)):
            if hasattr(self, "actionSerialPort"):
                self.actionSerialPort.setEnabled(True)
                self.set_buttons_enabled(False)
        
        # Inizializza variabile ora di fine (viene sovrascritta ogni volta)
        self.end_datetime = datetime.now().timestamp()
        self.stop_chronometer()

        if not self.definetly_saved:
            self.actionSave.setEnabled(True)
        self.actionClear.setEnabled(True)
        self.actionOpenFile.setEnabled(True)
        self.actionNewFile.setEnabled(True)

        print("🛑 Arrestato Monitoraggio Audio")





    #### GESTIONE DATI OTTIMIZZATA #####

    def on_new_fft_data(self, amplitudes, phases, max_amplitude, peak_bin, above_threshold, current_threshold):
        """Triggerato ad ogni nuova FFT ricevuta dal serial worker"""
        if not getattr(self, "is_acquiring", False):
            return

        # ACCUMULO DATI per salvataggio
        # (magnitudes + fasi)
        self.data_y_buffer = np.append(self.data_y_buffer, amplitudes)
        self.data_phase_buffer = np.append(self.data_phase_buffer, phases)

        # AGGIORNA DATI PLOT (sempre l'ultima FFT ricevuta)
        self.data_y_plot = amplitudes.copy()
        self.plot_needs_update = True

        # Click detection ULTRA-VELOCE (usa dati pre-calcolati) 
        if self.clicksDetectionStatus:
            self.check_for_clicks_optimized(max_amplitude, peak_bin, above_threshold)

        # ✅ AUTO-SAVE ogni N campioni (identico al voltage: 1000 campioni)
        if len(self.data_y_buffer) >= 15500:  # ~100 FFT * 155 campioni per FFT
            #print(f"💾 Auto-save triggered: {len(self.data_y_buffer)} campioni")
            self.save_fft_data()


    def check_for_clicks_optimized(self, max_amplitude, peak_bin, above_threshold):
        """Controlla se c'è un click basato sui dati FFT ricevuti"""

        current_time_us = time.time() * 1_000_000
        peak_frequency = self.data_x[peak_bin]

        if above_threshold:
            if not self.click_active:
                # ✅ INIZIO CLICK
                self.click_active = True
                self.click_start_time = current_time_us
                self.click_peak_frequency = peak_frequency
                self.click_peak_amplitude = max_amplitude
                self.click_peak_time = current_time_us
                self.click_fft_count = 1  # ✅ Prima FFT del click
                #salva il tempo di inizio da impostare poi come timestamp
                self.relative_timestamp = self.get_acquisition_time()
            else:
                # ✅ CLICK IN CORSO: Incrementa contatore FFT
                self.click_fft_count += 1
                
                # Aggiorna picco se maggiore
                if max_amplitude > self.click_peak_amplitude:
                    self.click_peak_amplitude = max_amplitude
                    self.click_peak_frequency = peak_frequency
                    self.click_peak_time = current_time_us
                    
        elif self.click_active:
            # ✅ FORMATO NUOVO: "N FFT" invece di microsecondi
            duration_str = f"{self.click_fft_count} FFT"
            
            self.FFTClicksDetectedTableWidget.add_data_row([
                self.relative_timestamp,
                f"{self.click_peak_frequency:.0f} Hz",
                f"{self.click_peak_amplitude:.3f} V",
                duration_str,  # ✅ NUOVO FORMATO
                ""
            ])
            
            # Reset contatore per prossimo click
            self.click_active = False
            self.click_fft_count = 0


    # GRAFICO FFT - AGGIORNATO SOLO DAL TIMER
    def update_plot(self):
        """Aggiorna il plot solo se necessario (chiamato dal timer a 60Hz)"""
        if self.plot_needs_update and len(self.data_y_plot) > 0:
            self.plot_widget_fft.plot.setData(self.data_x, self.data_y_plot)
            self.plot_needs_update = False




    #### GESTIONE CRONOMETRO ####

    def start_chronometer(self):
        """Avvia il cronometro solo se non già attivo"""
        if self.chrono_start_time == 0:
            self.chrono_start_time = time.time()
        self.chrono_timer.start(16)  # 60 FPS (~16.67ms)
        print("⏱️ Cronometro avviato.")

    def stop_chronometer(self):
        """Ferma il cronometro e aggiorna il tempo totale"""
        self.chrono_timer.stop()
        if self.chrono_start_time > 0:
            self.total_elapsed_time += time.time() - self.chrono_start_time
            self.chrono_start_time = 0  # Reset per la prossima ripresa

    def get_acquisition_time(self):
        """Restituisce il tempo totale di acquisizione effettiva"""
        if self.chrono_start_time > 0:
            return self.total_elapsed_time + (time.time() - self.chrono_start_time)
        else:
            return self.total_elapsed_time



   ##### SISTEMA SALVATAGGIO AUDIO #####

    def save_fft_data(self):
        """Salvataggio automatico - IDENTICO al voltage"""        
        # Scegli il file di destinazione (IDENTICO al voltage)
        if self._last_saved_file is not None:
            filename = self._last_saved_file
            #print(f"📝 Salvataggio dati in file definitivo: {filename}")
        else:
            if self._last_temp_file and os.path.dirname(self._last_temp_file) == tempfile.gettempdir():
                filename = self._last_temp_file
                #print(f"📝 Salvataggio dati in temp file: {filename}")
            else:
                filename = tempfile.mktemp(prefix='plantaudio_', suffix='.paudio')
                self._last_temp_file = filename
                #print(f"📝 Creazione nuovo file temporaneo: {filename}")

        # Prepara header solo se il file non esiste (IDENTICO al voltage)
        is_new_file = not os.path.exists(filename)
        header = None
        if is_new_file:
            header = self._create_header()

        # Copia buffer e svuota subito (IDENTICO al voltage)
        y_buffer = self.data_y_buffer.copy()
        self.data_y_buffer = np.array([])

        # Copia buffer fasi e svuota subito
        phase_buffer = self.data_phase_buffer.copy()
        self.data_phase_buffer = np.array([])

        # Avvia il worker in un thread separato (IDENTICO al voltage)
        self.save_thread = QThread()
        self.save_worker = AudioSaveWorker(filename, header, y_buffer, phase_buffer, None, is_new_file)
        self.save_worker.moveToThread(self.save_thread)
        
        self.save_thread.started.connect(self.save_worker.run)
        self.save_worker.finished.connect(self.save_thread.quit)
        self.save_worker.finished.connect(self.save_worker.deleteLater)
        self.save_thread.finished.connect(self.save_thread.deleteLater)
        self.save_worker.error.connect(lambda msg: print(f"❌ Errore salvataggio: {msg}"))
        
        self.save_thread.start()

        # Aggiorna riferimento temp file solo se necessario (IDENTICO al voltage)
        if not self._last_saved_file:
            self._last_temp_file = filename
            
        return filename

    def _create_header(self, header_data=None):
        """Crea header binario come nel voltage (128 byte)"""        
        if header_data is None:
            # Calcola data_points escludendo i NaN
            valid_points = len(self.data_y_buffer[~np.isnan(self.data_y_buffer)]) if len(self.data_y_buffer) > 0 else 0
            
            header = {
                'magic': b'PLANTAUDIO',  # 10 byte (come PLANTVOLT ma per audio)
                'version': 3.0, #AGGIORNATO 3.0 = con fase          # 4 byte. 
                'experiment_type': (self.type_of_experiment or 'Audio Test')[:20].ljust(20),  # 20 byte
                'fs': self.fs,           # 4 byte (sample rate)
                'fft_size': self.fft_size,  # 4 byte
                'freq_min': self.freq_min,  # 4 byte
                'freq_max': self.freq_max,  # 4 byte
                'threshold': getattr(self, 'threshold_value', 0.03),  # 4 byte
                'start_time': getattr(self, 'start_datetime', 0.0),  # 8 byte
                'end_time': getattr(self, 'end_datetime', 0.0),      # 8 byte
                'data_points': valid_points,      # 4 byte
                'acquisition_count': getattr(self, '_acquisition_count', 0),  # 4 byte
                'reserved': b'\x00' * 50         # 50 byte (padding)
            }
        else:
            header = header_data

        # Costruisci header binario (128 byte totali)
        header_bytes = bytearray()
        
        # Magic number (10 byte)
        magic = header['magic'][:10]
        header_bytes.extend(magic)
        header_bytes.extend(b'\x00' * (10 - len(magic)))  # Padding se necessario
        
        # Version (4 byte)
        header_bytes.extend(struct.pack('<f', header['version']))
        
        # Experiment type (20 byte)
        exp_type = header['experiment_type'].encode('ascii', errors='replace')[:20]
        exp_type += b'\x00' * (20 - len(exp_type))
        header_bytes.extend(exp_type)
        
        # Audio parameters (20 byte)
        header_bytes.extend(struct.pack('<I', header['fs']))         # 4 byte
        header_bytes.extend(struct.pack('<I', header['fft_size']))   # 4 byte
        header_bytes.extend(struct.pack('<I', header['freq_min']))   # 4 byte
        header_bytes.extend(struct.pack('<I', header['freq_max']))   # 4 byte
        header_bytes.extend(struct.pack('<f', header['threshold']))  # 4 byte
        
        # Timestamps (16 byte)
        header_bytes.extend(struct.pack('<d', header['start_time'])) # 8 byte
        header_bytes.extend(struct.pack('<d', header['end_time']))   # 8 byte
        
        # Counters (8 byte)
        header_bytes.extend(struct.pack('<I', header['data_points']))      # 4 byte
        header_bytes.extend(struct.pack('<I', header['acquisition_count'])) # 4 byte
        
        # Reserved space (50 byte)
        header_bytes.extend(header['reserved'][:50])
        
        # Verifica dimensione (come nel voltage)
        if len(header_bytes) != 128:
            raise ValueError(f"Dimensione header errata: {len(header_bytes)} byte (attesi 128)")

        return bytes(header_bytes)



    def save_file_action(self, ask_filename=True):
        from saving.audio_save_worker import AudioSaveActionWorker

        print("💾 Salvataggio manuale audio (solo FFT data)...")

        # --- Selezione file ---
        if ask_filename:
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Audio Data",
                f"audio_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.paudio",
                "PlantLeaf Audio (*.paudio);;All Files (*)"
            )
            if not filename:
                return False
            if not filename.endswith('.paudio'):
                filename += '.paudio'
            self._last_saved_file = filename
            print(f"📁 File definitivo scelto: {filename}")
        else:
            if self._last_saved_file:
                filename = self._last_saved_file
                print(f"💾 Salvataggio finale in file definitivo: {filename}")
            elif getattr(self, 'is_closing', False) and not getattr(self, 'is_cleaning', False):
                return False
            elif not self.is_cleaning:
                self.save_file_action(ask_filename=True)
                print("richiedo con salvataggio manuale")
                return True
            else:
                print("Stato di pulizia attivo, salvataggio automatico annullato.")
                return False

        try:
            # --- Progress Dialog ---
            self.progress_save = self.get_progress_widget("Saving Audio Data...")
            self.progress_save.setValue(0)
            self.progress_save.show()

            # --- Prepara dati da salvare ---
            all_fft_data = []
            all_phase_data = []

            # 1. Leggi TUTTI i dati dal file temporaneo (se esiste)
            source_file = None
            if self._last_temp_file and os.path.exists(self._last_temp_file):
                source_file = self._last_temp_file
                print(f"📊 Lettura dati da file temporaneo: {self._last_temp_file}")
                
                with open(source_file, 'rb') as f:
                    f.seek(128)  # Salta header
                    data = f.read()
                    
                    # Cerca marker click data
                    click_start = data.find(b'CLCK')
                    if click_start >= 0:
                        binary_data = data[:click_start]
                    else:
                        binary_data = data
                    
                    if binary_data:
                        # ✅ LETTURA INTERLACCIATA
                        # Ogni "campione FFT" = 5 byte (4B mag + 1B phase)
                        bytes_per_sample = 5
                        num_samples = len(binary_data) // bytes_per_sample
                        
                        for i in range(num_samples):
                            offset = i * bytes_per_sample
                            
                            # Leggi magnitudine (4 byte)
                            mag = struct.unpack('<f', binary_data[offset:offset+4])[0]
                            all_fft_data.append(mag)
                            
                            # Leggi fase (1 byte)
                            phase = struct.unpack('<b', binary_data[offset+4:offset+5])[0]
                            all_phase_data.append(phase)
                        
                        print(f"📊 Letti {num_samples} campioni (mags+fasi) da file temp")

            # 2. Aggiungi buffer corrente
            if len(self.data_y_buffer) > 0:
                all_fft_data.extend(self.data_y_buffer.tolist())
                all_phase_data.extend(self.data_phase_buffer.tolist())
                print(f"📊 Aggiunti {len(self.data_y_buffer)} campioni da buffer")

            # --- Prepara header ---
            all_fft_array = np.array(all_fft_data, dtype=np.float32)
            all_phase_array = np.array(all_phase_data, dtype=np.int8)
            
            valid_points = np.sum(~np.isnan(all_fft_array))
            
            header = self._create_header({
                'magic': b'PLANTAUDIO',
                'version': 3.0,  # VERSIONE AGGIORNATA PER FASI
                'experiment_type': (self.type_of_experiment or 'Audio Test')[:20].ljust(20),
                'fs': self.fs,
                'fft_size': self.fft_size,
                'freq_min': self.freq_min,
                'freq_max': self.freq_max,
                'threshold': getattr(self, 'threshold_value', 0.03),
                'start_time': getattr(self, 'start_datetime', 0.0),
                'end_time': getattr(self, 'end_datetime', 0.0),
                'data_points': valid_points,
                'acquisition_count': self._acquisition_count,
                'reserved': b'\x00' * 50
            })

            # --- Avvia worker in thread ---
            self.save_thread = QThread()
            self.save_worker = AudioSaveActionWorker(
                filename, header, 
                all_fft_array, 
                all_phase_array,  # NUOVO parametro
                None, 
                True
            )
            self.save_worker.moveToThread(self.save_thread)
            self.save_worker.progress.connect(self.progress_save.setValue)
            self.save_worker.finished.connect(self._on_save_finished)
            self.save_worker.error.connect(self._on_save_error)
            self.save_worker.cancelled.connect(self._on_save_cancelled)
            self.progress_save.canceled.connect(self.save_worker.cancel)
            self.save_thread.started.connect(self.save_worker.run)
            self.save_thread.start()

            # Svuota buffer
            self.data_y_buffer = np.array([])
            self.data_phase_buffer = np.array([])

            self._last_temp_file = None  # Resetta il temp file
            self.actionSave.setEnabled(False)  # Disabilita salvataggio multiplo
            self.actionSave.setToolTip(f"File already saved in {filename}")
            self.definetly_saved = True


            # Reset flag
            if hasattr(self, 'is_cleaning'):
                self.is_cleaning = False

            return filename

        except Exception as e:
            print(f"❌ Errore salvataggio: {e}")
            if 'progress' in locals():
                self.progress_save.close()
            self.show_error_dialog("Save Error", f"Cannot save file:\n{str(e)}")
            return False

    def _on_save_finished(self):
        self.progress_save.close()
        self.save_thread.quit()
        self.save_thread.wait()
        self.save_thread = None
        self.save_worker = None
        print(f"✅ Salvataggio completato.")
        # Se la chiusura era in sospeso, chiudi ora
        if hasattr(self, '_pending_close_event') and self._pending_close_event:
            self._pending_close_event = False
            print("✅ Ora posso chiudere la finestra dopo il salvataggio.")
            if getattr(self, 'opening_new_file', False):
                print("✅ Procedo con l'apertura del nuovo file dopo il salvataggio...")
                self.open_file_action()
                return
            if getattr(self, 'new_file_to_open', False):
                print("✅ Procedo con la creazione di un nuovo file dopo il salvataggio...")
                self.new_file_action()
                return
            if getattr(self, '_replay_after_save', False):
                self._replay_after_save = False
                self.replay_experiment_action()
                return
            if getattr(self, 'is_closing', False):
                self.finally_closing = True
                self.close()
                return
            if getattr(self, 'is_cleaning', False):
                print("✅ Procedo con la pulizia dei dati dopo il salvataggio...")
                if getattr(self, '_last_saved_file', None):
                    self._finalize_file_data(self._last_saved_file)
                self._clear_experiment_data()
                return
            if getattr(self, 'going_home', False):
                print("✅ Procedo con la navigazione alla home dopo il salvataggio...")
                if self.isFullScreen():
                    self.showNormal()
                self._navigate_home()
                return
            else:
                print("✅ Nessuna azione pendente dopo il salvataggio.")
                return

    def _on_save_error(self):
        self.progress_save.close()
        self.save_thread.quit()
        self.save_thread.wait()
        print(f"❌ Errore salvataggio:")
        self.show_error_dialog("Save Error", f"Cannot save current file.")
        self.save_thread = None
        self.save_worker = None

    def _on_save_cancelled(self):
        self.progress_save.close()
        self.save_thread.quit()
        self.save_thread.wait()
        print("⚠️ Salvataggio annullato dall'utente.")
        self.save_thread = None
        self.save_worker = None

    def save_click_data(self, audio_filename):
        """Salva click data INTEGRATI nel file .paudio ORA CON FINALIZE"""
        try:
            # Estrai click data dalla tabella esistente
            if hasattr(self, 'FFTClicksDetectedTableWidget'):
                click_data = self.FFTClicksDetectedTableWidget.export_click_data()

                print("🔍 DEBUG - Click data raw:")
                for i, click in enumerate(click_data[:3]):  # Primi 3 per debug
                    print(f"   Click {i}: {click}")

                if click_data:
                    # ✅ APPENDE al file .paudio esistente
                    with open(audio_filename, 'ab') as f:
                        import json, zlib
                        
                        # Marker per identificare inizio click data
                        f.write(b'CLCK')  # 4 byte marker
                        
                        # JSON compresso per efficienza
                        click_json = json.dumps(click_data, separators=(',', ':'))
                        click_compressed = zlib.compress(click_json.encode('utf-8'))
                        
                        # Lunghezza dati click (per lettura)
                        f.write(struct.pack('<I', len(click_compressed)))  # 4 byte
                        
                        # Dati click compressi
                        f.write(click_compressed)

                    self._click_data_saved = True
                    print(f"📊 Click data integrati nel file: {len(click_data)} eventi")
                else:
                    print("📊 Nessun click data da integrare")
            
        except Exception as e:
            print(f"⚠️ Errore integrazione click data: {e}")


    def _finalize_file_data(self, filename):
        """
        OVERRIDE: Implementazione del gancio di finalizzazione.
        Questo metodo viene chiamato da BaseWindow SOLO durante il salvataggio finale
        (chiusura, pulizia, etc.) per aggiungere i dati dei click al file.
        """
        self.save_click_data(filename)



##### PULSANTI ######

    #GIÀ INTEGRATO NELLO STESSO DEL CAMBIO STILE   
    #def on_clicks_detection_status_changed(self, status):
     #   print(f"🔄Clicks Detection changed: {status}")
      #  self.clicksDetectionStatus = status


    def update_chrono_label(self):
        elapsed = int(self.get_acquisition_time())
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60
        self.FFTTimePassedLabelTime.setText(f"{hours}:{minutes:02}:{seconds:02}")

        """
    def update_table_data(self, frequency, amplitude, duration, duration_us, notes):
        if hasattr(self, 'chrono_start_time') and self.chrono_start_time > 0:
            self.relative_timestamp = self.get_acquisition_time()
        else:
            self.relative_timestamp = 0.0

        # Passa SOLO 5 parametri: timestamp, frequency, amplitude, duration, notes
        self.FFTClicksDetectedTableWidget.add_data_row([
            self.relative_timestamp, frequency, amplitude, duration, notes
        ])
        """





    #cambio valore spinBox = THRESHOLD e manda al microcontrollore
    def on_fft_threshold_changed(self, value):
        """Gestisce il cambio manuale della soglia dallo SpinBox."""
            
        print(f"🎚️ Threshold cambiata manualmente a: {value:.3f}V")
        
        # Registra il momento della modifica manuale
        self._last_user_threshold_change = time.time()

        try:
            # Rimuovi la vecchia threshold line
            if hasattr(self, 'threshold_curve_fft') and self.threshold_curve_fft is not None:
                self.plot_widget_fft.remove_curve(self.threshold_curve_fft)
                self.threshold_curve_fft = None

            # Crea la nuova threshold line
            x = self.data_x
            y = np.full_like(x, value)
            self.threshold_curve_fft = self.plot_widget_fft.add_threshold(
                x=x, y=y, name="Threshold FFT", pen='r'
            )

            # Invia al microcontrollore solo se connesso
            if (hasattr(self, 'serial_worker') and 
                hasattr(self.serial_worker, 'ser') and 
                self.serial_worker.ser and 
                self.serial_worker.ser.is_open):
                
                try:
                    threshold_cmd = f"!threshold {value:.3f}".encode('utf-8')
                    self.serial_worker.ser.write(threshold_cmd)
                    print(f"📡 Soglia inviata al micro: {value:.3f}V")
                except Exception as e:
                    print(f"❌ Errore invio soglia: {e}")

            # Aggiorna valore locale
            self.threshold_value = value
            
        except Exception as e:
            print(f"Errore durante l'aggiornamento della soglia: {e}")




    ###### SETUP #####
    def _setup_ui(self):
        # Setup controlli
        self.clicksDetectionStatus = True  # Stato iniziale
        self.FFTClicksDetectorButton.setText("ON")
        self.FFTClicksDetectorButton.setFont(self.font_manager.create_fonts()['button'])
        self.FFTClicksDetectorButton.setCheckable(True)
        self.FFTClicksDetectorButton.setChecked(True)  # Stato iniziale ON
        #self.FFTClicksDetectorButton.setStyleSheet(self.theme_manager.get_toggle_button_style(self.clicksDetectionStatus))
        self.FFTClicksDetectorButton.clicked.connect(self.toggle_clicks_detection)
        #self.FFTClicksDetectorButton.clicked.connect(lambda: self.on_clicks_detection_status_changed(self.clicksDetectionStatus))
        #disattiva temporaneamente
        #self.FFTClicksDetectorButton.setEnabled(False)
        #self.FFTClicksDetectorButton.setToolTip("Not available yet")

        # SETUP TIMER: cronometro + plot a 60Hz
        self.chrono_timer = QTimer(self)
        self.chrono_timer.timeout.connect(self.update_chrono_label)
        self.chrono_timer.timeout.connect(self.update_plot)  # Plot refresh a 60Hz
        self.FFTTimePassedLabelTime.setText("0:00:00")

        # Configura FFTThresholdSpinBox
        self.FFTThresholdSpinBox.setMinimum(0.001) 
        self.FFTThresholdSpinBox.setMaximum(1.65)
        
        # FIXED: Imposta valore di default PRIMA di connettere il signal
        self.threshold_value = 0.03  # Valore di default
        self.FFTThresholdSpinBox.setValue(self.threshold_value)
        
        self.FFTThresholdSpinBox.setDecimals(3)
        self.FFTThresholdSpinBox.setSingleStep(0.001)
        self.FFTThresholdSpinBox.setSuffix(" V")
        self.FFTThresholdSpinBox.setEnabled(False)
        
        # Connetti DOPO aver impostato il valore
        self.FFTThresholdSpinBox.valueChanged.connect(self.on_fft_threshold_changed)

    def create_initial_threshold(self):
        """Crea la threshold line iniziale"""
        if hasattr(self, 'data_x') and len(self.data_x) > 0:
            x = self.data_x
            y = np.full_like(x, self.threshold_value)
            
            self.threshold_curve_fft = self.plot_widget_fft.add_threshold(
                x=x,
                y=y,
                name="Threshold FFT",
                pen='r'
            )
            print(f"🎚️ Threshold iniziale creata: {self.threshold_value:.2f} V")

    def _setup_table_fonts(self):
        """Imposta i font per le tabelle"""
        fonts = self.font_manager.create_fonts()
        self.FFTClicksDetectedLabel.setFont(fonts['label'])
        self.FFTClicksDetectedTableWidget.horizontalHeader().setFont(fonts['table_header'])
        self.FFTClicksDetectedTableWidget.setFont(fonts['table_content'])
        
        self.FFTClicksDetectedTableWidget.setToolTip(
            "Le note sono editabili (max 20 caratteri). "
            "Clicca su una cella nella colonna 'Notes' per aggiungere appunti."
        )

    def toggle_clicks_detection(self):
        #CAMBIA STILE E ANCHE STATO
        self.clicksDetectionStatus = not self.clicksDetectionStatus
        button_text = "ON" if self.clicksDetectionStatus else "OFF"
        self.FFTClicksDetectorButton.setText(button_text)
        self.FFTClicksDetectorButton.setChecked(self.clicksDetectionStatus)
        #self.FFTClicksDetectorButton.setStyleSheet(self.theme_manager.get_toggle_button_style(self.clicksDetectionStatus))
        self.clicks_detector_toggled.emit(self.clicksDetectionStatus)
        pass


    def setup_splitter(self):
        """Configura gli splitter"""
        if (hasattr(self, 'FFTPlotHLayout') and hasattr(self, 'FFTPotWidget') and 
            hasattr(self, 'FFTTableVLayout') and hasattr(self, 'FFTClicksDetectedLabel') and 
            hasattr(self, 'FFTClicksDetectedTableWidget')):
            self.setup_tab_splitter(
                self.FFTPlotHLayout,
                self.FFTPotWidget, 
                self.FFTTableVLayout,
                self.FFTClicksDetectedLabel,
                self.FFTClicksDetectedTableWidget,
                'fft_splitter'
            )

    def setup_tab_splitter(self, horizontal_layout, plot_widget, vertical_layout, 
                          label_widget, table_widget, splitter_name):
        """Configura lo splitter per un singolo tab"""
        horizontal_layout.removeWidget(plot_widget)
        horizontal_layout.removeItem(vertical_layout)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        setattr(self, splitter_name, splitter)
        
        splitter.addWidget(plot_widget)
        
        from PySide6.QtWidgets import QWidget, QVBoxLayout
        table_container = QWidget()
        container_layout = QVBoxLayout(table_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        vertical_layout.removeWidget(label_widget)
        vertical_layout.removeWidget(table_widget)
        
        container_layout.addWidget(label_widget)
        container_layout.addWidget(table_widget)
        
        splitter.addWidget(table_container)
        splitter.setSizes([700, 350])
        horizontal_layout.addWidget(splitter)


    def start_experiment_action(self):
        self.FFTStartStopButton.click()

    def sampling_settings_action(self):
        from PySide6.QtWidgets import QDialog
        popup = AudioSamplingSettingsPopup(self.theme_manager, parent=self)
        # Imposta il valore corrente
        popup.set_existing_settings(self.type_of_experiment)
        result = popup.exec()
        if result == QDialog.Accepted:
            settings = popup.get_settings()
            self.type_of_experiment = settings["experiment_type"]
            print("Impostazioni audio aggiornate:", self.type_of_experiment)

    def on_serial_port_selected(self, port):
        """AGGIORNATO: connette al nuovo segnale con 5 parametri"""
        self.serial_worker = AudioSerialWorker(port)
        self.set_buttons_enabled(True)
        # NUOVO SEGNALE con 5 parametri
        self.serial_worker.new_data.connect(self.on_new_fft_data)
        
        # ✅ CONNETTE SEGNALI DI DISCONNESSIONE
        try:
            self.serial_worker.error_popup.connect(self.show_serial_error)
            self.serial_worker.serial_connection_status_bool.connect(self.handle_connection_status)
        except Exception as e:
            print(f"Errore connessione funzioni error_popup: {e}")
        
        self.serial_worker.connection()
        print(f"Porta seriale selezionata: {port}")

    def handle_connection_status(self, is_connected):
        """✅ GESTISCE STATO CONNESSIONE SERIALE"""
        if not is_connected:
            if hasattr(self, "actionSerialPort"):
                self.actionSerialPort.setEnabled(True)

    def set_buttons_enabled(self, enabled: bool):
        """Abilita o disabilita i pulsanti"""
        self.actionStart.setEnabled(enabled)
        self.FFTStartStopButton.setEnabled(enabled)
        self.FFTThresholdSpinBox.setEnabled(enabled)
