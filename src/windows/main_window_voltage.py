"""
Finestra principale per il monitoraggio Voltage
"""

from PySide6.QtCore import Signal, Qt, QThread
import numpy as np
import pyqtgraph as pg
import time

from core.base_window import BaseWindow
from .ui.ui_MainWindowVoltage import Ui_MainWindowVoltage
from components.start_stop_button import StartStopButton
from components.sampling_settings import VoltageSamplingSettingsPopup
from core.special_component import replace_widget
from plotting.plot_manager import BasePlotWidget
from serial_communication.voltage_read import VoltageSerialWorker
from saving.voltage_save_worker import VoltageSaveWorker


#import per salvataggio dati
import struct
from datetime import datetime
import os



class MainWindowVoltage(BaseWindow, Ui_MainWindowVoltage):
    """Finestra principale per il monitoraggio voltage delle piante"""
    # Inizializza segnali dei toggle
    amplified_data_toggled = Signal(bool)
    connectivity_check_toggled = Signal(bool)
    
    def __init__(self, parent=None):
        Ui_MainWindowVoltage.__init__(self)
        BaseWindow.__init__(self, parent)

        self.setupUi(self)

        # Inizializza linea che separa start e stop
        self.pause_markers = []

        # Inizializza array per i dati
        self.data_x_buffer = np.array([])
        self.data_y_buffer = np.array([])
        #INIZIALIZZA ARRAY PER VALORI DA PLOTTARE
        self.data_x_plot = np.array([], dtype=float)
        self.data_y_plot = np.array([], dtype=float)

        # Inizializza timer con tempo assoluto
        self.total_elapsed_time = 0
        self.chrono_start_time = 0

        # Visualizzazione ultimi x campioni
        self.max_points = 60000  # Limite di campioni da visualizzare


        # Sostituzione pulsante start/stop
        custom_btn = StartStopButton(self.theme_manager, parent=self)
        replace_widget(self, "startStopButton", custom_btn)
        self.startStopButton = custom_btn
        
        custom_plot = BasePlotWidget(
            x_label="Time",
            y_label="Voltage",
            x_range=(0, 15),
            y_range=(-1.7, 1.7),
            x_min=0, x_max=None, y_min=-1.7, y_max=1.7,
            unit_x="s", unit_y="V",
            parent=self
        )
        replace_widget(self, "plotWidget", custom_plot)
        self.plot_widget = custom_plot

        # Crea la curva principale con la penna desiderata
        self.plot_widget.plot = self.plot_widget.plot_widget.plot(name="Voltage Data")


        self.setWindowTitle("Voltage Monitor")

        self.startStopButton.started.connect(self.on_start)
        self.startStopButton.stopped.connect(self.on_stop)


        self.amplified_data_toggled.connect(self.on_amplified_data_changed)
        #manda il seganle startup
        self.amplified_data_toggled.emit(True)  # Imposta lo stato iniziale
        #self.connectivity_check_toggled.connect(self.on_connectivity_check_changed)

        self._setup_specific_ui()

        self.layout_manager.center_window_on_screen(self)


        self.time_window_enabled = True
        self.actionUpdatedTimeWindow.setToolTip("Turn off time window update")

        self.setup_toolbar_actions()
        self.setup_menubar_actions()

        #applica il tema dinamico al widget plot
        # la curva viene creata in voltage_plot.py
        self.theme_manager.apply_theme_to_plot(self.plot_widget.plot_widget, self.plot_widget.plot)

        self.setStatusBar(None)  # Disabilita la status bar predefinita

        # Imposta l'azione di avvio dell'esperimento come disattivata e anche i pulsanti di start/stop
        self.startStopButton.setEnabled(False)
        self.actionStart.setEnabled(False)
        self.amplifiedDataButton.setEnabled(True)
        self.connectivityCheckButton.setEnabled(False)

        # Variabili per salvare le impostazioni di sampling_settings
        self.duration = 120 #default
        self.sampling_rate = 500 #default
        self.type_of_experiment = "Test"  # Default, può essere modificato in sampling_settings


        # Inizializza variabile per amplified data
       # self.amplified_data_toggled.emit(True)  # Imposta lo stato iniziale a True

        # Variabili per salvataggio dati
        self._last_temp_file = None
        self._last_saved_file = None  # Per tenere traccia dell'ultimo file salvato
        self._acquisition_count = 0       
        self.definetly_saved = False

        self.is_acquiring = False

        # mostra a tutto schermo mantenendo le grafiche
        self.showMaximized()



    ##### GESTIONE GRAFICO #####
    
    #connesso a dati raccolti dalla seriale
    def on_new_voltage_data(self, value):
        if not getattr(self, "is_acquiring", False):
            return
        elapsed = self.total_elapsed_time + (time.time() - self.chrono_start_time)
        value *= float(self.multiplier)
        # Aggiungi ai buffer di salvataggio
        self.data_x_buffer = np.append(self.data_x_buffer, elapsed)
        self.data_y_buffer = np.append(self.data_y_buffer, value)
        # Aggiungi ai dati per il plot (mantieni solo gli ultimi max_points)
        self.data_x_plot = np.append(self.data_x_plot, elapsed)
        self.data_y_plot = np.append(self.data_y_plot, value)
        if len(self.data_x_plot) > self.max_points:
            self.data_x_plot = self.data_x_plot[-self.max_points:]
            self.data_y_plot = self.data_y_plot[-self.max_points:]
        self.update_plot()
        # se la finestra temporale è abilitata, aggiorna la visualizzazione
        if self.time_window_enabled:
            if len(self.data_x_plot) > 0:
                xmin = self.data_x_plot[0]
            else:
                xmin = 0
            self.plot_widget.update_time_window(x_data_seconds=self.data_x_plot, xmin=xmin)


        if len(self.data_x_buffer) >= 1000:
            self.save_voltage_data()
    

    def update_plot(self):
        if len(self.data_x_plot) > 0:
            self.plot_widget.plot.setData(self.data_x_plot, self.data_y_plot)
            # Imposta i limiti degli assi in base ai dati correnti SOLO se è passato un intervallo di tempo sufficiente
            interval = self.max_points / self.sampling_rate
            if self.total_elapsed_time > interval:
            # Calcola l'intervallo di tempo da visualizzare
                x_max = self.data_x_plot[-1]
                x_min = x_max - interval
                self.plot_widget.set_axis_limits(x_min=x_min, x_max=x_max, y_min=-1.7, y_max=1.7)


    def setup_chronometer(self):
        from PySide6.QtCore import QTimer
        self.chrono_timer = QTimer(self)
        self.chrono_timer.timeout.connect(self.on_chrono_tick)

    def on_chrono_tick(self):
        # Aggiorna solo la finestra temporale del plot, se serve
        if self.time_window_enabled and len(self.data_x_plot) > 0:
            max_points = self.max_points  # 60000
            if len(self.data_x_plot) > max_points:
                xmin = self.data_x_plot[-max_points]
            else:
                xmin = self.data_x_plot[0]
            self.plot_widget.update_time_window(x_data_seconds=self.data_x_plot, xmin=xmin)


    #sovrascrive il metodo di voltage_plot
    def start_chronometer(self):
        self.chrono_start_time = time.time()
        self.chrono_timer.start(50) #GESTIONE REFRESHING RATE DEL GRAFICO (ora è 20fps)


    def stop_chronometer(self):
        # Aggiorna il tempo totale trascorso
        if hasattr(self, 'chrono_start_time'):
            self.total_elapsed_time += time.time() - self.chrono_start_time
        self.chrono_timer.stop()



    def on_start(self):
        # ✅ CONTROLLO SICUREZZA COMPLETO
        if (not hasattr(self, 'serial_worker') or 
            self.serial_worker is None or 
            not getattr(self.serial_worker, 'is_connected', False)):
            
            print("❌ Impossibile avviare: porta seriale non connessa")
            
            # ✅ RIABILITA azione SerialPort
            if hasattr(self, 'actionSerialPort'):
                self.actionSerialPort.setEnabled(True)
                
            return  # ✅ ESCI SUBITO - FONDAMENTALE!
        
        # ✅ Solo se tutto OK, procedi
        self.is_acquiring = True
        self.serial_worker.start()

        self.start_chronometer()



        # Disabilita azioni
        try:
            if self.serial_worker.is_connected:
                self.actionClear.setEnabled(False)
                self.amplifiedDataButton.setEnabled(False)
                self.actionSamplingSettings.setEnabled(False)
                self.actionSerialPort.setEnabled(False)
                self.actionSave.setEnabled(False)
                self.actionOpenFile.setEnabled(False)
                self.actionNewFile.setEnabled(False)
                print("serial off")
        except Exception as e:
            print(f"errore non so perché: {e}")
            return

        # Inizializza variabile ora di inizio
        if not hasattr(self, 'start_datetime'):
            # Inizializza start_datetime solo la prima volta
            self.start_datetime = datetime.now().timestamp()



    def on_stop(self):
        if not self.isVisible():  # Se la finestra sta chiudendosi, non salvare
            return
        self.is_acquiring = False
        
        # Inizializza variabile ora di fine (viene sovrascritta ogni volta)
        self.end_datetime = datetime.now().timestamp()
        
        self.stop_chronometer()

        # Chiama il metodo sicuro centralizzato in BaseWindow
        self._safe_stop_serial_worker()
        
        # ✅ CONTROLLO SICUREZZA per riabilitazione porta
        if (hasattr(self, 'serial_worker') and 
            self.serial_worker is not None and 
            not getattr(self.serial_worker, 'is_connected', False)):
            if hasattr(self, "actionSerialPort"):
                self.actionSerialPort.setEnabled(True)
                self.set_buttons_enabled(False)
        
        if len(self.data_x_plot) > 0:
            pause_x = self.data_x_plot[-1]
            vline = pg.InfiniteLine(pos=pause_x, angle=90, pen=pg.mkPen('r', width=2, style=Qt.DashLine))
            self.plot_widget.plot_widget.addItem(vline)
            self.pause_markers.append(vline)
            self.data_x_plot = np.append(self.data_x_plot, np.nan)
            self.data_y_plot = np.append(self.data_y_plot, np.nan)
            self.plot_widget.plot.setData(self.data_x_plot, self.data_y_plot)


        self.remove_data_after_stop()

        max_points = self.max_points  # 60000
        if len(self.data_x_plot) > max_points:
            xmin = self.data_x_plot[-max_points]
        elif len(self.data_x_plot) > 0:
            xmin = self.data_x_plot[0]
        else:
            xmin = 0
        self.plot_widget.update_time_window(x_data_seconds=self.data_x_plot, xmin=xmin)
        
        self.actionClear.setEnabled(True)
        if not self.definetly_saved:
            self.actionSave.setEnabled(True)
        self.actionOpenFile.setEnabled(True)
        self.actionNewFile.setEnabled(True)
        print("🛑 Arrestato Monitoraggio Voltage")



    # SI PUO scegliere solo priam di iniziare l'esperimento
    def on_amplified_data_changed(self, state):
        # Prendi i dati in base allo stato di amplified_data_state
        if state:
            # Lascia i valori amplificati
            self.multiplier = 1.0
            # imposta i limiti del grafico in base allo stato di amplified_data_state
            self.plot_widget.set_y_range(-1.7, 1.7)  
            #cambia titolo dell'asse y
            self.plot_widget.set_y_label('Voltage', 'V')
            #debug
            print("Amplified Data is ON, using amplified data values.")
        else:
            # Dividi per l'amplificazione per tornare ai valori originali
            self.multiplier = 76.92307692307693  # 76.92 è l'inverso di 1000/13
            # imposta i limiti del grafico in base allo stato di amplified_data_state
            self.plot_widget.set_y_range(-130, 130)
            #cambia titolo dell'asse y
            self.plot_widget.set_y_label('Voltage', 'mV')
            #debug
            print("Amplified Data is OFF, using original data values.")

    
    
    def on_connectivity_check_changed(self, state):
        print(f"🔗 Connectivity Check changed: {state}")



    def update_signal_stability(self, stability=str):
        """Aggiorna la stabilità del segnale"""
        if stability == "CONNECTED":
            self.boxFrameSignalStability.setStyleSheet("background-color: blue; color: white;")
            self.signalStabilityLabelInfo.setText(stability.upper())
        if stability == "PERFECT":
            self.boxFrameSignalStability.setStyleSheet("background-color: green; color: white;")
            self.signalStabilityLabelInfo.setText(stability.upper())
        elif stability == "GOOD":
            self.boxFrameSignalStability.setStyleSheet("background-color: red; color: white;")
            self.signalStabilityLabelInfo.setText(stability.upper())
        elif stability == "LOW DATA":
            self.boxFrameSignalStability.setStyleSheet("background-color: orange; color: white;")
            self.signalStabilityLabelInfo.setText(stability.upper())
        elif stability == "NO DATA":
            self.boxFrameSignalStability.setStyleSheet("background-color: yellow; color: black;")
            self.signalStabilityLabelInfo.setText(stability.upper())
        elif stability == "NOT CONNECTED":
            self.boxFrameSignalStability.setStyleSheet("background-color: gray; color: white;")
            self.signalStabilityLabelInfo.setText(stability.upper())
        elif stability == "STOPPED":
            self.boxFrameSignalStability.setStyleSheet("background-color: black; color: white;")
            self.signalStabilityLabelInfo.setText(stability.upper())


    #STILI

    def _setup_specific_ui(self):
        # Setup controlli amplified data
        self.amplified_data_state = True
        self.amplifiedDataButton.setText("ON")
        self.amplifiedDataButton.setFont(self.font_manager.create_fonts()['button'])
        #self.amplifiedDataButton.setStyleSheet(self.theme_manager.get_toggle_button_style(self.amplified_data_state))
        self.amplifiedDataButton.clicked.connect(self.toggle_amplified_data_style)
        #inizializzalo come cliccato
        self.amplifiedDataButton.setChecked(self.amplified_data_state)

        # Setup controlli connectivity check
        self.connectivity_check_state = False
        self.connectivityCheckButton.setText("OFF")
        self.connectivityCheckButton.setFont(self.font_manager.create_fonts()['button'])
        #self.connectivityCheckButton.setStyleSheet(self.theme_manager.get_toggle_button_style(self.connectivity_check_state))
        self.connectivityCheckButton.clicked.connect(self.toggle_connectivity_check_style)
        #disabilita temporaneamente
        self.connectivityCheckButton.setEnabled(False)
        self.connectivityCheckButton.setToolTip("Not available yet")

        # stato segnale
        self.update_signal_stability("NOT CONNECTED")  # Inizializza con stato non connesso

        #test cronometro
        self.setup_chronometer()  # Inizializza il cronometro per il graf

    
    def toggle_amplified_data_style(self):
        """Toggle amplified data state"""
        self.amplified_data_state = not self.amplified_data_state
        button_text = "ON" if self.amplified_data_state else "OFF"
        self.amplifiedDataButton.setText(button_text)
        #self.amplifiedDataButton.setStyleSheet(self.theme_manager.get_toggle_button_style(self.amplified_data_state))
        print(f"📊 Amplified Data: {button_text}")
        self.amplified_data_toggled.emit(self.amplified_data_state)  # <--- emetti segnale



    def toggle_connectivity_check_style(self):
        """Toggle connectivity check state"""
        self.connectivity_check_state = not self.connectivity_check_state
        button_text = "ON" if self.connectivity_check_state else "OFF"
        self.connectivityCheckButton.setText(button_text)
        #self.connectivityCheckButton.setStyleSheet(self.theme_manager.get_toggle_button_style(self.connectivity_check_state))
        print(f"🔗 Connectivity Check: {button_text}")
        self.connectivity_check_toggled.emit(self.connectivity_check_state)  # <--- emetti segnale


    def start_experiment_action(self):
        # Simula il click sul pulsante Start/Stop
        self.startStopButton.click()

    def sampling_settings_action(self):
        from PySide6.QtWidgets import QDialog
        popup = VoltageSamplingSettingsPopup(self.theme_manager, parent=self)
        # Imposta i valori correnti
        popup.type_edit.setText(self.type_of_experiment)
        popup.rate_spin.setValue(self.sampling_rate)
        result = popup.exec()
        if result == QDialog.Accepted:
            settings = popup.get_settings()
            # Salva nelle variabili della finestra
            self.sampling_rate = settings["sampling_rate"]
            self.type_of_experiment = settings["experiment_type"]
            print("Impostazioni aggiornate:", self.sampling_rate, self.type_of_experiment)

        # Se esiste un worker seriale già creato, aggiorna il suo valore di sampling
        if hasattr(self, 'serial_worker') and self.serial_worker is not None:
            try:
                self.serial_worker.sampling_value = self.sampling_rate
                print(f"⚙️ Aggiornato sampling_value del serial_worker: {self.sampling_rate} Hz")
            except Exception as e:
                print(f"⚠️ Errore aggiornando sampling_value sul serial_worker: {e}")



    #FUNZIONI BACKEND
    def on_serial_port_selected(self, port):
        self.serial_worker = VoltageSerialWorker(port, self.sampling_rate)
        self.serial_worker.serial_connection_status_bool.connect(self.set_buttons_enabled)
        self.serial_worker.new_data.connect(self.on_new_voltage_data)
        self.serial_worker.error_popup.connect(self.show_serial_error)
        self.serial_worker.stability_status_changed.connect(self.update_signal_stability)
        self.serial_worker.connection()
        #aspetta 1.5 secondo per stabilire la connessione
        time.sleep(1.5)
        print(f"Porta seriale selezionata: {port}")



    #funzione per abilitare/disabilitare i pulsanti
    def set_buttons_enabled(self, enabled: bool):
        """Abilita o disabilita i pulsanti di start/stop e sampling settings"""
        self.startStopButton.setEnabled(enabled)
        self.actionStart.setEnabled(enabled)
        #self.connectivityCheckButton.setEnabled(enabled)


    def remove_data_after_stop(self):
        """Rimuove tutti i dati raccolti dopo lo stop (cioè dopo total_elapsed_time)."""
        mask = self.data_x_plot <= self.total_elapsed_time
        self.data_x_plot = self.data_x_plot[mask]
        self.data_y_plot = self.data_y_plot[mask]
        self.plot_widget.plot.setData(self.data_x_plot, self.data_y_plot)




    # SALVATAGGIO DATI
    def save_voltage_data(self):
        """Salva i dati nel file giusto: temporaneo o definitivo, in un thread separato."""
        import tempfile, os

        # Scegli il file di destinazione
        if self._last_saved_file is not None:
            filename = self._last_saved_file
            #print(f"Salvataggio dati in: {filename}")
        else:
            if self._last_temp_file and os.path.dirname(self._last_temp_file) == tempfile.gettempdir():
                filename = self._last_temp_file
                #print(f"Salvataggio dati in: {filename}")
            else:
                filename = tempfile.mktemp(prefix='plantvolt_', suffix='.pvolt')
                self._last_temp_file = filename
                print(f"Creazione nuovo file temporaneo: {filename}")

        # Prepara header solo se il file non esiste
        is_new_file = not os.path.exists(filename)
        header = None
        if is_new_file:
            header = self._create_header()

        # Copia i buffer e svuota subito (così non perdi dati se arrivano nuovi)
        x_buffer = self.data_x_buffer.copy()
        y_buffer = self.data_y_buffer.copy()
        self.data_x_buffer = np.array([])
        self.data_y_buffer = np.array([])

        # Avvia il worker in un thread separato
        self.save_thread = QThread()
        self.save_worker = VoltageSaveWorker(filename, header, x_buffer, y_buffer, is_new_file)
        self.save_worker.moveToThread(self.save_thread)
        self.save_thread.started.connect(self.save_worker.run)
        self.save_worker.finished.connect(self.save_thread.quit)
        self.save_worker.finished.connect(self.save_worker.deleteLater)
        self.save_thread.finished.connect(self.save_thread.deleteLater)
        self.save_worker.error.connect(lambda msg: print(f"Errore salvataggio: {msg}"))
        self.save_thread.start()

        # Aggiorna il riferimento al file temporaneo solo se stai usando il temporaneo
        if not self._last_saved_file:
            self._last_temp_file = filename
        return filename


    def _create_header(self, header_data=None):
        """Crea un header garantendo il magic number corretto e dimensione 128 byte"""
        if header_data is None: 
            # Calcola data_points escludendo i NaN
            valid_points = len(self.data_x_buffer[~np.isnan(self.data_x_buffer)])
            header = {
                'magic': b'PLANTVOLT',  # 9 byte
                'version': 2.0,         # 4 byte
                'experiment_type': (self.type_of_experiment or 'None')[:20].ljust(20),  # 20 byte
                'sampling_rate': self.sampling_rate,   # 4 byte
                'duration': self.total_elapsed_time,   # 4 byte
                'amplified': self.amplified_data_state, # 1 byte
                'start_time': getattr(self, 'start_datetime', 0.0), # 8 byte
                'end_time': getattr(self, 'end_datetime', 0.0),     # 8 byte
                'data_points': valid_points,           # 4 byte
                'acquisition_count': self._acquisition_count, # 4 byte
                'reserved': b'\x00' * 62              # 62 byte
            }
        else:
            header = header_data

        header_bytes = bytearray()
        header_bytes.extend(header['magic'][:9])  # 9 byte
        header_bytes.extend(struct.pack('<f', header['version']))  # 4
        exp_type = header['experiment_type'].encode('ascii', errors='replace')[:20]
        exp_type += b'\x00' * (20 - len(exp_type))
        header_bytes.extend(exp_type)  # 20
        header_bytes.extend(struct.pack('<f', header['sampling_rate']))  # 4
        header_bytes.extend(struct.pack('<f', header['duration']))  # 4 (espressa in secondi)
        header_bytes.extend(struct.pack('<?', header['amplified']))  # 1
        header_bytes.extend(struct.pack('<d', header['start_time']))  # 8
        header_bytes.extend(struct.pack('<d', header['end_time']))    # 8
        header_bytes.extend(struct.pack('<I', header['data_points']))  # 4
        header_bytes.extend(struct.pack('<I', header['acquisition_count']))  # 4
        header_bytes.extend(header['reserved'][:62])  # 62

        # Verifica dimensione
        if len(header_bytes) != 128:
            raise ValueError(f"Dimensione header errata: {len(header_bytes)} byte (attesi 128)")

        return bytes(header_bytes)


      
    def save_file_action(self, ask_filename=True):
        print("💾 Salvataggio manuale o finale...")
        """Salvataggio manuale o finale: scrive header e TUTTI i dati raccolti (file temporaneo + buffer)"""
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        # --- Selezione file ---
        if ask_filename:
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Voltage Data",
                f"voltage_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pvolt",
                "PlantLeaf Voltage (*.pvolt);;All Files (*)"
            )
            if not filename:
                return False
            if not filename.endswith('.pvolt'):
                filename += '.pvolt'
            self._last_saved_file = filename
            print(f"📁 File definitivo scelto: {filename}")
        else:
            # Salvataggio automatico: usa il file definitivo se esiste
            if self._last_saved_file:
                filename = self._last_saved_file
                print(f"Salvataggio finale in file definitivo {filename}")
            else:
                if self.is_closing and not self.is_cleaning:  # Se la finestra sta chiudendosi, non salvare
                    print("La finestra sta chiudendosi, salvataggio automatico annullato.")
                    return False
                elif not self.is_cleaning:
                    self.save_file_action(ask_filename=True)
                    print("richiedo con salvataggio manuale")
                    return True  # Dopo il salvataggio manuale, esci
                else:
                    print("Stato di pulizia attivo, salvataggio automatico annullato.")
                    return False

        try:
            # Progress dialog
            progress = self.get_progress_widget("Saving Voltage Data...")
            progress.setValue(0)
            progress.show()
            # Ricostruisci tutti i dati dal file temporaneo + buffer attuale
            all_x = []
            all_y = []

            # Scegli il file di origine dati: se hai già un file definitivo, usa quello!
            source_file = None
            if self._last_saved_file and os.path.exists(self._last_saved_file):
                source_file = self._last_saved_file
            elif self._last_temp_file and os.path.exists(self._last_temp_file):
                source_file = self._last_temp_file

            if source_file:
                with open(source_file, 'rb') as f:
                    f.seek(128)  # Salta header
                    while True:
                        chunk = f.read(8)
                        if not chunk or len(chunk) < 8:
                            break
                        x, y = struct.unpack('<ff', chunk)
                        all_x.append(x)
                        all_y.append(y)
            # Aggiungi i dati attualmente nel buffer
            all_x.extend(self.data_x_buffer)
            all_y.extend(self.data_y_buffer)
            print(f"Totale punti da salvare: {len(all_x)}")

            # Scrivi sempre tutti i dati (anche se non è final save)
            with open(filename, 'wb') as f:
                # Calcola i punti validi (escludi NaN)
                valid_points = np.sum(~np.isnan(all_x) & ~np.isnan(all_y))
                header = self._create_header({
                    'magic': b'PLANTVOLT',
                    'version': 2.0,
                    'experiment_type': (self.type_of_experiment or 'None')[:20].ljust(20),
                    'sampling_rate': self.sampling_rate,
                    'duration': self.total_elapsed_time,
                    'amplified': self.amplified_data_state,
                    'start_time': getattr(self, 'start_datetime', 0.0),
                    'end_time': getattr(self, 'end_datetime', 0.0),
                    'data_points': valid_points,
                    'acquisition_count': self._acquisition_count,
                    'reserved': b'\x00' * 62
                })
                f.write(header)
                print(f"Header scritto con punti validi: {valid_points}")


                total_points = len(all_x)
                for i, (x, y) in enumerate(zip(all_x, all_y)):
                    try:
                        if np.isnan(x) or np.isnan(y):
                            f.write(struct.pack('<ff', float('nan'), float('nan')))
                        else:
                            f.write(struct.pack('<ff', x, y))
                    except Exception as e:
                        print(f"Errore nel salvataggio del punto ({x}, {y}): {e}")
                    # Aggiorna progress ogni 500 punti
                    if i % 500 == 0 or i == total_points - 1:
                        percent = int((i + 1) / total_points * 100)
                        progress.setValue(percent)
                        from PySide6.QtWidgets import QApplication
                        QApplication.processEvents()
                        if progress.wasCanceled():
                            print("Salvataggio annullato dall'utente")
                            progress.close()
                            return False

            progress.setValue(100)
            print(f"Dati salvati in: {filename}")
            progress.close()


            # Dopo il salvataggio manuale, cancella il file temporaneo
            if ask_filename and self._last_temp_file and os.path.exists(self._last_temp_file):
                try:
                    os.remove(self._last_temp_file)
                    self._last_temp_file = None
                    print(f"File temporaneo {self._last_temp_file} cancellato con successo.")
                except Exception as e:
                    print(f"⚠️ Impossibile cancellare il file temporaneo: {e}")

            # Svuota i buffer dopo il salvataggio definitivo
            self.data_x_buffer = np.array([])
            self.data_y_buffer = np.array([])

            self.is_cleaning = False  # Reset flag di pulizia

            self._last_temp_file = None  # Resetta il temp file
            self.actionSave.setEnabled(False)  # Disabilita salvataggio multiplo
            self.actionSave.setToolTip(f"File already saved in {filename}")
            self.definetly_saved = True

            return True
        except Exception as e:
            print(f"Errore durante il salvataggio: {e}")
            if 'progress' in locals():
                progress.close()
            QMessageBox.critical(self, "Errore di Salvataggio", f"Impossibile salvare il file:\n{str(e)}")
            return False


    def _finalize_file_data(self, last_saved_file):
        self.save_file_action(ask_filename=False)
        print(f"File finalizzato in: {last_saved_file}")
    ##### 
