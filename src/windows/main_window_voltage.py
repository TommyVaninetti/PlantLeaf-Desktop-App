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
        self.max_pause_markers = 50  # Limite massimo di pause markers

        # 🔥 CIRCULAR BUFFER per i dati di salvataggio
        self.save_buffer_size = 10000  # Buffer di salvataggio (salvato ogni 1000 campioni)
        self.data_x_buffer = np.zeros(self.save_buffer_size, dtype=np.float64)
        self.data_y_buffer = np.zeros(self.save_buffer_size, dtype=np.float32)
        self.save_buffer_index = 0
        
        # 🔥 CIRCULAR BUFFER per il plot (ottimizzato per memoria)
        self.max_points = 60000  # Ultimi 60000 campioni visibili (~2 minuti a 500 Hz)
        self.plot_buffer_size = self.max_points * 2  # Doppia dimensione per sicurezza
        self.data_x_plot = np.zeros(self.plot_buffer_size, dtype=np.float64)
        self.data_y_plot = np.zeros(self.plot_buffer_size, dtype=np.float32)
        self.plot_buffer_index = 0
        self.plot_buffer_full = False  # Flag per sapere se abbiamo riempito il buffer

        # Inizializza timer con tempo assoluto
        self.total_elapsed_time = 0
        self.chrono_start_time = 0


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

        # 🔥 SETUP WORKER DI SALVATAGGIO PERSISTENTE (Soluzione 3)
        self._setup_persistent_save_worker()

        # mostra a tutto schermo mantenendo le grafiche
        self.showMaximized()



    ##### GESTIONE GRAFICO #####
    
    def _setup_persistent_save_worker(self):
        """Inizializza worker di salvataggio persistente per evitare memory leak"""
        self.save_worker = VoltageSaveWorker()
        self.save_thread = QThread()
        self.save_worker.moveToThread(self.save_thread)
        
        # Connetti segnali
        self.save_worker.finished.connect(self._on_save_finished)
        self.save_worker.error.connect(self._on_save_error)
        
        # Avvia il thread UNA VOLTA (resta attivo per tutta la sessione)
        self.save_thread.start()
        
        print("✅ Persistent save worker initialized")
    
    def _on_save_finished(self, filename):
        """Callback per salvataggio completato"""
        pass  # Silent success
    
    def _on_save_error(self, error_msg):
        """Callback per errore salvataggio"""
        print(f"❌ Save error: {error_msg}")
    
    #connesso a dati raccolti dalla seriale
    def on_new_voltage_data(self, value):
        """
        🔥 OTTIMIZZATO: Usa circular buffer invece di np.append()
        Elimina il memory leak O(n²) con approccio O(1)
        """
        if not getattr(self, "is_acquiring", False):
            return
        
        elapsed = self.total_elapsed_time + (time.time() - self.chrono_start_time)
        value *= float(self.multiplier)
        
        # 🔥 CIRCULAR BUFFER per salvataggio (10000 campioni)
        self.data_x_buffer[self.save_buffer_index] = elapsed
        self.data_y_buffer[self.save_buffer_index] = value
        self.save_buffer_index += 1
        
        # Salva quando raggiungi 1000 campioni
        if self.save_buffer_index >= 1000:
            self.save_voltage_data()
        
        # 🔥 CIRCULAR BUFFER per plot (60000 campioni)
        idx = self.plot_buffer_index % self.plot_buffer_size
        self.data_x_plot[idx] = elapsed
        self.data_y_plot[idx] = value
        self.plot_buffer_index += 1
        
        # Segna quando riempiamo il buffer per la prima volta
        if self.plot_buffer_index >= self.plot_buffer_size:
            self.plot_buffer_full = True
        
        # Aggiorna plot (ogni N campioni per performance)
        if self.plot_buffer_index % 10 == 0:  # Aggiorna ogni 10 campioni
            self.update_plot()
        
        # Aggiorna finestra temporale se abilitata
        if self.time_window_enabled and self.plot_buffer_index % 50 == 0:  # Ogni 50 campioni
            self._update_time_window()
    
    def _update_time_window(self):
        """Aggiorna la finestra temporale del plot"""
        visible_data_x, visible_data_y = self._get_visible_plot_data()
        if len(visible_data_x) > 0:
            xmin = visible_data_x[0]
            self.plot_widget.update_time_window(x_data_seconds=visible_data_x, xmin=xmin)
    
    def _get_visible_plot_data(self):
        """
        🔥 Estrae i dati visibili dal circular buffer
        Gestisce correttamente wrapping e limiti
        Restituisce SEMPRE gli ultimi max_points campioni
        """
        if self.plot_buffer_index == 0:
            return np.array([]), np.array([])
        
        if not self.plot_buffer_full:
            # Buffer non ancora pieno: restituisci gli ultimi max_points (o tutti se meno)
            valid_count = min(self.plot_buffer_index, self.max_points)
            start_idx = max(0, self.plot_buffer_index - valid_count)
            return (self.data_x_plot[start_idx:self.plot_buffer_index].copy(), 
                    self.data_y_plot[start_idx:self.plot_buffer_index].copy())
        else:
            # Buffer pieno: mostra ultimi max_points campioni con wrapping
            start_idx = self.plot_buffer_index % self.plot_buffer_size
            
            # Se start_idx == 0, il buffer è perfettamente pieno
            if start_idx == 0:
                end_slice = self.plot_buffer_size
            else:
                end_slice = start_idx
            
            # Calcola quanti campioni prendere
            count = min(self.max_points, self.plot_buffer_size)
            
            # Estrai gli ultimi count campioni (gestendo wrapping)
            if count <= end_slice:
                # Dati contigui alla fine
                x_data = self.data_x_plot[end_slice - count:end_slice].copy()
                y_data = self.data_y_plot[end_slice - count:end_slice].copy()
            else:
                # Dati wrappati: prendi dalla fine e dall'inizio
                first_part = count - end_slice
                x_data = np.concatenate([
                    self.data_x_plot[-first_part:],
                    self.data_x_plot[:end_slice]
                ])
                y_data = np.concatenate([
                    self.data_y_plot[-first_part:],
                    self.data_y_plot[:end_slice]
                ])
            
            return x_data, y_data

    def update_plot(self):
        """Aggiorna il grafico con i dati dal circular buffer"""
        visible_data_x, visible_data_y = self._get_visible_plot_data()
        
        if len(visible_data_x) > 0:
            self.plot_widget.plot.setData(visible_data_x, visible_data_y)
            
            # Imposta i limiti degli assi SOLO se abbiamo abbastanza dati
            interval = self.max_points / self.sampling_rate
            current_time = self.total_elapsed_time + (time.time() - self.chrono_start_time)
            
            if current_time > interval:
                # Calcola l'intervallo di tempo da visualizzare
                x_max = visible_data_x[-1]
                x_min = x_max - interval
                self.plot_widget.set_axis_limits(x_min=x_min, x_max=x_max, y_min=-1.7, y_max=1.7)


    def setup_chronometer(self):
        from PySide6.QtCore import QTimer
        self.chrono_timer = QTimer(self)
        self.chrono_timer.timeout.connect(self.on_chrono_tick)

    def on_chrono_tick(self):
        """Aggiorna solo la finestra temporale del plot"""
        if self.time_window_enabled:
            self._update_time_window()


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
        """
        🔥 OTTIMIZZATO: Gestione pause markers con limite massimo
        """
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
        
        # 🔥 Aggiungi pause marker CON LIMITE per evitare accumulo
        visible_data_x, _ = self._get_visible_plot_data()
        if len(visible_data_x) > 0:
            pause_x = visible_data_x[-1]
            vline = pg.InfiniteLine(pos=pause_x, angle=90, pen=pg.mkPen('r', width=2, style=Qt.DashLine))
            self.plot_widget.plot_widget.addItem(vline)
            self.pause_markers.append(vline)
            
            # 🔥 LIMITA numero di pause markers (rimuovi i più vecchi)
            if len(self.pause_markers) > self.max_pause_markers:
                old_marker = self.pause_markers.pop(0)
                self.plot_widget.plot_widget.removeItem(old_marker)
                del old_marker  # Explicit cleanup
        
        # Salva dati rimanenti nel buffer
        self.save_voltage_data()

        # Aggiorna vista
        self._update_time_window()
        
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
        """
        🔥 DEPRECATO: Non più necessario con circular buffer
        Il circular buffer gestisce automaticamente il limite di memoria
        """
        pass  # Metodo mantenuto per compatibilità ma non fa nulla




    # SALVATAGGIO DATI
    def save_voltage_data(self):
        """
        🔥 OTTIMIZZATO: Usa worker persistente invece di creare nuovi thread
        Elimina il memory leak da thread accumulation
        """
        import tempfile, os

        # Determina quanti dati salvare dal buffer
        count_to_save = min(self.save_buffer_index, 1000)
        
        if count_to_save == 0:
            return None  # Niente da salvare
        
        # Scegli il file di destinazione
        if self._last_saved_file is not None:
            filename = self._last_saved_file
        else:
            if self._last_temp_file and os.path.dirname(self._last_temp_file) == tempfile.gettempdir():
                filename = self._last_temp_file
            else:
                filename = tempfile.mktemp(prefix='plantvolt_', suffix='.pvolt')
                self._last_temp_file = filename
                print(f"📄 Creazione nuovo file temporaneo: {filename}")

        # Prepara header solo se il file non esiste
        is_new_file = not os.path.exists(filename)
        header = None
        if is_new_file:
            header = self._create_header()

        # 🔥 Copia SOLO i dati validi dal buffer circolare
        x_buffer = self.data_x_buffer[:count_to_save].copy()
        y_buffer = self.data_y_buffer[:count_to_save].copy()
        
        # Reset indice buffer di salvataggio
        self.save_buffer_index = 0

        # 🔥 USA IL WORKER PERSISTENTE (invece di crearne uno nuovo)
        try:
            # Invia richiesta di salvataggio al worker esistente
            self.save_worker.save_data_signal.emit(filename, header, x_buffer, y_buffer, is_new_file)
        except Exception as e:
            print(f"❌ Errore invio dati a save_worker: {e}")

        # Aggiorna il riferimento al file temporaneo solo se stai usando il temporaneo
        if not self._last_saved_file:
            self._last_temp_file = filename
        
        return filename


    def _create_header(self, header_data=None):
        """Crea un header garantendo il magic number corretto e dimensione 128 byte"""
        if header_data is None: 
            # 🔥 Calcola data_points dal buffer circolare
            header = {
                'magic': b'PLANTVOLT',  # 9 byte
                'version': 2.0,         # 4 byte
                'experiment_type': (self.type_of_experiment or 'None')[:20].ljust(20),  # 20 byte
                'sampling_rate': self.sampling_rate,   # 4 byte
                'duration': self.total_elapsed_time,   # 4 byte
                'amplified': self.amplified_data_state, # 1 byte
                'start_time': getattr(self, 'start_datetime', 0.0), # 8 byte
                'end_time': getattr(self, 'end_datetime', 0.0),     # 8 byte
                'data_points': self.save_buffer_index,  # 4 byte (conta buffer corrente)
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
        """
        � OTTIMIZZATO: Salvataggio con supporto circular buffer
        """
        print("💾 Salvataggio manuale o finale...")
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
                if self.is_closing and not self.is_cleaning:
                    print("La finestra sta chiudendosi, salvataggio automatico annullato.")
                    return False
                elif not self.is_cleaning:
                    self.save_file_action(ask_filename=True)
                    print("richiedo con salvataggio manuale")
                    return True
                else:
                    print("Stato di pulizia attivo, salvataggio automatico annullato.")
                    return False

        try:
            # Progress dialog
            progress = self.get_progress_widget("Saving Voltage Data...")
            progress.setValue(0)
            progress.show()
            
            # 🔥 Raccogli TUTTI i dati (file + buffer circolare)
            all_x = []
            all_y = []

            # Scegli il file di origine dati
            source_file = None
            if self._last_saved_file and os.path.exists(self._last_saved_file):
                source_file = self._last_saved_file
            elif self._last_temp_file and os.path.exists(self._last_temp_file):
                source_file = self._last_temp_file

            # Leggi dati esistenti
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
            
            # 🔥 Aggiungi dati dal buffer di salvataggio (solo i validi)
            if self.save_buffer_index > 0:
                all_x.extend(self.data_x_buffer[:self.save_buffer_index])
                all_y.extend(self.data_y_buffer[:self.save_buffer_index])
            
            print(f"💾 Totale punti da salvare: {len(all_x)}")
            progress.setValue(30)

            # Scrivi file completo
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
                progress.setValue(50)

                # Scrivi tutti i dati
                total_points = len(all_x)
                for i, (x, y) in enumerate(zip(all_x, all_y)):
                    try:
                        if np.isnan(x) or np.isnan(y):
                            f.write(struct.pack('<ff', float('nan'), float('nan')))
                        else:
                            f.write(struct.pack('<ff', x, y))
                    except Exception as e:
                        print(f"⚠️ Errore salvataggio punto: {e}")
                    
                    # Update progress
                    if i % 1000 == 0:
                        progress.setValue(50 + int((i / total_points) * 45))
                        from PySide6.QtWidgets import QApplication
                        QApplication.processEvents()
                        if progress.wasCanceled():
                            print("❌ Salvataggio annullato dall'utente")
                            progress.close()
                            return False
            
            progress.setValue(100)
            print(f"✅ Dati salvati in: {filename}")
            progress.close()

            # Cancella file temporaneo se presente
            if ask_filename and self._last_temp_file and os.path.exists(self._last_temp_file):
                try:
                    os.remove(self._last_temp_file)
                    self._last_temp_file = None
                    print(f"🗑️ File temporaneo cancellato")
                except Exception as e:
                    print(f"⚠️ Impossibile cancellare file temporaneo: {e}")

            # 🔥 Reset buffer di salvataggio
            self.save_buffer_index = 0

            self.actionSave.setEnabled(False)
            self.actionSave.setToolTip(f"File already saved in {filename}")
            self.definetly_saved = True

            print(f"✅ File finalizzato: {os.path.basename(filename)} ({valid_points} punti)")
            return True
            
        except Exception as e:
            print(f"❌ Errore durante il salvataggio: {e}")
            if 'progress' in locals():
                progress.close()
            QMessageBox.critical(self, "Errore di Salvataggio", f"Impossibile salvare il file:\n{str(e)}")
            return False


    def _finalize_file_data(self, last_saved_file):
        self.save_file_action(ask_filename=False)
        print(f"File finalizzato in: {last_saved_file}")
    
    def closeEvent(self, event):
        """
        🔥 Override per cleanup thread persistente prima della chiusura
        """
        # Cleanup worker persistente
        if hasattr(self, 'save_worker') and hasattr(self, 'save_thread'):
            try:
                if self.save_thread.isRunning():
                    print("🧹 Fermando save_thread persistente...")
                    self.save_thread.quit()
                    self.save_thread.wait(2000)  # Aspetta max 2 secondi
                    print("✅ Save_thread fermato")
            except Exception as e:
                print(f"⚠️ Errore fermando save_thread: {e}")
        
        # 🔥 Cleanup pause markers per evitare leak di oggetti grafici
        if hasattr(self, 'pause_markers'):
            for marker in self.pause_markers:
                try:
                    self.plot_widget.plot_widget.removeItem(marker)
                except:
                    pass
            self.pause_markers.clear()
            print("🧹 Pause markers puliti")
        
        # Chiama il closeEvent della classe base
        super().closeEvent(event)
    ##### 
