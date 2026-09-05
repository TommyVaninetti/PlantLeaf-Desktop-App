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

"""
Finestra principale per il monitoraggio Audio
"""

import os
import tempfile
import struct
from pathlib import Path

from PySide6.QtWidgets import QMessageBox, QFileDialog, QProgressDialog, QApplication
from PySide6.QtCore import Signal, QTimer, Qt, QThread

from core import BaseWindow
from core.wake_lock_manager import WakeLockManager
from .ui.ui_MainWindowAudio import Ui_MainWindowAudio
from core.special_component import replace_widget
from components.start_stop_button import StartStopButton
from components.events_table import EventsTable
from plotting.plot_manager import BasePlotWidget
from components.sampling_settings import AudioSamplingSettingsPopup
from serial_communication.audio_reader import AudioSerialWorker
from saving.audio_save_worker import AudioSaveWorker
from ml import default_model_path

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

        # Gli splitter (mainSplitter / graphsSplitter) arrivano gia' montati dal
        # file .ui: non c'e' piu' nessuna chirurgia di layout a runtime.
        # replace_widget sa gia' sostituire un widget figlio di uno QSplitter.

        self._setup_ui()

        # Sostituzione tabelle
        customTableFFT = EventsTable(
            self.theme_manager,
            parent=self,
            settings_manager=self.settings_manager,
        )

        replace_widget(self, "FFTClicksDetectedTableWidget", customTableFFT)

        self.FFTClicksDetectedTableWidget = customTableFFT

        # ⚠️ DOPO la sostituzione: prima di questa riga l'attributo e' ancora il
        # QTableWidget segnaposto del .ui, che non ha il segnale eventSelected.
        self.FFTClicksDetectedTableWidget.eventSelected.connect(self.on_event_selected)

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

        # Curva di riferimento per la modalita' Region FFT: lo spettro dell'intero
        # frame resta visibile in grigio tratteggiato sotto quello della regione,
        # esattamente come nel RegionFFTDialog.
        self.reference_curve_fft = self.plot_widget_fft.plot_widget.plot(
            name="Frame FFT (transmitted)",
            pen={'color': '#888888', 'width': 1, 'style': Qt.PenStyle.DashLine}
        )
        self.reference_curve_fft.setVisible(False)

        #iFFT (ricostruzione nel tempo dell'evento, centrata sul picco)
        custom_plot_ifft = BasePlotWidget(
            x_label="Time",
            y_label="Amplitude",
            x_range=(-1.28, 1.28),
            y_range=(-0.05, 0.05),
            unit_x="ms", unit_y="V",
            parent=self
        )
        replace_widget(self, "IFFTPlotWidget", custom_plot_ifft)
        self.plot_widget_ifft = custom_plot_ifft

        # Asse normale, NON TimeAxisItem: quello formatta H:MM:SS.ss, illeggibile
        # su un frame da 2.56 ms.
        self.plot_widget_ifft.plot = self.plot_widget_ifft.plot_widget.plot(name="iFFT")

        self.setWindowTitle("Audio Monitor")

        self.layout_manager.center_window_on_screen(self)

        self.setup_toolbar_actions()
        self.setup_menubar_actions()

        self.theme_manager.apply_theme_to_plot(self.plot_widget_fft.plot_widget, self.plot_widget_fft.plot)
        self.theme_manager.apply_theme_to_plot(self.plot_widget_ifft.plot_widget, self.plot_widget_ifft.plot)

        #riapplica modifiche tema, font, layout
        self.layout_manager.adjust_window_size_for_content(self)

        self.type_of_experiment = "Test"  # Default, può essere modificato in sampling_settings

        #richiamo il sistemafont
        self._load_saved_settings()

        self.setStatusBar(None)  # Disabilita la status bar predefinita

        # Imposta l'azione di avvio dell'esperimento come disattivata e anche i pulsanti di start/stop
        self.FFTStartStopButton.setEnabled(False)
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

        # X axis = the true FFT bin center frequencies transmitted by the
        # firmware: (bin_start + k) * bin_freq for k = 0..num_bins-1, i.e.
        # 19921.875 .. 79687.5 Hz in 390.625 Hz steps. A linspace between
        # the nominal 20 kHz / 80 kHz band edges would skew every label by
        # up to ~312 Hz at the top of the band, because the true bin grid
        # neither starts at exactly 20 kHz nor is spaced 60 kHz/153.
        self.data_x = np.arange(bin_start, bin_start + num_bins) * bin_freq

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

        # Modello SVM in uso. Solo il PERCORSO all'avvio: joblib.load costa ~100 ms
        # e finche' nessuno classifica non serve a niente caricarlo.
        self.svm_model_path = default_model_path()   # ml/v6/plantleaf_svm_v6_DEPLOYED.pkl
        self.svm_model = None
        self._update_svm_action_tooltip()

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

        # Defense in depth: peak_bin comes from the serial stream. The reader
        # validates frame framing, but a corrupted frame must never be able to
        # crash the GUI thread with an IndexError here - drop it instead.
        if not (0 <= peak_bin < len(self.data_x)):
            return

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
            # TRANSITORIO — questo ramo esiste solo finche' il firmware attuale
            # continua a mandare TUTTI i frame e la rilevazione e' un semplice
            # attraversamento di soglia. Riempie le sole chiavi dello schema v6
            # che a questo stadio sono effettivamente note, cosi' la tabella e'
            # gia' popolata e navigabile mentre il firmware a eventi viene
            # scritto. Il reader a eventi sostituira' tutto questo con l'evento
            # completo di feature, fft_mags e phases.
            self.FFTClicksDetectedTableWidget.add_event({
                'timestamp_s': self.relative_timestamp,
                'FPE_hz': self.click_peak_frequency,
                'label': '',
                'note': '',
                # Chiavi fuori schema, lette solo da export_click_data per tenere
                # in vita il blocco CLCK dei file .paudio gia' salvati.
                'peak_amplitude_v': self.click_peak_amplitude,
                'duration_us': int(self.click_fft_count * self.fft_interval_us),
            })

            # Reset contatore per prossimo click
            self.click_active = False
            self.click_fft_count = 0


    # GRAFICO FFT - AGGIORNATO SOLO DAL TIMER
    def update_plot(self):
        """Aggiorna il plot solo se necessario (chiamato dal timer a 60Hz)"""
        # In Region FFT il grafico appartiene all'evento selezionato: il flusso
        # live non deve sovrascriverlo ad ogni frame.
        if self.fft_mode != self.FFT_MODE_FRAME:
            return
        if self.plot_needs_update and len(self.data_y_plot) > 0:
            self.plot_widget_fft.plot.setData(self.data_x, self.data_y_plot)
            self.plot_needs_update = False




    #### EVENTI: SELEZIONE, GRAFICI, MODELLO SVM ####

    #: Indici del FFTModeComboBox. Nominati perche' compaiono in tre posti.
    FFT_MODE_FRAME = 0
    FFT_MODE_REGION = 1

    def on_event_selected(self, row):
        """Una riga della tabella eventi e' stata selezionata: ridisegna i grafici."""
        event = self.FFTClicksDetectedTableWidget.event_at(row)
        if event is None:
            self.IFFTTitleLabel.setText("iFFT — no event")
            return
        self._render_event(event)

    def on_fft_mode_changed(self, index):
        """Frame FFT (spettro trasmesso) vs Region FFT (spettro del solo click)."""
        self.fft_mode = index
        self.reference_curve_fft.setVisible(index == self.FFT_MODE_REGION)

        event = self.FFTClicksDetectedTableWidget.current_event()
        if event is not None:
            self._render_event(event)
        elif index == self.FFT_MODE_FRAME:
            # Nessun evento selezionato: torna semplicemente al flusso live.
            self.plot_needs_update = True
            self.update_plot()

    def _render_event(self, event):
        """
        ⚠️ QUESTO E' IL PUNTO DI AGGANCIO DEL BACKEND A EVENTI.

        Si aspetta due chiavi extra sull'evento, che OGGI NESSUNO SCRIVE:

            event['fft_mags']  ndarray, le magnitudini del frame trasmesso
            event['phases']    ndarray int8, le fasi quantizzate

        Quando ci saranno, la ricostruzione nel tempo e' gia' scritta e testata:

            core.click_pipeline_v5.reconstruct_frame_v5(mags, phases,
                fs=self.fs, fft_size=self.fft_size, normalize=False)['signal']

        e lo spettro della regione si ottiene con
        core.spectral_analysis.compute_spectrum() sulla finestra di decadimento.

        Finche' mancano, il metodo dichiara esplicitamente che non c'e' forma
        d'onda e pulisce le curve: una riga senza waveform deve VEDERSI, non
        lasciare a schermo l'evento precedente.
        """
        mags = event.get('fft_mags')
        phases = event.get('phases')
        frame_idx = event.get('frame_idx')
        where = f"frame {frame_idx}" if frame_idx not in (None, '') else \
                f"t={event.get('timestamp_s', 0):.2f}s"

        if mags is None or phases is None:
            self.IFFTTitleLabel.setText(f"iFFT — {where} · no waveform yet")
            self.plot_widget_ifft.plot.setData([], [])
            if self.fft_mode == self.FFT_MODE_REGION:
                self.plot_widget_fft.plot.setData([], [])
                self.reference_curve_fft.setData([], [])
            return

        # --- da qui in poi: codice che il backend rendera' raggiungibile ---
        from core.click_pipeline_v5 import reconstruct_frame_v5

        result = reconstruct_frame_v5(
            np.asarray(mags), np.asarray(phases),
            fs=self.fs, fft_size=self.fft_size, normalize=False
        )
        if result is None:
            self.IFFTTitleLabel.setText(f"iFFT — {where} · reconstruction failed")
            self.plot_widget_ifft.plot.setData([], [])
            return

        signal = np.asarray(result['signal'])
        # Centrato sull'evento: t = 0 e' il campione di picco, cosi' eventi
        # diversi sono confrontabili a colpo d'occhio.
        peak_idx = int(np.argmax(np.abs(signal)))
        t_ms = (np.arange(signal.size) - peak_idx) * (1000.0 / self.fs)
        self.plot_widget_ifft.plot.setData(t_ms, signal)
        self.IFFTTitleLabel.setText(f"iFFT — {where} (centred on peak)")

        if self.fft_mode == self.FFT_MODE_FRAME:
            self.plot_widget_fft.plot.setData(self.data_x, np.asarray(mags))
        else:
            # Region FFT: lo spettro del frame resta come riferimento grigio.
            self.reference_curve_fft.setData(self.data_x, np.asarray(mags))
            self._render_region_spectrum(event, signal)

    def _render_region_spectrum(self, event, signal):
        """
        Spettro della sola regione del click. Backend seam: serve la finestra di
        decadimento (onset, decay_len) che oggi nessun evento porta con se'.
        """
        from core.spectral_analysis import compute_spectrum

        onset = event.get('peak_abs')
        decay_len = event.get('decay_len')
        if onset in (None, '') or decay_len in (None, ''):
            self.plot_widget_fft.plot.setData([], [])
            return

        start = max(0, int(onset) % self.fft_size)
        stop = min(signal.size, start + int(decay_len))
        segment = signal[start:stop]
        if segment.size < 4:
            self.plot_widget_fft.plot.setData([], [])
            return

        spec = compute_spectrum(segment, self.fs)
        self.plot_widget_fft.plot.setData(spec.freqs, spec.mags)

    def _update_svm_action_tooltip(self, model=None):
        """Mostra sull'azione quale .pkl e' in uso, e cosa contiene se caricato."""
        if not hasattr(self, 'actionSVM'):
            return
        name = self.svm_model_path.name if self.svm_model_path else "none"
        if model is None:
            self.actionSVM.setToolTip(f"SVM model: {name} (not loaded yet)")
            return
        try:
            summary = (f"kernel={model['kernel']} "
                       f"thr={model['threshold']:.3f} "
                       f"feat={len(model['features'])}")
        except Exception:
            summary = "loaded"
        self.actionSVM.setToolTip(f"SVM model: {name} — {summary}")

    def svm_model_action(self):
        """Sceglie il .pkl con cui classificare. Stesso flusso del model browser
        del Data Collection dialog, cosi' i due si comportano allo stesso modo."""
        from core.click_pipeline_v5 import load_svm_model

        start_dir = str(self.svm_model_path.parent) if self.svm_model_path \
            else self.settings_manager.get_last_directory("svm_model")

        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select SVM Model", start_dir, "SVM model (*.pkl)"
        )
        if not filepath:
            return

        try:
            # joblib, non pickle: il modello contiene buffer numpy grezzi.
            model = load_svm_model(Path(filepath))
        except Exception as e:
            # La selezione precedente resta valida: un modello illeggibile non
            # deve lasciare la finestra senza modello.
            print(f"❌ Modello SVM non caricabile: {e}")
            self.show_error_dialog("Model Error", f"Cannot load SVM model:\n{e}")
            return

        self.svm_model_path = Path(filepath)
        self.svm_model = model
        self.settings_manager.set_last_directory("svm_model", filepath)
        self._update_svm_action_tooltip(model)
        print(f"🧠 Modello SVM selezionato: {self.svm_model_path.name}")

    def reset_svm_model_action(self):
        """Torna al modello v6 distribuito con l'app (ml/__init__.default_model_path)."""
        self.svm_model_path = default_model_path()
        self.svm_model = None
        self._update_svm_action_tooltip()
        print(f"🧠 Modello SVM ripristinato: {self.svm_model_path.name}")




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
            start_dir = self.settings_manager.get_last_directory("save_audio")
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Audio Data",
                os.path.join(start_dir, f"audio_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.paudio"),
                "PlantLeaf Audio (*.paudio);;All Files (*)"
            )
            if not filename:
                return False
            if not filename.endswith('.paudio'):
                filename += '.paudio'
            self._last_saved_file = filename
            self.settings_manager.set_last_directory("save_audio", filename)
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


    def set_fft_threshold(self, value):
        """
        Aggiorna la soglia: linea rossa sul plot + comando al microcontrollore.

        Non c'e' piu' uno SpinBox che la chiami — resta perche' il valore viaggia
        ancora nel protocollo seriale e nell'header del file, e perche' il nuovo
        firmware avra' comunque bisogno di riceverlo una volta.
        """
        print(f"🎚️ Threshold impostata a: {value:.3f}V")

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

        # La soglia non e' piu' regolabile dalla UI: con il nuovo firmware lo
        # Stage 1 gira a bordo e il valore qui serve solo come riferimento (linea
        # rossa sul plot, campo `threshold` dell'header .paudio, argomento di
        # serial_worker.start). Il comando seriale !threshold resta disponibile.
        self.threshold_value = 0.03

        # Frame FFT (quello trasmesso) vs Region FFT (solo il click). L'indice 0
        # e' la modalita' live di sempre.
        self.fft_mode = self.FFT_MODE_FRAME
        self.FFTModeComboBox.setCurrentIndex(self.FFT_MODE_FRAME)
        self.FFTModeComboBox.currentIndexChanged.connect(self.on_fft_mode_changed)

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
        
        # Il tooltip lo imposta EventsTable.setup_table: descrive labelling,
        # navigazione e menu delle colonne, che questa finestra non conosce.

    def toggle_clicks_detection(self):
        #CAMBIA STILE E ANCHE STATO
        self.clicksDetectionStatus = not self.clicksDetectionStatus
        button_text = "ON" if self.clicksDetectionStatus else "OFF"
        self.FFTClicksDetectorButton.setText(button_text)
        self.FFTClicksDetectorButton.setChecked(self.clicksDetectionStatus)
        #self.FFTClicksDetectorButton.setStyleSheet(self.theme_manager.get_toggle_button_style(self.clicksDetectionStatus))
        self.clicks_detector_toggled.emit(self.clicksDetectionStatus)
        pass


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
