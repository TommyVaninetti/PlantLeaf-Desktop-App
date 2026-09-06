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

from PySide6.QtWidgets import (QMessageBox, QFileDialog, QProgressDialog,
                               QApplication, QInputDialog)
from PySide6.QtCore import Signal, QTimer, Qt, QThread, QMetaObject
from PySide6.QtGui import QActionGroup

from core import BaseWindow
from core.wake_lock_manager import WakeLockManager
from .ui.ui_MainWindowAudio import Ui_MainWindowAudio
from core.special_component import replace_widget
from components.start_stop_button import StartStopButton
from components.events_table import (EventsTable, FILTER_MODES, FILTER_LABELS,
                                     FILTER_SURVIVORS)
from core.live_event_worker import LiveEventWorker
from plotting.plot_manager import BasePlotWidget
from components.sampling_settings import AudioSamplingSettingsPopup
from serial_communication.audio_reader import AudioSerialWorker
from saving.audio_save_worker import AudioSaveWorker
from ml import default_model_path
from core import paudio_format as pf

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

        self._setup_event_pipeline()
        self._setup_experiment_menu()
        self._refresh_events_label()

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
        
        # A recording is one event stream: frame indices restart at 0 with
        # !start!, so anything the worker still held from the previous run would
        # be stitched to the wrong frames.
        self.event_worker.reset()
        self._board_stats = None
        self._event_status = {}
        # The metadata log follows the FILE, not the acquisition: several
        # acquisitions append to one temp file, and the footer is positional
        # (record i describes body frame i). frame_idx restarting at 0 is how a
        # reader sees the boundary between them.
        if not (self._last_temp_file and os.path.exists(self._last_temp_file)):
            self._event_meta = []
        self.event_worker.model_path = self.svm_model_path
        self.event_worker.k = self.stage1_k

        # ✅ Solo se tutto OK, procedi
        self.is_acquiring = True
        self._wake_lock.acquire()  # ☀️ Previeni sleep durante acquisizione
        self.start_chronometer()
        self.serial_worker.start(self.threshold_value)  # PASSA LA SOGLIA CORRENTE AL METODO START

        # Disabilita azioni ma verifica se dopo serial_worker.start() è andcora tutto attivo:
        try:
            if self.serial_worker.is_connected:
                # The firmware refuses a mode change while running; disable the
                # control rather than let the user send a command that fails.
                self.FFTClicksDetectorButton.setEnabled(False)
                if hasattr(self, 'actionStage1K'):
                    self.actionStage1K.setEnabled(False)
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

        self.FFTClicksDetectorButton.setEnabled(True)
        if hasattr(self, 'actionStage1K'):
            self.actionStage1K.setEnabled(True)

        # Resolve the candidates still waiting for a neighbour that will now
        # never arrive, and let their rows reach the table before anything
        # exports or saves. Blocking is deliberate: eventReady is a queued
        # signal, so the rows are delivered by processEvents, not inline.
        if self.event_mode:
            QMetaObject.invokeMethod(self.event_worker, 'flush',
                                     Qt.ConnectionType.BlockingQueuedConnection)
            QApplication.processEvents()

        # The board's own counters, read by AudioSerialWorker.stop() while the
        # port was still open. `drop` is the ONLY evidence of frames the ADC
        # queue lost: they never reach Stage 1, so frame_idx does not skip over
        # them and every timestamp derived from it runs early.
        stats = getattr(self.serial_worker, 'last_stats_line', None)
        if stats:
            self._board_stats = stats
            drop = self._parse_stats_field(stats, 'drop')
            if drop:
                print(f"⚠️ Il firmware ha perso {drop} frame: i timestamp "
                      f"derivati da frame_idx anticipano di {drop * 2.56:.0f} ms")
        self._refresh_events_label()
        
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





    #### MODALITA' EVENTI (firmware v3) ####

    #: Stage 1 k. The firmware's own default; its floor is a firmware rule.
    STAGE1_K_DEFAULT = 1.5
    STAGE1_K_MIN = 1.2
    STAGE1_K_MAX = 10.0

    def _setup_event_pipeline(self):
        """
        Build the live pipeline and the thread it runs on.

        The click-detection ON/OFF button is the MODE SWITCH: ON puts the board
        in event mode, where Stage 1 runs on the MCU and only candidates plus
        their neighbours are transmitted; OFF puts it back in full mode, where
        every frame arrives and this window behaves exactly as it always has.
        The firmware refuses either command while recording, so the button is
        disabled while acquiring rather than sending a command it knows will
        come back as '!err busy'.
        """
        self.event_mode = True          # matches the firmware's boot default
        self.stage1_k = self.STAGE1_K_DEFAULT
        self._pending_mode = None       # mode whose ack we are waiting for
        self._board_stats = None        # last '!stats ...' line
        self._event_status = {}
        #: One 21-byte record per transmitted event frame, for the .paudio v4
        #: footer. Kept here rather than in the worker because it must survive
        #: a candidate that fails feature extraction.
        self._event_meta = []

        self.event_worker = LiveEventWorker(
            fs=self.fs, fft_size=self.fft_size,
            model_path=self.svm_model_path, k=self.stage1_k,
        )
        self.event_thread = QThread(self)
        self.event_worker.moveToThread(self.event_thread)
        self.event_thread.started.connect(self.event_worker.start)
        self.event_worker.eventReady.connect(self.on_event_ready)
        self.event_worker.statusChanged.connect(self.on_event_status)
        self.event_worker.error.connect(
            lambda msg: print(f"⚠️ Live pipeline: {msg}"))
        self.event_thread.start()

    def _setup_experiment_menu(self):
        """
        Two controls that have no home in the generated .ui, put in the menu on
        purpose: ui_MainWindowAudio.py is Qt Designer output and anything added
        to it by hand is lost the next time the .ui is regenerated.
        """
        if not hasattr(self, 'menuExperiment'):
            return
        self.menuExperiment.addSeparator()

        self.actionStage1K = self.menuExperiment.addAction("Stage 1 threshold (k)…")
        self.actionStage1K.setToolTip(
            "Multiplier on the adaptive energy floor. Runs on the board.")
        self.actionStage1K.triggered.connect(self.stage1_k_action)

        filter_menu = self.menuExperiment.addMenu("Events shown")
        self._filter_group = QActionGroup(self)
        self._filter_group.setExclusive(True)
        self._filter_actions = {}
        for mode in FILTER_MODES:
            act = filter_menu.addAction(FILTER_LABELS[mode])
            act.setCheckable(True)
            act.setChecked(mode == FILTER_SURVIVORS)
            act.triggered.connect(lambda _checked, m=mode: self.set_event_filter(m))
            self._filter_group.addAction(act)
            self._filter_actions[mode] = act

    # ── board configuration ─────────────────────────────────────────────────

    def _push_board_config(self):
        """
        Make the board agree with the UI. Only legal while stopped — the
        firmware rejects both commands during a recording, which is the
        behaviour this mirrors rather than works around.
        """
        worker = getattr(self, 'serial_worker', None)
        if worker is None or self.is_acquiring:
            return
        self._pending_mode = self.event_mode
        worker.send_command('!eventmode!' if self.event_mode else '!fullmode!')
        if self.event_mode:
            worker.send_command(f'!k {self.stage1_k:.3f}')

    def on_board_reply(self, line):
        """
        One '!' line from the board. These are the only confirmation that a mode
        or k change actually took, so a rejection has to move the UI back rather
        than leave it claiming something the board is not doing.
        """
        print(f"📟 {line}")
        if line.startswith('!ok eventmode') or line.startswith('!ok fullmode'):
            self._pending_mode = None
        elif line.startswith('!err busy'):
            if self._pending_mode is not None:
                self.event_mode = not self._pending_mode
                self._pending_mode = None
                self._sync_mode_button()
                self.show_error_dialog(
                    "Mode locked",
                    "The board refuses to change mode while it is recording.\n"
                    "Stop the recording first.")
        elif line.startswith('!err range'):
            self.show_error_dialog(
                "Value refused",
                f"The board rejected k = {self.stage1_k:.3f}.")
        elif line.startswith('!stats'):
            self._board_stats = line
            self._refresh_events_label()

    def stage1_k_action(self):
        """Stage 1's threshold multiplier. Runs on the board, so it is a command."""
        if self.is_acquiring:
            self.show_error_dialog(
                "Recording",
                "k can only be changed while the recording is stopped.")
            return
        value, ok = QInputDialog.getDouble(
            self, "Stage 1 threshold",
            "Candidate when E_i > k x floor(i).\n"
            f"Minimum {self.STAGE1_K_MIN} (the firmware refuses less).",
            self.stage1_k, self.STAGE1_K_MIN, self.STAGE1_K_MAX, 2)
        if not ok:
            return
        self.stage1_k = float(value)
        self.event_worker.k = self.stage1_k
        self._push_board_config()

    def set_event_filter(self, mode):
        self.FFTClicksDetectedTableWidget.set_filter_mode(mode)
        self._refresh_events_label()

    def _sync_mode_button(self):
        self.clicksDetectionStatus = self.event_mode
        self.FFTClicksDetectorButton.setText("ON" if self.event_mode else "OFF")
        self.FFTClicksDetectorButton.setChecked(self.event_mode)

    # ── the event stream ────────────────────────────────────────────────────

    def on_new_event(self, evt):
        """
        One 802-byte event frame. Candidate AND neighbour alike arrive here —
        the triples are reassembled by the worker, which is the only thing that
        knows when a candidate is decidable.
        """
        if not getattr(self, "is_acquiring", False):
            return

        # Saved exactly like a full-mode frame, so the .paudio body stays
        # byte-identical to v3 and every existing reader keeps working. What
        # makes the file an event recording is the footer, not the body.
        self.data_y_buffer = np.append(self.data_y_buffer, evt['fft_mags'])
        self.data_phase_buffer = np.append(self.data_phase_buffer, evt['phases'])
        self._event_meta.append((evt['frame_idx'], evt['flags'], evt['E_i'],
                                 evt['E_hat_floor'], evt['noise_floor'],
                                 evt['std_noise']))

        # The live spectrum now updates only when something was transmitted,
        # which is the honest thing for it to show in event mode.
        self.data_y_plot = evt['fft_mags'].copy()
        self.plot_needs_update = True

        self.event_worker.submit(evt)

        if len(self.data_y_buffer) >= 15500:
            self.save_fft_data()

    def on_event_ready(self, row):
        """One fully annotated event: Stages 2, 3 and 4 have all had their say."""
        self.FFTClicksDetectedTableWidget.add_event(row)
        self._refresh_events_label()

    def on_event_status(self, status):
        self._event_status = status
        self._refresh_events_label()

    def _refresh_events_label(self):
        """
        The table's own heading carries the counts, so no widget had to be added
        to a generated .ui file to say what the run is doing.
        """
        table = getattr(self, 'FFTClicksDetectedTableWidget', None)
        if table is None:
            return
        if not self.event_mode:
            table_rows = table.rowCount()
            self.FFTClicksDetectedLabel.setText(
                f"Events — {table_rows} (full mode, threshold crossings)")
            return

        st = self._event_status
        parts = [f"Events — {table.visible_count()} shown / "
                 f"{st.get('candidates', 0)} candidates"]

        # Every way data can go missing, named. Silence here would be a lie:
        # two of these three are invisible in the data itself.
        losses = []
        if st.get('board_overflow'):
            losses.append(f"{st['board_overflow']} board overflow")
        if st.get('inbox_dropped'):
            losses.append(f"{st['inbox_dropped']} host backlog")
        if st.get('incomplete'):
            losses.append(f"{st['incomplete']} partial context")
        if table.n_discarded:
            losses.append(f"{table.n_discarded} rows aged out")
        if st.get('failed'):
            losses.append(f"{st['failed']} failed")
        if losses:
            parts.append("lost: " + ", ".join(losses))

        if self._board_stats:
            drop = self._parse_stats_field(self._board_stats, 'drop')
            if drop:
                parts.append(f"board dropped {drop} frames "
                             f"(timestamps run early)")
        self.FFTClicksDetectedLabel.setText("  ·  ".join(parts))

    @staticmethod
    def _parse_stats_field(line, field):
        """Pull one integer out of '!stats frames N drop M ...'."""
        tokens = line.split()
        if field in tokens:
            i = tokens.index(field)
            if i + 1 < len(tokens):
                try:
                    return int(tokens[i + 1])
                except ValueError:
                    return None
        return None

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
        # Full mode only. In event mode the board has already run Stage 1 and
        # the rows come from the real pipeline, so letting this add rows too
        # would mix two different notions of 'event' in one table.
        if self.clicksDetectionStatus and not self.event_mode:
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
        Disegna un evento: iFFT nel tempo, e FFT del frame o della regione.

        Un evento prodotto dalla pipeline live porta con se' `ctx_signal`, cioe'
        la traccia prev|curr|next su cui TUTTE le feature sono state calcolate.
        Quella e' la curva da disegnare, e non una ricostruzione rifatta qui:

          * `peak_abs`, `onset` e `decay_end` sono indici DENTRO quella traccia.
            Ricostruendo il solo frame corrente si otteneva un array tre volte
            piu' corto, e la vecchia riga `int(onset) % self.fft_size` in
            _render_region_spectrum nascondeva il disallineamento invece di
            risolverlo: la finestra di decadimento poteva cadere su campioni
            diversi da quelli misurati.
          * la ricostruzione locale usava `normalize=False`, mentre la pipeline
            misura su `normalize=True` (correzione microfono, 0.55x-1.49x e
            dipendente dalla frequenza). Le due curve non hanno la stessa scala.

        Gli eventi della modalita' full (semplice attraversamento di soglia) non
        hanno ne' traccia ne' spettro: il metodo lo dichiara e pulisce le curve,
        perche' una riga senza waveform deve VEDERSI, non lasciare a schermo
        l'evento precedente.
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

        signal = event.get('ctx_signal')
        if signal is None:
            # Riga senza contesto (import legacy): ricostruisci il solo frame,
            # con la stessa normalizzazione della pipeline.
            from core.click_pipeline_v5 import reconstruct_frame_v5
            result = reconstruct_frame_v5(
                np.asarray(mags), np.asarray(phases),
                fs=self.fs, fft_size=self.fft_size, normalize=True
            )
            if result is None:
                self.IFFTTitleLabel.setText(f"iFFT — {where} · reconstruction failed")
                self.plot_widget_ifft.plot.setData([], [])
                return
            signal = result['signal']

        signal = np.asarray(signal)
        # Centrato sull'evento: t = 0 e' il campione di picco, cosi' eventi
        # diversi sono confrontabili a colpo d'occhio.
        peak_idx = int(np.argmax(np.abs(signal)))
        t_ms = (np.arange(signal.size) - peak_idx) * (1000.0 / self.fs)
        self.plot_widget_ifft.plot.setData(t_ms, signal)

        n_frames = max(1, int(round(signal.size / float(self.fft_size))))
        span = "" if n_frames == 1 else f", {n_frames} frames stitched"
        self.IFFTTitleLabel.setText(f"iFFT — {where} (centred on peak{span})")

        if self.fft_mode == self.FFT_MODE_FRAME:
            self.plot_widget_fft.plot.setData(self.data_x, np.asarray(mags))
        else:
            # Region FFT: lo spettro del frame resta come riferimento grigio.
            self.reference_curve_fft.setData(self.data_x, np.asarray(mags))
            self._render_region_spectrum(event, signal)

    def _render_region_spectrum(self, event, signal):
        """
        Spettro della sola regione del click.

        La finestra arriva dall'evento come `ctx_region` = (onset, decay_end+1),
        gli stessi indici che _feat_v6_spectral usa per misurare: cosi' la curva
        a schermo e i numeri in tabella descrivono gli stessi campioni.
        """
        from core.spectral_analysis import compute_spectrum

        region = event.get('ctx_region')
        if not region:
            self.plot_widget_fft.plot.setData([], [])
            return

        start = max(0, min(int(region[0]), signal.size))
        stop = max(start, min(int(region[1]), signal.size))
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
                'version': self._file_version(),  # 3.0 continuo, 4.0 a eventi
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
                    f.seek(pf.HEADER_SIZE)  # Salta header
                    data = f.read()

                    # Il corpo finisce al PRIMO marker di footer, qualunque
                    # sia: cercare solo CLCK trasformerebbe i byte di EVNT in
                    # magnitudini se il file fosse troncato fra i due.
                    binary_data, _, _ = pf.split_sections(data)
                    
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
                'version': self._file_version(),
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

    def _file_version(self):
        """
        3.0 for a continuous recording, 4.0 for an event one.

        The distinction is not cosmetic: in a v4 file body frame i is NOT the
        i-th frame of the signal. Where it actually sits is in the EVNT footer,
        and a reader that assumes contiguity gets a plausible, wrong answer.
        """
        return (pf.VERSION_EVENT if getattr(self, 'event_mode', False)
                else pf.VERSION_CONTINUOUS)

    def save_click_data(self, audio_filename):
        """
        Append the footers: CLCK always, then EVNT for an event recording.

        ⚠️ In v4 the CLCK block is written EVEN WHEN EMPTY. Every reader in this
        repo finds the end of the body with find(b'CLCK'); without a CLCK block
        they would run past it and decode the EVNT bytes as magnitudes. The
        empty block is what makes the sidecar footer safe for readers that know
        nothing about it.
        """
        try:
            if not hasattr(self, 'FFTClicksDetectedTableWidget'):
                return
            click_data = self.FFTClicksDetectedTableWidget.export_click_data()
            is_event = self._file_version() >= pf.VERSION_EVENT
            meta = getattr(self, '_event_meta', []) if is_event else []

            if not click_data and not is_event:
                print("📊 Nessun click data da integrare")
                return

            if getattr(self, '_click_data_saved', False):
                # Appending a second footer pair would leave the first one
                # inside the body as far as any reader is concerned.
                print("📊 Footer gia' scritto, non lo riscrivo")
                return

            if is_event:
                # The EVNT block is POSITIONAL: record i describes body frame i.
                # If the two counts disagree the file is not readable as an
                # event recording, so say so instead of writing it anyway.
                body = os.path.getsize(audio_filename) - pf.HEADER_SIZE
                n_body = body // pf.FRAME_BYTES
                if n_body != len(meta):
                    print(f"⚠️ EVNT: {len(meta)} record per {n_body} frame nel "
                          f"corpo — il footer non viene scritto")
                    is_event = False

            with open(audio_filename, 'ab') as f:
                f.write(pf.pack_click_footer(click_data))
                if is_event:
                    f.write(pf.pack_event_footer(meta))

            self._click_data_saved = True
            print(f"📊 Click data integrati nel file: {len(click_data)} eventi")
            if is_event:
                print(f"📊 Footer EVNT: {len(meta)} frame trasmessi")

        except Exception as e:
            print(f"⚠️ Errore integrazione click data: {e}")


    def cleanup_resources(self):
        """Stop the live pipeline thread as well as everything BaseWindow knows."""
        try:
            if getattr(self, 'event_thread', None) is not None:
                QMetaObject.invokeMethod(self.event_worker, 'stop',
                                         Qt.ConnectionType.BlockingQueuedConnection)
                self.event_thread.quit()
                if not self.event_thread.wait(2000):
                    self.event_thread.terminate()
                    self.event_thread.wait(500)
        except Exception as e:      # noqa: BLE001
            print(f"⚠️ Chiusura pipeline eventi: {e}")
        super().cleanup_resources()

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
        """
        ON = event mode: Stage 1 runs on the board, only candidates and their
        neighbours are transmitted, and this window runs Stages 2-4 on each one.
        OFF = full mode: every frame arrives and the window behaves exactly as
        it did before the event firmware existed.

        Refused while recording. That is not caution, it is the firmware's own
        rule — it answers '!err busy' — and asking anyway would leave the button
        showing a mode the board is not in.
        """
        if self.is_acquiring:
            self._sync_mode_button()      # undo the click
            self.show_error_dialog(
                "Recording",
                "Click detection can only be switched while the recording is "
                "stopped.")
            return

        self.event_mode = not self.event_mode
        self._sync_mode_button()
        self._push_board_config()
        self._refresh_events_label()
        self.clicks_detector_toggled.emit(self.clicksDetectionStatus)


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
        # 802-byte event frames and the board's '!' replies.
        self.serial_worker.new_event.connect(self.on_new_event)
        self.serial_worker.board_reply.connect(self.on_board_reply)
        
        # ✅ CONNETTE SEGNALI DI DISCONNESSIONE
        try:
            self.serial_worker.error_popup.connect(self.show_serial_error)
            self.serial_worker.serial_connection_status_bool.connect(self.handle_connection_status)
        except Exception as e:
            print(f"Errore connessione funzioni error_popup: {e}")
        
        self.serial_worker.connection()
        # The board boots in event mode; say so explicitly anyway, so the UI and
        # the firmware cannot disagree after a reconnect or a reflash.
        self._push_board_config()
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
