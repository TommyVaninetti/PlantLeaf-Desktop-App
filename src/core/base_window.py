"""
Classe base per tutte le finestre dell'applicazione PlantLeaf
"""
import numpy as np
import os
import tempfile

from PySide6.QtWidgets import QMainWindow, QMessageBox, QProgressDialog
from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt

from .font_manager import FontManager
from .theme_manager import ThemeManager
from .layout_manager import LayoutManager
from .settings_manager import SettingsManager
from .file_handler_mixin import FileHandlerMixin
from config.app_config import AppConfig
from components.choose_serial_port import ChooseSerialPort



class BaseWindow(FileHandlerMixin, QMainWindow):
    """
    Classe base che fornisce funzionalità comuni a tutte le finestre:
    - Gestione font scalabili
    - Gestione temi
    - Gestione layout
    - Gestione impostazioni
    - Gestione azioni toolBar e menuBar
    """
    
    def __init__(self, parent=None):
        self.settings_manager = SettingsManager()
        super().__init__(parent)

        # Inizializza il connettore seriale
        self.serial_worker = None

        self.is_closing = False  # Variabile per tracciare se la finestra sta chiudendo
        self.is_cleaning = False #flag per pulizia esperimento
        self.going_home = False #flag per navigazione home
        self.replay_requested = False #flag per replay esperimento
        self.finally_closing = False  # Flag per chiusura finale
        self.new_file_to_open = False  # Flag per aprire un nuovo file dopo il salvataggio

        # Inizializza manager base

        self.font_manager = FontManager(self.settings_manager.settings)
        self.layout_manager = LayoutManager(self.font_manager)
        self.theme_manager = ThemeManager(self.settings_manager.settings, self.font_manager)

        # Imposta icona della finestra
        self.setWindowIcon(QIcon(AppConfig.LOGO_DIR))

        #INIZIALIZZAZIONE COMPONENTI BASE
        #if hasattr(self, 'actionUpdatedTimeWindow'):
        #    self.time_window_enabled = True  # Variabile per gestire la finestra mobile del grafico
        #    self.actionUpdatedTimeWindow.setToolTip("Disattiva finestra mobile")

        self.grid_visible = True  # Variabile per tenere traccia della visibilità della griglia


        #Inizializza azioni base finestra
        #self.setup_toolbar_actions()
        #self.setup_menubar_actions  #SONO DA CHIAMARE NELLA CLASSE FIGLIA


        # Carica impostazioni salvate per tema etc.
        self._load_saved_settings()


    def _load_saved_settings(self):
        """Carica tutte le impostazioni salvate"""
        saved_font_scale = self.font_manager.load_font_scale()
        self.font_manager.current_font_scale = saved_font_scale
        saved_theme = self.theme_manager.load_saved_theme()
        self.theme_manager.apply_theme(self, saved_theme)




    # METODO UNICO per aggiornare tema e/o font scale 
    def update_style(self, theme_name=None, font_scale=None):
        """Aggiorna tema e/o scala font in modo centralizzato"""
        if font_scale is not None:
            self.font_manager.save_font_scale(font_scale)
            self.font_manager.current_font_scale = font_scale
        if theme_name is not None:
            self.theme_manager.apply_theme(self, theme_name) #APPLICA FONT (con scala dinamica, uso questa funzione perché è collegata)
        else:
            self.theme_manager.apply_theme(self, self.theme_manager.current_theme) #APPLICA SIA FONT CHE TEMA

        # Aggiorna stile del widget grafico se specificato
        if hasattr(self, 'plot_widget'):
            self.update_graph_style(plot_widget_name=self.plot_widget, plot_instance=self.plot_widget.plot)
            print("voltage theme changed")
        elif hasattr(self, 'plot_widget_fft'):
            self.update_graph_style(plot_widget_name=self.plot_widget_fft, plot_instance=self.plot_widget_fft.plot)
            print("audio theme changed")

        self.setStatusBar(None)



    def update_graph_style(self, plot_widget_name=None, plot_instance=None):
        """Aggiorna lo stile del widget grafico specificato"""
        if plot_widget_name is not None and plot_instance is not None:
            self.theme_manager.apply_theme_to_plot(
                plot_widget_name.plot_widget,
                plot_instance
            )
        else:
            print("❌ Nome del widget grafico o istanza non specificati per l'aggiornamento dello stile.")




    #AZIONI TOOLBAR (da chiamare nelle window figlie)
    def setup_toolbar_actions(self):
        """Collega le azioni della toolbar ai rispettivi metodi"""
        actions = [
            (self.actionHome, self.go_to_home),
            (self.actionNewFile, self.new_file_action),
            (self.actionOpenFile, self.open_file_action),
            (self.actionSave, self.save_file_action),
            (self.actionSerialPort, self.serial_port_action),
            (self.actionViewGrid, self.view_grid_action),
            (self.actionSamplingSettings, self.sampling_settings_action),
        ]
        # Collega le azioni della toolbar ai metodi
        for action, slot in actions:
            try:
                action.triggered.disconnect()
            except Exception:
                print(f"azione {action} non trovata")
                pass
            action.triggered.connect(slot)

        # ✅ FIX WINDOWS: Applica icone con stato disabilitato generato automaticamente
        for action, _ in actions:
            icon_path = os.path.join(AppConfig.ICON_DIR, action.objectName() + ".png")
            if os.path.exists(icon_path):
                icon = QIcon(icon_path)
                
                # ✅ Genera automaticamente l'icona disabilitata (semitrasparente)
                from PySide6.QtGui import QPixmap, QPainter
                from PySide6.QtCore import Qt
                
                pixmap = QPixmap(icon_path)
                disabled_pixmap = QPixmap(pixmap.size())
                disabled_pixmap.fill(Qt.transparent)
                
                painter = QPainter(disabled_pixmap)
                painter.setOpacity(0.4)  # 40% di opacità
                painter.drawPixmap(0, 0, pixmap)
                painter.end()
                
                # Aggiungi la versione disabilitata all'icona
                icon.addPixmap(disabled_pixmap, QIcon.Disabled)
                
                action.setIcon(icon)

        ###SEPARATO PER UPDATE TIME WINDOW
        if hasattr(self, 'actionUpdatedTimeWindow'):
            self.actionUpdatedTimeWindow.triggered.connect(self.toggle_time_window)
            
            # ✅ Applica lo stesso fix per questa icona
            icon_path = os.path.join(AppConfig.ICON_DIR, self.actionUpdatedTimeWindow.objectName() + ".png")
            if os.path.exists(icon_path):
                icon = QIcon(icon_path)
                
                from PySide6.QtGui import QPixmap, QPainter
                from PySide6.QtCore import Qt
                
                pixmap = QPixmap(icon_path)
                disabled_pixmap = QPixmap(pixmap.size())
                disabled_pixmap.fill(Qt.transparent)
                
                painter = QPainter(disabled_pixmap)
                painter.setOpacity(0.4)
                painter.drawPixmap(0, 0, pixmap)
                painter.end()
                
                icon.addPixmap(disabled_pixmap, QIcon.Disabled)
                self.actionUpdatedTimeWindow.setIcon(icon)



    #alcune azioni sono anche comuni con menu
    def go_to_home(self):
        # Se l'utente annulla, l'operazione si interrompe.
        self.going_home = True  # Flag per indicare che si sta andando alla home

        if not self.handle_unsaved_data():
            print("❌ Ritorno alla home annullato dall'utente.")
            if not getattr(self, '_pending_close_event', False):
                self.going_home = False
                return
            self.going_home = True  # Continua con la navigazione se c'è un evento di chiusura in sospeso
            return

        # Se l'utente ha scelto "Save" o "Don't Save", procedi.
        if self.isFullScreen():
            self.showNormal()
        
        self._navigate_home()
        print("✅ Navigazione verso la home completata.")

    def _navigate_home(self):
        """Funzione interna per navigare alla home"""
        self.is_closing = True
        print("Navigating to home...")
        
        # ✅ SALVATAGGIO AUTOMATICO SE NECESSARIO
        if getattr(self, '_last_saved_file', None):
            print("💾 Auto-save prima di chiudere")
            #solo per main window voltage
            if hasattr(self, 'plot_widget'):
                self.save_file_action(ask_filename=False)

        # Se si tratta di audio, aspetta che il salvataggio sia completato
        if hasattr(self, "on_stop"):
            self.on_stop()
        if getattr(getattr(self, 'serial_worker', None), 'is_connected', False):
            try:
                self.serial_worker.stop()
            except Exception as e:
                print(f"Errore durante lo stop della seriale: {e}")
        
        from windows.main_window_home import MainWindowHome
        self.home_window = MainWindowHome()
        self.home_window.show()
        self.close()
            


    def new_file_action(self):
        """Crea un nuovo file/esperimento con gestione salvataggio intelligente"""
        print("🆕 New file action triggered")
        # 1. Gestisci eventuali dati non salvati
        if not self.handle_unsaved_data():
            print("❌ Creazione nuovo file annullata dall'utente.")
            # Se c'è un salvataggio asincrono in corso, sospendi l'azione
            if getattr(self, '_pending_close_event', False):
                self.new_file_to_open = True  # Flag per aprire nuovo file dopo il salvataggio
            return

        # 2. Se il salvataggio è asincrono, aspetta il callback
        if getattr(self, '_pending_close_event', False):
            print("⏳ New file sospeso: attendo il termine del salvataggio...")
            self.new_file_to_open = True
            return

        # 3. Determina il tipo di finestra da aprire
        from windows.main_window_voltage import MainWindowVoltage
        from windows.main_window_audio import MainWindowAudio
        window_class = MainWindowVoltage if hasattr(self, 'plot_widget') else MainWindowAudio

        # 4. Crea e mostra la nuova finestra
        new_window = window_class()
        self.layout_manager.center_window_on_screen(new_window)
        new_window.show()
        self.close()

    ######

    def serial_port_action(self):
        self.serial_ports_dialog = ChooseSerialPort(self.theme_manager)
        self.serial_ports_dialog.serial_port_selected.connect(self.on_serial_port_selected)
        self.serial_ports_dialog.exec()


    def view_grid_action(self): #toggle griglia
        """Attiva o disattiva la griglia del grafico"""
        self.grid_visible = not self.grid_visible
        # Aggiorna la visibilità della griglia nel widget grafico
        if hasattr(self, 'plot_widget'):
            # Se il widget grafico è già stato creato, aggiorna la griglia
            self.plot_widget.set_grid(not self.grid_visible)
        elif hasattr(self, 'plot_widget_fft'):
            # Se il widget grafico con dominio temporale è stato creato, aggiorna la griglia
            self.plot_widget_fft.set_grid(not self.grid_visible)        
    
    def sampling_settings_action(self): #sovrascitta del metodo della classe figlia
        print("⚙️ Sampling Settings action triggered")
        pass  # Da implementare

    def toggle_time_window(self):
        self.time_window_enabled = not self.time_window_enabled
        stato = "ATTIVA" if self.time_window_enabled else "DISATTIVA"
        print(f"⏳ Finestra mobile {stato}")
        self.actionUpdatedTimeWindow.setToolTip("Turn on time window update" if not self.time_window_enabled else "Turn off time window update")



    #AZIONI DEL MENU
    def setup_menubar_actions(self):
        """Collega le azioni della menuBar ai rispettivi metodi"""
        actions = [
            (self.actionHome, self.go_to_home),
            (self.actionQuit, self.close),
            (self.actionInfo, self.info_action),
            (self.actionVersion, self.version_action),
            (self.actionWebSite, self.website_action),
            (self.actionLicense, self.license_action),
            (self.actionNewFile, self.new_file_action),
            (self.actionOpenFile, self.open_file_action),
            (self.actionSave, self.save_file_action),
            (self.actionStart, self.start_experiment_action),
            (self.actionClear, self.clear_experiment_action),
            (self.actionReplay, self.replay_experiment_action),
            (self.actionSerialPort, self.serial_port_action),
            (self.actionSamplingSettings, self.sampling_settings_action),
            (self.actionDark, lambda: self.update_style(theme_name="dark.css")),
            (self.actionDark_Green, lambda: self.update_style(theme_name="dark_green.css")),
            (self.actionDark_Blue, lambda: self.update_style(theme_name="dark_blue.css")),
            (self.actionDark_Amber, lambda: self.update_style(theme_name="dark_amber.css")),
            (self.actionLight, lambda: self.update_style(theme_name="light.css")),
            (self.actionLight_Green, lambda: self.update_style(theme_name="light_green.css")),
            (self.actionLight_Blue, lambda: self.update_style(theme_name="light_blue.css")),
            (self.actionLight_Amber, lambda: self.update_style(theme_name="light_amber.css")),
            (self.actionVery_Small, lambda: self.update_style(font_scale=1.15)),
            (self.actionSmall, lambda: self.update_style(font_scale=1.25)),
            (self.actionMedium, lambda: self.update_style(font_scale=1.35)),
            (self.actionLarge, lambda: self.update_style(font_scale=1.4)),
            (self.actionAbout, self.about_action),
            (self.actionDocumentation, self.documentation_action),
        ]
        for action, slot in actions:
            try:
                action.triggered.disconnect()
            except Exception:
                print(f"azione {action} non trovata")
                pass
            action.triggered.connect(slot)
        
        #anche icona di clear
        icon_path = os.path.join(AppConfig.ICON_DIR, self.actionClear.objectName() + ".png")
        self.actionClear.setIcon(QIcon(icon_path))

    def open_file_with_default_app(self,filepath):
        import platform
        import subprocess

        if platform.system() == "Darwin":
            # Forza l'apertura con TextEdit (o altra app di testo)
            subprocess.call(["open", "-a", "TextEdit", filepath])
        elif platform.system() == "Windows":
            os.startfile(filepath)
        else:  # Linux
            subprocess.call(["xdg-open", filepath])

    # AZIONI MENU (SOLAMENTE DEL MENU, NON DELLA TOOLBAR)
    def info_action(self):
        self.open_file_with_default_app(AppConfig.README_PATH)


    def save_file_action(self, ask_filename=True):
        """Metodo da implementare nelle classi figlie per il salvataggio specifico"""
        pass

    def version_action(self):
        """Azione per mostrare la versione dell'applicazione"""
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Version")
        msg.setText(
            "The current version is 1.0.0 (Beta)."
        )
        msg.setStyleSheet("""
            QMessageBox {
                background-color: #fafafa;
                color: #222;
            }
            QLabel {
                color: #222;
            }
            QPushButton {
                background-color: #e0e0e0;
                color: #222;
                border-radius: 6px;
                padding: 6px 12px;
            }
        """)
        msg.exec()

    def website_action(self):
        """Azione per aprire il sito web dell'applicazione"""
        import webbrowser
        webbrowser.open(AppConfig.WEBSITE_URL)
        

    def license_action(self):
        self.open_file_with_default_app(AppConfig.LICENSES_PATH)  # o il file scelto dall'utente

    def start_experiment_action(self): #sovrascitta del metodo della classe figlia (perché varia il nome dei pulsanti)
        #TODO:DA COLLEGARE A PULSANTE START E STOP!!
        pass

    def clear_experiment_action(self):
        """Pulisce l'esperimento corrente, garantendo la finalizzazione dei dati."""
        self.is_cleaning = True

        # 1. Chiedi di salvare SOLO se ci sono dati in un file temporaneo.
        if not self.handle_unsaved_data():
            if not getattr(self, '_pending_close_event', False):
                self.is_cleaning = False # L'utente ha annullato
                print("❌ Pulizia esperimento annullata dall'utente.")
                return
            self.is_cleaning = True  # Continua con la pulizia se c'è un evento di chiusura in sospeso
            return

        # 2. Finalizza il file permanente, se esiste.
        if getattr(self, '_last_saved_file', None):
            print("⚙️ Finalizzazione del file permanente prima della pulizia...")
            self._finalize_file_data(self._last_saved_file)

        # 3. Solo ora, procedi con la pulizia effettiva dei dati.
        self._clear_experiment_data()
        
    def _clear_experiment_data(self): 
        """Pulisce tutti i dati dell'esperimento corrente e resetta lo stato della finestra."""
        # Salva i dati se necessario (solo se c'è un file temporaneo)
        #self.save_file_action(ask_filename=False)
        print("🧹 Pulizia dati esperimento in corso...")

        #imposta la vista all'inizio
        if hasattr(self, 'plot_widget'):
            self.plot_widget.set_x_range(0, 16) # Reset vista asse X con impostazioni di default NON CENTRALIZZATE
        elif hasattr(self, 'plot_widget_time'):
            self.plot_widget_time.set_x_range(0, 16) # Reset vista asse X con impostazioni di default NON CENTRALIZZATE

        # Svuota tutti i buffer di dati (plot e salvataggio)
        if hasattr(self, 'data_x_buffer'):
            self.data_x_buffer = np.array([], dtype=float)
        if hasattr(self, 'data_y_buffer'):
            self.data_y_buffer = np.array([], dtype=float)
        if hasattr(self, 'data_phase_buffer'):
            self.data_phase_buffer = np.array([], dtype=np.int8)
        if hasattr(self, 'data_x_plot'):
            self.data_x_plot = np.array([], dtype=float)
        if hasattr(self, 'data_y_plot'):
            self.data_y_plot = np.array([], dtype=float)

        # Resetta il conteggio e il tempo
        if hasattr(self, '_acquisition_count'):
            self._acquisition_count = 0
        self.total_elapsed_time = 0

        # Pulisci il grafico
        if hasattr(self, 'plot_widget'):
            self.plot_widget.plot.clear()
            self.plot_widget.plot.setData([], [])
            print("grafico pulito")
        elif hasattr(self, 'plot_widget_fft'):
            self.plot_widget_fft.plot.clear()
            self.plot_widget_fft.plot.setData([], [])
        
        #specifico per audio
        if hasattr(self, 'click_active'):
            self.click_active = False
            self.click_start_time = 0
            self.click_peak_frequency = 0
            self.click_peak_amplitude = 0
            self._last_table_update = 0
            if hasattr(self, 'FFTClicksDetectedTableWidget'):
                self.FFTClicksDetectedTableWidget.setRowCount(0)
            
            self.FFTTimePassedLabelTime.setText(f"0:00:00")
            if hasattr(self, 'FFTClicksDetectedTableWidget'):
                self.FFTClicksDetectedTableWidget.setRowCount(0)
            self.FFTTimePassedLabelTime.setText(f"0:00:00")

        # Rimuovi i marker di pausa
        if hasattr(self, "pause_markers"):
            for marker in self.pause_markers:
                self.plot_widget.plot_widget.removeItem(marker)
            self.pause_markers = []

        # Azzera il riferimento al file temporaneo e a file definitivo e crea un nuovo file temporaneo
        self._last_temp_file = None
        self._last_saved_file = None
        self.reset_temp_file()

        # Riabilita i controlli della UI se necessario
        if hasattr(self, "amplifiedDataButton"):
            self.amplifiedDataButton.setEnabled(True)
        if hasattr(self, "actionSamplingSettings"):
            self.actionSamplingSettings.setEnabled(True)
        
        if hasattr(self, '_click_data_saved'):
            self._click_data_saved = False

        self.is_closing = False
        self.is_cleaning = False



    def replay_experiment_action(self):
        print("🔄 Replay action triggered")

        # 1. Se ci sono dati non salvati, gestisci il salvataggio tramite la logica centralizzata
        if self.has_unsaved_data():
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Before Replay Save Data")
            msg.setText("Before replaying, please save your data.")
            msg.setStyleSheet("""
                QMessageBox {
                    background-color: #fafafa;
                    color: #222;
                }
                QLabel {
                    color: #222;
                }
                QPushButton {
                    background-color: #e0e0e0;
                    color: #222;
                    border-radius: 6px;
                    padding: 6px 12px;
                }
            """)
            msg.exec()
            return

        # 2. Se esiste un file salvato e finalizzato, esegui il replay
        if getattr(self, '_last_saved_file', None):
            self.is_closing = True
            self.replay_requested = True
            self.file_path = self._last_saved_file
            self._finalize_file_data(self.file_path)
            self.open_file_action(file_path=self.file_path)
            return

        # 3. Se non c'è nessun file da riprodurre, mostra avviso
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("No file to replay")
        msg.setText("There is no saved experiment file to replay.")
        msg.setStyleSheet("""
            QMessageBox {
                background-color: #fafafa;
                color: #222;
            }
            QLabel {
                color: #222;
            }
            QPushButton {
                background-color: #e0e0e0;
                color: #222;
                border-radius: 6px;
                padding: 6px 12px;
            }
        """)
        msg.exec()


    def about_action(self):
        """Azione per mostrare informazioni sull'applicazione"""
        #esegui lo stesso di documentazione per ora
        self.documentation_action()

    def documentation_action(self):
        """Azione per mostrare la documentazione"""
        #apri repo su github
        import webbrowser
        webbrowser.open(AppConfig.DOCUMENTATION_URL)


    #SOSTITUISCE QWIDGET DELL'UI CON IL WIDGET DEL GRAFICO
    def replace_widget_with_plot(self, widget_name: str, plot_widget_class, *args, **kwargs):
        """
        Sostituisce un QWidget (identificato dal suo nome) con un nuovo widget grafico.
        - widget_name: nome del QWidget da sostituire (come in Qt Designer)
        - plot_widget_class: classe del widget grafico da inserire (es. PlotWidget)
        - *args, **kwargs: argomenti necessari al costruttore del widget grafico (plot_widget_class)
        """
        # Trova il widget da sostituire
        old_widget = getattr(self, widget_name, None) # nome del QWidget .ui
        if old_widget is None:
            print(f"❌ Widget '{widget_name}' non trovato!")
            return

        # Crea il nuovo widget grafico
        plot_widget = plot_widget_class(*args, **kwargs)


        # Trova il layout padre
        parent_layout = old_widget.parentWidget().layout()
        if parent_layout is None:
            print(f"❌ Layout padre non trovato per '{widget_name}'!")
            return

        # Trova la posizione del vecchio widget nel layout
        for i in range(parent_layout.count()):
            if parent_layout.itemAt(i).widget() == old_widget:
                # Rimuovi il vecchio widget
                parent_layout.removeWidget(old_widget)
                old_widget.setParent(None)
                # Inserisci il nuovo widget nella stessa posizione
                parent_layout.insertWidget(i, plot_widget)
                # Aggiorna il riferimento nell'istanza
                setattr(self, widget_name, plot_widget)
                print(f"✅ Sostituito '{widget_name}' con il nuovo widget grafico.")
                return

        print(f"❌ Widget '{widget_name}' non trovato nel layout!")
        


    #FUNZIONI BACKEND

    def has_unsaved_temp_file(self):
        """Versione più robusta del controllo"""
        return (hasattr(self, '_last_temp_file') and 
            self._last_temp_file is not None and 
            os.path.exists(self._last_temp_file) and
            os.path.getsize(self._last_temp_file) > 128)  # Verifica che contenga dati    



    #ALTRE FUNZIONI UTILI
    def closeEvent(self, event):
        print("🚪 Tentativo di chiusura finestra...")

        # 1. Chiedi all'utente se vuole salvare (gestione Save/Don't Save/Cancel)
        if not self.finally_closing:
            if not self.prepare_for_exit():
                if getattr(self, '_pending_close_event', False):
                    print("⏳ Chiusura posticipata fino al completamento del salvataggio...")
                    self.is_closing = True
                else:
                    self.is_closing = False
                print("❌ Uscita annullata, l'acquisizione continua.")
                event.ignore()
                return

        # 2. Se il thread di salvataggio è attivo, blocca la chiusura e aspetta
        if hasattr(self, 'save_thread') and self.save_thread is not None:
            try:
                if self.save_thread.isRunning():
                    print("⏳ Salvataggio in corso, chiusura sospesa.")
                    self._pending_close_event = True
                    event.ignore()
                    return
            except RuntimeError:
                print("⚠️ save_thread già distrutto, ignoro controllo.")

        # 3. Ora è il momento giusto per fermare l'acquisizione e chiudere
        print(f"✅ Uscita confermata, fermo l'acquisizione...")
        self.is_closing = True

        if hasattr(self, 'is_acquiring') and self.is_acquiring:
            self.is_acquiring = False
            self._safe_stop_serial_worker()

        event.accept()
            

    def cleanup_resources(self):
        """Pulizia risorse - non cancella il file se è stato salvato permanentemente"""
        # Pulisci SOLO se è un file temporaneo (in /tmp/ o simile)
        if hasattr(self, '_last_temp_file') and self._last_temp_file:
            if os.path.dirname(self._last_temp_file) == tempfile.gettempdir():
                self.cleanup_temp_files()
            else:
                # Se è un file permanente, solo resetta il conteggio
                self._acquisition_count = 0
                self._last_temp_file = None
        if getattr(getattr(self, 'serial_worker', None), 'is_connected', False):
            self.serial_worker.stop()
            print("🔌 Serial worker stopped")


    def reset_temp_file(self):
        """Crea un nuovo file temporaneo solo se non esiste già un file permanente"""
        if not hasattr(self, '_last_temp_file') or not self._last_temp_file:
            self._last_temp_file = tempfile.mktemp(prefix='plantvolt_', suffix='.bin')
            self._acquisition_count = 0
            print(f"🔄 Creato nuovo file temporaneo: {self._last_temp_file}")

    


    def prepare_for_exit(self):
        """
        Prepara l'uscita con salvataggio finale.
        Ora il salvataggio finale crea sempre un nuovo file con header e dati reali.
        """
        # 1. Chiedi all'utente di salvare SOLO se ci sono dati in un file temporaneo.
        if not self.handle_unsaved_data():
            print("❌ Uscita annullata dall'utente")
            return False # L'utente ha premuto "Cancel"

        # 2. ✅ SOLUZIONE: Esegui la finalizzazione SE ESISTE un file permanente.
        # Questo cattura il caso in cui l'utente ha salvato, continuato a registrare e poi esce.
        if getattr(self, '_last_saved_file', None):
            print("⚙️ Finalizzazione del file permanente prima dell'uscita...")
            self._finalize_file_data(self._last_saved_file)

        # 3. Pulisci le risorse rimanenti (file temporanei, etc.)
        self.cleanup_resources()
        self._save_current_settings()
        return True

    def handle_unsaved_data(self):
        """
        Gestisce il salvataggio quando necessario. Diventa il punto centrale per tutte le decisioni.
        Restituisce True se l'operazione può continuare, False se è stata annullata.
        """
        # Se non ci sono dati non salvati, procedi.
        if not self.has_unsaved_data():
            print("✅ Nessun dato non salvato.")
            self._pending_close_event = False
            return True

        # Chiedi all'utente cosa fare (Save, Don't Save, Cancel)
        result = self._prompt_save_if_needed()

        if result == "cancel":
            print("❌ Operazione annullata dall'utente.")
            self._pending_close_event = False
            return False
        
        elif result == "save":
            # Esegui il salvataggio finale, che ora include il gancio di finalizzazione.
            return self._perform_final_save()
        
        elif result == "dont_save":
            # L'utente non vuole salvare, pulisci i file temporanei e procedi.
            self._pending_close_event = False
            self.cleanup_temp_files()
            return True
            
        return False # Default in caso di problemi
    
    def _perform_final_save(self):
        filename = None
        success = False

        # 1. Se il thread di salvataggio è attivo, blocca la chiusura/pulizia/navigazione
        try:
            if hasattr(self, 'save_thread') and self.save_thread is not None and self.save_thread.isRunning():
                print("⏳ Attendo il termine del salvataggio prima di chiudere/pulire/navigare...")
                self._pending_close_event = True  # Flag per chiusura post-salvataggio
                return False  # Blocca la chiusura per ora
        except Exception as e:
            print(f"⚠️ Errore durante il controllo del salvataggio: {e}")

        # 2. Avvia il salvataggio (se serve)
        if hasattr(self, 'save_file_action'):
            if getattr(self, '_last_saved_file', None):
                print("💾 Salvataggio finale su file esistente")
                filename = self._last_saved_file
                success = self.save_file_action(ask_filename=False)
            elif self.has_unsaved_temp_file():
                print("💾 Salvataggio finale con richiesta filename")
                result = self.save_file_action(ask_filename=True)
                if isinstance(result, str):  # Se restituisce il path del file
                    filename = result
                    success = True
                else:  # Se restituisce un booleano
                    success = result
                print(f"💾 Salvataggio finale completato con successo: {success}, file: {filename}")
            else:
                print("ℹ️ Nessun dato da salvare")
                return True  # Considerato un successo
            
            # se è voltage, ritorna subito il success
            if hasattr(self, 'plot_widget'):
                return success

            # 3. Se il salvataggio è sincrono (non parte un thread), chiama la finalizzazione subito
            # Se invece è asincrono, la finalizzazione e l'azione sospesa saranno gestite nel callback (audio)
            if hasattr(self, 'plot_widget_fft'):
                try:
                    if hasattr(self, 'save_thread') and self.save_thread is not None and self.save_thread.isRunning():
                        print("⏳ Salvataggio asincrono in corso, attendo callback.")
                        self._pending_close_event = True
                        return False
                    else:
                        if success and filename:
                            self._finalize_file_data(filename)
                        return success
                except Exception as e:
                    print(f"⚠️ Errore durante la finalizzazione post-salvataggio: {e}")
                    return False
        return False

    def _finalize_file_data(self, filename):
        """
        Gancio (Hook) per le classi figlie.
        Chiamato dopo il salvataggio finale per aggiungere dati supplementari.
        La classe base non fa nulla.
        """
        print(f"📄 File '{os.path.basename(filename)}' finalizzato. Nessun dato extra da aggiungere dalla classe base.")
        pass
    
    def _prompt_save_if_needed(self):
        """INTERNO: Mostra popup se ci sono dati non salvati"""
        if not self.has_unsaved_data():
            return "dont_save"
            
        from components.not_saved_popup import show_not_saved_popup
        return show_not_saved_popup(self)

    def has_unsaved_data(self):
        """
        Controlla se ci sono dati non salvati che richiedono azione utente.
        La condizione è vera SOLO se esiste un file temporaneo con dei dati.
        """
        return (hasattr(self, '_last_temp_file') and 
                self._last_temp_file is not None and 
                os.path.exists(self._last_temp_file) and
                os.path.getsize(self._last_temp_file) > 128) # Assicurati che non sia vuoto

    def cleanup_temp_files(self):
        """Pulisce SOLO i file temporanei creati dall'applicazione"""
        if hasattr(self, '_last_temp_file') and self._last_temp_file:
            try:
                # Verifica se è un file temporaneo (nella cartella temp)
                if os.path.dirname(self._last_temp_file) == tempfile.gettempdir():
                    if os.path.exists(self._last_temp_file):
                        os.remove(self._last_temp_file)
                        print(f"🧹 File temporaneo rimosso: {self._last_temp_file}")
                # In ogni caso, resetta il riferimento
                self._last_temp_file = None
            except Exception as e:
                print(f"⚠️ Errore durante la pulizia del file temporaneo: {e}")


    def _save_current_settings(self):
        """Salva le impostazioni correnti (tema e font)"""
        if hasattr(self, 'theme_manager'):
            # Salva il tema corrente
            current_theme = getattr(self.theme_manager, 'current_theme', None)
            if current_theme:
                self.theme_manager.save_theme(current_theme)
                print(f"🎨 Tema salvato: {current_theme}")
            
            # Salva la scala del font
            font_scale = getattr(self.font_manager, 'current_font_scale', None)
            if font_scale:
                self.font_manager.save_font_scale(font_scale)
                print(f"🔠 Scala font salvata: {font_scale}")
        
        # Salva altre impostazioni della finestra (posizione, dimensione)
        self.settings_manager.save_window_geometry(self)
        print("💾 Impostazioni finestra salvate")

    
    def show_serial_error(self, port_name=""):
        """Mostra un popup di errore per la disconnessione seriale e ferma i thread."""
        print(f"🚨 Errore sulla porta seriale {port_name}. Tentativo di stop sicuro.")
        
        # Usa il metodo sicuro per fermare il thread
        # Emetti segnale di stop del pulsante start/stop
        try:
            if hasattr(self, 'startStopButton') and self.startStopButton is not None:
                # Chiama stop() che esegue toggle solo se era running;
                # questo emetterà `stopped` e attiverà on_stop collegato.
                self.startStopButton.stop()
                # Assicuriamoci di aggiornare lo stato di acquisizione
                if hasattr(self, 'is_acquiring'):
                    self.is_acquiring = False
        except Exception as e:
            print(f"⚠️ Impossibile fermare tramite pulsante: {e}")
        
        # Riabilita i controlli principali
        if hasattr(self, 'actionSerialPort'):
            self.actionSerialPort.setEnabled(True)
        
        if hasattr(self, 'startStopButton'):
            self.startStopButton.setEnabled(False)
        elif hasattr(self, 'FFTStartStopButton'):
            self.FFTStartStopButton.setEnabled(False)


        # Mostra il messaggio di errore
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Serial Port Error")
        msg.setText(f"The serial port '{port_name}' has been disconnected or an error occurred.")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()

    def _safe_stop_serial_worker(self):
        """
        Metodo centralizzato e sicuro per fermare il thread del serial worker,
        prevenendo crash all'uscita.
        """
        if (hasattr(self, 'serial_worker') and 
            self.serial_worker is not None and
            self.serial_worker.isRunning()):
            
            print("🔄 Fermando il thread seriale in modo sicuro...")
            
            # 1. Chiedi al thread di terminare il suo loop di eventi
            try:
                self.serial_worker.stop() 
                print("⏳ Attendo che il thread seriale termini...")
                self.serial_worker.quit()
            except Exception as e:
                print(f"Errore nella chiusura del thread seriale: {e}")

            # 2. Attendi che il thread sia effettivamente terminato (max 3 secondi)
            if not self.serial_worker.wait(3000):
                print("⚠️ Il thread non ha risposto. Terminazione forzata.")
                self.serial_worker.terminate()
                self.serial_worker.wait(1000) # Breve attesa dopo la terminazione forzata
            else:
                print("✅ Thread seriale fermato correttamente.")
        else:
            print("ℹ️ Nessun thread seriale in esecuzione da fermare.")
    



