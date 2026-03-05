from PySide6.QtCore import QObject, Signal, Slot
import numpy as np
import struct

class VoltageSaveWorker(QObject):
    """
    🔥 WORKER PERSISTENTE per salvataggio voltage data
    Riutilizzabile via segnali invece di creare nuovi thread ogni volta
    """
    # Segnali OUTPUT
    finished = Signal(str)
    error = Signal(str)
    
    # 🔥 NUOVO: Segnale INPUT per richiedere salvataggio
    save_data_signal = Signal(str, object, np.ndarray, np.ndarray, bool)

    def __init__(self, parent=None):
        """
        🔥 NUOVO: Costruttore senza parametri (worker riutilizzabile)
        """
        super().__init__(parent)
        # Connetti il segnale di input allo slot di esecuzione
        self.save_data_signal.connect(self._execute_save)

    @Slot(str, object, np.ndarray, np.ndarray, bool)
    def _execute_save(self, filename, header, x_buffer, y_buffer, is_new_file):
        """
        🔥 Slot per eseguire il salvataggio (chiamato dal segnale)
        """
        try:
            # Scrivi header se necessario
            if is_new_file and header is not None:
                with open(filename, 'wb') as f:
                    f.write(header)
            
            # Scrivi dati in append
            with open(filename, 'ab') as f:
                for x, y in zip(x_buffer, y_buffer):
                    try:
                        if np.isnan(x) or np.isnan(y):
                            f.write(struct.pack('<ff', float('nan'), float('nan')))
                        else:
                            f.write(struct.pack('<ff', x, y))
                    except Exception as e:
                        # Logga ma non blocca il salvataggio
                        print(f"⚠️ Errore salvataggio punto ({x}, {y}): {e}")
            
            self.finished.emit(filename)
        except Exception as e:
            self.error.emit(str(e))

    @Slot()
    def run(self):
        """
        🔥 DEPRECATO: Mantenuto per compatibilità con vecchio codice
        Il nuovo metodo usa _execute_save() via segnali
        """
        # Questo metodo è chiamato solo nel vecchio sistema (creazione thread al volo)
        # Nel nuovo sistema persistente, usiamo _execute_save() via segnali
        pass