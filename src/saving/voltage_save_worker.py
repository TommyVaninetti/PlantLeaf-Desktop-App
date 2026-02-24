from PySide6.QtCore import QObject, Signal, Slot
import numpy as np
import struct

class VoltageSaveWorker(QObject):
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, filename, header, x_buffer, y_buffer, is_new_file, parent=None):
        super().__init__(parent)
        self.filename = filename
        self.header = header
        self.x_buffer = x_buffer
        self.y_buffer = y_buffer
        self.is_new_file = is_new_file

    @Slot()
    def run(self):
        try:
            # Scrivi header se necessario
            if self.is_new_file:
                with open(self.filename, 'wb') as f:
                    f.write(self.header)
                    print("Salvato Header iniziale")
            # Scrivi dati in append
            with open(self.filename, 'ab') as f:
                for x, y in zip(self.x_buffer, self.y_buffer):
                    try:
                        if np.isnan(x) or np.isnan(y):
                            f.write(struct.pack('<ff', float('nan'), float('nan')))
                            print("Salvato NaN")
                        else:
                            f.write(struct.pack('<ff', x, y))
                    except Exception as e:
                        # Logga ma non blocca il salvataggio
                        print(f"Errore nel salvataggio del punto ({x}, {y}): {e}")
            self.finished.emit(self.filename)
        except Exception as e:
            self.error.emit(str(e))