import numpy as np
import struct
from PySide6.QtCore import QObject, Signal, Slot
import os

class AudioSaveWorker(QObject):
    """Worker per salvataggio audio - segue pattern voltage ESATTO"""
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, filename, header, y_buffer, phase_buffer, click_data, is_new_file, parent=None):
        super().__init__(parent)
        self.filename = filename
        self.header = header
        self.y_buffer = y_buffer  # magnitudes (float32)
        self.phase_buffer = phase_buffer  # fasi (int8) ✅ NUOVO
        self.click_data = click_data
        self.is_new_file = is_new_file

    @Slot()
    def run(self):
        try:
            if self.is_new_file:
                with open(self.filename, 'wb') as f:
                    f.write(self.header)
                    print("📋 Header scritto per nuovo file")
            
            # ✅ SCRITTURA SEMPLICE SENZA SEPARATORI
            with open(self.filename, 'ab') as f:
                mags = np.array(self.y_buffer, dtype=np.float32)
                phases = np.array(self.phase_buffer, dtype=np.int8)

                if len(mags) != len(phases):
                    raise ValueError(f"Mismatch: {len(mags)} mags vs {len(phases)} phases")
                
                # ✅ SCRIVI TUTTI I CAMPIONI SEQUENZIALMENTE
                for i in range(len(mags)):
                    f.write(struct.pack('<f', mags[i]))
                    f.write(struct.pack('<b', phases[i]))
            
            file_size_mb = os.path.getsize(self.filename) / (1024 * 1024)
            self.finished.emit(f"💾 {len(mags)} campioni salvati ({file_size_mb:.1f} MB)")
            
        except Exception as e:
            self.error.emit(str(e))
                



class AudioSaveActionWorker(QObject):
    finished = Signal()
    error = Signal()
    progress = Signal(int)
    cancelled = Signal()
    
    def __init__(self, filename, header, y_buffer, phase_buffer, click_data, is_new_file, parent=None):
        super().__init__(parent)
        self.filename = filename
        self.header = header
        self.y_buffer = y_buffer
        self.phase_buffer = phase_buffer 
        self.click_data = click_data
        self.is_new_file = is_new_file
        self._cancelled = False
        self.already_emitted = False

    @Slot()
    def run(self):
        try:
            if self.is_new_file:
                with open(self.filename, 'wb') as f:
                    f.write(self.header)
            
            # ✅ SCRITTURA INTERLACCIATA CON PROGRESS
            with open(self.filename, 'ab') as f:
                # Pre-converti a numpy per efficienza
                mags = np.array(self.y_buffer, dtype=np.float32)
                phases = np.array(self.phase_buffer, dtype=np.int8)
                
                # Verifica coerenza
                if len(mags) != len(phases):
                    raise ValueError(f"Mismatch: {len(mags)} mags vs {len(phases)} phases")
                
                total_points = len(mags)
                
                # Interlaccia i dati con progress report
                for i, (mag, phase) in enumerate(zip(mags, phases)):
                    if self._cancelled:
                        return
                    
                    f.write(struct.pack('<f', mag))   # 4 byte
                    f.write(struct.pack('<b', phase)) # 1 byte
                    
                    # Progress ogni 500 campioni o alla fine
                    if i % 500 == 0 or i == total_points - 1:
                        percent = int((i + 1) / total_points * 100)
                        self.progress.emit(percent)
            
            num_ffts = len(mags) // 154
            print(f"💾 Salvate {num_ffts} FFT in {self.filename}")
            self.finished.emit()
            print("✅ Salvataggio completato.")
            self.already_emitted = True
            
        except Exception as e:
            print(f"❌ Errore salvataggio: {e}")
            self.error.emit()

    def cancel(self):
        if self.already_emitted:
            return
        print("❌ Salvataggio annullato.")
        self._cancelled = True
        self.cancelled.emit()