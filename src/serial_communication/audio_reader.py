import serial
import numpy as np
from PySide6.QtCore import Signal
from PySide6 import QtCore
import struct


class AudioSerialWorker(QtCore.QThread):
    new_data = Signal(np.ndarray, np.ndarray, float, int, bool, float)
    # ↑ magnitudes, ↑ phases, max_amp, peak_bin, above_threshold, threshold

    serial_connection_status_bool = Signal(bool)  # <--- aggiungi questo segnale
    error_popup = Signal(str)  # aggiungi questo segnale


    def __init__(self, serial_port):
        super().__init__()
        self.serial_port = serial_port
        self.is_connected = True
        self.is_running = False
        self._already_disconnected = False
        self._stopped_by_user = False



    def connection(self):
        try:
            print(f"Tentativo di apertura porta seriale: {self.serial_port}")
            self.ser = serial.Serial(self.serial_port, baudrate=115200) #la connessione è VIRTUAL COM quindi non imposta il baudrate
            self.is_connected = True
            self.serial_connection_status_bool.emit(self.is_connected)  # <--- emetti il segnale quando la porta si disconnette
            print(f"🔌 Connessione seriale avvenuta su {self.serial_port}")

        except serial.SerialException as e:
            print(f"Errore apertura seriale: {e}")
            self.handle_disconnection()
            return



    def run(self):
        if not self.is_connected:
            return

        try:
            while self.is_running:
                # ✅ CONTROLLA SE LA PORTA È ANCORA APERTA
                if not self.ser.is_open:
                    print("⚠️ Porta seriale chiusa durante il loop.")
                    break

                # Leggi header (4 byte)
                header = self.ser.read(4)
                if len(header) < 4:
                    continue

                if header[0] != 0xAA or header[1] != 0x55:
                    continue

                # Lunghezza payload
                payload_length = struct.unpack('<H', header[2:4])[0]
                

                # Leggi payload
                packet_data = self.ser.read(payload_length)

                # Verifica dimensione effettiva
                if len(packet_data) != payload_length:
                    continue

                offset = 0

                # === METADATA (11 byte) ===
                max_amplitude = struct.unpack('<f', packet_data[offset:offset+4])[0]
                offset += 4
                peak_bin = struct.unpack('<H', packet_data[offset:offset+2])[0]
                offset += 2
                above_threshold = bool(struct.unpack('<B', packet_data[offset:offset+1])[0])
                offset += 1
                current_threshold = struct.unpack('<f', packet_data[offset:offset+4])[0]
                offset += 4

                # === MAGNITUDINI FFT (154 bin × 4 byte = 616 byte) ===
                num_bins = 154  # ✅ FISSO per 20-80 kHz
                fft_bytes_size = num_bins * 4  # 616 byte
                
                if len(packet_data) - offset < fft_bytes_size:
                    continue

                fft_data_bytes = packet_data[offset:offset + fft_bytes_size]
                
                if len(fft_data_bytes) % 4 != 0:
                    continue
                
                fft_data = np.frombuffer(fft_data_bytes, dtype=np.float32)
                offset += fft_bytes_size

                # === FASI FFT (154 bin × 1 byte = 154 byte) ===
                phase_bytes_size = num_bins  # 154 byte
                
                if len(packet_data) - offset < phase_bytes_size:
                    continue

                phase_data_bytes = packet_data[offset:offset + phase_bytes_size]
                fft_phases = np.frombuffer(phase_data_bytes, dtype=np.int8)
                offset += phase_bytes_size

                # Emetti dati
                self.new_data.emit(fft_data, fft_phases, max_amplitude, peak_bin, 
                                above_threshold, current_threshold)

        except serial.SerialException as e:
            # ✅ ERRORE SERIALE (porta disconnessa fisicamente)
            if not self._stopped_by_user:
                print(f"❌ Errore seriale (porta disconnessa): {e}")
                self.handle_disconnection()
        
        except OSError as e:
            # ✅ BAD FILE DESCRIPTOR (porta già chiusa)
            if e.errno == 9:  # Bad file descriptor
                print("⚠️ Porta già chiusa (bad file descriptor), ignoro.")
            else:
                print(f"❌ Errore OS generico: {e}")
                if not self._stopped_by_user:
                    self.handle_disconnection()
        
        except Exception as e:
            print(f"❌ Errore generico nel thread seriale: {e}")
            if not self._stopped_by_user:
                self.handle_disconnection()


    #handle per porta che viene disconnessa
    def handle_disconnection(self):
        if self._already_disconnected:
            return
            
        print("🔌 Gestione disconnessione...")
        self.is_running = False
        self.is_connected = False
        
        # ✅ CHIUDI PORTA SERIALE SOLO SE ANCORA APERTA
        if hasattr(self, 'ser') and self.ser is not None:
            try:
                if self.ser.is_open:
                    self.ser.write(b"!stop!\n")
                    self.ser.close()
                    print("✅ Porta chiusa in seguito a disconnessione.")
            except Exception as e:
                print(f"⚠️ Porta già chiusa o non disponibile: {e}")
        
        self._already_disconnected = True
        
        # ✅ SEGNALI FINALI (ordine importante!)
        self.serial_connection_status_bool.emit(False)
        self.error_popup.emit(self.serial_port)
        print(f"🔌 Disconnessione dalla porta seriale {self.serial_port} avvenuta.")


    def start(self, current_threshold=0.03):
        """Avvia l'acquisizione, riaprendo la porta se necessario."""
        if not self.is_running:
            try:
                # ✅ RIAPRI LA PORTA SE È STATA CHIUSA
                if not self.ser.is_open:
                    self.ser.open()
                    print("✅ Porta seriale riaperta.")

                self.is_running = True
                self._already_disconnected = False
                
                # 1. INVIA LA SOGLIA ATTUALE PRIMA DI AVVIARE
                threshold_cmd = f"!threshold {current_threshold:.3f}\n".encode('utf-8')
                self.ser.write(threshold_cmd)
                print(f"📡 Soglia {current_threshold:.3f}V inviata prima dello start.")
                
                # 2. INVIA IL COMANDO DI START
                self.ser.write(b"!start!\n")

                print(f"🔌 Connessione seriale avviata su {self.serial_port}")
                super().start()
                print("serial thread started")
            except Exception as e:
                print(f"Errore scrittura su seriale in start: {e}")
                self.handle_disconnection()
  
  

    def stop(self):
        """✅ STOP SICURO DEL THREAD con chiusura della porta."""
        print("🔄 Fermando AudioSerialWorker...")
        
        # Ferma il loop principale
        self.is_running = False
        self._stopped_by_user = True

        # Invia comandi di stop al dispositivo
        if hasattr(self, 'ser') and self.ser.is_open and self._already_disconnected == False:
            try:
                #resetta la threshold sul micro a 0.08
                self.ser.write(b"!threshold 0.08\n")
                self.ser.write(b"!stop!\n")
                print("✅ Comandi di stop e reset inviati.")
                
                # ✅ CHIUDI LA PORTA SERIALE
                self.ser.close()
                print("✅ Porta seriale chiusa.")

            except Exception as e:
                print(f"⚠️ Errore durante lo stop: {e}")
        
        # NON impostare is_connected a False, la porta è solo chiusa, non persa.
        # Segnala che la porta è stata chiusa correttamente dall'utente
        if not self._already_disconnected:
            self.serial_connection_status_bool.emit(True)  # Porta ancora valida
        self._already_disconnected = True  # Evita doppia gestione disconnessione
