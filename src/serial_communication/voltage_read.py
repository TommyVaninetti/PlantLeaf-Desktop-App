import serial
import numpy as np
from PySide6.QtCore import Signal
from PySide6 import QtCore

class VoltageSerialWorker(QtCore.QThread):
    new_data = Signal(np.ndarray)  # Emesso per ogni valore
    stability_status_changed = Signal(str)   # Nuovo segnale per lo stato della connessione
    serial_connection_status_bool = Signal(bool)  # <--- aggiungi questo segnale
    error_popup = Signal(str)  # aggiungi questo segnale


    def __init__(self, serial_port, sampling_value):
        super().__init__()
        self.serial_port = serial_port #esempio: "/dev/ttyUSB0" o "COM3"
        self.sampling_value = sampling_value
        self.is_connected = True
        self.is_running = False
        self._block_counter = 0
        self._values_in_block = 0
        self._already_disconnected = False
        self._stopped_by_user = False



    def connection(self):
        try:
            print(f"Tentativo di apertura porta seriale: {self.serial_port}")
            self.ser = serial.Serial(self.serial_port, baudrate=115200)
            self.is_connected = True
            self.serial_connection_status = "CONNECTED"
            self.stability_status_changed.emit(self.serial_connection_status)
            self.serial_connection_status_bool.emit(self.is_connected)  # <--- emetti il segnale quando la porta si disconnette
            #invia comando per calcolare offset all'avvio
            self.ser.write(b"!offset\n")
            print("offset command sent")

            print(f"🔌 Connessione seriale avvenuta su {self.serial_port}")

        #eccezzione porta seriale occupata da implementare

        except serial.SerialException as e:
            print(f"Errore apertura seriale: {e}")
            self.handle_disconnection()
            return
        


    def run(self):
        if not self.is_connected:
            print(f"🔌 Porta seriale {self.serial_port} non connessa.")
            self.is_connected = False
            return

        try:
            #invia messaggio di start al microcontrollore
            #self.ser.write(f"!start {self.sampling_value}\n".encode()) 
            #print(f"start message sent: {self.sampling_value}")
            while self.is_running:
                #print("ciclo attivo")
                try:
                    #print("Bytes in attesa:", self.ser.in_waiting)
                    if not hasattr(self, 'ser') or not self.ser.is_open:
                        print("⚠️ Porta seriale chiusa, esco dal ciclo di acquisizione.")
                        break
                    if self._stopped_by_user and not self.is_running:
                        print("🔌 Stop richiesto dall'utente, esco dal ciclo di acquisizione.")
                        break
                    if self.ser.in_waiting:
                        raw_line = self.ser.readline()
                        try:
                            line = raw_line.decode('utf-8').strip()
                        except UnicodeDecodeError as e:
                            print(f"⚠️ Errore decodifica: {e}")
                            line = raw_line.decode('utf-8', errors='replace').strip()
                        #print("line read")
                        if line == "END_BLOCK":
                            self._block_counter += 1
                            if self._block_counter > 1:
                                if self._values_in_block == 256:
                                    self.serial_connection_status = "PERFECT"
                                elif self._values_in_block < 256 and self._values_in_block > 250:
                                    self.serial_connection_status = "GOOD"
                                elif self._values_in_block < 250 and self._values_in_block > 0:
                                    self.serial_connection_status = "LOW DATA"
                                elif self._values_in_block == 0:
                                    self.serial_connection_status = "NO DATA"
                                else:
                                    self.serial_connection_status = f"ERROR ({self._values_in_block} values)"
                                self.stability_status_changed.emit(self.serial_connection_status)
                            self._values_in_block = 0
                            continue
                        try:
                            # converte la stringa in float
                            value = float(line)
                            self.new_data.emit(float(value))    #invia solo un dato         
                            #print(value)   #debug           
                            self._values_in_block += 1
                        except ValueError:
                            pass
                except serial.SerialException as e:
                    print(f"SerialException: {e}")
                    if not self._stopped_by_user:
                        self.handle_disconnection()
                    break
                except Exception as e:
                    print(f"Errore lettura seriale: {e}")
                    self.handle_disconnection()
                    break
        finally:
            print("Serial worker stopped")
            if not self._stopped_by_user:
                self.handle_disconnection()
            self._stopped_by_user = False  # reset per eventuali riutilizzi
    
    #handle per porta che viene disconnessa
    def handle_disconnection(self):
        if self._already_disconnected:
            return
        self.is_running = False
        self.is_connected = False
        if hasattr(self, 'ser') and self.ser and self.ser.is_open:
            try:
                self.ser.write(b"!stop\n")
                print("stop message sent")
                self.ser.close()
            except Exception as e:
                print(f"Errore scrittura su seriale in disconnessione: {e}")

        self._already_disconnected = True
        self.stability_status_changed.emit("NOT CONNECTED")
        self.serial_connection_status_bool.emit(self.is_connected)

        # Mostra popup di errore solo se la finestra non sta chiudendo
        if getattr(self, 'is_closing', False) == False:
            self.error_popup.emit(self.serial_port)
        print(f"🔌 Disconnessione dalla porta seriale {self.serial_port} avvenuta.")


        # Chiamare self.wait() SOLO se NON siamo nel thread stesso
        if QtCore.QThread.currentThread() != self:
            self.wait(2000)


    #metodi per avviare e fermare la connessione seriale
    def start(self):
        """Avvia l'acquisizione, riaprendo la porta se necessario."""
        if not self.is_running:
            try:
                # ✅ RIAPRI LA PORTA SE È STATA CHIUSA
                if not self.ser.is_open:
                    self.ser.open()
                    print("✅ Porta seriale riaperta.")

                self.is_running = True
                self._already_disconnected = False
                #invia messaggio di start al microcontrollore
                self.ser.write(f"!start {self.sampling_value}\n".encode())
                print(f"start message sent: {self.sampling_value}")
                super().start()
            except Exception as e:
                print(f"Errore scrittura su seriale in start: {e}")
                self.handle_disconnection()
        


    def stop(self):
        """Ferma il thread in modo sicuro e chiude la porta seriale."""
        print("🔌 Chiusura richiesta per il thread voltage...")
        self.is_running = False
        self._stopped_by_user = True

        if hasattr(self, 'ser') and self.ser.is_open:
            try:
                self.ser.write(b"!stop\n")
                print("✅ Comando di stop inviato.")
                
                self.ser.close()
                print("✅ Porta seriale chiusa.")
            except Exception as e:
                print(f"⚠️ Errore durante lo stop: {e}")
        
        # NON impostare is_connected a False qui, la porta è solo chiusa, non persa.
        self.stability_status_changed.emit("STOPPED")
