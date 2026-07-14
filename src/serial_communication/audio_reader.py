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

import serial
import numpy as np
from PySide6.QtCore import Signal
from PySide6 import QtCore
import struct


class AudioSerialWorker(QtCore.QThread):
    new_data = Signal(np.ndarray, np.ndarray, float, int, bool, float)
    # ↑ magnitudes, ↑ phases, max_amp, peak_bin, above_threshold, threshold

    # Frame layout (must match the firmware's sendFFT_20_80kHz_with_peak):
    #   header:   0xAA 0x55, uint16 LE payload length
    #   payload:  float32 max_amplitude, uint16 peak_bin, uint8 above_threshold,
    #             float32 threshold, 154 x float32 magnitudes, 154 x int8 phases
    NUM_BINS = 154                                   # bins 51..204 = 20-80 kHz
    METADATA_SIZE = 4 + 2 + 1 + 4                    # 11 bytes
    EXPECTED_PAYLOAD = METADATA_SIZE + NUM_BINS * 4 + NUM_BINS   # 781 bytes

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
            # Virtual COM port: the baudrate value is ignored by USB CDC.
            # The read timeout keeps the acquisition loop supervisable: a
            # blocked read() returns after 1 s instead of hanging forever,
            # so the thread re-checks is_running and can exit cleanly
            # (previously shutdown relied on close() raising in the reader).
            self.ser = serial.Serial(self.serial_port, baudrate=115200, timeout=1.0)
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

                # === FRAME SYNC (byte-slip resync) ===
                # Scan the byte stream one byte at a time until a real frame
                # header is found. Advancing a single byte at a time (instead
                # of a fixed 4-byte read) guarantees deterministic recovery
                # within one frame after any desync: a truncated frame, a
                # dropped byte, or connecting mid-stream. A 0xAA 0x55 pair
                # can also legitimately occur INSIDE float payload data, so
                # a candidate header is only accepted if its length field
                # matches the one valid frame size - fake headers are
                # rejected before their garbage 'length' can swallow real
                # data (the old reader could read up to 64 KB of good bytes
                # as one bogus payload, and its garbage peak_bin could crash
                # the GUI with an IndexError).
                b = self.ser.read(1)
                if len(b) < 1 or b[0] != 0xAA:
                    continue  # timeout or not a header start: slide one byte

                # Consume a run of 0xAA bytes so that in '... AA AA 55' the
                # last AA is still recognized as the true header start.
                b = self.ser.read(1)
                while len(b) == 1 and b[0] == 0xAA:
                    b = self.ser.read(1)
                if len(b) < 1 or b[0] != 0x55:
                    continue  # not a header: keep scanning

                # === LENGTH VALIDATION ===
                len_bytes = self.ser.read(2)
                if len(len_bytes) < 2:
                    continue
                payload_length = struct.unpack('<H', len_bytes)[0]
                if payload_length != self.EXPECTED_PAYLOAD:
                    # Fake header (sync pattern inside payload data) or a
                    # protocol change: reject and resume scanning. Nothing
                    # is consumed beyond the 4 header bytes.
                    continue

                # === PAYLOAD (781 bytes) ===
                packet_data = self.ser.read(payload_length)
                if len(packet_data) != payload_length:
                    continue  # short read (timeout/disconnect): resync

                # === METADATA (11 bytes) ===
                offset = 0
                max_amplitude = struct.unpack('<f', packet_data[offset:offset+4])[0]
                offset += 4
                peak_bin = struct.unpack('<H', packet_data[offset:offset+2])[0]
                offset += 2
                above_threshold = bool(packet_data[offset])
                offset += 1
                current_threshold = struct.unpack('<f', packet_data[offset:offset+4])[0]
                offset += 4

                # Sanity check on decoded content: peak_bin is an index into
                # the 154 transmitted bins. An out-of-range value means the
                # frame is corrupt even though the framing looked right.
                # NOTE: framing + length checks cannot catch every corruption.
                # If a frame is truncated mid-stream (e.g. firmware TX abort),
                # its intact header swallows the following bytes as payload;
                # at most ~2 frames (~5 ms) are lost before the scanner
                # re-locks on the next real header. Only a checksum in the
                # wire protocol could close that gap completely.
                if peak_bin >= self.NUM_BINS:
                    continue

                # === FFT MAGNITUDES (154 x float32 = 616 bytes) ===
                fft_bytes_size = self.NUM_BINS * 4
                fft_data = np.frombuffer(
                    packet_data[offset:offset + fft_bytes_size], dtype=np.float32)
                offset += fft_bytes_size

                # Content sanity: real magnitudes are finite voltages (the
                # firmware sends |FFT|/N in volts). NaN/Inf here means the
                # payload bytes are not a real frame - drop it before it
                # reaches the plot, the click detector or the save buffer.
                if not (np.isfinite(max_amplitude) and
                        np.isfinite(current_threshold) and
                        np.isfinite(fft_data).all()):
                    continue

                # === FFT PHASES (154 x int8 = 154 bytes) ===
                fft_phases = np.frombuffer(
                    packet_data[offset:offset + self.NUM_BINS], dtype=np.int8)

                # Emit the decoded frame to the GUI thread
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
