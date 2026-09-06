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
import threading
import time


class AudioSerialWorker(QtCore.QThread):
    """
    Reader for BOTH firmware wire formats, told apart by the LENGTH FIELD.

        classic frame   payload 781   every FFT frame        (all firmwares)
        event frame     payload 802   click candidates only  (v3, event mode)

    That the length alone distinguishes them is the whole compatibility story:
    this reader talks to any firmware in the family without being configured,
    and a reader that only knows 781 rejects event frames cleanly as bad sync
    instead of misparsing them.

    The classic 11-byte metadata block sits at the SAME OFFSETS in both, by
    firmware design, so the classic decode below is shared; an event frame just
    carries 21 more bytes after it.

    Reference implementation, tested against real silicon and kept beside the
    firmware:  Audio_FIrmware_THIRD_realtime/tests/hardware/protocol.py
    """

    new_data = Signal(np.ndarray, np.ndarray, float, int, bool, float)
    # ↑ magnitudes, ↑ phases, max_amp, peak_bin, above_threshold, threshold
    # CLASSIC (781-byte) frames only. Deliberately unchanged: full mode must
    # reach zero new code, or "the old firmware still works" stops being true.

    new_event = Signal(dict)
    # EVENT (802-byte) frames. One dict per transmitted frame - candidate AND
    # neighbour alike; the consumer assembles the prev|curr|next triples.

    board_reply = Signal(str)
    # One '!...' ASCII line from the board (!ok / !err / !stats / !prof).
    # These arrive interleaved with binary frames and used to be discarded by
    # the sync scanner, which left the mode switch with no confirmation.

    # ── Wire layout, little-endian ─────────────────────────────────────────
    #   0xAA 0x55 | uint16 payload_len
    #   float32 max_amplitude | uint16 peak_bin | uint8 above_threshold
    #   float32 threshold
    #  ── event frames only, 21 bytes ──
    #   uint32 frame_idx | uint8 flags
    #   float32 E_i | E_hat_floor | noise_floor | std_noise
    #  ── both ──
    #   154 x float32 magnitudes | 154 x int8 phases
    NUM_BINS = 154                                   # bins 51..204 = 20-80 kHz
    METADATA_SIZE = 4 + 2 + 1 + 4                    # 11 bytes
    EVENT_EXTRA_SIZE = 4 + 1 + 4 + 4 + 4 + 4         # 21 bytes
    SPECTRUM_SIZE = NUM_BINS * 4 + NUM_BINS          # 770 bytes

    PAYLOAD_CLASSIC = METADATA_SIZE + SPECTRUM_SIZE                    # 781
    PAYLOAD_EVENT = METADATA_SIZE + EVENT_EXTRA_SIZE + SPECTRUM_SIZE   # 802
    VALID_PAYLOADS = (PAYLOAD_CLASSIC, PAYLOAD_EVENT)

    # Retained under its old name: other code reads it.
    EXPECTED_PAYLOAD = PAYLOAD_CLASSIC

    # flags byte of an event frame
    FLAG_CANDIDATE = 0x01
    FLAG_NEIGHBOUR = 0x02
    FLAG_OVERFLOW = 0x04    # events were lost BEFORE this one (board FIFO full)

    #: Longest '!' line the board sends; past this it is not a reply.
    MAX_REPLY_LEN = 200

    serial_connection_status_bool = Signal(bool)  # <--- aggiungi questo segnale
    error_popup = Signal(str)  # aggiungi questo segnale


    def __init__(self, serial_port):
        super().__init__()
        self.serial_port = serial_port
        self.is_connected = True
        self.is_running = False
        self._already_disconnected = False
        self._stopped_by_user = False

        # Link health. The reader had no counters at all, so a degraded link
        # was indistinguishable from a quiet room.
        self.n_frames_classic = 0
        self.n_frames_event = 0
        self.n_resync = 0          # bytes discarded looking for a header
        self.n_bad_payload = 0     # sync word with a length we do not accept
        self.n_short_read = 0      # header accepted, payload never arrived
        self.n_bad_content = 0     # framing fine, decoded values impossible

        # Partial '!' reply being accumulated by the sync scanner.
        self._reply_buf = bytearray()

        # Last '!stats ...' line the board sent, filled by stop().
        self.last_stats_line = None

        # Writes come from the GUI thread while this thread is reading.
        self._write_lock = threading.Lock()



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
                # matches a valid frame size - fake headers are rejected
                # before their garbage 'length' can swallow real data (the old
                # reader could read up to 64 KB of good bytes as one bogus
                # payload, and its garbage peak_bin could crash the GUI with
                # an IndexError).
                #
                # Every rejected byte is offered to the reply scanner first:
                # the board's '!ok' / '!err' / '!stats' lines travel on the
                # same pipe, between frames, and used to be thrown away here.
                b = self.ser.read(1)
                if len(b) < 1:
                    continue  # read timeout: loop and re-check is_running
                if b[0] != 0xAA:
                    self._scan_reply_byte(b[0])
                    self.n_resync += 1
                    continue

                # Consume a run of 0xAA bytes so that in '... AA AA 55' the
                # last AA is still recognized as the true header start.
                b = self.ser.read(1)
                while len(b) == 1 and b[0] == 0xAA:
                    b = self.ser.read(1)
                if len(b) < 1:
                    continue
                if b[0] != 0x55:
                    self._scan_reply_byte(b[0])
                    self.n_resync += 1
                    continue  # not a header: keep scanning

                # === LENGTH VALIDATION ===
                len_bytes = self.ser.read(2)
                if len(len_bytes) < 2:
                    continue
                payload_length = struct.unpack('<H', len_bytes)[0]
                if payload_length not in self.VALID_PAYLOADS:
                    # Fake header (sync pattern inside payload data) or a
                    # protocol change: reject and resume scanning. Nothing
                    # is consumed beyond the 4 header bytes.
                    self.n_bad_payload += 1
                    continue
                is_event = (payload_length == self.PAYLOAD_EVENT)

                # === PAYLOAD (781 or 802 bytes) ===
                packet_data = self.ser.read(payload_length)
                if len(packet_data) != payload_length:
                    self.n_short_read += 1
                    continue  # short read (timeout/disconnect): resync

                # === METADATA (11 bytes, same offsets in both formats) ===
                offset = 0
                max_amplitude = struct.unpack('<f', packet_data[offset:offset+4])[0]
                offset += 4
                peak_bin = struct.unpack('<H', packet_data[offset:offset+2])[0]
                offset += 2
                above_threshold = bool(packet_data[offset])
                offset += 1
                current_threshold = struct.unpack('<f', packet_data[offset:offset+4])[0]
                offset += 4

                # === EVENT METADATA (21 bytes, event frames only) ===
                # noise_floor and std_noise are here because the host CANNOT
                # recompute them: they come from a minimum-statistics estimator
                # over the QUIET frames, which event mode never transmits, and
                # every SNR feature plus the Stage 2 gates are calibrated
                # against them.
                frame_idx = flags = None
                E_i = E_hat_floor = noise_floor = std_noise = None
                if is_event:
                    (frame_idx, flags, E_i, E_hat_floor,
                     noise_floor, std_noise) = struct.unpack_from(
                        '<IBffff', packet_data, offset)
                    offset += self.EVENT_EXTRA_SIZE

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
                    self.n_bad_content += 1
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
                    self.n_bad_content += 1
                    continue

                # === FFT PHASES (154 x int8 = 154 bytes) ===
                fft_phases = np.frombuffer(
                    packet_data[offset:offset + self.NUM_BINS], dtype=np.int8)

                if not is_event:
                    # Emit the decoded frame to the GUI thread
                    self.n_frames_classic += 1
                    self.new_data.emit(fft_data, fft_phases, max_amplitude,
                                       peak_bin, above_threshold,
                                       current_threshold)
                    continue

                # The four event scalars are divisors and feature inputs; one
                # non-finite value would poison a whole event row.
                if not all(np.isfinite(v) for v in
                           (E_i, E_hat_floor, noise_floor, std_noise)):
                    self.n_bad_content += 1
                    continue

                # frombuffer returns a read-only view onto packet_data. A
                # classic frame is consumed synchronously, but an event dict is
                # queued and outlives this iteration, so it takes its own
                # writable copy.
                self.n_frames_event += 1
                self.new_event.emit({
                    'frame_idx': int(frame_idx),
                    'flags': int(flags),
                    'is_candidate': bool(flags & self.FLAG_CANDIDATE),
                    'is_neighbour': bool(flags & self.FLAG_NEIGHBOUR),
                    'had_overflow': bool(flags & self.FLAG_OVERFLOW),
                    'E_i': float(E_i),
                    'E_hat_floor': float(E_hat_floor),
                    'noise_floor': float(noise_floor),
                    'std_noise': float(std_noise),
                    'fft_mags': fft_data.copy(),
                    'phases': fft_phases.copy(),
                    'max_amplitude': float(max_amplitude),
                    'peak_bin': int(peak_bin),
                    'above_threshold': above_threshold,
                    'threshold': float(current_threshold),
                })

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


    # ─────────────────────────────────────────────────────────────────────
    #  Board replies and commands
    # ─────────────────────────────────────────────────────────────────────

    def _scan_reply_byte(self, byte):
        """
        Reassemble the board's ASCII '!' lines out of bytes the frame scanner
        rejects.

        Replies travel on the same pipe as frames, between them, so every byte
        of one lands in the reject path of run(). Accumulation starts only at
        '!' and ends at a newline, which is what keeps random payload bytes
        from ever forming a line: a stray '!' inside float data opens a buffer
        that the next non-printable byte, the next sync word or the length cap
        throws away before anything is emitted.
        """
        if byte in (0x0A, 0x0D):                       # LF / CR
            if self._reply_buf:
                line = self._reply_buf.decode('ascii', errors='replace').strip()
                self._reply_buf.clear()
                if line.startswith('!'):
                    self.board_reply.emit(line)
            return

        if not self._reply_buf:
            if byte == 0x21:                           # '!'
                self._reply_buf.append(byte)
            return

        if len(self._reply_buf) >= self.MAX_REPLY_LEN:
            self._reply_buf.clear()                    # not a reply after all
            return

        if 0x20 <= byte < 0x7F:                        # printable ASCII
            self._reply_buf.append(byte)
        else:
            self._reply_buf.clear()                    # binary: abandon the line

    def send_command(self, command):
        """
        Send one '!...' command to the board. Returns True if it went out.

        The answer comes back as a board_reply signal, not as a return value:
        at 390 fps a reply can sit behind several frames, so nothing here
        waits for it.

        Commands are written from the GUI thread while the reader thread is
        inside read(), hence the lock - a torn write would desync the board's
        command parser, not just this one reply.
        """
        if not (hasattr(self, 'ser') and self.ser is not None and self.ser.is_open):
            print(f"⚠️ Comando '{command.strip()}' non inviato: porta chiusa.")
            return False
        if not command.endswith('\n'):
            command += '\n'
        try:
            with self._write_lock:
                self.ser.write(command.encode('ascii'))
            return True
        except Exception as e:
            print(f"⚠️ Errore invio comando '{command.strip()}': {e}")
            return False

    def query_stats(self, timeout_s=0.5):
        """
        Ask the board for its cumulative counters and return the '!stats' line.

        MUST be called with the reader thread already joined: this reads the
        port directly, and two readers on one pipe would each get half the
        bytes. stop() arranges exactly that before closing.

        Why it matters: the firmware's `drop` counter is the ONLY evidence of
        frames the ADC queue lost. Those frames never reach Stage 1, so
        frame_idx does not skip over them - it simply counts fewer frames, and
        every timestamp derived from it then runs early with nothing in-band
        to say so.
        """
        if not (hasattr(self, 'ser') and self.ser is not None and self.ser.is_open):
            return None
        line = None
        saved_timeout = self.ser.timeout
        try:
            self.ser.timeout = 0.05
            with self._write_lock:
                self.ser.write(b"!stats!\n")
            deadline = time.monotonic() + timeout_s
            self._reply_buf.clear()
            while time.monotonic() < deadline:
                chunk = self.ser.read(max(1, self.ser.in_waiting))
                if not chunk:
                    continue
                for byte in chunk:
                    if byte in (0x0A, 0x0D):
                        if self._reply_buf:
                            text = self._reply_buf.decode('ascii', errors='replace').strip()
                            self._reply_buf.clear()
                            if text.startswith('!'):
                                self.board_reply.emit(text)
                                if text.startswith('!stats'):
                                    line = text
                    elif not self._reply_buf:
                        if byte == 0x21:
                            self._reply_buf.append(byte)
                    elif 0x20 <= byte < 0x7F:
                        self._reply_buf.append(byte)
                    else:
                        self._reply_buf.clear()
                if line is not None:
                    break
        except Exception as e:
            print(f"⚠️ !stats! non leggibile: {e}")
        finally:
            try:
                self.ser.timeout = saved_timeout
            except Exception:
                pass
        return line

    def link_health(self):
        """Host-side frame accounting, for the status line."""
        return {
            'classic': self.n_frames_classic,
            'event': self.n_frames_event,
            'resync': self.n_resync,
            'bad_payload': self.n_bad_payload,
            'short_read': self.n_short_read,
            'bad_content': self.n_bad_content,
        }

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
  
  

    def stop(self, collect_stats=True):
        """✅ STOP SICURO DEL THREAD con chiusura della porta."""
        print("🔄 Fermando AudioSerialWorker...")
        
        # Ferma il loop principale
        self.is_running = False
        self._stopped_by_user = True

        # Join the reader BEFORE touching the port. The old code closed the
        # port while this thread could still be inside read() - which is where
        # the 'bad file descriptor' branch in run() comes from - and it also
        # made reading a reply impossible, because two readers on one pipe
        # each get half the bytes.
        if self.isRunning() and QtCore.QThread.currentThread() is not self:
            self.wait(2000)

        self.last_stats_line = None

        # Invia comandi di stop al dispositivo
        if hasattr(self, 'ser') and self.ser.is_open and self._already_disconnected == False:
            try:
                # The board's own frame/drop counters, while the port is still
                # open and nothing else is reading it.
                if collect_stats:
                    self.last_stats_line = self.query_stats()

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
