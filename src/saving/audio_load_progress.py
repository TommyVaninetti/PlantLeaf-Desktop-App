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

from PySide6.QtCore import QObject, Signal, Slot
import numpy as np
import struct
import json
import zlib
import os
from core.click_pipeline_v5 import (  #keep functions and constants in sync with click_pipeline_v5.py
    _normalize_fft,
    reconstruct_frame_v5,
    compute_hilbert_envelope,
    compute_fft_energy as _compute_fft_energy_v5,
    AdaptiveNoiseEstimatorV5,
    _BIN_START as V5_BIN_START,
    _BIN_END   as V5_BIN_END,
    FS         as V5_FS,
    FFT_SIZE   as V5_FFT_SIZE,
)

class AudioLoadWorker(QObject):
    progress = Signal(int)      # percentuale
    finished = Signal(dict)          # dict con tutti i dati caricati
    error = Signal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self._cancelled = False

    @Slot()
    def run(self):
        try:
            if not self.file_path or not os.path.exists(self.file_path):
                self.error.emit(f"File non trovato: {self.file_path}")
                return

            with open(self.file_path, 'rb') as f:
                # STEP 1: Load header
                self.progress.emit(1)
                header_data = f.read(128)
                if len(header_data) != 128:
                    self.error.emit("Header incompleto")
                    return

                header_info = {
                    'magic': header_data[0:10].rstrip(b'\x00').decode('ascii'),
                    'version': struct.unpack('<f', header_data[10:14])[0],
                    'experiment': header_data[14:34].rstrip(b'\x00').decode('ascii'),
                    'fs': struct.unpack('<I', header_data[34:38])[0],
                    'fft_size': struct.unpack('<I', header_data[38:42])[0],
                    'freq_min': struct.unpack('<I', header_data[42:46])[0],
                    'freq_max': struct.unpack('<I', header_data[46:50])[0],
                    'threshold': struct.unpack('<f', header_data[50:54])[0],
                    'start_time': struct.unpack('<d', header_data[54:62])[0],
                    'end_time': struct.unpack('<d', header_data[62:70])[0],
                    'data_points': struct.unpack('<I', header_data[70:74])[0],
                    'acquisition_count': struct.unpack('<I', header_data[74:78])[0]
                }
                
                if header_info['magic'] != 'PLANTAUDIO':
                    self.error.emit("Magic number non valido")
                    return

                file_version = header_info['version']

                # STEP 2: Load FFT data
                self.progress.emit(5)
                remaining_data = f.read()
                click_start = remaining_data.find(b'CLCK')
                
                if click_start >= 0:
                    fft_bytes = remaining_data[:click_start]
                    click_section = remaining_data[click_start:]
                else:
                    fft_bytes = remaining_data
                    click_section = None

                fft_data = []
                phase_data = []
                samples_per_fft = 154
                
                if file_version >= 3:
                    # === AUTO-DETECT: Con o Senza Separatori? ===
                    samples_per_frame = 154
                    bytes_per_sample = 5
                    separator_bytes = 5
                    
                    # ✅ PROVA A LEGGERE PRIMO FRAME
                    test_offset = samples_per_frame * bytes_per_sample  # 770 byte
                    
                    has_separators = False
                    if test_offset + separator_bytes <= len(fft_bytes):
                        # Controlla se c'è NaN dopo il primo frame
                        test_mag = struct.unpack('<f', fft_bytes[test_offset:test_offset+4])[0]
                        test_phase = struct.unpack('<b', fft_bytes[test_offset+4:test_offset+5])[0]
                        
                        if np.isnan(test_mag) and test_phase == -1:
                            has_separators = True
                            print("📋 Rilevato formato v3.0 OLD (con separatori)")
                        else:
                            print("📋 Rilevato formato v3.0 NEW (senza separatori)")
                    
                    # === PARSING ADATTIVO ===
                    offset = 0
                    frame_count = 0
                    total_bytes = len(fft_bytes)
                    last_progress = 5
                    
                    while offset < len(fft_bytes):
                        if self._cancelled:
                            return
                        
                        frame_mags = []
                        frame_phases = []
                        
                        # Leggi 154 campioni
                        for i in range(samples_per_frame):
                            if offset + bytes_per_sample > len(fft_bytes):
                                break
                            
                            mag = struct.unpack('<f', fft_bytes[offset:offset+4])[0]
                            phase = struct.unpack('<b', fft_bytes[offset+4:offset+5])[0]
                            offset += bytes_per_sample
                            
                            frame_mags.append(mag)
                            frame_phases.append(phase)
                        
                        if len(frame_mags) == samples_per_frame:
                            fft_data.append(np.array(frame_mags, dtype=np.float32))
                            phase_data.append(np.array(frame_phases, dtype=np.int8))
                            frame_count += 1
                        
                        # ✅ SKIPPA SEPARATORE SOLO SE PRESENTE
                        if has_separators and offset + separator_bytes <= len(fft_bytes):
                            # Verifica che sia davvero un separatore
                            sep_mag = struct.unpack('<f', fft_bytes[offset:offset+4])[0]
                            if np.isnan(sep_mag):
                                offset += separator_bytes
                        
                        # Progress throttling
                        if frame_count % 500 == 0 and frame_count > 0:
                            new_progress = 5 + int((offset / total_bytes) * 53) # 5-58%
                            if new_progress > last_progress:
                                self.progress.emit(new_progress)
                                last_progress = new_progress
                                        
                else:
                    # === VERSIONE 2.0: Legacy ===
                    if len(fft_bytes) == 0:
                        self.error.emit("File vuoto")
                        return
                    
                    if len(fft_bytes) % 4 != 0:
                        self.error.emit("Dati FFT corrotti")
                        return
                    
                    fft_array = np.frombuffer(fft_bytes, dtype=np.float32)
                    
                    temp_fft = []
                    for value in fft_array:
                        if self._cancelled:
                            return
                        if np.isnan(value):
                            if len(temp_fft) > 0:
                                samples_per_fft = len(temp_fft)
                                break
                            temp_fft = []
                        else:
                            temp_fft.append(value)
                    
                    if samples_per_fft == 0 or samples_per_fft > 200:
                        samples_per_fft = 154
                    
                    current_fft = []
                    frame_idx = 0
                    last_progress = 5

                    for i, value in enumerate(fft_array):
                        if self._cancelled:
                            return
                        if np.isnan(value):
                            if len(current_fft) == samples_per_fft:
                                fft_data.append(np.array(current_fft))
                                phase_data.append(np.zeros(samples_per_fft, dtype=np.int8))
                                frame_idx += 1
                            current_fft = []
                        else:
                            current_fft.append(value)
                        
                        # THROTTLING: Ogni 5000 campioni
                        if i % 5000 == 0 and i > 0:
                            new_progress = 5 + int((i / len(fft_array)) * 53)  # 5-58%
                            if new_progress > last_progress:
                                self.progress.emit(new_progress)
                                last_progress = new_progress
                    
                    if len(current_fft) == samples_per_fft:
                        fft_data.append(np.array(current_fft))
                        phase_data.append(np.zeros(samples_per_fft, dtype=np.int8))

                # CHECKPOINT 58%
                self.progress.emit(58)
                
                total_frames = len(fft_data)
                if total_frames == 0:
                    self.error.emit("Nessun dato FFT valido")
                    return

                if samples_per_fft == 0 or samples_per_fft > 200:
                    samples_per_fft = 154

                freq_min = header_info['freq_min']
                freq_max = header_info['freq_max']
                frequency_axis = np.linspace(freq_min, freq_max, samples_per_fft)

                # STEP 3: Click events
                # CHECKPOINT 59%
                self.progress.emit(59)
                
                click_events = []
                try:
                    if click_section and len(click_section) >= 8:
                        marker = click_section[0:4]
                        if marker == b'CLCK':
                            click_length = struct.unpack('<I', click_section[4:8])[0]
                            if len(click_section) >= 8 + click_length:
                                compressed_data = click_section[8:8+click_length]
                                try:
                                    decompressed = zlib.decompress(compressed_data)
                                    click_events = json.loads(decompressed.decode('utf-8'))
                                except:
                                    click_events = []
                except:
                    click_events = []

                # STEP 4: Metadata and timing calculations
                                
                estimated_fft_rate = 390.0
                frame_duration_ms = 1000.0 / estimated_fft_rate
                total_duration_sec = (total_frames * frame_duration_ms / 1000.0)

            # NEW STEP 4.5: Precompute FFT means + adaptive noise  (was in main thread)
            # Runs in the worker thread so the UI never freezes.
            # Progress goes from 60 → 92 % during this loop.
            self.progress.emit(60)

            fs_h       = header_info.get('fs',       V5_FS)
            fft_size_h = header_info.get('fft_size', V5_FFT_SIZE)
            full_freq_axis = np.arange(fft_size_h // 2) * (fs_h / fft_size_h)

            n = total_frames
            fft_means_arr   = np.empty(n, dtype=np.float32)
            E_hat_floors    = np.empty(n, dtype=np.float32)
            noise_floors    = np.empty(n, dtype=np.float32)
            std_noises      = np.empty(n, dtype=np.float32)
            fft_timestamps  = np.empty(n, dtype=np.float64)

            estimator = AdaptiveNoiseEstimatorV5()

            for i in range(n):
                if self._cancelled:
                    return

                fft_frame = fft_data[i]
                fft_timestamps[i] = i * frame_duration_ms / 1000.0

                # Pad into full half-spectrum and normalize
                full_mags = np.zeros(fft_size_h // 2, dtype=np.float64)
                mags_raw  = np.asarray(fft_frame, dtype=np.float64)
                n_bins = min(len(mags_raw), V5_BIN_END - V5_BIN_START + 1)
                full_mags[V5_BIN_START : V5_BIN_START + n_bins] = mags_raw[:n_bins]
                fft_norm = _normalize_fft(full_mags, full_freq_axis)

                # FIX: compute energy [V²] FIRST, store it in fft_means_arr
                # (previously stored mean amplitude [V] → mismatch with E_hat_floors [V²])
                E_i = float(_compute_fft_energy_v5(fft_norm[V5_BIN_START : V5_BIN_END + 1]))
                fft_means_arr[i] = E_i   # [V²] — now matches E_hat_floor_arr units

                # iFFT + Hilbert for B2 buffer
                if i < len(phase_data):
                    frame_data_r = reconstruct_frame_v5(
                        mags_raw, phase_data[i], fs_h, fft_size_h, normalize=True
                    )
                else:
                    frame_data_r = None

                if frame_data_r is not None:
                    env        = compute_hilbert_envelope(frame_data_r['signal'])
                    env_mean_i = float(np.mean(env))
                    env_std_i  = float(np.std(env))
                else:
                    # FIX: fallback must pass amplitude [V], not energy [V²]
                    env_mean_i = float(np.sqrt(max(E_i, 0.0)))
                    env_std_i  = 0.0

                noise = estimator.update(E_i, env_mean_i, env_std_i)
                E_hat_floors[i] = noise['E_hat_floor']
                noise_floors[i] = noise['noise_floor']
                std_noises[i]   = noise['std_noise']

                # Report progress: map i/n → 60..92 %
                if i % 500 == 0:
                    pct = 60 + int((i / n) * 32)
                    self.progress.emit(pct)

            self.progress.emit(93)

            # STEP 5: Overview — now instant (reads fft_means_arr)
            self.progress.emit(94)
            overview_fps    = 10
            overview_points = int(total_duration_sec * overview_fps)
            frame_step      = max(1, total_frames // overview_points)
            overview_x = []
            overview_y = []
            for i in range(0, total_frames, frame_step):
                if self._cancelled:
                    return
                overview_x.append(i * frame_duration_ms / 1000.0)
                overview_y.append(float(fft_means_arr[i]))
            overview_x = np.array(overview_x)
            overview_y = np.array(overview_y)

            # STEP 6: Streaming buffer — also instant
            self.progress.emit(96)
            start_time  = 0.0
            end_time    = min(total_duration_sec, 20.0)
            start_frame = 0
            end_frame   = min(int(end_time * 1000 / frame_duration_ms), total_frames)
            stream_x = []
            stream_y = []
            for frame_idx in range(start_frame, end_frame):
                if self._cancelled:
                    return
                stream_x.append(frame_idx * frame_duration_ms / 1000.0)
                stream_y.append(float(fft_means_arr[frame_idx]))
            streaming_x = np.array(stream_x)
            streaming_y = np.array(stream_y)

            # STEP 7: Emit result
            self.progress.emit(98)

            data_dict = {
                'header_info'          : header_info,
                'fft_data'             : fft_data,
                'phase_data'           : phase_data,
                'frequency_axis'       : frequency_axis.tolist(),
                'total_frames'         : total_frames,
                'frame_duration_ms'    : frame_duration_ms,
                'total_duration_sec'   : total_duration_sec,
                'click_events'         : click_events,
                'overview_x'           : overview_x.tolist(),
                'overview_y'           : overview_y.tolist(),
                'streaming_x'          : streaming_x.tolist(),
                'streaming_y'          : streaming_y.tolist(),
                'streaming_start_time' : start_time,
                'streaming_end_time'   : end_time,
                # pre-computed arrays — main thread just assigns, no recompute
                'fft_means'            : fft_means_arr,
                'fft_timestamps'       : fft_timestamps,
                'E_hat_floor_arr'      : E_hat_floors,
                'noise_floor_arr'      : noise_floors,
                'std_noise_arr'        : std_noises,
            }

            self.finished.emit(data_dict)
            self.progress.emit(100)

        except Exception as e:
            self.error.emit(str(e))

    def cancel_load(self):
        self._cancelled = True