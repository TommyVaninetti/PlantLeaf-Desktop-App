from PySide6.QtCore import QObject, Signal, Slot
import numpy as np
import struct
import json
import zlib
import os

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
                            new_progress = 5 + int((offset / total_bytes) * 80)
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
                    last_progress = 5  # ✅ INIZIA DA 5

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
                        
                        # ✅ THROTTLING: Ogni 5000 campioni (non 1000)
                        if i % 5000 == 0 and i > 0:
                            new_progress = 5 + int((i / len(fft_array)) * 80)  # 5-85%
                            if new_progress > last_progress:
                                self.progress.emit(new_progress)
                                last_progress = new_progress
                    
                    if len(current_fft) == samples_per_fft:
                        fft_data.append(np.array(current_fft))
                        phase_data.append(np.zeros(samples_per_fft, dtype=np.int8))

                # ✅ CHECKPOINT 85%
                self.progress.emit(85)
                
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
                # ✅ CHECKPOINT 90%
                self.progress.emit(90)
                
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

                # STEP 4: Metadata
                # ✅ CHECKPOINT 93%
                self.progress.emit(93)
                
                estimated_fft_rate = 390.0
                frame_duration_ms = 1000.0 / estimated_fft_rate
                total_duration_sec = (total_frames * frame_duration_ms / 1000.0)

                # STEP 5: Overview
                # ✅ CHECKPOINT 95%
                self.progress.emit(95)
                
                overview_fps = 10
                overview_points = int(total_duration_sec * overview_fps)
                frame_step = max(1, total_frames // overview_points)
                overview_x = []
                overview_y = []
                
                for i in range(0, total_frames, frame_step):
                    if self._cancelled:
                        return
                    frame_time = (i * frame_duration_ms) / 1000.0
                    energy = np.mean(np.abs(fft_data[i]))
                    overview_x.append(frame_time)
                    overview_y.append(energy)
                
                overview_x = np.array(overview_x)
                overview_y = np.array(overview_y)

                # STEP 6: Streaming buffer
                # ✅ CHECKPOINT 97%
                self.progress.emit(97)
                
                streaming_fps = 100
                window_size = 20.0
                start_time = 0
                end_time = min(total_duration_sec, window_size)
                start_frame = int((start_time * 1000) / frame_duration_ms)
                end_frame = int((end_time * 1000) / frame_duration_ms)
                end_frame = min(end_frame, total_frames)
                
                stream_x = []
                stream_y = []
                for frame_idx in range(start_frame, end_frame, 1):
                    if self._cancelled:
                        return
                    frame_time = (frame_idx * frame_duration_ms) / 1000.0
                    signal_sample = np.mean(np.abs(fft_data[frame_idx]))
                    stream_x.append(frame_time)
                    stream_y.append(signal_sample)
                
                streaming_x = np.array(stream_x)
                streaming_y = np.array(stream_y)

                # STEP 7: Emit result
                # ✅ CHECKPOINT 98%
                self.progress.emit(98)
                
                data_dict = {
                    'header_info': header_info,
                    'fft_data': fft_data,
                    'phase_data': phase_data,
                    'frequency_axis': frequency_axis.tolist(),
                    'total_frames': total_frames,
                    'frame_duration_ms': frame_duration_ms,
                    'total_duration_sec': total_duration_sec,
                    'click_events': click_events,
                    'overview_x': overview_x.tolist(),
                    'overview_y': overview_y.tolist(),
                    'streaming_x': streaming_x.tolist(),
                    'streaming_y': streaming_y.tolist(),
                    'streaming_start_time': start_time,
                    'streaming_end_time': end_time,
                }
                
                self.finished.emit(data_dict)

                # ✅ FINALE 100%
                self.progress.emit(100)

        except Exception as e:
            self.error.emit(str(e))

    def cancel_load(self):
        self._cancelled = True