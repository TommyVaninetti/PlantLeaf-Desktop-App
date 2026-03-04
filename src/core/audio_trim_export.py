"""
Audio Trim Exporter - Sistema rigoroso per il taglio di file .paudio
Struttura identica a voltage_trim_export.py ma ottimizzata per FFT frames

FILE FORMAT .paudio:
- Header: 128 bytes (PLANTAUDIO magic, version 3.0, metadata)
- Data: Sequenza di frame FFT, ogni frame = 154 campioni
  - Ogni campione = 5 bytes (magnitude float32 + phase int8)
  - Frame size = 154 * 5 = 770 bytes
- Footer: Click events (opzionale)
  - Marker: b'CLCK' (4 bytes)
  - Length: uint32 (4 bytes)
  - Compressed JSON data (zlib)

IMPORTANTE: NON usiamo overwrite mode per audio (file troppo grandi)
"""

import struct
import numpy as np
import json
import zlib
import os
from datetime import datetime
from PySide6.QtWidgets import QProgressDialog, QMessageBox
from PySide6.QtCore import Qt


class AudioTrimExporter:
    """Gestisce l'export trimmed di file audio con integrità garantita"""
    
    def __init__(self, parent, file_path, metadata):
        """
        Args:
            parent: ReplayWindowAudio instance
            file_path: Path del file .paudio
            metadata: Dict con header info
        """
        self.parent = parent
        self.file_path = file_path
        self.metadata = metadata
        
        # Parametri FFT
        self.fs = metadata.get('fs', 200000)
        self.fft_size = metadata.get('fft_size', 512)
        self.freq_min = metadata.get('freq_min', 20000)
        self.freq_max = metadata.get('freq_max', 80000)
        
        # Calcola frame rate
        self.frame_rate = self.fs / self.fft_size  # ~390 FPS
        self.frame_duration_ms = 1000.0 / self.frame_rate
        
        # Dimensioni dati
        self.samples_per_frame = 154  # Bins 20-80kHz
        self.bytes_per_sample = 5  # float32 (4) + int8 (1)
        self.bytes_per_frame = self.samples_per_frame * self.bytes_per_sample  # 770
    
    def execute_trim_export(self, params):
        """
        Esegue l'export trimmed di un file audio.
        
        Args:
            params: dict con:
                - start_time: float (secondi)
                - end_time: float (secondi)
                - output_path: str (path nuovo file, NON overwrite)
                - include_clicks: bool
        
        Returns:
            bool: True se successo, False altrimenti
        """
        try:
            # === FASE 1: VALIDAZIONE PARAMETRI ===
            start_time = params['start_time']
            end_time = params['end_time']
            output_path = params['output_path']
            include_clicks = params.get('include_clicks', True)
            
            duration = end_time - start_time
            
            if duration < 0.1:
                QMessageBox.warning(self.parent, "Invalid Range", 
                                  "Duration must be at least 0.1 seconds")
                return False
            
            # === FASE 2: CALCOLA RANGE FRAME ===
            total_duration = self.metadata.get('duration', 0)
            estimated_total_frames = int(total_duration * self.frame_rate)
            
            start_frame = int(start_time * self.frame_rate)
            end_frame = int(end_time * self.frame_rate)
            new_frame_count = end_frame - start_frame
            
            print(f"\n📊 Audio Trim Export Parameters:")
            print(f"   Time Range: {start_time:.2f}s - {end_time:.2f}s (duration: {duration:.2f}s)")
            print(f"   Frame Range: {start_frame} - {end_frame} (total: {new_frame_count})")
            print(f"   Frame Rate: {self.frame_rate:.2f} FPS")
            print(f"   Samples per frame: {self.samples_per_frame}")
            print(f"   Bytes per frame: {self.bytes_per_frame}")
            
            # === FASE 3: SETUP PROGRESS DIALOG ===
            progress = QProgressDialog("Exporting trimmed audio file...", "Cancel", 0, 100, self.parent)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            progress.setWindowTitle("Export Progress")
            
            # === FASE 4: VERIFICA FILE OUTPUT ===
            if os.path.exists(output_path):
                reply = QMessageBox.question(
                    self.parent, "File Exists",
                    f"File already exists:\n{output_path}\n\nOverwrite?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    progress.close()
                    return False
            
            progress.setValue(5)
            if progress.wasCanceled():
                return False
            
            # === FASE 5: LEGGI E FILTRA CLICK EVENTS (se richiesto) ===
            filtered_clicks = []
            
            if include_clicks:
                progress.setLabelText("Reading click events...")
                progress.setValue(10)
                
                all_clicks = self._read_click_events_from_file()
                
                # Filtra click completamente dentro la regione
                for click in all_clicks:
                    # Click event format: {'timestamp': sec, 'frequency': Hz, 'amplitude': V, 'duration_us': μs}
                    click_time = click.get('timestamp', click.get('time', 0))  # Supporta entrambi i nomi
                    
                    # Click deve essere completamente nella regione
                    if start_time <= click_time <= end_time:
                        # Deep copy e re-timestamp
                        new_click = click.copy()
                        new_click['timestamp'] = click_time - start_time
                        filtered_clicks.append(new_click)
                        
                        print(f"   ✅ Click exported: {click_time:.3f}s → {new_click['timestamp']:.3f}s")
                
                print(f"\n📊 Click events summary: {len(all_clicks)} total, {len(filtered_clicks)} exported")
            
            progress.setValue(15)
            if progress.wasCanceled():
                return False
            
            # === FASE 6: CREA NUOVO HEADER ===
            progress.setLabelText("Creating new header...")
            progress.setValue(20)
            
            new_data_points = new_frame_count * self.samples_per_frame
            
            new_header = self._create_trimmed_header(
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                data_points=new_data_points
            )
            
            # === FASE 7: SCRIVI FILE OUTPUT ===
            progress.setLabelText(f"Writing to {os.path.basename(output_path)}...")
            progress.setValue(25)
            
            print(f"📖 Reading from: {os.path.basename(self.file_path)}")
            print(f"📝 Writing to: {os.path.basename(output_path)}")
            
            with open(output_path, 'wb') as out_f:
                # Scrivi header
                out_f.write(new_header)
                print(f"✅ Header written (128 bytes)")
                
                # Apri file sorgente
                with open(self.file_path, 'rb') as in_f:
                    # Verifica dimensione file sorgente
                    in_f.seek(0, 2)
                    source_size = in_f.tell()
                    in_f.seek(0)
                    print(f"📏 Source file size: {source_size} bytes ({source_size / (1024*1024):.2f} MB)")
                    
                    # Salta header (128 bytes)
                    in_f.seek(128)
                    
                    # Calcola offset frame da copiare
                    start_offset = 128 + (start_frame * self.bytes_per_frame)
                    
                    print(f"📍 Seeking to offset: {start_offset} (start_frame={start_frame})")
                    
                    # Posizionati all'inizio dei frame da copiare
                    in_f.seek(start_offset)
                    current_pos = in_f.tell()
                    print(f"📍 Current position after seek: {current_pos}")
                    
                    # Copia frame a blocchi con progress
                    chunk_frames = 1000  # 1000 frame per chunk (~2.5s)
                    frames_written = 0
                    
                    while frames_written < new_frame_count:
                        if progress.wasCanceled():
                            # Rimuovi file parziale
                            out_f.close()
                            if os.path.exists(output_path):
                                os.remove(output_path)
                            print("❌ Export cancelled by user")
                            return False
                        
                        # Calcola quanti frame leggere in questo chunk
                        frames_in_chunk = min(chunk_frames, new_frame_count - frames_written)
                        
                        # Leggi chunk (770 bytes per frame)
                        chunk_bytes = in_f.read(frames_in_chunk * self.bytes_per_frame)
                        
                        if not chunk_bytes:
                            print(f"⚠️  No data read at position {in_f.tell()}, expected {frames_in_chunk * self.bytes_per_frame} bytes")
                            break
                        
                        if len(chunk_bytes) != frames_in_chunk * self.bytes_per_frame:
                            print(f"⚠️  Partial read: {len(chunk_bytes)} bytes instead of {frames_in_chunk * self.bytes_per_frame}")
                            # Scrivi comunque quello che c'è
                        
                        # Scrivi chunk direttamente (NO re-timestamp necessario per audio)
                        out_f.write(chunk_bytes)
                        frames_written += frames_in_chunk
                        
                        # Aggiorna progress (25-85%)
                        copy_progress = int(25 + (frames_written / new_frame_count) * 60)
                        progress.setValue(copy_progress)
                        progress.setLabelText(
                            f"Copying frames: {frames_written:,} / {new_frame_count:,}"
                        )
                    
                    print(f"✅ Data written: {frames_written} frames ({frames_written * self.bytes_per_frame} bytes)")
            
            progress.setValue(85)
            
            # === FASE 8: SCRIVI FOOTER CON CLICK EVENTS (se presenti) ===
            if filtered_clicks:
                progress.setLabelText("Writing click events footer...")
                progress.setValue(90)
                
                with open(output_path, 'ab') as f:
                    # Marker
                    f.write(b'CLCK')
                    
                    # Comprimi JSON
                    click_json = json.dumps(filtered_clicks, separators=(',', ':'))
                    click_compressed = zlib.compress(click_json.encode('utf-8'))
                    
                    # Lunghezza dati
                    f.write(struct.pack('<I', len(click_compressed)))
                    
                    # Dati compressi
                    f.write(click_compressed)
                    
                    print(f"✅ Footer written: {len(filtered_clicks)} click events")
            
            progress.setValue(95)
            
            # === FASE 9: VALIDAZIONE FINALE ===
            progress.setLabelText("Validating output file...")
            
            if not self._validate_output_file(output_path, new_frame_count):
                QMessageBox.warning(self.parent, "Validation Warning", 
                                  "File created but validation failed. Check manually.")
            
            progress.setValue(100)

            return True
            
        except Exception as e:
            import traceback
            error_msg = f"Export failed: {str(e)}\n\n{traceback.format_exc()}"
            print(f"❌ {error_msg}")
            QMessageBox.critical(self.parent, "Export Error", error_msg)
            return False
    
    def _create_trimmed_header(self, start_time, end_time, duration, data_points):
        """Crea un header per il file trimmed"""
        # Copia metadata originale
        orig_meta = self.metadata.copy()
        
        # Crea header con dati aggiornati
        header_data = {
            'magic': b'PLANTAUDIO',
            'version': orig_meta.get('version', 3.0),
            'experiment_type': orig_meta.get('experiment', 'Trimmed')[:20].ljust(20),
            'fs': orig_meta['fs'],
            'fft_size': orig_meta['fft_size'],
            'freq_min': orig_meta['freq_min'],
            'freq_max': orig_meta['freq_max'],
            'threshold': orig_meta.get('threshold', 0.03),
            'start_time': datetime.now().timestamp(),  # Nuovo timestamp
            'end_time': datetime.now().timestamp(),
            'data_points': data_points,
            'acquisition_count': orig_meta.get('acquisition_count', 0),
            'reserved': b'\x00' * 50
        }
        
        # Costruisci header bytes (128 totali)
        header_bytes = bytearray()
        
        # Magic number (10 byte)
        magic = header_data['magic'][:10]
        header_bytes.extend(magic)
        header_bytes.extend(b'\x00' * (10 - len(magic)))
        
        # Version (4 byte)
        header_bytes.extend(struct.pack('<f', header_data['version']))
        
        # Experiment type (20 byte)
        exp_type = header_data['experiment_type'].encode('ascii', errors='replace')[:20]
        exp_type += b'\x00' * (20 - len(exp_type))
        header_bytes.extend(exp_type)
        
        # Audio parameters (20 byte)
        header_bytes.extend(struct.pack('<I', header_data['fs']))
        header_bytes.extend(struct.pack('<I', header_data['fft_size']))
        header_bytes.extend(struct.pack('<I', header_data['freq_min']))
        header_bytes.extend(struct.pack('<I', header_data['freq_max']))
        header_bytes.extend(struct.pack('<f', header_data['threshold']))
        
        # Timestamps (16 byte)
        header_bytes.extend(struct.pack('<d', header_data['start_time']))
        header_bytes.extend(struct.pack('<d', header_data['end_time']))
        
        # Counters (8 byte)
        header_bytes.extend(struct.pack('<I', header_data['data_points']))
        header_bytes.extend(struct.pack('<I', header_data['acquisition_count']))
        
        # Reserved (50 byte)
        header_bytes.extend(header_data['reserved'][:50])
        
        # Verifica dimensione
        if len(header_bytes) != 128:
            raise ValueError(f"Header size mismatch: {len(header_bytes)} bytes (expected 128)")
        
        return bytes(header_bytes)
    
    def _read_click_events_from_file(self):
        """Legge click events dal file originale"""
        try:
            with open(self.file_path, 'rb') as f:
                # Salta header
                f.seek(128)
                
                # Leggi tutto il resto
                remaining_data = f.read()
                
                # Cerca marker CLCK
                click_start = remaining_data.find(b'CLCK')
                
                if click_start < 0:
                    print("📊 No click events found in file")
                    return []
                
                # Leggi click section
                click_section = remaining_data[click_start:]
                
                if len(click_section) < 8:
                    print("⚠️  Click section too short")
                    return []
                
                # Verifica marker
                marker = click_section[0:4]
                if marker != b'CLCK':
                    print("⚠️  Invalid click marker")
                    return []
                
                # Leggi lunghezza
                click_length = struct.unpack('<I', click_section[4:8])[0]
                
                if len(click_section) < 8 + click_length:
                    print("⚠️  Incomplete click data")
                    return []
                
                # Decomprimi JSON
                compressed_data = click_section[8:8+click_length]
                decompressed = zlib.decompress(compressed_data)
                click_events = json.loads(decompressed.decode('utf-8'))
                
                print(f"📊 Found {len(click_events)} click events in file")
                return click_events
                
        except Exception as e:
            print(f"⚠️  Error reading click events: {e}")
            return []
    
    def _validate_output_file(self, output_path, expected_frames):
        """Valida il file output"""
        try:
            with open(output_path, 'rb') as f:
                # Verifica magic number
                magic = f.read(10).rstrip(b'\x00').decode('ascii')
                if magic != 'PLANTAUDIO':
                    print(f"❌ Validation failed: Invalid magic number '{magic}'")
                    return False
                
                # Leggi data_points dal header
                f.seek(70)
                data_points = struct.unpack('<I', f.read(4))[0]
                
                expected_data_points = expected_frames * self.samples_per_frame
                
                if data_points != expected_data_points:
                    print(f"⚠️  Data points mismatch: {data_points} in header vs {expected_data_points} expected")
                
                # Verifica dimensione file
                f.seek(0, 2)
                file_size = f.tell()
                
                # Dimensione minima = header + data
                min_size = 128 + (expected_frames * self.bytes_per_frame)
                
                if file_size < min_size:
                    print(f"❌ Validation failed: File too small ({file_size} < {min_size})")
                    return False
                
                print(f"✅ Validation passed: {file_size} bytes, {data_points} data points")
                return True
                
        except Exception as e:
            print(f"❌ Validation error: {e}")
            return False
