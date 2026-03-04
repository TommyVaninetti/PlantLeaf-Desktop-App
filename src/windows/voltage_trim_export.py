"""
Export trimmed voltage files - Modulo separato per la logica di export
"""
import os
import struct
import json
import shutil
from datetime import datetime
from PySide6.QtWidgets import QProgressDialog, QMessageBox
from PySide6.QtCore import Qt


class VoltageTrimExporter:
    """Gestisce l'export di regioni trimmed per file voltage"""
    
    def __init__(self, parent_window):
        self.parent = parent_window
        self.file_path = parent_window.file_path
        self.metadata = parent_window.metadata
    
    def execute_trim_export(self, params):
        """
        Esegue l'export trimmed di un file voltage.
        
        Args:
            params: dict con:
                - start_time: float (secondi)
                - end_time: float (secondi)
                - output_path: str | None (None se overwrite)
                - overwrite_mode: bool
                - include_analyses: bool
        
        Returns:
            bool: True se successo, False altrimenti
        """
        try:
            # === FASE 1: VALIDAZIONE PARAMETRI ===
            start_time = params['start_time']
            end_time = params['end_time']
            duration = end_time - start_time
            
            if duration < 0.1:
                QMessageBox.critical(self.parent, "Invalid Range", 
                                   f"Duration too short: {duration:.2f}s")
                return False
            
            # === FASE 2: CALCOLA RANGE SAMPLE ===
            sampling_rate = self.metadata['sampling_rate']
            total_points = self.metadata['data_points']
            
            start_sample = int(start_time * sampling_rate)
            end_sample = int(end_time * sampling_rate)
            new_data_points = end_sample - start_sample
            
            print(f"📊 Trim Export Parameters:")
            print(f"   Time Range: {start_time:.2f}s - {end_time:.2f}s (duration: {duration:.2f}s)")
            print(f"   Sample Range: {start_sample} - {end_sample} (total: {new_data_points})")
            print(f"   Sampling Rate: {sampling_rate} Hz")
            
            # === FASE 3: SETUP PROGRESS DIALOG ===
            progress = QProgressDialog("Exporting trimmed file...", "Cancel", 0, 100, self.parent)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            progress.setWindowTitle("Export Progress")
            
            # === FASE 4: DETERMINA FILE OUTPUT ===
            if params['overwrite_mode']:
                # Crea backup
                backup_path = self.file_path.replace('.pvolt', '_backup.pvolt')
                progress.setLabelText(f"Creating backup: {os.path.basename(backup_path)}...")
                progress.setValue(5)
                
                shutil.copy2(self.file_path, backup_path)
                print(f"💾 Backup created: {backup_path}")
                
                output_path = self.file_path
            else:
                output_path = params['output_path']
                if not output_path:
                    QMessageBox.critical(self.parent, "Error", "No output path specified")
                    return False
            
            progress.setValue(10)
            if progress.wasCanceled():
                return False
            
            # === FASE 5: LEGGI E FILTRA ANALISI (se richiesto) ===
            filtered_analyses = {}
            if params['include_analyses']:
                progress.setLabelText("Reading saved analyses...")
                progress.setValue(15)
                
                all_analyses = self.parent._read_analyses_from_file()
                
                for analysis_id, analysis in all_analyses.items():
                    a_start = analysis['parameters']['general']['start_time']
                    a_end = analysis['parameters']['general']['end_time']
                    
                    # Includi solo se COMPLETAMENTE nel range
                    if start_time <= a_start and a_end <= end_time:
                        # Re-timestamp: sottrai start_time
                        new_analysis = analysis.copy()
                        new_analysis['parameters'] = analysis['parameters'].copy()
                        new_analysis['parameters']['general'] = analysis['parameters']['general'].copy()
                        new_analysis['parameters']['general']['start_time'] = a_start - start_time
                        new_analysis['parameters']['general']['end_time'] = a_end - start_time
                        
                        filtered_analyses[analysis_id] = new_analysis
                
                print(f"📊 Analyses: {len(all_analyses)} total, {len(filtered_analyses)} in range")
            
            progress.setValue(20)
            if progress.wasCanceled():
                return False
            
            # === FASE 6: CREA NUOVO HEADER ===
            progress.setLabelText("Creating new header...")
            progress.setValue(25)
            
            new_header = self._create_trimmed_header(
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                data_points=new_data_points
            )
            
            # === FASE 7: SCRIVI FILE OUTPUT ===
            progress.setLabelText(f"Writing to {os.path.basename(output_path)}...")
            progress.setValue(30)
            
            with open(output_path, 'wb') as out_f:
                # Scrivi header
                out_f.write(new_header)
                print(f"✅ Header written (128 bytes)")
                
                # Apri file sorgente
                with open(self.file_path, 'rb') as in_f:
                    # Salta header (128 bytes)
                    in_f.seek(128)
                    
                    # Calcola offset dati da copiare
                    bytes_per_sample = 8  # 2 floats (4 bytes ciascuno)
                    start_offset = 128 + (start_sample * bytes_per_sample)
                    bytes_to_copy = new_data_points * bytes_per_sample
                    
                    # Posizionati all'inizio dei dati da copiare
                    in_f.seek(start_offset)
                    
                    # Copia dati a blocchi con progress
                    chunk_size = 100000 * bytes_per_sample  # 100k samples per chunk
                    bytes_copied = 0
                    
                    while bytes_copied < bytes_to_copy:
                        if progress.wasCanceled():
                            # Rimuovi file parziale
                            out_f.close()
                            if os.path.exists(output_path):
                                os.remove(output_path)
                            return False
                        
                        # Leggi chunk
                        chunk_bytes = min(chunk_size, bytes_to_copy - bytes_copied)
                        data_chunk = in_f.read(chunk_bytes)
                        
                        if not data_chunk:
                            break
                        
                        # Scrivi chunk
                        out_f.write(data_chunk)
                        bytes_copied += len(data_chunk)
                        
                        # Aggiorna progress (30-85%)
                        copy_progress = int(30 + (bytes_copied / bytes_to_copy) * 55)
                        progress.setValue(copy_progress)
                        progress.setLabelText(
                            f"Copying data: {bytes_copied/(1024*1024):.1f} / "
                            f"{bytes_to_copy/(1024*1024):.1f} MB"
                        )
                    
                    print(f"✅ Data copied: {bytes_copied} bytes ({new_data_points} samples)")
            
            progress.setValue(85)
            
            # === FASE 8: SCRIVI FOOTER CON ANALISI (se presenti) ===
            if filtered_analyses:
                progress.setLabelText("Writing analyses footer...")
                progress.setValue(90)
                
                with open(output_path, 'r+b') as f:
                    # Vai alla fine del file (dopo i dati)
                    f.seek(0, 2)
                    footer_offset = f.tell()
                    
                    # Scrivi JSON footer
                    footer_data = {
                        "file_format_version": "1.0",
                        "analyses": filtered_analyses
                    }
                    json_str = json.dumps(footer_data, indent=2)
                    json_bytes = json_str.encode('utf-8')
                    
                    f.write(json_bytes)
                    
                    # Scrivi offset del footer (ultimi 8 byte)
                    f.write(struct.pack('<Q', footer_offset))
                    
                    print(f"✅ Footer written: {len(filtered_analyses)} analyses")
            
            progress.setValue(95)
            
            # === FASE 9: VALIDAZIONE FINALE ===
            progress.setLabelText("Validating output file...")
            
            if not self._validate_output_file(output_path, new_data_points):
                QMessageBox.warning(self.parent, "Validation Warning", 
                                  "File created but validation failed. Check manually.")
            
            progress.setValue(100)
            
            # === FASE 10: MESSAGGIO SUCCESSO ===
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            
            msg = f"✅ Export completed successfully!\n\n"
            msg += f"Output: {os.path.basename(output_path)}\n"
            msg += f"Duration: {duration:.2f}s\n"
            msg += f"Data Points: {new_data_points:,}\n"
            msg += f"File Size: {file_size_mb:.1f} MB\n"
            
            if params['include_analyses'] and filtered_analyses:
                msg += f"Analyses Exported: {len(filtered_analyses)}\n"
            
            if params['overwrite_mode']:
                msg += f"\n💾 Backup saved as:\n{os.path.basename(backup_path)}"
            
            QMessageBox.information(self.parent, "Export Successful", msg)
            
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
            'magic': b'PLANTVOLT',
            'version': orig_meta.get('version', 2.0),
            'experiment_type': orig_meta.get('experiment_type', 'Trimmed')[:20].ljust(20),
            'sampling_rate': orig_meta['sampling_rate'],
            'duration': duration,
            'amplified': orig_meta.get('amplified', False),
            'start_time': datetime.now().timestamp(),  # Nuovo timestamp
            'end_time': datetime.now().timestamp(),
            'data_points': data_points,
            'acquisition_count': orig_meta.get('acquisition_count', 0),
            'reserved': b'\x00' * 62
        }
        
        # Costruisci header bytes
        header_bytes = bytearray()
        header_bytes.extend(header_data['magic'][:9])  # 9
        header_bytes.extend(struct.pack('<f', header_data['version']))  # 4
        
        exp_type = header_data['experiment_type'].encode('ascii', errors='replace')[:20]
        exp_type += b'\x00' * (20 - len(exp_type))
        header_bytes.extend(exp_type)  # 20
        
        header_bytes.extend(struct.pack('<f', header_data['sampling_rate']))  # 4
        header_bytes.extend(struct.pack('<f', header_data['duration']))  # 4
        header_bytes.extend(struct.pack('<?', header_data['amplified']))  # 1
        header_bytes.extend(struct.pack('<d', header_data['start_time']))  # 8
        header_bytes.extend(struct.pack('<d', header_data['end_time']))  # 8
        header_bytes.extend(struct.pack('<I', header_data['data_points']))  # 4
        header_bytes.extend(struct.pack('<I', header_data['acquisition_count']))  # 4
        header_bytes.extend(header_data['reserved'][:62])  # 62
        
        # Verifica dimensione
        if len(header_bytes) != 128:
            raise ValueError(f"Header size mismatch: {len(header_bytes)} bytes (expected 128)")
        
        return bytes(header_bytes)
    
    def _validate_output_file(self, file_path, expected_points):
        """Valida il file output creato"""
        try:
            file_size = os.path.getsize(file_path)
            expected_size = 128 + (expected_points * 8)  # Header + data
            
            # Leggi header
            with open(file_path, 'rb') as f:
                header_bytes = f.read(128)
                
                # Verifica magic number
                if header_bytes[:9] != b'PLANTVOLT':
                    print("⚠️ Validation: Invalid magic number")
                    return False
                
                # Verifica data_points
                stored_points = struct.unpack('<I', header_bytes[58:62])[0]
                if stored_points != expected_points:
                    print(f"⚠️ Validation: Point count mismatch ({stored_points} vs {expected_points})")
                    return False
            
            print(f"✅ Validation passed: {file_size} bytes, {expected_points} points")
            return True
            
        except Exception as e:
            print(f"⚠️ Validation error: {e}")
            return False
