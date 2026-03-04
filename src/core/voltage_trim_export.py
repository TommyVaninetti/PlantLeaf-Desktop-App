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
            backup_path = None  # ✅ Inizializza sempre
            
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
            analyses_with_insufficient_pre_data = []
            
            if params['include_analyses']:
                progress.setLabelText("Reading saved analyses...")
                progress.setValue(15)
                
                all_analyses = self.parent._read_analyses_from_file()
                
                for analysis_id, analysis in all_analyses.items():
                    a_start = analysis['parameters']['general']['start_time']
                    a_end = analysis['parameters']['general']['end_time']
                    
                    # Includi solo se COMPLETAMENTE nel range
                    if start_time <= a_start and a_end <= end_time:
                        # DEEP COPY completo dell'analisi
                        import copy
                        new_analysis = copy.deepcopy(analysis)
                        
                        # Re-timestamp di TUTTI i parametri temporali
                        time_offset = start_time
                        
                        # 1. Parametri generali
                        new_analysis['parameters']['general']['start_time'] = a_start - time_offset
                        new_analysis['parameters']['general']['end_time'] = a_end - time_offset
                        
                        # 2. time_peak (se presente)
                        if 'time_peak' in new_analysis['parameters']['general']:
                            orig_time_peak = new_analysis['parameters']['general']['time_peak']
                            new_analysis['parameters']['general']['time_peak'] = orig_time_peak - time_offset
                        
                        # 3. Exponential Return: t0
                        if 'exponential_return' in new_analysis['parameters']:
                            if 't0' in new_analysis['parameters']['exponential_return']:
                                orig_t0 = new_analysis['parameters']['exponential_return']['t0']
                                new_analysis['parameters']['exponential_return']['t0'] = orig_t0 - time_offset
                        
                        # 4. Action Potential: t_peak
                        if 'action_potential' in new_analysis['parameters']:
                            if 't_peak' in new_analysis['parameters']['action_potential']:
                                orig_t_peak = new_analysis['parameters']['action_potential']['t_peak']
                                new_analysis['parameters']['action_potential']['t_peak'] = orig_t_peak - time_offset
                        
                        filtered_analyses[analysis_id] = new_analysis
                        # ✅ CONTROLLO: Verifica se l'analisi avrà abbastanza dati "pre" nel file trimmed
                        new_start = new_analysis['parameters']['general']['start_time']
                        min_pre_time = 2.0  # Serve almeno 2s prima per la baseline

                        if new_start < min_pre_time:
                            analysis_name = new_analysis.get('metadata', {}).get('name', 'Unnamed')
                            insufficient_time = min_pre_time - new_start
                            analyses_with_insufficient_pre_data.append({
                                'id': analysis_id,
                                'name': analysis_name,
                                'start': new_start,
                                'missing': insufficient_time
                            })
                            print(f"⚠️  Warning: Analysis starts at {new_start:.3f}s (needs {insufficient_time:.3f}s more pre-data)")
                        
                        # Logging dettagliato per debug
                        print(f"\n✅ Analysis re-timestamped: {analysis_id}")
                        print(f"   Model: {new_analysis.get('model', {}).get('active_model', 'Unknown')}")
                        print(f"   Time range: {a_start:.3f}s → {a_end:.3f}s  ==>  "
                              f"{new_analysis['parameters']['general']['start_time']:.3f}s → "
                              f"{new_analysis['parameters']['general']['end_time']:.3f}s")
                        
                        # Log time_peak se presente
                        if 'time_peak' in new_analysis['parameters']['general']:
                            orig_tp = analysis['parameters']['general'].get('time_peak', 0)
                            new_tp = new_analysis['parameters']['general']['time_peak']
                            print(f"   time_peak: {orig_tp:.3f}s  ==>  {new_tp:.3f}s")
                        
                        # Log t0 (exponential) se presente
                        if 'exponential_return' in new_analysis['parameters']:
                            if 't0' in new_analysis['parameters']['exponential_return']:
                                orig_t0 = analysis['parameters']['exponential_return']['t0']
                                new_t0 = new_analysis['parameters']['exponential_return']['t0']
                                print(f"   exponential t0: {orig_t0:.3f}s  ==>  {new_t0:.3f}s")
                        
                        # Log t_peak (action potential) se presente
                        if 'action_potential' in new_analysis['parameters']:
                            if 't_peak' in new_analysis['parameters']['action_potential']:
                                orig_tp_ap = analysis['parameters']['action_potential']['t_peak']
                                new_tp_ap = new_analysis['parameters']['action_potential']['t_peak']
                                print(f"   action_potential t_peak: {orig_tp_ap:.3f}s  ==>  {new_tp_ap:.3f}s")
                
                print(f"\n📊 Analyses summary: {len(all_analyses)} total, {len(filtered_analyses)} exported")
                
                # ⚠️ MOSTRA WARNING se ci sono analisi problematiche
                if analyses_with_insufficient_pre_data:
                    warning_msg = "⚠️ Warning: Some analyses may not open correctly!\n\n"
                    warning_msg += f"{len(analyses_with_insufficient_pre_data)} analysis/analyses start before 2s:\n\n"
                    
                    for item in analyses_with_insufficient_pre_data:
                        warning_msg += f"• \"{item['name']}\"\n"
                        warning_msg += f"  Starts at: {item['start']:.3f}s\n"
                        warning_msg += f"  Missing pre-data: {item['missing']:.3f}s\n\n"
                    
                    warning_msg += "These analyses need at least 1 second of data before their start time\n"
                    warning_msg += "for proper baseline calculation.\n\n"
                    warning_msg += "RECOMMENDATION: Export a region that includes at least 1 second\n"
                    warning_msg += "before the first analysis start time.\n\n"
                    warning_msg += "Do you want to continue anyway?"
                    
                    from PySide6.QtWidgets import QMessageBox
                    reply = QMessageBox.warning(
                        self.parent,
                        "Insufficient Pre-Data for Analyses",
                        warning_msg,
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No
                    )
                    
                    if reply == QMessageBox.No:
                        progress.close()
                        print("❌ Export cancelled by user (insufficient pre-data)")
                        return False
                    else:
                        print("⚠️  User chose to continue despite warnings")
            
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
            
            # ✅ FIX: In overwrite mode, usa il backup come sorgente
            source_file = backup_path if params['overwrite_mode'] else self.file_path
            print(f"📖 Reading from: {os.path.basename(source_file)}")
            print(f"📝 Writing to: {os.path.basename(output_path)}")
            
            with open(output_path, 'wb') as out_f:
                # Scrivi header
                out_f.write(new_header)
                print(f"✅ Header written (128 bytes)")
                
                # Apri file sorgente (backup se overwrite, originale altrimenti)
                with open(source_file, 'rb') as in_f:
                    # Verifica dimensione file sorgente
                    in_f.seek(0, 2)  # Vai alla fine
                    source_size = in_f.tell()
                    in_f.seek(0)  # Torna all'inizio
                    print(f"📏 Source file size: {source_size} bytes ({source_size / (1024*1024):.2f} MB)")
                    
                    # Salta header (128 bytes)
                    in_f.seek(128)
                    
                    # Calcola offset dati da copiare
                    bytes_per_sample = 8  # 2 floats (4 bytes ciascuno)
                    start_offset = 128 + (start_sample * bytes_per_sample)
                    
                    print(f"📍 Seeking to offset: {start_offset} (start_sample={start_sample}, bytes_per_sample={bytes_per_sample})")
                    
                    # Posizionati all'inizio dei dati da copiare
                    in_f.seek(start_offset)
                    current_pos = in_f.tell()
                    print(f"📍 Current position after seek: {current_pos}")
                    
                    # Copia dati a blocchi con progress e RE-TIMESTAMP
                    chunk_size = 100000  # 100k samples per chunk
                    samples_written = 0
                    
                    while samples_written < new_data_points:
                        if progress.wasCanceled():
                            # Rimuovi file parziale
                            out_f.close()
                            if os.path.exists(output_path):
                                os.remove(output_path)
                            return False
                        
                        # Calcola quanti samples leggere in questo chunk
                        samples_in_chunk = min(chunk_size, new_data_points - samples_written)
                        
                        # Leggi chunk (ogni sample = 8 bytes: time + voltage)
                        chunk_bytes = in_f.read(samples_in_chunk * bytes_per_sample)
                        
                        if not chunk_bytes:
                            print(f"⚠️  No data read at position {in_f.tell()}, expected {samples_in_chunk * bytes_per_sample} bytes")
                            break
                        
                        # Processa chunk: ricalcola timestamp
                        processed_chunk = bytearray()
                        for i in range(0, len(chunk_bytes), 8):
                            # Leggi coppia (time_original, voltage)
                            time_orig = struct.unpack('<f', chunk_bytes[i:i+4])[0]
                            voltage = struct.unpack('<f', chunk_bytes[i+4:i+8])[0]
                            
                            # Ricalcola timestamp: nuovo_time = (samples_written + offset_in_chunk) / sampling_rate
                            offset_in_chunk = i // 8
                            new_time = (samples_written + offset_in_chunk) / sampling_rate
                            
                            # Scrivi coppia con nuovo timestamp
                            processed_chunk.extend(struct.pack('<f', new_time))
                            processed_chunk.extend(struct.pack('<f', voltage))
                        
                        # Scrivi chunk processato
                        out_f.write(processed_chunk)
                        samples_written += samples_in_chunk
                        
                        # Aggiorna progress (30-85%)
                        copy_progress = int(30 + (samples_written / new_data_points) * 55)
                        progress.setValue(copy_progress)
                        progress.setLabelText(
                            f"Re-timestamping data: {samples_written:,} / {new_data_points:,} samples"
                        )
                    
                    print(f"✅ Data re-timestamped and written: {samples_written} samples")
            
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
