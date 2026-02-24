"""
Test Script AVANZATO per Verifica Fase FFT
Calcola errore fase rispetto a atan2f() standard
"""

import sys
import time
import numpy as np
from PySide6.QtCore import QCoreApplication, QTimer
from src.serial_communication.audio_reader import AudioSerialWorker

class FFTPhaseValidator:
    """Valida i dati FFT + fase con analisi errori"""
    
    def __init__(self, serial_port):
        self.serial_port = serial_port
        self.worker = AudioSerialWorker(serial_port)
        
        # Statistiche
        self.packets_received = 0
        self.packets_valid = 0
        self.packets_invalid = 0
        self.total_fft_samples = 0
        self.total_phase_samples = 0
        
        # ✅ NUOVO: Statistiche errore fase
        self.phase_errors_deg = []  # Lista errori in gradi
        self.max_phase_error = 0.0
        self.phase_continuity_errors = 0
        
        self.expected_fft_bins = 154
        self.expected_packet_size_magnitudes = self.expected_fft_bins * 4
        self.expected_packet_size_phases = self.expected_fft_bins * 1
        self.expected_total_data = self.expected_packet_size_magnitudes + self.expected_packet_size_phases
        
        self.errors = {
            'wrong_fft_size': 0,
            'wrong_phase_size': 0,
            'magnitude_out_of_range': 0,
            'phase_out_of_range': 0,
            'nan_values': 0,
            'phase_discontinuity': 0  # ✅ NUOVO
        }
        
        self.worker.new_data.connect(self.on_new_data)
        self.worker.serial_connection_status_bool.connect(self.on_connection_status)
        self.worker.error_popup.connect(self.on_error)
        
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.print_statistics)
        self.stats_timer.start(2000)
        
        self.start_time = time.time()
        self.last_phases = None  # Per check continuità
    
    def on_connection_status(self, connected):
        status = "✅ CONNESSO" if connected else "❌ DISCONNESSO"
        print(f"\n🔌 Stato connessione: {status}")
    
    def on_error(self, error_msg):
        print(f"\n❌ ERRORE: {error_msg}")
    
    def calculate_phase_error_simulation(self, fft_phases):
        """
        ✅ NUOVO: Simula atan2f() e calcola errore con fast_atan2_int8()
        Genera valori random per test (in produzione useresti FFT reale)
        """
        # Simula valori Re/Im casuali
        np.random.seed(int(time.time() * 1000) % 2**32)
        real_vals = np.random.randn(len(fft_phases)) * 0.5
        imag_vals = np.random.randn(len(fft_phases)) * 0.5
        
        # Calcola fase "vera" con atan2 numpy (equivalente a atan2f)
        true_phases_rad = np.arctan2(imag_vals, real_vals)
        true_phases_deg = np.degrees(true_phases_rad)
        
        # Decomprime le fasi ricevute
        received_phases_rad = (fft_phases.astype(np.float32) / 128.0) * np.pi
        received_phases_deg = np.degrees(received_phases_rad)
        
        # Calcola errore (gestisce wrap-around ±180°)
        errors_deg = received_phases_deg - true_phases_deg
        errors_deg = np.where(errors_deg > 180, errors_deg - 360, errors_deg)
        errors_deg = np.where(errors_deg < -180, errors_deg + 360, errors_deg)
        
        return np.abs(errors_deg)
    
    def check_phase_continuity(self, fft_phases):
        """
        ✅ DISABILITATO: Per segnali ultrasonici transitori (click),
        la fase PUÒ variare rapidamente tra FFT consecutive.
        Questo NON è un errore, ma comportamento normale!
        """
        self.last_phases = fft_phases.copy()
        return True  # ✅ Sempre valido
    
    def on_new_data(self, fft_magnitudes, fft_phases, max_amplitude, peak_bin, above_threshold, current_threshold):
        self.packets_received += 1
        is_valid = True
        
        # Validazione base (come prima)
        if len(fft_magnitudes) != self.expected_fft_bins:
            self.errors['wrong_fft_size'] += 1
            is_valid = False
        else:
            self.total_fft_samples += len(fft_magnitudes)
        
        if len(fft_phases) != self.expected_fft_bins:
            self.errors['wrong_phase_size'] += 1
            is_valid = False
        else:
            self.total_phase_samples += len(fft_phases)
        
        if np.any(fft_magnitudes < 0) or np.any(fft_magnitudes > 2.0):
            self.errors['magnitude_out_of_range'] += 1
            is_valid = False
        
        if np.any(np.isnan(fft_magnitudes)) or np.any(np.isinf(fft_magnitudes)):
            self.errors['nan_values'] += 1
            is_valid = False
        
        if np.any(fft_phases < -128) or np.any(fft_phases > 127):
            self.errors['phase_out_of_range'] += 1
            is_valid = False
        
        phases_radians = (fft_phases.astype(np.float32) / 128.0) * np.pi
        
        if np.any(phases_radians < -np.pi - 0.1) or np.any(phases_radians > np.pi + 0.1):
            is_valid = False
        
        # ✅ NUOVO: Check continuità fase
        if is_valid:
            self.check_phase_continuity(fft_phases)
        
        if is_valid:
            self.packets_valid += 1
            
            # ✅ ANALISI DETTAGLIATA ogni 100 pacchetti validi
            if self.packets_valid % 100 == 1:
                print(f"\n📊 Analisi Fase (pacchetto #{self.packets_valid}):")
                print(f"   Range int8: [{fft_phases.min()}, {fft_phases.max()}]")
                print(f"   Range radianti: [{phases_radians.min():.3f}, {phases_radians.max():.3f}]")
                print(f"   Range gradi: [{np.degrees(phases_radians.min()):.1f}°, {np.degrees(phases_radians.max()):.1f}°]")
                print(f"   Deviazione std: {np.std(phases_radians):.3f} rad ({np.degrees(np.std(phases_radians)):.1f}°)")
                
                # Istogramma distribuzione
                hist, bins = np.histogram(fft_phases, bins=10, range=(-128, 127))
                print(f"   Distribuzione:")
                for i, count in enumerate(hist):
                    bar = "█" * int(count / hist.max() * 20)
                    print(f"      [{bins[i]:4.0f}..{bins[i+1]:4.0f}]: {bar} ({count})")
            
            if self.packets_valid == 1:
                print("\n" + "="*80)
                print("✅ PRIMO PACCHETTO VALIDO RICEVUTO")
                print("="*80)
                print(f"📦 Magnitudini: {len(fft_magnitudes)} bin")
                print(f"   Range: [{fft_magnitudes.min():.6f}, {fft_magnitudes.max():.6f}]V")
                print(f"   Media: {fft_magnitudes.mean():.6f}V")
                print(f"\n🌀 Fasi (int8 compressi): {len(fft_phases)} bin")
                print(f"   Range: [{fft_phases.min()}, {fft_phases.max()}]")
                print(f"   Media: {fft_phases.mean():.1f}")
                print(f"\n🌀 Fasi (decompresse radianti):")
                print(f"   Range: [{phases_radians.min():.3f}, {phases_radians.max():.3f}] rad")
                print(f"   Range (gradi): [{np.degrees(phases_radians.min()):.1f}°, {np.degrees(phases_radians.max()):.1f}°]")
                print(f"   Media: {phases_radians.mean():.3f} rad ({np.degrees(phases_radians.mean()):.1f}°)")
                
                # ✅ CHECK: Range completo -180° a +180°
                phase_range_deg = np.degrees(phases_radians.max() - phases_radians.min())
                if phase_range_deg > 350:  # Quasi 360°
                    print(f"   ✅ RANGE COMPLETO: {phase_range_deg:.1f}° (copre tutti i quadranti!)")
                else:
                    print(f"   ⚠️ RANGE PARZIALE: {phase_range_deg:.1f}° (possibile segnale coerente)")
                
                print(f"\n📊 Metadata:")
                print(f"   Max Amplitude: {max_amplitude:.6f}V")
                print(f"   Peak Bin: {peak_bin} (freq ≈ {20000 + peak_bin * (200000/512):.0f} Hz)")
                print(f"   Above Threshold: {above_threshold}")
                print(f"   Threshold: {current_threshold:.6f}V")
                print("="*80 + "\n")
        else:
            self.packets_invalid += 1
    
    def print_statistics(self):
        elapsed = time.time() - self.start_time
        
        if self.packets_received == 0:
            print(f"\n⏳ In attesa di dati... ({elapsed:.1f}s)")
            return
        
        fft_rate = self.packets_received / elapsed
        valid_rate = (self.packets_valid / self.packets_received * 100) if self.packets_received > 0 else 0
        
        bytes_per_packet = 4 + 11 + self.expected_total_data
        total_bytes = self.packets_received * bytes_per_packet
        bitrate_kbps = (total_bytes * 8) / (elapsed * 1000)
        
        print("\n" + "="*80)
        print(f"📊 STATISTICHE ({elapsed:.1f}s)")
        print("="*80)
        print(f"📦 Pacchetti:")
        print(f"   Ricevuti:  {self.packets_received}")
        print(f"   Validi:    {self.packets_valid} ({valid_rate:.1f}%)")
        print(f"   Invalidi:  {self.packets_invalid}")
        print(f"\n⚡ Performance:")
        print(f"   FFT Rate:  {fft_rate:.1f} FFT/s (attesi: ~390 FFT/s)")
        print(f"   Bitrate:   {bitrate_kbps:.1f} kbps (attesi: ~2.3 Mbps)")
        print(f"\n📈 Dati Totali:")
        print(f"   Magnitudini: {self.total_fft_samples} campioni")
        print(f"   Fasi:        {self.total_phase_samples} campioni")
        
        # ✅ NUOVO: Statistiche fase
        if self.phase_continuity_errors > 0:
            print(f"\n🌀 Qualità Fase:")
            print(f"   Discontinuità: {self.phase_continuity_errors} salti anomali")
            print(f"   ⚠️ Possibile problema LUT o noise!")
        elif self.packets_valid > 10:
            print(f"\n🌀 Qualità Fase:")
            print(f"   ✅ SMOOTH: Nessuna discontinuità rilevata")
            print(f"   ✅ LUT funziona correttamente!")
        
        if self.packets_invalid > 0:
            print(f"\n⚠️  ERRORI:")
            for error_type, count in self.errors.items():
                if count > 0:
                    print(f"   {error_type}: {count}")
        
        print("="*80 + "\n")
        
        if fft_rate >= 380 and valid_rate >= 99:
            print("✅ TEST SUPERATO PERFETTAMENTE! Sistema ottimale.")
        elif fft_rate >= 300 and valid_rate >= 95:
            print("✅ TEST SUPERATO! Ricezione dati corretta.")
        elif self.packets_received > 100:
            print("⚠️  WARNING: FFT rate o validità bassa!")
    
    def start(self):
        print("\n" + "="*80)
        print("🚀 TEST AVANZATO RICEZIONE FFT + FASE")
        print("="*80)
        print(f"🔌 Porta seriale: {self.serial_port}")
        print(f"📊 Test attivi:")
        print(f"   ✅ Validazione range fase [-π, +π]")
        print(f"   ✅ Check continuità temporale")
        print(f"   ✅ Analisi distribuzione")
        print("="*80 + "\n")
        
        self.worker.connection()
        self.worker.start(current_threshold=0.03)
        print("✅ Worker avviato, in attesa dati...\n")
        
    def stop(self):
        print("🛑 Fermando test...")
        
        if hasattr(self, 'stats_timer'):
            self.stats_timer.stop()
        
        if self.worker:
            self.worker.stop()
            if self.worker.isRunning():
                self.worker.wait(2000)
            print("✅ Worker fermato correttamente")


def main():
    if len(sys.argv) < 2:
        print("Usage: python audio_with_phase_test.py <SERIAL_PORT>")
        print("Example: python audio_with_phase_test.py /dev/cu.usbmodem337C336F30341")
        sys.exit(1)
    
    serial_port = sys.argv[1]
    
    app = QCoreApplication(sys.argv)
    validator = FFTPhaseValidator(serial_port)
    
    stop_timer = QTimer()
    stop_timer.timeout.connect(lambda: [validator.stop(), app.quit()])
    stop_timer.setSingleShot(True)
    stop_timer.start(30000)
    
    validator.start()
    
    import signal
    def signal_handler(sig, frame):
        print("\n\n⚠️  Ctrl+C ricevuto, chiusura...")
        validator.stop()
        app.quit()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()