#!/usr/bin/env python3
"""
📊 TEST REALTÀ SISTEMA PLANTLEAF
===============================

NO TEORIE, NO CALCOLI SBAGLIATI.
Solo misurazione diretta di cosa succede VERAMENTE.

Usage: python3 real_test.py [porta_seriale]
"""

import sys
import time
import os

# Aggiungi src al path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)

print(f"📂 Project root: {project_root}")
print(f"📂 Src path: {src_path}")

try:
    from PySide6.QtCore import QCoreApplication
    from serial_communication.audio_reader import AudioSerialWorker
    print("✅ Import riusciti")
except ImportError as e:
    print(f"❌ Errore import: {e}")
    sys.exit(1)

class RealTest:
    def __init__(self, port="/dev/cu.usbmodem337C336F30341"):
        self.port = port
        self.packet_count = 0
        self.start_time = None
        self.worker = None
        
    def start_test(self):
        print(f"\n🔌 TEST REALTÀ - Connessione a {self.port}")
        print("=" * 50)
        
        self.worker = AudioSerialWorker(self.port)
        self.worker.new_data.connect(self.on_data_received)
        self.worker.error_popup.connect(self.on_error)
        
        try:
            # Prima connetti
            self.worker.connection()
            
            # Poi avvia
            self.worker.start()
            
            print("✅ Connesso! Misurazione in corso...")
            self.start_time = time.time()
            return True
            
        except Exception as e:
            print(f"❌ Errore: {e}")
            return False
    
    def on_data_received(self, amplitudes, peak_bin, click_detected, click_duration_us, click_peak_amp):
        self.packet_count += 1
        self.last_amplitudes = amplitudes
        if click_detected:
            print(f"🔊 Click rilevato! Durata: {click_duration_us} μs | Ampiezza: {click_peak_amp:.3f} V | Bin: {peak_bin}")
        if self.packet_count % 100 == 0:
            elapsed = time.time() - self.start_time
            rate = self.packet_count / elapsed
            print(f"📦 FFT: {self.packet_count} | Rate: {rate:.1f}/s | Bin: {peak_bin}")

    def on_error(self, error_msg):
        print(f"❌ Errore seriale: {error_msg}")
    
    def stop_test(self):
        if self.worker:
            self.worker.stop()
        
        if self.start_time:
            elapsed = time.time() - self.start_time
            rate = self.packet_count / elapsed
            
            print(f"\n📊 RISULTATI REALI:")
            print(f"⏱️  Durata: {elapsed:.1f}s")
            print(f"📦 FFT ricevute: {self.packet_count}")
            print(f"📈 Rate REALE: {rate:.1f} FFT/s")
            print(f"🎯 Questo è il rate VERO del sistema!")
            
            # Calcoliamo i byte reali
            data_per_fft = len(self.last_amplitudes) if hasattr(self, 'last_amplitudes') else 155  # Stima
            bytes_per_fft = data_per_fft * 4  # float32
            total_bytes = self.packet_count * bytes_per_fft
            
            print(f"\n📊 ANALISI TRAFFICO USB:")
            print(f"   Punti per FFT: ~{data_per_fft}")
            print(f"   Byte per FFT: ~{bytes_per_fft}")
            print(f"   Byte totali: {total_bytes}")
            print(f"   Throughput USB: {total_bytes/elapsed/1024:.1f} KB/s")

def main():
    app = QCoreApplication(sys.argv)
    
    port = "/dev/cu.usbmodem337C336F30341"
    if len(sys.argv) > 1:
        port = sys.argv[1]
    
    test = RealTest(port)
    
    if test.start_test():
        try:
            # Timer per auto-stop
            from PySide6.QtCore import QTimer
            timer = QTimer()
            timer.timeout.connect(lambda: [test.stop_test(), app.quit()])
            timer.start(30000)  # 30 secondi
            
            print("⏳ Test 30 secondi... (Ctrl+C per fermare)")
            app.exec()
            
        except KeyboardInterrupt:
            print("\n🛑 Test interrotto")
            test.stop_test()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
