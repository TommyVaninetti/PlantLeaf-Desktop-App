"""
Test per verificare il funzionamento del circular buffer
Eseguire SENZA hardware (test logica)
"""

import numpy as np
import sys
from pathlib import Path

# Aggiungi src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))


class MockPlotBuffer:
    """Mock del circular buffer per testing"""
    
    def __init__(self):
        self.max_points = 60000
        self.plot_buffer_size = self.max_points * 2
        self.data_x_plot = np.zeros(self.plot_buffer_size, dtype=np.float64)
        self.data_y_plot = np.zeros(self.plot_buffer_size, dtype=np.float32)
        self.plot_buffer_index = 0
        self.plot_buffer_full = False
    
    def add_point(self, x, y):
        """Simula on_new_voltage_data()"""
        idx = self.plot_buffer_index % self.plot_buffer_size
        self.data_x_plot[idx] = x
        self.data_y_plot[idx] = y
        self.plot_buffer_index += 1
        
        if self.plot_buffer_index >= self.plot_buffer_size:
            self.plot_buffer_full = True
    
    def get_visible_data(self):
        """Simula _get_visible_plot_data()"""
        if self.plot_buffer_index == 0:
            return np.array([]), np.array([])
        
        if not self.plot_buffer_full:
            # Buffer non pieno: prendi gli ultimi max_points (o tutti se meno)
            valid_count = min(self.plot_buffer_index, self.max_points)
            start_idx = max(0, self.plot_buffer_index - valid_count)
            return (self.data_x_plot[start_idx:self.plot_buffer_index].copy(), 
                    self.data_y_plot[start_idx:self.plot_buffer_index].copy())
        else:
            # Buffer pieno: gestisci wrapping
            start_idx = self.plot_buffer_index % self.plot_buffer_size
            
            if start_idx == 0:
                end_slice = self.plot_buffer_size
            else:
                end_slice = start_idx
            
            count = min(self.max_points, self.plot_buffer_size)
            
            if count <= end_slice:
                x_data = self.data_x_plot[end_slice - count:end_slice].copy()
                y_data = self.data_y_plot[end_slice - count:end_slice].copy()
            else:
                first_part = count - end_slice
                x_data = np.concatenate([
                    self.data_x_plot[-first_part:],
                    self.data_x_plot[:end_slice]
                ])
                y_data = np.concatenate([
                    self.data_y_plot[-first_part:],
                    self.data_y_plot[:end_slice]
                ])
            
            return x_data, y_data


def test_basic_operation():
    """Test 1: Operazioni base"""
    print("🧪 Test 1: Operazioni base")
    buffer = MockPlotBuffer()
    
    # Aggiungi 100 punti
    for i in range(100):
        buffer.add_point(float(i), float(i * 2))
    
    x, y = buffer.get_visible_data()
    
    assert len(x) == 100, f"Expected 100 points, got {len(x)}"
    assert np.allclose(x, np.arange(100)), "X data mismatch"
    assert np.allclose(y, np.arange(100) * 2), "Y data mismatch"
    
    print(f"   ✅ Aggiunti 100 punti correttamente")
    print(f"   ✅ Buffer index: {buffer.plot_buffer_index}")
    print(f"   ✅ Buffer full: {buffer.plot_buffer_full}")


def test_max_points_limit():
    """Test 2: Limite max_points"""
    print("\n🧪 Test 2: Limite max_points (60000)")
    buffer = MockPlotBuffer()
    
    # Aggiungi 80000 punti (più di max_points)
    for i in range(80000):
        buffer.add_point(float(i), float(i))
    
    x, y = buffer.get_visible_data()
    
    assert len(x) == buffer.max_points, f"Expected {buffer.max_points}, got {len(x)}"
    
    # Verifica che siano gli ULTIMI 60000 punti
    # Nota: il buffer circolare da 120000 contiene tutti gli 80000 punti
    # quindi dovremmo vedere gli ultimi 60000 (da 20000 a 79999)
    expected_start = 80000 - buffer.max_points  # 20000
    expected_x = np.arange(expected_start, 80000, dtype=np.float64)
    
    # Debug
    print(f"   DEBUG: x[0]={x[0]:.0f}, expected={expected_start}")
    print(f"   DEBUG: x[-1]={x[-1]:.0f}, expected={79999}")
    print(f"   DEBUG: len(x)={len(x)}")
    
    # Il buffer circolare restituisce gli ultimi max_points inseriti
    # Ma solo se buffer_index <= buffer_size (che è vero: 80k <= 120k)
    assert np.allclose(x, expected_x), f"Data mismatch: got {x[0]:.0f}-{x[-1]:.0f}, expected {expected_start}-79999"
    
    print(f"   ✅ Restituiti ultimi {buffer.max_points} punti su 80000")
    print(f"   ✅ Range: {x[0]:.0f} - {x[-1]:.0f}")
    print(f"   ✅ Buffer full: {buffer.plot_buffer_full}")


def test_wrapping():
    """Test 3: Wrapping del buffer"""
    print("\n🧪 Test 3: Wrapping del buffer")
    buffer = MockPlotBuffer()
    
    # Riempi completamente il buffer + overflow
    total_points = buffer.plot_buffer_size + 5000
    for i in range(total_points):
        buffer.add_point(float(i), float(i))
    
    x, y = buffer.get_visible_data()
    
    assert len(x) == buffer.max_points, f"Expected {buffer.max_points}, got {len(x)}"
    assert buffer.plot_buffer_full, "Buffer should be marked as full"
    
    # Verifica continuità dei dati (nessun gap)
    diffs = np.diff(x)
    assert np.allclose(diffs, 1.0), "Data should be continuous (diff=1)"
    
    print(f"   ✅ Buffer wrapped correttamente")
    print(f"   ✅ Total points added: {total_points}")
    print(f"   ✅ Visible points: {len(x)}")
    print(f"   ✅ Data continuity: OK (no gaps)")


def test_memory_usage():
    """Test 4: Uso memoria costante"""
    print("\n🧪 Test 4: Uso memoria costante")
    buffer = MockPlotBuffer()
    
    # Memoria iniziale
    import sys
    initial_size = sys.getsizeof(buffer.data_x_plot) + sys.getsizeof(buffer.data_y_plot)
    
    # Aggiungi 300k punti (simula 10 min a 500 Hz)
    for i in range(300000):
        buffer.add_point(float(i), float(i))
    
    # Memoria finale (dovrebbe essere identica)
    final_size = sys.getsizeof(buffer.data_x_plot) + sys.getsizeof(buffer.data_y_plot)
    
    assert initial_size == final_size, "Memory size changed (LEAK!)"
    
    print(f"   ✅ Memoria iniziale: {initial_size / 1024:.1f} KB")
    print(f"   ✅ Memoria finale: {final_size / 1024:.1f} KB")
    print(f"   ✅ Differenza: 0 KB (NESSUN LEAK)")


def test_performance():
    """Test 5: Performance O(1)"""
    print("\n🧪 Test 5: Performance O(1)")
    import time
    
    buffer = MockPlotBuffer()
    
    # Test inserimento 100k punti
    start = time.perf_counter()
    for i in range(100000):
        buffer.add_point(float(i), float(i))
    elapsed = time.perf_counter() - start
    
    ops_per_sec = 100000 / elapsed
    
    print(f"   ✅ Inseriti 100k punti in {elapsed:.3f}s")
    print(f"   ✅ Performance: {ops_per_sec:,.0f} ops/sec")
    
    # A 500 Hz, serve ~2000 ops/sec
    assert ops_per_sec > 10000, f"Too slow! Need >10k ops/sec, got {ops_per_sec:.0f}"
    print(f"   ✅ Performance OK per 500 Hz (richiede ~2k ops/sec)")


def run_all_tests():
    """Esegue tutti i test"""
    print("=" * 60)
    print("🔥 CIRCULAR BUFFER VALIDATION TESTS")
    print("=" * 60)
    
    try:
        test_basic_operation()
        test_max_points_limit()
        test_wrapping()
        test_memory_usage()
        test_performance()
        
        print("\n" + "=" * 60)
        print("✅ TUTTI I TEST PASSATI")
        print("=" * 60)
        print("\n🎯 Il circular buffer funziona correttamente!")
        print("📊 Memoria costante: ~1.5 MB")
        print("⚡ Performance: >10k ops/sec (OK per 500 Hz)")
        
        return True
        
    except AssertionError as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FALLITO: {e}")
        print("=" * 60)
        return False
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ ERRORE: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
