import serial
import threading
import time
import sys

# --- CONFIGURA QUI ---
PORTA_SERIALE = "/dev/cu.usbmodem335C335C31341"  # Modifica con la tua porta seriale
BAUDRATE = 115200
CAMPIONI_PER_BLOCCO = 256
# ---------------------

# Variabili globali per il thread di lettura
ser = None
running = True

def reader_thread():
    """
    Questo thread legge continuamente dalla porta seriale
    e stampa tutto ciò che riceve.
    """
    print("--- Avvio del thread di lettura ---")
    
    campioni_contati = 0
    blocco_contato = 0
    
    # Timestamp per misurare la velocità
    primo_campione_ts = None
    ultimo_campione_ts = None

    while running:
        if ser is None or not ser.is_open:
            time.sleep(0.1)
            continue
            
        try:
            # Leggiamo una riga (fino a '\n')
            linea_bytes = ser.readline()
            
            if not linea_bytes:
                continue # Timeout o porta chiusa

            # Registra il timestamp di arrivo
            timestamp_arrivo = time.perf_counter()

            # Decodifica da bytes a stringa, ignorando errori
            linea_str = linea_bytes.decode('utf-8', errors='ignore').strip()

            if not linea_str:
                continue

            # Logica di analisi
            if "END_BLOCK" in linea_str:
                blocco_contato += 1
                durata_blocco = (timestamp_arrivo - primo_campione_ts) * 1000  # in ms
                
                print("\n----------------------------------------------------")
                print(f"✅ BLOCCO {blocco_contato} RICEVUTO ({campioni_contati} campioni)")
                print(f"   -> Tempo di arrivo del blocco: {durata_blocco:.2f} ms")
                
                # Calcola la velocità di ricezione (quanti campioni al secondo
                # sono arrivati in quel burst)
                velocita_burst = campioni_contati / (durata_blocco / 1000)
                print(f"   -> Velocità 'Burst' interna: {velocita_burst:.0f} campioni/s")
                
                if ultimo_campione_ts:
                    # Calcola il tempo di "pausa" tra la fine
                    # del blocco precedente e l'inizio di questo.
                    pausa_ms = (primo_campione_ts - ultimo_campione_ts) * 1000
                    print(f"   -> Tempo di 'Pausa' dal blocco precedente: {pausa_ms:.2f} ms")

                print("----------------------------------------------------\n")

                # Salva il timestamp di questo "END_BLOCK"
                ultimo_campione_ts = timestamp_arrivo
                
                # Resetta i contatori per il prossimo blocco
                campioni_contati = 0
                primo_campione_ts = None
                
            elif linea_str.startswith("Comando") or \
                 linea_str.startswith("Errore") or \
                 linea_str.startswith("Offset") or \
                 linea_str.startswith("Avvio"):
                # Messaggio di stato dal firmware
                print(f"[FIRMWARE] {linea_str}")
                
            else:
                # È un dato numerico
                if campioni_contati == 0:
                    # Questo è il primo campione del blocco
                    primo_campione_ts = timestamp_arrivo
                
                campioni_contati += 1
                
                # Stampa solo il primo e l'ultimo campione per non
                # intasare il terminale. Commenta per vederli tutti.
                if campioni_contati == 1 or campioni_contati == CAMPIONI_PER_BLOCCO:
                     print(f"  ... Campione {campioni_contati}: {linea_str}")
                elif campioni_contati == 2:
                     print("      ...")

        except serial.SerialException as e:
            print(f"--- Errore seriale: {e} ---")
            break
        except Exception as e:
            print(f"--- Errore sconosciuto: {e} ---")
            pass # Ignora errori di decodifica minori

    print("--- Thread di lettura terminato ---")


def main():
    global ser, running
    
    print(f"--- Script di Diagnosi Seriale ---")
    print(f"Collegamento a {PORTA_SERIALE} a {BAUDRATE} baud...")

    try:
        ser = serial.Serial(PORTA_SERIALE, BAUDRATE, timeout=0.1)
    except serial.SerialException as e:
        print(f"ERRORE: Impossibile aprire la porta {PORTA_SERIALE}.")
        print(f"{e}")
        return

    # Aggiungi un piccolo ritardo DOPO l'apertura
    # per la "race condition" di cui abbiamo discusso.
    print("Porta aperta. Attendo 200ms che l'STM32 sia pronto...")
    time.sleep(0.2)
    ser.flushInput() # Pulisce qualsiasi vecchio dato
    print("Pronto.")

    # Avvia il thread che legge
    t = threading.Thread(target=reader_thread, daemon=True)
    t.start()

    print("\n--- Inserisci i comandi (es. !offset, !start 500, !stop) ---")
    print("--- Digita 'exit' per uscire ---")

    try:
        while True:
            cmd = input("> ")
            if cmd.lower() == 'exit':
                break
            
            # Invia il comando con il '\n' (Invio)
            ser.write(f"{cmd}\n".encode('utf-8'))
            
    except KeyboardInterrupt:
        print("--- Uscita forzata ---")
    finally:
        running = False
        if ser and ser.is_open:
            ser.write(b"!stop\n") # Prova a fermare il dispositivo
            ser.close()
        print("--- Connessione chiusa. ---")
        t.join(timeout=1) # Attendi il thread

if __name__ == "__main__":
    main()