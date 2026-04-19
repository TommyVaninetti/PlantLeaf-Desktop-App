#!/usr/bin/env python3
"""
csv_to_pvolt.py  —  Converti file CSV Logger Pro → formato .pvolt
═══════════════════════════════════════════════════════════════════

Formato CSV atteso (Logger Pro, separatore ; oppure ,):
  • Righe di intestazione (opzionali) prima della riga delle unità
  • Prima colonna = tempo (s)  — Logger Pro la chiama "Time" o "Tempo"
  • Seconda colonna = tensione (V o mV) — Logger Pro la chiama "Potential" / "Voltage"
  • Separatore: ; (europeo) o , (anglosassone) — rilevato automaticamente
  • Decimale: , oppure .  — rilevato automaticamente

Formato binario .pvolt  (v2.0):
  ┌─────────────────────────────────────────────────────────────────┐
  │ HEADER  128 byte (little-endian)                                │
  │  [0:9]   magic         b'PLANTVOLT'         9 B                 │
  │  [9:13]  version       float32  (2.0)       4 B                 │
  │ [13:33]  experiment_type  ASCII 20 B (null-padded)              │
  │ [33:37]  sampling_rate float32              4 B                 │
  │ [37:41]  duration      float32  (s)         4 B                 │
  │ [41:42]  amplified     bool                 1 B                 │
  │ [42:50]  start_time    float64  (UNIX ts)   8 B  ← v2 only      │
  │ [50:58]  end_time      float64  (UNIX ts)   8 B  ← v2 only      │
  │ [58:62]  data_points   uint32               4 B                 │
  │ [62:66]  acquisition_count uint32           4 B                 │
  │ [66:128] reserved      62 B (zeroed)                            │
  └─────────────────────────────────────────────────────────────────┘
  Dati: sequenza di coppie float32 (time_s, voltage_V)
        → ogni punto = 8 byte, little-endian

Utilizzo:
  python csv_to_pvolt.py                          # selezione guidata
  python csv_to_pvolt.py file.csv                 # converti un file
  python csv_to_pvolt.py file1.csv file2.csv ...  # batch
"""

import sys
import os
import struct
import csv
import math
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════════

def _ask(prompt, default=None, choices=None, cast=str):
    """Prompt interattivo con valore di default e validazione."""
    hint = ""
    if default is not None:
        hint += f" [{default}]"
    if choices:
        hint += f" ({'/'.join(str(c) for c in choices)})"
    while True:
        raw = input(f"{prompt}{hint}: ").strip()
        if raw == "" and default is not None:
            return cast(default)
        if not raw:
            print("  ⚠️  Valore obbligatorio.")
            continue
        if choices and raw not in [str(c) for c in choices]:
            print(f"  ⚠️  Scegli tra: {choices}")
            continue
        try:
            return cast(raw)
        except (ValueError, TypeError):
            print(f"  ⚠️  Valore non valido (atteso {cast.__name__}).")


def _detect_delimiter(path):
    """Rileva il separatore del CSV (';' o ',') e il separatore decimale."""
    with open(path, encoding='utf-8-sig', errors='replace') as f:
        lines = [f.readline() for _ in range(15)]

    # Conta i ';' nelle righe: se presenti → separatore campi = ';'
    semicolon_lines = [l for l in lines if l.strip() and l.count(';') > 0]
    comma_heavy     = [l for l in lines if l.strip() and l.count(',') > 1]

    if semicolon_lines:
        # Separatore campi = ';'
        # Per trovare il separatore decimale, salta la prima riga (intestazione)
        # e cerca nelle righe dati se ci sono virgole dentro le celle
        data_lines = semicolon_lines[1:] if len(semicolon_lines) > 1 else semicolon_lines
        decimal_sep = '.'
        for dl in data_lines:
            parts = dl.split(';')
            if any(',' in p for p in parts):
                decimal_sep = ','
                break
        return ';', decimal_sep

    if comma_heavy:
        return ',', '.'

    return ',', '.'


def _parse_loggerpro_csv(path):
    """
    Legge un file CSV Logger Pro e restituisce:
      (t_array, v_array, sampling_rate_hz, unit_voltage)

    Gestisce i formati:
      A) riga 0: nomi colonne, riga 1: unità, riga 2+: dati   (Logger Pro EN)
      B) riga 0: nomi colonne con unità inline (es. "Ultimo: Tempo (s)"),
         riga 1+: dati                                         (Logger Pro IT)
      C) solo dati numerici senza intestazione

    Separatori supportati: ';' o ','  —  Decimale: ',' o '.'
    """
    delim, dec_sep = _detect_delimiter(path)
    print(f"  Rilevato: separatore='{delim}', decimale='{dec_sep}'")

    rows = []
    with open(path, encoding='utf-8-sig', errors='replace') as f:
        reader = csv.reader(f, delimiter=delim)
        for row in reader:
            # Rimuovi virgolette esterne da ogni cella
            rows.append([c.strip().strip('"').strip("'") for c in row])

    if not rows:
        raise ValueError("File CSV vuoto.")

    # ── Trova la prima riga completamente numerica ────────────────
    def _is_numeric_cell(cell):
        c = cell.strip()
        if dec_sep != '.':
            c = c.replace(dec_sep, '.')
        try:
            float(c)
            return True
        except ValueError:
            return False

    def _is_numeric_row(row):
        """True se almeno 2 celle della riga sono numeri validi."""
        return sum(1 for c in row if c and _is_numeric_cell(c)) >= 2

    data_start = 0
    unit_row   = None
    header_row = None

    for i, row in enumerate(rows):
        if _is_numeric_row(row):
            data_start = i
            # Riga precedente: potrebbe essere unità oppure header con unità inline
            if i > 0:
                unit_row = i - 1
            if i > 1:
                header_row = i - 2
            break
    else:
        raise ValueError(
            "Impossibile trovare righe di dati numerici nel CSV.\n"
            f"  Separatore rilevato: '{delim}', Decimale: '{dec_sep}'\n"
            f"  Prime 3 righe: {rows[:3]}"
        )

    print(f"  Riga dati inizia a: {data_start}")
    if header_row is not None:
        print(f"  Intestazione colonne: {rows[header_row]}")
    if unit_row is not None:
        print(f"  Riga unità/descrizione: {rows[unit_row]}")

    # ── Determina unità della tensione ────────────────────────────
    # Cerca sia nella riga unità che nell'intestazione colonne (formato IT inline)
    unit_voltage = 'V'
    candidates = []
    if unit_row is not None and len(rows[unit_row]) >= 2:
        candidates.append(rows[unit_row][1])
    if header_row is not None and len(rows[header_row]) >= 2:
        candidates.append(rows[header_row][1])
    # Anche la riga immediatamente sopra i dati (data_start-1) se contiene testo
    if data_start > 0 and not _is_numeric_row(rows[data_start - 1]):
        candidates.append(rows[data_start - 1][1] if len(rows[data_start - 1]) >= 2 else '')

    for c in candidates:
        cl = c.lower()
        if 'mv' in cl or 'millivolt' in cl:
            unit_voltage = 'mV'
            break
        if '(v)' in cl or cl.strip() == 'v':
            unit_voltage = 'V'
            break

    print(f"  Unità tensione rilevata: {unit_voltage}")

    # ── Parsing dati ──────────────────────────────────────────────
    t_list, v_list = [], []
    skipped = 0
    for row in rows[data_start:]:
        if len(row) < 2:
            continue
        try:
            t_str = row[0].strip()
            v_str = row[1].strip()
            if dec_sep != '.':
                t_str = t_str.replace(dec_sep, '.')
                v_str = v_str.replace(dec_sep, '.')
            t = float(t_str)
            v = float(v_str)
            if math.isfinite(t) and math.isfinite(v):
                t_list.append(t)
                v_list.append(v)
            else:
                skipped += 1
        except (ValueError, IndexError):
            skipped += 1
            continue

    if skipped > 0:
        print(f"  ℹ️  {skipped} righe saltate (non numeriche o NaN).")

    if len(t_list) < 2:
        raise ValueError(f"Trovati solo {len(t_list)} punti validi nel CSV.")

    # ── Converti in Volt se necessario ───────────────────────────
    if unit_voltage == 'mV':
        v_list = [v / 1000.0 for v in v_list]
        print("  Conversione mV → V applicata.")

    # ── Stima sampling rate (mediana dei Δt) ─────────────────────
    diffs = sorted(t_list[i+1] - t_list[i] for i in range(min(500, len(t_list)-1)))
    median_dt = diffs[len(diffs) // 2]
    if median_dt <= 0:
        raise ValueError("Timestep non valido (dt ≤ 0) — controlla la colonna del tempo.")
    sampling_rate = round(1.0 / median_dt, 4)
    print(f"  Sampling rate stimato: {sampling_rate:.2f} Hz")

    return t_list, v_list, sampling_rate, unit_voltage


# ═══════════════════════════════════════════════════════════════════
# HEADER BUILDER
# ═══════════════════════════════════════════════════════════════════

def _build_header(
    experiment_type: str,
    sampling_rate: float,
    duration: float,
    amplified: bool,
    start_time: float,
    end_time: float,
    data_points: int,
    acquisition_count: int = 1,
) -> bytes:
    """
    Costruisce il blocco header da 128 byte nel formato .pvolt v2.0.

    Layout identico a _create_header() in main_window_voltage.py:
      [0:9]   b'PLANTVOLT'
      [9:13]  float32  version = 2.0
      [13:33] ASCII 20 byte  experiment_type  (null-padded)
      [33:37] float32  sampling_rate
      [37:41] float32  duration  (s)
      [41:42] bool     amplified
      [42:50] float64  start_time  (UNIX timestamp)
      [50:58] float64  end_time    (UNIX timestamp)
      [58:62] uint32   data_points
      [62:66] uint32   acquisition_count
      [66:128] 62 byte  reserved  (zeroed)
    """
    buf = bytearray()

    buf.extend(b'PLANTVOLT')                                   # [0:9]   9 B
    buf.extend(struct.pack('<f', 2.0))                         # [9:13]  4 B
    exp = experiment_type.encode('ascii', errors='replace')[:20]
    exp = exp + b'\x00' * (20 - len(exp))
    buf.extend(exp)                                            # [13:33] 20 B
    buf.extend(struct.pack('<f', sampling_rate))               # [33:37] 4 B
    buf.extend(struct.pack('<f', duration))                    # [37:41] 4 B
    buf.extend(struct.pack('<?', amplified))                   # [41:42] 1 B
    buf.extend(struct.pack('<d', start_time))                  # [42:50] 8 B  ← v2
    buf.extend(struct.pack('<d', end_time))                    # [50:58] 8 B  ← v2
    buf.extend(struct.pack('<I', data_points))                 # [58:62] 4 B
    buf.extend(struct.pack('<I', acquisition_count))           # [62:66] 4 B
    buf.extend(b'\x00' * 62)                                   # [66:128] 62 B

    assert len(buf) == 128, f"BUG: header = {len(buf)} byte (attesi 128)"
    return bytes(buf)


# ═══════════════════════════════════════════════════════════════════
# WRITER
# ═══════════════════════════════════════════════════════════════════

def _write_pvolt(out_path, header_bytes, t_array, v_array):
    """Scrive il file .pvolt: header + coppie float32 (t, v)."""
    data = bytearray()
    for t, v in zip(t_array, v_array):
        data.extend(struct.pack('<f', t))
        data.extend(struct.pack('<f', v))

    with open(out_path, 'wb') as f:
        f.write(header_bytes)
        f.write(data)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"  ✅ Scritto: {out_path}  ({len(t_array)} punti, {size_kb:.1f} KB)")


# ═══════════════════════════════════════════════════════════════════
# CONVERSIONE DI UN SINGOLO FILE
# ═══════════════════════════════════════════════════════════════════

def convert_file(csv_path, interactive=True, defaults=None):
    """
    Converte un singolo CSV → .pvolt.

    Parameters
    ----------
    csv_path     : str   percorso del file CSV sorgente
    interactive  : bool  se True, chiede i metadati a terminale
    defaults     : dict  valori pre-impostati (usati in batch mode)
    """
    defaults = defaults or {}

    print(f"\n{'─'*60}")
    print(f"📂 File:  {os.path.basename(csv_path)}")
    print(f"{'─'*60}")

    # ── Parsing CSV ──────────────────────────────────────────────
    t_list, v_list, sr_auto, unit_v = _parse_loggerpro_csv(csv_path)

    n_points  = len(t_list)
    t0        = t_list[0]
    duration  = t_list[-1] - t_list[0]

    print(f"  Punti validi    : {n_points}")
    print(f"  Durata          : {duration:.3f} s")
    print(f"  Intervallo t    : [{t_list[0]:.4f}, {t_list[-1]:.4f}] s")
    print(f"  Range tensione  : [{min(v_list):.4f}, {max(v_list):.4f}] V")

    # ── Normalizza il tempo a partire da 0 se necessario ─────────
    if abs(t0) > 0.001:
        print(f"  ℹ️  Offset temporale rilevato ({t0:.4f} s) — rimosso.")
        t_list = [t - t0 for t in t_list]

    # ── Controllo range tensione ──────────────────────────────────
    v_max_abs = max(abs(min(v_list)), abs(max(v_list)))
    PVOLT_RANGE = 1.65   # limite display ±1.65 V dell'app (amplified=True)
    scale_factor = 1.0

    if v_max_abs > PVOLT_RANGE:
        print(f"\n  ⚠️  Range tensione rilevato: ±{v_max_abs:.3f} V  (limite display app: ±{PVOLT_RANGE} V)")
        print(      "  ┌─────────────────────────────────────────────────────┐")
        print(      "  │  OPZIONI DI GESTIONE DEL RANGE:                     │")
        print(      "  │  1 = Scala linearmente in [-1.65, +1.65] V          │")
        print(      "  │      (R², forme d'onda e fit invarianti per scala)  │")
        print(      "  │  2 = Salva i valori originali (grafico clippato      │")
        print(      "  │      visivamente, ma calcoli su dati reali)          │")
        print(      "  └─────────────────────────────────────────────────────┘")
        choice = _ask("  Scegli", default='1', choices=['1', '2'])

        if choice == '1':
            scale_factor = PVOLT_RANGE / v_max_abs
            v_list = [v * scale_factor for v in v_list]
            print(f"\n  ✅ Scala applicata: ×{scale_factor:.6f}  "
                  f"(per tornare ai V originali: valore_display / {scale_factor:.6f})")
            print(      "  ℹ️  R², pendenza normalizzata e shape dei fit sono invarianti per scala lineare.")
            amplified_forced = True
        else:
            scale_factor = 1.0
            print(f"\n  ℹ️  Dati salvati a ±{v_max_abs:.3f} V originali.")
            print(      "  ℹ️  Il grafico mostrerà l'asse Y clippato a ±1.65 V ma i dati sono intatti.")
            amplified_forced = True   # usa comunque amplified=True per il range V (non mV)
    else:
        amplified_forced = None   # lascia decidere all'utente sotto

    # ── Metadati interattivi ─────────────────────────────────────
    print()

    if interactive:
        print("─── Metadati esperimento ─────────────────────────────────")

        exp_type = _ask(
            "Tipo esperimento (max 20 car.)",
            default=defaults.get('experiment_type', 'LoggerPro'),
        )[:20]

        sr = _ask(
            f"Sampling rate [Hz]",
            default=defaults.get('sampling_rate', round(sr_auto)),
            cast=float,
        )

        if amplified_forced is not None:
            # Range già gestito sopra — non richerere
            amplified = amplified_forced
            print(f"  Amplified: {'sì' if amplified else 'no'}  (impostato automaticamente)")
        else:
            amplified_input = _ask(
                "Dati amplificati? (range ±1.7 V)",
                default=defaults.get('amplified', 'n'),
                choices=['y', 'n'],
            )
            amplified = amplified_input.lower() == 'y'

        # Timestamp di inizio — opzionale
        start_ts_str = _ask(
            "Data/ora inizio (YYYY-MM-DD HH:MM:SS) oppure invio per ora corrente",
            default=defaults.get('start_time_str', ''),
        )
        if start_ts_str.strip():
            try:
                start_ts = datetime.strptime(start_ts_str, "%Y-%m-%d %H:%M:%S").timestamp()
            except ValueError:
                print("  ⚠️  Formato non riconosciuto — uso ora corrente.")
                start_ts = datetime.now().timestamp()
        else:
            start_ts = datetime.now().timestamp()

    else:
        # Modalità non interattiva: usa defaults
        exp_type  = defaults.get('experiment_type', 'LoggerPro')[:20]
        sr        = float(defaults.get('sampling_rate', sr_auto))
        amplified = amplified_forced if amplified_forced is not None else bool(defaults.get('amplified', False))
        start_ts  = float(defaults.get('start_time', datetime.now().timestamp()))

    end_ts = start_ts + duration

    # ── Costruisci header ────────────────────────────────────────
    header = _build_header(
        experiment_type  = exp_type,
        sampling_rate    = sr,
        duration         = duration,
        amplified        = amplified,
        start_time       = start_ts,
        end_time         = end_ts,
        data_points      = n_points,
        acquisition_count= 1,
    )

    # ── Percorso output ──────────────────────────────────────────
    base = os.path.splitext(csv_path)[0]
    out_path = base + '.pvolt'

    if os.path.exists(out_path):
        overwrite = _ask(
            f"  ⚠️  '{os.path.basename(out_path)}' esiste già. Sovrascrivere?",
            default='y',
            choices=['y', 'n'],
        )
        if overwrite.lower() != 'y':
            print("  Saltato.")
            return None

    # ── Scrivi ──────────────────────────────────────────────────
    _write_pvolt(out_path, header, t_list, v_list)

    if scale_factor != 1.0:
        print(f"  📐 Fattore di scala applicato: {scale_factor:.8f}")
        print(f"     Per ricostruire i valori originali: V_reale = V_display / {scale_factor:.8f}")
        print(f"     Oppure: V_reale = V_display × {1.0/scale_factor:.8f}")

    return out_path


# ═══════════════════════════════════════════════════════════════════
# MODALITÀ BATCH
# ═══════════════════════════════════════════════════════════════════

def batch_convert(csv_paths):
    """
    Converte più file chiedendo i metadati comuni una volta sola,
    poi li applica a tutti i file senza ulteriori prompt.
    """
    print(f"\n{'═'*60}")
    print(f"  BATCH MODE  —  {len(csv_paths)} file")
    print(f"{'═'*60}")
    print("Inserisci i metadati comuni (verranno applicati a tutti i file).\n")

    exp_type = _ask("Tipo esperimento (max 20 car.)", default='LoggerPro')[:20]
    amplified_input = _ask(
        "Dati amplificati? (range ±1.7 V)",
        default='n',
        choices=['y', 'n'],
    )
    amplified = amplified_input.lower() == 'y'

    start_ts_str = _ask(
        "Data/ora inizio primo file (YYYY-MM-DD HH:MM:SS) oppure invio per ora corrente",
        default='',
    )
    if start_ts_str.strip():
        try:
            start_ts = datetime.strptime(start_ts_str, "%Y-%m-%d %H:%M:%S").timestamp()
        except ValueError:
            print("  ⚠️  Formato non riconosciuto — uso ora corrente.")
            start_ts = datetime.now().timestamp()
    else:
        start_ts = datetime.now().timestamp()

    defaults = {
        'experiment_type': exp_type,
        'amplified':       amplified,
        'start_time':      start_ts,
        'start_time_str':  '',        # non richerere di nuovo
    }

    results = []
    for path in csv_paths:
        try:
            out = convert_file(path, interactive=False, defaults=defaults)
            results.append((path, out, None))
        except Exception as e:
            print(f"  ❌ Errore su {os.path.basename(path)}: {e}")
            results.append((path, None, str(e)))

    print(f"\n{'═'*60}")
    print("  RIEPILOGO BATCH")
    print(f"{'═'*60}")
    ok  = [r for r in results if r[2] is None]
    err = [r for r in results if r[2] is not None]
    print(f"  ✅ Convertiti con successo : {len(ok)}")
    if err:
        print(f"  ❌ Errori                 : {len(err)}")
        for _, _, msg in err:
            print(f"       {msg}")
    print()


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   csv_to_pvolt  —  Logger Pro CSV → PlantLeaf .pvolt    ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── Argomenti da riga di comando ─────────────────────────────
    args = sys.argv[1:]

    if args:
        csv_paths = []
        for a in args:
            if not os.path.isfile(a):
                print(f"⚠️  File non trovato: {a}")
                continue
            if not a.lower().endswith('.csv'):
                print(f"⚠️  Non è un .csv: {a}")
                continue
            csv_paths.append(a)

        if not csv_paths:
            print("❌ Nessun file CSV valido fornito.")
            sys.exit(1)

        if len(csv_paths) == 1:
            convert_file(csv_paths[0], interactive=True)
        else:
            batch_convert(csv_paths)

    else:
        # ── Modalità interattiva: chiedi il percorso ─────────────
        print("\nNessun file specificato — modalità interattiva.\n")
        raw = input("Inserisci il percorso del file CSV (o più file separati da spazio): ").strip()
        if not raw:
            print("❌ Nessun file specificato.")
            sys.exit(1)

        # Gestisci percorsi con spazi tra virgolette
        import shlex
        try:
            paths = shlex.split(raw)
        except ValueError:
            paths = raw.split()

        csv_paths = []
        for p in paths:
            p = p.strip('"\'')
            if not os.path.isfile(p):
                print(f"⚠️  File non trovato: {p}")
                continue
            csv_paths.append(p)

        if not csv_paths:
            print("❌ Nessun file valido.")
            sys.exit(1)

        if len(csv_paths) == 1:
            convert_file(csv_paths[0], interactive=True)
        else:
            batch_convert(csv_paths)

    print("🏁 Fatto.\n")


if __name__ == '__main__':
    main()
