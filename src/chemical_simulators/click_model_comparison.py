"""
click_model_comparison.py
=========================
Confronto di un click reale con ciò che un modello fisico può produrre,
misurando entrambi con LO STESSO metro.

Perché serve
------------
Il click reale non è la pressione emessa: è ciò che resta dopo microfono,
FFT del firmware (frame da 512 campioni, 154 bin 20–80 kHz, fase int8),
ricostruzione (correzione mic al 50 %, taper, Gibbs) e stimatore v6 di τ/f.
Confrontare la pressione simulata "grezza" con il click ricostruito, o
confrontare segnali campionati a passi diversi, dà numeri senza significato.

Qui il segnale simulato fa lo stesso percorso:

    p(t) modello, griglia fine (16 × 200 kHz)
      → decimazione a 200 kHz (FFT, banda limitata)
      → risposta del microfono (hybrid.channel_model.colorize)
      → frame del firmware (hybrid.frame_emulator.forward)
      → reconstruct_frame_v5 → build_click_context → resolve_click
      → compute_features_v5   (τ, R², f esattamente come il detector v6)

Modelli
-------
- 'vessel'  Dutta et al. 2022 (vessel_resonance.py). Sinusoide smorzata (f, τ):
            si CALIBRANO f e τ "veri" finché, misurati dalla catena, danno la
            f e il τ del click reale; poi R = √(4ητ/ρ) e L = v_eff/(2f).
- 'bubble'  Bolla libera di Minnaert con smorzamento radiativo, viscoso e
            termico (rayleigh_plesset.py). Si calibra solo f (→ R0); τ è
            PREDETTO dalla fisica, senza parametri liberi, e si confronta il
            τ simulato misurato con quello reale.

In entrambi i casi l'ampiezza è portata al picco del click reale (le soglie
del detector dipendono dal rapporto segnale/rumore) e l'innesco è allineato
al picco reale. La forma d'onda si confronta con R² nella sola regione che
v6 considera click (onset → decay_end), con ampiezza e fase libere; l'impulso
iniziale non spiegato dal ringing è misurato a parte (vedi _waveform_fit).

Nessuna dipendenza da Qt: usabile dalla UI e da script di popolazione.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from hybrid.pipeline_loader import load_pipeline          # noqa: E402
from hybrid import frame_emulator as fe                   # noqa: E402
from hybrid import channel_model as ch                    # noqa: E402

import rayleigh_plesset as rp                             # noqa: E402
import vessel_resonance as vr                             # noqa: E402

_cp = load_pipeline()
FS = _cp.FS
FFT_SIZE = _cp.FFT_SIZE
BIN_HZ = FS / FFT_SIZE

OVERSAMPLE = 16          # griglia fine per la sintesi (3.2 MHz)
PAD = 1024               # campioni di guardia ai lati (evitano l'effetto circolare delle FFT)
MAX_ITER = 10
TAU_TOL = 0.02           # tolleranza relativa su τ misurato
F_BOUNDS = (15_000.0, 95_000.0)
R0_BOUNDS = (3e-6, 3e-3)

REGION_NFFT = 4096       # zero-padding per la frequenza di picco della regione
REGION_MIN_LEN = 16      # campioni minimi della regione per stimarne lo spettro
MAX_PEAK_SHIFT = 64
# Sotto questo τ la catena + stimatore v6 non risolve più il decadimento
# (misurato in Step B: a 30 kHz τ_vero 0.04 ms → τ_misurato 0.057 ms; a 0.02 ms
# il fit fallisce). I risultati per click più corti sono segnalati, non scartati.
TAU_RESOLUTION_MS = 0.06      # spostamento massimo dell'innesco per iterazione [campioni]

MODEL_VESSEL = 'vessel'
MODEL_BUBBLE = 'bubble'


# =============================================================================
# MISURA (specchio di core.candidate_analysis.analyse_candidate, senza Qt)
# =============================================================================

def region_peak_frequency(signal, onset, decay_end, fs=None):
    """
    Frequenza di picco dello spettro della REGIONE del click [Hz]: segmento
    onset → decay_end, finestra di Hann, zero-padding a REGION_NFFT, argmax in
    20–80 kHz con interpolazione parabolica. Nessuna dipendenza da BLAS.

    Perché non FPE_hz del detector: FPE_hz è l'argmax dello spettro del solo
    frame corrente, e quando il click sta a cavallo di due frame quel frame ne
    contiene solo un pezzo — la stima salta tra bin lontani (19.9 → 25 → 30 kHz)
    e la calibrazione non converge. È anche il metodo di Dutta et al. 2022
    (DFT sulla finestra temporale del click).
    """
    fs = FS if fs is None else fs
    seg = np.asarray(signal[max(0, onset):decay_end + 1], dtype=np.float64)
    if len(seg) < REGION_MIN_LEN:
        return float('nan')
    seg = (seg - np.mean(seg)) * np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(seg, n=REGION_NFFT))
    freqs = np.fft.rfftfreq(REGION_NFFT, 1.0 / fs)
    band = np.where((freqs >= _cp.BIN_START_HZ) & (freqs <= _cp.BIN_END_HZ))[0]
    k = band[np.argmax(spec[band])]
    if 0 < k < len(spec) - 1:
        a, b, c = spec[k - 1], spec[k], spec[k + 1]
        den = a - 2 * b + c
        shift = 0.5 * (a - c) / den if den != 0 else 0.0
        return float((k + shift) * fs / REGION_NFFT)
    return float(freqs[k])


def measure_frames(frames, frame_idx, noise_floor, std_noise):
    """
    Misura un click da tre frame (mags, phases) — prev/curr/next, None se
    assenti — con le stesse chiamate di core.candidate_analysis.analyse_candidate
    (senza Buffer 3: τ, R², FPE_hz e peak_amp non ne dipendono).

    Returns:
        dict o None se il frame corrente non è ricostruibile.
    """
    def rec(fr):
        if fr is None:
            return None
        return _cp.reconstruct_frame_v5(np.asarray(fr[0]), np.asarray(fr[1]),
                                        FS, FFT_SIZE, normalize=True)

    prev, curr, nxt = (rec(f) for f in frames)
    if curr is None:
        return None
    ctx = _cp.build_click_context(prev['signal'] if prev else None, curr['signal'],
                                  nxt['signal'] if nxt else None)
    resolved = _cp.resolve_click(ctx, noise_floor, std_noise)
    feats = _cp.compute_features_v5(ctx, resolved, curr['fft_norm'], curr['freq_axis'],
                                    noise_floor, std_noise, FS, p_noise_psd=None)
    signal = np.asarray(ctx['signal'], dtype=np.float64)
    return {
        'tau_ms': float(feats['tau_ms']),
        'R2': float(feats['R2']),
        'fit_valid': int(feats.get('fit_valid', 0)),
        'f_hz': float(feats['FPE_hz']),
        'f_region_hz': region_peak_frequency(signal, int(resolved['onset']), int(resolved['decay_end'])),
        'peak_amp': float(resolved['peak_amp']),
        'onset': int(resolved['onset']),
        'peak': int(resolved['peak']),
        'decay_end': int(resolved['decay_end']),
        'signal': signal,
        'fft_norm': np.asarray(curr['fft_norm']),
        'freq_axis': np.asarray(curr['freq_axis']),
        # Indice nel layout a 3 frame del campione 0 del contesto: 512 se manca prev.
        'layout_offset': 0 if prev is not None else FFT_SIZE,
    }


# =============================================================================
# SINTESI → CATENA DI MISURA
# =============================================================================

def _fine_time(t0_layout_s):
    """Griglia fine [s] rispetto all'innesco, sul layout a 3 frame + guardie."""
    n = (3 * FFT_SIZE + 2 * PAD) * OVERSAMPLE
    t_layout = (np.arange(n) / (FS * OVERSAMPLE)) - PAD / FS
    return t_layout - t0_layout_s


def _hilbert_quadrature(x):
    """Parte immaginaria del segnale analitico (x sfasato di 90°), numpy puro."""
    X = np.fft.fft(x)
    n = len(x)
    h = np.zeros(n)
    h[0] = 1.0
    if n % 2 == 0:
        h[1:n // 2] = 2.0
        h[n // 2] = 1.0
    else:
        h[1:(n + 1) // 2] = 2.0
    return np.fft.ifft(X * h).imag


def render_pressure(model, params, t0_layout_s, quadrature=False):
    """
    Pressione del modello sulla griglia fine.

    params: vessel → {'f': Hz, 'tau': s};  bubble → {'R0': m}
    """
    t = _fine_time(t0_layout_s)
    if model == MODEL_VESSEL:
        p = vr.synthesize_vessel_pressure(t, params['f'], params['tau'],
                                          phase=np.pi / 2 if quadrature else 0.0)
    elif model == MODEL_BUBBLE:
        R0 = params['R0']
        osc = rp.synthesize_bubble_oscillation(R0, t=t)
        p = rp.compute_radiated_pressure(osc['R'], osc['V'], t, R0)
        p = p - p[0]
        if quadrature:
            p = _hilbert_quadrature(p)
    else:
        raise ValueError(f"unknown model {model!r}")
    return p


def pressure_to_frames(p_fine, present):
    """
    Griglia fine → 200 kHz → microfono → frame del firmware.

    Args:
        p_fine  : pressione sulla griglia fine (da render_pressure)
        present : (prev, curr, next) booleani — stessi frame disponibili del click reale

    Returns:
        (frames, x200) — frames come (mags, phases) o None; x200 il segnale a 200 kHz
    """
    X = np.fft.rfft(p_fine)
    n_c = len(p_fine) // OVERSAMPLE
    x = np.fft.irfft(X[:n_c // 2 + 1], n=n_c) * (n_c / len(p_fine))
    x = ch.colorize(x, FS)
    x = x[PAD:PAD + 3 * FFT_SIZE]
    frames = []
    for k, ok in enumerate(present):
        frames.append(fe.forward(x[k * FFT_SIZE:(k + 1) * FFT_SIZE]) if ok else None)
    return frames, x


def _scale_frames(frames, gain):
    """Moltiplica le ampiezze (la catena è lineare nelle magnitudini)."""
    return [None if f is None else (f[0] * gain, f[1]) for f in frames]


# =============================================================================
# CONFRONTO FORMA D'ONDA
# =============================================================================

def _pearson(x, y):
    """Correlazione di Pearson con sole somme elementari (niente BLAS, vedi sotto)."""
    x = x - np.mean(x)
    y = y - np.mean(y)
    den = np.sqrt(np.sum(x * x) * np.sum(y * y))
    return float(np.sum(x * y) / den) if den > 0 else float('nan')


def _waveform_fit(real, sim_s, sim_c):
    """
    Confronto della forma d'onda nella regione che v6 considera click:
    [onset, decay_end] di resolve_click (= ctx_region del detector), niente margini
    e niente ritardi (la calibrazione allinea già il picco).

    Ampiezza e fase del simulato sono LIBERE, stimate ai minimi quadrati:

        y ≈ a·S + b·C      (S, C: simulato e sua versione a 90°, dalla catena)
        R² = 1 − Σ (y − a·S − b·C)² / Σ (y − ȳ)²

    Perché l'ampiezza è libera: non è una predizione di nessuno dei due modelli
    (energia della sorgente e accoppiamento col microfono sono ignoti). La versione
    "stesso picco" (Step B.1) penalizzava il modello per l'impulso iniziale dei click
    reali — 1–2 cicli fino a ~3–5× più alti del ringing che segue, che un'unica
    sinusoide smorzata non descrive — dando R² fino a −3 su click la cui oscillazione
    coincide con il modello (vedi CHANGELOG Step B.2).

    Quell'impulso è misurato a parte:

        impulse_factor = picco d'inviluppo reale / picco d'inviluppo del modello adattato

    ≈ 1: il click è tutto ringing; ≫ 1: un impulso iniziale domina il picco.
    """
    y_full = real['signal']
    n = min(len(y_full), len(sim_s), len(sim_c))
    lo = max(0, real['onset'])
    hi = min(n, real['decay_end'] + 1)
    empty = {'r2_wave': float('nan'), 'r_wave': float('nan'), 'env_corr': float('nan'),
             'impulse_factor': float('nan'), 'phase': float('nan'),
             'sim_fit': np.zeros_like(y_full), 'window': (lo, hi)}
    if hi - lo < 4:
        return empty
    y = y_full[lo:hi]
    # Scala comune: segnali in V (~1e-4), prodotti ~1e-8 — lontani dai limiti del
    # float64 ma normalizzati per sicurezza. Solo somme elementari: niente BLAS
    # (np.linalg / @ vanno in segfault dentro un QThread su macOS).
    scale = max(np.max(np.abs(sim_s[lo:hi])), np.max(np.abs(sim_c[lo:hi])), 1e-300)
    S, C = sim_s[lo:hi] / scale, sim_c[lo:hi] / scale
    sss, scc, ssc = np.sum(S * S), np.sum(C * C), np.sum(S * C)
    sys_, syc = np.sum(y * S), np.sum(y * C)
    det = sss * scc - ssc * ssc
    if not np.isfinite(det) or det <= 1e-12 * sss * scc:
        return empty
    a = (sys_ * scc - syc * ssc) / det
    b = (syc * sss - sys_ * ssc) / det

    fit = np.zeros_like(y_full)
    fit[:n] = (a * sim_s[:n] + b * sim_c[:n]) / scale

    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - np.sum((y - fit[lo:hi]) ** 2) / ss_tot if ss_tot > 0 else float('nan')
    env_r = _cp.compute_hilbert_envelope(y_full)[lo:hi]
    env_s = _cp.compute_hilbert_envelope(fit)[lo:hi]
    peak_s = np.max(env_s)
    return {'r2_wave': float(r2),
            # Correlazione di forma, indipendente dalla scala.
            'r_wave': _pearson(y, fit[lo:hi]),
            'env_corr': _pearson(env_r, env_s),
            'impulse_factor': float(np.max(env_r) / peak_s) if peak_s > 0 else float('nan'),
            'phase': float(np.arctan2(b, a)),
            'sim_fit': fit, 'window': (lo, hi)}


# =============================================================================
# CALIBRAZIONE SUL CLICK REALE
# =============================================================================

def _bubble_R0_for_frequency(f):
    """R0 della bolla libera che risuona a f [Hz] (Minnaert con κ effettivo)."""
    g = lambda R0: rp.minnaert_frequency(R0)[0] - f
    return brentq(g, *R0_BOUNDS, xtol=1e-9)


def fit_model(model, real, frames_present, frame_idx, noise_floor, std_noise):
    """
    Calibra il modello sul click reale già misurato (`real` da measure_frames).

    Returns:
        dict con i parametri calibrati, la misura del simulato e il confronto,
        oppure {'ok': False, 'reason': ...}.
    """
    f_target = real['f_region_hz']
    if not np.isfinite(f_target):
        return {'ok': False, 'reason': 'regione del click troppo corta per stimarne la frequenza'}
    tau_target = real['tau_ms'] / 1e3 if (real['fit_valid'] and real['tau_ms'] > 0) else None
    if model == MODEL_VESSEL and tau_target is None:
        return {'ok': False, 'reason': 'τ del click reale non valido (fit_valid = 0)'}

    offset = real['layout_offset']
    t0 = (real['onset'] + offset) / FS
    f_true = f_target
    tau_true = tau_target
    gain = 1.0
    sim = None
    history = []
    converged = False

    def params_for(f, tau):
        if model == MODEL_VESSEL:
            return {'f': f, 'tau': tau}
        return {'R0': _bubble_R0_for_frequency(f)}

    best = None   # (errore, f_true, tau_true, t0) — il miglior punto visto, se non converge

    for it in range(MAX_ITER):
        params = params_for(f_true, tau_true)
        frames, _ = pressure_to_frames(render_pressure(model, params, t0), frames_present)
        sim = measure_frames(_scale_frames(frames, gain), frame_idx, noise_floor, std_noise)
        if sim is None or sim['peak_amp'] <= 0:
            return {'ok': False, 'reason': 'il click simulato non è ricostruibile'}
        # 1) ampiezza: stesso picco del reale (lineare → esatta in un passo)
        gain *= real['peak_amp'] / sim['peak_amp']
        sim = measure_frames(_scale_frames(frames, gain), frame_idx, noise_floor, std_noise)
        history.append((f_true, tau_true, sim['f_hz'], sim['tau_ms']))

        f_err = (abs(sim['f_region_hz'] - f_target) / (0.01 * f_target)
                 if np.isfinite(sim['f_region_hz']) else np.inf)
        tau_valid = bool(sim['fit_valid']) and sim['tau_ms'] > 0
        tau_err = (0.0 if model == MODEL_BUBBLE else
                   abs(sim['tau_ms'] / 1e3 / tau_target - 1.0) if tau_valid else np.inf)
        peak_err = abs(sim['peak'] - real['peak'])
        err = f_err + tau_err / TAU_TOL + peak_err
        if best is None or err < best[0]:
            best = (err, f_true, tau_true, t0)
        if f_err < 1.0 and tau_err < TAU_TOL and peak_err <= 1:
            converged = True
            break

        # 2) aggiornamenti di punto fisso, con limiti: sotto ~0.05 ms la catena
        #    non risolve più τ e la relazione τ_vero → τ_misurato smette di essere
        #    monotona, quindi un passo non protetto può divergere.
        if np.isfinite(sim['f_region_hz']):
            f_true = float(np.clip(f_true + (f_target - sim['f_region_hz']), *F_BOUNDS))
        if model == MODEL_VESSEL and tau_valid:
            tau_true = float(np.clip(tau_true * tau_target / (sim['tau_ms'] / 1e3),
                                     tau_target / 3.0, tau_target * 3.0))
        if 0 < peak_err <= MAX_PEAK_SHIFT:
            t0 += (real['peak'] - sim['peak']) / FS

    if not converged and best is not None:
        _, f_true, tau_true, t0 = best

    params = params_for(f_true, tau_true)
    p_s = render_pressure(model, params, t0)
    p_c = render_pressure(model, params, t0, quadrature=True)
    fr_s, _ = pressure_to_frames(p_s, frames_present)
    fr_c, _ = pressure_to_frames(p_c, frames_present)
    sim_s = measure_frames(_scale_frames(fr_s, gain), frame_idx, noise_floor, std_noise)
    sim_c = measure_frames(_scale_frames(fr_c, gain), frame_idx, noise_floor, std_noise)
    wave = _waveform_fit(real, sim_s['signal'], sim_c['signal'])

    out = {
        'ok': True, 'model': model, 'converged': converged, 'iterations': len(history),
        'below_resolution': bool(0 < real['tau_ms'] < TAU_RESOLUTION_MS),
        'f_true_hz': f_true,
        'sim_tau_ms': sim_s['tau_ms'], 'sim_R2': sim_s['R2'], 'sim_fit_valid': sim_s['fit_valid'],
        'sim_f_hz': sim_s['f_region_hz'],
        'sim_fpe_hz': sim_s['f_hz'],
        'r2_wave': wave['r2_wave'], 'r_wave': wave['r_wave'], 'env_corr': wave['env_corr'],
        'impulse_factor': wave['impulse_factor'],
        'phase': wave['phase'],
        'sim_signal': wave['sim_fit'], 'window': wave['window'],
        'sim_fft_norm': sim_s['fft_norm'],
    }
    if model == MODEL_VESSEL:
        geo = vr.vessel_from_click(f_true, tau_true)
        out.update({
            'tau_true_ms': tau_true * 1e3,
            'Q': geo['Q'],
            'R_um': geo['R'] * 1e6,
            'L_mm': geo['L'] * 1e3,
            'v_eff': geo['v_eff'],
        })
    else:
        R0 = params['R0']
        props = rp.bubble_linear_properties(R0)
        osc = rp.synthesize_bubble_oscillation(R0, n_points=2000,
                                               t_max=min(6 * props['tau'], 2e-3))
        out.update({
            'R0_um': R0 * 1e6,
            'kappa': props['kappa'],
            'tau_pred_ms': props['tau'] * 1e3,
            'Q_pred': props['Q'],
            'delta_rad': props['delta_rad'], 'delta_vis': props['delta_vis'],
            'delta_th': props['delta_th'],
            # τ misurati con lo stesso stimatore: >1 = il click reale dura più della bolla
            'tau_ratio_real_over_sim': (real['tau_ms'] / sim_s['tau_ms']
                                        if real['tau_ms'] > 0 and sim_s['tau_ms'] > 0
                                        else float('nan')),
            'bubble_t': osc['t'], 'bubble_R': osc['R'],
        })
    return out


def analyse_click(frames, frame_idx, noise_floor, std_noise, models=(MODEL_VESSEL, MODEL_BUBBLE)):
    """
    Misura il click reale e lo confronta con ciascun modello.

    Args:
        frames      : (prev, curr, next) come (mags, phases) o None
        frame_idx   : indice del frame nella registrazione
        noise_floor, std_noise : stato del rumore al click [V] (dal detector)

    Returns:
        dict {'real': misura, 'vessel': risultato, 'bubble': risultato}
        con 'real' None se il click non è ricostruibile.
    """
    real = measure_frames(frames, frame_idx, noise_floor, std_noise)
    out = {'real': real}
    if real is None:
        return out
    present = tuple(f is not None for f in frames)
    for m in models:
        try:
            out[m] = fit_model(m, real, present, frame_idx, noise_floor, std_noise)
        except Exception as e:  # noqa: BLE001 — un click difettoso non deve fermare un batch
            out[m] = {'ok': False, 'reason': f'{type(e).__name__}: {e}'}
    if MODEL_VESSEL in out and MODEL_BUBBLE in out and out[MODEL_VESSEL].get('ok') \
            and out[MODEL_BUBBLE].get('ok'):
        # τ "vero" del click reale (calibrato dalla catena) contro τ previsto dalla bolla
        out[MODEL_BUBBLE]['tau_true_real_ms'] = out[MODEL_VESSEL]['tau_true_ms']
        out[MODEL_BUBBLE]['Q_real'] = out[MODEL_VESSEL]['Q']
        # Sempre definito, anche quando il τ previsto dalla bolla è troppo corto
        # perché lo stimatore v6 lo misuri sul simulato (fit_valid = 0).
        out[MODEL_BUBBLE]['tau_ratio_true_over_pred'] = (
            out[MODEL_VESSEL]['tau_true_ms'] / out[MODEL_BUBBLE]['tau_pred_ms'])
    return out


# Colonne per l'export CSV (una riga per click)
CSV_COLUMNS = [
    'file', 'frame_idx', 'timestamp_s',
    'real_tau_ms', 'real_R2', 'real_fit_valid', 'real_f_hz', 'real_FPE_hz', 'real_peak_amp_V',
    'real_below_tau_resolution',
    'vessel_ok', 'vessel_converged', 'vessel_f_true_hz', 'vessel_tau_true_ms', 'vessel_Q',
    'vessel_R_um', 'vessel_L_mm', 'vessel_sim_tau_ms', 'vessel_r2_wave', 'vessel_r_wave', 'vessel_env_corr', 'vessel_impulse_factor',
    'bubble_ok', 'bubble_converged', 'bubble_R0_um', 'bubble_tau_pred_ms', 'bubble_Q_pred',
    'bubble_sim_tau_ms', 'bubble_tau_ratio_real_over_sim', 'bubble_tau_ratio_true_over_pred',
    'bubble_r2_wave', 'bubble_r_wave', 'bubble_env_corr', 'bubble_impulse_factor',
    'notes',
]


def result_to_row(res, file='', frame_idx=None, timestamp_s=None):
    """Appiattisce il risultato di analyse_click in una riga CSV."""
    nan = float('nan')
    real = res.get('real') or {}
    v = res.get(MODEL_VESSEL) or {}
    b = res.get(MODEL_BUBBLE) or {}
    notes = '; '.join(f"{k}: {d['reason']}" for k, d in ((MODEL_VESSEL, v), (MODEL_BUBBLE, b))
                      if d and not d.get('ok') and d.get('reason'))
    return {
        'file': file, 'frame_idx': frame_idx, 'timestamp_s': timestamp_s,
        'real_tau_ms': real.get('tau_ms', nan), 'real_R2': real.get('R2', nan),
        'real_fit_valid': real.get('fit_valid', ''), 'real_f_hz': real.get('f_region_hz', nan),
        'real_FPE_hz': real.get('f_hz', nan),
        'real_peak_amp_V': real.get('peak_amp', nan),
        'real_below_tau_resolution': int(0 < real.get('tau_ms', 0) < TAU_RESOLUTION_MS),
        'vessel_ok': int(bool(v.get('ok'))), 'vessel_converged': int(bool(v.get('converged'))),
        'vessel_f_true_hz': v.get('f_true_hz', nan), 'vessel_tau_true_ms': v.get('tau_true_ms', nan),
        'vessel_Q': v.get('Q', nan), 'vessel_R_um': v.get('R_um', nan), 'vessel_L_mm': v.get('L_mm', nan),
        'vessel_sim_tau_ms': v.get('sim_tau_ms', nan), 'vessel_r2_wave': v.get('r2_wave', nan),
        'vessel_r_wave': v.get('r_wave', nan),
        'vessel_env_corr': v.get('env_corr', nan), 'vessel_impulse_factor': v.get('impulse_factor', nan),
        'bubble_ok': int(bool(b.get('ok'))), 'bubble_converged': int(bool(b.get('converged'))),
        'bubble_R0_um': b.get('R0_um', nan), 'bubble_tau_pred_ms': b.get('tau_pred_ms', nan),
        'bubble_Q_pred': b.get('Q_pred', nan), 'bubble_sim_tau_ms': b.get('sim_tau_ms', nan),
        'bubble_tau_ratio_real_over_sim': b.get('tau_ratio_real_over_sim', nan),
        'bubble_tau_ratio_true_over_pred': b.get('tau_ratio_true_over_pred', nan),
        'bubble_r2_wave': b.get('r2_wave', nan), 'bubble_r_wave': b.get('r_wave', nan),
        'bubble_env_corr': b.get('env_corr', nan), 'bubble_impulse_factor': b.get('impulse_factor', nan),
        'notes': notes,
    }
