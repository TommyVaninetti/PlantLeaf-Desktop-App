"""
run_acoustic_simulation.py
==========================
Pipeline completa del simulatore acustico PlantLeaf — VERSIONE FISICA
(modello a risonanza di bolla smorzata, vedi rayleigh_plesset.py).

Collega la risonanza di bolla (Minnaert + smorzamento per irraggiamento
acustico) → propagazione acustica nel tessuto → risposta del microfono →
estrazione diagnostica, e produce il click simulato finale.

R0 è l'unico parametro fisico libero: frequenza e tau del click sono
determinati da R0 tramite formule chiuse (nessun parametro arbitrario
da tarare, a differenza di un precedente tentativo con "rigidità
elastica" del vaso). La calibrazione consiste quindi nel trovare l'R0
che riproduce il tau e/o la frequenza osservati nel click reale, tramite
una semplice e stabile ricerca 1D (bisezione), non più un'ottimizzazione
a due parametri.

Se sono disponibili sia tau_target_ms sia freq_target_hz, si dà priorità
alla frequenza (più direttamente legata alla fisica di risonanza) e si
riporta il tau ottenuto come diagnostica, dato che un solo parametro
libero (R0) non può in generale soddisfare esattamente entrambi i
target contemporaneamente.

Utilizzo:
    from run_acoustic_simulation import run_simulation
    result = run_simulation(R0=50e-6)
    result = run_simulation(tau_target_ms=0.25)
    result = run_simulation(tau_target_ms=0.25, freq_target_hz=45000)
"""

import numpy as np
from scipy.signal import hilbert
from scipy.stats import linregress

from acoustic_parameters import (
    BubbleParameters,
    XylemPressure,
    PlantLeafConfig
)
from rayleigh_plesset import (
    simulate_bubble_collapse,
    solve_R0_for_freq,
    solve_R0_for_tau,
    minnaert_frequency,
    damping_time_constant,
    damping_components
)
from acoustic_propagation import apply_propagation


# =============================================================================
# FUNZIONE PRINCIPALE
# =============================================================================

def run_simulation(R0=None, P_inf=None, distance_m=None, tau_target_ms=None, freq_target_hz=None, real_signal_for_fit=None):
    """
    Esegue la simulazione fisica completa del click ultrasonico.

    Args:
        R0             : raggio di equilibrio della bolla [m]. Default: 50 µm.
                         Ignorato se tau_target_ms o freq_target_hz sono
                         specificati (viene calibrato numericamente).
        P_inf          : tensione idrica dello xilema [Pa]. Solo per
                         display (non entra nella fisica del modello di
                         risonanza). Default: -0.3 MPa.
        distance_m     : distanza bolla-microfono [m]. Default: 1 cm.
        tau_target_ms  : se specificato, calibra R0 per riprodurre
                         questo tau di decadimento.
        freq_target_hz : se specificato, calibra R0 per riprodurre
                         questa frequenza dominante (ha priorità su
                         tau_target_ms se entrambi sono presenti).

    Returns:
        dict con le chiavi:
            'bubble'      : risultati dinamica bolla (da rayleigh_plesset)
            'propagation' : risultati propagazione (da acoustic_propagation)
            'diagnostics' : parametri diagnostici estratti dal segnale
            'plantleaf'   : segnale ricampionato sull'asse PlantLeaf
            'calibration' : info sulla calibrazione (None se non richiesta)
    """
    if P_inf is None:
        P_inf = XylemPressure.P_INF_DEFAULT
    if distance_m is None:
        distance_m = 0.01

    calibration_info = None

    extra_damping_rate = 0.0

    if real_signal_for_fit is not None:
        fit_result = fit_R0_and_damping_for_correlation(real_signal_for_fit, distance_m)
        R0 = fit_result['R0']
        extra_damping_rate = fit_result['extra_damping_rate']
        f0_achieved, omega0_achieved = minnaert_frequency(R0)
        tau_achieved = damping_time_constant(R0, omega0_achieved, extra_damping_rate)
        correlation_achieved = fit_result['correlation']

        calibration_info = {
            'converged': correlation_achieved > 0.5,
            'freq_target_hz': freq_target_hz,
            'freq_achieved_hz': f0_achieved,
            'tau_target_ms': tau_target_ms,
            'tau_achieved_ms': tau_achieved * 1000.0,
            'extra_damping_rate': extra_damping_rate,
            'correlation_achieved': correlation_achieved,
            'note': (
                f"R0 e smorzamento calibrati per massimizzare direttamente la "
                f"correlazione con il click reale (fitting sulla forma "
                f"dell'inviluppo, non solo su frequenza/tau separati).\n"
                f"R0={R0*1e6:.1f}µm, smorzamento extra={extra_damping_rate:.0f} 1/s, "
                f"frequenza risultante={f0_achieved/1000:.1f}kHz, "
                f"tau risultante={tau_achieved*1000:.3f}ms.\n"
                f"Correlazione ottenuta: {correlation_achieved:.3f}"
            )
        }
    elif freq_target_hz is not None and freq_target_hz > 0:
        R0, freq_achieved = calibrate_R0_for_freq_e2e(P_inf, freq_target_hz, distance_m)
        f0_theory, omega0_theory = minnaert_frequency(R0)
        freq_converged = freq_achieved is not None and abs(freq_achieved - freq_target_hz) / freq_target_hz < 0.15

        tau_achieved = damping_time_constant(R0, omega0_theory)
        tau_note = ""
        tau_converged = True

        if tau_target_ms is not None and tau_target_ms > 0:
            extra_damping_rate, tau_achieved_e2e = fit_extra_damping_for_tau(
                R0, tau_target_ms / 1000.0, distance_m
            )
            tau_converged = (
                tau_achieved_e2e is not None and
                abs(tau_achieved_e2e - tau_target_ms) / tau_target_ms < 0.10
            )
            tau_achieved = tau_achieved_e2e / 1000.0 if tau_achieved_e2e else tau_achieved
            tau_note = (
                f" Smorzamento aggiuntivo del vaso stimato (fitting): "
                f"{extra_damping_rate:.0f} 1/s. "
                f"Tau target={tau_target_ms:.3f}ms → ottenuto="
                f"{(tau_achieved_e2e if tau_achieved_e2e else tau_achieved*1000):.3f}ms."
            )

        converged = freq_converged and tau_converged
        calibration_info = {
            'converged': converged,
            'freq_target_hz': freq_target_hz,
            'freq_achieved_hz': freq_achieved if freq_achieved else f0_theory,
            'tau_target_ms': tau_target_ms,
            'tau_achieved_ms': tau_achieved * 1000.0,
            'extra_damping_rate': extra_damping_rate,
            'note': (
                f"R0 calibrato a {R0*1e6:.1f} µm per riprodurre la frequenza "
                f"{freq_target_hz/1000:.1f} kHz misurata sul segnale finale, "
                f"ottenuta: {(freq_achieved/1000 if freq_achieved else f0_theory/1000):.1f} kHz."
                + tau_note
            )
        }
    elif tau_target_ms is not None and tau_target_ms > 0:
        R0 = solve_R0_for_tau(tau_target_ms / 1000.0)
        f0_achieved, omega0_achieved = minnaert_frequency(R0)
        tau_achieved = damping_time_constant(R0, omega0_achieved)
        calibration_info = {
            'converged': True,
            'freq_target_hz': None,
            'freq_achieved_hz': f0_achieved,
            'tau_target_ms': tau_target_ms,
            'tau_achieved_ms': tau_achieved * 1000.0,
            'note': (
                f"R0 calibrato a {R0*1e6:.1f} µm per riprodurre tau = "
                f"{tau_target_ms:.3f} ms (ottenuto: {tau_achieved*1000:.3f} ms). "
                f"Frequenza risultante: {f0_achieved/1000:.1f} kHz."
            )
        }
    elif R0 is None:
        R0 = BubbleParameters.R0_DEFAULT

    # --- Dinamica della bolla (risonanza smorzata) ---
    bubble = simulate_bubble_collapse(R0=R0, P_inf=P_inf, extra_damping_rate=extra_damping_rate)

    # --- Propagazione acustica (tessuto + geometria + microfono) ---
    propagation = apply_propagation(
        p_source=bubble['p_source'],
        t=bubble['t'],
        distance_m=distance_m
    )

    # --- Estrazione parametri diagnostici (identica alla pipeline reale) ---
    diagnostics = extract_diagnostics(
        signal=propagation['signal'],
        t=bubble['t']
    )

    # --- Ricampionamento sull'asse frequenze PlantLeaf ---
    plantleaf = resample_to_plantleaf(
        freq=propagation['freq'],
        spectrum=propagation['spectrum']
    )

    return {
        'bubble': bubble,
        'propagation': propagation,
        'diagnostics': diagnostics,
        'plantleaf': plantleaf,
        'calibration': calibration_info
    }

# =============================================================================
# CALIBRAZIONE END-TO-END SULLA FREQUENZA (dopo propagazione nel tessuto)
# =============================================================================

def _measure_freq_zerocrossing(signal, t):
    """
    Misura la frequenza dominante di un segnale tramite conteggio degli
    attraversamenti dello zero sulla porzione isolata attorno al picco
    (stesso metodo usato per estrarre la frequenza dal click reale).
    """
    try:
        envelope = np.abs(hilbert(signal))
        peak_idx = int(np.argmax(envelope))
        peak_amp = envelope[peak_idx]
        if peak_amp <= 0:
            return None
        level = peak_amp * 0.1

        start_idx = 0
        for i in range(peak_idx - 1, -1, -1):
            if envelope[i] < level:
                start_idx = i + 1
                break

        end_idx = len(envelope) - 1
        for i in range(peak_idx + 1, len(envelope)):
            if envelope[i] < level:
                end_idx = i
                break

        segment = signal[start_idx:end_idx + 1]
        if len(segment) < 4:
            return None

        signs = np.sign(segment)
        signs[signs == 0] = 1
        crossings = np.where(np.diff(signs) != 0)[0]
        if len(crossings) < 2:
            return None

        n_cycles = len(crossings) / 2.0
        dt = t[1] - t[0]
        duration_s = len(segment) * dt
        return n_cycles / duration_s if duration_s > 0 else None
    except Exception:
        return None


def _freq_e2e_for_R0(R0, P_inf, distance_m):
    """
    Esegue l'intera pipeline (bolla + propagazione) per un dato R0 e
    misura la frequenza dominante del segnale finale, con lo stesso
    metodo usato per il click reale — così la calibrazione centra
    esattamente ciò che viene confrontato in UI.
    """
    try:
        bubble = simulate_bubble_collapse(R0=R0, P_inf=P_inf)
        propagation = apply_propagation(
            p_source=bubble['p_source'],
            t=bubble['t'],
            distance_m=distance_m
        )
        return _measure_freq_zerocrossing(propagation['signal'], bubble['t'])
    except Exception:
        return None


def calibrate_R0_for_freq_e2e(P_inf, freq_target_hz, distance_m, n_scan=60):
    """
    Trova l'R0 che, attraverso l'intera pipeline fisica (bolla +
    propagazione nel tessuto), produce una frequenza dominante il più
    vicino possibile a freq_target_hz. Usa una scansione fitta invece
    di una bisezione, per essere robusta anche quando alcune zone del
    range biologico non producono una misura valida (es. R0 molto
    piccoli, frequenze troppo alte per un conteggio affidabile).

    Returns:
        (R0, freq_achieved_hz) — freq_achieved_hz è None se nessun
        punto della scansione ha prodotto una misura valida.
    """
    R0_min = BubbleParameters.R0_MIN
    R0_max = BubbleParameters.R0_MAX

    R0_grid = np.linspace(R0_min, R0_max, n_scan)
    best_R0, best_err, best_freq = BubbleParameters.R0_DEFAULT, 1e18, None

    for R0_candidate in R0_grid:
        f = _freq_e2e_for_R0(R0_candidate, P_inf, distance_m)
        if f is None:
            continue
        err = abs(f - freq_target_hz)
        if err < best_err:
            best_err, best_R0, best_freq = err, R0_candidate, f

    return best_R0, best_freq

# =============================================================================
# FITTING DELLO SMORZAMENTO AGGIUNTIVO DEL VASO (per riprodurre il τ reale)
# =============================================================================

def _tau_e2e_for_extra_damping(R0, extra_damping_rate, P_inf, distance_m):
    """
    Esegue l'intera pipeline (bolla + propagazione) per un dato
    smorzamento aggiuntivo e misura il tau del segnale finale, con lo
    stesso metodo diagnostico usato per il confronto in UI.
    """
    try:
        bubble = simulate_bubble_collapse(R0=R0, P_inf=P_inf, extra_damping_rate=extra_damping_rate)
        propagation = apply_propagation(
            p_source=bubble['p_source'],
            t=bubble['t'],
            distance_m=distance_m
        )
        diag = extract_diagnostics(propagation['signal'], bubble['t'])
        tau = diag['tau']
        return tau * 1000.0 if tau else None
    except Exception:
        return None


def fit_extra_damping_for_tau(R0, tau_target_s, distance_m, max_iter=5):
    """
    Calcola lo smorzamento aggiuntivo (rappresentante l'attrito con la
    parete del vaso xilematico, non incluso nel modello fisico di base)
    necessario per riprodurre il tau osservato nel click reale, a un
    dato R0 (già calibrato sulla frequenza).

    Questo è un FITTING esplicito, non una previsione fisica: il valore
    trovato è specifico del singolo click analizzato — la variabilità
    osservata tra click diversi (fino a 3-4x) suggerisce che riflette
    differenze reali tra vasi xilematici, non un singolo meccanismo
    fisico universale.

    Metodo: parte da una stima algebrica diretta (formula chiusa), poi
    raffina con poche correzioni dirette (non una ricerca/ottimizzazione
    instabile) per compensare il piccolo bias sistematico introdotto
    dal metodo di misura del tau (identico a quello usato sui dati
    reali, per coerenza nel confronto).

    Returns:
        (extra_damping_rate, tau_achieved_ms)
    """
    f0, omega0 = minnaert_frequency(R0)
    b_rad, b_vis = damping_components(R0, omega0)
    b_target = 1.0 / tau_target_s
    extra_damping_rate = max(0.0, b_target - (b_rad + b_vis))

    tau_target_ms = tau_target_s * 1000.0
    tau_achieved_ms = None

    for _ in range(max_iter):
        tau_achieved_ms = _tau_e2e_for_extra_damping(R0, extra_damping_rate, XylemPressure.P_INF_DEFAULT, distance_m)
        if tau_achieved_ms is None:
            break
        error_ratio = tau_achieved_ms / tau_target_ms
        if abs(error_ratio - 1.0) < 0.02:
            break
        b_total_current = b_rad + b_vis + extra_damping_rate
        b_total_new = b_total_current * error_ratio
        extra_damping_rate = max(0.0, b_total_new - (b_rad + b_vis))

    return extra_damping_rate, tau_achieved_ms

# =============================================================================
# SMOOTHING (identico a PlantLeaf: media mobile 4 campioni, elimina la
# portante di decadimento senza alterare la vera forma del segnale)
# =============================================================================

def apply_smoothing_4samples(signal):
    """
    Media mobile a 4 campioni, identica a quella usata dall'algoritmo di
    PlantLeaf (compute_decay_r2, DECAY_FIT_SMOOTH_WIN=4) per calcolare
    tau e R² sul click reale. Applicarla anche qui, prima del confronto,
    garantisce che stiamo confrontando forme d'onda trattate allo stesso
    modo — non è un filtro nuovo per "ripulire" il dato, è coerenza con
    il metro di misura già usato per il riferimento reale.
    """
    n = len(signal)
    if n < 4:
        return signal.copy()
    kernel = np.ones(4) / 4.0
    return np.convolve(signal, kernel, mode='same')


# =============================================================================
# FITTING DIRETTO SULLA CORRELAZIONE (invece di frequenza+tau separati)
# =============================================================================

def _envelope_hilbert(signal):
    """Inviluppo di Hilbert, identico al metodo usato altrove nell'app."""
    N = len(signal)
    Xf = np.fft.rfft(signal, n=N)
    h = np.zeros(N // 2 + 1, dtype=np.float64)
    h[0] = 1.0
    if N % 2 == 0:
        h[1:-1] = 2.0
        h[-1] = 1.0
    else:
        h[1:] = 2.0
    hilbert_part = np.fft.irfft(Xf * h, n=N)
    return np.sqrt(signal ** 2 + hilbert_part ** 2)


def _find_envelope_bounds(signal, level_fraction=0.1, max_search=500):
    """
    Trova inizio, picco e fine di un segnale tramite soglia
    sull'inviluppo di Hilbert — identico al metodo usato in UI
    (_find_click_envelope_bounds), per garantire che il fitting
    ottimizzi esattamente la stessa metrica che viene poi mostrata.
    """
    if signal is None or len(signal) < 10:
        return None
    try:
        envelope = _envelope_hilbert(signal)
        peak_idx = int(np.argmax(envelope))
        peak_amp = envelope[peak_idx]
        if peak_amp <= 0:
            return None
        level = peak_amp * level_fraction

        start_idx = max(peak_idx - max_search, 0)
        for i in range(peak_idx - 1, max(peak_idx - max_search, -1), -1):
            if envelope[i] < level:
                start_idx = i + 1
                break

        end_idx = min(peak_idx + max_search, len(envelope) - 1)
        for i in range(peak_idx + 1, end_idx + 1):
            if envelope[i] < level:
                end_idx = i
                break

        return start_idx, peak_idx, end_idx
    except Exception:
        return None


def _correlation_for_params(R0, extra_damping_rate, real_signal_smoothed, distance_m):
    """
    Simula con (R0, extra_damping_rate), applica la stessa pipeline
    completa e la stessa media mobile del segnale reale, poi calcola la
    correlazione usando ESATTAMENTE lo stesso metodo di allineamento
    (soglia sull'inviluppo, non solo il picco assoluto con finestra
    fissa) usato per il numero mostrato in UI — così il fitting
    ottimizza davvero la metrica che poi si vede.
    """
    try:
        bubble = simulate_bubble_collapse(R0=R0, extra_damping_rate=extra_damping_rate)
        propagation = apply_propagation(
            p_source=bubble['p_source'], t=bubble['t'], distance_m=distance_m
        )
        sim_signal = apply_smoothing_4samples(propagation['signal'])

        real_bounds = _find_envelope_bounds(real_signal_smoothed)
        sim_bounds = _find_envelope_bounds(sim_signal)
        if real_bounds is None or sim_bounds is None:
            return 0.0

        real_start, real_peak, real_end = real_bounds
        sim_start, sim_peak, sim_end = sim_bounds

        pre = max(real_peak - real_start, sim_peak - sim_start)
        post = max(real_end - real_peak, sim_end - sim_peak)
        pre = min(pre, real_peak, sim_peak)
        post = min(post, len(real_signal_smoothed) - real_peak - 1, len(sim_signal) - sim_peak - 1)
        if post <= 3:
            return 0.0

        real_env = _envelope_hilbert(real_signal_smoothed)
        sim_env = _envelope_hilbert(sim_signal)

        r_win = real_env[real_peak - pre: real_peak + post + 1]
        s_win = sim_env[sim_peak - pre: sim_peak + post + 1]
        n = min(len(r_win), len(s_win))
        if n <= 6:
            return 0.0
        r_win = r_win[:n]
        s_win = s_win[:n]

        r_norm = r_win / (np.max(r_win) + 1e-30)
        s_norm = s_win / (np.max(s_win) + 1e-30)

        corr = np.corrcoef(r_norm, s_norm)[0, 1]
        return float(corr) if np.isfinite(corr) else 0.0
    except Exception:
        return 0.0


def fit_R0_and_damping_for_correlation(real_signal, distance_m,
                                        R0_min=None, R0_max=None,
                                        n_scan_R0=15, n_scan_damping=12):
    """
    Trova la coppia (R0, smorzamento extra) che massimizza direttamente
    la correlazione tra l'inviluppo del click reale e quello simulato —
    invece di tarare frequenza e tau separatamente sperando che la
    correlazione ne segua. Scansione a griglia (stabile, sempre
    termina), poi raffinamento locale attorno al miglior punto trovato.

    Returns:
        dict con 'R0', 'extra_damping_rate', 'correlation'
    """
    if R0_min is None:
        R0_min = BubbleParameters.R0_MIN
    if R0_max is None:
        R0_max = BubbleParameters.R0_MAX

    real_signal_smoothed = apply_smoothing_4samples(real_signal)

    R0_grid = np.linspace(R0_min, R0_max, n_scan_R0)
    damping_grid = np.logspace(2, 5, n_scan_damping)  # 100 - 100000 1/s

    best_corr = -2.0
    best_R0 = BubbleParameters.R0_DEFAULT
    best_damping = 0.0

    for R0c in R0_grid:
        for dc in damping_grid:
            c = _correlation_for_params(R0c, dc, real_signal_smoothed, distance_m)
            if c > best_corr:
                best_corr, best_R0, best_damping = c, R0c, dc

    # Raffinamento locale: griglia più fitta attorno al miglior punto
    R0_span = (R0_max - R0_min) / n_scan_R0
    R0_grid_fine = np.linspace(max(R0_min, best_R0 - R0_span), min(R0_max, best_R0 + R0_span), 10)
    damping_grid_fine = np.linspace(max(0, best_damping * 0.5), best_damping * 1.5 + 1, 10)

    for R0c in R0_grid_fine:
        for dc in damping_grid_fine:
            c = _correlation_for_params(R0c, dc, real_signal_smoothed, distance_m)
            if c > best_corr:
                best_corr, best_R0, best_damping = c, R0c, dc

    return {'R0': best_R0, 'extra_damping_rate': best_damping, 'correlation': best_corr}

# =============================================================================
# ESTRAZIONE PARAMETRI DIAGNOSTICI
# =============================================================================

def extract_diagnostics(signal, t):
    """
    Estrae i parametri diagnostici dal segnale simulato usando
    la stessa procedura dell'algoritmo PlantLeaf v4.0.
    """
    diagnostics = {}

    if len(signal) < 10 or np.max(np.abs(signal)) < 1e-20:
        return {
            'tau': None, 'SPR': None,
            'R_spectral': None, 'asymmetry': None,
            'peak_amplitude': None, 'rms': None
        }

    analytic = hilbert(signal)
    envelope = np.abs(analytic)

    peak_amplitude = np.max(envelope)
    rms = np.sqrt(np.mean(signal ** 2))
    diagnostics['peak_amplitude'] = peak_amplitude
    diagnostics['rms'] = rms

    diagnostics['SPR'] = peak_amplitude / rms if rms > 0 else None

    tau = compute_tau(envelope, t)
    diagnostics['tau'] = tau

    mid = len(signal) // 2
    energy_first = np.sum(signal[:mid] ** 2)
    energy_second = np.sum(signal[mid:] ** 2)
    if (energy_first + energy_second) > 0:
        diagnostics['asymmetry'] = energy_first / (energy_first + energy_second)
    else:
        diagnostics['asymmetry'] = None

    diagnostics['R_spectral'] = compute_r_spectral(signal, t)

    return diagnostics


def compute_tau(envelope, t):
    """
    Calcola la costante di smorzamento τ tramite regressione log-lineare
    sull'inviluppo di Hilbert dopo il picco.
    """
    try:
        peak_idx = np.argmax(envelope)

        env_post = envelope[peak_idx:]
        t_post = t[peak_idx:]

        valid = env_post > 0
        if np.sum(valid) < 5:
            return None

        log_env = np.log(env_post[valid])
        t_valid = t_post[valid]

        slope, intercept, r_value, p_value, std_err = linregress(t_valid, log_env)

        if slope >= 0:
            return None

        tau = -1.0 / slope
        return tau

    except Exception:
        return None


def compute_r_spectral(signal, t):
    """
    Calcola R_spectral: correlazione tra lo spettro del segnale simulato
    e uno spettro di riferimento gaussiano centrato a 25 kHz.
    """
    try:
        n = len(signal)
        dt = t[1] - t[0]

        S_fft = np.abs(np.fft.rfft(signal)) / n
        freq_hz = np.fft.rfftfreq(n, d=dt)

        mask = (freq_hz >= PlantLeafConfig.FREQ_MIN) & \
               (freq_hz <= PlantLeafConfig.FREQ_MAX)
        S_filtered = S_fft[mask]
        freq_filtered = freq_hz[mask]

        if len(S_filtered) < 2:
            return None

        f_center = 25000.0
        f_sigma = 10000.0
        S_ref = np.exp(-0.5 * ((freq_filtered - f_center) / f_sigma) ** 2)

        S_filtered_norm = S_filtered / (np.max(S_filtered) + 1e-30)
        S_ref_norm = S_ref / (np.max(S_ref) + 1e-30)

        correlation = np.corrcoef(S_filtered_norm, S_ref_norm)[0, 1]
        return float(correlation)

    except Exception:
        return None


# =============================================================================
# RICAMPIONAMENTO SULL'ASSE PLANTLEAF
# =============================================================================

def resample_to_plantleaf(freq, spectrum):
    """
    Ricampiona lo spettro simulato sull'asse frequenze di PlantLeaf
    (154 bins da 20 a 80 kHz) per il confronto diretto con i dati reali.
    """
    plantleaf_freq = PlantLeafConfig.get_freq_axis()

    if len(freq) == 0:
        spectrum_resampled = np.zeros(len(plantleaf_freq))
    else:
        spectrum_resampled = np.interp(
            plantleaf_freq,
            freq,
            spectrum,
            left=0.0,
            right=0.0
        )

    return {
        'freq': plantleaf_freq,
        'spectrum': spectrum_resampled
    }