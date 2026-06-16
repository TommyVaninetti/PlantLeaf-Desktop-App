"""
run_acoustic_simulation.py
==========================
Pipeline completa del simulatore acustico PlantLeaf.

Collega Step 1, 2, 3 e produce il click simulato finale con
i parametri diagnostici estratti con la stessa procedura
dell'algoritmo PlantLeaf v4.0.

Utilizzo:
    from run_acoustic_simulation import run_simulation
    result = run_simulation(R0=50e-6, P_inf=-0.5e6)
"""

import numpy as np
from scipy.signal import hilbert
from scipy.stats import linregress

from acoustic_parameters import (
    BubbleParameters,
    XylemPressure,
    PlantLeafConfig
)
from rayleigh_plesset import simulate_bubble_collapse
from acoustic_propagation import apply_propagation


# =============================================================================
# FUNZIONE PRINCIPALE
# =============================================================================

def run_simulation(R0=None, P_inf=None, distance_m=None, tau_target_ms=None):
    """
    Esegue la simulazione completa del click ultrasonico.

    Args:
        R0         : raggio iniziale della bolla [m].
                     Default: 50 µm
        P_inf      : tensione idrica dello xilema [Pa].
                     Default: -0.3 MPa
        distance_m : distanza bolla-microfono [m].
                     Default: 1 cm

    Returns:
        dict con le chiavi:
            'bubble'      : risultati dinamica bolla (da rayleigh_plesset)
            'propagation' : risultati propagazione (da acoustic_propagation)
            'diagnostics' : parametri diagnostici estratti dal segnale
            'plantleaf'   : segnale ricampionato sull'asse PlantLeaf
    """
    # Valori di default
    if R0 is None:
        R0 = BubbleParameters.R0_DEFAULT
    if P_inf is None:
        P_inf = XylemPressure.P_INF_DEFAULT

    # --- Step 2: dinamica della bolla ---
    bubble = simulate_bubble_collapse(R0=R0, P_inf=P_inf, tau_target_ms=tau_target_ms)

    # --- Step 3: propagazione acustica ---
    propagation = apply_propagation(
        p_source=bubble['p_source'],
        t=bubble['t'],
        distance_m=distance_m
    )

    # --- Estrazione parametri diagnostici ---
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
        'plantleaf': plantleaf
    }


# =============================================================================
# ESTRAZIONE PARAMETRI DIAGNOSTICI
# =============================================================================

def extract_diagnostics(signal, t):
    """
    Estrae i parametri diagnostici dal segnale simulato usando
    la stessa procedura dell'algoritmo PlantLeaf v4.0.

    Parametri estratti:
        - tau    : costante di smorzamento [s] — via regressione log-lineare
                   sull'inviluppo di Hilbert
        - SPR    : Signal Peak Ratio — rapporto picco/RMS
        - R_spec : R_spectral — coefficiente di correlazione spettrale
        - asym   : asymmetry ratio — asimmetria del segnale

    Args:
        signal : array del segnale simulato [Pa]
        t      : array dei tempi [s]

    Returns:
        dict con i parametri diagnostici
    """
    diagnostics = {}

    # Protezione: segnale vuoto o costante
    if len(signal) < 10 or np.max(np.abs(signal)) < 1e-20:
        return {
            'tau': None, 'SPR': None,
            'R_spectral': None, 'asymmetry': None,
            'peak_amplitude': None, 'rms': None
        }

    # --- Inviluppo di Hilbert ---
    analytic = hilbert(signal)
    envelope = np.abs(analytic)

    # --- Ampiezza di picco e RMS ---
    peak_amplitude = np.max(envelope)
    rms = np.sqrt(np.mean(signal ** 2))
    diagnostics['peak_amplitude'] = peak_amplitude
    diagnostics['rms'] = rms

    # --- SPR: Signal Peak Ratio ---
    diagnostics['SPR'] = peak_amplitude / rms if rms > 0 else None

    # --- tau: costante di smorzamento ---
    # Regressione log-lineare sull'inviluppo dopo il picco
    tau = compute_tau(envelope, t)
    diagnostics['tau'] = tau

    # --- Asymmetry ratio ---
    # Rapporto tra energia nella prima e nella seconda metà del segnale
    mid = len(signal) // 2
    energy_first = np.sum(signal[:mid] ** 2)
    energy_second = np.sum(signal[mid:] ** 2)
    if (energy_first + energy_second) > 0:
        diagnostics['asymmetry'] = energy_first / (energy_first + energy_second)
    else:
        diagnostics['asymmetry'] = None

    # --- R_spectral: coefficiente di correlazione spettrale ---
    # Correlazione tra spettro simulato e spettro atteso dal modello
    diagnostics['R_spectral'] = compute_r_spectral(signal, t)

    return diagnostics


def compute_tau(envelope, t):
    """
    Calcola la costante di smorzamento τ tramite regressione log-lineare
    sull'inviluppo di Hilbert dopo il picco.

    Il modello è: envelope(t) = A · exp(-t/τ)
    In forma logaritmica: log(envelope) = log(A) - t/τ
    → regressione lineare di log(envelope) su t dopo il picco.

    Args:
        envelope : array inviluppo di Hilbert
        t        : array dei tempi [s]

    Returns:
        float: costante di smorzamento τ [s], o None se il fit fallisce
    """
    try:
        # Trova l'indice del picco
        peak_idx = np.argmax(envelope)

        # Usa solo i campioni dopo il picco
        env_post = envelope[peak_idx:]
        t_post = t[peak_idx:]

        # Rimuovi valori <= 0 per il logaritmo
        valid = env_post > 0
        if np.sum(valid) < 5:
            return None

        log_env = np.log(env_post[valid])
        t_valid = t_post[valid]

        # Regressione lineare
        slope, intercept, r_value, p_value, std_err = linregress(t_valid, log_env)

        # τ = -1/slope (slope deve essere negativo per smorzamento)
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

    Args:
        signal : array del segnale simulato
        t      : array dei tempi [s]

    Returns:
        float: coefficiente di correlazione R_spectral (0–1)
    """
    try:
        n = len(signal)
        dt = t[1] - t[0]

        # Spettro del segnale simulato
        S_fft = np.abs(np.fft.rfft(signal)) / n
        freq_hz = np.fft.rfftfreq(n, d=dt)

        # Filtra range PlantLeaf
        mask = (freq_hz >= PlantLeafConfig.FREQ_MIN) & \
               (freq_hz <= PlantLeafConfig.FREQ_MAX)
        S_filtered = S_fft[mask]
        freq_filtered = freq_hz[mask]

        if len(S_filtered) < 2:
            return None

        # Spettro di riferimento: gaussiana centrata a 25 kHz
        f_center = 25000.0
        f_sigma = 10000.0
        S_ref = np.exp(-0.5 * ((freq_filtered - f_center) / f_sigma) ** 2)

        # Normalizza entrambi
        S_filtered_norm = S_filtered / (np.max(S_filtered) + 1e-30)
        S_ref_norm = S_ref / (np.max(S_ref) + 1e-30)

        # Correlazione di Pearson
        correlation = np.corrcoef(S_filtered_norm, S_ref_norm)[0, 1]
        return float(correlation)

    except Exception:
        return None


# =============================================================================
# RICAMPIONAMENTO SULL'ASSE PLANTLEAF
# =============================================================================

def resample_to_plantleaf(freq, spectrum):
    plantleaf_freq = PlantLeafConfig.get_freq_axis()
    
    if len(freq) == 0 or len(spectrum) == 0:
        return {
            'freq': plantleaf_freq,
            'spectrum': np.zeros(len(plantleaf_freq))
        }

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