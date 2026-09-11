"""
acoustic_propagation.py
=======================
Modella la propagazione del click acustico dalla bolla al microfono.

Pipeline:
    p_source(t)  →  attenuazione tessuto  →  decadimento geometrico
                 →  risposta microfono  →  segnale PlantLeaf

Fonti:
    - Khait et al. (2023): propagazione acustica nel tessuto vegetale
    - Knowles SPU0410LR5H datasheet: risposta in frequenza microfono
"""

import numpy as np
from acoustic_parameters import (
    WaterProperties,
    MicrophoneResponse,
    PlantLeafConfig,
    PropagationParameters
)


# =============================================================================
# FUNZIONE PRINCIPALE
# =============================================================================

def apply_propagation(p_source, t, distance_m=None):
    """
    Applica tutte le trasformazioni fisiche al segnale di pressione
    dalla sorgente fino al microfono.

    Args:
        p_source   : array pressione irradiata alla sorgente [Pa]
        t          : array dei tempi [s]
        distance_m : distanza bolla-microfono [m].
                     Default: PropagationParameters.DISTANCE_DEFAULT (1 cm)

    Returns:
        dict con le chiavi:
            'signal'         : segnale finale al microfono [Pa]
            'signal_tissue'  : segnale dopo attenuazione tessuto (prima del microfono)
            'freq'           : asse frequenze della FFT [Hz]
            'spectrum'       : spettro in ampiezza del segnale finale
            'distance_m'     : distanza usata [m]
    """
    if distance_m is None:
        distance_m = PropagationParameters.DISTANCE_DEFAULT

    # Step 1 — attenuazione nel tessuto vegetale
    signal_tissue = apply_tissue_attenuation(p_source, t, distance_m)

    # Step 2 — decadimento geometrico sferico
    signal_geometric = apply_geometric_decay(signal_tissue, distance_m)

    # Step 3 — risposta del microfono
    signal_mic = apply_microphone_response(signal_geometric, t)

    # Calcola spettro del segnale finale
    freq, spectrum = compute_spectrum(signal_mic, t)

    return {
        'signal': signal_mic,
        'signal_tissue': signal_tissue,
        'freq': freq,
        'spectrum': spectrum,
        'distance_m': distance_m
    }


# =============================================================================
# ATTENUAZIONE NEL TESSUTO VEGETALE
# =============================================================================

def apply_tissue_attenuation(p_source, t, distance_m):
    """
    Applica l'attenuazione viscoelastica del tessuto vegetale al segnale.

    L'attenuazione dipende dalla frequenza: α [dB/cm/kHz]. Per un segnale
    a banda stretta come una risonanza di bolla smorzata, filtrare
    bin-per-bin nel dominio della frequenza attenuerebbe in modo
    differenziale le componenti spettrali vicine attorno al picco di
    risonanza, distorcendo artificialmente la forma del decadimento nel
    tempo (allungando il tau misurato). Per evitarlo, si usa un singolo
    fattore di attenuazione calcolato alla frequenza dominante del
    segnale, applicato uniformemente — fisicamente corretto per una
    sorgente essenzialmente monocromatica.

    Args:
        p_source   : array pressione sorgente [Pa]
        t          : array dei tempi [s]
        distance_m : distanza di propagazione [m]

    Returns:
        np.ndarray: segnale attenuato nel dominio del tempo [Pa]
    """
    n = len(p_source)
    dt = t[1] - t[0]

    P_fft = np.fft.rfft(p_source)
    freq_hz = np.fft.rfftfreq(n, d=dt)

    if len(P_fft) > 1:
        dominant_freq_hz = freq_hz[np.argmax(np.abs(P_fft[1:])) + 1]
    else:
        dominant_freq_hz = 0.0

    distance_cm = distance_m * 100.0
    freq_khz = dominant_freq_hz / 1000.0
    attenuation_db = PropagationParameters.ATTENUATION_COEFF * freq_khz * distance_cm
    attenuation_linear = 10.0 ** (-attenuation_db / 20.0)

    return p_source * attenuation_linear


# =============================================================================
# DECADIMENTO GEOMETRICO SFERICO
# =============================================================================

def apply_geometric_decay(signal, distance_m):
    """
    Applica il decadimento geometrico sferico al segnale.

    L'ampiezza dell'onda sferica decade come 1/r.
    Normalizziamo rispetto a una distanza di riferimento di 1 cm.

    Args:
        signal     : array del segnale [Pa]
        distance_m : distanza bolla-microfono [m]

    Returns:
        np.ndarray: segnale scalato per il decadimento geometrico [Pa]
    """
    # Distanza di riferimento: 1 cm
    r_ref = 0.01  # m

    # Fattore di decadimento geometrico
    geometric_factor = r_ref / max(distance_m, 1e-6)

    return signal * geometric_factor


# =============================================================================
# RISPOSTA DEL MICROFONO SPU0410LR5H
# =============================================================================

def apply_microphone_response(signal, t):
    """
    Applica la risposta in frequenza del microfono SPU0410LR5H al segnale.

    Per un segnale a banda stretta (come una risonanza di bolla smorzata),
    applicare la curva di risposta bin-per-bin nel dominio della frequenza
    distorcerebbe artificialmente la forma del decadimento nel tempo,
    esattamente come accadeva con l'attenuazione del tessuto (vedi
    apply_tissue_attenuation). Si usa quindi un singolo guadagno,
    valutato alla frequenza dominante del segnale, applicato
    uniformemente — corretto per una sorgente essenzialmente
    monocromatica.

    Args:
        signal : array del segnale in ingresso al microfono [Pa]
        t      : array dei tempi [s]

    Returns:
        np.ndarray: segnale misurato dal microfono [Pa]
    """
    n = len(signal)
    dt = t[1] - t[0]

    S_fft = np.fft.rfft(signal)
    freq_hz = np.fft.rfftfreq(n, d=dt)

    if len(S_fft) > 1:
        dominant_freq_hz = freq_hz[np.argmax(np.abs(S_fft[1:])) + 1]
    else:
        dominant_freq_hz = 0.0

    mic_gain = MicrophoneResponse.get_response_linear(dominant_freq_hz)

    return signal * mic_gain


# =============================================================================
# CALCOLO SPETTRO
# =============================================================================

def compute_spectrum(signal, t):
    """
    Calcola lo spettro in ampiezza del segnale nel range PlantLeaf (20–80 kHz).

    Args:
        signal : array del segnale nel tempo [Pa]
        t      : array dei tempi [s]

    Returns:
        tuple (freq, spectrum):
            freq     : array frequenze [Hz], range 20–80 kHz
            spectrum : array ampiezze normalizzate [Pa]
    """
    n = len(signal)
    dt = t[1] - t[0]

    # FFT e asse frequenze
    S_fft = np.fft.rfft(signal)
    freq_hz = np.fft.rfftfreq(n, d=dt)

    # Ampiezza normalizzata
    amplitude = np.abs(S_fft) / n

    # Filtra solo il range PlantLeaf: 20–80 kHz
    mask = (freq_hz >= PlantLeafConfig.FREQ_MIN) & (freq_hz <= PlantLeafConfig.FREQ_MAX)
    freq_filtered = freq_hz[mask]
    amplitude_filtered = amplitude[mask]

    return freq_filtered, amplitude_filtered