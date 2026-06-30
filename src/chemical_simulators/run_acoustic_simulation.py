"""
run_acoustic_simulation.py
==========================
Modello fenomenologico del click ultrasonico xilematico.
Genera una sinusoide smorzata calibrata sui parametri reali del click.
"""

import numpy as np
from acoustic_parameters import (
    BubbleParameters,
    XylemPressure,
    PlantLeafConfig
)


def run_simulation(R0=None, P_inf=None, distance_m=None, tau_target_ms=None):
    if R0 is None:
        R0 = BubbleParameters.R0_DEFAULT
    if P_inf is None:
        P_inf = XylemPressure.P_INF_DEFAULT
    if distance_m is None:
        distance_m = 0.01

    # Parametri del modello fenomenologico
    tau_s = (tau_target_ms / 1000.0) if tau_target_ms and tau_target_ms > 0 else 0.000249
    fs = PlantLeafConfig.FS
    n_points = PlantLeafConfig.FFT_SIZE
    t = np.linspace(0, n_points / fs, n_points)

    # Frequenza centrale stimata da P_inf
    # Più P_inf è negativo, più la frequenza è alta
    f_center = 25000 + abs(P_inf) / 1e6 * 5000
    f_center = np.clip(f_center, 20000, 80000)

    # Ampiezza scalata con R0
    amplitude = (R0 / BubbleParameters.R0_DEFAULT) * 1000.0

    # Segnale: sinusoide smorzata
    signal = amplitude * np.exp(-t / tau_s) * np.cos(2 * np.pi * f_center * t)

    # Decadimento geometrico
    r_ref = 0.01
    signal = signal * (r_ref / max(distance_m, 1e-6))

    # Spettro
    S_fft = np.abs(np.fft.rfft(signal)) / n_points
    freq_hz = np.fft.rfftfreq(n_points, d=1.0/fs)
    mask = (freq_hz >= PlantLeafConfig.FREQ_MIN) & (freq_hz <= PlantLeafConfig.FREQ_MAX)
    freq_filtered = freq_hz[mask]
    amplitude_filtered = S_fft[mask]

    # Diagnostics
    envelope = np.abs(signal)
    peak_amplitude = np.max(envelope)
    rms = np.sqrt(np.mean(signal ** 2))
    spr = peak_amplitude / rms if rms > 0 else None

    mid = len(signal) // 2
    e1 = np.sum(signal[:mid] ** 2)
    e2 = np.sum(signal[mid:] ** 2)
    asymmetry = e1 / (e1 + e2) if (e1 + e2) > 0 else None

    plantleaf_freq = PlantLeafConfig.get_freq_axis()
    if len(freq_filtered) == 0:
        spectrum_resampled = np.zeros(len(plantleaf_freq))
    else:
        spectrum_resampled = np.interp(
            plantleaf_freq, freq_filtered, amplitude_filtered,
            left=0.0, right=0.0
        )

    # Bubble dict (compatibilità con UI)
    bubble = {
        't': t,
        'R': np.ones(n_points) * R0,
        'V': np.zeros(n_points),
        'p_source': signal,
        'R0': R0,
        'P_inf': P_inf,
        'P_gas0': 0,
        'collapsed': True
    }

    propagation = {
        'signal': signal,
        'signal_tissue': signal,
        'freq': freq_filtered,
        'spectrum': amplitude_filtered,
        'distance_m': distance_m
    }

    diagnostics = {
        'tau': tau_s,
        'SPR': spr,
        'R_spectral': None,
        'asymmetry': asymmetry,
        'peak_amplitude': peak_amplitude,
        'rms': rms
    }

    plantleaf = {
        'freq': plantleaf_freq,
        'spectrum': spectrum_resampled
    }

    return {
        'bubble': bubble,
        'propagation': propagation,
        'diagnostics': diagnostics,
        'plantleaf': plantleaf
    }