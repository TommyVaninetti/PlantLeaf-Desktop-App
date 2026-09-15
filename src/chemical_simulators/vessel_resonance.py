"""
vessel_resonance.py
===================
Il vaso xilematico come risuonatore acustico ("canna d'organo"), secondo

    Dutta S. et al. (2022), "Ultrasound Pulse Emission Spectroscopy Method to
    Characterize Xylem Conduits in Plant Stems", Research 2022:9790438,
    doi:10.34133/2022/9790438

Idea del modello: la formazione di una bolla nel vaso rilascia l'energia
elastica della colonna d'acqua in tensione; l'impulso eccita l'onda
stazionaria longitudinale dell'elemento di vaso (lunghezza L, raggio R), che
risuona e decade per viscosità della linfa. Il click è quindi una sinusoide
smorzata con

    f = m · v_eff / (2L)                                   Eq. (1), (9)
    1/v_eff² = 1/v_l² + ρ_l · β,   β = 2R / (h·E)          Eq. (2)
    τ_s = ρ_l · R² / (4 η_l)   →   R = √(4 η_l τ_s / ρ_l)  Eq. (5)

dove τ_s è il tempo in cui l'inviluppo cala di un fattore e.

Dalla coppia misurata (f, τ) si ricavano quindi R (da τ) e L (da f e R):
sono QUESTI i numeri da confrontare con l'anatomia della pianta — il modello
riproduce sempre la forma "sinusoide smorzata", il test è se R e L sono
realistici (come nella validazione del paper: raggio acustico vs microscopia).

Nota su Eq. (5): dalle Eq. (3)–(4) del paper, ζ = 4η·L/(ρ·v_l·R²) e
f_L = v_l/(2L), quindi 1/(ζ·f_L) = ρR²/(2η), mentre l'Eq. (5) stampata e il
calcolo dei raggi acustici usano ρR²/(4η). Qui si usa la forma STAMPATA,
perché è quella con cui gli autori hanno ottenuto i raggi validati. Se fosse
corretta l'altra, i raggi ricavati andrebbero divisi per √2 (vedi
CHANGELOG_SIMULATOR.md, Step B).

Nessuna dipendenza da Qt.
"""

import numpy as np

from acoustic_parameters import VesselParameters as VP


def effective_sound_speed(R, h=None, E=None):
    """v_eff [m/s] in un tubo a parete elastica di raggio R [m] (Eq. 2)."""
    h = VP.WALL_THICKNESS if h is None else h
    E = VP.YOUNG_MODULUS if E is None else E
    beta = 2.0 * R / (h * E)
    return 1.0 / np.sqrt(1.0 / VP.SOUND_SPEED_LIQUID ** 2 + VP.DENSITY_LIQUID * beta)


def settling_time(R):
    """τ_s [s] per un vaso di raggio R [m] (Eq. 5, forma stampata)."""
    return VP.DENSITY_LIQUID * R ** 2 / (4.0 * VP.VISCOSITY_LIQUID)


def radius_from_settling_time(tau_s):
    """Raggio acustico R [m] dal tempo di assestamento τ_s [s] (Eq. 5 invertita)."""
    return np.sqrt(4.0 * VP.VISCOSITY_LIQUID * tau_s / VP.DENSITY_LIQUID)


def resonance_frequency(L, R, h=None, E=None, m=None):
    """Frequenza del modo m [Hz] per un elemento di vaso lungo L e di raggio R [m]."""
    m = VP.MODE_ORDER if m is None else m
    return m * effective_sound_speed(R, h, E) / (2.0 * L)


def length_from_frequency(f, R, h=None, E=None, m=None):
    """Lunghezza acustica dell'elemento di vaso L [m] da f [Hz] e R [m] (Eq. 9)."""
    m = VP.MODE_ORDER if m is None else m
    return m * effective_sound_speed(R, h, E) / (2.0 * f)


def vessel_from_click(f, tau_s, h=None, E=None, m=None):
    """
    Geometria del vaso che, secondo il modello, produce un click (f, τ_s).

    Returns:
        dict con 'R' [m], 'L' [m], 'v_eff' [m/s], 'Q' (= π·f·τ_s)
    """
    R = radius_from_settling_time(tau_s)
    return {
        'R': R,
        'L': length_from_frequency(f, R, h, E, m),
        'v_eff': effective_sound_speed(R, h, E),
        'Q': np.pi * f * tau_s,
    }


def synthesize_vessel_pressure(t, f, tau_s, phase=0.0):
    """
    Forma d'onda del modello: sinusoide smorzata alla frequenza f con
    inviluppo exp(−t/τ_s), a partire dall'innesco t = 0 (zero prima).

    Per confrontarla con il modello a bolla usa la stessa accensione graduale
    di rayleigh_plesset.synthesize_bubble_oscillation, (1 − e^(−t/t_r))² con
    t_r = un quarto di periodo, così i due modelli differiscono solo per la
    fisica e non per un dettaglio di sintesi.

    Args:
        t      : tempi [s], t = 0 all'innesco
        f      : frequenza [Hz]
        tau_s  : tempo di assestamento dell'inviluppo [s]
        phase  : fase iniziale [rad]

    Returns:
        np.ndarray: pressione (ampiezza arbitraria, picco ~1)
    """
    t = np.asarray(t, dtype=np.float64)
    tp = np.clip(t, 0.0, None)
    omega = 2.0 * np.pi * f
    tr = (np.pi / 2.0) / omega
    p = (1.0 - np.exp(-tp / tr)) ** 2 * np.exp(-tp / tau_s) * np.sin(omega * tp + phase)
    p[t < 0] = 0.0
    return p
