"""
rayleigh_plesset.py
====================
Modello di risonanza di bolla smorzata da irraggiamento acustico, per il
click ultrasonico di cavitazione xilematica.

A differenza di un tentativo precedente (integrazione diretta di
Rayleigh-Plesset sotto tensione P∞ < 0), che risulta fisicamente instabile
— senza equilibrio, la bolla cresce indefinitamente (vedi nota sotto) —
qui si usa il modello validato in letteratura per l'evento acustico:
un nucleo di cavitazione, una volta formatosi, oscilla ed è smorzato
attorno a un raggio di equilibrio R0 con la frequenza naturale di
Minnaert, smorzata principalmente per irraggiamento acustico.

Frequenza:
    f0 = (1/2πR0) · √[(3γP0 + (3γ-1)·2σ/R0) / ρ]

Smorzamento (irraggiamento + viscoso):
    b_rad = ω0²·R0 / (2c)      τ_rad = 1/b_rad
    b_vis = 2µ / (ρR0²)         τ_vis = 1/b_vis
    τ = 1 / (b_rad + b_vis)

Il raggio oscilla quindi come:
    R(t) = R0 + ΔR·exp(-t/τ)·cos(ω0·t)

con ΔR = R0 · BubbleResonance.PERTURBATION_FRACTION.

Nota sul perché non si integra più l'equazione RP classica sotto
tensione: con P∞ negativo (tensione idrica dello xilema) e pressione del
gas che cala con la crescita della bolla, non esiste un punto di
equilibrio stabile — la bolla cresce senza fermarsi. Il modello di
risonanza qui adottato bypassa il problema modellando direttamente
l'oscillazione smorzata attorno a un raggio di equilibrio fisico (R0),
coerente con l'osservazione che il click è un evento oscillante e
limitato in ampiezza, non un'espansione o un collasso violento.

Fonti:
    - Minnaert (1933)
    - Brennen (1995), Cap. 2 — Cavitation and Bubble Dynamics
    - Plesset & Prosperetti (1977)
"""

import numpy as np
from scipy.optimize import brentq

from acoustic_parameters import (
    WaterProperties,
    BubbleParameters,
    XylemPressure,
    BubbleResonance
)


# =============================================================================
# FREQUENZA DI RISONANZA E SMORZAMENTO (formule chiuse, no parametri tarati)
# =============================================================================

def minnaert_frequency(R0):
    """
    Calcola la frequenza di risonanza di Minnaert per una bolla di raggio
    R0 in equilibrio a pressione atmosferica.

    Args:
        R0 : raggio della bolla [m]

    Returns:
        (f0, omega0) : frequenza [Hz] e pulsazione angolare [rad/s]
    """
    p0 = BubbleParameters.P_ATM
    rho = WaterProperties.DENSITY
    sigma = WaterProperties.SURFACE_TENSION
    gamma = WaterProperties.GAMMA_GAS

    omega0_sq = (3.0 * gamma * p0 + (3.0 * gamma - 1.0) * 2.0 * sigma / R0) / (rho * R0 ** 2)
    omega0 = np.sqrt(omega0_sq)
    f0 = omega0 / (2.0 * np.pi)
    return f0, omega0


def damping_time_constant(R0, omega0, extra_damping_rate=0.0):
    """
    Calcola il tempo di decadimento τ dovuto a smorzamento per
    irraggiamento acustico, viscoso, e un eventuale smorzamento
    aggiuntivo (extra_damping_rate) che rappresenta l'attrito con la
    parete del vaso xilematico — un meccanismo non incluso nel modello
    "bolla libera in acqua infinita", e che varia da vaso a vaso.
    Quando fornito, questo termine è stimato dal fitting sul click
    reale (vedi run_acoustic_simulation.py), non da una legge fisica
    universale.

    Args:
        R0                  : raggio della bolla [m]
        omega0              : pulsazione angolare di risonanza [rad/s]
        extra_damping_rate  : tasso di smorzamento aggiuntivo [1/s]

    Returns:
        float: tau [s]
    """
    rho = WaterProperties.DENSITY
    mu = WaterProperties.VISCOSITY
    c = WaterProperties.SPEED_OF_SOUND

    b_rad = omega0 ** 2 * R0 / (2.0 * c)
    b_vis = 2.0 * mu / (rho * R0 ** 2)
    return 1.0 / (b_rad + b_vis + extra_damping_rate)


def damping_components(R0, omega0):
    """
    Restituisce separatamente i due contributi di smorzamento fisico
    (irraggiamento e viscoso), utile per calcolare quanto smorzamento
    aggiuntivo serve per riprodurre un tau osservato.

    Returns:
        (b_rad, b_vis) : tassi di smorzamento [1/s]
    """
    rho = WaterProperties.DENSITY
    mu = WaterProperties.VISCOSITY
    c = WaterProperties.SPEED_OF_SOUND

    b_rad = omega0 ** 2 * R0 / (2.0 * c)
    b_vis = 2.0 * mu / (rho * R0 ** 2)
    return b_rad, b_vis


# =============================================================================
# SINTESI DELL'OSCILLAZIONE R(t), V(t)
# =============================================================================

def synthesize_bubble_oscillation(R0, n_points=5000, t_max=None, extra_damping_rate=0.0):
    """
    Sintetizza l'oscillazione smorzata del raggio della bolla attorno a
    R0, con un'accensione graduale (velocità E accelerazione iniziali
    esattamente zero, per evitare transienti artificiali all'istante
    t=0) seguita dall'oscillazione smorzata vera e propria.

    Args:
        R0                  : raggio di equilibrio della bolla [m]
        n_points            : numero di campioni temporali
        t_max               : durata della simulazione [s]. Se None,
                               stimata come multiplo di tau, con un
                               minimo che copre sempre la finestra
                               visibile nei grafici dell'app (150 µs).
        extra_damping_rate  : smorzamento aggiuntivo [1/s].

    Returns:
        dict con 't', 'R', 'V', 'f0', 'omega0', 'tau'
    """
    f0, omega0 = minnaert_frequency(R0)
    tau = damping_time_constant(R0, omega0, extra_damping_rate)

    if t_max is None:
        t_max = min(max(6.0 * tau, 150e-6), 2e-3)

    t = np.linspace(0, t_max, n_points)
    deltaR = R0 * BubbleResonance.PERTURBATION_FRACTION

    tr = (np.pi / 2.0) / omega0

    rise = (1.0 - np.exp(-t / tr)) ** 2
    decay = np.exp(-t / tau)
    osc = np.sin(omega0 * t)

    R = R0 + deltaR * rise * decay * osc
    V = np.gradient(R, t)

    return {'t': t, 'R': R, 'V': V, 'f0': f0, 'omega0': omega0, 'tau': tau}


# =============================================================================
# FUNZIONE PRINCIPALE DI SIMULAZIONE
# =============================================================================

def simulate_bubble_collapse(R0=None, P_inf=None, t_max=None, n_points=5000,
                              tau_target_ms=None, G_shell=None, extra_damping_rate=0.0):
    """
    Simula l'oscillazione della bolla e calcola la pressione irradiata.

    Args:
        R0                  : raggio di equilibrio della bolla [m].
                               Default: 50 µm.
        P_inf               : tensione idrica dello xilema [Pa]. Solo
                               per display (vedi docstring del modulo).
        t_max               : durata della simulazione [s].
        n_points            : numero di campioni temporali.
        tau_target_ms       : se specificato e R0 non è dato, stima R0
                               (uso raro: preferire la calibrazione in
                               run_acoustic_simulation.py).
        G_shell             : parametro deprecato, ignorato.
        extra_damping_rate  : smorzamento aggiuntivo del vaso [1/s],
                               stimato dal fitting sul click reale.

    Returns:
        dict con 't', 'R', 'V', 'p_source', 'R0', 'P_inf', 'P_gas0',
        'f0', 'extra_damping_rate', 'collapsed' (sempre False)
    """
    if R0 is None:
        R0 = BubbleParameters.R0_DEFAULT
    if P_inf is None:
        P_inf = XylemPressure.P_INF_DEFAULT

    if tau_target_ms is not None and tau_target_ms > 0 and R0 is None:
        R0 = solve_R0_for_tau(tau_target_ms / 1000.0)

    osc = synthesize_bubble_oscillation(R0, n_points=n_points, t_max=t_max,
                                         extra_damping_rate=extra_damping_rate)
    t, R, V = osc['t'], osc['R'], osc['V']

    p_source = compute_radiated_pressure(R, V, t, R0)

    return {
        't': t,
        'R': R,
        'V': V,
        'p_source': p_source,
        'R0': R0,
        'P_inf': P_inf,
        'P_gas0': BubbleParameters.P_ATM,
        'f0': osc['f0'],
        'extra_damping_rate': extra_damping_rate,
        'collapsed': False
    }


# =============================================================================
# PRESSIONE IRRADIATA DALLA BOLLA
# =============================================================================

def compute_radiated_pressure(R, V, t, R0):
    """
    Calcola la pressione irradiata dalla bolla oscillante, usando
    differenziazione numerica centrale (più accurata ai bordi rispetto
    alla differenza in avanti, importante ora che il segnale parte da
    una condizione perfettamente liscia).

    Args:
        R  : array del raggio nel tempo [m]
        V  : array della velocità dR/dt [m/s]
        t  : array dei tempi [s]
        R0 : raggio di equilibrio [m]

    Returns:
        np.ndarray: pressione irradiata [Pa]
    """
    rho = WaterProperties.DENSITY
    dV_dt = np.gradient(V, t)
    p_source = rho * R ** 2 * (R * dV_dt + 2.0 * V ** 2) / R0
    return p_source


# =============================================================================
# INVERSIONE: TROVA R0 DATO UN TARGET DI FREQUENZA O TAU (bisezione)
# =============================================================================

def solve_R0_for_freq(freq_target_hz, R0_bounds=None):
    """
    Trova l'R0 [m] per cui la frequenza di Minnaert è pari a
    freq_target_hz, tramite bisezione (la relazione è monotona e
    numericamente ben condizionata, sempre stabile).

    Args:
        freq_target_hz : frequenza target [Hz]
        R0_bounds      : tupla (min, max) di R0 in metri per la ricerca.
                          Default: range biologico esteso.

    Returns:
        float: R0 [m]
    """
    if R0_bounds is None:
        R0_bounds = (5e-6, 500e-6)

    def f(R0):
        freq, _ = minnaert_frequency(R0)
        return freq - freq_target_hz

    try:
        return brentq(f, R0_bounds[0], R0_bounds[1], xtol=1e-10)
    except Exception:
        # Target fuori dal range raggiungibile: restituisci l'estremo più vicino
        f_min = f(R0_bounds[0])
        f_max = f(R0_bounds[1])
        return R0_bounds[0] if abs(f_min) < abs(f_max) else R0_bounds[1]


def solve_R0_for_tau(tau_target_s, R0_bounds=None):
    """
    Trova l'R0 [m] per cui il tau di decadimento teorico è pari a
    tau_target_s, tramite bisezione.

    Args:
        tau_target_s : tau target [s]
        R0_bounds     : tupla (min, max) di R0 in metri per la ricerca.

    Returns:
        float: R0 [m]
    """
    if R0_bounds is None:
        R0_bounds = (5e-6, 500e-6)

    def f(R0):
        _, omega0 = minnaert_frequency(R0)
        tau = damping_time_constant(R0, omega0)
        return tau - tau_target_s

    try:
        return brentq(f, R0_bounds[0], R0_bounds[1], xtol=1e-10)
    except Exception:
        f_min = f(R0_bounds[0])
        f_max = f(R0_bounds[1])
        return R0_bounds[0] if abs(f_min) < abs(f_max) else R0_bounds[1]