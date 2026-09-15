"""
rayleigh_plesset.py
====================
Modello di risonanza di bolla smorzata, per il click ultrasonico di
cavitazione xilematica (teoria lineare della bolla libera in acqua).

A differenza di un tentativo precedente (integrazione diretta di
Rayleigh-Plesset sotto tensione P∞ < 0), che risulta fisicamente instabile
— senza equilibrio, la bolla cresce indefinitamente (vedi nota sotto) —
qui si usa il modello validato in letteratura per l'evento acustico:
un nucleo di cavitazione, una volta formatosi, oscilla ed è smorzato
attorno a un raggio di equilibrio R0 con la frequenza naturale di
Minnaert.

Frequenza (Minnaert con tensione superficiale ed esponente politropico
effettivo κ, Prosperetti 1977):
    ω0² = [3κ·p_g0 − 2σ/R0] / (ρR0²),   p_g0 = p0 + 2σ/R0
    (con κ = γ si ritrova f0 = (1/2πR0)·√[(3γp0 + (3γ−1)·2σ/R0)/ρ])

Smorzamento (tassi d'ampiezza, inviluppo ∝ exp(−b·t)):
    b_rad = ω0²·R0 / (2c)                 irraggiamento acustico
    b_vis = 2µ / (ρR0²)                    viscosità
    b_th  = p_g0·Im(Φ) / (2ρ·ω0·R0²)       conduzione termica nel gas
    τ = 1 / (b_rad + b_vis + b_th),   Q = ω0 / (2b) = π·f0·τ

    Φ = 3γ / (1 − 3(γ−1)·iχ·[√(i/χ)·coth√(i/χ) − 1]),  χ = D/(ω·R0²),  κ = Re(Φ)/3
    D = diffusività termica del gas alla pressione p_g0.

    Nota (Step B, 2026-09): lo smorzamento termico mancava nella versione
    precedente. Per 20–80 kHz è il termine DOMINANTE (Q ≈ 9–14 invece di ~55):
    senza di esso τ risultava sovrastimato di ~4 volte.

Il raggio oscilla quindi come:
    R(t) = R0 + ΔR·rise(t)·exp(-t/τ)·sin(ω0·t)

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
    - Plesset & Prosperetti (1977); Prosperetti (1977) JASA 61:17
    - Devin (1959) JASA 31:1654; Ainslie & Leighton (2011) JASA 130:3184
"""

import numpy as np
from scipy.optimize import brentq

from acoustic_parameters import (
    WaterProperties,
    BubbleParameters,
    XylemPressure,
    BubbleResonance,
    GasProperties,
)


# =============================================================================
# FREQUENZA DI RISONANZA E SMORZAMENTO (formule chiuse, no parametri tarati)
# =============================================================================

def _gas_pressure(R0):
    """Pressione del gas all'equilibrio p_g0 = p0 + 2σ/R0 [Pa]."""
    return BubbleParameters.P_ATM + 2.0 * WaterProperties.SURFACE_TENSION / R0


def prosperetti_phi(R0, omega):
    """
    Funzione complessa Φ della teoria lineare di Prosperetti (1977) per il
    gas nella bolla: Re(Φ)/3 è l'esponente politropico effettivo κ (tra 1,
    isoterma, e γ, adiabatica), Im(Φ) dà lo smorzamento termico.

        Φ = 3γ / (1 − 3(γ−1)·iχ·[√(i/χ)·coth√(i/χ) − 1]),   χ = D/(ω·R0²)
    """
    gamma = WaterProperties.GAMMA_GAS
    D = GasProperties.thermal_diffusivity(_gas_pressure(R0))
    chi = D / (omega * R0 ** 2)
    s = np.sqrt(1j / chi)
    return 3.0 * gamma / (1.0 - 3.0 * (gamma - 1.0) * 1j * chi * (s / np.tanh(s) - 1.0))


def bubble_linear_properties(R0, n_iter=30):
    """
    Frequenza, esponente politropico e smorzamenti di una bolla d'aria
    libera di raggio R0 in acqua a 1 atm, dalla teoria lineare.

    κ dipende da ω e ω dipende da κ: si risolve per iterazione di punto
    fisso partendo dal caso adiabatico (converge in pochi passi).

    Returns:
        dict con f0 [Hz], omega0 [rad/s], kappa, b_rad, b_vis, b_th [1/s],
        tau [s], Q, delta_rad, delta_vis, delta_th, delta_tot
        (δ = 2b/ω0, costanti di smorzamento adimensionali; Q = 1/δ_tot).
    """
    rho = WaterProperties.DENSITY
    sigma = WaterProperties.SURFACE_TENSION
    mu = WaterProperties.VISCOSITY
    c = WaterProperties.SPEED_OF_SOUND
    p_g0 = _gas_pressure(R0)

    kappa = WaterProperties.GAMMA_GAS
    omega = np.sqrt((3.0 * kappa * p_g0 - 2.0 * sigma / R0) / (rho * R0 ** 2))
    for _ in range(n_iter):
        phi = prosperetti_phi(R0, omega)
        kappa = phi.real / 3.0
        omega_new = np.sqrt((3.0 * kappa * p_g0 - 2.0 * sigma / R0) / (rho * R0 ** 2))
        if abs(omega_new - omega) < 1e-10 * omega:
            omega = omega_new
            break
        omega = omega_new
    phi = prosperetti_phi(R0, omega)

    b_rad = omega ** 2 * R0 / (2.0 * c)
    b_vis = 2.0 * mu / (rho * R0 ** 2)
    b_th = p_g0 * phi.imag / (2.0 * rho * omega * R0 ** 2)
    b_tot = b_rad + b_vis + b_th
    return {
        'f0': omega / (2.0 * np.pi), 'omega0': omega, 'kappa': phi.real / 3.0,
        'b_rad': b_rad, 'b_vis': b_vis, 'b_th': b_th,
        'tau': 1.0 / b_tot, 'Q': omega / (2.0 * b_tot),
        'delta_rad': 2 * b_rad / omega, 'delta_vis': 2 * b_vis / omega,
        'delta_th': 2 * b_th / omega, 'delta_tot': 2 * b_tot / omega,
    }


def minnaert_frequency(R0):
    """
    Calcola la frequenza di risonanza di Minnaert per una bolla di raggio
    R0 in equilibrio a pressione atmosferica, con tensione superficiale ed
    esponente politropico effettivo κ (non più γ fisso: una bolla di
    decine di µm a decine di kHz non è adiabatica).

    Args:
        R0 : raggio della bolla [m]

    Returns:
        (f0, omega0) : frequenza [Hz] e pulsazione angolare [rad/s]
    """
    props = bubble_linear_properties(R0)
    return props['f0'], props['omega0']


def damping_time_constant(R0, omega0, extra_damping_rate=0.0):
    """
    Calcola il tempo di decadimento τ dovuto a smorzamento per
    irraggiamento acustico, viscoso, termico, e un eventuale smorzamento
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
    b_rad, b_vis, b_th = damping_components(R0, omega0)
    return 1.0 / (b_rad + b_vis + b_th + extra_damping_rate)


def damping_components(R0, omega0):
    """
    Restituisce separatamente i tre contributi di smorzamento fisico
    (irraggiamento, viscoso, termico), utile per calcolare quanto
    smorzamento aggiuntivo serve per riprodurre un tau osservato.

    Returns:
        (b_rad, b_vis, b_th) : tassi di smorzamento [1/s]
    """
    rho = WaterProperties.DENSITY
    mu = WaterProperties.VISCOSITY
    c = WaterProperties.SPEED_OF_SOUND

    b_rad = omega0 ** 2 * R0 / (2.0 * c)
    b_vis = 2.0 * mu / (rho * R0 ** 2)
    b_th = _gas_pressure(R0) * prosperetti_phi(R0, omega0).imag / (
        2.0 * rho * omega0 * R0 ** 2)
    return b_rad, b_vis, b_th


# =============================================================================
# SINTESI DELL'OSCILLAZIONE R(t), V(t)
# =============================================================================

def synthesize_bubble_oscillation(R0, n_points=5000, t_max=None, extra_damping_rate=0.0,
                                  t=None):
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
        t                   : griglia temporale esplicita [s], con t = 0
                               all'innesco (campioni con t < 0 → R = R0).
                               Se data, n_points e t_max sono ignorati.

    Returns:
        dict con 't', 'R', 'V', 'f0', 'omega0', 'tau'
    """
    f0, omega0 = minnaert_frequency(R0)
    tau = damping_time_constant(R0, omega0, extra_damping_rate)

    if t is None:
        if t_max is None:
            t_max = min(max(6.0 * tau, 150e-6), 2e-3)
        t = np.linspace(0, t_max, n_points)
    t = np.asarray(t, dtype=np.float64)
    deltaR = R0 * BubbleResonance.PERTURBATION_FRACTION

    tr = (np.pi / 2.0) / omega0
    tp = np.clip(t, 0.0, None)

    rise = (1.0 - np.exp(-tp / tr)) ** 2
    decay = np.exp(-tp / tau)
    osc = np.sin(omega0 * tp)

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

    Sorgente monopolare: p(r,t) = ρ·V''/(4πr), con V = (4/3)πR³, da cui

        p(r,t) = ρ/r · (R²·R'' + 2·R·R'²) = ρ/r · R · (R·R'' + 2·R'²)

    con r = R0. Unità risultanti: Pascal [Pa]. (La forma con R² al posto
    di R ha unità N/m, non Pa: vedi CAVITATION_MODEL_REVIEW.md §3.)

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
    p_source = rho * R * (R * dV_dt + 2.0 * V ** 2) / R0
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