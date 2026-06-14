"""
rayleigh_plesset.py
===================
Risoluzione numerica dell'equazione di Rayleigh-Plesset per la dinamica
della bolla di cavitazione nello xilema.

Equazione di Rayleigh-Plesset:
    ρ · (R·R'' + 3/2·R'²) = P_gas(t) - P∞ - 4µ·R'/R - 2σ/R

dove:
    R(t)    = raggio della bolla nel tempo [m]
    R'(t)   = dR/dt — velocità di variazione del raggio [m/s]
    R''(t)  = d²R/dt² — accelerazione del raggio [m/s²]
    P_gas   = pressione interna del gas (legge adiabatica) [Pa]
    P∞      = tensione idrica dello xilema (parametro biologico) [Pa]
    µ       = viscosità dinamica dell'acqua [Pa·s]
    σ       = tensione superficiale [N/m]
    ρ       = densità dell'acqua [kg/m³]

Fonti:
    - Rayleigh (1917), Plesset & Prosperetti (1977)
    - Brennen (1995), Cap. 2
    - Khait et al. (2023)
"""

import numpy as np
from scipy.integrate import odeint

from chemical_simulators.acoustic_parameters import (
    WaterProperties,
    BubbleParameters,
    XylemPressure,
    PropagationParameters
)


# =============================================================================
# SISTEMA DI EQUAZIONI DIFFERENZIALI
# =============================================================================

def rayleigh_plesset_ode(y, t, R0, P_inf, P_gas0):
    """
    Sistema ODE del primo ordine equivalente all'equazione di Rayleigh-Plesset.

    L'equazione originale è del secondo ordine in R(t).
    La trasformiamo in un sistema di due equazioni del primo ordine
    introducendo la variabile ausiliaria V = dR/dt:

        dy[0]/dt = y[1]               (dR/dt = V)
        dy[1]/dt = f(R, V, t)         (dV/dt = R'')

    Args:
        y      : array [R, V] — stato corrente della bolla
        t      : istante temporale corrente [s]
        R0     : raggio iniziale della bolla [m]
        P_inf  : tensione idrica dello xilema [Pa]
        P_gas0 : pressione iniziale del gas nella bolla [Pa]

    Returns:
        [dR/dt, dV/dt] — derivate dello stato
    """
    R = y[0]  # raggio corrente [m]
    V = y[1]  # velocità di variazione del raggio [m/s]

    # Protezione numerica: evita divisione per zero se R collassa a zero
    if R < BubbleParameters.R_COLLAPSE_THRESHOLD:
        return [0.0, 0.0]

    # Proprietà fisiche dell'acqua
    rho = WaterProperties.DENSITY
    mu  = WaterProperties.VISCOSITY
    sigma = WaterProperties.SURFACE_TENSION
    gamma = WaterProperties.GAMMA_GAS

    # Pressione interna del gas — legge adiabatica: P_gas = P_gas0 * (R0/R)^(3γ)
    P_gas = P_gas0 * (R0 / R) ** (3.0 * gamma)

    # Termine di pressione netta sulla parete della bolla
    pressure_term = P_gas - P_inf - (4.0 * mu * V / R) - (2.0 * sigma / R)

    # Equazione di Rayleigh-Plesset risolta per R'':
    # R'' = (pressure_term/ρ - 3/2 · V²) / R
    R_ddot = (pressure_term / rho - 1.5 * V ** 2) / R

    return [V, R_ddot]


# =============================================================================
# FUNZIONE PRINCIPALE DI SIMULAZIONE
# =============================================================================

def simulate_bubble_collapse(R0=None, P_inf=None, t_max=None, n_points=5000):
    """
    Simula il collasso di una bolla di cavitazione nello xilema.

    Risolve l'equazione di Rayleigh-Plesset dal momento di nucleazione
    fino al collasso completo o fino a t_max.

    Args:
        R0       : raggio iniziale della bolla [m].
                   Default: BubbleParameters.R0_DEFAULT (50 µm)
        P_inf    : tensione idrica dello xilema [Pa].
                   Default: XylemPressure.P_INF_DEFAULT (-0.3 MPa)
        t_max    : durata massima della simulazione [s].
                   Default: calcolata automaticamente da R0
        n_points : numero di punti temporali della soluzione

    Returns:
        dict con le chiavi:
            't'         : array tempi [s]
            'R'         : array raggi [m]
            'V'         : array velocità dR/dt [m/s]
            'p_source'  : array pressione irradiata alla sorgente [Pa]
            'R0'        : raggio iniziale usato [m]
            'P_inf'     : tensione xilematica usata [Pa]
            'P_gas0'    : pressione iniziale del gas [Pa]
            'collapsed' : True se la bolla ha raggiunto il collasso
    """
    # Valori di default dallo Step 1
    if R0 is None:
        R0 = BubbleParameters.R0_DEFAULT
    if P_inf is None:
        P_inf = XylemPressure.P_INF_DEFAULT

    # Condizione di equilibrio iniziale della bolla:
    # P_gas0 = P_atm + 2σ/R0 - P_inf
    # La bolla è in equilibrio quando la pressione interna bilancia
    # la pressione esterna più la tensione superficiale
    sigma = WaterProperties.SURFACE_TENSION
    P_gas0 = (BubbleParameters.P_ATM
              + 2.0 * sigma / R0
              - P_inf)

    # Durata simulazione: scala con R0 e P_inf
    # Una bolla più grande o con meno tensione impiega più tempo a collassare
    if t_max is None:
        rho = WaterProperties.DENSITY
        # Tempo di Rayleigh — stima analitica del tempo di collasso
        # T_R = 0.915 * R0 * sqrt(ρ / |P_inf|)
        T_rayleigh = 0.915 * R0 * np.sqrt(rho / abs(P_inf))
        # Simuliamo fino a 3 volte il tempo di Rayleigh per catturare
        # anche le oscillazioni post-collasso
        t_max = 3.0 * T_rayleigh

    # Array temporale
    t = np.linspace(0, t_max, n_points)

    # Condizioni iniziali: bolla ferma al raggio R0
    y0 = [R0, 0.0]  # [R(0) = R0, V(0) = 0]

    # Risoluzione ODE con scipy
    solution = odeint(
        rayleigh_plesset_ode,
        y0,
        t,
        args=(R0, P_inf, P_gas0),
        rtol=1e-8,   # tolleranza relativa
        atol=1e-12,  # tolleranza assoluta
        mxstep=5000  # max passi interni per intervallo
    )

    R = solution[:, 0]  # raggio nel tempo
    V = solution[:, 1]  # velocità nel tempo

    # Protezione: forza R >= soglia di collasso
    R = np.maximum(R, BubbleParameters.R_COLLAPSE_THRESHOLD)

    # Calcola pressione irradiata alla sorgente
    p_source = compute_radiated_pressure(R, V, t, R0)

    # Controlla se il collasso è avvenuto
    collapsed = np.any(R <= BubbleParameters.R_COLLAPSE_THRESHOLD * 2)

    return {
        't': t,
        'R': R,
        'V': V,
        'p_source': p_source,
        'R0': R0,
        'P_inf': P_inf,
        'P_gas0': P_gas0,
        'collapsed': collapsed
    }


# =============================================================================
# PRESSIONE IRRADIATA DALLA BOLLA
# =============================================================================

def compute_radiated_pressure(R, V, t, R0):
    """
    Calcola la pressione irradiata dalla bolla in collasso.

    La pressione irradiata è proporzionale alla seconda derivata
    del volume della bolla nel tempo. In forma semplificata:

        p_source(t) ≈ ρ/r · (R·R'' + 2·R'²) · R²

    dove r è la distanza dalla bolla (qui normalizzata a R0).

    Args:
        R  : array del raggio nel tempo [m]
        V  : array della velocità dR/dt [m/s]
        t  : array dei tempi [s]
        R0 : raggio iniziale [m]

    Returns:
        np.ndarray: pressione irradiata [Pa], stessa lunghezza di R
    """
    rho = WaterProperties.DENSITY

    # Derivata seconda di R calcolata numericamente da V
    # (V è già dR/dt, quindi dV/dt = d²R/dt²)
    dt = np.diff(t)
    dV_dt = np.diff(V) / dt
    # Estendi all'ultimo punto per mantenere la lunghezza
    dV_dt = np.append(dV_dt, dV_dt[-1])

    # Pressione irradiata alla distanza R0 dalla bolla
    # p = ρ · R² · (R·R'' + 2·V²) / R0
    p_source = rho * R ** 2 * (R * dV_dt + 2.0 * V ** 2) / R0

    return p_source