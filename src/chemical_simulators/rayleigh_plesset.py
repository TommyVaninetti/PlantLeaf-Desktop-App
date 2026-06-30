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

from acoustic_parameters import (
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

def simulate_bubble_collapse(R0=None, P_inf=None, t_max=None, n_points=5000, tau_target_ms=None):
    if R0 is None:
        R0 = BubbleParameters.R0_DEFAULT
    if P_inf is None:
        P_inf = XylemPressure.P_INF_DEFAULT

    rho = WaterProperties.DENSITY
    sigma = WaterProperties.SURFACE_TENSION

    # Se abbiamo un tau target, stimiamo R0 da esso
    if tau_target_ms is not None and tau_target_ms > 0:
        tau_s = tau_target_ms / 1000.0
        R0_estimated = tau_s / (0.915 * np.sqrt(rho / abs(P_inf)))
        R0_estimated = np.clip(R0_estimated, 1e-7, 500e-6)
        R0 = R0_estimated
        print(f"R0 stimato da tau={tau_target_ms:.3f} ms: {R0*1e6:.2f} µm")

    P_gas0 = BubbleParameters.P_ATM + 2.0 * sigma / R0

    if t_max is None:
        T_rayleigh = 0.915 * R0 * np.sqrt(rho / abs(P_inf))
        t_max = max(5.0 * T_rayleigh, 1e-4)

    t = np.linspace(0, t_max, n_points)
    y0 = [R0, 0.0]

    solution = odeint(
        rayleigh_plesset_ode,
        y0,
        t,
        args=(R0, P_inf, P_gas0),
        rtol=1e-8,
        atol=1e-12,
        mxstep=10000
    )

    R = solution[:, 0]
    V = solution[:, 1]
    R = np.maximum(R, BubbleParameters.R_COLLAPSE_THRESHOLD)

    p_source = compute_radiated_pressure(R, V, t, R0)
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
    del volume della bolla nel tempo (sorgente monopolare). In forma semplificata:

        p_source(t) ≈ ρ/r · (R²·R'' + 2·R·R'²)

    dove r è la distanza dalla bolla (qui normalizzata a R0).
    Unità risultanti: Pascal [Pa].

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

    # Pressione irradiata alla distanza R0 dalla bolla (sorgente monopolare):
    # p = ρ/r · (R²·R'' + 2·R·V²) = ρ/R0 · R · (R·R'' + 2·V²)   [Pa]
    p_source = rho * R * (R * dV_dt + 2.0 * V ** 2) / R0

    return p_source