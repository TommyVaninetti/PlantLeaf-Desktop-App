"""
stress_analysis.py
==================
Analisi sistematica della relazione tra stress idrico e parametri del click.

Lancia il simulatore su una griglia di valori P∞ × R0 e confronta
la distribuzione di τ simulati con i τ misurati dai click reali PlantLeaf.

Fonti:
    - Tyree & Sperry (1988): relazione stress idrico - cavitazione
    - Khait et al. (2023): validazione statistica del modello
"""

import numpy as np
from acoustic_parameters import (
    BubbleParameters,
    XylemPressure
)
from run_acoustic_simulation import run_simulation # type: ignore


# =============================================================================
# GRIGLIA DI SIMULAZIONE
# =============================================================================

def build_simulation_grid(n_pressure=20, n_radius=10):
    """
    Costruisce la griglia di valori P∞ × R0 per l'analisi sistematica.

    Args:
        n_pressure : numero di valori di P∞ (default 20)
        n_radius   : numero di valori di R0 (default 10)

    Returns:
        dict con le chiavi:
            'P_inf_grid' : array valori P∞ [Pa]
            'R0_grid'    : array valori R0 [m]
            'n_total'    : numero totale di simulazioni
    """
    P_inf_grid = np.linspace(
        XylemPressure.P_INF_MIN,
        XylemPressure.P_INF_MAX,
        n_pressure
    )

    R0_grid = np.linspace(
        BubbleParameters.R0_MIN,
        BubbleParameters.R0_MAX,
        n_radius
    )

    return {
        'P_inf_grid': P_inf_grid,
        'R0_grid':    R0_grid,
        'n_total':    n_pressure * n_radius
    }


# =============================================================================
# ESECUZIONE GRIGLIA
# =============================================================================

def run_stress_analysis(n_pressure=20, n_radius=10, progress_callback=None):
    """
    Esegue il simulatore su tutta la griglia P∞ × R0 ed estrae τ
    da ciascun click simulato.

    Args:
        n_pressure        : numero di valori di P∞
        n_radius          : numero di valori di R0
        progress_callback : funzione opzionale chiamata ad ogni simulazione
                            con argomento (i, n_total) per aggiornare
                            una barra di progresso nella UI

    Returns:
        dict con le chiavi:
            'P_inf_grid'  : array valori P∞ usati [Pa]
            'R0_grid'     : array valori R0 usati [m]
            'tau_matrix'  : matrice τ [s], shape (n_pressure, n_radius)
                            None dove la simulazione non è conversa
            'tau_flat'    : array τ appiattito (tutti i valori validi)
            'P_inf_flat'  : array P∞ corrispondenti ai τ validi [Pa]
            'R0_flat'     : array R0 corrispondenti ai τ validi [m]
            'n_success'   : numero di simulazioni riuscite
            'n_failed'    : numero di simulazioni fallite
    """
    grid = build_simulation_grid(n_pressure, n_radius)
    P_inf_grid = grid['P_inf_grid']
    R0_grid    = grid['R0_grid']
    n_total    = grid['n_total']

    # Matrice risultati
    tau_matrix = np.full((n_pressure, n_radius), np.nan)

    n_success = 0
    n_failed  = 0
    counter   = 0

    for i, P_inf in enumerate(P_inf_grid):
        for j, R0 in enumerate(R0_grid):
            counter += 1

            # Aggiorna progresso se fornito callback
            if progress_callback is not None:
                progress_callback(counter, n_total)

            try:
                result = run_simulation(R0=R0, P_inf=P_inf)
                tau = result['diagnostics'].get('tau', None)

                if tau is not None and tau > 0:
                    tau_matrix[i, j] = tau
                    n_success += 1
                else:
                    n_failed += 1

            except Exception as e:
                print(f"⚠️ Simulazione fallita P∞={P_inf/1e6:.2f} MPa, "
                      f"R0={R0*1e6:.0f} µm: {e}")
                n_failed += 1

    # Appiattisci rimuovendo NaN
    valid_mask   = ~np.isnan(tau_matrix)
    tau_flat     = tau_matrix[valid_mask]

    # Costruisci array P∞ e R0 corrispondenti
    P_inf_2d = np.tile(P_inf_grid[:, np.newaxis], (1, n_radius))
    R0_2d    = np.tile(R0_grid[np.newaxis, :],    (n_pressure, 1))
    P_inf_flat = P_inf_2d[valid_mask]
    R0_flat    = R0_2d[valid_mask]

    print(f"✅ Analisi completata: {n_success} successi, {n_failed} fallimenti")

    return {
        'P_inf_grid':  P_inf_grid,
        'R0_grid':     R0_grid,
        'tau_matrix':  tau_matrix,
        'tau_flat':    tau_flat,
        'P_inf_flat':  P_inf_flat,
        'R0_flat':     R0_flat,
        'n_success':   n_success,
        'n_failed':    n_failed
    }


# =============================================================================
# CONFRONTO CON DATI REALI
# =============================================================================

def compare_with_real_data(tau_simulated, tau_measured):
    """
    Confronta la distribuzione di τ simulati con i τ misurati
    dai click reali di PlantLeaf.

    Args:
        tau_simulated : array τ simulati [s]
        tau_measured  : array τ misurati dai click reali [s]

    Returns:
        dict con le chiavi:
            'tau_sim_ms'    : τ simulati in millisecondi
            'tau_meas_ms'   : τ misurati in millisecondi
            'sim_mean_ms'   : media τ simulati [ms]
            'meas_mean_ms'  : media τ misurati [ms]
            'sim_std_ms'    : deviazione standard τ simulati [ms]
            'meas_std_ms'   : deviazione standard τ misurati [ms]
            'overlap'       : indice di sovrapposizione delle distribuzioni (0-1)
    """
    tau_sim_ms  = np.array(tau_simulated) * 1000.0
    tau_meas_ms = np.array(tau_measured)  * 1000.0

    # Statistiche descrittive
    sim_mean  = np.mean(tau_sim_ms)
    sim_std   = np.std(tau_sim_ms)
    meas_mean = np.mean(tau_meas_ms)
    meas_std  = np.std(tau_meas_ms)

    # Indice di sovrapposizione: basato sulla distanza tra le medie
    # normalizzata per la deviazione standard media
    pooled_std = (sim_std + meas_std) / 2.0
    if pooled_std > 0:
        overlap = max(0.0, 1.0 - abs(sim_mean - meas_mean) / (2.0 * pooled_std))
    else:
        overlap = 1.0 if abs(sim_mean - meas_mean) < 1e-6 else 0.0

    return {
        'tau_sim_ms':   tau_sim_ms,
        'tau_meas_ms':  tau_meas_ms,
        'sim_mean_ms':  sim_mean,
        'meas_mean_ms': meas_mean,
        'sim_std_ms':   sim_std,
        'meas_std_ms':  meas_std,
        'overlap':      overlap
    }


# =============================================================================
# STIMA P∞ DAI DATI REALI
# =============================================================================

def estimate_p_inf(stress_results, tau_measured):
    """
    Stima il valore di P∞ che meglio corrisponde ai τ misurati dai
    click reali — stima indiretta della tensione idrica dello xilema.

    Per ogni valore di P∞ nella griglia calcola la media dei τ simulati
    (su tutti gli R0) e trova quello più vicino alla media dei τ misurati.

    Args:
        stress_results : output di run_stress_analysis()
        tau_measured   : array τ misurati dai click reali [s]

    Returns:
        dict con le chiavi:
            'P_inf_estimated_pa'  : P∞ stimato [Pa]
            'P_inf_estimated_mpa' : P∞ stimato [MPa]
            'stress_label'        : etichetta stress idrico
            'tau_sim_mean_ms'     : τ medio simulato al P∞ stimato [ms]
            'tau_meas_mean_ms'    : τ medio misurato [ms]
            'P_inf_curve_pa'      : array P∞ della curva [Pa]
            'tau_mean_curve_ms'   : array τ medio per ogni P∞ [ms]
    """
    from acoustic_parameters import XylemPressure

    tau_matrix  = stress_results['tau_matrix']
    P_inf_grid  = stress_results['P_inf_grid']

    # Media dei τ simulati per ogni valore di P∞ (su tutti gli R0)
    tau_mean_per_p = np.nanmean(tau_matrix, axis=1)  # shape (n_pressure,)

    # Media dei τ misurati
    tau_meas_mean = np.mean(tau_measured)

    # Trova P∞ la cui τ media simulata è più vicina a quella misurata
    diff = np.abs(tau_mean_per_p - tau_meas_mean)
    best_idx = np.nanargmin(diff)
    P_inf_best = P_inf_grid[best_idx]

    return {
        'P_inf_estimated_pa':  P_inf_best,
        'P_inf_estimated_mpa': XylemPressure.to_mpa(P_inf_best),
        'stress_label':        XylemPressure.get_stress_label(P_inf_best),
        'tau_sim_mean_ms':     tau_mean_per_p[best_idx] * 1000.0,
        'tau_meas_mean_ms':    tau_meas_mean * 1000.0,
        'P_inf_curve_pa':      P_inf_grid,
        'tau_mean_curve_ms':   tau_mean_per_p * 1000.0
    }