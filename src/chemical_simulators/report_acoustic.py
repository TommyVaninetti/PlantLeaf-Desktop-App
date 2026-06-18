"""
report_acoustic.py
==================
Genera il report PDF scientifico per il simulatore acustico PlantLeaf.

Contenuto:
    - Grafico 1: dinamica della bolla R(t)
    - Grafico 2: pressione irradiata p_source(t)
    - Grafico 3: segnale simulato vs click reale PlantLeaf
    - Grafico 4: distribuzione τ simulati vs τ misurati
    - Tabella:   parametri fittati con incertezze 1σ

Dipendenze:
    pip install matplotlib reportlab
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # backend non interattivo per generazione PDF
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import os
from datetime import datetime

from acoustic_parameters import XylemPressure


# =============================================================================
# FUNZIONE PRINCIPALE
# =============================================================================

def generate_report(simulation_result, stress_result=None,
                    fit_result=None, tau_measured=None,
                    output_path=None):
    """
    Genera il report PDF completo della simulazione acustica.

    Args:
        simulation_result : output di run_simulation() — risultati simulazione
        stress_result     : output di run_stress_analysis() — analisi sistematica
                            (opzionale, se None il Grafico 4 viene saltato)
        fit_result        : output di fit_damped_sine() — parametri del fit
                            (opzionale)
        tau_measured      : array τ misurati dai click reali [s]
                            (opzionale, per il confronto)
        output_path       : percorso del file PDF di output.
                            Default: 'report_acoustic_YYYYMMDD_HHMMSS.pdf'

    Returns:
        str: percorso del file PDF generato
    """
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'report_acoustic_{timestamp}.pdf'

    with PdfPages(output_path) as pdf:

        # --- Pagina 1: dinamica bolla + pressione sorgente ---
        fig1 = _plot_bubble_dynamics(simulation_result)
        pdf.savefig(fig1, bbox_inches='tight')
        plt.close(fig1)

        # --- Pagina 2: segnale simulato vs reale ---
        fig2 = _plot_signal_comparison(simulation_result, tau_measured)
        pdf.savefig(fig2, bbox_inches='tight')
        plt.close(fig2)

        # --- Pagina 3: distribuzione τ (solo se stress_result disponibile) ---
        if stress_result is not None and tau_measured is not None:
            fig3 = _plot_tau_distribution(stress_result, tau_measured)
            pdf.savefig(fig3, bbox_inches='tight')
            plt.close(fig3)

        # --- Pagina 4: tabella parametri ---
        fig4 = _plot_parameters_table(simulation_result, fit_result)
        pdf.savefig(fig4, bbox_inches='tight')
        plt.close(fig4)

        # Metadati PDF
        d = pdf.infodict()
        d['Title'] = 'PlantLeaf Acoustic Simulator — Report'
        d['Author'] = 'PlantLeaf v1.0'
        d['Subject'] = 'Cavitation bubble dynamics and acoustic click simulation'
        d['CreationDate'] = datetime.now()

    print(f'✅ Report generato: {output_path}')
    return output_path


# =============================================================================
# GRAFICO 1 + 2: DINAMICA BOLLA E PRESSIONE SORGENTE
# =============================================================================

def _plot_bubble_dynamics(simulation_result):
    """Grafico 1 e 2: R(t) e p_source(t)"""

    bubble = simulation_result['bubble']
    t = bubble['t']
    R = bubble['R']
    p_source = bubble['p_source']
    R0 = bubble['R0']
    P_inf = bubble['P_inf']

    # Scala temporale in microsecondi
    t_us = t * 1e6

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(
        f'Bubble Dynamics — R₀ = {R0*1e6:.0f} µm, '
        f'P∞ = {XylemPressure.to_mpa(P_inf):.2f} MPa '
        f'({XylemPressure.get_stress_label(P_inf)})',
        fontsize=13, fontweight='bold', y=0.98
    )

    # --- Grafico 1: R(t) ---
    ax1 = axes[0]
    ax1.plot(t_us, R * 1e6, color='#2196F3', linewidth=1.5, label='R(t)')
    ax1.axhline(y=R0 * 1e6, color='gray', linestyle='--',
                linewidth=1.0, alpha=0.6, label=f'R₀ = {R0*1e6:.0f} µm')
    ax1.set_xlabel('Time [µs]', fontsize=11)
    ax1.set_ylabel('Bubble radius [µm]', fontsize=11)
    ax1.set_title('Bubble radius R(t) during collapse', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(t_us[0], t_us[-1])

    # --- Grafico 2: p_source(t) ---
    ax2 = axes[1]
    ax2.plot(t_us, p_source / 1e3, color='#E53935', linewidth=1.5,
             label='p_source(t)')
    ax2.set_xlabel('Time [µs]', fontsize=11)
    ax2.set_ylabel('Radiated pressure [kPa]', fontsize=11)
    ax2.set_title('Radiated pressure at source p_source(t)', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(t_us[0], t_us[-1])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# =============================================================================
# GRAFICO 3: SEGNALE SIMULATO VS CLICK REALE
# =============================================================================

def _plot_signal_comparison(simulation_result, tau_measured=None):
    """Grafico 3: segnale simulato vs click reale nel tempo e in frequenza"""

    bubble      = simulation_result['bubble']
    propagation = simulation_result['propagation']
    plantleaf   = simulation_result['plantleaf']
    diagnostics = simulation_result['diagnostics']

    t        = bubble['t']
    signal   = propagation['signal']
    freq_sim = plantleaf['freq']
    spec_sim = plantleaf['spectrum']

    t_us = t * 1e6

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Simulated click vs PlantLeaf measurement',
                 fontsize=13, fontweight='bold')

    # --- Segnale nel tempo ---
    ax1 = axes[0]
    ax1.plot(t_us, signal, color='#2196F3', linewidth=1.2,
             label='Simulated click', alpha=0.9)
    ax1.set_xlabel('Time [µs]', fontsize=11)
    ax1.set_ylabel('Pressure [Pa]', fontsize=11)
    ax1.set_title('Time domain', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(t_us[0], t_us[-1])

    # Annotazione τ
    tau = diagnostics.get('tau', None)
    if tau is not None:
        ax1.annotate(
            f'τ = {tau*1000:.3f} ms',
            xy=(0.05, 0.92), xycoords='axes fraction',
            fontsize=10, color='#2196F3',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )

    ax1.legend(fontsize=10)

    # --- Spettro in frequenza ---
    ax2 = axes[1]
    freq_khz = freq_sim / 1000.0
    ax2.plot(freq_khz, spec_sim, color='#2196F3', linewidth=1.5,
             label='Simulated spectrum', alpha=0.9)
    ax2.axvline(x=25.0, color='gray', linestyle='--',
                linewidth=1.0, alpha=0.5, label='Mic resonance (25 kHz)')
    ax2.set_xlabel('Frequency [kHz]', fontsize=11)
    ax2.set_ylabel('Amplitude [Pa]', fontsize=11)
    ax2.set_title('Frequency domain (20–80 kHz)', fontsize=11)
    ax2.set_xlim(20, 80)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# =============================================================================
# GRAFICO 4: DISTRIBUZIONE τ SIMULATI VS MISURATI
# =============================================================================

def _plot_tau_distribution(stress_result, tau_measured):
    """Grafico 4: istogrammi sovrapposti τ simulati vs τ misurati"""

    tau_flat    = stress_result['tau_flat']
    tau_sim_ms  = tau_flat * 1000.0
    tau_meas_ms = np.array(tau_measured) * 1000.0

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle('τ distribution: simulated vs measured',
                 fontsize=13, fontweight='bold')

    # Bins comuni
    all_tau = np.concatenate([tau_sim_ms, tau_meas_ms])
    bins = np.linspace(np.min(all_tau), np.max(all_tau), 30)

    ax.hist(tau_sim_ms, bins=bins, alpha=0.5, color='#2196F3',
            label=f'Simulated (n={len(tau_sim_ms)})', density=True)
    ax.hist(tau_meas_ms, bins=bins, alpha=0.5, color='#E53935',
            label=f'Measured (n={len(tau_meas_ms)})', density=True)

    # Linee medie
    ax.axvline(x=np.mean(tau_sim_ms), color='#2196F3', linestyle='--',
               linewidth=1.5, label=f'Sim mean = {np.mean(tau_sim_ms):.3f} ms')
    ax.axvline(x=np.mean(tau_meas_ms), color='#E53935', linestyle='--',
               linewidth=1.5, label=f'Meas mean = {np.mean(tau_meas_ms):.3f} ms')

    ax.set_xlabel('τ [ms]', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# =============================================================================
# TABELLA PARAMETRI
# =============================================================================

def _plot_parameters_table(simulation_result, fit_result=None):
    """Tabella: parametri fittati con incertezze e semplificazioni dichiarate"""

    bubble      = simulation_result['bubble']
    diagnostics = simulation_result['diagnostics']

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle('Model parameters and declared simplifications',
                 fontsize=13, fontweight='bold')

    # --- Tabella 1: parametri simulazione ---
    ax1 = axes[0]
    ax1.axis('off')

    rows_sim = [
        ['Parameter', 'Value', 'Unit', 'Source'],
        ['R₀', f"{bubble['R0']*1e6:.1f}", 'µm', 'Input'],
        ['P∞', f"{XylemPressure.to_mpa(bubble['P_inf']):.2f}", 'MPa', 'Input'],
        ['P_gas0', f"{bubble['P_gas0']/1e3:.1f}", 'kPa', 'Equilibrium condition'],
        ['τ (Hilbert)', f"{diagnostics['tau']*1000:.3f}" if diagnostics['tau'] else 'N/A', 'ms', 'PlantLeaf v4.0'],
        ['SPR', f"{diagnostics['SPR']:.2f}" if diagnostics['SPR'] else 'N/A', '-', 'PlantLeaf v4.0'],
        ['Asymmetry', f"{diagnostics['asymmetry']:.3f}" if diagnostics['asymmetry'] else 'N/A', '-', 'PlantLeaf v4.0'],
        ['R_spectral', f"{diagnostics['R_spectral']:.3f}" if diagnostics['R_spectral'] else 'N/A', '-', 'PlantLeaf v4.0'],
    ]

    # Aggiungi parametri fit se disponibili
    if fit_result is not None and fit_result.get('success'):
        rows_sim += [
            ['τ (fit)', f"{fit_result['tau']*1000:.3f} ± {fit_result['tau_err']*1000:.3f}", 'ms', 'curve_fit'],
            ['f₀ (fit)', f"{fit_result['f0']/1000:.1f} ± {fit_result['f0_err']/1000:.1f}", 'kHz', 'curve_fit'],
            ['A (fit)', f"{fit_result['A']:.4f} ± {fit_result['A_err']:.4f}", 'Pa', 'curve_fit'],
        ]

    table1 = ax1.table(
        cellText=rows_sim[1:],
        colLabels=rows_sim[0],
        cellLoc='center',
        loc='center'
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(9)
    table1.scale(1, 1.4)
    ax1.set_title('Simulation parameters', fontsize=11, pad=10)

    # --- Tabella 2: semplificazioni dichiarate ---
    ax2 = axes[1]
    ax2.axis('off')

    rows_simp = [
        ['Simplification', 'Impact'],
        ['Spherical bubble geometry', 'Low — valid for R << vessel diameter'],
        ['Adiabatic gas law (γ=1.4)', 'Low — collapse is fast (<10 µs)'],
        ['Uniform xylem pressure P∞', 'Medium — real P∞ varies along vessel'],
        ['Homogeneous tissue attenuation', 'Medium — real tissue is heterogeneous'],
        ['Microphone response from datasheet', 'Low — individual unit variation <2 dB'],
        ['No bubble-bubble interaction', 'Low — single embolism event assumed'],
    ]

    table2 = ax2.table(
        cellText=rows_simp[1:],
        colLabels=rows_simp[0],
        cellLoc='center',
        loc='center'
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1, 1.4)
    ax2.set_title('Declared simplifications and expected impact', fontsize=11, pad=10)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig