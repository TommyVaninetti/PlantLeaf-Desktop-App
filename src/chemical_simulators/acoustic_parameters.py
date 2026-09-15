"""
acoustic_parameters.py
=======================
Parametri fisici e costanti del modello di cavitazione xilematica.

Fonti scientifiche:
    - Khait et al. (2023): modello acustico per emissioni ultrasoniche nelle piante
    - Tyree & Sperry (1988): teoria della cavitazione nello xilema
    - Brennen (1995): Cavitation and Bubble Dynamics, Oxford University Press

Utilizzo:
    from acoustic_parameters import WaterProperties, BubbleParameters, MicrophoneResponse, PlantLeafConfig
"""

import sys
from pathlib import Path

import numpy as np

# Constants come from click_pipeline_v5 (single source of truth), loaded through
# hybrid.pipeline_loader rather than `from core.click_pipeline_v5 import ...`:
# importing the `core` package runs core/__init__.py, which pulls in the Qt
# windows and makes this module unusable from a headless analysis script.
_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from hybrid.pipeline_loader import load_pipeline  # noqa: E402

_cp = load_pipeline()
_PL_FS         = _cp.FS
_PL_FFT_SIZE   = _cp.FFT_SIZE
_PL_BIN_START  = _cp.BIN_START_HZ
_PL_BIN_END    = _cp.BIN_END_HZ
_MIC_FREQ_HZ   = _cp._MIC_FREQ_HZ
_MIC_RESP_DB   = _cp._MIC_RESP_DB


# =============================================================================
# COSTANTI FISICHE DELL'ACQUA (a 20°C)
# =============================================================================

class WaterProperties:
    """
    Proprietà fisiche dell'acqua liquida a 20°C e pressione atmosferica standard.
    Tutte le unità sono SI.

    Fonte: Brennen (1995), Tabella A.1
    """

    # Densità [kg/m³]
    DENSITY = 1000.0  # ρ

    # Viscosità dinamica [Pa·s]
    VISCOSITY = 1.0e-3  # µ

    # Tensione superficiale acqua-aria [N/m]
    SURFACE_TENSION = 0.072  # σ

    # Pressione di vapore saturo a 20°C [Pa]
    VAPOR_PRESSURE = 2300.0  # P_v

    # Modulo di compressibilità bulk [Pa]
    BULK_MODULUS = 2.2e9  # K

    # Velocità del suono in acqua [m/s]
    SPEED_OF_SOUND = 1480.0  # c_water

    # Esponente adiabatico del gas interno alla bolla (aria)
    GAMMA_GAS = 1.4  # γ — usato nella legge adiabatica P·V^γ = costante


# =============================================================================
# PARAMETRI GEOMETRICI E FISICI DELLA BOLLA
# =============================================================================

class BubbleParameters:
    """
    Parametri geometrici della bolla di cavitazione nello xilema.

    Il raggio iniziale R0 dipende dalla specie vegetale e dal diametro
    del vaso xilematico. Il range biologico tipico è 20–100 µm.

    Fonte: Tyree & Sperry (1988), Khait et al. (2023)
    """

    # Raggio iniziale della bolla — valore default [m]
    # Corrisponde a 50 µm, valore centrale del range biologico
    R0_DEFAULT = 50.0e-6  # R0 in metri

    # Range biologico plausibile per R0 [m]
    R0_MIN = 20.0e-6   # 20 µm — vasi stretti, specie con tensione idrica elevata
    R0_MAX = 100.0e-6  # 100 µm — vasi larghi, piante ben idratate

    # Pressione interna iniziale del gas nella bolla [Pa]
    # Calcolata dalla condizione di equilibrio iniziale:
    # P_gas0 = P_atm + 2σ/R0 - P_∞
    # (calcolata dinamicamente in rayleigh_plesset.py in base a R0 e P_∞)
    # Qui salviamo solo la pressione atmosferica di riferimento
    P_ATM = 101325.0  # Pa

    # Soglia minima del raggio per considerare il collasso completato [m]
    R_COLLAPSE_THRESHOLD = 1.0e-7  # 0.1 µm — collasso praticamente completo

class BubbleResonance:
    """
    Parametri del modello di risonanza di bolla smorzata da irraggiamento
    acustico (Minnaert + smorzamento radiativo/viscoso).

    A differenza del precedente modello con guscio elastico arbitrario,
    qui frequenza e tempo di decadimento del click dipendono SOLO da R0,
    tramite formule fisiche standard (letteratura: Minnaert 1933; Brennen
    1995; validato numericamente per il caso xylematico).
    """

    # Frazione di perturbazione iniziale del raggio rispetto a R0
    # (ampiezza dell'oscillazione, non influenza frequenza o tau)
    PERTURBATION_FRACTION = 0.10


# =============================================================================
# TENSIONE IDRICA DELLO XILEMA
# =============================================================================

class XylemPressure:
    """
    Tensione idrica dello xilema P∞ — il parametro biologico chiave.

    P∞ è la pressione del liquido lontano dalla bolla (bulk pressure).
    Valori negativi indicano tensione (acqua sotto pressione negativa).

    Questo è il parametro che varia con lo stress idrico della pianta:
        - Pianta ben idratata:   P∞ ≈ -0.3 MPa
        - Stress idrico moderato: P∞ ≈ -0.8 MPa
        - Stress idrico severo:  P∞ ≈ -1.5 MPa

    Fonte: Tyree & Sperry (1988)
    """

    # Valore default — pianta ben idratata [Pa]
    P_INF_DEFAULT = -0.3e6  # -0.3 MPa in Pa

    # Range per l'analisi sistematica (Step 6) [Pa]
    P_INF_MIN = -1.5e6  # -1.5 MPa — stress severo
    P_INF_MAX = -0.3e6  # -0.3 MPa — pianta idratata

    # Numero di punti per la griglia di analisi (Step 6)
    P_INF_GRID_POINTS = 20

    @staticmethod
    def to_mpa(p_pa):
        """Converte pressione da Pa a MPa per la visualizzazione."""
        return p_pa / 1.0e6

    @staticmethod
    def to_pa(p_mpa):
        """Converte pressione da MPa a Pa per i calcoli."""
        return p_mpa * 1.0e6

    @staticmethod
    def get_stress_label(p_pa):
        """
        Restituisce una descrizione qualitativa dello stress idrico
        in base alla tensione xilematica.

        Args:
            p_pa (float): Tensione idrica in Pa (valore negativo)

        Returns:
            str: Etichetta descrittiva dello stress
        """
        p_mpa = abs(p_pa) / 1.0e6
        if p_mpa < 0.5:
            return "Well hydrated"
        elif p_mpa < 0.9:
            return "Mild water stress"
        elif p_mpa < 1.2:
            return "Moderate water stress"
        else:
            return "Severe water stress"


# =============================================================================
# RISPOSTA IN FREQUENZA DEL MICROFONO SPU0410LR5H
# =============================================================================

class MicrophoneResponse:
    """
    Risposta in frequenza del microfono MEMS SPU0410LR5H (Knowles).

    I valori di frequenza e risposta sono importati da core.click_pipeline_v5
    (_MIC_FREQ_HZ, _MIC_RESP_DB) per garantire che la simulazione usi
    esattamente la stessa curva di risposta usata dal pipeline di rilevamento.

    Fonte: Knowles SPU0410LR5H datasheet, Fig. 3
    """

    # Frequenze di campionamento [Hz] — identiche a click_pipeline_v5._MIC_FREQ_HZ
    FREQ_POINTS_HZ = _MIC_FREQ_HZ

    # Risposta corrispondente [dB] — identica a click_pipeline_v5._MIC_RESP_DB
    RESPONSE_DB = _MIC_RESP_DB

    # Frequenza di risonanza principale [Hz]
    F_RESONANCE = 25000.0  # 25 kHz

    # Picco di risonanza [dB]
    RESONANCE_PEAK_DB = 10.5

    @staticmethod
    def get_response_linear(freq_hz):
        """
        Interpola la risposta del microfono a una frequenza arbitraria.

        Args:
            freq_hz (float or np.ndarray): Frequenza/e in Hz

        Returns:
            np.ndarray: Risposta lineare (non in dB) — fattore moltiplicativo
        """
        response_db = np.interp(
            freq_hz,
            MicrophoneResponse.FREQ_POINTS_HZ,
            MicrophoneResponse.RESPONSE_DB
        )
        return 10.0 ** (response_db / 20.0)


# =============================================================================
# PARAMETRI DI ACQUISIZIONE PLANTLEAF
# =============================================================================

class PlantLeafConfig:
    """
    Configurazione del sistema di acquisizione PlantLeaf.

    I valori numerici sono importati da core.click_pipeline_v5 (single source of
    truth) per garantire coerenza con il firmware e con il pipeline di rilevamento.
    """

    # Frequenza di campionamento [Hz]
    FS = _PL_FS            # 200 kHz

    # Dimensione FFT [campioni]
    FFT_SIZE = _PL_FFT_SIZE  # 512

    # Range di frequenza analizzato [Hz]
    FREQ_MIN = _PL_BIN_START  # 20 kHz
    FREQ_MAX = _PL_BIN_END    # 80 kHz

    # Durata di un singolo frame FFT [s]
    FFT_DURATION = _PL_FFT_SIZE / _PL_FS  # = 2.56 ms

    # Passo in frequenza per bin FFT [Hz/bin]
    BIN_FREQ = _PL_FS / _PL_FFT_SIZE  # = 390.625 Hz/bin

    @staticmethod
    def get_freq_axis():
        """
        Restituisce l'asse delle frequenze del sistema PlantLeaf.
        Identico a self.data_x in MainWindowAudio.

        Returns:
            np.ndarray: Array di frequenze in Hz, da FREQ_MIN a FREQ_MAX
        """
        bin_freq = PlantLeafConfig.FS / PlantLeafConfig.FFT_SIZE
        bin_start = int(PlantLeafConfig.FREQ_MIN / bin_freq)
        bin_end = int(PlantLeafConfig.FREQ_MAX / bin_freq)
        # True STFT bin-center frequencies f[k] = k * (FS/FFT_SIZE), k = bin_start..bin_end.
        # (linspace(FREQ_MIN, FREQ_MAX, num_bins) would mislabel the top bin by ~1 bin.)
        return np.arange(bin_start, bin_end + 1) * bin_freq


# =============================================================================
# PARAMETRI DI PROPAGAZIONE ACUSTICA NEL TESSUTO VEGETALE
# =============================================================================

class PropagationParameters:
    """
    Parametri per la modellazione della propagazione acustica
    attraverso il tessuto vegetale (parenchima fogliare).

    Fonte: Khait et al. (2023), sezione Materials and Methods
    """

    # Attenuazione dipendente dalla frequenza [dB/cm/kHz]
    ATTENUATION_COEFF = 0.5  # α

    # Distanza bolla–microfono di default [m]
    DISTANCE_DEFAULT = 0.01  # 1 cm

    # Velocità del suono nel tessuto vegetale [m/s]
    SPEED_OF_SOUND_TISSUE = 1200.0  # c_tissue

    @staticmethod
    def attenuation_factor(freq_hz, distance_m):
        """
        Calcola il fattore di attenuazione totale (lineare) per una data
        frequenza e distanza di propagazione.

        Args:
            freq_hz (float or np.ndarray): Frequenza in Hz
            distance_m (float): Distanza bolla-microfono in metri

        Returns:
            float or np.ndarray: Fattore di attenuazione lineare (0–1)
        """
        distance_cm = distance_m * 100.0
        freq_khz = np.asarray(freq_hz) / 1000.0

        attenuation_db = PropagationParameters.ATTENUATION_COEFF * freq_khz * distance_cm
        attenuation_linear = 10.0 ** (-attenuation_db / 20.0)
        geometric_decay = 0.01 / max(distance_m, 1e-6)

        return attenuation_linear * geometric_decay


# =============================================================================
# PARAMETRI PER IL FITTING FENOMENOLOGICO (Step 5)
# =============================================================================

class DampedSineFitParams:
    """
    Bounds e valori iniziali per il fitting della sinusoide smorzata:
        s(t) = A · exp(-t/τ) · sin(2π·f₀·t + φ)
    """

    BOUNDS_LOWER = [0.0,    0.02e-3,  20000.0, -np.pi]
    BOUNDS_UPPER = [10.0,   2.0e-3,   80000.0,  np.pi]

    F0_INIT = 25000.0
    PHI_INIT = 0.0

# =============================================================================
# GAS NELLA BOLLA — per lo smorzamento termico (aggiunto in Step B)
# =============================================================================

class GasProperties:
    """
    Proprietà termiche dell'aria a 20 °C e 1 atm, necessarie allo smorzamento
    termico della bolla (teoria lineare di Prosperetti 1977).

    La diffusività termica D = k / (ρ_g · c_p) scala come 1/p: alla pressione
    interna della bolla p_g0 = p0 + 2σ/R0 va riscalata (vedi rayleigh_plesset).

    Fonte: valori standard dell'aria secca a 20 °C (k = 0.0257 W/m/K,
    ρ_g = 1.204 kg/m³, c_p = 1005 J/kg/K).
    """

    THERMAL_CONDUCTIVITY = 0.0257   # k   [W/(m·K)]
    DENSITY_1ATM = 1.204            # ρ_g [kg/m³] a 1 atm
    SPECIFIC_HEAT_CP = 1005.0       # c_p [J/(kg·K)]

    @staticmethod
    def thermal_diffusivity(p_gas):
        """Diffusività termica dell'aria [m²/s] alla pressione p_gas [Pa]."""
        d_1atm = GasProperties.THERMAL_CONDUCTIVITY / (
            GasProperties.DENSITY_1ATM * GasProperties.SPECIFIC_HEAT_CP)
        return d_1atm * BubbleParameters.P_ATM / p_gas


# =============================================================================
# VASO XILEMATICO COME RISUONATORE — Dutta et al. 2022 (aggiunto in Step B)
# =============================================================================

class VesselParameters:
    """
    Parametri del modello "vaso come canna d'organo" di Dutta et al. (2022),
    Research 2022:9790438, doi:10.34133/2022/9790438 — valori COME RIPORTATI
    nel paper, da verificare per le succulente (vedi report §7):

        v_l = ~1482 m/s   velocità del suono nell'acqua a 20 °C
        ρ_l = 996 kg/m³   densità della linfa (acqua)
        η_l = 8.9e-4 Pa·s viscosità dinamica
        h   = ~1 µm       spessore della parete (crio-SEM)
        E   = 0.2 ± 0.1 GPa  modulo di Young, fusti freschi idratati
        m   = 1           modo fondamentale
    """

    SOUND_SPEED_LIQUID = 1482.0     # v_l [m/s]
    DENSITY_LIQUID = 996.0          # ρ_l [kg/m³]
    VISCOSITY_LIQUID = 8.9e-4       # η_l [Pa·s]
    WALL_THICKNESS = 1.0e-6         # h   [m]
    YOUNG_MODULUS = 0.2e9           # E   [Pa]
    MODE_ORDER = 1                  # m

    # Valori pubblicati usati come test di validazione (Tabella 1 e testo).
    # Hydrangea quercifolia: raggio acustico 11.2 ± 0.5 µm, f 51.2 ± 1.0 kHz,
    # lunghezza acustica dell'elemento di vaso 0.99 ± 0.08 mm.
    REFERENCE_HYDRANGEA = {'R_um': 11.2, 'f_khz': 51.2, 'L_mm': 0.99, 'L_err_mm': 0.08}
