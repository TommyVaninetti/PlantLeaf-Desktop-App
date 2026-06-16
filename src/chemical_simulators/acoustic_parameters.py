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

import numpy as np


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

    Dati estratti dal datasheet ufficiale Knowles SPU0410LR5H-QB.
    La risposta è espressa in dB relativo al livello piatto (0 dB = risposta piatta).

    Il microfono ha un picco di risonanza a ~25 kHz (+10.5 dB) e una
    caduta progressiva oltre i 40 kHz, con un secondo picco più basso
    attorno a 60 kHz.

    Fonte: Knowles SPU0410LR5H datasheet, Fig. 3
    """

    # Frequenze di campionamento [Hz] — valori dal datasheet
    FREQ_POINTS_HZ = np.array([
        20000, 22000, 24000, 25000, 26000, 28000,
        30000, 32000, 35000, 40000, 45000, 50000,
        55000, 60000, 65000, 70000, 75000, 80000
    ], dtype=float)

    # Risposta corrispondente [dB] — relativa al livello piatto
    RESPONSE_DB = np.array([
        0.0,   1.5,   5.0,  10.5,  8.0,   4.0,
        1.0,  -1.0,  -2.5,  -5.0,  -4.0,  -2.0,
       -3.5,   0.5,  -4.0,  -7.0, -10.0, -14.0
    ], dtype=float)

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
        # Interpolazione lineare dei valori dB
        response_db = np.interp(
            freq_hz,
            MicrophoneResponse.FREQ_POINTS_HZ,
            MicrophoneResponse.RESPONSE_DB
        )
        # Conversione dB → lineare
        return 10.0 ** (response_db / 20.0)


# =============================================================================
# PARAMETRI DI ACQUISIZIONE PLANTLEAF
# =============================================================================

class PlantLeafConfig:
    """
    Configurazione del sistema di acquisizione PlantLeaf.

    Questi parametri devono essere coerenti con il firmware del microcontrollore
    e con i parametri usati in MainWindowAudio.
    """

    # Frequenza di campionamento [Hz]
    FS = 200000  # 200 kHz

    # Dimensione FFT [campioni]
    FFT_SIZE = 512

    # Range di frequenza analizzato [Hz]
    FREQ_MIN = 20000   # 20 kHz
    FREQ_MAX = 80000   # 80 kHz

    # Durata di un singolo frame FFT [s]
    FFT_DURATION = FFT_SIZE / FS  # = 2.56 ms

    # Passo in frequenza per bin FFT [Hz/bin]
    BIN_FREQ = FS / FFT_SIZE  # = 390.625 Hz/bin

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
        num_bins = bin_end - bin_start + 1
        return np.linspace(PlantLeafConfig.FREQ_MIN, PlantLeafConfig.FREQ_MAX, num_bins)


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