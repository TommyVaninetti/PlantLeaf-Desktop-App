# Copyright (C) 2026 Tommaso Vaninetti
#
# This file is part of PlantLeaf.
#
# PlantLeaf is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# PlantLeaf is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with PlantLeaf. If not, see <https://www.gnu.org/licenses/>.

"""
PlantLeaf — Click Detection Pipeline v5.0
==========================================

Full pipeline for automatic ultrasonic click detection using an adaptive noise
estimator and a Support Vector Machine (SVM) classifier.

Architecture (four stages):
  Stage 1  – Adaptive energy threshold + run-length filter
  Stage 3  – Feature extraction (17 features, no hard thresholds)
  Stage 4  – SVM classification
  Stage 5  – Deduplication

Stage 2 (FFT hard-threshold filters from v4) has been removed. SPR and peak FFT
amplitude are now features fed to the SVM rather than hard gates, so the pipeline
learns their discriminative power from data rather than from hand-tuned thresholds.

Implementation notes:
  • All numpy operations — no scipy.linalg / BLAS / LAPACK calls — ensuring
    thread-safety on macOS (avoids segfaults from Apple Accelerate inside QThreads).
  • Every tuneable constant is defined once in the '── ALGORITHM CONSTANTS ──'
    section below and referenced by name throughout. Nothing is buried inline.

Reference: CLICK_DETECTION_ALGORITHM_v5.md (June 2026)
"""

import numpy as np
from typing import Optional


# ── ALGORITHM CONSTANTS ──────────────────────────────────────────────────────
# Every parameter that controls algorithm behaviour is defined here.
# Physically motivated values include a brief citation or derivation.
# Empirically chosen values are flagged with "→ verify experimentally".
#
# NOISE ESTIMATOR (§4)
W_NOISE        = 750    # Circular-buffer length for the noise estimator.
                        # At ~390 FPS: 750 frames ≈ 1.92 s.
                        # Long enough that a 30-50 frame burst (≤130 ms) adds
                        # <7% of entries and barely shifts the minimum.
                        # Short enough to track 1-5 s environmental transitions
                        # (wind gusts, passing vehicles).

BETA           = 1.5    # Martin (2001) minimum-statistics bias correction.
                        # The minimum of a finite buffer of noisy values
                        # systematically underestimates the true floor;
                        # β = 1.5 corrects this bias (validated in the original
                        # Martin 2001 paper on speech noise estimation).

ALPHA          = 2.0    # Burst exclusion multiplier: a frame is considered
                        # energetic (and excluded from buffers) when
                        #   E_i  >  ALPHA × Ê_floor(i-1).
                        # → verify experimentally on outdoor recordings.

WARM_UP_FRAMES = 375    # First W/2 frames: accept ALL into buffers unconditionally.
                        # Burst protection requires a reliable Ê_floor estimate,
                        # which we do not have before seeing enough silent frames.

# SYSTEM PARAMETERS (§3) — match firmware / hardware settings
FS             = 200_000   # Sampling rate [sps]
FFT_SIZE       = 512       # FFT window length [samples]
BIN_START_HZ   = 20_000    # Lower edge of analysis band [Hz]
BIN_END_HZ     = 80_000    # Upper edge of analysis band [Hz]

# Derived from the above — do not edit directly.
_BIN_FREQ      = FS / FFT_SIZE                  # Frequency per bin [Hz/bin]
_BIN_START     = int(BIN_START_HZ / _BIN_FREQ)  # First bin index (inclusive)
_BIN_END       = int(BIN_END_HZ   / _BIN_FREQ)  # Last  bin index (inclusive)
_K_BINS        = _BIN_END - _BIN_START + 1      # Number of analysis bins (= 154)

# STAGE 1 (§5)
MAX_RUN          = 3    # Maximum run length of consecutive above-threshold frames.
                        # Runs longer than this are discarded as sustained noise.
                        # A genuine cavitation click lasts ≤ 0.5 ms ≈ 1–2 frames;
                        # a run of 4+ frames (≥10.24 ms) is virtually always noise.
K_STAGE1_DEFAULT = 1.5  # Default Stage 1 threshold multiplier k.
                        # A frame is a candidate if E_i > k × Ê_floor.
                        # 1.5 casts a wide net for data collection.
                        # After SVM training a tighter value (e.g. 2.0–3.46) is used.

# iFFT RECONSTRUCTION (§7)
TUKEY_TAPER_FRACTION = 0.10  # Fraction of analysis-band bins used for each
                              # edge of the Tukey taper applied to the complex
                              # spectrum. taper_len = max(5, round(K × fraction)).
                              # Smoothly ramps the spectral edges to zero to reduce
                              # Gibbs ringing in the reconstructed time-domain signal.

# GIBBS SUPPRESSION (§7, Step 3)
GIBBS_CHECK_SAMPLES = 15   # Samples at each frame border examined for Gibbs energy.
                            # At 200 kHz: 15 samples = 75 µs.
GIBBS_FACTOR        = 2.5  # Both borders must exceed GIBBS_FACTOR × interior RMS to
                            # confirm the symmetric AND condition for Gibbs suppression.
                            # Using AND (not OR) preserves real clicks near frame edges.

# MIC NORMALIZATION — Knowles SPU0410LR5H-QB, 50 % conservative (§6.1)
# Datasheet frequency-response points used for piecewise-linear interpolation.
# 50 % conservative means we apply only half the datasheet correction, keeping
# the amplitude error within ±2.9 dB across the analysis band.
_MIC_FREQ_HZ       = np.array([20, 25, 30, 40, 50, 60, 70, 80], dtype=np.float64) * 1000.0
_MIC_RESP_DB       = np.array([ 8.0, 10.5,  6.0, -2.0, -6.0, -7.0, -6.0, -4.0], dtype=np.float64)
_MIC_NORM_FRACTION = 0.5   # 50 % conservative normalization factor
# ─────────────────────────────────────────────────────────────────────────────


# =============================================================================
# ADAPTIVE NOISE ESTIMATOR
# =============================================================================

class AdaptiveNoiseEstimatorV5:
    """
    Adaptive minimum-statistics noise floor estimator (§4 of the v5 spec).

    Maintains three circular buffers, each of length W_NOISE frames:

      B1       – FFT frame energy E_i [V²]
                 → used to compute Ê_floor for Stage 1 threshold
      B2_mean  – Hilbert-envelope mean of the reconstructed iFFT [V]
                 → used to compute noise_floor for Stage 3 features
      B2_std   – Hilbert-envelope std of the reconstructed iFFT [V]
                 → used to compute std_noise for Stage 3 features

    All three buffers share the same burst-protection gate (§4.4): if a frame's
    FFT energy exceeds ALPHA × current Ê_floor, the frame is considered energetic
    (a burst or click candidate) and is NOT written into any buffer. This prevents
    transient events from inflating — and thus corrupting — the noise estimate.

    Offline mode (the normal use case for .paudio files):
        Process frames in chronological order, calling update() once per frame.
        This simulates the real-time behaviour of the estimator exactly.

    Usage example:
        est = AdaptiveNoiseEstimatorV5()
        for i, frame in enumerate(frames):
            E_i        = compute_fft_energy(frame.fft_mags)
            env        = compute_hilbert_envelope(reconstruct_ifft(frame))
            env_mean_i = float(np.mean(env))
            env_std_i  = float(np.std(env))
            result     = est.update(E_i, env_mean_i, env_std_i)
            # result['E_hat_floor'] → use for Stage 1 check
            # result['noise_floor'] → use for Stage 3 feature computation
            # result['std_noise']   → use for Stage 3 feature computation
    """

    def __init__(self):
        # Three circular buffers initialised to zero.
        # Entries are only meaningful up to index self._fill - 1
        # (or the entire array once the buffer has wrapped).
        self._B1      = np.zeros(W_NOISE, dtype=np.float64)  # FFT energy [V²]
        self._B2_mean = np.zeros(W_NOISE, dtype=np.float64)  # env mean [V] (or more likely uV)
        self._B2_std  = np.zeros(W_NOISE, dtype=np.float64)  # env std  [V]

        self._buf_idx     = 0      # Next write position in the circular buffer
        self._fill        = 0      # Number of valid entries (0 → W_NOISE)
        self._frame_count = 0      # Total frames processed (including bursts)

        # Cached estimates — updated every call to update().
        # Initialised to 0; meaningless until at least one frame has been written.
        self._E_hat_floor = 0.0
        self._noise_floor = 0.0
        self._std_noise   = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, E_i: float, env_mean_i: float, env_std_i: float) -> dict:
        """
        Process one frame and return the updated noise estimates.

        This is the core per-frame update described in §4.8 of the spec.
        Call once per frame, in chronological order.

        Parameters
        ----------
        E_i : float
            FFT frame energy for the current frame [V²].
            Computed as (1/K) × Σ|A[k]|² over the K analysis bins.
        env_mean_i : float
            Mean of the Hilbert envelope of the reconstructed iFFT [V].
        env_std_i : float
            Standard deviation of the Hilbert envelope [V].

        Returns
        -------
        dict with keys:
            'E_hat_floor' : float  – Ê_floor(i) [V²]  for Stage 1 check
            'noise_floor' : float  – noise_floor(i) [V] for Stage 3 features
            'std_noise'   : float  – std_noise(i) [V]  for Stage 3 features
            'is_burst'    : bool   – True if this frame was excluded from buffers
            'in_warmup'   : bool   – True if still in the warm-up period
            'buffer_fill' : int    – number of valid entries currently in buffers
        """
        self._frame_count += 1
        in_warmup = (self._frame_count <= WARM_UP_FRAMES)

        # ── Burst protection (§4.4) ───────────────────────────────────────────
        # During warm-up: accept everything — we have no reliable estimate yet.
        # After warm-up: exclude energetic frames to protect the noise estimate.
        if in_warmup or self._fill == 0:
            is_burst = False
        else:
            is_burst = E_i > ALPHA * self._E_hat_floor

        # ── Buffer update ─────────────────────────────────────────────────────
        # Write this frame into the circular buffers only if it is not a burst.
        if not is_burst:
            idx = self._buf_idx % W_NOISE          # Write position (wraps at W)
            self._B1[idx]      = E_i
            self._B2_mean[idx] = env_mean_i
            self._B2_std[idx]  = env_std_i

            self._buf_idx += 1                     # Advance write pointer
            if self._fill < W_NOISE:               # Track how many entries are valid
                self._fill += 1

        # ── Estimate update ───────────────────────────────────────────────────
        # Compute new floor and std estimates from whatever is in the buffers.
        # We must read only valid entries (indices 0.._fill-1 when not yet full,
        # or all W entries once the buffer has wrapped).
        if self._fill > 0:
            valid_B1      = self._B1[:self._fill]
            valid_B2_mean = self._B2_mean[:self._fill]
            valid_B2_std  = self._B2_std[:self._fill]

            # BETA corrects the systematic downward bias of the minimum
            # (Martin 2001 — see module constants for details).
            self._E_hat_floor = BETA * float(np.min(valid_B1))
            self._noise_floor = BETA * float(np.min(valid_B2_mean))

            # std_noise uses the MEAN of per-frame stds, not the minimum.
            # We want the typical noise variability, not its lowest value.
            self._std_noise   = float(np.mean(valid_B2_std))

        return {
            'E_hat_floor' : self._E_hat_floor,
            'noise_floor' : self._noise_floor,
            'std_noise'   : self._std_noise,
            'is_burst'    : is_burst,
            'in_warmup'   : in_warmup,
            'buffer_fill' : self._fill,
        }

    def reset(self):
        """Reset the estimator to its initial state (all buffers cleared)."""
        self._B1[:]       = 0.0
        self._B2_mean[:]  = 0.0
        self._B2_std[:]   = 0.0
        self._buf_idx     = 0
        self._fill        = 0
        self._frame_count = 0
        self._E_hat_floor = 0.0
        self._noise_floor = 0.0
        self._std_noise   = 0.0

    # ------------------------------------------------------------------
    # Read-only properties — convenient access without going through update()
    # ------------------------------------------------------------------

    @property
    def E_hat_floor(self) -> float:
        """Current Ê_floor estimate [V²] (for Stage 1 threshold)."""
        return self._E_hat_floor

    @property
    def noise_floor(self) -> float:
        """Current noise_floor estimate [V] (for Stage 3 features)."""
        return self._noise_floor

    @property
    def std_noise(self) -> float:
        """Current std_noise estimate [V] (for Stage 3 features)."""
        return self._std_noise

    @property
    def is_warmed_up(self) -> bool:
        """True once the warm-up period is over and burst protection is active."""
        return self._frame_count > WARM_UP_FRAMES

    @property
    def buffer_fill(self) -> int:
        """Number of valid (non-burst) frames currently stored in the buffers."""
        return self._fill


# =============================================================================
# HELPER — FFT frame energy (used by both the noise estimator and Stage 1)
# =============================================================================

def compute_fft_energy(fft_magnitudes: np.ndarray) -> float:
    """
    Compute the mean squared energy of a single FFT frame over the analysis band.

    This is the E_i scalar defined in §4.2 of the spec:

        E_i = (1 / K) × Σ_{k=0}^{K-1} |A_i[k]|²

    where the sum runs over the K = 154 bins covering 20–80 kHz.

    Parameters
    ----------
    fft_magnitudes : np.ndarray
        Raw (non-normalized) FFT magnitude array for a single frame.
        Expected length: _K_BINS (= 154 bins), i.e. only the analysis-band
        bins as transmitted by the firmware.

    Returns
    -------
    float
        Frame energy in V² (or in whatever units fft_magnitudes is expressed).
    """
    if len(fft_magnitudes) == 0:
        return 0.0
    # Use only as many bins as are present (guard against truncated frames)
    k = min(len(fft_magnitudes), _K_BINS)
    return float(np.mean(fft_magnitudes[:k] ** 2))


# =============================================================================
# SIGNAL UTILITIES
# (migrated from replay_window_audio.py — v4 detection logic removed,
#  these pure-signal functions remain valid and unchanged)
# =============================================================================

def suppress_edge_artifacts(signal: np.ndarray) -> np.ndarray:
    """
    Suppress Gibbs ringing at the frame borders of the reconstructed iFFT signal.

    Algorithm — symmetric AND condition (§7, Step 3):
        Gibbs ringing is SYMMETRIC: spectral truncation at bin_start and bin_end
        produces ringing on BOTH frame borders simultaneously and with comparable
        intensity. This gives us a reliable detection condition:

          1. Compute RMS energy of the first GIBBS_CHECK_SAMPLES samples (left border).
          2. Compute RMS energy of the last  GIBBS_CHECK_SAMPLES samples (right border).
          3. Compute RMS energy of the interior (samples [40 .. N-40]).
          4. If BOTH borders exceed GIBBS_FACTOR × interior RMS → genuine Gibbs.
             Apply a half-Hann fade on each border to suppress it.
          5. Otherwise → do not touch the signal.

        Using AND instead of OR protects real clicks near frame edges:
          - A click near the left border → left high, right low → AND fails → preserved ✓
          - Pure Gibbs → both borders high → AND fires → suppressed ✓

    Parameters
    ----------
    signal : np.ndarray
        Time-domain signal output from iFFT, typically FFT_SIZE (512) samples.

    Returns
    -------
    np.ndarray
        Signal with Gibbs-suppressed borders (copy), or unchanged copy if
        the AND condition was not met.
    """
    n      = len(signal) #(with fs=200kHz and fft_size=512, windows lasts 2.56 ms for a total of 512 samples)
    result = signal.copy()

    if n < 100: 
        return result  # Too short to apply meaningful suppression

    interior = result[40 : n - 40]
    if len(interior) < 10:
        return result

    energy_interior = float(np.sqrt(np.mean(interior ** 2)))
    if energy_interior < 1e-15: #veeery small value to avoid division by zero and false positives on silent frames
        return result  # Near-zero signal — nothing to suppress

    energy_left  = float(np.sqrt(np.mean(result[:GIBBS_CHECK_SAMPLES] ** 2)))
    energy_right = float(np.sqrt(np.mean(result[n - GIBBS_CHECK_SAMPLES:] ** 2)))

    left_suspicious  = energy_left  > GIBBS_FACTOR * energy_interior
    right_suspicious = energy_right > GIBBS_FACTOR * energy_interior

    # Apply fade only when BOTH borders are anomalous (symmetric Gibbs condition)
    if left_suspicious and right_suspicious: #Hann window (half-cosine) fade for smooth suppression of Gibbs ringing
        fade = 0.5 * (1.0 - np.cos(
            np.pi * np.arange(GIBBS_CHECK_SAMPLES) / GIBBS_CHECK_SAMPLES
        ))
        result[:GIBBS_CHECK_SAMPLES]       *= fade        # fade-in on left border
        result[n - GIBBS_CHECK_SAMPLES:]   *= fade[::-1]  # fade-out on right border

    return result


def compute_hilbert_envelope(signal: np.ndarray) -> np.ndarray:
    """
    Compute the instantaneous amplitude envelope via the Hilbert transform.

    Uses only numpy (no scipy) for thread-safety on macOS: scipy.signal.hilbert
    calls BLAS/LAPACK through openblas, which can segfault when invoked from a
    QThread on the Cocoa main run-loop. This implementation uses only np.fft,
    which is safe in any thread.

    Algorithm:
        1. Compute the full N-point FFT of the signal.
        2. Zero out all negative-frequency components and double all positive
           ones (DC and Nyquist stay at weight 1). This constructs the
           one-sided spectrum of the complex analytic signal.
        3. IFFT → complex analytic signal Z[n] = x[n] + j·H{x}[n].
        4. Return |Z[n]| — the instantaneous amplitude envelope.

    Note on the v4 bug: the previous implementation used rfft + irfft, which
    reconstructed a real signal from the weighted half-spectrum instead of the
    complex analytic signal. This caused a systematic amplitude bias and
    non-constant envelope for pure sinusoids. The correct approach requires
    the full fft + ifft path.

    Parameters
    ----------
    signal : np.ndarray
        Real-valued time-domain signal (e.g. iFFT output, 512 samples).

    Returns
    -------
    np.ndarray
        Instantaneous amplitude A[n] = |Z[n]|, same length as input, all ≥ 0.
    """
    N  = len(signal)
    Xf = np.fft.fft(signal)

    # One-sided weighting vector for the analytic signal spectrum:
    #   • DC (k=0) and Nyquist (k=N//2, even N only) → weight 1 (unchanged)
    #   • Positive frequencies (k=1..N//2-1) → weight 2 (doubled)
    #   • Negative frequencies (k=N//2+1..N-1) → weight 0 (zeroed)
    h = np.zeros(N, dtype=np.float64)
    h[0]         = 1.0            # DC
    h[1 : N//2]  = 2.0            # positive frequencies
    if N % 2 == 0:
        h[N // 2] = 1.0           # Nyquist (only present for even N)

    # IFFT of the one-sided spectrum → complex analytic signal
    analytic = np.fft.ifft(Xf * h)
    return np.abs(analytic)   # instantaneous amplitude = |analytic signal|


def find_peak(envelope: np.ndarray) -> tuple:
    """
    Find the index and amplitude of the maximum of the Hilbert envelope.

    Parameters
    ----------
    envelope : np.ndarray
        Hilbert amplitude envelope A[n] (all values ≥ 0).

    Returns
    -------
    (peak_idx, peak_amp) : (int, float)
        Index of the maximum and its value.
    """
    peak_idx = int(np.argmax(envelope))
    peak_amp = float(envelope[peak_idx])
    return peak_idx, peak_amp


# =============================================================================
# iFFT RECONSTRUCTION
# =============================================================================

def _normalize_fft(fft_mags: np.ndarray, freq_axis_hz: np.ndarray) -> np.ndarray:
    """
    Apply 50 % conservative microphone normalization to FFT magnitudes.

    Uses the Knowles SPU0410LR5H-QB datasheet frequency-response curve
    (stored in _MIC_FREQ_HZ / _MIC_RESP_DB) with piecewise-linear interpolation.
    Applying only _MIC_NORM_FRACTION (50 %) of the correction keeps the amplitude
    error within ±2.9 dB across the 20–80 kHz analysis band.

    Parameters
    ----------
    fft_mags : np.ndarray
        Raw FFT magnitude values covering the full half-spectrum (fft_size//2 bins).
    freq_axis_hz : np.ndarray
        Frequency in Hz for each bin of fft_mags (same length).

    Returns
    -------
    np.ndarray
        Normalized magnitudes (copy), same shape as fft_mags.
    """
    analysis_mask = (freq_axis_hz >= BIN_START_HZ) & (freq_axis_hz <= BIN_END_HZ)

    # Interpolate datasheet dB response at every analysis-band frequency
    mic_db   = np.interp(freq_axis_hz[analysis_mask], _MIC_FREQ_HZ, _MIC_RESP_DB)

    # Convert fractional dB correction to a linear gain factor
    gain     = 10.0 ** (-mic_db * _MIC_NORM_FRACTION / 20.0)

    normalized = fft_mags.copy()
    normalized[analysis_mask] *= gain
    return normalized


def reconstruct_frame_v5(
    fft_mags:   np.ndarray,
    phase_int8: np.ndarray,
    fs:         int = FS,
    fft_size:   int = FFT_SIZE,
) -> Optional[dict]:
    """
    Reconstruct the time-domain signal from a single FFT frame.

    Implements the pre-processing pipeline (§7, Steps 1–3):

      Step 1 — Build complex spectrum:
        • Zero-pad the analysis-band magnitudes into a full half-spectrum array
          (fft_size // 2 bins).
        • Apply 50 % conservative microphone normalization.
        • Decode int8 phase values to radians and combine with magnitudes
          to form a complex spectrum.
        • Apply a Tukey (cosine-bell) taper to the analysis-band edges of the
          complex spectrum to reduce spectral leakage.

      Step 2 — iFFT:
        • np.fft.irfft → fft_size real samples.

      Step 3 — Gibbs suppression:
        • suppress_edge_artifacts() with symmetric AND condition.

    Parameters
    ----------
    fft_mags : np.ndarray
        Raw FFT magnitudes for the analysis band only (up to _K_BINS = 154 values).
    phase_int8 : np.ndarray
        Phase values for the same bins, encoded as int8 [-127, +127] → [-π, +π].
    fs : int
        Sampling rate [Hz].
    fft_size : int
        FFT window size [samples].

    Returns
    -------
    dict with keys:
        'signal'      : np.ndarray – Gibbs-suppressed time-domain signal (fft_size samples)
        'fft_norm'    : np.ndarray – mic-normalized magnitudes (full half-spectrum)
        'freq_axis'   : np.ndarray – frequency axis for the full half-spectrum [Hz]
    Returns None if the frame data is invalid (empty phase array or iFFT failure).
    """
    if len(phase_int8) == 0:
        return None

    num_bins  = fft_size // 2           # Half-spectrum length (256 bins)
    bin_freq  = fs / fft_size           # Hz per bin
    bin_start = int(BIN_START_HZ / bin_freq)
    bin_end   = int(BIN_END_HZ   / bin_freq)

    # Frequency axis for the full half-spectrum (used for normalization)
    freq_axis = np.arange(num_bins, dtype=np.float64) * bin_freq

    # ── Step 1a: zero-pad into full half-spectrum ─────────────────────────────
    full_mag   = np.zeros(num_bins, dtype=np.float64)
    full_phase = np.zeros(num_bins, dtype=np.int8)

    # Number of analysis-band bins available in this frame (may be < _K_BINS if
    # the firmware sent a truncated packet)
    n_bins = min(len(fft_mags), len(phase_int8), bin_end - bin_start + 1)
    full_mag  [bin_start : bin_start + n_bins] = fft_mags  [:n_bins]
    full_phase[bin_start : bin_start + n_bins] = phase_int8[:n_bins]

    # ── Step 1b: mic normalization ────────────────────────────────────────────
    fft_norm = _normalize_fft(full_mag, freq_axis)

    # ── Step 1c: build complex spectrum ──────────────────────────────────────
    # int8 phase [-127, +127] maps linearly to [-π, +π]
    phases_rad       = (full_phase.astype(np.float64) / 127.0) * np.pi
    complex_spectrum = fft_norm * np.exp(1j * phases_rad)

    # ── Step 1d: Tukey taper on the analysis-band edges ──────────────────────
    # Smoothly ramp the complex spectrum to zero at the lower and upper edges of
    # the analysis band to reduce spectral leakage artifacts in the iFFT.
    taper_len = max(5, round(n_bins * TUKEY_TAPER_FRACTION))
    for i in range(taper_len):
        alpha = i / taper_len
        taper_val = 0.5 * (1.0 - np.cos(np.pi * alpha))
        complex_spectrum[bin_start + i]              *= taper_val  # left edge
        complex_spectrum[bin_start + n_bins - 1 - i] *= taper_val  # right edge

    # ── Step 2: iFFT ─────────────────────────────────────────────────────────
    try:
        signal_raw = np.fft.irfft(complex_spectrum, n=fft_size)
    except Exception:
        return None

    # ── Step 3: Gibbs suppression ─────────────────────────────────────────────
    signal = suppress_edge_artifacts(signal_raw)

    return {
        'signal'    : signal,
        'fft_norm'  : fft_norm,
        'freq_axis' : freq_axis,
    }


# =============================================================================
# STAGE 1 — Adaptive energy threshold + run-length filter
# =============================================================================

def run_stage1_v5(dm, k: float = K_STAGE1_DEFAULT) -> list:
    """
    Stage 1 of the v5 click detection pipeline (§5).

    Processes all frames in chronological order, maintaining an
    AdaptiveNoiseEstimatorV5 that tracks the noise floor as it changes over time.

    For every frame:
      1. Compute FFT energy E_i.
      2. Reconstruct iFFT → Hilbert envelope → env_mean_i, env_std_i.
      3. Update the noise estimator (burst protection applied internally).
      4. Check Stage 1 criterion: E_i  >  k × Ê_floor(i).

    Then apply the run-length filter: discard any run of consecutive
    above-threshold frames longer than MAX_RUN (sustained noise, not a click).

    Parameters
    ----------
    dm : AudioDataManager-like object
        Must expose:
            .fft_data    (list/array of 1-D float arrays — one per frame)
            .phase_data  (list/array of 1-D int8  arrays — one per frame)
            .header_info (dict with keys 'fs', 'fft_size')
            .total_frames (int)
    k : float
        Stage 1 threshold multiplier (default K_STAGE1_DEFAULT = 1.5 for data
        collection). After SVM training use a tighter value (e.g. 2.0–3.46).

    Returns
    -------
    list of dict, one entry per surviving candidate frame:
        'frame_idx'   : int   – index in the recording
        'E_i'         : float – FFT energy of this frame [V²]
        'E_hat_floor' : float – Ê_floor at detection time [V²]
        'noise_floor' : float – noise_floor at detection time [V]  (for Stage 3)
        'std_noise'   : float – std_noise at detection time [V]    (for Stage 3)
        'group_size'  : int   – run length this candidate belongs to (1 ≤ n ≤ MAX_RUN)
    """
    fs       = dm.header_info.get('fs',       FS)
    fft_size = dm.header_info.get('fft_size', FFT_SIZE)

    estimator      = AdaptiveNoiseEstimatorV5()
    above_threshold = []   # accumulates ALL above-threshold frames before run filter

    for frame_idx in range(dm.total_frames):
        fft_mags = dm.fft_data[frame_idx]

        # ── 1. FFT energy (cheap — no iFFT needed) ───────────────────────────
        E_i = compute_fft_energy(fft_mags)

        # ── 2. iFFT reconstruction for noise estimator B2 update ─────────────
        # We need env_mean and env_std from the Hilbert envelope of every frame
        # (not just candidates) so that the B2 buffer stays current.
        if frame_idx < len(dm.phase_data):
            frame_data = reconstruct_frame_v5(
                dm.fft_data[frame_idx], dm.phase_data[frame_idx], fs, fft_size
            )
        else:
            frame_data = None

        if frame_data is not None:
            envelope   = compute_hilbert_envelope(frame_data['signal'])
            env_mean_i = float(np.mean(envelope))
            env_std_i  = float(np.std(envelope))
            noise      = estimator.update(E_i, env_mean_i, env_std_i)
        else:
            # Phase data unavailable for this frame — skip estimator update
            # and use the cached estimates from the previous frame.
            # This is a rare edge case (incomplete file) and has minimal impact.
            noise = {
                'E_hat_floor' : estimator.E_hat_floor,
                'noise_floor' : estimator.noise_floor,
                'std_noise'   : estimator.std_noise,
            }

        # ── 3. Stage 1 threshold check ────────────────────────────────────────
        E_hat_floor = noise['E_hat_floor']

        # Guard: skip check if we have no floor estimate yet (very first frame
        # before any warm-up data, E_hat_floor == 0).
        if E_hat_floor > 0 and E_i > k * E_hat_floor:
            above_threshold.append({
                'frame_idx'   : frame_idx,
                'E_i'         : E_i,
                'E_hat_floor' : E_hat_floor,
                'noise_floor' : noise['noise_floor'],
                'std_noise'   : noise['std_noise'],
            })

    if not above_threshold:
        return []

    # ── 4. Run-length filter ──────────────────────────────────────────────────
    # Group consecutive above-threshold frames into runs.
    # Two frames are consecutive if their frame indices differ by exactly 1.
    runs    = []
    current = [above_threshold[0]]

    for i in range(1, len(above_threshold)):
        if above_threshold[i]['frame_idx'] == above_threshold[i - 1]['frame_idx'] + 1:
            current.append(above_threshold[i])   # extend current run
        else:
            runs.append(current)                  # close run, start new one
            current = [above_threshold[i]]
    runs.append(current)                          # close last run

    # Keep only runs that are short enough to be click candidates.
    # Tag each surviving frame with its run length for downstream use.
    survivors = []
    for run in runs:
        if len(run) <= MAX_RUN:
            for candidate in run:
                survivors.append({**candidate, 'group_size': len(run)})

    return survivors
