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
             [compute_features_v5 is called between Stage 1 and Stage 2 by the caller]
  Stage 2  – Valid-fit and OOD gate: R² ≥ STAGE2_R2_MIN, SPR < STAGE2_SPR_MAX
  Stage 3  – SVM classification (RBF kernel, 16 features, calibrated threshold)
  Stage 4  – Deduplication (merge nearby detections from the same physical click)

The old v4 Stage 2 (FFT hard-threshold filters on SPR and peak FFT amplitude) has
been removed. SPR and peak amplitude are now features fed to the SVM, so the
pipeline learns their discriminative weight from labelled data instead of fixed
hand-tuned thresholds.

Stage 2 applies two gates, both mirroring the pre-filters used on the training data
in train_svm.py so the SVM only receives inputs within its trained distribution:
  • R² ≥ 0.10: if the decay does not fit an exponential, τ and all decay-window
    features are unreliable — discard before the SVM.
  • SPR < 100: if the spectrum is extremely tonal (single dominant component),
    the sample is out-of-distribution for the model — discard before the SVM.

Implementation notes:
  • All numpy operations in the signal processing stages — no scipy.linalg /
    BLAS / LAPACK calls — ensuring thread-safety on macOS (avoids segfaults from
    Apple Accelerate inside QThreads).
  • sklearn's SVC.predict_proba() at INFERENCE time uses the libsvm C extension
    (no BLAS) plus Platt-sigmoid arithmetic (pure scalar), so Stage 3 is also
    safe to call from a QThread.
  • Every tuneable constant is defined once in the '── ALGORITHM CONSTANTS ──'
    section below and referenced by name throughout. Nothing is buried inline.

Reference: CLICK_DETECTION_ALGORITHM_v5.md (June 2026)
"""

import numpy as np
from typing import Optional
from pathlib import Path


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
M_SUBWINDOWS   = 10                        # Number of sub-windows
SUBWINDOW_SIZE = W_NOISE // M_SUBWINDOWS   # = 75 frames per sub-window (~192 ms at 390 FPS)
                        # Robustness argument: a single glitch frame affects at most 1 of 10 sub-window
                        # minima. The median of 10 values is insensitive to a single outlier — it would
                        # require ≥ 5/10 sub-windows contaminated to shift the median, which is
                        # physically implausible for isolated glitches. (Martin 2001)

BETA           = 1.3    # Martin (2001) minimum-statistics bias correction.
                        # The minimum of a finite buffer of noisy values
                        # systematically underestimates the true floor;
                        # β = 1.3 corrects this bias (validated in the original
                        # Martin 2001 paper on speech noise estimation).
                        # Slightly lower than usually used 1.5 since we operate on the median.

ALPHA          = 4.0    # Burst exclusion multiplier: a frame is considered
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
_MIC_NORM_FRACTION = 0.5   # 50 % conservative normalization factor in order to avoid amplifing
#                                              noise too much. See documentation for more details

# FIT PIPELINE (§10)
DECAY_SLOPE_WINDOW        = 4     # Local slope window length W [samples] (§10.2 Step A).
                                   # slope_local[n] = (A[n+W-1] - A[n]) / (W-1).
DECAY_SLOPE_THRESHOLD_FRAC = 0.01 # Slope threshold = fraction × peak_amp.
                                   # A window is "descending" if slope < -threshold.
DECAY_SLOPE_CONSECUTIVE   = 3     # Number of consecutive descending windows required
                                   # to confirm that the true decay has started.
DECAY_START_CAP           = 20    # Hard cap: decay_start ≤ peak_idx + CAP [samples].
                                   # = 100 µs @ 200 kHz. Enforces τ_min and avoids
                                   # searching far into the tail.
DECAY_START_FALLBACK      = 5     # Fallback offset from peak_idx [samples] when the
                                   # slope criterion is not met within the cap.
DECAY_END_CONSECUTIVE     = 4     # Y: consecutive sub-threshold samples required to
                                   # confirm the end of the decay (§10.2 Step B).
DECAY_END_NOISE_ALPHA     = 1.5   # decay_end threshold: noise_floor + ALPHA × std_noise.
                                   # 1.5 std above the floor provides a ~93 % confidence
                                   # level that the signal has returned to background.
GAUSS_SIGMA               = 2     # Gaussian smoothing kernel σ [samples] = 10 µs.
                                   # Applied to the decay segment before the OLS fit
                                   # to suppress high-frequency ripple without distorting
                                   # the decay shape (σ << τ_min). (§10.2 Step C)
MIN_FIT_SAMPLES           = 10    # Minimum fit-window length for a valid OLS fit.
                                   # Shorter windows → unreliable slope estimate.
FIT_LOG_EPSILON           = 1e-9  # Floor value used in log(max(x, ε)) to prevent
                                   # log(0) when the envelope touches zero.

# Derived: pre-computed Gaussian kernel (computed once at import time).
# Truncated at 3σ (covers 99.7 % of the Gaussian mass).
_GAUSS_RADIUS = int(np.ceil(3 * GAUSS_SIGMA))                          # = 6 samples
_k            = np.arange(-_GAUSS_RADIUS, _GAUSS_RADIUS + 1, dtype=np.float64)
_GAUSS_KERNEL = np.exp(-_k**2 / (2.0 * GAUSS_SIGMA**2))
_GAUSS_KERNEL = _GAUSS_KERNEL / _GAUSS_KERNEL.sum()                    # unit-sum normalisation

# Numerical guard used in run_fit_pipeline_v5: slopes whose absolute value is
# smaller than this are indistinguishable from zero in double precision and are
# treated as "no decay" (→ τ = −1). This is NOT a physical threshold on τ.
_SLOPE_ZERO_GUARD = 1e-10

# FEATURE COMPUTATION (§8)
LEVEL_STD_FACTOR     = 1.0  # LEVEL = noise_floor + LEVEL_STD_FACTOR × std_noise.
                              # Defines "the signal has emerged from the noise floor"
                              # for: rise_time, fall_time, pre-window boundary,
                              # kurtosis event_start. Factor of 1.0 = 1 σ above floor.
POST_WINDOW_SAMPLES  = 100  # Length of the post-SNR window P [samples] = 0.5 ms.
                              # Captures signal level immediately after the click ends.
PRE_WINDOW_SAMPLES   = 100  # Length of the pre-SNR window P [samples] = 0.5 ms.
                              # Symmetric with POST_WINDOW_SAMPLES: captures signal
                              # level immediately before the click emerged above LEVEL.
                              # The previous-frame extension ensures 100 samples are
                              # always available regardless of where in the frame the
                              # click lands.
PRE_MIN_SAMPLES      = 5    # Minimum samples needed for a meaningful pre-window RMS.
                              # If fewer samples are available, pre_SNR is set to 1.0
                              # (neutral — "unknown silence before click").
ZCR_CLICK_TAU_FACTOR = 1.0  # ZCR_click window half-width = τ_ms × fs/1000 × factor.
                              # Factor = 1.0 sets window to ±1 decay time constant,
                              # capturing the main oscillation period of the click.
ZCR_CLICK_FALLBACK_MS = 0.3 # Fallback ZCR_click half-width [ms] when τ is invalid
                              # (τ = −1). 0.3 ms covers the typical click duration.
MIN_CENTROID_SAMPLES  = 6   # Minimum segment length [samples] for a meaningful
                              # short-time FFT in the centroid_shift computation.

# Derived
_BIN_MID = int(40_000 / _BIN_FREQ)   # 40 kHz bin index in the full half-spectrum.
                                       # Splits the analysis band at 40 kHz for
                                       # R_spectral = E_low[20-40] / E_high[40-80].

# STAGE 2 — Valid-fit gate (§11)
STAGE2_R2_MIN  = 0.10   # Minimum acceptable R² from the log-linear decay fit.
                         # Below this value the fit explains less than 10 % of the
                         # variance → τ, fit_coverage, and every feature that reads
                         # decay_start/decay_end are unreliable. Candidate is dropped.
                         # Mirrors NOISE_FILTER_R2_MIN in train_svm.py.

STAGE2_TAU_MIN = 0.0    # τ must be strictly positive. _fit_decay_segment returns the
                        # −1.0 sentinel when no exponential decay could be fitted; such
                        # a candidate has no usable decay window, so every decay-derived
                        # feature is meaningless. Rejected under the same verdict as the
                        # R² gate (STAGE_BLOCKED_R2) — both mean "invalid fit".

STAGE2_SPR_MAX = 100.0  # Maximum acceptable Spectral Peak Ratio.
                         # SPR = max(power) / mean(power) over the 20–80 kHz band.
                         # SPR ≥ 100 means the spectrum is dominated by a single
                         # tonal component — extremely narrowband, never seen in
                         # real clicks (max SPR in the training click set ≈ 34).
                         # During training, noise samples with SPR ≥ 100 were
                         # removed by NOISE_FILTER_SPR_MAX in train_svm.py, so
                         # neither class was ever shown this region of feature space.
                         # A candidate with SPR ≥ 100 is therefore out-of-distribution
                         # for the SVM; its prediction would be undefined.
                         # Mirrors NOISE_FILTER_SPR_MAX in train_svm.py.

# STAGE 4 — Deduplication (§13)
# Deduplication now groups by the click's ABSOLUTE peak sample (peak_abs), not by
# frame-index proximity: once every candidate resolves its peak on the stitched
# prev|curr|next trace (resolve_click), the two Stage 1 candidates a straddling
# click produces resolve to the *same* peak_abs and collapse exactly. This makes
# click identity deterministic and model-independent (the old logic broke ties by
# svm_probability, so identity moved with every retrained model).
PEAK_MATCH_SAMPLES = 8   # Two detections belong to the same physical click when
                          # their peak_abs differ by ≤ this many samples (= 40 µs
                          # @ 200 kHz). A margin, not a window: the fi and fi+1
                          # candidates of one click resolve to an integer-identical
                          # peak_abs, but per-frame adaptive noise can nudge the
                          # onset/peak search by a sample or two.

DEDUP_WINDOW_FRAMES = 3  # DEPRECATED for Stage 4 (superseded by PEAK_MATCH_SAMPLES).
                          # Kept defined because src/ml/evaluate_candidates.py still
                          # imports it; do not reference it from run_stage4_v5.
                          # Historical meaning: max frame-index gap between two
                          # detections treated as the same physical click (~7.7 ms).

# CLICK RESOLUTION (§8 pre-processing) — used by resolve_click on the stitched trace
CLICK_PEAK_SEARCH_SAMPLES = FFT_SIZE  # After locating the click onset, the true peak
                                       # is argmax of the envelope over
                                       # [onset : onset + this]. One frame (2.56 ms)
                                       # comfortably contains a single click's rise +
                                       # peak while being short enough that a *distinct*
                                       # louder click rarely falls inside it. Both the
                                       # fi and fi+1 candidates share one onset, so they
                                       # search the identical window and agree on the peak.

# STAGE VERDICT LABELS — the value of the 'stage_blocked' key written by
# run_stages234_annotated(). The strings are part of the CSV contract: they must
# stay identical to the ones src/ml/evaluate_candidates.py writes, otherwise the
# in-app export and the offline CLI would disagree on the same recording.
STAGE_OK            = ''              # survived all four stages — a confirmed click
STAGE_BLOCKED_R2    = 'Stage2_R2'     # R² < STAGE2_R2_MIN   — invalid decay fit
STAGE_BLOCKED_SPR   = 'Stage2_SPR'    # SPR ≥ STAGE2_SPR_MAX — out-of-distribution
STAGE_BLOCKED_SVM   = 'Stage3_SVM'    # SVM scored it below the decision threshold
STAGE_BLOCKED_DEDUP = 'Stage4_dedup'  # duplicate of a higher-confidence detection
# ─────────────────────────────────────────────────────────────────────────────


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

    interior = result[40 : n - 40] # 40 is a safe margin to exclude the borders and capture the true signal energy in the middle of the frame
    if len(interior) < 10:
        return result

    energy_interior = float(np.sqrt(np.mean(interior ** 2)))
    if energy_interior < 1e-15: #veeery small value to avoid division by zero and false positives on silent frames. Even when working with uV-level signals, the interior energy should be above this threshold
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

    # One-sided weighting vector for the ANALYTIC signal spectrum:
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
    normalize:  bool = True #False = skip mic correction (raw FFT)
) -> Optional[dict]:
    """
    Reconstruct the time-domain signal from a single FFT frame. (inverseFFT)

    Implements the pre-processing pipeline (§7, Steps 1–3):

      Step 1 — Build complex spectrum:
        • Zero-pad the analysis-band magnitudes into a full half-spectrum array
          (fft_size // 2 bins).
        • Apply 50 % conservative microphone normalization.
        • Decode int8 phase values to radians and combine with magnitudes
          to form a complex spectrum.
        • Apply a Tukey (cosine-bell) taper to the analysis-band edges of the
          complex spectrum to reduce spectral leakage.
        • Multiply by N/2 to undo the firmware's own 2/N normalization — see
          the AMPLITUDE SCALE note below.

      Step 2 — iFFT:
        • np.fft.irfft → fft_size real samples, in VOLTS.

      Step 3 — Gibbs suppression:
        • suppress_edge_artifacts() with symmetric AND condition.

    AMPLITUDE SCALE
    ---------------
    The firmware transmits magnitudes that are ALREADY amplitudes in volts: it
    converts ADC counts to volts before the FFT (main_with_phase.c:118) and then
    scales every bin by 2/N afterwards (main_with_phase.c:224). np.fft.irfft, by
    contrast, expects RAW (unnormalized) rfft coefficients and supplies its own
    1/N. Feeding the transmitted magnitudes straight to irfft therefore applied
    2/N twice and produced a signal N/2 = 256x too small — clicks appeared as
    ~10 uV, which is 70x below one ADC quantization step (3.3/4095 = 806 uV) and
    hence physically impossible. Step 1e restores the raw-FFT scale, so `signal`
    is now in true volts and clicks read in mV.

    See docs/fft_and_ifft/IFFT_AMPLITUDE_SCALE_FIX.md for the full derivation and
    the proof that all 17 v5 features (being dimensionless ratios) are unchanged.

    Parameters
    ----------
    fft_mags : np.ndarray
        FFT magnitudes for the analysis band only (up to _K_BINS = 154 values).
        Already amplitude-normalized by the firmware → units of volts.
    phase_int8 : np.ndarray
        Phase values for the same bins, encoded as int8 on the firmware's
        128-counts-per-π scale: [-128, +127] → [-π, +π).
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
    if normalize: #default True
        fft_norm = _normalize_fft(full_mag, freq_axis)
    else:
        fft_norm = full_mag.copy()   # skip correction

    # ── Step 1c: build complex spectrum ──────────────────────────────────────
    # int8 phase decode on the firmware's 128-counts-per-π scale:
    # [-128, +127] → [-π, +π). The firmware LUT is generated as
    # round(atan(i/255)·128/π) with 64 = 90°, so the divisor must be 128
    # (see docs/fft_and_ifft/FFT_PHASE_TECHNICAL_SPECIFICATION.md §4.2).
    phases_rad       = (full_phase.astype(np.float64) / 128.0) * np.pi
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

    # ── Step 1e: restore the raw-FFT scale ───────────────────────────────────
    # The firmware already divides every magnitude by N/2 (main_with_phase.c:224,
    # FFT_NORMALIZATION_FACTOR_NORMAL = 2/FFT_BUFFER_SIZE), so fft_mags are
    # AMPLITUDES in volts, not raw FFT coefficients. np.fft.irfft expects raw
    # coefficients and applies its own 1/N — without this factor the 2/N lands
    # twice and the reconstruction comes out N/2 = 256x too small.
    complex_spectrum = complex_spectrum * (fft_size / 2.0)

    # ── Step 2: iFFT ─────────────────────────────────────────────────────────
    try:
        signal_raw = np.fft.irfft(complex_spectrum, n=fft_size)
    except Exception:
        print("iFFT reconstruction failed for frame with magnitudes:", fft_mags)
        return None

    # ── Step 3: Gibbs suppression ─────────────────────────────────────────────
    signal = suppress_edge_artifacts(signal_raw)

    return {
        'signal'    : signal,
        'fft_norm'  : fft_norm,
        'freq_axis' : freq_axis,
    }


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

    def _median_of_local_minima(self, buffer: np.ndarray) -> float:
        """
        Compute  median(m_1, …, m_M)  where  m_j = min(sub-window j).

        Divides `buffer` into M_SUBWINDOWS non-overlapping sub-windows of
        SUBWINDOW_SIZE entries each (§4.5, §4.6).

        Partial buffer (during warm-up):
          - If fewer than SUBWINDOW_SIZE entries are present → global min fallback
            (same conservative behaviour as before, only for the first 75 frames).
          - Otherwise: as many complete sub-windows as fit, plus a partial final
            sub-window for the remainder — median over however many m_j exist.

        NOTE: β (BETA) correction is NOT applied here — the caller multiplies
        by BETA so that the correction is clearly visible at the call site.
        """
        n = len(buffer)
        if n == 0:
            return 0.0

        if n < SUBWINDOW_SIZE:
            # Fewer entries than one sub-window: no basis for median yet.
            # Fall back to global minimum — same as previous behaviour.
            # This only occurs for the very first SUBWINDOW_SIZE = 75 frames.
            return float(np.min(buffer))

        local_mins = []
        for j in range(M_SUBWINDOWS):
            start = j * SUBWINDOW_SIZE
            end   = start + SUBWINDOW_SIZE
            if end <= n:
                # Complete sub-window
                local_mins.append(float(np.min(buffer[start:end])))
            elif start < n:
                # Partial final sub-window (warm-up: buffer not yet full)
                local_mins.append(float(np.min(buffer[start:n])))
                break  # no more sub-windows can start
            else:
                break  # start >= n: nothing left

        # Should never be empty here (n >= SUBWINDOW_SIZE guarantees ≥1 window),
        # but guard defensively.
        if not local_mins:
            return float(np.min(buffer))

        return float(np.median(local_mins))

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

        The class includes TWO Noise estimators: one for stage 1 and one for stage 3.
        Though stage 3 is accessed only when events pass stage 1, both estimators are
        updated every frame becuase they both need SILENT frames and are not that heavy.
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
        # media of local minima instead of global minimum to reduce bias from outliers and non-stationary noise.
        if self._fill > 0:
            valid_B1      = self._B1[:self._fill]
            valid_B2_mean = self._B2_mean[:self._fill]
            valid_B2_std  = self._B2_std[:self._fill]

            # §4.5: Ê_floor = β · median(m_1,…,m_M)  where m_j = min(sub-window j of B1)
            # β = BETA = 1.5 corrects the systematic downward bias of local minima (Martin 2001).
            self._E_hat_floor = BETA * self._median_of_local_minima(valid_B1)

            # §4.6: same method for noise_floor (B2_mean)
            self._noise_floor = BETA * self._median_of_local_minima(valid_B2_mean)

            # §4.6: std_noise uses MEAN of per-frame stds — captures typical
            # variability, not its floor. Unchanged from previous implementation.
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
    # Cast to float64 before squaring: float32 overflows for large (but physically
    # plausible) values that arrive from corrupted or high-amplitude frames.
    return float(np.mean(fft_magnitudes[:k].astype(np.float64) ** 2))


### STAGES ###


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
        collection). After SVM training use a tighter value (e.g. 2.0, but we'll see).

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

        # ── 1. iFFT reconstruction for noise estimator B2 update ─────────────
        # We need env_mean and env_std from the Hilbert envelope of every frame
        # (not just candidates) so that the B2 buffer stays current.
        if frame_idx < len(dm.phase_data):
            frame_data = reconstruct_frame_v5(
                dm.fft_data[frame_idx], dm.phase_data[frame_idx],
                fs, fft_size, normalize=True ###DEFAULT MIC CORRECTION ENABLED
            )
        else:
            frame_data = None

        # ── 2. FFT energy — from the MIC-CORRECTED spectrum ──────────────────
        # E_i must come from the same normalized data as everything else in the
        # pipeline: the envelope above, every Stage 3 feature, and the E_i that
        # AudioLoadWorker precomputes (audio_load_progress.py:304). Using the raw
        # magnitudes here while the noise floor is built from mic-corrected data
        # compares two different quantities — and because the mic correction is
        # frequency-dependent (0.55x-1.49x across 20-80 kHz, not a constant), it
        # does NOT cancel out of E_i > k * Ê_floor: it changes which frames become
        # candidates. reconstruct_frame_v5 already returns the normalized spectrum,
        # so reusing it here costs nothing.
        if frame_data is not None:
            E_i = compute_fft_energy(frame_data['fft_norm'][_BIN_START : _BIN_END + 1])
        else:
            # No phase data for this frame → no reconstruction, so fall back to the
            # raw magnitudes. Rare (incomplete file), and the estimator is not
            # updated for such frames anyway.
            E_i = compute_fft_energy(fft_mags)

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

        # GUARD: skip check if we have no floor estimate yet (very first frame
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


# =============================================================================
# FIT PIPELINE — τ and R²  (§10)
# =============================================================================

def run_fit_pipeline_v5(
    envelope:             np.ndarray,
    peak_idx:             int,
    noise_floor:          float,
    std_noise:            float,
    fs:                   int = FS,
    next_frame_envelope:  Optional[np.ndarray] = None,
) -> dict:
    """
    Estimate the exponential decay constant τ and goodness-of-fit R² from the
    Hilbert envelope of a click candidate (§10 of the v5 spec).

    This is the top-level entry point for the fit pipeline. It delegates to:
      • find_decay_window_v5 — Steps A+B: locate decay_start and decay_end
      • _fit_decay_segment   — Steps C+D: Gaussian smoothing + OLS log-linear fit

    Call this function from the feature extraction pipeline. The returned dict
    contains decay_start and decay_end so all other features (asymmetry_integral,
    fall_time_ms, ZCR_post, post_SNR) can read them directly without re-running
    the window search.

    If you only need the window boundaries (e.g. for pre-processing or debugging),
    call find_decay_window_v5 directly.

    Parameters
    ----------
    envelope : np.ndarray
        Hilbert amplitude envelope A[n] for the current frame (512 samples).
    peak_idx : int
        Index of the amplitude peak within `envelope`.
    noise_floor : float
        Estimated noise floor [V], from AdaptiveNoiseEstimatorV5.
    std_noise : float
        Estimated noise standard deviation [V], from AdaptiveNoiseEstimatorV5.
    fs : int
        Sampling rate [Hz]. Default: FS (200 000).
    next_frame_envelope : np.ndarray or None
        Hilbert envelope of the immediately following frame, passed through to
        find_decay_window_v5 to allow cross-frame decay detection.

    Returns
    -------
    dict with keys:
        'tau_ms'       : float – decay constant [ms]. −1 if no valid decay.
        'R2'           : float – goodness of fit in log space. 0 if invalid.
        'fit_coverage' : float – len(fit_window) / (decay_end − decay_start).
        'decay_start'  : int   – index of decay onset in `envelope`.
        'decay_end'    : int   – index of decay end in the (possibly extended)
                                 envelope (may exceed len(envelope)).
    """

    # Steps A and B: locate the decay window.
    window = find_decay_window_v5(envelope, peak_idx, noise_floor, std_noise,
                                  next_frame_envelope)
    decay_start  = window['decay_start']
    decay_end    = window['decay_end']
    extended     = window['extended_envelope']

    # Steps C and D: Gaussian smoothing + OLS fit on the located window.
    return _fit_decay_segment(extended, decay_start, decay_end, fs)


def _fit_decay_segment(
    extended:    np.ndarray,
    decay_start: int,
    decay_end:   int,
    fs:          int = FS,
) -> dict:
    """
    Steps C and D of the fit pipeline: Gaussian smoothing + OLS log-linear fit.

    Operates on a pre-located decay window [decay_start, decay_end] within
    `extended` (the possibly frame-concatenated envelope). Called internally by
    run_fit_pipeline_v5; exposed separately so it can be unit-tested in isolation.

    Step C — Gaussian smoothing:
        Convolve extended[decay_start:decay_end] with _GAUSS_KERNEL (σ=2 samples,
        mode='valid'). Suppresses high-frequency ripple without distorting the
        decay shape (kernel width << τ_min).

    Step D — OLS log-linear fit:
        Fit log(fit_window) vs. sample index. Closed-form formula — no BLAS.
        Returns τ_ms = −1000 / (slope × fs) and R² in log space.

    Parameters
    ----------
    extended : np.ndarray
        Full (possibly extended) envelope from find_decay_window_v5.
    decay_start : int
        Decay window start index in `extended`.
    decay_end : int
        Decay window end index in `extended`.
    fs : int
        Sampling rate [Hz].

    Returns
    -------
    dict with keys:
        'tau_ms'       : float – decay constant [ms]. −1 if no valid decay.
        'R2'           : float – goodness of fit in log space.
        'fit_coverage' : float – len(fit_window) / (decay_end − decay_start).
        'decay_start'  : int   – echoed back for convenience.
        'decay_end'    : int   – echoed back for convenience.
    """
    decay_len = decay_end - decay_start

    if decay_len < 1:
        return {'tau_ms': -1.0, 'R2': 0.0, 'fit_coverage': 0.0,
                'decay_start': decay_start, 'decay_end': decay_end}

    decay_segment = extended[decay_start : decay_end]

    # ── Step C: Gaussian smoothing ────────────────────────────────────────────
    if len(decay_segment) >= len(_GAUSS_KERNEL):
        # mode='valid' avoids zero-padding artefacts at segment boundaries.
        fit_window = np.convolve(decay_segment, _GAUSS_KERNEL, mode='valid')
    else:
        fit_window = decay_segment.copy()   # too short to convolve

    n_fit = len(fit_window)

    if n_fit < MIN_FIT_SAMPLES:
        return {'tau_ms': -1.0, 'R2': 0.0,
                'fit_coverage': n_fit / max(1, decay_len),
                'decay_start': decay_start, 'decay_end': decay_end}

    # ── Step D: OLS log-linear fit ────────────────────────────────────────────
    log_env = np.log(np.maximum(fit_window, FIT_LOG_EPSILON))
    n_array = np.arange(n_fit, dtype=np.float64)

    n_pts  = float(n_fit)
    sum_x  = float(np.sum(n_array))
    sum_y  = float(np.sum(log_env))
    sum_xx = float(np.sum(n_array * n_array))
    sum_xy = float(np.sum(n_array * log_env))
    denom  = n_pts * sum_xx - sum_x * sum_x

    if abs(denom) < 1e-30:
        return {'tau_ms': -1.0, 'R2': 0.0,
                'fit_coverage': n_fit / max(1, decay_len),
                'decay_start': decay_start, 'decay_end': decay_end}

    slope_m     = (n_pts * sum_xy - sum_x * sum_y) / denom
    intercept_b = (sum_y - slope_m * sum_x) / n_pts

    # Negative slope + above numerical zero-guard → genuine exponential decay
    if slope_m < -_SLOPE_ZERO_GUARD:
        tau_ms = -1000.0 / (slope_m * fs)
    else:
        tau_ms = -1.0

    y_pred = slope_m * n_array + intercept_b
    ss_res = float(np.sum((log_env - y_pred)        ** 2))
    ss_tot = float(np.sum((log_env - sum_y / n_pts) ** 2))
    R2     = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-30 else 0.0

    return {
        'tau_ms'      : tau_ms,
        'R2'          : R2,
        'fit_coverage': n_fit / max(1, decay_len),
        'decay_start' : decay_start,
        'decay_end'   : decay_end,
    }


def find_decay_window_v5(
    envelope:            np.ndarray,
    peak_idx:            int,
    noise_floor:         float,
    std_noise:           float,
    next_frame_envelope: Optional[np.ndarray] = None,
) -> dict:
    """
    Locate the decay window [decay_start, decay_end] of a click candidate.

    This implements Steps A and B of the fit pipeline (§10.2) and is separated
    from the OLS fit so that other features (asymmetry_integral, fall_time_ms,
    ZCR_post, post_SNR) can reuse the same window boundaries without re-running
    the full fit.

    Step A — decay_start (slope criterion):
        Scan forward from peak_idx. The decay is considered to have started at
        the first position n where DECAY_SLOPE_CONSECUTIVE consecutive sliding
        windows (each DECAY_SLOPE_WINDOW wide) all have:
            slope < −DECAY_SLOPE_THRESHOLD_FRAC × peak_amp
        Hard-capped at peak_idx + DECAY_START_CAP samples (= 100 µs).
        Fallback to peak_idx + DECAY_START_FALLBACK if criterion never met.

    Step B — decay_end (noise threshold):
        Build an extended envelope (current frame + next frame, if provided).
        Scan forward from decay_start. The decay ends at the first position n
        where DECAY_END_CONSECUTIVE consecutive samples all satisfy:
            A[n] < noise_floor + DECAY_END_NOISE_ALPHA × std_noise
        If not found, decay_end = end of available data.

    Parameters
    ----------
    envelope : np.ndarray
        Hilbert amplitude envelope A[n] for the current frame.
    peak_idx : int
        Index of the amplitude peak within `envelope`.
    noise_floor : float
        Estimated noise floor [V], from AdaptiveNoiseEstimatorV5.
    std_noise : float
        Estimated noise standard deviation [V], from AdaptiveNoiseEstimatorV5.
    next_frame_envelope : np.ndarray or None
        Hilbert envelope of the immediately following frame. Appended to allow
        the decay_end search to extend past the current frame boundary.

    Returns
    -------
    dict with keys:
        'decay_start'       : int        – index of decay onset in `envelope`.
        'decay_end'         : int        – index of decay end in the (possibly
                                           extended) envelope. May exceed
                                           len(envelope) when the click spills
                                           into the next frame.
        'extended_envelope' : np.ndarray – envelope used for the search
                                           (current frame, or current + next).
                                           Needed by _fit_decay_segment and by
                                           feature functions that operate on the
                                           full decay region.
    """
    peak_amp = float(envelope[peak_idx])

    # ── Step A — locate decay_start ──────────────────────────────────────────
    slope_threshold = DECAY_SLOPE_THRESHOLD_FRAC * peak_amp
    n_env      = len(envelope)
    cap        = peak_idx + DECAY_START_CAP
    # Upper bound: the last starting index that allows all consecutive windows
    # to fit within the current frame.
    search_end = min(cap, n_env - DECAY_SLOPE_WINDOW - DECAY_SLOPE_CONSECUTIVE + 1)

    decay_start = None
    for n in range(peak_idx, search_end):
        all_descending = True
        for j in range(DECAY_SLOPE_CONSECUTIVE):
            # Slope over window [n+j .. n+j+DECAY_SLOPE_WINDOW-1]
            slope = (envelope[n + j + DECAY_SLOPE_WINDOW - 1] - envelope[n + j]) \
                    / (DECAY_SLOPE_WINDOW - 1)
            if slope >= -slope_threshold:
                all_descending = False
                break
        if all_descending:
            decay_start = n
            break

    if decay_start is None:
        decay_start = peak_idx + DECAY_START_FALLBACK

    # ── Step B — locate decay_end ─────────────────────────────────────────────
    # Extend the envelope into the next frame so clicks that spill past the
    # current frame boundary are fully captured.
    if next_frame_envelope is not None:
        extended = np.concatenate([envelope, next_frame_envelope])
    else:
        extended = envelope

    level = noise_floor + DECAY_END_NOISE_ALPHA * std_noise
    Y     = DECAY_END_CONSECUTIVE
    n_ext = len(extended)

    decay_end = None
    for n in range(decay_start, n_ext - Y):
        if np.all(extended[n + 1 : n + 1 + Y] < level):
            decay_end = n
            break

    if decay_end is None:
        decay_end = n_ext - 1

    return {
        'decay_start'       : decay_start,
        'decay_end'         : decay_end,
        'extended_envelope' : extended,
    }

# =============================================================================
# FEATURE SUB-FUNCTIONS  (§8)
# Each function computes one logical group of features. All return plain floats
# or dicts of plain floats — no numpy arrays — so the caller can merge them
# directly into a flat feature dict.
# =============================================================================

def _build_pre_window(
    envelope:            np.ndarray,
    signal:              np.ndarray,
    peak_idx:            int,
    noise_floor:         float,
    std_noise:           float,
    prev_frame_envelope: Optional[np.ndarray] = None,
    prev_frame_signal:   Optional[np.ndarray] = None,
) -> dict:
    """
    Build the pre-window for a click candidate, extending into the previous frame.

    DEPRECATED / no longer on the live path. compute_features_v5 now resolves the
    click on the stitched prev|curr|next context (build_click_context +
    resolve_click) and slices the pre-window directly from it, so the previous
    frame no longer needs special-casing here. Kept for reference and any external
    caller; the two now-inlined behaviours (pre_env/pre_sig) are equivalent for a
    click fully inside the current frame.

    The pre-window is defined as the PRE_WINDOW_SAMPLES samples immediately
    before the click emerged from the noise floor — i.e., the last 100 samples
    before the first sample above LEVEL = noise_floor + LEVEL_STD_FACTOR × std_noise
    going backward from peak_idx.

    This is symmetric with the post-window (POST_WINDOW_SAMPLES after decay_end):
    both capture the IMMEDIATE silence around the click event rather than the
    entire pre/post-click silence region.

    Why extending into the previous frame still matters:
        If peak_idx is small (e.g. click at sample 20 of a 512-sample frame),
        the current frame provides only 20 samples of pre-click context.
        The previous frame guarantees that PRE_WINDOW_SAMPLES are always
        available regardless of where in the frame the click lands.

    Strategy:
        1. Build an extended array: [prev_tail | current_frame_up_to_peak].
           If no previous frame is available, work with the current frame only.
        2. Scan backward from the peak position for the last sub-LEVEL sample
           — this is the true pre-boundary (pre_end_in_ext).
        3. Take the PRE_WINDOW_SAMPLES samples immediately before pre_end_in_ext
           as the pre-window (symmetric with post_window).
        4. Return the boundary index mapped back to the current frame's coordinate
           system (for kurtosis event_start).

    Parameters
    ----------
    envelope : np.ndarray
        Hilbert envelope of the current frame.
    signal : np.ndarray
        Raw iFFT signal of the current frame.
    peak_idx : int
        Index of the peak in the current frame.
    noise_floor : float
        Adaptive noise floor [V].
    std_noise : float
        Adaptive noise std [V].
    prev_frame_envelope : np.ndarray or None
        Hilbert envelope of the immediately preceding frame. Appended on the
        left of the search array to give the boundary search more context.
    prev_frame_signal : np.ndarray or None
        Raw iFFT signal of the preceding frame. Used to build pre_sig.
        If None but prev_frame_envelope is given, pre_sig is extended with
        zeros for the previous portion (ZCR_pre will be 0 there — conservative).

    Returns
    -------
    dict with keys:
        'pre_env'          : np.ndarray – last PRE_WINDOW_SAMPLES of the envelope
                             before the click boundary. RMS of this / noise_floor → pre_SNR.
        'pre_sig'          : np.ndarray – last PRE_WINDOW_SAMPLES of the raw signal
                             before the click boundary. Fed to _compute_zcr for ZCR_pre.
        'boundary_in_frame': int – index in the *current frame* where the click
                             first emerged above LEVEL. Used as event_start for
                             kurtosis. Clamped to 0 if the boundary is in the
                             previous frame.
    """
    level = noise_floor + LEVEL_STD_FACTOR * std_noise

    # ── Build extended arrays ─────────────────────────────────────────────────
    if prev_frame_envelope is not None:
        ext_env = np.concatenate([prev_frame_envelope, envelope[:peak_idx + 1]])
        if prev_frame_signal is not None:
            ext_sig = np.concatenate([prev_frame_signal, signal[:peak_idx + 1]])
        else:
            ext_sig = np.concatenate([
                np.zeros(len(prev_frame_envelope), dtype=signal.dtype),
                signal[:peak_idx + 1]
            ])
        prev_len = len(prev_frame_envelope)
    else:
        ext_env  = envelope[:peak_idx + 1]
        ext_sig  = signal[:peak_idx + 1]
        prev_len = 0

    # ── Scan backward for the last sub-LEVEL sample ───────────────────────────
    peak_in_ext    = len(ext_env) - 1
    pre_end_in_ext = 0                  # default: no sub-LEVEL sample found
    for n in range(peak_in_ext - 1, -1, -1):
        if ext_env[n] < level:
            pre_end_in_ext = n + 1   # exclusive upper bound of pre-window
            break

    # ── Extract pre-window: last PRE_WINDOW_SAMPLES before the boundary ───────
    # Symmetric with post_window (POST_WINDOW_SAMPLES samples after decay_end).
    pre_start_in_ext = max(0, pre_end_in_ext - PRE_WINDOW_SAMPLES)
    pre_env = ext_env[pre_start_in_ext : pre_end_in_ext]
    pre_sig = ext_sig[pre_start_in_ext : pre_end_in_ext]

    # ── Map boundary back to current-frame coordinates ────────────────────────
    boundary_in_frame = max(0, min(peak_idx, pre_end_in_ext - prev_len))

    return {
        'pre_env'          : pre_env,
        'pre_sig'          : pre_sig,
        'boundary_in_frame': boundary_in_frame,
    }


def _compute_zcr(
    signal:    np.ndarray,
    std_noise: float,
) -> float:
    """
    Compute the hysteresis zero-crossing rate [crossings/ms] for a signal segment.

    A crossing is only counted when the signal COMPLETELY traverses the noise
    band [-std_noise, +std_noise], eliminating spurious crossings from noise
    micro-fluctuations:

        upward:   x[n-1] < −std_noise  AND  x[n] > +std_noise
        downward: x[n-1] > +std_noise  AND  x[n] < −std_noise

    Parameters
    ----------
    signal : np.ndarray
        Raw iFFT signal segment (not the envelope).
    std_noise : float
        Noise standard deviation [V] — defines the hysteresis band width.

    Returns
    -------
    float
        Crossings per millisecond [crossings/ms]. Returns 0.0 if segment < 2 samples.
    """
    if len(signal) < 2:
        return 0.0

    n_crossings = 0
    for i in range(1, len(signal)):
        upward   = signal[i - 1] < -std_noise and signal[i] > +std_noise
        downward = signal[i - 1] > +std_noise and signal[i] < -std_noise
        if upward or downward:
            n_crossings += 1

    duration_ms = len(signal) / FS * 1000.0
    return n_crossings / duration_ms if duration_ms > 0 else 0.0


def _segment_spectral_centroid(segment: np.ndarray, fs: int) -> float:
    """
    Compute the power-weighted mean frequency (spectral centroid) of a short
    time-domain segment, restricted to the 20–80 kHz analysis band.

    A Hann window is applied before the FFT to reduce spectral leakage from the
    sharp segment boundaries — essential for short windows (10–100 samples).

    Parameters
    ----------
    segment : np.ndarray
        Time-domain signal segment.
    fs : int
        Sampling rate [Hz].

    Returns
    -------
    float
        Spectral centroid [Hz], or 0.0 if the segment is too short or has no energy.
    """
    n = len(segment)
    if n < MIN_CENTROID_SAMPLES:
        return 0.0

    windowed = segment * np.hanning(n)
    Xf       = np.abs(np.fft.rfft(windowed))
    freq     = np.fft.rfftfreq(n, d=1.0 / fs)

    mask  = (freq >= BIN_START_HZ) & (freq <= BIN_END_HZ)
    power = Xf[mask] ** 2
    total = float(np.sum(power))

    if total < 1e-30:
        return 0.0

    return float(np.sum(freq[mask] * power) / total)


# ── Individual feature functions ──────────────────────────────────────────────

def _feat_peak_snr(peak_amp: float, noise_floor: float) -> float:
    """
    peak_SNR = peak_amp / noise_floor  (§8.1)

    Dimensionless amplitude relative to the adaptive noise floor. Replaces the
    absolute 130 µV threshold from v4 — invariant to hardware gain changes.
    """
    if noise_floor <= 0:
        return 0.0
    return peak_amp / noise_floor


def _feat_pre_snr(pre_env: np.ndarray, noise_floor: float) -> float:
    """
    pre_SNR = RMS(pre_window) / noise_floor  (§8.2)

    The pre-window (pre_env) is built by _build_pre_window, which extends into
    the previous frame when available — so this function always receives a
    representative silence window regardless of where in the frame the click is.

    Returns 1.0 (neutral sentinel) if the pre-window is too short or noise_floor
    is zero. 1.0 means "unknown silence before click" and is conservative.

    A genuine click preceded by silence → pre_SNR ≈ 1.0.
    An embedded event or sustained noise   → pre_SNR > 1.
    """
    if len(pre_env) < PRE_MIN_SAMPLES or noise_floor <= 0:
        return 1.0
    rms = float(np.sqrt(np.mean(pre_env ** 2)))
    return rms / noise_floor


def _feat_post_snr(
    extended_envelope: np.ndarray,
    decay_end:         int,
    noise_floor:       float,
) -> float:
    """
    post_SNR = RMS(post_window) / noise_floor  (§8.3)

    The post-window spans [decay_end : decay_end + POST_WINDOW_SAMPLES] in the
    extended envelope (current frame optionally concatenated with the next frame
    by find_decay_window_v5). If fewer than POST_WINDOW_SAMPLES are available,
    uses whatever remains.

    A genuine click returns to silence → post_SNR ≈ 1.0.
    A sustained event → post_SNR > 1. Complements pre_SNR.
    """
    post_window = extended_envelope[decay_end : decay_end + POST_WINDOW_SAMPLES]

    if len(post_window) == 0 or noise_floor <= 0:
        return 1.0   # neutral sentinel

    rms = float(np.sqrt(np.mean(post_window ** 2)))
    return rms / noise_floor


def _feat_rise_fall_time(
    envelope:    np.ndarray,
    peak_idx:    int,
    noise_floor: float,
    std_noise:   float,
    fs:          int,
) -> dict:
    """
    rise_time_ms and fall_time_ms  (§8.4)

    LEVEL = noise_floor + LEVEL_STD_FACTOR × std_noise.

    rise_time_ms : time from last sub-LEVEL sample before peak to peak.
    fall_time_ms : time from peak to first sub-LEVEL sample after peak.

    Both are computed on the Hilbert envelope. They replace the single
    asymmetry_ratio of v4 with two independent, physically meaningful quantities.
    """
    level = noise_floor + LEVEL_STD_FACTOR * std_noise
    n     = len(envelope)

    # Rise: scan backward from peak_idx for the last sample below LEVEL
    rise_start = 0   # default: signal was above LEVEL since frame start
    for i in range(peak_idx - 1, -1, -1):
        if envelope[i] < level:
            rise_start = i
            break
    rise_time_ms = (peak_idx - rise_start) / fs * 1000.0

    # Fall: scan forward from peak_idx for the first sample below LEVEL
    fall_end = n   # default: signal stays above LEVEL until frame end
    for i in range(peak_idx + 1, n):
        if envelope[i] < level:
            fall_end = i
            break
    fall_time_ms = (fall_end - peak_idx) / fs * 1000.0

    return {'rise_time_ms': rise_time_ms, 'fall_time_ms': fall_time_ms}


def _feat_asymmetry_integral(
    envelope: np.ndarray,
    peak_idx: int,
    decay_end: int,
) -> float:
    """
    Asymmetry integral  (§8.5)

    Measures whether the Hilbert envelope decays more slowly than it rises —
    the expected signature of a cavitation click.

        W = decay_end − peak_idx
        right_side   = A[peak_idx : peak_idx + W]
        left_side    = A[peak_idx − W : peak_idx]
        left_flipped = left_side[::-1]  (both sides start at peak)

        asymmetry_integral = Σ(right_side − left_flipped) / (W × peak_amp)

    W is capped to the samples actually available before the peak, so the
    integral is always computed over symmetric windows.

    Returns
    -------
    float
        Dimensionless. Positive → decay slower than rise (genuine click).
                       ≈ 0     → symmetric (EMI spike, oscillator).
                       Negative → rise slower than fall (rare).
    """
    peak_amp = float(envelope[peak_idx])
    if peak_amp <= 0:
        return 0.0

    W = decay_end - peak_idx   # desired window half-width
    if W <= 0:
        return 0.0

    # Limit W to the samples available before the peak (left side)
    W = min(W, peak_idx)
    if W == 0:
        return 0.0

    right_side   = envelope[peak_idx          : peak_idx + W]
    left_side    = envelope[peak_idx - W      : peak_idx]
    left_flipped = left_side[::-1]   # flip so index 0 = sample nearest peak

    # Both sides should be the same length W; guard against rounding
    min_len = min(len(right_side), len(left_flipped))
    if min_len == 0:
        return 0.0

    diff = right_side[:min_len] - left_flipped[:min_len]
    return float(np.sum(diff)) / (min_len * peak_amp)


def _feat_zcr_triple(
    signal:    np.ndarray,
    pre_sig:   np.ndarray,
    peak_idx:  int,
    decay_end: int,
    tau_ms:    float,
    std_noise: float,
    fs:        int,
) -> dict:
    """
    Three zero-crossing rate features  (§8.6)

    All ZCRs use the same hysteresis band [-std_noise, +std_noise] on the raw
    iFFT signal x[n] (not the envelope). Only complete band-traversals count.

    ZCR_pre   : in pre_sig (built by _build_pre_window, may span into previous frame)
                → low for genuine click (silence before)
    ZCR_click : in [peak_idx − W_click : peak_idx + W_click] of current-frame signal
                → measures click oscillation frequency
    ZCR_post  : in [peak_idx : decay_end] of current-frame signal
                → oscillation during decay (decreasing for genuine click)

    W_click = τ_ms × fs / 1000 × ZCR_CLICK_TAU_FACTOR  (or fallback if τ invalid).

    All returned as [crossings/ms].
    """
    # ZCR_pre: use the pre-window signal built with previous-frame context
    zcr_pre = _compute_zcr(pre_sig, std_noise)

    # ZCR_click: ±W_click samples around the peak in the current frame
    if tau_ms > 0:
        W_click = max(1, int(tau_ms * fs / 1000.0 * ZCR_CLICK_TAU_FACTOR))
    else:
        W_click = max(1, int(ZCR_CLICK_FALLBACK_MS * fs / 1000.0))
    click_start = max(0, peak_idx - W_click)
    click_end   = min(len(signal), peak_idx + W_click)
    zcr_click   = _compute_zcr(signal[click_start : click_end], std_noise)

    # ZCR_post: from peak to end of decay in the current frame
    zcr_post = _compute_zcr(signal[peak_idx : decay_end], std_noise)

    return {'ZCR_pre': zcr_pre, 'ZCR_click': zcr_click, 'ZCR_post': zcr_post}


def _feat_kurtosis(
    signal:      np.ndarray,
    event_start: int,
    decay_end:   int,
) -> float:
    """
    Excess kurtosis of the raw iFFT signal over the event window  (§8.7)

        K_excess = mean((x − mean(x))⁴) / std(x)⁴ − 3

    Window: [event_start : decay_end] — from first noise-level crossing before
    peak to end of decay. Captures the full click without surrounding silence
    (which would dilute the impulsivity measure).

    Typical values:
      Gaussian noise        ≈  0
      Sustained vibration   :  2 – 7
      Genuine click         : 15 – 50
      EMI spike             : > 100

    Returns 0.0 if the window has fewer than 4 samples or zero variance.
    """
    window = signal[event_start:decay_end]

    if len(window) < 4:
        return 0.0

    mu  = float(np.mean(window))
    std = float(np.std(window))
    if std < 1e-30:
        return 0.0

    K = float(np.mean(((window - mu) / std) ** 4))
    return K - 3.0   # excess kurtosis


def _feat_centroid_shift(
    signal:    np.ndarray,
    peak_idx:  int,
    decay_end: int,
    fs:        int,
) -> float:
    """
    Spectral centroid shift during the click decay  (§8.8)

    For a genuine click, higher frequencies attenuate faster due to
    frequency-dependent damping → the spectral centroid shifts downward
    during the decay (SC_early > SC_late → centroid_shift > 0).

    The centroid is computed on short FFTs of two segments of the iFFT
    signal (not directly from the single-frame FFT magnitudes), so it
    captures spectral evolution within the frame:

        SC_early : spectral centroid of signal[peak_idx : peak_idx + third]
        SC_late  : spectral centroid of signal[peak_idx + 2×third : decay_end]
        centroid_shift = SC_early − SC_late   [Hz]

    A Hann window is applied to each segment before FFT to reduce leakage
    (segments can be as short as 10–30 samples).

    Returns 0.0 if the decay segment is too short for meaningful computation.
    """
    decay_len = decay_end - peak_idx
    third     = max(1, decay_len // 3)

    early_seg = signal[peak_idx            : peak_idx + third]
    late_seg  = signal[peak_idx + 2*third  : decay_end]

    if len(early_seg) < MIN_CENTROID_SAMPLES or len(late_seg) < MIN_CENTROID_SAMPLES:
        return 0.0

    sc_early = _segment_spectral_centroid(early_seg, fs)
    sc_late  = _segment_spectral_centroid(late_seg,  fs)

    return sc_early - sc_late   # [Hz]; positive → high freqs decayed first ✓


def _feat_fft_features(
    fft_norm:  np.ndarray,
    freq_axis: np.ndarray,
) -> dict:
    """
    Three FFT-domain features derived from the mic-normalized spectrum  (§8.9)

    All computed over the analysis band only (bins _BIN_START .. _BIN_END,
    covering 20–80 kHz, 154 bins).

    SPR        : Spectral Peak Ratio = max(power) / mean(power).
                 Low SPR → broadband (good for genuine click).
                 High SPR → tonal/narrowband (EMI, oscillator artefact).

    R_spectral : Energy ratio E_low[20-40 kHz] / E_high[40-80 kHz].
                 Describes the spectral balance of the event.

    FPE        : Frequency of Peak Energy [Hz] = freq at max power bin.
                 Location of the dominant spectral component.
    """
    # Analysis-band magnitudes and power
    band_mag   = fft_norm[_BIN_START : _BIN_END + 1]
    band_freq  = freq_axis[_BIN_START : _BIN_END + 1]
    band_power = band_mag ** 2

    mean_power = float(np.mean(band_power))
    max_power  = float(np.max(band_power))

    # SPR
    spr = max_power / mean_power if mean_power > 1e-30 else 0.0 #avoid 0 division

    # R_spectral: split analysis band at 40 kHz
    # _BIN_MID is the absolute index; relative index within the band slice:
    mid_rel   = _BIN_MID - _BIN_START
    low_power = band_power[:mid_rel]
    high_power= band_power[mid_rel:]
    e_low     = float(np.mean(low_power))  if len(low_power)  > 0 else 0.0
    e_high    = float(np.mean(high_power)) if len(high_power) > 0 else 0.0
    r_spectral = e_low / e_high if e_high > 1e-30 else 0.0

    # FPE: frequency of maximum power bin
    peak_bin = int(np.argmax(band_power))
    fpe_hz   = float(band_freq[peak_bin])

    return {'SPR': spr, 'R_spectral': r_spectral, 'FPE_hz': fpe_hz}


# =============================================================================
# MASTER FEATURE FUNCTION
# =============================================================================

def build_click_context(
    prev_sig: Optional[np.ndarray],
    curr_sig: np.ndarray,
    next_sig: Optional[np.ndarray],
) -> dict:
    """
    Stitch up to three consecutive reconstructed frames into ONE continuous
    time-domain trace, so a click that straddles a frame boundary is analysed on
    a single array instead of being silently truncated at the 512-sample edge.

    Only the current frame (curr_sig) is mandatory; prev_sig / next_sig are None
    at the recording edges and are simply omitted (origin adjusts accordingly).

    The Hilbert envelope is computed on the STITCHED signal (not per-frame then
    concatenated), matching the Region-FFT path and giving a continuous envelope
    across the joins. Each frame was reconstructed with its own Tukey taper, so
    the joins carry a mild seam — their positions are returned in 'seams' so
    callers can draw them honestly rather than pretend they are not there.

    Parameters
    ----------
    prev_sig, next_sig : np.ndarray or None
        Reconstructed iFFT signal of the previous / next frame, or None at a
        recording boundary.
    curr_sig : np.ndarray
        Reconstructed iFFT signal of the current (candidate) frame.

    Returns
    -------
    dict with keys:
        'signal'   : np.ndarray – the stitched trace [prev | curr | next].
        'envelope' : np.ndarray – Hilbert envelope of the stitched trace.
        'origin'   : int        – index of curr_sig's first sample in 'signal'.
        'n_frame'  : int        – len(curr_sig); the current-frame span is
                                  [origin : origin + n_frame].
        'seams'    : list[int]  – sample indices of the frame joins within
                                  'signal' (for honest rendering).
    """
    parts: list = []
    seams: list = []
    if prev_sig is not None and len(prev_sig) > 0:
        parts.append(np.asarray(prev_sig, dtype=np.float64))
        seams.append(len(parts[-1]))
    origin = sum(len(p) for p in parts)
    parts.append(np.asarray(curr_sig, dtype=np.float64))
    if next_sig is not None and len(next_sig) > 0:
        seams.append(sum(len(p) for p in parts))
        parts.append(np.asarray(next_sig, dtype=np.float64))

    signal = np.concatenate(parts) if len(parts) > 1 else parts[0]
    return {
        'signal':   signal,
        'envelope': compute_hilbert_envelope(signal),
        'origin':   int(origin),
        'n_frame':  int(len(curr_sig)),
        'seams':    seams,
    }


def resolve_click(
    ctx:         dict,
    noise_floor: float,
    std_noise:   float,
) -> dict:
    """
    Locate the click on the stitched context and return its boundaries in a
    SINGLE coordinate system (indices into ctx['signal'] / ctx['envelope']).

    This is what makes borderline features frame-grid independent AND makes the
    two Stage 1 candidates of a straddling click collapse:

      1. Frame-local peak — argmax of the envelope over the current frame only
         (the excursion Stage 1 actually flagged for this candidate).
      2. Onset — scan BACKWARD from that peak to the last sample below
         LEVEL = noise_floor + LEVEL_STD_FACTOR × std_noise. Because the search
         runs over the stitched trace, an onset in the previous frame is found
         rather than clamped away. The fi and fi+1 candidates of one click reach
         the SAME onset here.
      3. True peak — argmax of the envelope over
         [onset : onset + CLICK_PEAK_SEARCH_SAMPLES]. Both candidates share the
         onset and the window, so they recover the SAME peak (the fi+1
         candidate's frame-local peak is only the decay tail; scanning back and
         re-maximising recovers the real peak).
      4. Decay window — find_decay_window_v5 on the stitched envelope from the
         true peak. next_frame_envelope=None because ctx['envelope'] already
         contains the next frame.

    Returns
    -------
    dict with keys (int indices into ctx arrays, except peak_amp):
        'onset', 'peak', 'decay_start', 'decay_end', 'peak_amp'
    """
    env     = ctx['envelope']
    origin  = ctx['origin']
    n_frame = ctx['n_frame']
    n_ext   = len(env)

    # 1. Frame-local peak — the Stage-1 excursion for THIS candidate's frame.
    frame_hi   = min(origin + n_frame, n_ext)
    frame_peak = origin + int(np.argmax(env[origin:frame_hi]))

    # 2. Onset — last sub-LEVEL sample before the frame-local peak.
    level = noise_floor + LEVEL_STD_FACTOR * std_noise
    onset = 0
    for n in range(frame_peak - 1, -1, -1):
        if env[n] < level:
            onset = n
            break

    # 3. True peak — global max of the click from the onset.
    search_hi = min(n_ext, onset + CLICK_PEAK_SEARCH_SAMPLES)
    peak = onset + int(np.argmax(env[onset:search_hi]))

    # 4. Decay window on the stitched envelope (already includes the next frame).
    window      = find_decay_window_v5(env, peak, noise_floor, std_noise,
                                       next_frame_envelope=None)
    decay_start = min(int(window['decay_start']), n_ext - 1)
    decay_end   = min(int(window['decay_end']),   n_ext - 1)
    if decay_end < decay_start:
        decay_end = decay_start

    return {
        'onset':       int(onset),
        'peak':        int(peak),
        'decay_start': int(decay_start),
        'decay_end':   int(decay_end),
        'peak_amp':    float(env[peak]),
    }


def click_event_key(ctx: dict, resolved: dict, frame_idx: int) -> tuple:
    """
    Absolute, model-independent identity of a resolved click.

    peak_abs            : the peak's absolute sample index in the recording,
                          independent of which frame's candidate resolved it. The
                          fi and fi+1 candidates of a straddling click yield an
                          integer-identical peak_abs (it is exact sample
                          arithmetic), which is what lets Stage 4 collapse them.
    canonical_frame_idx : the frame that OWNS the peak (peak_abs // FFT_SIZE).

    frame_idx is the candidate's own frame (the frame at ctx['origin']).
    """
    peak_abs = frame_idx * FFT_SIZE + (resolved['peak'] - ctx['origin'])
    canonical_frame_idx = peak_abs // FFT_SIZE
    return int(peak_abs), int(canonical_frame_idx)


def compute_features_v5(
    ctx:         dict,
    resolved:    dict,
    fft_norm:    np.ndarray,
    freq_axis:   np.ndarray,
    noise_floor: float,
    std_noise:   float,
    fs:          int = FS,
) -> dict:
    """
    Compute all 17 v5 features for a single resolved click candidate.

    Every time-domain feature is computed on the STITCHED context
    (ctx['signal'] / ctx['envelope']) using the context-relative indices in
    `resolved` (onset, peak, decay_start, decay_end). Because the arrays already
    span prev|curr|next, an index past the current-frame boundary addresses real
    samples instead of silently truncating — the frame-straddle bug that made
    ZCR_post / kurtosis / centroid_shift_hz / asymmetry_integral wrong for
    borderline clicks is structurally impossible here.

    The spectral features (SPR, R_spectral, FPE_hz via _feat_fft_features) still
    use the frame's transmitted FFT — migrating them to a Region-FFT window over
    [onset, decay_end] is a separate, deferred change.

    Parameters
    ----------
    ctx : dict
        Output of build_click_context — stitched 'signal'/'envelope' + 'origin'.
    resolved : dict
        Output of resolve_click — onset/peak/decay_start/decay_end (indices into
        the ctx arrays) + peak_amp.
    fft_norm, freq_axis : np.ndarray
        Mic-normalized frame FFT magnitudes and their frequency axis [Hz].
    noise_floor, std_noise : float
        Adaptive noise estimates [V] for the candidate frame.
    fs : int
        Sampling rate [Hz].

    Returns
    -------
    dict with exactly 17 keys (all floats):
        peak_SNR, pre_SNR, post_SNR,
        rise_time_ms, fall_time_ms,
        asymmetry_integral,
        ZCR_pre, ZCR_click, ZCR_post,
        kurtosis,
        centroid_shift_hz,
        tau_ms, R2, fit_coverage,
        SPR, R_spectral, FPE_hz
    """
    signal   = ctx['signal']
    envelope = ctx['envelope']
    n_ext    = len(envelope)

    onset       = resolved['onset']
    peak_idx    = resolved['peak']
    decay_start = resolved['decay_start']
    decay_end   = resolved['decay_end']
    peak_amp    = resolved['peak_amp']

    # One coordinate system: every index must land inside the stitched arrays.
    # Cheap guards that turn a would-be silent truncation into a loud failure if
    # resolve_click and build_click_context ever disagree.
    assert 0 <= onset <= peak_idx < n_ext, \
        f"onset/peak out of context bounds: {onset}/{peak_idx}/{n_ext}"
    assert peak_idx <= decay_start <= decay_end < n_ext, \
        f"decay window out of context bounds: {decay_start}/{decay_end}/{n_ext}"

    # ── Decay fit: τ, R², fit_coverage (§10) ──────────────────────────────────
    fit = _fit_decay_segment(envelope, decay_start, decay_end, fs)

    # ── Pre-window (shared by pre_SNR, ZCR_pre) — the PRE_WINDOW_SAMPLES of the
    #    stitched trace immediately before the onset. No previous-frame special
    #    casing: the previous frame is already part of ctx. ──────────────────────
    pre_end   = onset + 1                       # exclusive: first click sample
    pre_start = max(0, pre_end - PRE_WINDOW_SAMPLES)
    pre_env   = envelope[pre_start:pre_end]
    pre_sig   = signal[pre_start:pre_end]

    # ── All 17 features ────────────────────────────────────────────────────────
    features = {}

    # Amplitude / SNR (§8.1–8.3)
    features['peak_SNR'] = _feat_peak_snr(peak_amp, noise_floor)
    features['pre_SNR']  = _feat_pre_snr(pre_env, noise_floor)
    features['post_SNR'] = _feat_post_snr(envelope, decay_end, noise_floor)

    # Temporal shape (§8.4–8.5) — rise/fall now cross the seam; asymmetry sees
    # the full decay instead of a frame-clipped one.
    features.update(_feat_rise_fall_time(envelope, peak_idx, noise_floor, std_noise, fs))
    features['asymmetry_integral'] = _feat_asymmetry_integral(envelope, peak_idx, decay_end)

    # Zero-crossing rates (§8.6) — ZCR_post spans the whole decay [peak, decay_end]
    features.update(_feat_zcr_triple(signal, pre_sig, peak_idx, decay_end,
                                     fit['tau_ms'], std_noise, fs))

    # Impulsivity (§8.7) — event window is the whole click [onset, decay_end]
    features['kurtosis'] = _feat_kurtosis(signal, onset, decay_end)

    # Spectral evolution (§8.8) — late segment is no longer empty for borderline clicks
    features['centroid_shift_hz'] = _feat_centroid_shift(signal, peak_idx, decay_end, fs)

    # Fit results as features (§10)
    features['tau_ms']       = fit['tau_ms']
    features['R2']           = fit['R2']
    features['fit_coverage'] = fit['fit_coverage']

    # FFT-domain features (§8.9) — still the frame FFT (Region-FFT migration TODO)
    features.update(_feat_fft_features(fft_norm, freq_axis))

    return features


# =============================================================================
# SVM MODEL LOADER
# =============================================================================

def load_svm_model(model_path: Path) -> dict:
    """
    Load a PlantLeaf v5 SVM model from a .pkl file produced by train_svm.py.

    Called ONCE at application startup (or when the user loads a new model file)
    and the returned dict is kept in memory and passed to run_stage3_v5 for every
    recording. Do NOT call this per-frame.

    Expected dict keys in the .pkl (all written by train_svm.py):
        'pipeline'    : fitted sklearn Pipeline (SimpleImputer → StandardScaler → SVC)
        'threshold'   : float — optimal decision threshold from ROC optimisation
        'features'    : list[str] — ordered feature names the model was trained on
        'kernel'      : str  — 'linear' or 'rbf' (informational)
        'all_results' : dict — per-kernel AUC and threshold (informational)

    Parameters
    ----------
    model_path : Path
        Absolute path to the .pkl file.

    Returns
    -------
    dict
        The raw model dict (same keys as above). Pass directly to run_stage3_v5.

    Raises
    ------
    FileNotFoundError
        If model_path does not exist.
    ValueError
        If the .pkl is missing required keys (wrong file or corrupt save).
    """
    import joblib  # imported lazily so the module is usable without sklearn when
                   # only signal processing functions are needed (e.g. unit tests)

    if not model_path.exists():
        raise FileNotFoundError(f"SVM model not found: {model_path}")

    model = joblib.load(model_path)

    # Validate that all required keys are present before the first recording runs.
    # A clear error here is much easier to debug than an AttributeError deep in
    # Stage 3 during a live acquisition.
    required = ('pipeline', 'threshold', 'features')
    missing  = [k for k in required if k not in model]
    if missing:
        raise ValueError(
            f"SVM model at {model_path} is missing keys: {missing}. "
            f"Re-train with the current train_svm.py."
        )

    return model


def has_precomputed_stage1_arrays(dm) -> bool:
    """True when `dm` carries the per-frame arrays AudioLoadWorker computes at load."""
    return all(
        getattr(dm, name, None) is not None
        for name in ('fft_means', 'E_hat_floor_arr', 'noise_floor_arr', 'std_noise_arr')
    ) and len(dm.fft_means) == dm.total_frames


def run_stage1_v5_precomputed(dm, k: float = K_STAGE1_DEFAULT) -> list:
    """
    Stage 1 using the per-frame arrays AudioLoadWorker already computed at load time.

    Same criterion and same run-length filter as run_stage1_v5, and returns candidate
    dicts of exactly the same shape — but it is a vectorised threshold over arrays
    instead of an iFFT + Hilbert transform per frame, so it is orders of magnitude
    faster on a long recording.

    IMPORTANT — this is not merely an optimisation of run_stage1_v5, it can select a
    slightly different candidate set. AudioLoadWorker computes E_i from the *normalised*
    (mic-corrected) spectrum, whereas run_stage1_v5 computes it from the raw magnitudes.
    Every candidate CSV and therefore the trained SVM came from this path, so this is
    the authoritative one: prefer it whenever the arrays are available, and treat
    run_stage1_v5 as the fallback for files loaded without them.

    Parameters
    ----------
    dm : data manager carrying fft_means, E_hat_floor_arr, noise_floor_arr,
         std_noise_arr and total_frames (see has_precomputed_stage1_arrays).
    k  : Stage 1 threshold multiplier.

    Returns
    -------
    list of dict — identical keys to run_stage1_v5:
        'frame_idx', 'E_i', 'E_hat_floor', 'noise_floor', 'std_noise', 'group_size'
    """
    fft_means     = np.asarray(dm.fft_means,       dtype=np.float64)
    E_hat_arr     = np.asarray(dm.E_hat_floor_arr, dtype=np.float64)
    noise_fl_arr  = np.asarray(dm.noise_floor_arr, dtype=np.float64)
    std_noise_arr = np.asarray(dm.std_noise_arr,   dtype=np.float64)

    # Threshold: E_i > k × Ê_floor  (vectorised, no Python loop over frames).
    # Ê_floor == 0 means the estimator had not warmed up — those frames cannot
    # be judged and are skipped.
    valid   = E_hat_arr > 0
    flagged = valid & (fft_means > k * E_hat_arr)
    indices = np.where(flagged)[0]

    if len(indices) == 0:
        return []

    above_threshold = [
        {
            'frame_idx'  : int(i),
            'E_i'        : float(fft_means[i]),
            'E_hat_floor': float(E_hat_arr[i]),
            'noise_floor': float(noise_fl_arr[i]),
            'std_noise'  : float(std_noise_arr[i]),
        }
        for i in indices
    ]

    # Run-length filter: a run longer than MAX_RUN is sustained noise, not a click.
    runs, current = [], [above_threshold[0]]
    for j in range(1, len(above_threshold)):
        if above_threshold[j]['frame_idx'] == above_threshold[j - 1]['frame_idx'] + 1:
            current.append(above_threshold[j])
        else:
            runs.append(current)
            current = [above_threshold[j]]
    runs.append(current)

    candidates = []
    for run in runs:
        if len(run) <= MAX_RUN:
            for entry in run:
                candidates.append({**entry, 'group_size': len(run)})

    return candidates


# =============================================================================
# STAGE 2 — Valid-fit gate and single tone rejection  (§11)
# =============================================================================

def run_stage2_v5(candidates: list) -> tuple:
    """
    Stage 2 of the v5 click detection pipeline: valid-fit and OOD gate  (§11).

    Two hard thresholds guard the feature vector before it is passed to the SVM.
    Both were chosen to match the pre-filters applied to the training data in
    train_svm.py, so the SVM only receives inputs that fall within the region of
    feature space it was trained on.

    Gate 1 — valid exponential decay:
        R² ≥ STAGE2_R2_MIN  (default 0.10)

        An R² below 0.10 means the Gaussian-smoothed decay shows essentially no
        exponential character. In this case:
          • τ_ms is set to −1 (invalid sentinel) by _fit_decay_segment.
          • decay_start and decay_end are placed heuristically (fallback), not
            from the slope criterion.
          • Every feature that reads the decay window (asymmetry_integral,
            fall_time_ms, post_SNR, ZCR_post, centroid_shift_hz) is an artefact
            of the fallback placement rather than of the actual signal shape.
        Feeding such a vector to the SVM produces an unpredictable score.

    Gate 2 — in-distribution SPR:
        SPR < STAGE2_SPR_MAX  (default 100)

        SPR ≥ 100 means the spectrum is dominated by a single tonal component —
        extremely narrowband. This region was never shown to the SVM during
        training (noise pre-filter NOISE_FILTER_SPR_MAX = 100 in train_svm.py
        removed all such samples from BOTH training classes), so the model's
        prediction there is out-of-distribution and undefined. The maximum SPR
        observed in the training click set was ≈ 34, making this a generous
        safety margin rather than a tight discriminant.

    All other features (peak_SNR, FPE_hz, ZCR, kurtosis, …) are left entirely
    to the SVM. They vary continuously and the SVM boundary in those dimensions
    was learned from labelled data.

    Parameters
    ----------
    candidates : list of dict
        Stage 1 survivors with features already attached (i.e. each dict
        contains 'R2' and 'SPR' keys produced by compute_features_v5).

    Returns
    -------
    (survivors, n_rejected) : (list of dict, int)
        survivors   — candidates that passed both gates.
        n_rejected  — total candidates dropped (useful for UI statistics).
    """
    survivors  = []
    n_rejected = 0

    for cand in candidates:
        if _stage2_reason(cand) == STAGE_OK:
            survivors.append(cand)
        else:
            n_rejected += 1

    return survivors, n_rejected


def _stage2_reason(cand: dict) -> str:
    """
    Which Stage 2 gate rejects this candidate, if any.

    The single source of truth for the Stage 2 rule. run_stage2_v5 uses it to
    drop rejects; run_stages234_annotated uses it to *label* them. Keeping both
    on one implementation is what guarantees the survivor-only path and the
    annotated path can never disagree about the same candidate.

    Returns
    -------
    str
        STAGE_OK ('') if every gate passes, otherwise STAGE_BLOCKED_R2 or
        STAGE_BLOCKED_SPR — the first gate that rejects it, the fit gates first.
    """
    # Gate 1a: R² < 0.10 → decay fit is invalid; downstream features
    # that depend on decay_start / decay_end are unreliable.
    # _fit_decay_segment also returns R2=0.0 for degenerate windows
    # (too short or near-zero denominator) — those fail here too.
    if cand.get('R2', 0.0) < STAGE2_R2_MIN:
        return STAGE_BLOCKED_R2

    # Gate 1b: τ ≤ 0 → no decay was fitted at all. _fit_decay_segment returns the
    # −1 sentinel, and every decay-window feature (fall_time_ms, post_SNR, ZCR_post,
    # asymmetry_integral, centroid_shift_hz) is then an artefact of the fallback
    # window placement rather than of the signal. Same failure as Gate 1a — an
    # unusable exponential fit — so it carries the same verdict.
    if cand.get('tau_ms', -1.0) <= STAGE2_TAU_MIN:
        return STAGE_BLOCKED_R2

    # Gate 2: SPR ≥ 100 → extremely tonal signal, out-of-distribution
    # for the SVM (never seen in training for either class).
    if cand.get('SPR', 0.0) >= STAGE2_SPR_MAX:
        return STAGE_BLOCKED_SPR

    return STAGE_OK


# =============================================================================
# STAGE 3 — SVM classification  (§12)
# =============================================================================

def run_stage3_v5(candidates: list, svm_model: dict) -> tuple:
    """
    Stage 3 of the v5 click detection pipeline: SVM classification  (§12).

    Feeds the feature vector of every Stage 2 survivor into the pre-trained
    sklearn Pipeline (SimpleImputer → StandardScaler → SVC with Platt scaling)
    and applies the optimised decision threshold to obtain the binary prediction.

    The feature order and the decision threshold are both read from the model
    dict (loaded once by load_svm_model). The pipeline module never hardcodes
    which features are used — the model is the single source of truth.

    Thread-safety note:
        sklearn SVC.predict_proba at inference time uses the libsvm C extension
        for the kernel evaluations (no BLAS/LAPACK) and simple scalar arithmetic
        for the Platt sigmoid. StandardScaler.transform is pure numpy. This
        function is therefore safe to call from a QThread on macOS.

    Parameters
    ----------
    candidates : list of dict
        Stage 2 survivors. Each dict must contain every feature name listed in
        svm_model['features']. Any feature value that is NaN (e.g. from a
        degenerate frame) is handled by the SimpleImputer in the pipeline, which
        fills it with the training-set mean — safe but worth monitoring.

    svm_model : dict
        Dict returned by load_svm_model. Must contain:
            'pipeline'  : fitted sklearn Pipeline
            'threshold' : float — decision threshold (e.g. 0.220)
            'features'  : list[str] — ordered feature names

    Returns
    -------
    (detections, n_rejected) : (list of dict, int)
        detections  — candidates classified as clicks (svm_probability ≥ threshold).
                      Two new keys are added to each surviving dict:
                          'svm_probability' : float  (0.0–1.0)
                          'svm_prediction'  : int    (always 1 here)
        n_rejected  — candidates the SVM classified as noise (for the UI stats panel).
    """
    if not candidates:
        return [], 0

    threshold = float(svm_model['threshold'])
    proba     = _stage3_scores(candidates, svm_model)

    # ── Apply threshold and collect survivors ────────────────────────────────
    detections = []
    n_rejected = 0

    for cand, p in zip(candidates, proba):
        if p >= threshold:
            # Deep-copy the dict to avoid mutating Stage 2 output in place,
            # then annotate with the SVM result for downstream use and logging.
            det = dict(cand)
            det['svm_probability'] = float(p)
            det['svm_prediction']  = 1
            detections.append(det)
        else:
            n_rejected += 1

    return detections, n_rejected


def _stage3_scores(candidates: list, svm_model: dict) -> np.ndarray:
    """
    P(click) for EVERY candidate — including the ones the threshold will reject.

    run_stage3_v5 keeps only the survivors, so the probability of a rejected
    candidate is lost. That number is the most informative diagnostic there is
    ("how close was this to being a click?"), so the scoring step is factored
    out here where both the survivor-only path and run_stages234_annotated can
    reach it.

    Parameters
    ----------
    candidates : list of dict
        Stage 2 survivors. Each must carry every feature in svm_model['features'];
        a missing one becomes NaN and is filled by the Pipeline's SimpleImputer.
    svm_model : dict
        From load_svm_model. Only 'pipeline' and 'features' are read — the
        threshold is applied by the caller, so a caller may override it.

    Returns
    -------
    np.ndarray, shape (len(candidates),)
        P(class=1) = P(click), in the same order as `candidates`.
    """
    feat_names = svm_model['features']   # authoritative order from training

    # ── Build feature matrix (n_candidates × n_features) ────────────────────
    # Values are extracted in the exact column order the SVM was trained on.
    # float64 is used throughout: the SVC C extension requires it, and it avoids
    # the silent overflow that affected float32 on large peak_SNR values.
    X = np.array(
        [[cand.get(f, np.nan) for f in feat_names] for cand in candidates],
        dtype=np.float64,
    )

    # Column 1 of predict_proba = P(class=1) = P(click).
    # The pipeline internally applies SimpleImputer then StandardScaler before
    # the SVC — no manual scaling is needed here.
    return svm_model['pipeline'].predict_proba(X)[:, 1]


# =============================================================================
# STAGE 4 — Deduplication  (§13)
# =============================================================================

def _det_peak_abs(det: dict) -> int:
    """Absolute peak sample of a detection, with a frame-index fallback.

    Every candidate built through resolve_click / click_event_key carries
    'peak_abs'. The fallback keeps run_stage4_v5 usable for any legacy dict that
    only has 'frame_idx' (treated as the frame start).
    """
    pa = det.get('peak_abs')
    if pa is not None:
        return int(pa)
    return int(det['frame_idx']) * FFT_SIZE


def run_stage4_v5(detections: list) -> list:
    """
    Stage 4 of the v5 click detection pipeline: deduplication  (§13).

    A single physical cavitation click produces Stage 1 candidates in more than
    one consecutive frame when it straddles a frame boundary — Stage 1 detects it
    in the frame where it rises and again in the frame where it decays.

    Deduplication now groups by the click's ABSOLUTE peak sample (peak_abs), not
    by frame-index proximity. Once every candidate resolves its peak on the
    stitched prev|curr|next trace (resolve_click), the two candidates of one
    straddling click carry an integer-identical peak_abs, so they land in the
    same group by construction. This makes identity deterministic and
    model-independent — the old logic broke ties by svm_probability, so which
    frame "was" the click shifted with every retrained model.

    Algorithm:
        1. Sort detections by peak_abs.
        2. Start a new group whenever the peak_abs gap exceeds
           PEAK_MATCH_SAMPLES (a small margin for per-frame noise jitter, NOT a
           multi-millisecond window — two genuinely distinct clicks are now kept
           separate instead of being merged).
        3. From each group keep the CANONICAL representative: the candidate whose
           own frame owns the peak (frame_idx == canonical_frame_idx), because
           its context has the peak centred and therefore the cleanest envelope.
           If none/several qualify, fall back to highest svm_probability, then
           earliest frame.

    Parameters
    ----------
    detections : list of dict
        Stage 3 survivors. Each dict should carry 'peak_abs' and
        'canonical_frame_idx' (from click_event_key); 'frame_idx' and
        'svm_probability' are used for representative selection.

    Returns
    -------
    list of dict
        Deduplicated detections, sorted by peak_abs. The returned dicts are the
        same objects as in the input (no copying), so all keys are preserved.
    """
    if len(detections) <= 1:
        return list(detections)

    # ── Step 1: sort by absolute peak sample ─────────────────────────────────
    sorted_dets = sorted(detections, key=_det_peak_abs)

    # ── Step 2: group by peak-sample proximity ───────────────────────────────
    groups  = []
    current = [sorted_dets[0]]
    for det in sorted_dets[1:]:
        if _det_peak_abs(det) - _det_peak_abs(current[-1]) <= PEAK_MATCH_SAMPLES:
            current.append(det)   # same physical click → extend current group
        else:
            groups.append(current)
            current = [det]
    groups.append(current)

    # ── Step 3: keep the canonical representative from each group ─────────────
    def pick(group):
        canon = group[0].get('canonical_frame_idx')
        if canon is None:
            canon = _det_peak_abs(group[0]) // FFT_SIZE
        owners = [d for d in group if d.get('frame_idx') == canon]
        pool = owners if owners else group
        # highest svm_probability, ties broken toward the earliest frame
        return max(pool, key=lambda d: (d.get('svm_probability', 0.0),
                                        -int(d.get('frame_idx', 0))))

    return [pick(g) for g in groups]


# =============================================================================
# STAGES 2-4 — annotated run (keeps every candidate)
# =============================================================================

def run_stages234_annotated(
    candidates: list,
    svm_model: dict,
    threshold: float = None,
) -> list:
    """
    Run Stages 2, 3 and 4 while keeping EVERY input candidate.

    run_stage2_v5 / run_stage3_v5 / run_stage4_v5 each return survivors plus a
    reject count, which is what a live detector wants. An analyst wants the
    opposite: the full census, with a verdict attached to each candidate saying
    which stage rejected it and how confident the SVM was. That is this function.

    It is the in-app equivalent of src/ml/evaluate_candidates.py, and produces
    the same three columns with the same names and semantics — so a CSV exported
    from the GUI and one produced by the offline CLI are interchangeable.

    Three keys are added to every returned candidate:

        stage_blocked   : str   — STAGE_OK ('') if it survived all four stages,
                                  otherwise the label of the first stage that
                                  rejected it (STAGE_BLOCKED_R2 / _SPR / _SVM /
                                  _DEDUP).
        svm_probability : float — P(click) from the SVM.
                                  None if Stage 2 blocked it (it never reached
                                  the model, so no score exists).
        svm_prediction  : int   — 1 if P ≥ threshold, else 0.
                                  None if Stage 2 blocked it.

    Deduplication is applied within the given list, so callers must pass the
    candidates of ONE recording at a time — otherwise clicks at similar frame
    indices in different recordings would be merged.

    Parameters
    ----------
    candidates : list of dict
        Stage 1 survivors with features attached (compute_features_v5 output
        merged in). Not mutated — each returned dict is a shallow copy.
    svm_model : dict
        From load_svm_model.
    threshold : float, optional
        Overrides svm_model['threshold']. Lowering it trades precision for
        recall without retraining; this is what the UI's threshold control does.

    Returns
    -------
    list of dict
        One dict per input candidate, same order as the input, each annotated.
    """
    # Shallow-copy so the caller's dicts are never mutated, and pre-seed the
    # three annotation keys so every returned dict has an identical shape —
    # a CSV writer can then rely on them existing.
    annotated = []
    for cand in candidates:
        row = dict(cand)
        row['stage_blocked']   = _stage2_reason(row)
        row['svm_probability'] = None
        row['svm_prediction']  = None
        annotated.append(row)

    # ── Stage 3 — score everything that cleared the Stage 2 gates ────────────
    survivors = [r for r in annotated if r['stage_blocked'] == STAGE_OK]
    if not survivors:
        return annotated

    thr   = float(svm_model['threshold']) if threshold is None else float(threshold)
    proba = _stage3_scores(survivors, svm_model)

    for row, p in zip(survivors, proba):
        row['svm_probability'] = float(p)
        row['svm_prediction']  = int(p >= thr)
        if row['svm_prediction'] == 0:
            row['stage_blocked'] = STAGE_BLOCKED_SVM

    # ── Stage 4 — deduplicate the confirmed clicks ───────────────────────────
    # run_stage4_v5 returns the *same* dict objects it was given (its docstring
    # guarantees no copying), so the survivors are identifiable by identity and
    # everything else in `passers` is a duplicate that lost to a higher score.
    passers = [r for r in survivors if r['stage_blocked'] == STAGE_OK]
    if len(passers) > 1:
        kept_ids = {id(r) for r in run_stage4_v5(passers)}
        for row in passers:
            if id(row) not in kept_ids:
                row['stage_blocked'] = STAGE_BLOCKED_DEDUP

    return annotated


def stage_summary(annotated: list) -> dict:
    """
    Funnel counts for a run_stages234_annotated() result — for UI stats panels.

    Returns
    -------
    dict
        {'total', 'Stage2_R2', 'Stage2_SPR', 'Stage3_SVM', 'Stage4_dedup',
         'confirmed'}. The four blocked counts plus 'confirmed' sum to 'total'.
    """
    counts = {
        'total':               len(annotated),
        STAGE_BLOCKED_R2:      0,
        STAGE_BLOCKED_SPR:     0,
        STAGE_BLOCKED_SVM:     0,
        STAGE_BLOCKED_DEDUP:   0,
        'confirmed':           0,
    }

    for row in annotated:
        verdict = row.get('stage_blocked', STAGE_OK)
        if verdict == STAGE_OK:
            counts['confirmed'] += 1
        elif verdict in counts:
            counts[verdict] += 1

    return counts
