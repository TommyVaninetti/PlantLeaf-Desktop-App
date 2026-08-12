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
Channel model — bringing Khait/Dryad recordings into PlantLeaf's instrument space.

Two independent steps, both of which have been validated numerically (see
`test_scripts/verify_channel_model.py`):

1. Rate conversion, 500 kHz -> 200 kHz (section 5 of the spec).
2. Microphone colorization, flat-response -> SPU0410LR5H-QB (section 4).


Which space do we colorize *into*?
----------------------------------
This is the subtle part, and getting it wrong silently corrupts every
timing-derived feature. There are two candidate target spaces.

`_normalize_fft()` in click_pipeline_v5 multiplies the **magnitude only** by a
real positive gain |H_mic|^-0.5. Reconstruction then pairs that corrected
magnitude with the *firmware's measured phase*, which still carries the full mic
phase. So the post-normalization space a live click lands in is:

    magnitude = |P| * |H_mic|^0.5        <- half the coloration removed
    phase     = arg P + arg H_mic        <- phase untouched by normalization

i.e. half the magnitude coloration but ALL of the phase coloration. A colorizing
filter aiming at that space would need

    H = |H_mic|^0.5 * exp(1j * angle(minphase(|H_mic|)))

which is emphatically NOT `minphase(|H_mic| ** 0.5)` — that applies only half the
phase, an error of 0.73 rad in-band against a microphone whose in-band group
delay varies by ~50 us (10 samples at 200 kHz). That is easily enough to distort
rise_time_ms, fall_time_ms and asymmetry_integral.

This module sidesteps the whole question by targeting the OTHER space: **raw
recorded space**, the pre-normalization domain the firmware actually stores in a
.paudio file. A live recording there is simply

    FFT{ p(t) * h_mic(t) }  ->  magnitude |P|*|H_mic| , phase arg P + arg H_mic

so the colorizing filter is just the full minimum-phase mic response, magnitude
and phase together — `mic_filter(..., magnitude_exponent=1.0)`. The injected
click is then written to a .paudio and the pipeline applies its own
normalization exactly once, exactly as it does for a genuine recording. The
half-magnitude/full-phase asymmetry above then arises on its own instead of
being hand-constructed, and there is no way to double-apply or half-apply it.

`magnitude_exponent=0.5` is retained for callers that genuinely need the
post-normalization space (e.g. comparing against an already-normalized
spectrum); it produces the correct half-magnitude/full-phase filter, not the
naive one.


Minimum phase
-------------
Only the SPU0410's magnitude response is documented (8 datasheet points, no
phase or group-delay data — verified absent from the full datasheet). Phase
feeds straight into iFFT -> Hilbert envelope -> tau, rise/fall time, asymmetry,
so it cannot simply be ignored. A zero-phase filter would be non-causal and
smear the onset backwards in time, which is worse than wrong — it is
unphysical. We therefore reconstruct the minimum-phase response from the
magnitude via the real cepstrum, the standard resolution when phase calibration
data is unavailable (Terzic et al. take the same approach, via Bode's
gain-phase relation).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

# ─────────────────────────────────────────────────────────────────────────────
# Hardware constants
# ─────────────────────────────────────────────────────────────────────────────

DRYAD_FS = 500_000          # Avisoft UltraSoundGate 1216H sampling rate [sps]
PLANTLEAF_FS = 200_000      # PlantLeaf STM32 sampling rate [sps]

# 200/500 = 2/5 exactly, so the conversion is a clean polyphase rational resample.
RESAMPLE_UP = 2
RESAMPLE_DOWN = 5

# Knowles SPU0410LR5H-QB datasheet response. Identical to the arrays in
# click_pipeline_v5 (_MIC_FREQ_HZ / _MIC_RESP_DB); duplicated rather than
# imported so this module stays usable with no pipeline present, and asserted
# equal against the pipeline's copy in verify_channel_model.py.
MIC_FREQ_HZ = np.array([20, 25, 30, 40, 50, 60, 70, 80], dtype=np.float64) * 1000.0
MIC_RESP_DB = np.array([8.0, 10.5, 6.0, -2.0, -6.0, -7.0, -6.0, -4.0], dtype=np.float64)

# Spec section 3.2 / 16: the Elecrow board's LM321 gain stage response has never been
# measured, so the channel model uses the mic response alone. Flip this only
# once a calibrated tone/sweep measurement exists — at which point the only
# change needed is to cascade the measured response into `mic_response_db`.
BOARD_RESPONSE_MEASURED = False


# ─────────────────────────────────────────────────────────────────────────────
# Resampling
# ─────────────────────────────────────────────────────────────────────────────

def resample_500k_to_200k(x: np.ndarray) -> np.ndarray:
    """
    Convert a 500 kHz Dryad clip to PlantLeaf's 200 kHz rate.

    Polyphase resampling with scipy's default Kaiser anti-alias filter.
    Measured behaviour of that filter (verify_channel_model.py):

        20-80 kHz  (the analysis band) : +-0.02 dB  -- flat, no feature impact
        85/90/95 kHz                   : -0.05 / -0.66 / -2.4 dB (transition)
        >=120 kHz                      : <= -55 dB

    Only input content above 120 kHz can alias *into* 20-80 kHz (it folds to
    |f - 200 kHz|), and that is attenuated by at least 55 dB, so in-band aliasing
    is negligible. Discarding everything above the new 100 kHz Nyquist is correct
    emulation of PlantLeaf's hardware limit, not data loss: measured across the
    Dryad corpus only 4.8-13.6 % of clip energy sits above 100 kHz, while 79-91 %
    is already inside 20-80 kHz.

    A 1001-sample clip (2.002 ms) becomes 401 samples (2.005 ms).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"expected a 1-D signal, got shape {x.shape}")
    return resample_poly(x, up=RESAMPLE_UP, down=RESAMPLE_DOWN)


# ─────────────────────────────────────────────────────────────────────────────
# Minimum-phase reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def minimum_phase_from_magnitude(mag_full: np.ndarray, n_fft: int | None = None) -> np.ndarray:
    """
    Build the minimum-phase complex response having the given magnitude.

    Real-cepstrum method: take log|H|, transform to the cepstral domain, zero the
    anti-causal half (doubling the causal half to conserve energy), and transform
    back. Zeroing the second half is precisely what enforces causality, and hence
    minimum phase.

    Parameters
    ----------
    mag_full : np.ndarray
        Magnitude over the FULL n-point frequency axis, in `np.fft.fftfreq`
        order and Hermitian-symmetric (`mag_full[n-k] == mag_full[k]`). Build it
        with `mic_response_magnitude()`, or by mirroring a one-sided curve.
    n_fft : int, optional
        Length of `mag_full`. Validated if given.

    Returns
    -------
    np.ndarray
        Complex spectrum, length n, with `abs(result) == mag_full` to machine
        precision and minimum phase.

    Notes
    -----
    Do NOT pass a one-sided (n/2+1) magnitude array, and do NOT reach for
    `scipy.signal.hilbert()` on the log-magnitude as a shortcut. `hilbert()`'s
    implicit FFT length must match the full mirrored spectrum; applied to a
    truncated one-sided array it silently produces a non-causal, oscillating
    result that looks plausible and is wrong.

    Validation: against a genuine discrete-time minimum-phase FIR (a real
    polynomial with all zeros strictly inside the unit circle) this recovers the
    complex response to 5.2e-15 and the impulse response to 4.4e-16.

    A continuous-time prototype such as 1/(1 + j*f/f_c) is NOT a valid ground
    truth for this function: it has non-zero phase at Nyquist, which no real
    discrete-time minimum-phase system can have (the response must be real at
    both DC and Nyquist). Checking against one produces a spurious ~1.28 rad
    discrepancy concentrated near Nyquist that looks like a bug in this code but
    is a defect in the reference.
    """
    mag_full = np.asarray(mag_full, dtype=np.float64)
    n = len(mag_full)
    if n_fft is not None and n_fft != n:
        raise ValueError(f"n_fft={n_fft} does not match len(mag_full)={n}")
    if n < 4 or n % 2 != 0:
        raise ValueError(f"need an even FFT length >= 4, got {n}")

    # Floor the magnitude: log(0) is -inf and would poison the whole cepstrum.
    log_mag = np.log(np.maximum(mag_full, 1e-12))
    cepstrum = np.fft.ifft(log_mag).real

    c_min = np.zeros(n, dtype=np.float64)
    c_min[0] = cepstrum[0]                  # DC term: unchanged
    c_min[1:n // 2] = 2.0 * cepstrum[1:n // 2]   # fold the anti-causal half in
    c_min[n // 2] = cepstrum[n // 2]        # Nyquist: unchanged, NOT doubled
    # c_min[n//2 + 1:] stays zero -- this is what enforces causality.

    return np.exp(np.fft.fft(c_min))


# ─────────────────────────────────────────────────────────────────────────────
# Microphone response
# ─────────────────────────────────────────────────────────────────────────────

def mic_response_db(freq_hz: np.ndarray) -> np.ndarray:
    """
    SPU0410 datasheet response in dB at the given frequencies (piecewise linear).

    `np.interp` clamps outside the tabulated 20-80 kHz range, so this reports
    +8 dB at DC and -4 dB above 80 kHz. That extrapolation is not physical, but
    it is harmless in this pipeline because the frame emulator transmits only
    bins 51-204 (20-80 kHz) and discards everything else. Any caller that uses
    this response outside the analysis band must band-mask AFTER colorizing, not
    before.
    """
    return np.interp(np.abs(np.asarray(freq_hz, dtype=np.float64)), MIC_FREQ_HZ, MIC_RESP_DB)


def mic_response_magnitude(n_fft: int, fs: int = PLANTLEAF_FS) -> np.ndarray:
    """
    |H_mic| over the full n-point FFT axis, Hermitian-symmetric by construction.

    Uses `np.fft.fftfreq` ordering and takes `abs(f)`, so bin k and bin n-k get
    the same magnitude — a requirement for `minimum_phase_from_magnitude` and for
    the filtered signal coming back real.
    """
    freqs = np.fft.fftfreq(n_fft, d=1.0 / fs)
    return 10.0 ** (mic_response_db(freqs) / 20.0)


def mic_filter(n_fft: int, fs: int = PLANTLEAF_FS, magnitude_exponent: float = 1.0) -> np.ndarray:
    """
    The complex colorizing filter to apply to flat-response (Dryad) data.

        H(f) = |H_mic(f)| ** magnitude_exponent  *  exp(1j * angle(minphase(|H_mic|)))

    The phase term always carries the FULL mic phase regardless of
    `magnitude_exponent`, because in PlantLeaf's chain magnitude and phase are
    attenuated independently: normalization scales magnitude by a real positive
    gain and never touches phase. See the module docstring.

    Parameters
    ----------
    magnitude_exponent : float
        1.0 (default) targets **raw recorded space** — the pre-normalization
        domain stored in a .paudio. This is what the injector uses: write the
        mixture to a .paudio and let the pipeline normalize once, as it does for
        real recordings.

        0.5 targets **post-normalization space**, for callers comparing directly
        against an already-normalized spectrum. Note this is deliberately not
        `minphase(|H_mic| ** 0.5)`, which would apply only half the phase.

    Returns
    -------
    np.ndarray
        Complex, length n_fft, Hermitian-symmetric (real at DC and Nyquist), so
        filtering a real signal with it returns a real signal.
    """
    mag = mic_response_magnitude(n_fft, fs)
    h_min = minimum_phase_from_magnitude(mag, n_fft)

    filt = (mag ** magnitude_exponent) * np.exp(1j * np.angle(h_min))

    # Force exact Hermitian symmetry at the two self-conjugate bins. The cepstral
    # round trip leaves ~1e-17 imaginary residue there; left alone it produces a
    # microscopic imaginary part in the filtered signal that then gets silently
    # discarded by .real, which would hide a genuine symmetry bug if one appeared.
    filt[0] = np.abs(filt[0])
    if n_fft % 2 == 0:
        filt[n_fft // 2] = np.abs(filt[n_fft // 2])

    return filt


def colorize(x: np.ndarray, fs: int = PLANTLEAF_FS, magnitude_exponent: float = 1.0) -> np.ndarray:
    """
    Apply the microphone channel model to a flat-response time-domain signal.

    Linear-convolution caveat: this multiplies the signal's own n-point FFT by
    the filter, which is a CIRCULAR convolution — energy pushed past the end of
    the array wraps around to the start. The mic's minimum-phase impulse response
    concentrates 99 % of its energy within 12 samples (60 us), and Dryad clips
    open and close in near-silence with the click centred ~200 samples from
    either edge, so the wrapped contribution lands in noise and is negligible.
    The injector nonetheless colorizes the clip before placing it in the bed, so
    any wrap stays inside the template rather than smearing into the bed.

    Returns a real signal of the same length as `x`.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"expected a 1-D signal, got shape {x.shape}")

    n = len(x)
    if n % 2 != 0:
        # minimum_phase_from_magnitude needs an even length for the Nyquist bin
        # to exist. Colorize on n+1 -> n+1 is odd, so pad by one instead and trim.
        padded = np.append(x, 0.0)
        return colorize(padded, fs, magnitude_exponent)[:n]

    filt = mic_filter(n, fs, magnitude_exponent)
    return np.fft.ifft(np.fft.fft(x) * filt).real


def prepare_dryad_click(x_500k: np.ndarray, magnitude_exponent: float = 1.0) -> np.ndarray:
    """
    Full Dryad -> PlantLeaf channel model: resample, then colorize.

    Returns a 200 kHz signal in raw recorded space (by default), ready to be
    scaled by the injection gain and added to a noise bed.
    """
    return colorize(resample_500k_to_200k(x_500k), PLANTLEAF_FS, magnitude_exponent)
