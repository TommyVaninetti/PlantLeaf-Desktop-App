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
Spectral analysis of an arbitrary segment of a signal.

A GENERIC, domain-independent module: it takes `(signal, fs)` and nothing else.
It does not import Qt, does not import scipy, and knows nothing about clicks,
frames or microphones. It is used today by the audio "FFT of Region" dialog, but
it is written to be reused as-is for voltage analysis.

  - numpy only (same contract as click_pipeline_v5: scipy/BLAS can segfault
    inside QThreads on macOS, so the DSP chain stays pure numpy);
  - no imports from click_pipeline_v5 → no audio-specific constants.


THE KEY IDEA: RESOLUTION IS NOT THE NUMBER OF POINTS
=====================================================

The segment length `n_seg` and the transform length `n_fft` are two different
things and must be kept apart:

  * `n_seg` is fixed by PHYSICS (e.g. the decay region of a click). It must NOT
    be rounded to a power of two — that would change what is being analysed.

  * `n_fft` is a DISPLAY parameter. Zero-padding to a power of two makes the
    curve smooth and refines the peak-frequency readout.

Zero-padding **does not add resolution**: it interpolates the DTFT, it does not
separate two tones the window could not already separate. The true resolution is

    Δf = NENBW(w) · fs / n_seg

and it is a hard physical limit. That is why `Spectrum.enbw_hz` depends on
`n_seg` and NOT on `n_fft` — and why it is the number to show the user. Otherwise
a 4096-point curve looks like it has 48 Hz of resolution when it really has
several kHz.


AMPLITUDE SCALE
===============

The default scaling is 'amplitude': a sinusoid of amplitude A inside the region
produces a peak of exactly A, independently of `n_seg` and `n_fft`. This makes
regions of different lengths directly comparable, and puts the spectrum on the
same scale as the time-domain trace (mV, after the iFFT scale correction — see
docs/fft_and_ifft/IFFT_AMPLITUDE_SCALE_FIX.md).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Finestre
# ─────────────────────────────────────────────────────────────────────────────

WINDOWS = ('rectangular', 'hann', 'hamming', 'blackman', 'tukey')

DEFAULT_WINDOW = 'tukey'
DEFAULT_ALPHA = 0.25

#: Sotto questa lunghezza uno spettro non ha alcun significato utile.
MIN_SEGMENT_SAMPLES = 8


def next_pow2(n: int) -> int:
    """Smallest power of two >= n (>= 1)."""
    n = int(max(1, n))
    return 1 << (n - 1).bit_length()


def make_window(name: str = DEFAULT_WINDOW,
                n: int = 0,
                alpha: float = DEFAULT_ALPHA,
                left_only: bool = False) -> np.ndarray:
    """
    Build a length-`n` analysis window.

    Parameters
    ----------
    name : one of WINDOWS.
    alpha : Tukey taper fraction — the total fraction of the window that is
        tapered (alpha/2 at each end). Ignored by the other windows.
    left_only : make the taper ONE-SIDED (leading edge only).

        This exists for decay segments. The fitted region of a click *starts* at
        maximum amplitude — a hard discontinuity — and *ends* by decaying into
        the noise on its own. The right edge is therefore already tapered by the
        physics; tapering it again just throws away signal. A symmetric Hann is
        actively wrong here: it zeroes the onset, which is exactly where the
        click's energy lives.

    Notes
    -----
    The raised-cosine ramp 0.5*(1 - cos(pi*i/L)) is the same taper idiom used in
    click_pipeline_v5 (the frequency-domain Tukey in reconstruct_frame_v5 and the
    Gibbs fade in suppress_edge_artifacts).
    """
    n = int(n)
    if n <= 0:
        return np.ones(0, dtype=np.float64)
    if n == 1:
        return np.ones(1, dtype=np.float64)

    key = (name or DEFAULT_WINDOW).lower()

    if key == 'rectangular':
        return np.ones(n, dtype=np.float64)
    if key == 'hann':
        w = np.hanning(n)
    elif key == 'hamming':
        w = np.hamming(n)
    elif key == 'blackman':
        w = np.blackman(n)
    elif key == 'tukey':
        w = _tukey(n, alpha)
    else:
        raise ValueError(f"unknown window {name!r}; expected one of {WINDOWS}")

    if left_only:
        # Keep the rising half, flatten everything past the peak of the taper.
        peak = int(np.argmax(w))
        if peak > 0:
            w = w.copy()
            w[peak:] = 1.0
    return np.asarray(w, dtype=np.float64)


def _tukey(n: int, alpha: float) -> np.ndarray:
    """Tukey (tapered-cosine) window: flat top, raised-cosine ends."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0.0:
        return np.ones(n, dtype=np.float64)
    if alpha >= 1.0:
        return np.hanning(n)

    w = np.ones(n, dtype=np.float64)
    taper = int(np.floor(alpha * (n - 1) / 2.0))
    if taper < 1:
        return w
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(taper + 1) / taper))
    w[:taper + 1] = ramp
    w[n - taper - 1:] = ramp[::-1]
    return w


def nenbw(w: np.ndarray) -> float:
    """
    Normalized Equivalent Noise Bandwidth, in BINS.

        NENBW = n * sum(w^2) / (sum(w))^2

    Rectangular = 1.0 (the best possible). Everything else is worse; that is the
    price of suppressing leakage. Multiply by fs/n to get the true resolution in
    Hz.
    """
    w = np.asarray(w, dtype=np.float64)
    s = float(np.sum(w))
    if len(w) == 0 or abs(s) < 1e-30:
        return 0.0
    return float(len(w) * np.sum(w ** 2) / (s ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# Spettro
# ─────────────────────────────────────────────────────────────────────────────

SCALINGS = ('amplitude', 'psd', 'magnitude')

_UNITS = {
    'amplitude': 'V',
    'psd': 'V²/Hz',
    'magnitude': '',
}


@dataclass
class Spectrum:
    """One-sided spectrum of a finite segment."""
    freqs: np.ndarray      #: Hz
    mags: np.ndarray       #: in `unit`
    scaling: str           #: 'amplitude' | 'psd' | 'magnitude'
    unit: str              #: 'V' | 'V²/Hz' | ''
    n_seg: int             #: segment length  → sets the RESOLUTION
    n_fft: int             #: transform length → interpolation only
    fs: float
    window: str
    enbw_hz: float         #: TRUE resolution = nenbw(w) * fs / n_seg

    @property
    def duration_s(self) -> float:
        return self.n_seg / self.fs if self.fs else 0.0

    @property
    def bin_spacing_hz(self) -> float:
        """Spacing of the plotted points. NOT the resolution — see enbw_hz."""
        return self.fs / self.n_fft if self.n_fft else 0.0


def compute_spectrum(signal: np.ndarray,
                     fs: float,
                     *,
                     window: str = DEFAULT_WINDOW,
                     alpha: float = DEFAULT_ALPHA,
                     left_only: bool = False,
                     n_fft: Optional[int] = None,
                     scaling: str = 'amplitude',
                     detrend: bool = False) -> Spectrum:
    """
    One-sided spectrum of `signal`.

    Parameters
    ----------
    n_fft : transform length. None → `default_nfft(len(signal))`. Values smaller
        than the segment are raised to the segment length (never truncate the
        data). Zero-padding interpolates; it does NOT improve resolution.
    scaling :
        'amplitude' — 2·|X|/sum(w).  A sinusoid of amplitude A peaks at A. [V]
                      DC and Nyquist bins are NOT doubled.
        'psd'       — 2·|X|²/(fs·sum(w²)).  Welch scaling. [V²/Hz]
        'magnitude' — raw |X|, no compensation. Only for comparing against a
                      raw FFT on the same n_fft grid.
    detrend : subtract the segment mean first. A no-op for band-limited audio
        (there is no DC), but useful for voltage.
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    n_seg = len(x)

    if scaling not in SCALINGS:
        raise ValueError(f"unknown scaling {scaling!r}; expected one of {SCALINGS}")
    if n_seg == 0 or fs <= 0:
        return Spectrum(np.zeros(0), np.zeros(0), scaling, _UNITS[scaling],
                        0, 0, float(fs), window, 0.0)

    if detrend:
        x = x - float(np.mean(x))

    w = make_window(window, n_seg, alpha, left_only)
    xw = x * w

    if n_fft is None:
        n_fft = default_nfft(n_seg)
    n_fft = max(int(n_fft), n_seg)      # never truncate

    X = np.fft.rfft(xw, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)

    sum_w = float(np.sum(w))
    sum_w2 = float(np.sum(w ** 2))

    if scaling == 'amplitude':
        mags = np.abs(X) * (2.0 / sum_w if sum_w > 1e-30 else 0.0)
        # DC and Nyquist appear once in a one-sided spectrum, not twice.
        if len(mags) > 0:
            mags[0] /= 2.0
        if n_fft % 2 == 0 and len(mags) > 1:
            mags[-1] /= 2.0
    elif scaling == 'psd':
        denom = fs * sum_w2
        mags = (np.abs(X) ** 2) * (2.0 / denom if denom > 1e-30 else 0.0)
        if len(mags) > 0:
            mags[0] /= 2.0
        if n_fft % 2 == 0 and len(mags) > 1:
            mags[-1] /= 2.0
    else:                                # 'magnitude'
        mags = np.abs(X)

    return Spectrum(
        freqs=freqs,
        mags=mags,
        scaling=scaling,
        unit=_UNITS[scaling],
        n_seg=n_seg,
        n_fft=n_fft,
        fs=float(fs),
        window=window,
        enbw_hz=nenbw(w) * fs / n_seg,   # depends on n_seg, NOT on n_fft
    )


def default_nfft(n_seg: int, floor: int = 512, cap: int = 8192) -> int:
    """
    A sensible zero-padded transform length: enough interpolation for a smooth
    curve, without pretending to a resolution the segment cannot deliver.
    """
    return int(np.clip(next_pow2(4 * max(1, int(n_seg))), floor, cap))


def to_db(mags: np.ndarray,
          ref: Optional[float] = None,
          floor_db: float = -120.0) -> np.ndarray:
    """
    20·log10(mags / ref), floored. `ref` defaults to the peak of `mags`, i.e.
    dB relative to the maximum (0 dB at the peak).
    """
    m = np.asarray(mags, dtype=np.float64)
    if len(m) == 0:
        return m
    if ref is None:
        ref = float(np.max(np.abs(m)))
    if not np.isfinite(ref) or ref <= 0:
        return np.full_like(m, floor_db)
    with np.errstate(divide='ignore', invalid='ignore'):
        db = 20.0 * np.log10(np.abs(m) / ref)
    return np.maximum(np.nan_to_num(db, neginf=floor_db), floor_db)


# ─────────────────────────────────────────────────────────────────────────────
# Descrittori
# ─────────────────────────────────────────────────────────────────────────────

def band_descriptors(spec: Spectrum,
                     band: Optional[Tuple[float, float]] = None) -> dict:
    """
    Summary numbers for a spectrum, restricted to `band` (Hz) if given.

    Mirrors the quantities click_pipeline_v5 computes on the whole-frame
    transmitted spectrum (_feat_fft_features, _segment_spectral_centroid), but
    generically and on THIS segment — which is the point of the feature: the v5
    spectral features describe the whole 2.56 ms frame, mostly noise, whereas
    these describe the click itself.

    Returns (all floats; 0.0 where undefined):
        peak_freq_hz, centroid_hz, bandwidth_rms_hz, bw_3db_hz, bw_10db_hz,
        spr, r_spectral, band_energy, total_energy
    """
    empty = dict(peak_freq_hz=0.0, centroid_hz=0.0, bandwidth_rms_hz=0.0,
                 bw_3db_hz=0.0, bw_10db_hz=0.0, spr=0.0, r_spectral=0.0,
                 band_energy=0.0, total_energy=0.0)
    if spec is None or len(spec.freqs) == 0:
        return empty

    f = spec.freqs
    m = spec.mags
    total_energy = float(np.sum(m ** 2))

    if band is not None:
        mask = (f >= band[0]) & (f <= band[1])
        if not np.any(mask):
            return empty
        f = f[mask]
        m = m[mask]

    power = m ** 2
    tot = float(np.sum(power))
    if tot <= 1e-30:
        return empty

    peak_i = int(np.argmax(power))
    peak_freq = float(f[peak_i])

    centroid = float(np.sum(f * power) / tot)
    bandwidth_rms = float(np.sqrt(max(np.sum(((f - centroid) ** 2) * power) / tot, 0.0)))

    mean_power = float(np.mean(power))
    max_power = float(power[peak_i])
    spr = max_power / mean_power if mean_power > 1e-30 else 0.0

    # E[low half] / E[high half] of the band — descriptive, like v5's R_spectral
    mid = 0.5 * (f[0] + f[-1])
    lo = power[f < mid]
    hi = power[f >= mid]
    e_lo = float(np.mean(lo)) if len(lo) else 0.0
    e_hi = float(np.mean(hi)) if len(hi) else 0.0
    r_spectral = e_lo / e_hi if e_hi > 1e-30 else 0.0

    return dict(
        peak_freq_hz=peak_freq,
        centroid_hz=centroid,
        bandwidth_rms_hz=bandwidth_rms,
        bw_3db_hz=_width_at(f, m, peak_i, 10 ** (-3.0 / 20.0)),
        bw_10db_hz=_width_at(f, m, peak_i, 10 ** (-10.0 / 20.0)),
        spr=spr,
        r_spectral=r_spectral,
        band_energy=tot,
        total_energy=total_energy,
    )


def _width_at(freqs: np.ndarray, mags: np.ndarray,
              peak_i: int, frac: float) -> float:
    """Width [Hz] of the peak where the magnitude first falls below frac*peak on
    each side. Returns 0.0 if the peak never drops that far inside the band."""
    if len(freqs) < 2:
        return 0.0
    thr = float(mags[peak_i]) * frac
    if thr <= 0:
        return 0.0

    lo = peak_i
    while lo > 0 and mags[lo] > thr:
        lo -= 1
    hi = peak_i
    n = len(mags) - 1
    while hi < n and mags[hi] > thr:
        hi += 1

    if mags[lo] > thr or mags[hi] > thr:
        return 0.0   # never crossed → the peak fills the band; width undefined
    return float(freqs[hi] - freqs[lo])


# ─────────────────────────────────────────────────────────────────────────────
# v6 — the excess spectrum E[k] and the features built on it
#
# Specification: docs/fft_and_ifft/SPECTRAL_FEATURES_v6_PROPOSAL.md
#
# The v5 spectral features (SPR, R_spectral, FPE_hz) are computed on the whole
# 2.56 ms frame, of which a click occupies 30-80 of 512 samples — so they mostly
# describe noise (§0). Worse, the noise floor they are judged against is a SCALAR,
# so nothing can ask WHERE the excess energy sits. "Broadband", the defining
# property of a cavitation click, is therefore never measured (gap G1).
#
# The fix is the excess spectrum
#
#     E[m] = max(0, P_region[m] - P_noise[m])      [V²/Hz], m over the band grid
#
# and a small set of dimensionless statistics computed on it. Everything here is
# pure numpy and signal-agnostic; the audio-specific parts (Buffer 3, the mic
# correction, the analysis-band taper) live in click_pipeline_v5 and are handed
# in as plain arrays, so this module keeps its no-Qt / no-scipy / no-pipeline
# contract and therefore its QThread-safety guarantee.
# ─────────────────────────────────────────────────────────────────────────────

#: Analysis band of the v6 grid [Hz] (§4.3). Matches the transmitted band.
V6_BAND_LO_HZ = 20_000.0
V6_BAND_HI_HZ = 80_000.0

#: Number of bands, FIXED rather than variable (§4.3, decision D4).
#: Fixed M keeps events with different n_seg comparable and stops the feature
#: re-encoding region duration, which fall_time_ms already carries.
V6_N_BANDS = 12

#: Half-band split for spectral_tilt [Hz] (§5.3): 20-50 kHz vs 50-80 kHz.
V6_TILT_SPLIT_HZ = 50_000.0

#: Below this segment length the bands are no longer independent (§4.3):
#:     Δf = NENBW(Tukey α=0.25) · fs / n_seg ≤ 5 kHz,  NENBW ≈ 1.104
#: ⇒ n_seg ≥ 1.104 · 200000 / 5000 ≈ 45 samples (0.225 ms).
#: Entropy is biased optimistically (toward 1) below it. Do NOT special-case such
#: events away — record n_seg alongside every feature and flag them.
V6_MIN_NSEG = 45


def band_edges(lo: float = V6_BAND_LO_HZ,
               hi: float = V6_BAND_HI_HZ,
               n_bands: int = V6_N_BANDS) -> np.ndarray:
    """The M+1 band boundaries [Hz]. Default: 13 edges, 12 bands of 5 kHz."""
    return np.linspace(float(lo), float(hi), int(n_bands) + 1)


def band_centers(edges: np.ndarray) -> np.ndarray:
    """Centre frequency of each band [Hz]."""
    e = np.asarray(edges, dtype=np.float64)
    return 0.5 * (e[:-1] + e[1:])


#: np.trapz was removed in numpy 2.0 in favour of np.trapezoid.
_trapz = getattr(np, 'trapezoid', None) or np.trapz


def band_average(spec: 'Spectrum', edges: np.ndarray) -> np.ndarray:
    """
    Reduce a PSD to the coarse band grid: the average PSD inside each band,

        P[m] = (1 / Δf_band) · ∫ PSD(f) df   over [edges[m], edges[m+1]]

    computed by trapezoidal quadrature over the native bins, with the band edges
    themselves interpolated in. Average of POWERS, never of dB (§4.3).

    ⚠️ DELIBERATE DEVIATION FROM §4.3, MEASURED NOT ASSUMED. The spec says "mean
    of the native PSD bins falling inside it". That is a crude Riemann sum of the
    integral above, and because bins-per-band is not an integer (12.8 at
    n_fft = 512), some bands get 12 native bins and some 13. That unevenness is a
    pure artefact of the n_fft the user happened to pick, and it leaks straight
    into the feature — which is the one thing the band grid exists to prevent.

    Measured over three random broadband regions, spread of spectral_entropy
    across n_fft ∈ {512, 1024, 2048, 4096, 8192}:

        mean-of-bins (spec)   0.0062   0.0069   0.0070
        trapezoid    (here)   0.00005  0.00012  0.00034      → 20-130× better

    The quantity being estimated is unchanged; only the quadrature is correct.
    This preserves the spec's intent ("mean of powers") and delivers the n_fft
    invariance §4.2 asks for, which mean-of-bins does not.

    Why reduce at all (§4.2) — the honest reason is NOT variance reduction.
    Averaging ten adjacent bins of a zero-padded region spectrum gives roughly
    ONE independent estimate, not ten, because padding interpolates the DTFT and
    the correlation length is ≈ n_fft/n_seg bins. The real justification is that
    entropy computed on the padded native grid DEPENDS ON n_fft — more padding
    smooths the noise fluctuation and drives H upward. n_fft is a display choice.
    A feature that changes when the user picks 4096 instead of 512 is not a
    feature. Reducing to the true resolution removes the dependence entirely.

    Refuses anything but scaling='psd'. An amplitude- or magnitude-scaled input
    would be silently averaged as if it were power and every feature downstream
    would degrade without raising — the exact failure mode this whole family is
    supposed to avoid. Convert first, or ask for 'psd' from compute_spectrum.

    Returns
    -------
    (n_bands,) float64. Bands containing no native bin are NaN, not 0 — an empty
    band is unmeasured, and 0 would be indistinguishable from "measured, empty".
    """
    if spec is None or len(getattr(spec, 'freqs', ())) == 0:
        return np.full(len(np.asarray(edges)) - 1, np.nan)
    if spec.scaling != 'psd':
        raise ValueError(
            f"band_average needs a PSD spectrum, got scaling={spec.scaling!r}. "
            "E[k] is defined in V²/Hz so that P_noise, estimated on 512-sample "
            "frames, subtracts from a region of ANY length with no correction "
            "factor (§3.2). Averaging amplitudes here would corrupt every "
            "feature downstream without raising.")

    e = np.asarray(edges, dtype=np.float64)
    f = np.asarray(spec.freqs, dtype=np.float64)
    p = np.asarray(spec.mags, dtype=np.float64)
    out = np.full(len(e) - 1, np.nan)

    order = np.argsort(f)
    f, p = f[order], p[order]
    if len(f) < 2:
        return out

    for m in range(len(e) - 1):
        lo, hi = e[m], e[m + 1]
        if hi <= f[0] or lo >= f[-1]:
            continue                       # band lies outside the spectrum
        inner = (f > lo) & (f < hi)
        xs = np.concatenate(([lo], f[inner], [hi]))
        ys = np.concatenate(([np.interp(lo, f, p)], p[inner],
                             [np.interp(hi, f, p)]))
        width = hi - lo
        if width <= 0:
            continue
        out[m] = float(_trapz(ys, xs) / width)
    return out


def excess_spectrum(P_region: np.ndarray, P_noise: np.ndarray) -> np.ndarray:
    """
    E[m] = max(0, P_region[m] - P_noise[m])                            (D2, §3.1)

    ⚠️ ORDER MATTERS: band-average FIRST, rectify HERE. max(0, ·) is the classic
    spectral-subtraction artefact — for a bin whose true excess is zero, a single
    χ²₂ periodogram bin of mean μ gives E[max(0, X-μ)] = μ/e ≈ 0.368 μ, a spurious
    floor at 37 % of the noise level, and entropy is precisely the feature that
    would be poisoned by it. Averaging L estimates BEFORE rectifying drops that to
    ≈ 0.126 μ at L = 10 (§4.1). Rectifying per native bin and then band-averaging
    forfeits the reduction entirely — the bias would already be baked in.

    E[k] is a SINGLE spectrum, one number per band. It is not a per-bin time
    series and not a spectrogram: the region is 30-80 samples, so it cannot be
    subdivided into multiple transforms. There is no time axis left inside it.
    """
    a = np.asarray(P_region, dtype=np.float64)
    b = np.asarray(P_noise, dtype=np.float64)
    return np.maximum(0.0, a - b)


def spectral_entropy(E: np.ndarray) -> float:
    """
    Normalised Shannon entropy of the excess spectrum — measures spectral SPREAD,
    filling gap G1 (D5, §5.1).

        p(m) = E[m] / Σ E                        H = -Σ p log₂ p / log₂ M   ∈ [0,1]

    This is the OpenAE standard definition
    (openae.io/standards/features/latest/spectral-entropy/). log₂ rather than ln
    is not arbitrary: the base cancels between numerator and denominator, but log₂
    makes the readout N_eff = 2^(H·log₂M) = the effective number of occupied bands
    directly available, and it is cheaper on a Cortex-M4F.

    DOCUMENTED DEVIATION FROM THE STANDARD: OpenAE computes it from the raw power
    spectrum |X|². Acoustic-emission piezo sensors run at high SNR behind hardware
    thresholds, so their raw spectrum IS essentially the event. At PlantLeaf SNR it
    is not — a raw-spectrum entropy gives H → 1 for BOTH a low-SNR click and a pure
    noise region, i.e. it measures "how much noise is in this region", a badly
    conditioned proxy for peak_SNR. Computing it on E[k] is not an embellishment;
    without the subtraction the feature does not work in this regime.

    Zero bins are dropped: p·log p → 0 in the limit, but 0·log 0 is NaN in floating
    point. (This is also why spectral_flatness / Wiener entropy is unusable here —
    rectification produces exact zeros, one zero sends the geometric mean to zero,
    and flatness collapses to 0 for every event. Entropy degrades gracefully.)

    Returns NaN when the total excess is zero or the input is degenerate. NOT 0
    and NOT 1: the quantity is genuinely undefined there, and a sentinel inside
    the valid range is exactly the mistake R² = 0 makes today (§7.5.3).
    """
    e = np.asarray(E, dtype=np.float64).ravel()
    e = e[np.isfinite(e)]
    m = len(e)
    if m < 2:
        return float('nan')
    total = float(np.sum(e))
    if not np.isfinite(total) or total <= 0.0:
        return float('nan')

    p = e / total
    p = p[p > 0.0]
    if len(p) == 0:
        return float('nan')
    if len(p) == 1:
        return 0.0                      # all energy in one band: H = 0 exactly
    return float(-np.sum(p * np.log2(p)) / np.log2(m))


def effective_bands(entropy: float, n_bands: int = V6_N_BANDS) -> float:
    """N_eff = 2^(H·log₂M) — the effective number of occupied bands (§5.1).

    UI readout only. Multiply by the band width for BW_eff, an effective occupied
    bandwidth in Hz. Display that; feed the entropy to the SVM."""
    if not np.isfinite(entropy):
        return float('nan')
    return float(2.0 ** (entropy * np.log2(n_bands)))


def shape_novelty(P_region: np.ndarray, P_noise: np.ndarray) -> float:
    """
    1 − cos(P̂_region, P̂_noise) — measures spectral DIFFERENCE from ambient,
    filling gap G3 (D6, §5.2).                                          ∈ [0, 1]

    Note this takes P_region, NOT E. Amplitude is divided out entirely by the L2
    normalisation, so the feature sees SHAPE ONLY. That is the whole point:
    the hard negatives are frames that survived Stages 1 and 2, i.e. AMPLITUDE
    EXCURSIONS OF THE AMBIENT NOISE. A louder burst of the same noise has the same
    spectral shape → novelty ≈ 0. A genuine click has a different shape → novelty
    large. Every amplitude-based feature (peak_SNR, pre_SNR, post_SNR, kurtosis) is
    blind to this by construction; this is the orthogonal test, aimed precisely at
    the class the classifier finds hardest.

    IMPLEMENTATION TRAP (§5.2): computed on native grids this is biased. A region
    spectrum at n_fft = 512 is intrinsically smooth (correlation length ~4 kHz)
    while Buffer 3 is spiky at 390 Hz, and the cosine between a smooth curve and a
    spiky one is depressed regardless of true shape difference → high "novelty" for
    everything. Passing both on the M = 12 band grid, where they are equally
    smooth, fixes it for free. Both arguments must be on the SAME grid.

    Returns NaN if either vector has zero norm.
    """
    a = np.asarray(P_region, dtype=np.float64).ravel()
    b = np.asarray(P_noise, dtype=np.float64).ravel()
    if len(a) != len(b) or len(a) == 0:
        return float('nan')
    good = np.isfinite(a) & np.isfinite(b)
    a, b = a[good], b[good]
    if len(a) == 0:
        return float('nan')

    na = float(np.sqrt(np.sum(a * a)))
    nb = float(np.sqrt(np.sum(b * b)))
    if na <= 1e-300 or nb <= 1e-300:
        return float('nan')

    cos = float(np.dot(a, b) / (na * nb))
    return float(np.clip(1.0 - cos, 0.0, 1.0))


def spectral_tilt(P: np.ndarray,
                  freqs: np.ndarray,
                  split_hz: float = V6_TILT_SPLIT_HZ) -> float:
    """
    Spectral slope in dB/kHz — measures spectral TILT (D7, §5.3).

        tilt = 10·log₁₀(P̃_high / P̃_low) / (f_c,high - f_c,low)

    with P̃ the MEDIAN power over each half-band and f_c the half-band centres
    (35 and 65 kHz for the default split ⇒ denominator = 30 kHz).

    MEDIAN, NOT OLS ON dB. A single 40 kHz interferer bin would drag an OLS slope.
    Because log is monotonic, median(log P) = log(median P) — so take the medians
    of the RAW POWERS and call log10 twice, not once per bin.

    WHAT IT TARGETS: v5 §1 names "hardware coupling artefacts — PCB vibrations
    couple mechanically into the MEMS, producing low-frequency broadband bursts
    concentrated near the 20 kHz analysis band edge". That is a strongly negative
    tilt. An EMI spike approximates a delta function → flat. A cavitation click
    with a mid-band resonance → modest.

    ⚠️ HONESTY NOTE, to be carried into the report and not silently absorbed: the
    50 % microphone normalisation leaves a RESIDUAL SYSTEMATIC TILT of ≈ −0.1
    dB/kHz (+8 dB at 20 kHz → −4 dB at 80 kHz, halved = −6 dB over 60 kHz). It is a
    fixed offset that shifts the whole distribution, not a per-event error.

    Grid-agnostic: works on the 12-band grid or on a native spectrum. The band grid
    is the intended call (median of 6 band means still rejects a single contaminated
    band), but the maths does not care.

    Returns NaN if either half is empty or its median power is non-positive.
    """
    p = np.asarray(P, dtype=np.float64).ravel()
    f = np.asarray(freqs, dtype=np.float64).ravel()
    if len(p) != len(f) or len(p) == 0:
        return float('nan')
    good = np.isfinite(p) & np.isfinite(f)
    p, f = p[good], f[good]

    lo_mask = f < float(split_hz)
    hi_mask = ~lo_mask
    if not np.any(lo_mask) or not np.any(hi_mask):
        return float('nan')

    p_lo = float(np.median(p[lo_mask]))
    p_hi = float(np.median(p[hi_mask]))
    if p_lo <= 0.0 or p_hi <= 0.0:
        return float('nan')

    fc_lo = float(np.mean(f[lo_mask]))
    fc_hi = float(np.mean(f[hi_mask]))
    span_khz = (fc_hi - fc_lo) / 1000.0
    if abs(span_khz) < 1e-12:
        return float('nan')

    return float(10.0 * np.log10(p_hi / p_lo) / span_khz)


def temporal_concentration(env: np.ndarray,
                           region: Tuple[int, int]) -> float:
    """
    σ_t / T_region — the time-domain companion (D8, §5.4).            ∈ [0, 0.5]

        σ_t = RMS duration of A²[n] about its centroid, over the region

    Uniform energy across the region → 1/√12 = 0.289. Energy concentrated near
    the onset → lower.

    fs cancels (both numerator and denominator are times measured in the same
    units), so this is computed in SAMPLES and the sampling rate is not a
    parameter. That is not an oversight in the spec's signature.

    NOT redundant with fall_time_ms: fall_time measures the region's LENGTH, this
    measures the SHAPE of energy within it. A click concentrates energy near the
    peak; a noise burst fills the region uniformly.

    Honest assessment (§5.4): expected separation is ~0.15-0.25 (click) vs ~0.29
    (noise burst) — modest. Cheap to compute, worth testing, ranked last.

    Returns NaN for a degenerate region or zero total energy.
    """
    a = np.asarray(env, dtype=np.float64).ravel()
    i0, i1 = int(region[0]), int(region[1])
    i0 = max(0, min(i0, len(a)))
    i1 = max(i0, min(i1, len(a)))
    seg = a[i0:i1]
    n = len(seg)
    if n < 2:
        return float('nan')

    w = seg ** 2
    w = np.where(np.isfinite(w), w, 0.0)
    total = float(np.sum(w))
    if total <= 0.0:
        return float('nan')

    t = np.arange(n, dtype=np.float64)
    centroid = float(np.sum(t * w) / total)
    var = float(np.sum(((t - centroid) ** 2) * w) / total)
    if var < 0.0:
        return float('nan')

    # T_region = n samples (not n-1): the discrete uniform then gives
    # sqrt((n²-1)/12)/n → 1/sqrt(12) as n grows, which is the documented target.
    return float(np.sqrt(var) / n)


def spectral_quantiles(E: np.ndarray,
                       edges: np.ndarray,
                       qs: Tuple[float, ...] = (0.25, 0.5, 0.75)) -> np.ndarray:
    """
    Frequency quantiles of the cumulative excess energy (D18, §5.5.3).

        C(edges[m+1]) = Σ_{j≤m} E[j] / Σ_j E[j],      C(edges[0]) = 0
        f(q) = the frequency where C crosses q, LINEARLY INTERPOLATED

    Interpolation is what makes the output continuous rather than quantised to 12
    levels, which an SVM handles far better than a 12-level categorical.

    Why quantiles rather than a sliding-band maximum: a sliding-band peak needs a
    band width B — exactly the kind of tuned magic number this project rejects.
    Quantiles have no free parameters, use all 12 bands rather than only the
    winner (so far lower variance than an argmax), and are monotone in the
    underlying distribution.

    ⚠️ SPEC ERRATUM, corrected here. §5.5.3 says "linear interpolation between
    band CENTRES". That is wrong, and measurably so: the cumulative sum through
    band m accounts for ALL of band m's energy, so it is reached at that band's
    UPPER EDGE, not at its centre. Interpolating on centres therefore reports
    every quantile half a band too low — a fixed −2.5 kHz bias on this grid. The
    test case is unambiguous: a flat excess spectrum has its median at the exact
    centre of the analysis band, 50 kHz. Interpolating on centres returns 47.5 kHz;
    interpolating on edges returns 50.000 kHz.

    Using edges also removes the need for the anchoring hack the centre version
    required (C at the first centre is already E[0]/ΣE > 0, leaving low quantiles
    undefined): C = 0 at edges[0] and C = 1 at edges[-1] fall out for free.

    Returns
    -------
    (len(qs),) float64 in Hz, NaN where undefined. f_50 = q 0.5;
    IQR_f = f(0.75) - f(0.25).

    HELD BACK FROM THE SVM (D18): f_50 is the median of the same distribution FPE
    takes the mode of (expect ρ ≈ 0.75-0.9 — two estimators of one quantity, not
    complementarity), and IQR_f collides with spectral_entropy (both measure
    spread, expect ρ ≈ 0.6-0.85 unless the clicks are multi-modal). Compute them,
    put them in the CSV, decide on measurement.
    """
    e = np.asarray(E, dtype=np.float64).ravel()
    ed = np.asarray(edges, dtype=np.float64).ravel()
    out = np.full(len(qs), np.nan)
    if len(ed) != len(e) + 1 or len(e) == 0:
        return out

    e = np.where(np.isfinite(e), e, 0.0)
    total = float(np.sum(e))
    if total <= 0.0:
        return out

    y = np.concatenate(([0.0], np.cumsum(e) / total))     # C at each band edge
    y[-1] = 1.0                                            # kill rounding drift

    # np.interp needs a non-decreasing x; y is non-decreasing by construction.
    for j, q in enumerate(qs):
        out[j] = float(np.interp(float(q), y, ed))
    return out


def psd_from_amplitude(A: np.ndarray,
                       fs: float,
                       n_frame: int) -> np.ndarray:
    """
    Convert an AMPLITUDE spectrum to the PSD convention compute_spectrum uses.

        P[k] = A[k]² · N / (2·fs)                                      [V²/Hz]

    `A[k]` is the amplitude in volts of spectral component k — i.e. what
    compute_spectrum(scaling='amplitude') returns, and what the PlantLeaf firmware
    transmits (it already applies its own 2/N before sending, so the wire carries
    amplitudes, not raw FFT coefficients). `n_frame` is the length of the frame
    those amplitudes were measured on, with a rectangular window (Σw = Σw² = N).

    Derivation, for a rectangular window of length N:
        amplitude scaling  A = 2|X|/Σw = 2|X|/N        ⇒  |X| = A·N/2
        psd scaling        P = 2|X|²/(fs·Σw²) = 2|X|²/(fs·N)
                             = 2·(A·N/2)²/(fs·N) = A²·N/(2·fs)     ∎

    ⚠️ THE SPEC IS WRONG HERE, AND IT MATTERS. SPECTRAL_FEATURES_v6_PROPOSAL.md
    §2.3 gives `P = 2·|A[k]|²/(fs·Σw²)`. That formula is correct for RAW FFT
    coefficients and wrong for amplitudes — substituting an amplitude into it is
    out by (N/2)² = 65 536 at N = 512. Verified end-to-end against
    reconstruct_frame_v5 → compute_spectrum(scaling='psd'): the form above agrees
    to a ratio of 1.000000 at every in-band bin, the spec's form to 1/65536.

    This is the same class of error as the 256× iFFT bug
    (docs/fft_and_ifft/IFFT_AMPLITUDE_SCALE_FIX.md), one power up. Had it shipped,
    P_noise would have been ~65 000× too small, max(0, P_region - P_noise) would
    have returned P_region essentially untouched, and the whole feature family
    would have silently degraded into "entropy of the raw region spectrum" — the
    exact failure mode §5.1 exists to prevent. It would not have raised.

    PSD is not a stylistic choice (§3.2). Of the three signal classes present,
    stationary NOISE is invariant under PSD scaling and falls as N^(-1/2) under
    amplitude scaling. So P_noise, estimated on 512-sample frames, subtracts from a
    region spectrum of ANY length with no correction factor at all. In amplitude
    scaling you would need √(B_eff^region / B_eff^frame) per event, recomputed for
    every click, exactly right, or the subtraction silently over- or
    under-subtracts. Energy-signal vs power-signal: structural, not cosmetic.
    """
    a = np.asarray(A, dtype=np.float64)
    return a * a * (float(n_frame) / (2.0 * float(fs)))


def spectrum_from_psd(freqs: np.ndarray,
                      psd: np.ndarray,
                      fs: float,
                      n_seg: int,
                      n_fft: int,
                      window: str = 'rectangular') -> Spectrum:
    """Wrap an already-computed PSD array as a Spectrum so it can be fed to
    band_average alongside spectra produced by compute_spectrum."""
    f = np.asarray(freqs, dtype=np.float64)
    return Spectrum(freqs=f, mags=np.asarray(psd, dtype=np.float64),
                    scaling='psd', unit=_UNITS['psd'],
                    n_seg=int(n_seg), n_fft=int(n_fft), fs=float(fs),
                    window=window, enbw_hz=float(fs) / max(1, int(n_seg)))


def v6_spectral_features(region_spec: Spectrum,
                         noise_psd: np.ndarray,
                         noise_freqs: np.ndarray,
                         *,
                         env: Optional[np.ndarray] = None,
                         region: Optional[Tuple[int, int]] = None,
                         edges: Optional[np.ndarray] = None,
                         band: Tuple[float, float] = (V6_BAND_LO_HZ, V6_BAND_HI_HZ)) -> dict:
    """
    Assemble the v6 spectral feature set for one resolved click.

    Parameters
    ----------
    region_spec : Spectrum
        compute_spectrum(region_segment, fs, scaling='psd', ...) — the PADDED
        NATIVE spectrum of the onset→decay_end region, taken from the
        RECONSTRUCTED time signal (so it already carries the analysis-band Tukey
        taper).
    noise_psd, noise_freqs :
        Buffer 3's per-bin noise PSD [V²/Hz] on the transmitted 154-bin grid,
        ALREADY multiplied by the squared analysis-band taper — see trap (a) in
        the caller. Must be built from mic-normalised data — trap (b).
    env, region :
        Hilbert envelope and (i0, i1) region bounds, for temporal_concentration.
        Optional; the feature is NaN without them.

    ─────────────────────────────────────────────────────────────────────────
    ⚠️  TWO DIFFERENT INPUTS, DELIBERATELY. THIS LOOKS LIKE AN INCONSISTENCY
        AND IS NOT. (§5.5.2)

        FPE_hz                              ← the PADDED NATIVE spectrum
        entropy / novelty / tilt / quantiles ← the 12-BAND GRID

        - entropy, novelty and the quantiles are DISTRIBUTIONAL statistics.
          Their VALUE CHANGES with n_fft: padding smooths the noise fluctuation
          and drives H upward. n_fft is a display choice, so a feature that moves
          when the user picks 4096 instead of 512 is not a feature. They must live
          on the true-resolution grid, which is what the band average provides.

        - FPE is a LOCATION statistic. Padding cannot improve resolution, but it
          DOES interpolate the DTFT and genuinely refine the argmax readout
          (REGION_FFT_FEATURE.md §2). The 12-band grid would quantise FPE to 12
          levels and destroy exactly that.

        The zero-padding argument cuts differently for the two kinds of statistic.
        Do not "harmonise" these onto one grid.
    ─────────────────────────────────────────────────────────────────────────

    ⚠️ SPEC GAP, resolved here: the spec never says how P_noise reaches the
    region's frequency grid. It cannot — the two do not share one. Buffer 3 lives
    on the transmitted 390.625 Hz grid; the region spectrum lives on fs/n_fft.
    P_noise is therefore linearly interpolated IN POWER (a PSD is a density, and
    both grids are uniform) onto region_spec.freqs for the native FPE path. The
    band-grid path needs no interpolation — both sides are averaged onto the same
    12 bands independently.
    """
    if edges is None:
        edges = band_edges(band[0], band[1], V6_N_BANDS)
    centers = band_centers(edges)
    out = {
        'spectral_entropy': float('nan'),
        'shape_novelty': float('nan'),
        'spectral_tilt': float('nan'),
        'temporal_concentration': float('nan'),
        'FPE_hz': float('nan'),
        'f_50_hz': float('nan'),
        'IQR_f': float('nan'),
        'N_eff': float('nan'),
        'BW_eff_hz': float('nan'),
        'n_seg': int(getattr(region_spec, 'n_seg', 0) or 0),
        'n_seg_valid': 0,
        'E_bands': np.full(len(centers), np.nan),
        'P_region_bands': np.full(len(centers), np.nan),
        'P_noise_bands': np.full(len(centers), np.nan),
    }
    # §4.3: below V6_MIN_NSEG the bands are correlated and entropy is biased
    # toward 1. Flagged, never silently special-cased away.
    out['n_seg_valid'] = int(out['n_seg'] >= V6_MIN_NSEG)

    if env is not None and region is not None:
        out['temporal_concentration'] = temporal_concentration(env, region)

    if region_spec is None or len(region_spec.freqs) == 0:
        return out
    # noise_psd is None whenever Buffer 3 has not yet closed a sub-window. That is
    # a normal startup state, not an error — return the NaN skeleton rather than
    # substituting a zero floor, which would make every band's excess equal the
    # region itself and look like a perfectly clean detection.
    if noise_psd is None or noise_freqs is None:
        return out

    nf = np.asarray(noise_freqs, dtype=np.float64).ravel()
    npsd = np.asarray(noise_psd, dtype=np.float64).ravel()
    if len(nf) == 0 or len(nf) != len(npsd):
        return out

    # ── Native-resolution path: FPE only ──────────────────────────────────────
    noise_on_region = np.interp(region_spec.freqs, nf, npsd, left=np.nan, right=np.nan)
    e_native = np.asarray(region_spec.mags, dtype=np.float64) - noise_on_region
    in_band = ((region_spec.freqs >= band[0]) & (region_spec.freqs <= band[1])
               & np.isfinite(e_native))
    if np.any(in_band):
        # argmax(max(0, x)) == argmax(x) whenever the max is positive, so the
        # rectification is a no-op for a LOCATION statistic; skip it.
        idx = np.flatnonzero(in_band)
        out['FPE_hz'] = float(region_spec.freqs[idx[int(np.argmax(e_native[idx]))]])

    # ── Band-grid path: everything distributional ─────────────────────────────
    p_region_b = band_average(region_spec, edges)
    noise_spec = spectrum_from_psd(nf, npsd, region_spec.fs,
                                   n_seg=region_spec.n_seg, n_fft=len(nf) * 2)
    p_noise_b = band_average(noise_spec, edges)

    out['P_region_bands'] = p_region_b
    out['P_noise_bands'] = p_noise_b

    ok = np.isfinite(p_region_b) & np.isfinite(p_noise_b)
    if not np.any(ok):
        return out

    e_bands = excess_spectrum(np.where(ok, p_region_b, 0.0),
                              np.where(ok, p_noise_b, 0.0))
    e_bands = np.where(ok, e_bands, np.nan)
    out['E_bands'] = e_bands

    out['spectral_entropy'] = spectral_entropy(e_bands[ok])
    # shape_novelty takes P_region, NOT E — the L2 normalisation divides amplitude
    # out, so it sees shape only, which is the entire point (§5.2).
    out['shape_novelty'] = shape_novelty(p_region_b[ok], p_noise_b[ok])
    # ⚠️ COMPUTED ON P_region, NOT ON E — and that is a correction, not a slip.
    # §5.3 writes the formula over "P̃, the median power" without saying which
    # spectrum. An earlier pass used E[k], for symmetry with the quantiles. That is
    # measurably wrong: max(0, ·) produces exact zeros, so the median of a half-band
    # is 0 whenever half its bands rectify away, and the feature returns NaN.
    # Measured NaN rate of tilt-on-E by occupied-band count:
    #
    #     <= 4 bands  100 %      6 bands  58 %      8 bands  5 %     >= 10 bands  0 %
    #     overall, uniform occupancy: 48.6 %   (on P_region: 0.0 %)
    #
    # The missingness is therefore CORRELATED WITH NARROWBANDNESS — i.e. with
    # spectral_entropy, the very quantity this family exists to measure. Phase 4's
    # SimpleImputer would fill precisely the high-Q clicks of §9 with a median
    # tilt, which is the silent-collapse failure §7.5.3 is about, reappearing.
    #
    # P_region always carries the noise pedestal, so it has no exact zeros and the
    # tilt is always defined. §6's own expectation table agrees: it predicts
    # "≈ −0.1 (= ambient offset)" for an ambient amplitude excursion, which is a
    # statement about the REGION's spectrum — on E an ambient excursion would have
    # no well-defined tilt at all.
    #
    # The frame-level Stage 2 GATE version (D7) is a different measurement on a
    # different spectrum and is not computed here.
    out['spectral_tilt'] = spectral_tilt(p_region_b[ok], centers[ok])

    out['N_eff'] = effective_bands(out['spectral_entropy'], int(np.count_nonzero(ok)))
    if np.isfinite(out['N_eff']) and len(edges) > 1:
        out['BW_eff_hz'] = float(out['N_eff'] * (edges[1] - edges[0]))

    q = spectral_quantiles(np.where(ok, e_bands, 0.0), edges, qs=(0.25, 0.5, 0.75))
    out['f_50_hz'] = float(q[1])
    out['IQR_f'] = float(q[2] - q[0]) if np.all(np.isfinite(q[[0, 2]])) else float('nan')
    return out



# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

def _self_test() -> int:
    fs = 200_000.0
    fails = []

    def check(name, cond, detail=''):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ''))
        if not cond:
            fails.append(name)

    print("\n1. NENBW table (bins)")
    expect = {'rectangular': 1.000, 'tukey': 1.104, 'hamming': 1.368,
              'hann': 1.508, 'blackman': 1.735}
    for name, want in expect.items():
        got = nenbw(make_window(name, 200, alpha=0.25))
        check(f"NENBW({name}) ≈ {want}", abs(got - want) < 0.01, f"got {got:.3f}")

    print("\n2. Amplitude scaling: a 3 mV tone must read 3 mV")
    amp = 0.003
    n = 400
    t = np.arange(n) / fs
    x = amp * np.sin(2 * np.pi * 50_000 * t)
    for win in WINDOWS:
        s = compute_spectrum(x, fs, window=win, n_fft=8192, scaling='amplitude')
        got = float(np.max(s.mags))
        check(f"peak with {win:12s}", abs(got - amp) / amp < 0.03,
              f"{got*1e3:.4f} mV")

    print("\n3. Zero-padding interpolates but does NOT change resolution")
    peaks, enbws = [], []
    for nf in (512, 1024, 2048, 4096, 8192):
        s = compute_spectrum(x, fs, window='tukey', n_fft=nf, scaling='amplitude')
        peaks.append(float(np.max(s.mags)))
        enbws.append(s.enbw_hz)
    check("peak amplitude invariant to n_fft",
          (max(peaks) - min(peaks)) / max(peaks) < 0.03,
          f"{min(peaks)*1e3:.4f}–{max(peaks)*1e3:.4f} mV")
    check("enbw_hz invariant to n_fft",
          (max(enbws) - min(enbws)) < 1e-6, f"{enbws[0]:.1f} Hz for all")
    check("bin spacing DOES shrink with n_fft",
          compute_spectrum(x, fs, n_fft=8192).bin_spacing_hz <
          compute_spectrum(x, fs, n_fft=512).bin_spacing_hz)

    print("\n4. Resolution formula Δf = NENBW·fs/n_seg")
    s = compute_spectrum(x, fs, window='tukey', alpha=0.25)
    want = nenbw(make_window('tukey', n, alpha=0.25)) * fs / n
    check("enbw_hz matches the formula", abs(s.enbw_hz - want) < 1e-6,
          f"{s.enbw_hz:.1f} Hz")

    print("\n5. Parseval for scaling='psd'")
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 0.002, 2048)
    s = compute_spectrum(noise, fs, window='rectangular', scaling='psd', n_fft=2048)
    integral = float(np.sum(s.mags) * (fs / s.n_fft))     # ∫PSD df
    variance = float(np.mean(noise ** 2))
    check("∫PSD·df ≈ mean(x²)", abs(integral - variance) / variance < 0.02,
          f"{integral:.3e} vs {variance:.3e}")

    print("\n6. scaling='magnitude' reproduces a raw rfft")
    s = compute_spectrum(x, fs, window='rectangular', scaling='magnitude', n_fft=n)
    raw = np.abs(np.fft.rfft(x, n=n))
    check("|X| matches np.fft.rfft", np.allclose(s.mags, raw))

    print("\n7. left_only taper preserves the onset of a decay")
    k = np.arange(300)
    decay = 0.02 * np.exp(-k / 60.0) * np.sin(2 * np.pi * 50_000 * k / fs)
    w_sym = make_window('hann', 300)
    w_left = make_window('tukey', 300, 0.25, left_only=True)
    kept_sym = float(np.sum((decay * w_sym) ** 2) / np.sum(decay ** 2))
    kept_left = float(np.sum((decay * w_left) ** 2) / np.sum(decay ** 2))
    check("one-sided Tukey keeps more decay energy than symmetric Hann",
          kept_left > kept_sym,
          f"{kept_left*100:.0f}% vs {kept_sym*100:.0f}%")

    print("\n8. band_descriptors on a known tone")
    s = compute_spectrum(x, fs, window='tukey', n_fft=8192)
    d = band_descriptors(s, band=(20_000, 80_000))
    check("peak_freq ≈ 50 kHz", abs(d['peak_freq_hz'] - 50_000) < 500,
          f"{d['peak_freq_hz']/1e3:.2f} kHz")
    check("centroid ≈ 50 kHz", abs(d['centroid_hz'] - 50_000) < 2_000,
          f"{d['centroid_hz']/1e3:.2f} kHz")

    print("\n9. Degenerate inputs do not raise")
    try:
        compute_spectrum(np.zeros(0), fs)
        compute_spectrum(np.zeros(10), fs)
        band_descriptors(compute_spectrum(np.zeros(10), fs), (20e3, 80e3))
        to_db(np.zeros(5))
        check("empty / all-zero inputs handled", True)
    except Exception as e:                                # noqa: BLE001
        check("empty / all-zero inputs handled", False, repr(e))

    # ── v6 ───────────────────────────────────────────────────────────────────

    print("\n10. PSD convention (v6 spec §8.1 — BLOCKING)")
    # The self-contained half: an amplitude spectrum converted with
    # psd_from_amplitude must equal compute_spectrum(scaling='psd') bin for bin.
    # The end-to-end half, which needs reconstruct_frame_v5 and the analysis-band
    # taper, lives in test_scripts/verify_psd_convention_v6.py — importing
    # click_pipeline_v5 here would break this module's no-pipeline contract.
    n_fr = 512
    tone = 0.005 * np.cos(2 * np.pi * 50_000 * np.arange(n_fr) / fs)
    amp = compute_spectrum(tone, fs, window='rectangular', n_fft=n_fr,
                           scaling='amplitude')
    psd = compute_spectrum(tone, fs, window='rectangular', n_fft=n_fr,
                           scaling='psd')
    conv = psd_from_amplitude(amp.mags, fs, n_fr)
    k50 = int(round(50_000 / (fs / n_fr)))
    check("A²·N/(2·fs) reproduces scaling='psd' at the tone",
          abs(conv[k50] - psd.mags[k50]) / psd.mags[k50] < 1e-12,
          f"ratio {psd.mags[k50] / conv[k50]:.6f}")
    rng = np.random.default_rng(11)
    wide = rng.standard_normal(n_fr) * 1e-3
    a2 = compute_spectrum(wide, fs, window='rectangular', n_fft=n_fr, scaling='amplitude')
    p2 = compute_spectrum(wide, fs, window='rectangular', n_fft=n_fr, scaling='psd')
    c2 = psd_from_amplitude(a2.mags, fs, n_fr)
    sl = slice(51, 205)                       # the transmitted band
    rel = np.max(np.abs(c2[sl] - p2.mags[sl]) / p2.mags[sl])
    check("...and on every bin of a broadband signal", rel < 1e-12,
          f"max rel err {rel:.2e} over bins 51-204")
    spec_formula = 2.0 * a2.mags[k50] ** 2 / (fs * n_fr)
    check("the spec's §2.3 formula is wrong by (N/2)² = 65536",
          abs(p2.mags[k50] / spec_formula - (n_fr // 2) ** 2) / (n_fr // 2) ** 2 < 1e-9,
          f"measured factor {p2.mags[k50] / spec_formula:.0f}")

    print("\n11. spectral_entropy on known distributions (§5.1)")
    edges12 = band_edges()
    check("flat excess spectrum → H ≈ 1",
          abs(spectral_entropy(np.ones(12)) - 1.0) < 1e-12,
          f"H = {spectral_entropy(np.ones(12)):.6f}")
    spike = np.zeros(12)
    spike[5] = 1.0
    check("single-band spike → H ≈ 0", abs(spectral_entropy(spike)) < 1e-12,
          f"H = {spectral_entropy(spike):.6f}")
    half = np.zeros(12)
    half[:6] = 1.0
    check("6 of 12 bands occupied → H = log₂6/log₂12",
          abs(spectral_entropy(half) - np.log2(6) / np.log2(12)) < 1e-12,
          f"H = {spectral_entropy(half):.4f}, N_eff = {effective_bands(spectral_entropy(half)):.2f}")
    check("all-zero excess → NaN, not 0 and not 1",
          np.isnan(spectral_entropy(np.zeros(12))))

    print("\n12. shape_novelty (§5.2)")
    v = np.array([1.0, 4.0, 2.0, 9.0, 3.0, 5.0, 1.0, 2.0, 8.0, 3.0, 1.0, 6.0])
    check("identical spectra → 0", abs(shape_novelty(v, v)) < 1e-12,
          f"{shape_novelty(v, v):.2e}")
    check("scaled copy → 0 (shape only, amplitude divided out)",
          abs(shape_novelty(v, 137.0 * v)) < 1e-12,
          f"{shape_novelty(v, 137.0 * v):.2e}")
    o1 = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    o2 = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    check("orthogonal spectra → 1", abs(shape_novelty(o1, o2) - 1.0) < 1e-12,
          f"{shape_novelty(o1, o2):.6f}")
    check("zero-norm input → NaN", np.isnan(shape_novelty(v, np.zeros(12))))

    print("\n13. spectral_tilt recovers a known slope (§5.3)")
    centers12 = band_centers(edges12)
    for want in (-0.20, -0.05, 0.0, 0.10):
        # A spectrum whose power in dB is exactly linear in frequency.
        p_db = want * (centers12 / 1000.0)
        p_lin = 10.0 ** (p_db / 10.0)
        got = spectral_tilt(p_lin, centers12)
        check(f"slope {want:+.2f} dB/kHz recovered",
              abs(got - want) < 1e-9, f"got {got:+.6f}")
    # The whole reason for the median form (§5.3): "a single 40 kHz interferer bin
    # would drag an OLS slope". Assert that against an actual OLS slope on dB
    # rather than against exact invariance — with an even band count the median is
    # the mean of the two middle values, so one contaminated band does move it a
    # little. The claim is relative robustness, so test the relative claim.
    p_lin = 10.0 ** ((-0.10 * (centers12 / 1000.0)) / 10.0)
    spiked = p_lin.copy()
    spiked[3] *= 1000.0                        # a 30 dB tonal interferer at 37.5 kHz

    def _ols_tilt(p):
        return float(np.polyfit(centers12 / 1000.0, 10.0 * np.log10(p), 1)[0])

    d_med = abs(spectral_tilt(spiked, centers12) - spectral_tilt(p_lin, centers12))
    d_ols = abs(_ols_tilt(spiked) - _ols_tilt(p_lin))
    # Regression guard for the E-vs-P_region choice: tilt must be DEFINED for a
    # narrowband spectrum. On E[k] a spike rectifies its neighbours to zero and the
    # half-band median is 0, so the feature would be NaN for exactly the high-Q
    # clicks §9 asks about — and NaN correlated with narrowbandness is worse than
    # useless once an imputer fills it.
    spike12 = np.zeros(12)
    spike12[7] = 1.0
    check("tilt is NaN on a rectified single-band excess (why E is NOT used)",
          not np.isfinite(spectral_tilt(spike12, centers12)))
    check("...but DEFINED on P_region, which keeps its noise pedestal",
          np.isfinite(spectral_tilt(spike12 + 1e-3, centers12)),
          f"{spectral_tilt(spike12 + 1e-3, centers12):+.4f} dB/kHz")

    check("median form is far more interferer-robust than OLS on dB",
          d_med < 0.2 * d_ols,
          f"median moves {d_med:.4f} dB/kHz, OLS moves {d_ols:.4f} "
          f"({d_ols / max(d_med, 1e-12):.0f}× more)")

    print("\n14. temporal_concentration (§5.4)")
    n_reg = 58
    uniform = np.ones(n_reg)
    got = temporal_concentration(uniform, (0, n_reg))
    want = np.sqrt((n_reg ** 2 - 1) / 12.0) / n_reg     # exact discrete uniform
    check("uniform energy over the region → 1/√12 ≈ 0.289",
          abs(got - 1.0 / np.sqrt(12)) < 1e-3 and abs(got - want) < 1e-12,
          f"{got:.5f}  (discrete exact {want:.5f}, 1/√12 = {1/np.sqrt(12):.5f})")
    conc = np.zeros(n_reg)
    conc[:4] = 1.0
    check("energy concentrated at the onset → much lower",
          temporal_concentration(conc, (0, n_reg)) < 0.05,
          f"{temporal_concentration(conc, (0, n_reg)):.5f}")
    check("degenerate region → NaN",
          np.isnan(temporal_concentration(np.ones(5), (2, 2))))

    print("\n15. n_fft INVARIANCE — the whole reason for the band grid (§4.2)")
    # A noisy broadband region. On the padded native grid, entropy climbs with
    # n_fft because padding smooths the fluctuation. On the band grid it must not
    # move at all. If this fails, the grid is wrong.
    noise_f = np.arange(51, 205) * (fs / 512)
    noise_p = np.full(154, 1e-11)
    worst_band = worst_native = 0.0
    for seed in (1234, 7, 99):
        rng = np.random.default_rng(seed)
        seg = rng.standard_normal(58) * 1e-3
        ent, nov, ent_native = [], [], []
        for nf_try in (512, 1024, 2048, 4096, 8192):
            rs = compute_spectrum(seg, fs, window='tukey', n_fft=nf_try, scaling='psd')
            f6 = v6_spectral_features(rs, noise_p, noise_f)
            ent.append(f6['spectral_entropy'])
            nov.append(f6['shape_novelty'])
            # The same statistic on the PADDED NATIVE grid, for contrast.
            m = (rs.freqs >= 20e3) & (rs.freqs <= 80e3)
            e_nat = np.maximum(0.0, rs.mags[m] - np.interp(rs.freqs[m], noise_f, noise_p))
            ent_native.append(spectral_entropy(e_nat))
        worst_band = max(worst_band, max(ent) - min(ent), max(nov) - min(nov))
        worst_native = max(worst_native, max(ent_native) - min(ent_native))

    check("spectral_entropy / shape_novelty invariant to n_fft on the band grid",
          worst_band < 1e-3, f"worst spread {worst_band:.6f} over 512→8192, 3 seeds")
    # The contrast is the actual §4.2 argument: on the native grid H climbs
    # MONOTONICALLY with padding (more padding smooths the noise fluctuation), and
    # that is the dependence the grid exists to remove — not merely shrink.
    check("  ...whereas on the padded native grid it drifts an order more",
          worst_native > 20 * worst_band,
          f"native spread {worst_native:.6f} vs band {worst_band:.6f} "
          f"({worst_native / max(worst_band, 1e-12):.0f}×)")
    check("band_average refuses a non-PSD spectrum",
          _raises(lambda: band_average(
              compute_spectrum(np.ones(58), fs, scaling='amplitude'), edges12),
              ValueError))

    print("\n16. spectral_quantiles (§5.5.3)")
    sym = np.zeros(12)
    sym[5] = sym[6] = 1.0                     # symmetric about 50 kHz
    q = spectral_quantiles(sym, edges12, (0.25, 0.5, 0.75))
    check("symmetric excess → f_50 = 50 kHz", abs(q[1] - 50_000) < 1e-6,
          f"f_50 = {q[1]/1e3:.3f} kHz")
    flat_q = spectral_quantiles(np.ones(12), edges12, (0.25, 0.5, 0.75))
    check("flat excess → f_50 = 50 kHz, IQR = 30 kHz",
          abs(flat_q[1] - 50_000) < 1e-6 and abs((flat_q[2] - flat_q[0]) - 30_000) < 1e-6,
          f"f_50 = {flat_q[1]/1e3:.2f} kHz, IQR = {(flat_q[2]-flat_q[0])/1e3:.2f} kHz")
    narrow = np.zeros(12)
    narrow[1] = 1.0
    wide_q = spectral_quantiles(np.ones(12), edges12, (0.25, 0.75))
    nar_q = spectral_quantiles(narrow, edges12, (0.25, 0.75))
    check("narrowband IQR < broadband IQR",
          (nar_q[1] - nar_q[0]) < (wide_q[1] - wide_q[0]),
          f"{(nar_q[1]-nar_q[0])/1e3:.2f} vs {(wide_q[1]-wide_q[0])/1e3:.2f} kHz")
    check("all-zero excess → NaN, not 0",
          bool(np.all(np.isnan(spectral_quantiles(np.zeros(12), edges12, (0.5,))))))

    print("\n17. v6 degenerate inputs do not raise")
    try:
        empty = compute_spectrum(np.zeros(0), fs, scaling='psd')
        v6_spectral_features(empty, noise_p, noise_f)
        v6_spectral_features(compute_spectrum(np.zeros(30), fs, scaling='psd'),
                             np.zeros(154), noise_f)
        excess_spectrum(np.zeros(12), np.zeros(12))
        band_average(empty, edges12)
        check("empty / all-zero v6 inputs handled", True)
    except Exception as e:                                # noqa: BLE001
        check("empty / all-zero v6 inputs handled", False, repr(e))

    print(f"\n{'ALL PASSED' if not fails else 'FAILED: ' + ', '.join(fails)}\n")
    return 0 if not fails else 1


def _raises(fn, exc) -> bool:
    """True if `fn()` raises `exc`. Used by the self-test only."""
    try:
        fn()
    except exc:
        return True
    except Exception:                                     # noqa: BLE001
        return False
    return False


if __name__ == '__main__':
    import sys
    sys.exit(_self_test())
