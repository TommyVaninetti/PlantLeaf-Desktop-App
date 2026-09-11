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

    print(f"\n{'ALL PASSED' if not fails else 'FAILED: ' + ', '.join(fails)}\n")
    return 0 if not fails else 1


if __name__ == '__main__':
    import sys
    sys.exit(_self_test())
