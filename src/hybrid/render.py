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
Batch rendering — one PNG per injected Dryad clip, plus per-class contact sheets.

Four panels per clip:

  1. Time domain — the stitched prev|curr|next reconstruction with its Hilbert
     envelope, the resolved onset/peak/decay markers, the two decision levels
     that produced them, and the frame seams drawn honestly rather than hidden.
  2. Region FFT — the spectrum of the onset->decay_end segment, which is what
     "the click's spectrum" actually means here, with the transmitted 512-point
     frame FFT overlaid for reference.
  3. Native-rate comparison — 500 kHz against 200 kHz, in time and in frequency,
     showing exactly what resampling discarded. This runs spec section 5's
     validation visually on every clip instead of as a one-off script.
  4. Footer — provenance and the full v5 feature vector.

Uses matplotlib's Agg backend, not pyqtgraph: this runs headless under
multiprocessing, and `container.render()` on a GL-backed PlotWidget segfaults on
macOS (the same reason `data_collection_dialog_v5._render_candidate_screenshot`
hand-rolls QPainter instead). Palette and axis styling follow
`src/ml/analyze_dataset.py` so the output looks like the rest of the project's
generated plots.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import channel_model as cm
from . import frame_emulator as fe
from .injector import InjectionResult
from .pipeline_loader import load_pipeline, load_spectral

# Palette — matches src/ml/analyze_dataset.py:574-580
BG = "#1A1A2E"
PANEL_BG = "#16162A"
SIGNAL_COLOR = "#6FA8DC"
ENVELOPE_COLOR = "#E84040"
REFERENCE_COLOR = "#8888AA"
ACCENT_COLOR = "#F5C542"
NATIVE_COLOR = "#7FD17F"
TEXT_COLOR = "#CCCCCC"
MUTED_COLOR = "#8A8AA0"
GRID_COLOR = "#2A2A4A"
SPINE_COLOR = "#444466"
SEAM_COLOR = "#5A5A80"
BAND_COLOR = "#E84040"

FIG_W_IN = 16.0
FIG_H_IN = 12.0
DPI = 100                       # 1600 x 1200 px

BAND = (float(fe.FREQ_MIN_HZ), float(fe.FREQ_MAX_HZ))

_MPL_READY = False


def _ensure_matplotlib():
    """
    Import matplotlib with the Agg backend, once per process.

    matplotlib is commented out in requirements.txt ("ONLY if running
    src/ml/analyze_dataset.py"), so it is an optional dependency here too and
    the error message says how to get it rather than dumping an ImportError.
    """
    global _MPL_READY
    if _MPL_READY:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for rendering but is not installed "
            "(it is optional in requirements.txt). Install it with: "
            "pip install matplotlib"
        ) from exc
    _MPL_READY = True


def _style_axis(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color=TEXT_COLOR, fontsize=9, loc="left", pad=4)
    ax.tick_params(colors=MUTED_COLOR, labelsize=7, length=3)
    ax.set_xlabel(xlabel, color=MUTED_COLOR, fontsize=8)
    ax.set_ylabel(ylabel, color=MUTED_COLOR, fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, zorder=0)


def _draw_time_domain(ax, result: InjectionResult, *, zoom: bool):
    """
    Stitched reconstruction, envelope, resolved markers, decision levels.

    Drawn twice per figure: once over the full three-frame context to show the
    click against its noise bed, and once zoomed to the resolved click. Without
    the zoom the click is a ~0.1 ms spike inside a 7.7 ms window — legible as a
    detection, useless for judging shape, which is what manual labelling needs.
    """
    render = result.render
    ctx, resolved = render["ctx"], render["resolved"]
    signal, envelope = ctx["signal"], ctx["envelope"]

    # Time relative to the current frame's start, in ms.
    t_ms = (np.arange(len(signal)) - ctx["origin"]) / fe.FS * 1e3

    ax.plot(t_ms, signal * 1e3, color=SIGNAL_COLOR, linewidth=0.7, zorder=3, label="iFFT signal")
    ax.plot(t_ms, envelope * 1e3, color=ENVELOPE_COLOR, linewidth=1.1, zorder=4, label="Hilbert envelope")
    ax.plot(t_ms, -envelope * 1e3, color=ENVELOPE_COLOR, linewidth=1.1, zorder=4, alpha=0.45)

    nf, sd = render["noise_floor"], render["std_noise"]
    cp = load_pipeline()
    # The two levels that actually produced the markers, so the plot explains itself.
    ax.axhline((nf + cp.LEVEL_STD_FACTOR * sd) * 1e3, color=ACCENT_COLOR, linewidth=0.8,
               linestyle="--", zorder=2, label=f"onset level (nf+{cp.LEVEL_STD_FACTOR:g}sd)")
    ax.axhline((nf + cp.DECAY_END_NOISE_ALPHA * sd) * 1e3, color=MUTED_COLOR, linewidth=0.8,
               linestyle=":", zorder=2, label=f"decay-end level (nf+{cp.DECAY_END_NOISE_ALPHA:g}sd)")

    # Frame joins: each frame was tapered independently, so the seams are real.
    for seam in ctx["seams"]:
        if 0 <= seam < len(t_ms):
            ax.axvline(t_ms[seam], color=SEAM_COLOR, linewidth=0.7, alpha=0.6, zorder=1)

    if zoom:
        lo = max(0, int(resolved["onset"]) - 40)                     # 0.2 ms of lead-in
        hi = min(len(t_ms) - 1, int(resolved["decay_end"]) + 80)     # 0.4 ms of tail
        ax.set_xlim(t_ms[lo], t_ms[hi])
        span = envelope[lo:hi + 1]
        ax.set_ylim(-1.35 * float(np.max(span)) * 1e3, 1.35 * float(np.max(span)) * 1e3)

    # Markers last, so they can be placed against the final axis limits.
    y_top = ax.get_ylim()[1]
    for key, colour, style in (("onset", ACCENT_COLOR, "-"), ("peak", ENVELOPE_COLOR, "-"),
                               ("decay_start", MUTED_COLOR, "--"), ("decay_end", MUTED_COLOR, "-")):
        idx = int(resolved[key])
        if not (0 <= idx < len(t_ms)):
            continue
        x = t_ms[idx]
        if zoom and not (ax.get_xlim()[0] <= x <= ax.get_xlim()[1]):
            continue
        ax.axvline(x, color=colour, linewidth=0.9, linestyle=style, alpha=0.85, zorder=5)
        if zoom:
            ax.annotate(key, xy=(x, y_top), fontsize=6.5, color=colour, rotation=90,
                        va="top", ha="right", zorder=6)

    decay_len = int(resolved["decay_end"] - resolved["decay_start"])
    tau = result.features.get("tau_ms", float("nan"))
    note = f"tau={tau:.3f} ms" if tau > 0 else "tau = -1 (fit failed)"
    if result.features.get("fit_dead_zone", 0.0):
        note += f"  [decay_len={decay_len} in the 13-21 fit dead zone]"

    if zoom:
        _style_axis(ax, f"1b. Zoom on the resolved click   -   {note}   "
                        f"decay_len={decay_len} samples",
                    "time relative to frame start [ms]", "amplitude [mV]")
        ax.legend(loc="upper right", fontsize=6, framealpha=0.25, facecolor=BG,
                  edgecolor=SPINE_COLOR, labelcolor=TEXT_COLOR)
    else:
        _style_axis(ax, "1a. Full reconstructed context (prev|curr|next) in the noise bed",
                    "", "amplitude [mV]")


def _panel_region_fft(ax, result: InjectionResult):
    """Region FFT over onset->decay_end, with the transmitted frame FFT overlaid."""
    sa = load_spectral()
    render = result.render
    ctx, resolved = render["ctx"], render["resolved"]

    lo = int(resolved["onset"])
    hi = min(int(resolved["decay_end"]) + 1, len(ctx["signal"]))
    segment = ctx["signal"][lo:hi]

    if len(segment) < sa.MIN_SEGMENT_SAMPLES:
        ax.text(0.5, 0.5, f"region too short ({len(segment)} samples)", color=MUTED_COLOR,
                fontsize=9, ha="center", va="center", transform=ax.transAxes)
        _style_axis(ax, "2. Region FFT (onset -> decay end)", "frequency [kHz]", "magnitude [dB]")
        return

    # left_only=False deliberately: with an onset-anchored region the larger
    # discontinuity is at the TRAILING edge, so a symmetric taper is right.
    # Same reasoning as region_fft_dialog.py:451.
    spectrum = sa.compute_spectrum(segment, fe.FS, window="tukey", alpha=0.25)
    db = sa.to_db(spectrum.mags)
    ax.plot(spectrum.freqs / 1e3, db, color=SIGNAL_COLOR, linewidth=1.0, zorder=3,
            label=f"region FFT (n_seg={spectrum.n_seg}, n_fft={spectrum.n_fft})")

    # Reference: the 512-point frame FFT the hardware actually transmitted.
    fft_norm, freq_axis = render["fft_norm"], render["freq_axis"]
    mask = (freq_axis >= BAND[0]) & (freq_axis <= BAND[1])
    if np.any(mask) and np.max(fft_norm[mask]) > 0:
        ref_db = sa.to_db(fft_norm[mask])
        ax.plot(freq_axis[mask] / 1e3, ref_db, color=REFERENCE_COLOR, linewidth=0.9,
                linestyle="--", alpha=0.8, zorder=2, label="transmitted 512-pt frame FFT")

    ax.axvspan(BAND[0] / 1e3, BAND[1] / 1e3, color=BAND_COLOR, alpha=0.06, zorder=0)
    for edge in BAND:
        ax.axvline(edge / 1e3, color=BAND_COLOR, linewidth=0.7, alpha=0.5, zorder=1)

    desc = sa.band_descriptors(spectrum, BAND)
    ax.axvline(desc["peak_freq_hz"] / 1e3, color=ACCENT_COLOR, linewidth=0.9, zorder=4)
    ax.set_xlim(0, fe.FS / 2 / 1e3)
    ax.set_ylim(-90, 5)

    _style_axis(
        ax,
        "2. Region FFT (onset -> decay end)   -   "
        f"FPE {desc['peak_freq_hz'] / 1e3:.1f} kHz, centroid {desc['centroid_hz'] / 1e3:.1f} kHz, "
        f"SPR {desc['spr']:.2f}, R_spec {desc['r_spectral']:.2f}",
        "frequency [kHz]", "magnitude [dB rel. peak]",
    )
    ax.legend(loc="upper right", fontsize=6, framealpha=0.25, facecolor=BG,
              edgecolor=SPINE_COLOR, labelcolor=TEXT_COLOR)


def _fit_tau_fixed_window(envelope: np.ndarray, fs: int, *,
                          floor_frac: float = 0.05, max_ms: float = 0.5) -> float:
    """
    Reference tau from a level-bounded post-peak window, for the 500-vs-200 kHz check.

    The pipeline's own tau cannot be used for this comparison: it depends on
    where `decay_end` lands, which depends on the noise floor, and the native
    500 kHz clip has essentially no floor. This fit is defined purely by the
    signal, so it means the same thing at both rates.

    The window runs from the envelope peak down to `floor_frac` of the peak
    (3 time constants at 5 %), capped at `max_ms`. Bounding by LEVEL rather than
    by a fixed sample count is what makes it robust: Dryad clips carry secondary
    ringing 0.1-0.3 ms after the click, and a fixed 80-sample window swallows it
    and returns a badly inflated tau (1.1 ms on clips that actually decay in
    ~0.15 ms). Stopping at 5 % of peak excludes the ringing.

    Expressed as a duration rather than a sample count so the two rates get
    genuinely equivalent windows.

    Returns tau in ms, or nan if the segment does not decay.
    """
    peak = int(np.argmax(envelope))
    tail = envelope[peak:peak + int(max_ms * 1e-3 * fs)]
    if len(tail) < 10:
        return float("nan")

    peak_amp = float(tail[0])
    if peak_amp <= 0:
        return float("nan")

    below = np.where(tail < floor_frac * peak_amp)[0]
    end = int(below[0]) if len(below) else len(tail)
    seg = tail[:end]
    if len(seg) < 10:
        return float("nan")

    log_env = np.log(np.maximum(seg, 1e-12))
    n = np.arange(len(seg), dtype=np.float64)
    slope = float(np.polyfit(n, log_env, 1)[0])
    return float(-1000.0 / (slope * fs)) if slope < 0 else float("nan")


def _panel_native_comparison(ax_time, ax_freq, result: InjectionResult):
    """500 kHz vs 200 kHz: what resampling kept and what it discarded."""
    cp = load_pipeline()
    render = result.render
    native = render["native_500k"]
    resampled = cm.resample_500k_to_200k(native)

    env_native = cp.compute_hilbert_envelope(native)
    env_200k = cp.compute_hilbert_envelope(resampled)

    # Normalise both to their own peak: the vertical scales are unrelated (the
    # Dryad units are arbitrary), so only the shapes are comparable.
    scale_n = max(np.max(np.abs(env_native)), 1e-12)
    scale_r = max(np.max(np.abs(env_200k)), 1e-12)

    t_native = np.arange(len(native)) / cm.DRYAD_FS * 1e3
    t_200k = np.arange(len(resampled)) / fe.FS * 1e3
    ax_time.plot(t_native, native / scale_n, color=NATIVE_COLOR, linewidth=0.5,
                 alpha=0.55, zorder=2, label="500 kHz native")
    ax_time.plot(t_200k, resampled / scale_r, color=SIGNAL_COLOR, linewidth=0.7,
                 zorder=3, label="200 kHz resampled (2/5)")
    ax_time.plot(t_native, env_native / scale_n, color=NATIVE_COLOR, linewidth=1.1, zorder=4)
    ax_time.plot(t_200k, env_200k / scale_r, color=ENVELOPE_COLOR, linewidth=1.1, zorder=5)

    tau_native = _fit_tau_fixed_window(env_native, cm.DRYAD_FS)
    tau_200k = _fit_tau_fixed_window(env_200k, fe.FS)
    delta = (100.0 * (tau_200k - tau_native) / tau_native
             if np.isfinite(tau_native) and tau_native > 0 and np.isfinite(tau_200k) else float("nan"))

    # Labelled "primary decay" to keep it distinct from the pipeline's tau in
    # panel 1b: this one is bounded by signal level (peak -> 5 % of peak), that
    # one by the noise floor, so they measure different parts of the envelope and
    # are not expected to match. What matters here is 500 kHz vs 200 kHz
    # agreement under one definition — spec section 5's validation.
    _style_axis(
        ax_time,
        "3a. Native 500 kHz vs resampled 200 kHz (peak-normalised)   -   "
        f"primary decay (peak -> 5 % of peak): {tau_native:.3f} ms vs {tau_200k:.3f} ms "
        f"({delta:+.1f} %)",
        "time [ms]", "normalised amplitude",
    )
    ax_time.legend(loc="upper right", fontsize=6, framealpha=0.25, facecolor=BG,
                   edgecolor=SPINE_COLOR, labelcolor=TEXT_COLOR)

    # Spectra, both to their own Nyquist.
    sa = load_spectral()
    for sig, fs, colour, label in ((native, cm.DRYAD_FS, NATIVE_COLOR, "500 kHz native"),
                                   (resampled, fe.FS, SIGNAL_COLOR, "200 kHz resampled")):
        spectrum = sa.compute_spectrum(sig, fs, window="tukey", alpha=0.25)
        ax_freq.plot(spectrum.freqs / 1e3, sa.to_db(spectrum.mags), color=colour,
                     linewidth=0.9, alpha=0.9, label=label)

    ax_freq.axvline(fe.FS / 2 / 1e3, color=ACCENT_COLOR, linewidth=1.0, linestyle="--",
                    label="200 kHz Nyquist (100 kHz)")
    ax_freq.axvspan(BAND[0] / 1e3, BAND[1] / 1e3, color=BAND_COLOR, alpha=0.06, zorder=0)
    ax_freq.set_xlim(0, cm.DRYAD_FS / 2 / 1e3)
    ax_freq.set_ylim(-90, 5)

    # How much of the clip's energy the 100 kHz limit actually discards.
    win = np.hanning(len(native))
    power = np.abs(np.fft.rfft(native * win)) ** 2
    freqs = np.fft.rfftfreq(len(native), 1.0 / cm.DRYAD_FS)
    total = power.sum()
    frac_band = power[(freqs >= BAND[0]) & (freqs <= BAND[1])].sum() / total if total else 0.0
    frac_lost = power[freqs >= fe.FS / 2].sum() / total if total else 0.0

    _style_axis(
        ax_freq,
        f"3b. Spectra to each Nyquist   -   {100 * frac_band:.1f} % of energy in 20-80 kHz, "
        f"{100 * frac_lost:.1f} % discarded above 100 kHz",
        "frequency [kHz]", "magnitude [dB rel. peak]",
    )
    ax_freq.legend(loc="upper right", fontsize=6, framealpha=0.25, facecolor=BG,
                   edgecolor=SPINE_COLOR, labelcolor=TEXT_COLOR)


_FOOTER_FEATURES = (
    "peak_SNR", "pre_SNR", "post_SNR", "rise_time_ms", "fall_time_ms",
    "asymmetry_integral", "ZCR_pre", "ZCR_click", "ZCR_post", "kurtosis",
    "centroid_shift_hz", "tau_ms", "R2", "fit_coverage", "SPR", "R_spectral", "FPE_hz",
)


def _panel_footer(ax, result: InjectionResult):
    ax.axis("off")
    ax.set_facecolor(BG)

    detected = "DETECTED by Stage 1" if result.detected else "NOT detected by Stage 1"
    lines = [
        f"{result.clip_id}   |   {result.class_name}  ({result.species}/{result.condition}, "
        f"{'click' if result.is_click else 'noise'})   |   {detected}",
        f"bed {result.bed_id}   room={result.room}   regime={result.regime}   "
        f"noise_floor={result.noise_floor * 1e3:.3f} mV   std_noise={result.std_noise * 1e3:.3f} mV",
        f"injection: t0={result.t0_sample}  sub-frame phase={result.subframe_phase}/512  "
        f"frame={result.frame_idx}  gain={result.gain:.4g}   "
        f"peak_SNR target={result.target_peak_snr:.2f}  measured={result.measured_peak_snr:.2f}",
    ]
    for i, line in enumerate(lines):
        ax.text(0.005, 0.95 - i * 0.16, line, color=TEXT_COLOR, fontsize=8.5,
                family="monospace", va="top", transform=ax.transAxes)

    cells = []
    for name in _FOOTER_FEATURES:
        value = result.features.get(name)
        if value is None:
            continue
        if abs(value) >= 1000:
            text = f"{value:.0f}"
        elif abs(value) >= 1:
            text = f"{value:.3f}"
        else:
            text = f"{value:.4f}"
        cells.append(f"{name}={text}")

    per_row = 6
    for row in range(math.ceil(len(cells) / per_row)):
        chunk = cells[row * per_row:(row + 1) * per_row]
        ax.text(0.005, 0.44 - row * 0.15, "   ".join(f"{c:<26}" for c in chunk),
                color=MUTED_COLOR, fontsize=7.5, family="monospace", va="top",
                transform=ax.transAxes)


def render_clip(result: InjectionResult, out_path: str | Path) -> Path:
    """
    Render the four-panel figure for one injected clip.

    Requires `result.render` — inject with `keep_render_payload=True`.
    """
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    if result.render is None:
        raise ValueError(
            f"{result.clip_id} has no render payload; "
            "call inject_batch(..., keep_render_payload=True)"
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI)
    fig.patch.set_facecolor(BG)
    grid = fig.add_gridspec(6, 1, height_ratios=[1.3, 2.6, 2.4, 1.9, 1.9, 1.5],
                            hspace=0.62, left=0.055, right=0.985, top=0.968, bottom=0.03)

    _draw_time_domain(fig.add_subplot(grid[0]), result, zoom=False)
    _draw_time_domain(fig.add_subplot(grid[1]), result, zoom=True)
    _panel_region_fft(fig.add_subplot(grid[2]), result)
    _panel_native_comparison(fig.add_subplot(grid[3]), fig.add_subplot(grid[4]), result)
    _panel_footer(fig.add_subplot(grid[5]), result)

    fig.savefig(out_path, dpi=DPI, facecolor=BG)
    plt.close(fig)
    return out_path


def render_contact_sheet(png_paths: list[Path], out_path: str | Path, *,
                         cols: int = 3, rows: int = 4, title: str = "") -> Path | None:
    """
    Tile already-rendered PNGs into one sheet for fast visual triage.

    Reads the PNGs back rather than re-rendering: the per-clip figures are the
    expensive part, and a sheet is only ever a downscaled view of them.
    """
    _ensure_matplotlib()
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    png_paths = [Path(p) for p in png_paths if Path(p).is_file()]
    if not png_paths:
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 4.0), dpi=90)
    fig.patch.set_facecolor(BG)
    for ax, path in zip(np.ravel(axes), png_paths[:rows * cols]):
        ax.imshow(mpimg.imread(path))
        ax.set_title(path.stem, color=TEXT_COLOR, fontsize=6, pad=2)
        ax.axis("off")
    for ax in np.ravel(axes)[len(png_paths):]:
        ax.axis("off")

    if title:
        fig.suptitle(title, color=TEXT_COLOR, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97 if title else 1.0))
    fig.savefig(out_path, dpi=90, facecolor=BG)
    plt.close(fig)
    return out_path
