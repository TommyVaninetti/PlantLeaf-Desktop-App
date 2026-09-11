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
Click injection — transplanting Dryad clicks into PlantLeaf noise beds.

Implements section 9 of HYBRID_TRAINING_NOISE_INJECTION_SPEC.md.


Why inject at all
-----------------
Two problems, one mechanism. A bare 2 ms clip has no history for the adaptive
estimator to derive `noise_floor` from, so `peak_SNR` is undefined; and Khait's
recordings are anechoic-chamber-quiet while PlantLeaf's are real rooms, so every
noise-normalized feature would be trained on an unrealistically clean
population.

This is measurable, not theoretical. Running the pipeline on Dryad clicks padded
with their OWN anechoic noise gives peak_SNR 51-313 (PlantLeaf's real clicks:
median 12.8) and tau up to 1.13 ms with R^2 = 0.07 (PlantLeaf: median 0.188 ms).
The cause is precisely spec section 6(b): `decay_end` fires when the envelope
drops below `noise_floor + 1.5*std_noise`, and against an anechoic floor it never
does, so the decay window runs 4-10x too long and drags tau with it. Fitted over
a fixed window instead, Dryad tau is 0.144-0.175 ms — very close to PlantLeaf's
0.188 ms. The physics already agrees; only the noise regime disagrees.


Where the addition happens
--------------------------
In the time domain, in raw recorded space, on the bed's own frame grid. The bed
arrives from `noise_bed` via `frame_emulator.inverse_raw()` (never
`reconstruct_frame_v5` — see that module's docstring), and the click arrives
from `channel_model.prepare_dryad_click()` colorized with the FULL minimum-phase
mic response. Both are therefore pre-normalization, so they simply add. The
mixture is re-framed and written as a .paudio, and the pipeline applies its own
mic normalization exactly once when it reads the file — exactly as it does for a
genuine recording.

Adding frame-by-frame in the frequency domain would be the tempting shortcut and
is a trap: it forces the click onto the 512-sample grid, needs a hand-written
split for straddling clicks, and re-quantises already-quantised phases. Working
in the time domain makes `t0` a free parameter at any sample, which matters
because every Dryad clip has its envelope peak at index 499 — a constant that
must be broken up by randomising the insertion point, including its sub-frame
phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from . import channel_model as cm
from . import frame_emulator as fe
from .dryad_io import ClipAudio, DryadClip
from .noise_bed import BedWindow
from .pipeline_loader import FrameDataManager, compute_stage1_arrays, load_pipeline

# PlantLeaf's own confirmed clicks, from Dataset_20June2026.csv (91 positives):
# peak_SNR median 12.79, p10 7.19, p90 39.11, heavily right-skewed.
PLANTLEAF_MEDIAN_PEAK_SNR = 12.79
PLANTLEAF_PEAK_SNR_P10 = 7.19
PLANTLEAF_PEAK_SNR_P90 = 39.11

DEFAULT_SPACING_S = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# Amplitude model
# ─────────────────────────────────────────────────────────────────────────────
#
# AMPLITUDE_GLOBAL_G is the one used for anything feeding training.
# AMPLITUDE_FIXED_SNR forces every clip to one target and exists only so the
# visualization PNGs share a common scale.
#
# Why a single global gain, and not a target SNR
# ----------------------------------------------
# Forcing every clip to a fixed peak_SNR makes peak_SNR a LABEL LEAK: injected
# clicks would sit at one value and injected negatives at another, so the feature
# alone identifies the class. But flattening both classes onto one distribution
# is equally wrong -- peak_SNR carries real physical information (single-feature
# AUC 0.823 on the 285 real rows, a top-3 permutation-importance feature).
#
# A single global gain solves both. It IMPORTS Khait's real click-to-click
# amplitude variation instead of inventing a distribution:
#
#     peak_SNR = GLOBAL_G * A_peak / nf_bed
#
# with spread coming from A_peak (Khait's real amplitudes) and nf_bed (PlantLeaf's
# real 2.43-3.49 mV session floors). Both physical; neither hand-specified. The
# click-vs-negative separation then emerges from Khait's own data.
AMPLITUDE_GLOBAL_G = "global-g"
AMPLITUDE_FIXED_SNR = "fixed-snr"
AMPLITUDE_MODES = (AMPLITUDE_GLOBAL_G, AMPLITUDE_FIXED_SNR)

# Volts per Dryad amplitude unit. Derived by `calibrate_global_g()` over 400
# randomly drawn click-class clips (seed 20260813) run through the full channel
# model and `measure_template_peak`, against the median noise floor of the 12
# screened bed sessions (2.987 mV):
#
#     median A_peak = 5370.03   ->   GLOBAL_G = 12.79 * 2.987e-3 / 5370.03
#
# Calibrated on `measure_template_peak` output, NOT the raw int16 peak: peak_SNR
# is defined on the reconstructed envelope, and the mic filter is
# frequency-dependent, so int16 peak -> A_peak is not a pure scalar.
#
# Resulting synthetic distribution vs PlantLeaf's real clicks:
#
#            p10     median    p90
#   synth    5.61    12.79     37.05
#   real     7.19    12.79     39.11
#
# One constant reproduces the real dynamic range to within ~10% at both tails,
# and the synthetic low end extends BELOW the real p10 -- so Khait's detection
# threshold (a hard wall at peak/background ~7-10) has not truncated the range
# that matters. That is why no coverage multiplier is applied: the measurement
# says the range is already there.
GLOBAL_G = 7.1142433e-06

# Reference floor the constant above was calibrated against. Only used to
# re-derive GLOBAL_G; injection always divides by the bed's own measured floor.
CALIBRATION_NF_BED_V = 2.987e-3

# How far from the predicted peak frame a Stage-1 candidate may sit and still be
# considered this injection's detection. A click straddling a frame boundary
# produces candidates at both fi and fi+1, which resolve_click collapses.
_DETECTION_TOLERANCE_FRAMES = 2

# Strata. Greenhouse Noises is acoustically a different room from the anechoic
# chamber the rest of the corpus was recorded in -- background RMS median 942
# against 284-312 -- so its template is windowed to the event before injection
# (channel_model.window_to_event) and its provenance is tagged separately.
STRATUM_ANECHOIC = "anechoic"
STRATUM_GREENHOUSE = "greenhouse"
_GREENHOUSE_CLASSES = ("Greenhouse Noises",)


def stratum_for(class_name: str) -> str:
    """Which acoustic environment a Dryad class was recorded in."""
    return STRATUM_GREENHOUSE if class_name in _GREENHOUSE_CLASSES else STRATUM_ANECHOIC


@dataclass
class InjectionResult:
    """One injected click: what went in, what the pipeline measured coming out."""
    # provenance — source
    clip_id: str
    class_name: str
    species: str
    condition: str
    is_click: bool
    source_path: str
    plant_id: int
    sound_id: int
    # provenance — bed and placement
    bed_id: str
    session_id: str
    room: str
    regime: str
    stratum: str                # anechoic | greenhouse (acoustic environment)
    amplitude_mode: str         # global-g | fixed-snr
    t0_sample: int
    subframe_phase: int
    frame_idx: int
    gain: float
    target_peak_snr: float
    # measured by the pipeline
    detected: bool
    measured_peak_snr: float
    noise_floor: float
    std_noise: float
    features: dict = field(default_factory=dict)
    # payload kept only when the caller asks (rendering); never serialised
    render: dict | None = field(default=None, repr=False)

    def provenance(self) -> dict:
        """
        Flat, JSON-serialisable record.

        Required by the split-leakage checks of spec section 12: augmentation-group
        level (a parent clip and every synthetic child stay on one side of a
        split) and bed level (a bed segment must not appear on both sides).
        """
        out = {
            "clip_id": self.clip_id, "class_name": self.class_name,
            "species": self.species, "condition": self.condition,
            "is_click": self.is_click, "source_path": self.source_path,
            "plant_id": self.plant_id, "sound_id": self.sound_id,
            "bed_id": self.bed_id, "session_id": self.session_id,
            "room": self.room, "regime": self.regime,
            "stratum": self.stratum, "amplitude_mode": self.amplitude_mode,
            "t0_sample": self.t0_sample, "subframe_phase": self.subframe_phase,
            "frame_idx": self.frame_idx, "gain": self.gain,
            "target_peak_snr": self.target_peak_snr,
            "detected": self.detected,
            "measured_peak_snr": self.measured_peak_snr,
            "noise_floor_V": self.noise_floor, "std_noise_V": self.std_noise,
        }
        out.update({f"feat_{k}": v for k, v in self.features.items()})
        return out


def measure_template_peak(template: np.ndarray) -> tuple[float, int]:
    """
    Envelope peak of a colorized click template, measured AFTER the pipeline's
    own reconstruction path.

    This is deliberately not `max(abs(hilbert(template)))` on the raw waveform.
    `peak_SNR` is defined as `peak_amp / noise_floor` where `peak_amp` comes from
    the Hilbert envelope of the *reconstructed* signal
    (click_pipeline_v5.py:1483), and reconstruction band-limits to 20-80 kHz,
    applies a spectral Tukey taper and applies mic normalization — all of which
    change the peak. Scaling against the raw peak would put a systematic bias
    into every injection gain.

    The template is placed centred in the middle of three frames so it sits far
    from the frame edges, where the taper and Gibbs suppression would distort it.
    That gives the clean reference amplitude; the actual injection lands at a
    random sub-frame phase and will differ slightly, which is exactly why the
    measured SNR is recorded rather than the target.

    Returns (peak_amplitude, peak_offset_within_template).
    """
    cp = load_pipeline()
    n = fe.FFT_SIZE

    buf = np.zeros(3 * n, dtype=np.float64)
    start = n + max(0, (n - len(template)) // 2)
    usable = min(len(template), len(buf) - start)
    buf[start:start + usable] = template[:usable]

    mags, phases = fe.frames_from_signal(buf)
    signals = []
    for i in range(3):
        frame = cp.reconstruct_frame_v5(mags[i], phases[i], fe.FS, fe.FFT_SIZE, normalize=True)
        signals.append(frame["signal"] if frame is not None else np.zeros(n))

    ctx = cp.build_click_context(signals[0], signals[1], signals[2])
    envelope = ctx["envelope"]
    peak_idx = int(np.argmax(envelope))

    return float(envelope[peak_idx]), int(peak_idx - start)


def injection_gain(target_peak_snr: float, bed_noise_floor: float, template_peak: float) -> float:
    """
    Fixed-SNR gain: g such that the click lands at approximately `target_peak_snr`.

        peak_SNR = peak_amp / noise_floor  ->  g = S * nf_bed / A_peak

    First-order only: the Hilbert envelope of a sum is not the sum of the
    envelopes, so bed noise at the peak shifts the result either way. Spec
    section 15.6 — always store the pipeline's measured value as ground truth,
    never this target.

    Use this ONLY for visualization output. It flattens every clip onto one
    amplitude and so makes peak_SNR a label leak; see `AMPLITUDE_GLOBAL_G`.
    """
    if template_peak <= 0:
        raise ValueError("template has zero amplitude; cannot scale it")
    return float(target_peak_snr) * float(bed_noise_floor) / float(template_peak)


def calibrate_global_g(template_peaks, *, target_median_snr: float = PLANTLEAF_MEDIAN_PEAK_SNR,
                       nf_bed: float = CALIBRATION_NF_BED_V) -> float:
    """
    Re-derive `GLOBAL_G` from a sample of measured template peaks.

    `template_peaks` must come from `measure_template_peak()` on click-class
    clips that have been through the full channel model — not raw int16 peaks.

    Kept in the module rather than in a one-off script so the constant's
    derivation travels with the constant, and so it can be re-run when the corpus
    or the channel model changes.
    """
    peaks = np.asarray(list(template_peaks), dtype=np.float64)
    peaks = peaks[peaks > 0]
    if len(peaks) < 20:
        raise ValueError(f"need at least 20 valid template peaks, got {len(peaks)}")
    return float(target_median_snr) * float(nf_bed) / float(np.median(peaks))


def global_gain(template_peak: float) -> float:
    """
    Global-gain injection: one constant for every clip, click and negative alike.

    Deliberately ignores both the clip's own amplitude and the bed's floor. The
    resulting peak_SNR = GLOBAL_G * A_peak / nf_bed then varies exactly as the
    two physical quantities do, which is the whole point — see AMPLITUDE_GLOBAL_G.

    Takes `template_peak` only to reject degenerate templates, so that both gain
    functions have the same failure behaviour.
    """
    if template_peak <= 0:
        raise ValueError("template has zero amplitude; cannot scale it")
    return GLOBAL_G


def plan_placements(bed: BedWindow, n_clicks: int, template_len: int,
                    rng: np.random.Generator, *, spacing_s: float = DEFAULT_SPACING_S) -> list[int]:
    """
    Choose `n_clicks` insertion samples inside the bed's usable region.

    Clicks are placed on a regular grid of `spacing_s` and then jittered inside
    their own slot. The jitter does two jobs at once: it decorrelates the
    injection times, and because it is drawn in samples rather than frames it
    gives each click a random sub-frame phase relative to the 512-sample grid.
    That is required, not cosmetic — every Dryad clip has its envelope peak at
    index 499, so without jitter every synthetic positive would share an
    identical peak-within-frame offset, and frame-straddling behaviour would
    never be exercised.

    Raises if the bed is too short for the requested layout, rather than silently
    packing clicks closer together than asked.
    """
    spacing = int(round(spacing_s * fe.FS))
    if spacing <= template_len:
        raise ValueError(
            f"spacing {spacing_s}s = {spacing} samples does not fit a "
            f"{template_len}-sample template"
        )

    start = bed.usable_start_sample
    available = len(bed.signal) - start - template_len
    needed = n_clicks * spacing
    if available < needed:
        raise ValueError(
            f"bed '{bed.bed_id}' has {available / fe.FS:.1f}s usable after warm-up, "
            f"but {n_clicks} clicks at {spacing_s}s spacing need {needed / fe.FS:.1f}s"
        )

    slack = spacing - template_len
    return [int(start + i * spacing + rng.integers(0, slack)) for i in range(n_clicks)]


def inject_batch(bed: BedWindow, clips: Sequence[ClipAudio],
                 rng: np.random.Generator, *,
                 amplitude_mode: str = AMPLITUDE_GLOBAL_G,
                 target_peak_snr: float = PLANTLEAF_MEDIAN_PEAK_SNR,
                 spacing_s: float = DEFAULT_SPACING_S,
                 keep_render_payload: bool = True) -> tuple[np.ndarray, np.ndarray, list[InjectionResult]]:
    """
    Inject a batch of Dryad clips into one bed and analyse the result.

    Implements spec section 9 steps 1-10 for every clip, sharing one bed and one
    pipeline pass. Returns `(mags, phases, results)` — the frames are ready for
    `frame_emulator.write_paudio`.

    `amplitude_mode` selects how loud each clip is injected:
      AMPLITUDE_GLOBAL_G  — one constant for every clip. Use for anything feeding
                            training; preserves Khait's real amplitude spread.
      AMPLITUDE_FIXED_SNR — every clip forced to `target_peak_snr`. Visualization
                            only: one common scale makes PNGs comparable, but it
                            makes peak_SNR a label leak, so never train on it.

    All clicks in a batch share a bed, which makes the batch the natural
    bed-level grouping unit for splitting (spec section 12.4): keep a batch's
    clicks on one side of any train/validation split.

    Clips that fail to trip Stage 1 are still fully analysed and returned with
    `detected=False`. Rendering a PNG for every clip is the point of this run, and
    a missed detection is itself information worth seeing.
    """
    cp = load_pipeline()
    if amplitude_mode not in AMPLITUDE_MODES:
        raise ValueError(f"amplitude_mode must be one of {AMPLITUDE_MODES}, got {amplitude_mode!r}")

    # ── 1-3. Colorize each clip and measure its reconstructed peak ────────────
    # Greenhouse clips are windowed to the event first: their chamber background
    # is ~3x the anechoic set's and would otherwise be transplanted into the bed.
    templates, peaks, offsets, strata = [], [], [], []
    for audio in clips:
        stratum = stratum_for(audio.clip.class_name)
        template = cm.prepare_dryad_click(
            audio.samples, window_event=(stratum == STRATUM_GREENHOUSE))
        peak, offset = measure_template_peak(template)
        templates.append(template)
        peaks.append(peak)
        offsets.append(offset)
        strata.append(stratum)

    template_len = max(len(t) for t in templates)

    # ── 4-6. Gain, then randomised placement ─────────────────────────────────
    placements = plan_placements(bed, len(clips), template_len, rng, spacing_s=spacing_s)
    if amplitude_mode == AMPLITUDE_GLOBAL_G:
        gains = [global_gain(p) for p in peaks]
    else:
        gains = [injection_gain(target_peak_snr, bed.noise_floor, p) for p in peaks]

    # ── 7. Mix in the time domain, raw recorded space ─────────────────────────
    mixed = bed.signal.copy()
    for template, gain, t0 in zip(templates, gains, placements):
        mixed[t0:t0 + len(template)] += gain * template

    # ── 8. Re-frame and run the pipeline as if live-acquired ─────────────────
    mags, phases = fe.frames_from_signal(mixed)
    arrays = compute_stage1_arrays(mags, phases, fs=fe.FS, fft_size=fe.FFT_SIZE)
    dm = FrameDataManager(mags, phases, fs=fe.FS, fft_size=fe.FFT_SIZE)
    dm.attach_stage1_arrays(arrays)
    candidates = cp.run_stage1_v5_precomputed(dm, k=cp.K_STAGE1_DEFAULT)
    candidate_frames = {c["frame_idx"]: c for c in candidates}

    # ── 9-10. Measure and record ─────────────────────────────────────────────
    results = []
    for audio, template, gain, t0, offset, stratum in zip(
            clips, templates, gains, placements, offsets, strata):
        expected_peak = t0 + offset
        frame_idx = expected_peak // fe.FFT_SIZE

        detected = False
        anchor = frame_idx
        for delta in range(-_DETECTION_TOLERANCE_FRAMES, _DETECTION_TOLERANCE_FRAMES + 1):
            if frame_idx + delta in candidate_frames:
                detected = True
                anchor = frame_idx + delta
                break

        # In global-g mode there is no per-clip target; record what the amplitude
        # model implies for this clip so target-vs-measured stays checkable.
        target = (target_peak_snr if amplitude_mode == AMPLITUDE_FIXED_SNR
                  else gain * peaks[len(results)] / max(bed.noise_floor, 1e-30))

        result = _analyse_injection(
            cp, mags, phases, arrays, anchor, audio.clip, bed,
            t0=t0, gain=gain, target=target, detected=detected,
            amplitude_mode=amplitude_mode, stratum=stratum,
            template=template if keep_render_payload else None,
            native=audio.samples if keep_render_payload else None,
        )
        results.append(result)

    return mags, phases, results


def _analyse_injection(cp, mags, phases, arrays, frame_idx: int, clip: DryadClip,
                       bed: BedWindow, *, t0: int, gain: float, target: float,
                       detected: bool, amplitude_mode: str, stratum: str,
                       template, native) -> InjectionResult:
    """
    Reconstruct the prev|curr|next context around `frame_idx` and extract features.

    Uses the same three-frame stitched context the live detector uses, so a click
    straddling a frame boundary is resolved on one continuous trace rather than
    truncated at the 512-sample edge.

    The noise floor comes from the per-frame arrays at this frame — the value the
    pipeline itself would report there — not from the bed's summary statistic, so
    slow drift within the bed is reflected honestly.
    """
    n_frames = len(mags)

    def reconstruct(i):
        if i < 0 or i >= n_frames:
            return None
        frame = cp.reconstruct_frame_v5(mags[i], phases[i], fe.FS, fe.FFT_SIZE, normalize=True)
        return frame

    curr = reconstruct(frame_idx)
    prev = reconstruct(frame_idx - 1)
    nxt = reconstruct(frame_idx + 1)
    if curr is None:
        raise RuntimeError(f"frame {frame_idx} of {n_frames} could not be reconstructed")

    ctx = cp.build_click_context(
        prev["signal"] if prev else None,
        curr["signal"],
        nxt["signal"] if nxt else None,
    )

    noise_floor = float(arrays["noise_floor_arr"][frame_idx])
    std_noise = float(arrays["std_noise_arr"][frame_idx])

    resolved = cp.resolve_click(ctx, noise_floor, std_noise)
    features = cp.compute_features_v5(
        ctx, resolved, curr["fft_norm"], curr["freq_axis"], noise_floor, std_noise, fe.FS
    )

    # Kept because decay_len explains most of the variation in tau and R2, and
    # short windows are exactly where Dryad clicks differ from PlantLeaf's
    # (median 24 vs 39). No dead-zone flag any more: the [13,21] hole in
    # _fit_decay_segment was fixed August 2026.
    features["decay_len_samples"] = float(resolved["decay_end"] - resolved["decay_start"])

    render = None
    if template is not None:
        render = {
            "ctx": ctx, "resolved": resolved,
            "fft_norm": curr["fft_norm"], "freq_axis": curr["freq_axis"],
            "template_200k": template, "native_500k": native,
            "noise_floor": noise_floor, "std_noise": std_noise,
        }

    return InjectionResult(
        clip_id=clip.clip_id, class_name=clip.class_name, species=clip.species,
        condition=clip.condition, is_click=clip.is_click, source_path=str(clip.path),
        plant_id=clip.plant_id, sound_id=clip.sound_id,
        bed_id=bed.bed_id, session_id=bed.session_id, room=bed.room, regime=bed.regime,
        stratum=stratum, amplitude_mode=amplitude_mode,
        t0_sample=int(t0), subframe_phase=int(t0 % fe.FFT_SIZE), frame_idx=int(frame_idx),
        gain=float(gain), target_peak_snr=float(target), detected=bool(detected),
        measured_peak_snr=float(features.get("peak_SNR", float("nan"))),
        noise_floor=noise_floor, std_noise=std_noise,
        features={k: float(v) for k, v in features.items()},
        render=render,
    )
