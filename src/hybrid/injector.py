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
# peak_SNR median 12.8, p10 7.2, p90 39.1, heavily right-skewed.
PLANTLEAF_MEDIAN_PEAK_SNR = 12.8

# The augmentation ladder for stage 2 — spanning p10 to p90 of the distribution
# above. Step 1 renders one image per clip, so it draws a single target.
SNR_LADDER = (7.0, 13.0, 25.0, 40.0)

DEFAULT_SPACING_S = 1.0

# How far from the predicted peak frame a Stage-1 candidate may sit and still be
# considered this injection's detection. A click straddling a frame boundary
# produces candidates at both fi and fi+1, which resolve_click collapses.
_DETECTION_TOLERANCE_FRAMES = 2

# Decay-window lengths for which `_fit_decay_segment` ALWAYS returns tau = -1,
# even on a noiseless exponential. Cause (click_pipeline_v5.py:1110-1117): the
# Gaussian smoothing uses mode='valid', which drops len(kernel)-1 = 12 samples,
# but the convolution is skipped entirely when the segment is shorter than the
# 13-sample kernel. So n_fit jumps 12 -> 1 as decay_len goes 12 -> 13, and does
# not clear MIN_FIT_SAMPLES = 10 again until decay_len reaches 22.
#
# Verified against a pure exponential: decay_len 12 -> tau 0.150 ms, 13..21 ->
# tau -1, 22 -> tau 0.150 ms.
#
# This is a pre-existing defect in the shared pipeline, not something injection
# introduces, and it is deliberately NOT patched here: changing feature
# computation would invalidate the trained SVM and every exported dataset.
# It happens to be latent for PlantLeaf's own data (0 of 285 rows in
# Dataset_20June2026.csv land in the zone; decay windows there run 39 samples
# median for clicks, 82 for negatives), but Dryad clicks decay faster and hit it
# often at low target SNR. Flagged per-row so the affected rows can be excluded
# rather than silently read as "no decay".
DEAD_ZONE_LO = 13
DEAD_ZONE_HI = 21


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
    g such that the injected click lands at approximately `target_peak_snr`.

        peak_SNR = peak_amp / noise_floor  ->  g = S * nf_bed / A_peak

    First-order only: the Hilbert envelope of a sum is not the sum of the
    envelopes, so bed noise at the peak shifts the result either way. Spec
    section 15.6 — always store the pipeline's measured value as ground truth,
    never this target.
    """
    if template_peak <= 0:
        raise ValueError("template has zero amplitude; cannot scale it")
    return float(target_peak_snr) * float(bed_noise_floor) / float(template_peak)


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
                 target_peak_snr: float = PLANTLEAF_MEDIAN_PEAK_SNR,
                 spacing_s: float = DEFAULT_SPACING_S,
                 keep_render_payload: bool = True) -> tuple[np.ndarray, np.ndarray, list[InjectionResult]]:
    """
    Inject a batch of Dryad clips into one bed and analyse the result.

    Implements spec section 9 steps 1-10 for every clip, sharing one bed and one
    pipeline pass. Returns `(mags, phases, results)` — the frames are ready for
    `frame_emulator.write_paudio`.

    All clicks in a batch share a bed, which makes the batch the natural
    bed-level grouping unit for splitting (spec section 12.4): keep a batch's
    clicks on one side of any train/validation split.

    Clips that fail to trip Stage 1 are still fully analysed and returned with
    `detected=False`. Rendering a PNG for every clip is the point of this run, and
    a missed detection is itself information worth seeing.
    """
    cp = load_pipeline()

    # ── 1-3. Colorize each clip and measure its reconstructed peak ────────────
    templates, peaks, offsets = [], [], []
    for audio in clips:
        template = cm.prepare_dryad_click(audio.samples)
        peak, offset = measure_template_peak(template)
        templates.append(template)
        peaks.append(peak)
        offsets.append(offset)

    template_len = max(len(t) for t in templates)

    # ── 4-6. Gain from the bed's own measured floor; randomised placement ─────
    placements = plan_placements(bed, len(clips), template_len, rng, spacing_s=spacing_s)
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
    for audio, template, gain, t0, offset in zip(clips, templates, gains, placements, offsets):
        expected_peak = t0 + offset
        frame_idx = expected_peak // fe.FFT_SIZE

        detected = False
        anchor = frame_idx
        for delta in range(-_DETECTION_TOLERANCE_FRAMES, _DETECTION_TOLERANCE_FRAMES + 1):
            if frame_idx + delta in candidate_frames:
                detected = True
                anchor = frame_idx + delta
                break

        result = _analyse_injection(
            cp, mags, phases, arrays, anchor, audio.clip, bed,
            t0=t0, gain=gain, target=target_peak_snr, detected=detected,
            template=template if keep_render_payload else None,
            native=audio.samples if keep_render_payload else None,
        )
        results.append(result)

    return mags, phases, results


def _analyse_injection(cp, mags, phases, arrays, frame_idx: int, clip: DryadClip,
                       bed: BedWindow, *, t0: int, gain: float, target: float,
                       detected: bool, template, native) -> InjectionResult:
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

    decay_len = int(resolved["decay_end"] - resolved["decay_start"])
    features["decay_len_samples"] = float(decay_len)
    features["fit_dead_zone"] = float(DEAD_ZONE_LO <= decay_len <= DEAD_ZONE_HI)

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
        t0_sample=int(t0), subframe_phase=int(t0 % fe.FFT_SIZE), frame_idx=int(frame_idx),
        gain=float(gain), target_peak_snr=float(target), detected=bool(detected),
        measured_peak_snr=float(features.get("peak_SNR", float("nan"))),
        noise_floor=noise_floor, std_noise=std_noise,
        features={k: float(v) for k, v in features.items()},
        render=render,
    )
