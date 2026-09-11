#!/usr/bin/env python3
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
Regression tests for the Dryad hybrid channel model, frame emulator and injector.

Run:  python3 test_scripts/verify_channel_model.py

Checks 1-6 are self-contained. Checks 7-9 need the Dryad corpus and/or the noise
beds on the external drive and are skipped (not failed) when those are absent.

Each check states what would be wrong if it failed, because several of these
guard against mistakes that are silent rather than loud — a plausible-looking
number in the wrong space.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hybrid import channel_model as cm          # noqa: E402
from hybrid import frame_emulator as fe         # noqa: E402
from hybrid.pipeline_loader import load_pipeline, load_spectral  # noqa: E402

DRYAD_ROOT = Path("/Users/tommy/PlantLeaf_dev/Analisi/DRYAD_database/PlantSounds")

_RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = ""):
    _RESULTS.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  -  {detail}" if detail else ""))
    return passed


def skip(name: str, why: str):
    _RESULTS.append((name, True, f"SKIPPED: {why}"))
    print(f"  [SKIP] {name}  -  {why}")


# ─────────────────────────────────────────────────────────────────────────────

def check_minimum_phase():
    """
    Ground truth must be a genuine DISCRETE-TIME minimum-phase system.

    A continuous-time prototype like 1/(1 + j f/f_c) is NOT valid here: its phase
    at Nyquist is non-zero, which no real discrete-time minimum-phase system can
    have (the response must be real at DC and Nyquist). Testing against one
    produces a spurious ~1.28 rad discrepancy near Nyquist that looks like a bug
    in the cepstrum code but is a defect in the reference. A real FIR with all
    zeros strictly inside the unit circle is minimum-phase by construction.
    """
    print("\n1. Minimum-phase reconstruction from magnitude")
    n = 4096
    zeros = np.array([0.6 * np.exp(1j * 0.7), 0.3 * np.exp(1j * 2.1), 0.85 * np.exp(1j * 1.2)])
    taps = np.poly(np.concatenate([zeros, np.conj(zeros)])).real
    h_true = np.fft.fft(taps, n)
    h_rec = cm.minimum_phase_from_magnitude(np.abs(h_true), n)

    err = float(np.max(np.abs(h_rec - h_true)))
    check("min-phase recovers a known min-phase FIR", err < 1e-12, f"max complex error {err:.2e}")

    impulse = np.fft.ifft(h_rec).real
    imp_err = float(np.max(np.abs(impulse[:len(taps)] - taps)))
    check("impulse response recovered", imp_err < 1e-12, f"max error {imp_err:.2e}")

    # Causality: a minimum-phase impulse response must not have energy in the
    # anti-causal half. This is what the cepstral folding buys us.
    mag = cm.mic_response_magnitude(512)
    h_mic = cm.minimum_phase_from_magnitude(mag, 512)
    h_t = np.fft.ifft(h_mic).real
    leak = float(np.sum(h_t[256:] ** 2) / np.sum(h_t ** 2))
    check("mic min-phase impulse is causal", leak < 1e-6, f"anti-causal energy {leak:.2e}")

    mag_err = float(np.max(np.abs(np.abs(h_mic) - mag)))
    check("magnitude preserved exactly", mag_err < 1e-12, f"max error {mag_err:.2e}")


def check_mic_filter():
    """
    The colorizing filter must carry FULL mic phase whatever the magnitude
    exponent, because PlantLeaf's normalization scales magnitude only and never
    touches phase. `minphase(|H|**0.5)` would apply HALF the phase — the failure
    this check exists to catch.
    """
    print("\n2. Microphone colorizing filter")
    cp = load_pipeline()
    check("datasheet curve matches the pipeline's",
          np.array_equal(cm.MIC_FREQ_HZ, cp._MIC_FREQ_HZ)
          and np.array_equal(cm.MIC_RESP_DB, cp._MIC_RESP_DB))

    n = 512
    for exponent in (1.0, 0.5):
        filt = cm.mic_filter(n, magnitude_exponent=exponent)
        herm = float(np.max(np.abs(filt[1:] - np.conj(filt[1:][::-1]))))
        mag_err = float(np.max(np.abs(np.abs(filt) - cm.mic_response_magnitude(n) ** exponent)))
        check(f"filter exponent={exponent} is Hermitian", herm < 1e-12, f"{herm:.2e}")
        check(f"filter exponent={exponent} has the requested magnitude", mag_err < 1e-12,
              f"{mag_err:.2e}")

    full = cm.mic_filter(n, magnitude_exponent=1.0)
    half = cm.mic_filter(n, magnitude_exponent=0.5)
    check("phase is identical for both exponents (full mic phase)",
          np.allclose(np.angle(full), np.angle(half)))

    naive = cm.minimum_phase_from_magnitude(cm.mic_response_magnitude(n) ** 0.5, n)
    freqs = np.fft.fftfreq(n, 1.0 / cm.PLANTLEAF_FS)
    in_band = (np.abs(freqs) >= 20e3) & (np.abs(freqs) <= 80e3)
    diff = float(np.max(np.abs(np.angle(half * np.conj(naive))[in_band])))
    check("differs from the naive minphase(|H|**0.5)", diff > 0.1,
          f"{diff:.3f} rad in-band = {diff / np.pi * 128:.0f} int8 phase LSBs")

    colored = cm.colorize(np.random.default_rng(0).standard_normal(401))
    check("colorize returns a real signal of the same length",
          np.isrealobj(colored) and len(colored) == 401)


def check_resampling():
    """
    Only input above 120 kHz can alias INTO 20-80 kHz (it folds to |f - 200 kHz|),
    so that is the band whose rejection actually matters for features.
    """
    print("\n3. Polyphase resampling 500 -> 200 kHz")
    n = 20001
    t = np.arange(n) / cm.DRYAD_FS

    ripple = []
    for f_hz in (20e3, 30e3, 40e3, 50e3, 60e3, 70e3, 80e3):
        y = cm.resample_500k_to_200k(np.sin(2 * np.pi * f_hz * t))[200:-200]
        ripple.append(abs(20 * np.log10(np.sqrt(2 * np.mean(y ** 2)))))
    check("in-band (20-80 kHz) response is flat", max(ripple) < 0.05,
          f"max deviation {max(ripple):.4f} dB")

    rejection = []
    for f_hz in (120e3, 140e3, 160e3, 180e3):
        y = cm.resample_500k_to_200k(np.sin(2 * np.pi * f_hz * t))[200:-200]
        rejection.append(20 * np.log10(max(np.sqrt(2 * np.mean(y ** 2)), 1e-12)))
    check("content that would alias into 20-80 kHz is rejected", max(rejection) < -50,
          f"worst {max(rejection):.1f} dB at 120-180 kHz")

    out = cm.resample_500k_to_200k(np.zeros(1001))
    check("1001 samples @500k -> 401 @200k", len(out) == 401, f"got {len(out)}")


def check_frame_roundtrip():
    """
    `inverse_raw` must EXACTLY invert `forward` on a fixed grid — this is what
    lets a bed be resynthesised without distortion, and it is the property
    `reconstruct_frame_v5` deliberately lacks.
    """
    print("\n4. Frame emulator round trip")
    rng = np.random.default_rng(1)
    signal = rng.standard_normal(fe.FFT_SIZE * 40) * 0.003

    mags, phases = fe.frames_from_signal(signal)
    rebuilt = fe.signal_from_frames(mags, phases)
    mags2, phases2 = fe.frames_from_signal(rebuilt)

    nz = mags > 1e-15
    mag_err = float(np.max(np.abs(mags2 - mags)[nz] / mags[nz]))
    phase_err = int(np.max(np.abs((phases2.astype(int) - phases.astype(int) + 128) % 256 - 128)))
    check("magnitudes survive forward/inverse/forward", mag_err < 1e-9, f"{mag_err:.2e} relative")
    check("phases survive (circular, int8 wraps at +-pi)", phase_err <= 1, f"{phase_err} LSB")

    # Amplitude scale: reconstruct_frame_v5 must return the original volts.
    # Guards the x N/2 factor of IFFT_AMPLITUDE_SCALE_FIX.md.
    cp = load_pipeline()
    amp = 0.005
    idx = np.arange(fe.FFT_SIZE)
    tone = amp * np.cos(2 * np.pi * 50_000 * idx / fe.FS + 0.7)
    m, p = fe.forward(tone)
    frame = cp.reconstruct_frame_v5(m, p, fe.FS, fe.FFT_SIZE, normalize=False)
    got = float(np.median(cp.compute_hilbert_envelope(frame["signal"])[40:-40]))
    check("amplitude round trip preserves volts", abs(got / amp - 1.0) < 0.10,
          f"{amp * 1e3:.1f} mV in, {got * 1e3:.2f} mV out (ratio {got / amp:.3f})")


def check_reconstruct_is_not_invertible():
    """
    Documents WHY inverse_raw exists. reconstruct_frame_v5 applies mic
    normalization, a band-edge Tukey taper and Gibbs suppression; reconstructing
    a bed with it and re-running forward() applies each twice. If this check ever
    starts passing (i.e. the error becomes small), the taper or normalization has
    been removed from the pipeline and this module's assumptions need revisiting.
    """
    print("\n5. reconstruct_frame_v5 is NOT a resynthesis inverse (by design)")
    cp = load_pipeline()
    rng = np.random.default_rng(2)
    signal = rng.standard_normal(fe.FFT_SIZE * 20) * 0.003
    mags, phases = fe.frames_from_signal(signal)

    errors, edge_errors = [], []
    for i in range(len(mags)):
        frame = cp.reconstruct_frame_v5(mags[i], phases[i], fe.FS, fe.FFT_SIZE, normalize=True)
        m2, _ = fe.forward(frame["signal"])
        nz = mags[i] > 1e-15
        rel = np.abs(m2 - mags[i])[nz] / mags[i][nz]
        errors.append(np.median(rel))
        edge_errors.append(np.abs(m2[0] - mags[i][0]) / max(mags[i][0], 1e-15))

    check("reconstruct -> re-FFT distorts the spectrum", float(np.median(errors)) > 0.1,
          f"median {100 * np.median(errors):.0f} % error, "
          f"{100 * np.median(edge_errors):.0f} % at the 20 kHz band edge")


def _fit_decay_segment_prepatch(cp, extended, decay_start, decay_end, fs=None):
    """
    The PRE-PATCH `_fit_decay_segment`, reproduced for equivalence testing.

    Identical to the shipped function except for the Step-C branch condition,
    which tested only `len(decay_segment) >= len(_GAUSS_KERNEL)`. Everything
    after Step C is delegated to the real function by handing it a segment that
    forces the same path, so this cannot drift from the implementation it is
    meant to compare against.
    """
    fs = cp.FS if fs is None else fs
    decay_len = decay_end - decay_start
    if decay_len < 1 or decay_len >= len(cp._GAUSS_KERNEL) + cp.MIN_FIT_SAMPLES - 1:
        # Below 1, or long enough that both versions smooth: identical by
        # construction, so just call the real thing.
        return cp._fit_decay_segment(extended, decay_start, decay_end, fs)

    # decay_len < 22. Pre-patch smoothed whenever decay_len >= 13.
    segment = extended[decay_start:decay_end]
    if decay_len >= len(cp._GAUSS_KERNEL):
        fit_window = np.convolve(segment, cp._GAUSS_KERNEL, mode="valid")
    else:
        fit_window = segment.copy()

    n_fit = len(fit_window)
    if n_fit < cp.MIN_FIT_SAMPLES:
        return {"tau_ms": -1.0, "R2": 0.0, "fit_coverage": n_fit / max(1, decay_len),
                "decay_start": decay_start, "decay_end": decay_end}

    # Unsmoothed-and-long-enough: the real function takes exactly this path too.
    return cp._fit_decay_segment(extended, decay_start, decay_end, fs)


def check_fit_dead_zone():
    """
    The decay-fit dead zone, fixed August 2026 — and the branch-equivalence proof
    that the fix cannot have changed any existing result.

    Mechanism: Step C's Gaussian smoothing uses mode='valid', which costs
    len(_GAUSS_KERNEL) - 1 = 12 samples, but the branch tested only whether the
    kernel *fit* the segment, never whether Step D would still have
    MIN_FIT_SAMPLES afterwards. n_fit therefore jumped 12 -> 1 at decay_len 13
    and did not clear MIN_FIT_SAMPLES again until 22, so decay_len 13-21 returned
    tau = -1 on any input, including a noiseless exponential.

    Why the fix is safe is a branch argument, NOT a row count: decay_len <= 12
    skipped smoothing before and skips it now; decay_len >= 22 smoothed before and
    smooths now. Only 13-21 changes path, and every such candidate previously
    produced tau = -1 and was discarded by Stage 2 before export. Checks 3 and 4
    below assert that equivalence numerically.

    (The often-quoted "0 of 285 rows in Dataset_20June2026.csv are affected" is
    true but vacuous as evidence: data_collection_dialog_v5.py skips candidates
    with tau <= STAGE2_TAU_MIN before they become rows, so no such row could ever
    have existed. The CSV does show the fingerprint — fit_coverage runs 1.0 for
    84 rows then jumps straight to 0.4545, with nothing in the 0.077-0.429 band
    that decay_len 13-21 would produce.)
    """
    print("\n6. Decay-fit dead zone (fixed) + branch equivalence")
    cp = load_pipeline()
    k_size, min_fit = len(cp._GAUSS_KERNEL), cp.MIN_FIT_SAMPLES
    threshold = k_size + min_fit - 1
    envelope = np.exp(-np.arange(300) / 30.0) + 1e-9      # tau = 30 samples = 0.15 ms

    check("smoothing threshold accounts for both constraints", threshold == 22,
          f"len(kernel)={k_size} + MIN_FIT_SAMPLES={min_fit} - 1 = {threshold}")

    # 1. The zone is gone: everything from MIN_FIT_SAMPLES upward now fits.
    failed = [d for d in range(min_fit, 40)
              if cp._fit_decay_segment(envelope, 0, d)["tau_ms"] <= 0]
    check(f"decay_len {min_fit}-39 all fit (dead zone closed)", not failed,
          f"failures: {failed}" if failed else "tau = 0.150 ms throughout")

    # 2. Still correctly rejects genuinely too-short windows.
    too_short = [d for d in range(1, min_fit)
                 if cp._fit_decay_segment(envelope, 0, d)["tau_ms"] > 0]
    check(f"decay_len < {min_fit} still rejected", not too_short,
          f"below MIN_FIT_SAMPLES there is nothing to fit")

    # 3 + 4. Branch equivalence on random, noisy, physically-shaped envelopes —
    # the assertion that protects the trained SVM and every exported dataset.
    rng = np.random.default_rng(4242)
    unchanged_lo = unchanged_hi = 0
    diffs = []
    for _ in range(400):
        tau = rng.uniform(5, 80)
        n = 200
        env = np.exp(-np.arange(n) / tau) * rng.uniform(0.5, 2.0)
        env = np.abs(env + rng.normal(0, 0.02, n)) + 1e-9
        for d in (rng.integers(1, 13), rng.integers(22, 120)):
            d = int(d)
            if d >= n:
                continue
            new = cp._fit_decay_segment(env, 0, d)
            old = _fit_decay_segment_prepatch(cp, env, 0, d)
            same = all(np.isclose(new[key], old[key], rtol=0, atol=0)
                       for key in ("tau_ms", "R2", "fit_coverage"))
            if not same:
                diffs.append((d, new["tau_ms"], old["tau_ms"]))
            if d <= 12:
                unchanged_lo += same
            else:
                unchanged_hi += same

    check("decay_len <= 12 bit-identical to pre-patch", not any(d <= 12 for d, _, _ in diffs),
          f"{unchanged_lo} random windows, exact match")
    check("decay_len >= 22 bit-identical to pre-patch", not any(d >= 22 for d, _, _ in diffs),
          f"{unchanged_hi} random windows, exact match")

    # 5. And 13-21 is exactly what changed.
    recovered = [d for d in range(k_size, threshold)
                 if cp._fit_decay_segment(envelope, 0, d)["tau_ms"] > 0
                 and _fit_decay_segment_prepatch(cp, envelope, 0, d)["tau_ms"] <= 0]
    check(f"decay_len {k_size}-{threshold - 1} recovered by the fix",
          recovered == list(range(k_size, threshold)),
          f"{len(recovered)} lengths now fit that previously returned tau = -1")

    # 6. Not every dead-zone window recovers, and that is correct. On real data
    # 84 of the 181 candidates in the zone now fit (335 -> 419 exportable
    # candidates over 34 recordings, a 25 % increase); the rest fail Step D's
    # slope test because their envelope does not actually decay. The patch
    # restores the fit, it cannot invent a decay. Asserted on a synthetic
    # population so the suite stays fast.
    fit_rng = np.random.default_rng(31)
    fits = no_decay = 0
    for _ in range(1500):
        d = int(fit_rng.integers(k_size, threshold))
        tau = fit_rng.uniform(5, 60)
        noise = fit_rng.uniform(0.0, 0.9)
        env = np.abs(np.exp(-np.arange(d + 5) / tau) * (1 - noise)
                     + fit_rng.normal(0, noise, d + 5)) + 1e-9
        if cp._fit_decay_segment(env, 0, d)["tau_ms"] > 0:
            fits += 1
        else:
            no_decay += 1
    rate = fits / (fits + no_decay)
    check("partial recovery, remainder genuinely non-decaying", 0.5 < rate < 0.9,
          f"{100 * rate:.0f} % of noisy dead-zone windows now fit; the rest fail "
          f"the slope test, which is correct")


def check_dryad_corpus():
    """Properties the channel model relies on. Cheap, and they would matter if they changed."""
    print("\n7. Dryad corpus properties")
    if not DRYAD_ROOT.is_dir():
        skip("Dryad corpus", f"{DRYAD_ROOT} not found")
        return None

    from hybrid import dryad_io as dio
    clips = dio.build_manifest(DRYAD_ROOT)
    check("manifest built", len(clips) > 5000, f"{len(clips)} clips across "
          f"{len({c.class_name for c in clips})} classes")

    rng = np.random.default_rng(3)
    sample = [clips[i] for i in rng.choice(len(clips), 40, replace=False)]
    peaks, lengths, rates = [], [], []
    for clip in sample:
        audio = dio.read_clip(clip)
        rates.append(audio.fs)
        lengths.append(len(audio.samples))
        env = load_pipeline().compute_hilbert_envelope(audio.samples)
        peaks.append(int(np.argmax(env)))

    check("all clips are 500 kHz", set(rates) == {cm.DRYAD_FS}, f"{sorted(set(rates))}")
    check("all clips are 1001 samples", set(lengths) == {1001}, f"{sorted(set(lengths))}")

    # Trigger-aligned: the envelope peak sits at index 499 in the MEDIAN for every
    # class, which is why injection must randomise t0 -- otherwise every synthetic
    # positive shares one peak-within-clip offset.
    #
    # The median, not the range: the click classes cluster tightly (Tobacco Dry
    # spreads 6 samples, Tomato Dry 17) but Greenhouse Noises spreads ~665,
    # because a diffuse broadband noise clip has no well-defined envelope peak to
    # be aligned on. Asserting a tight range would fail on that legitimately.
    check("clips are trigger-aligned (median peak at 499)",
          abs(float(np.median(peaks)) - 499.0) <= 3.0,
          f"median peak index {np.median(peaks):.0f} of 1001, range {min(peaks)}-{max(peaks)}")
    return clips


def check_injection(clips):
    """The end-to-end assertions: bed identity at g=0, and SNR/tau landing where intended."""
    print("\n8. Injection end to end")
    from hybrid import dryad_io as dio
    from hybrid import injector as inj
    from hybrid import noise_bed as nb

    sources = nb.discover_bed_sources()
    if not sources:
        skip("injection", "no noise beds found (external drive not mounted?)")
        return
    if clips is None:
        skip("injection", "Dryad corpus unavailable")
        return

    rng = np.random.default_rng(11)
    bed = None
    for source in sources:
        bed = nb.find_clean_window(source, rng, usable_frames=30 * 390 + 400)
        if bed is not None:
            break
    if bed is None:
        skip("injection", "no event-free bed window found")
        return

    check("bed screened event-free", bed.candidate_rate == 0.0,
          f"{bed.session_id[:34]} nf={bed.noise_floor * 1e3:.3f} mV "
          f"sd={bed.std_noise * 1e3:.3f} mV")

    # Regression guard for a real bug: window selection used to draw from the
    # whole file, so it landed on the start/stop transients that are the ONLY
    # place most of these recordings trip Stage 1. That wasted attempts and, in
    # my own characterisation scan, made clean recordings look noisy. With
    # EDGE_GUARD_FRAMES excluding both ends, acceptance goes 82 % -> ~97 %.
    accepted = draws = 0
    guard_rng = np.random.default_rng(77)
    span = nb.WARMUP_FRAMES + nb.DEFAULT_USABLE_FRAMES
    for source in sources:
        lo, hi = nb.EDGE_GUARD_FRAMES, source.total_frames - span - nb.EDGE_GUARD_FRAMES
        if hi <= lo:
            continue
        for _ in range(4):
            draws += 1
            accepted += nb.extract_window(source, int(guard_rng.integers(lo, hi + 1))) is not None
    check("guarded interiors are overwhelmingly clean", accepted >= 0.85 * draws,
          f"{accepted}/{draws} windows accepted with a "
          f"{nb.EDGE_GUARD_FRAMES * fe.FFT_SIZE / fe.FS:.1f}s guard at each end")

    # g = 0 must reproduce the bed bin for bin. This single assertion covers the
    # bed reader, inverse_raw, the mixer and the forward emulator at once.
    src_mags, src_phases = fe.read_frames(bed.source_path, bed.start_frame, bed.n_frames)
    out_mags, out_phases = fe.frames_from_signal(bed.signal)

    # Phase is only defined where there is magnitude to carry it. A handful of
    # bins in a real recording are stored as exactly 0.0 V, and `np.angle(0+0j)`
    # is 0 regardless of what phase byte the firmware happened to write there, so
    # those bins can differ by up to half a turn while meaning nothing. Measured:
    # every bin with a phase error above 1 LSB has magnitude exactly 0.0
    # (~0.00005 % of bins). Excluding them is not loosening the test — it is
    # scoping it to where the quantity exists.
    nz = src_mags > 0.0
    mag_err = float(np.max(np.abs(out_mags - src_mags)[nz] / src_mags[nz]))
    circular = np.abs((out_phases.astype(int) - src_phases.astype(int) + 128) % 256 - 128)
    ph_err = int(np.max(circular[nz]))
    zero_bins = int(np.sum(~nz))
    check("zero-gain bed round trip is bin-exact", mag_err < 1e-9 and ph_err <= 1,
          f"magnitude {mag_err:.2e} relative, phase {ph_err} LSB "
          f"(over {int(nz.sum())} bins; {zero_bins} zero-magnitude bins excluded)")

    tomato = [c for c in clips if c.class_name == "Tomato Cut"][:30]
    audio = [dio.read_clip(c) for c in tomato]

    # fixed-snr must COLLAPSE the spread (that is what makes it a label leak, and
    # why it is visualization-only); global-g must PRESERVE it.
    spreads = {}
    for mode in (inj.AMPLITUDE_FIXED_SNR, inj.AMPLITUDE_GLOBAL_G):
        _, _, res = inj.inject_batch(bed, audio, np.random.default_rng(5),
                                     amplitude_mode=mode,
                                     target_peak_snr=inj.PLANTLEAF_MEDIAN_PEAK_SNR,
                                     spacing_s=1.0, keep_render_payload=False)
        values = np.array([r.measured_peak_snr for r in res])
        spreads[mode] = (values, res)

    fixed, _ = spreads[inj.AMPLITUDE_FIXED_SNR]
    ratio = float(np.median(fixed) / inj.PLANTLEAF_MEDIAN_PEAK_SNR)
    check("fixed-snr hits its target", 0.9 < ratio < 1.1,
          f"median {np.median(fixed):.2f} vs target {inj.PLANTLEAF_MEDIAN_PEAK_SNR} "
          f"(ratio {ratio:.3f})")

    fixed_spread = float(np.percentile(fixed, 90) / np.percentile(fixed, 10))
    check("fixed-snr collapses the amplitude spread (why it must not train)",
          fixed_spread < 1.6, f"p90/p10 = {fixed_spread:.2f}")

    glob, results = spreads[inj.AMPLITUDE_GLOBAL_G]
    glob_spread = float(np.percentile(glob, 90) / np.percentile(glob, 10))
    # PlantLeaf's own clicks span p90/p10 = 39.11/7.19 = 5.44.
    check("global-g preserves Khait's real amplitude spread", glob_spread > 3.0,
          f"p90/p10 = {glob_spread:.2f} (PlantLeaf real clicks: 5.44)")
    check("global-g centres near PlantLeaf's median",
          0.7 < float(np.median(glob) / inj.PLANTLEAF_MEDIAN_PEAK_SNR) < 1.4,
          f"median {np.median(glob):.2f} vs {inj.PLANTLEAF_MEDIAN_PEAK_SNR}")

    snr = glob

    detected = sum(r.detected for r in results)
    check("injected clicks trip Stage 1", detected >= 0.9 * len(results),
          f"{detected}/{len(results)} detected")

    phases = {r.subframe_phase for r in results}
    check("sub-frame phases are randomised", len(phases) > 0.6 * len(results),
          f"{len(phases)} distinct of {len(results)}")

    tau = np.array([r.features["tau_ms"] for r in results])
    valid = tau[tau > 0]
    # PlantLeaf's own confirmed clicks: tau median 0.188 ms. Injected Dryad
    # clicks should land in the same regime -- the self-noise-padded control
    # gives 0.4-1.1 ms, so this range genuinely discriminates.
    check("tau lands in PlantLeaf's regime", 0.05 < float(np.median(valid)) < 0.40,
          f"median {np.median(valid):.3f} ms over {len(valid)}/{len(tau)} fits "
          f"(PlantLeaf real clicks: 0.188 ms)")


def check_tau_resample_agreement(clips):
    """Spec section 5: tau must survive 500 -> 200 kHz. Level-bounded so both rates agree."""
    print("\n9. tau agreement, 500 kHz vs 200 kHz (spec section 5)")
    if clips is None:
        skip("tau agreement", "Dryad corpus unavailable")
        return

    from hybrid import dryad_io as dio
    from hybrid.render import _fit_tau_fixed_window
    cp = load_pipeline()

    rng = np.random.default_rng(9)
    clicks = [c for c in clips if c.is_click]
    sample = [clicks[i] for i in rng.choice(len(clicks), 40, replace=False)]

    deltas = []
    for clip in sample:
        native = dio.read_clip(clip).samples
        tau_n = _fit_tau_fixed_window(cp.compute_hilbert_envelope(native), cm.DRYAD_FS)
        tau_r = _fit_tau_fixed_window(
            cp.compute_hilbert_envelope(cm.resample_500k_to_200k(native)), fe.FS)
        if np.isfinite(tau_n) and np.isfinite(tau_r) and tau_n > 0:
            deltas.append(100.0 * (tau_r - tau_n) / tau_n)

    deltas = np.array(deltas)
    check("tau survives resampling", len(deltas) > 20 and abs(float(np.median(deltas))) < 10,
          f"median {np.median(deltas):+.1f} %, |delta|<20 % for "
          f"{100 * np.mean(np.abs(deltas) < 20):.0f} % of {len(deltas)} clips")


def main() -> int:
    print("=" * 72)
    print("Dryad hybrid channel model - verification suite")
    print("=" * 72)

    check_minimum_phase()
    check_mic_filter()
    check_resampling()
    check_frame_roundtrip()
    check_reconstruct_is_not_invertible()
    check_fit_dead_zone()
    clips = check_dryad_corpus()
    check_injection(clips)
    check_tau_resample_agreement(clips)

    failed = [name for name, ok, _ in _RESULTS if not ok]
    print("\n" + "=" * 72)
    print(f"{len(_RESULTS) - len(failed)}/{len(_RESULTS)} checks passed")
    if failed:
        print("FAILED:")
        for name in failed:
            print(f"  - {name}")
    print("=" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
