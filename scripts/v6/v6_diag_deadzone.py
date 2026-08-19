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
v6_diag_deadzone.py — Diagnostic 2 of SPECTRAL_FEATURES_v6_PROPOSAL.md §7.5.2
=============================================================================

    "The 5.4 % is of Stage-1 candidates, which are overwhelmingly noise.
     HOW MANY OF THE 181 WERE GENUINE CLICKS? That figure is what the dead zone
     actually cost in recall, and it is separate from the loss attributed to the
     R² > 0.1 gate."

This script replays raw .paudio recordings through Stage 1 and the decay-window
search, isolates every candidate whose decay window falls in the former dead zone
(decay_len 13-21 samples), computes the full v5 feature vector for each, and
reports how many of them look click-like.

READ-ONLY. It labels nothing, writes nothing to any dataset, and never touches
Dataset_20June2026.csv beyond reading it to derive plausibility ranges.

WHERE THE "CLICK-LIKE" RANGES COME FROM
---------------------------------------
Not invented. They are percentile envelopes of the 91 CONFIRMED POSITIVES
(label == 1) in Dataset_20June2026.csv, on peak_SNR / kurtosis / rise_time_ms.
An event is counted click-like when all three land inside the [p_lo, p_hi]
envelope of the confirmed clicks. The percentile is a reporting choice, so the
result is reported at several percentiles rather than at one tuned value — the
project rejects magic numbers, and this is a diagnostic, not a classifier.

WHY A CUSTOM READER
-------------------
src/saving/audio_load_progress.py parses .paudio with a per-sample struct.unpack
loop (154 calls per frame). Over the 44.6 GB / ~58 M frame corpus that is days.
The frame layout is a packed C array — 154 x (float32 magnitude, int8 phase),
itemsize 5 — so np.frombuffer with a structured dtype reads it bit-identically
in one call. `--verify-reader` asserts the two agree on a sample of frames.

Usage
-----
    python3 scripts/v6_diag_deadzone.py [ROOT ...] [options]

    ROOT               directories (or .paudio files) to scan recursively.
                       Default: ~/PlantLeaf_dev/audio_tests/OFFICIALS
    --k FLOAT          Stage 1 threshold multiplier (default 1.5 = K_STAGE1_DEFAULT,
                       the data-collection value that produced the CSVs)
    --max-frames N     stop each recording after N frames (smoke test)
    --limit N          process at most N recordings
    --out PATH         write the per-event table here (default:
                       docs/fft_and_ifft/v6_plots/v6_deadzone_events.csv). A NEW
                       diagnostic file; not a dataset, nothing consumes it.
    --restart          discard previous progress instead of resuming
    --verify-batch     assert the batched reconstruction == the per-frame one
    --verify-reader    cross-check the fast reader against struct.unpack, then exit
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import struct
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]

# The dead zone, as fixed in August 2026. Both bounds are DERIVED, not chosen:
#   lower = len(_GAUSS_KERNEL)                     -> smoothing became possible
#   upper = len(_GAUSS_KERNEL) + MIN_FIT_SAMPLES-1 -> ...and finally left enough
# They are re-derived from the module constants below rather than hard-coded.

_DEFAULT_ROOT = Path.home() / "PlantLeaf_dev" / "audio_tests" / "OFFICIALS"
_DATASET = (Path.home() / "PlantLeaf_dev" / "Analisi" / "v5" / "SVM_Training"
            / "Dataset_20June2026.csv")

_HEADER_BYTES = 128
_BINS_PER_FRAME = 154
#: 154 x (float32 magnitude + int8 phase). itemsize is exactly 5 — no padding.
_FRAME_DTYPE = np.dtype([("mag", "<f4"), ("ph", "i1")])


def load_pipeline():
    """Load click_pipeline_v5 by path so core/__init__'s Qt imports never run."""
    path = _REPO / "src" / "core" / "click_pipeline_v5.py"
    spec = importlib.util.spec_from_file_location("click_pipeline_v5", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ─────────────────────────────────────────────────────────────────────────────
# Fast .paudio reader
# ─────────────────────────────────────────────────────────────────────────────

def read_paudio(path: Path, max_frames: int | None = None):
    """
    Read a .paudio v3 file into (mags, phases, header).

    Returns
    -------
    mags   : float32 (n_frames, 154) — raw transmitted magnitudes [V], NOT
             mic-corrected (normalisation happens inside reconstruct_frame_v5).
    phases : int8    (n_frames, 154)
    header : dict
    """
    with open(path, "rb") as fh:
        head = fh.read(_HEADER_BYTES)
        if len(head) != _HEADER_BYTES:
            raise ValueError(f"{path.name}: header incomplete")
        magic = head[0:10].rstrip(b"\x00").decode("ascii", "replace")
        if magic != "PLANTAUDIO":
            raise ValueError(f"{path.name}: bad magic {magic!r}")
        header = {
            "version": struct.unpack("<f", head[10:14])[0],
            "fs": struct.unpack("<I", head[34:38])[0],
            "fft_size": struct.unpack("<I", head[38:42])[0],
        }
        blob = fh.read()

    # A trailing click section, if present, is not frame data.
    click_start = blob.find(b"CLCK")
    if click_start >= 0:
        blob = blob[:click_start]

    stride = _BINS_PER_FRAME * _FRAME_DTYPE.itemsize          # 770 bytes

    # Format v3.0 OLD interleaves a 5-byte (NaN, -1) separator between frames.
    # Same auto-detection as audio_load_progress.py:106-116.
    has_sep = False
    if len(blob) >= stride + 5:
        t_mag = struct.unpack("<f", blob[stride:stride + 4])[0]
        t_ph = struct.unpack("<b", blob[stride + 4:stride + 5])[0]
        has_sep = bool(np.isnan(t_mag) and t_ph == -1)

    step = stride + 5 if has_sep else stride
    n_frames = len(blob) // step
    if max_frames is not None:
        n_frames = min(n_frames, int(max_frames))
    if n_frames == 0:
        return (np.zeros((0, _BINS_PER_FRAME), np.float32),
                np.zeros((0, _BINS_PER_FRAME), np.int8), header)

    usable = memoryview(blob)[:n_frames * step]
    if has_sep:
        # Drop the separator by viewing each frame as `step` raw bytes first.
        raw = np.frombuffer(usable, dtype=np.uint8).reshape(n_frames, step)
        raw = np.ascontiguousarray(raw[:, :stride])
        rec = raw.view(_FRAME_DTYPE).reshape(n_frames, _BINS_PER_FRAME)
    else:
        rec = np.frombuffer(usable, dtype=_FRAME_DTYPE)
        rec = rec.reshape(n_frames, _BINS_PER_FRAME)

    return (np.ascontiguousarray(rec["mag"]),
            np.ascontiguousarray(rec["ph"]), header)


def verify_reader(path: Path, n_check: int = 200) -> bool:
    """Assert the vectorised reader matches a literal struct.unpack transcription."""
    mags, phases, _ = read_paudio(path, max_frames=n_check)
    with open(path, "rb") as fh:
        fh.seek(_HEADER_BYTES)
        blob = fh.read(n_check * (_BINS_PER_FRAME * 5 + 5))

    stride = _BINS_PER_FRAME * 5
    t_mag = struct.unpack("<f", blob[stride:stride + 4])[0]
    t_ph = struct.unpack("<b", blob[stride + 4:stride + 5])[0]
    step = stride + 5 if (np.isnan(t_mag) and t_ph == -1) else stride

    ok = True
    n = min(len(mags), 20)
    for f in range(n):
        base = f * step
        for i in range(_BINS_PER_FRAME):
            off = base + i * 5
            m = struct.unpack("<f", blob[off:off + 4])[0]
            p = struct.unpack("<b", blob[off + 4:off + 5])[0]
            same_m = (m == mags[f, i]) or (np.isnan(m) and np.isnan(mags[f, i]))
            if not same_m or p != phases[f, i]:
                print(f"  MISMATCH frame {f} bin {i}: "
                      f"struct=({m}, {p})  fast=({mags[f, i]}, {phases[f, i]})")
                ok = False
    print(f"  {'PASS' if ok else 'FAIL'}  fast reader == struct.unpack "
          f"over {n} frames x {_BINS_PER_FRAME} bins  ({path.name})")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Batched reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def _batch_reconstruct(cp, mags, phases, fs, fft_size):
    """
    reconstruct_frame_v5 + compute_hilbert_envelope, vectorised over frames.

    Bit-identical to calling them per frame — asserted by --verify-batch. The
    per-frame path costs ~110 us/frame, almost all of it Python overhead
    (per-frame allocations, np.interp for the mic correction, and a 15-iteration
    Python loop for the taper). Over the 58 M-frame corpus that is ~2 h; batching
    turns it into minutes, which is the difference between a diagnostic you can
    iterate on and one you run once overnight.

    Returns (signals, fft_norms, envelopes, freq_axis).
    """
    n = len(mags)
    n_half = fft_size // 2
    freq_axis = np.arange(n_half) * (fs / fft_size)

    # ── Step 1a/1b: zero-pad to the half spectrum, then mic-normalise ────────
    full = np.zeros((n, n_half), dtype=np.float64)
    k = min(mags.shape[1], cp._BIN_END + 1 - cp._BIN_START)
    full[:, cp._BIN_START:cp._BIN_START + k] = mags[:, :k]

    mask = (freq_axis >= cp.BIN_START_HZ) & (freq_axis <= cp.BIN_END_HZ)
    mic_db = np.interp(freq_axis[mask], cp._MIC_FREQ_HZ, cp._MIC_RESP_DB)
    gain = 10.0 ** (-mic_db * cp._MIC_NORM_FRACTION / 20.0)
    fft_norms = full.copy()
    fft_norms[:, mask] *= gain

    # ── Step 1c/1d: complex spectrum, then the analysis-band Tukey taper ─────
    phase_rad = (phases.astype(np.float64) / 128.0) * np.pi
    spec = np.zeros((n, n_half), dtype=np.complex128)
    spec[:, cp._BIN_START:cp._BIN_START + k] = (
        fft_norms[:, cp._BIN_START:cp._BIN_START + k]
        * np.exp(1j * phase_rad[:, :k]))

    n_bins = cp._BIN_END - cp._BIN_START + 1
    taper = np.ones(n_bins)
    taper_len = max(5, round(n_bins * cp.TUKEY_TAPER_FRACTION))
    for i in range(taper_len):
        val = 0.5 * (1.0 - np.cos(np.pi * (i / taper_len)))
        taper[i] = val
        taper[n_bins - 1 - i] = val
    spec[:, cp._BIN_START:cp._BIN_END + 1] *= taper

    # ── Step 1e + 2: restore the raw-FFT scale, then iFFT ────────────────────
    spec *= (fft_size / 2.0)
    signals = np.fft.irfft(spec, n=fft_size, axis=1)

    # ── Step 3: Gibbs suppression (conditional, per frame) ───────────────────
    g = cp.GIBBS_CHECK_SAMPLES
    interior = signals[:, 40:fft_size - 40]
    e_int = np.sqrt(np.mean(interior ** 2, axis=1))
    e_lft = np.sqrt(np.mean(signals[:, :g] ** 2, axis=1))
    e_rgt = np.sqrt(np.mean(signals[:, fft_size - g:] ** 2, axis=1))
    fire = ((e_int >= 1e-15)
            & (e_lft > cp.GIBBS_FACTOR * e_int)
            & (e_rgt > cp.GIBBS_FACTOR * e_int))
    if np.any(fire):
        fade = 0.5 * (1.0 - np.cos(np.pi * np.arange(g) / g))
        signals[np.ix_(fire, np.arange(g))] *= fade
        signals[np.ix_(fire, np.arange(fft_size - g, fft_size))] *= fade[::-1]

    # ── Hilbert envelope, batched ────────────────────────────────────────────
    h = np.zeros(fft_size, dtype=np.float64)
    h[0] = 1.0
    h[1:fft_size // 2] = 2.0
    if fft_size % 2 == 0:
        h[fft_size // 2] = 1.0
    envelopes = np.abs(np.fft.ifft(np.fft.fft(signals, axis=1) * h, axis=1))

    return signals, fft_norms, envelopes, freq_axis


def verify_batch(cp, path: Path, n_check: int = 300) -> bool:
    """Assert the batched path reproduces the shipped per-frame functions."""
    mags, phases, header = read_paudio(path, max_frames=n_check)
    fs = header.get("fs") or cp.FS
    fft_size = header.get("fft_size") or cp.FFT_SIZE
    sig_b, norm_b, env_b, _ = _batch_reconstruct(cp, mags, phases, fs, fft_size)

    d_sig = d_norm = d_env = 0.0
    for i in range(len(mags)):
        fr = cp.reconstruct_frame_v5(mags[i], phases[i], fs, fft_size, normalize=True)
        d_sig = max(d_sig, float(np.max(np.abs(fr["signal"] - sig_b[i]))))
        d_norm = max(d_norm, float(np.max(np.abs(fr["fft_norm"] - norm_b[i]))))
        d_env = max(d_env, float(np.max(np.abs(
            cp.compute_hilbert_envelope(fr["signal"]) - env_b[i]))))

    ok = d_sig < 1e-18 and d_norm < 1e-18 and d_env < 1e-18
    print(f"  {'PASS' if ok else 'FAIL'}  batched == per-frame over {len(mags)} "
          f"frames ({path.name[:38]})")
    print(f"        max|Δsignal|={d_sig:.2e}  max|Δfft_norm|={d_norm:.2e}  "
          f"max|Δenvelope|={d_env:.2e}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 + decay window replay
# ─────────────────────────────────────────────────────────────────────────────

def replay(cp, mags, phases, header, k: float):
    """
    Run Stage 1 over a whole recording, then resolve every surviving candidate.

    Mirrors run_stage1_v5 (click_pipeline_v5.py:862) exactly — same estimator,
    same mic-corrected E_i, same run-length filter — but keeps the reconstructed
    signals it needs instead of re-reading them, and returns decay_len per
    candidate.

    Returns (n_candidates, events) where each event is a dict carrying the
    resolved indices and the full v5 feature vector.
    """
    fs = header.get("fs") or cp.FS
    fft_size = header.get("fft_size") or cp.FFT_SIZE
    n_frames = len(mags)
    band = slice(cp._BIN_START, cp._BIN_END + 1)

    # ── PASS 1: Stage 1 over every frame, storing NO reconstructed signals ───
    # The earlier version kept signals[] and fft_norms[] for all frames. At
    # float64 that is n_frames x 512 x 8 B — 10 GB for a 2.5 M-frame recording,
    # 16 GB with the spectra, which is what got the first run killed. Candidates
    # are sparse (~200 per recording), so pass 2 re-derives only the few frames
    # actually needed and peak memory becomes O(CHUNK) instead of O(n_frames).
    #
    # Chunked, but the estimator is SEQUENTIAL and is carried across chunk
    # boundaries unchanged — a per-chunk estimator would silently re-run warm-up
    # every 8000 frames and change which frames become candidates.
    estimator = cp.AdaptiveNoiseEstimatorV5()
    freq_axis = None
    above = []
    E_series_chunks = []   # full per-frame energy, needed by the v5.1 peak test

    CHUNK = 8_000
    for c0 in range(0, n_frames, CHUNK):
        c1 = min(c0 + CHUNK, n_frames)
        _, norm_c, env_c, freq_axis = _batch_reconstruct(
            cp, mags[c0:c1], phases[c0:c1], fs, fft_size)

        E_all = np.mean(norm_c[:, band] ** 2, axis=1)
        mu_all = np.mean(env_c, axis=1)
        sd_all = np.std(env_c, axis=1)
        del norm_c, env_c
        E_series_chunks.append(np.asarray(E_all, dtype=np.float64).copy())

        for j in range(c1 - c0):
            E_i = float(E_all[j])
            noise = estimator.update(E_i, float(mu_all[j]), float(sd_all[j]))
            if noise["E_hat_floor"] > 0 and E_i > k * noise["E_hat_floor"]:
                above.append({
                    "frame_idx": c0 + j,
                    "E_i": E_i,
                    "E_hat_floor": noise["E_hat_floor"],
                    "noise_floor": noise["noise_floor"],
                    "std_noise": noise["std_noise"],
                })

    E_all_frames = (np.concatenate(E_series_chunks) if E_series_chunks
                    else np.zeros(0))
    if not above:
        return 0, []

    # Candidate selection — delegated to the pipeline's OWN _stage1_select, the
    # same function run_stage1_v5 and run_stage1_v5_precomputed call. This script
    # used to carry a third hand-written copy of the run-length filter; once Stage 1
    # became v5.1 that copy would silently have kept measuring the old algorithm.
    by_frame = {c["frame_idx"]: c for c in above}
    picks = cp._stage1_select(E_all_frames, sorted(by_frame), k=k)
    candidates = [{**by_frame[p["frame_idx"]], **p} for p in picks]

    lo = len(cp._GAUSS_KERNEL)                                  # 13
    hi = len(cp._GAUSS_KERNEL) + cp.MIN_FIT_SAMPLES - 1         # 22 (exclusive)

    # ── PASS 2: reconstruct only the prev|curr|next triples we actually need ──
    events = []
    for cand in candidates:
        i = cand["frame_idx"]
        a = max(0, i - 1)
        b = min(n_frames, i + 2)
        sig_t, norm_t, _, _ = _batch_reconstruct(cp, mags[a:b], phases[a:b],
                                                 fs, fft_size)
        loc = i - a
        ctx = cp.build_click_context(
            sig_t[loc - 1] if loc > 0 else None,
            sig_t[loc],
            sig_t[loc + 1] if loc + 1 < len(sig_t) else None,
        )
        resolved = cp.resolve_click(ctx, cand["noise_floor"], cand["std_noise"])
        decay_len = int(resolved["decay_end"] - resolved["decay_start"])
        if not (lo <= decay_len < hi):
            continue

        feats = cp.compute_features_v5(ctx, resolved, norm_t[loc], freq_axis,
                                       cand["noise_floor"], cand["std_noise"], fs)
        peak_abs, canonical = cp.click_event_key(ctx, resolved, i)
        events.append({
            "frame_idx": i,
            "canonical_frame_idx": canonical,
            "peak_abs": peak_abs,
            "decay_len": decay_len,
            **{key: feats.get(key) for key in cp_feature_names()},
        })

    return len(candidates), events


def cp_feature_names():
    """The 17 v5 features, in the order compute_features_v5 emits them."""
    return ["peak_SNR", "pre_SNR", "post_SNR",
            "rise_time_ms", "fall_time_ms", "asymmetry_integral",
            "ZCR_pre", "ZCR_click", "ZCR_post",
            "kurtosis", "centroid_shift_hz",
            "tau_ms", "R2", "fit_coverage",
            "SPR", "R_spectral", "FPE_hz"]


# ─────────────────────────────────────────────────────────────────────────────
# Plausibility envelope from the confirmed positives
# ─────────────────────────────────────────────────────────────────────────────

_PLAUSIBILITY_KEYS = ("peak_SNR", "kurtosis", "rise_time_ms")


def positive_envelope(dataset: Path, pct: float):
    """
    [p, 100-p] percentile box of the confirmed clicks (label == 1), on the three
    features the brief names. Returns {feature: (lo, hi)} or None if unavailable.
    """
    if not dataset.exists():
        return None
    rows = []
    with open(dataset, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                if float(row.get("label") or "nan") != 1.0:
                    continue
                rows.append([float(row[k]) for k in _PLAUSIBILITY_KEYS])
            except (TypeError, ValueError):
                continue
    if len(rows) < 10:
        return None
    arr = np.asarray(rows, dtype=np.float64)
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    return {k: (float(np.percentile(arr[:, j], pct)),
                float(np.percentile(arr[:, j], 100.0 - pct)))
            for j, k in enumerate(_PLAUSIBILITY_KEYS)}, len(arr)


def count_clicklike(events, box) -> int:
    n = 0
    for ev in events:
        vals = [ev.get(k) for k in _PLAUSIBILITY_KEYS]
        if any(v is None or not np.isfinite(v) for v in vals):
            continue
        if all(box[k][0] <= ev[k] <= box[k][1] for k in _PLAUSIBILITY_KEYS):
            n += 1
    return n


# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*", default=[str(_DEFAULT_ROOT)])
    ap.add_argument("--k", type=float, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(_REPO / "docs" / "fft_and_ifft" / "v6_plots"
                                     / "v6_deadzone_events.csv"))
    ap.add_argument("--verify-reader", action="store_true")
    ap.add_argument("--verify-batch", action="store_true",
                    help="assert the batched reconstruction == the per-frame one")
    ap.add_argument("--restart", action="store_true",
                    help="discard previous progress instead of resuming")
    args = ap.parse_args()

    files: list[Path] = []
    for root in args.roots:
        p = Path(root).expanduser()
        if p.is_file() and p.suffix == ".paudio":
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.rglob("*.paudio")))
    files = sorted(set(files))
    if args.limit:
        files = files[:args.limit]

    if not files:
        print("no .paudio files found", file=sys.stderr)
        return 1

    cp = load_pipeline()
    k = args.k if args.k is not None else cp.K_STAGE1_DEFAULT

    if args.verify_reader:
        print("\nFast-reader equivalence")
        return 0 if all(verify_reader(f) for f in files[:3]) else 1

    if args.verify_batch:
        print("\nBatched-reconstruction equivalence")
        return 0 if all(verify_batch(cp, f) for f in files[:3]) else 1

    lo = len(cp._GAUSS_KERNEL)
    hi = len(cp._GAUSS_KERNEL) + cp.MIN_FIT_SAMPLES - 1
    print(f"\nDiagnostic 2 — dead-zone candidate census")
    print(f"  dead zone      : decay_len {lo}..{hi - 1} "
          f"(len(_GAUSS_KERNEL)={lo}, MIN_FIT_SAMPLES={cp.MIN_FIT_SAMPLES})")
    print(f"  Stage 1 k      : {k}")
    print(f"  recordings     : {len(files)}")
    print(f"  total size     : {sum(f.stat().st_size for f in files) / 1e9:.1f} GB\n")

    # Resume support. The first attempt at this ran for 20 minutes and was killed
    # (it kept every reconstructed frame in RAM), losing everything because the
    # per-event table was only written at the end. Now every recording is
    # appended as it completes and a rerun skips what is already done.
    out = Path(args.out)
    tally = out.with_suffix(".progress.csv")
    cols = (["file", "frame_idx", "canonical_frame_idx", "peak_abs", "decay_len"]
            + cp_feature_names())

    done: dict = {}
    all_events: list[dict] = []
    if tally.exists() and not args.restart:
        with open(tally, newline="") as fh:
            for row in csv.DictReader(fh):
                done[row["file"]] = int(row["n_candidates"])
        if out.exists():
            with open(out, newline="") as fh:
                all_events = [r for r in csv.DictReader(fh) if r["file"] in done]
        print(f"  resuming: {len(done)} recordings already done, "
              f"{len(all_events)} dead-zone events on disk\n")
    else:
        for p in (out, tally):
            p.unlink(missing_ok=True)

    if not tally.exists():
        with open(tally, "w", newline="") as fh:
            csv.writer(fh).writerow(["file", "n_candidates", "n_deadzone"])
    if not out.exists():
        with open(out, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore").writeheader()

    total_candidates = sum(done.values())
    t_start = time.time()

    for n, path in enumerate(files, 1):
        if path.stem in done:
            print(f"[{n:3d}/{len(files)}] {path.name[:58]:58s}  (already done)")
            continue
        t0 = time.time()
        try:
            mags, phases, header = read_paudio(path, args.max_frames)
            n_cand, events = replay(cp, mags, phases, header, k)
            del mags, phases
        except Exception as exc:                                  # noqa: BLE001
            print(f"[{n:3d}/{len(files)}] {path.name[:58]:58s}  SKIPPED  {exc}",
                  flush=True)
            continue

        total_candidates += n_cand
        for ev in events:
            ev["file"] = path.stem
        all_events.extend(events)

        with open(out, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore").writerows(events)
        with open(tally, "a", newline="") as fh:
            csv.writer(fh).writerow([path.stem, n_cand, len(events)])

        print(f"[{n:3d}/{len(files)}] {path.name[:58]:58s}  "
              f"cand={n_cand:5d}  deadzone={len(events):4d}  "
              f"{time.time() - t0:6.1f}s", flush=True)

    n_dz = len(all_events)
    print(f"\n{'=' * 78}")
    print(f"Stage-1 candidates          : {total_candidates}")
    print(f"In the dead zone ({lo}-{hi - 1})   : {n_dz}"
          + (f"   ({100.0 * n_dz / total_candidates:.1f} %)" if total_candidates else ""))
    print(f"Elapsed                     : {(time.time() - t_start) / 60:.1f} min")

    if n_dz:
        fitted = sum(1 for e in all_events
                     if e.get("tau_ms") is not None and e["tau_ms"] > 0)
        print(f"\nOf those {n_dz}:")
        print(f"  produce a valid fit now (tau > 0) : {fitted}"
              f"   ({100.0 * fitted / n_dz:.1f} %)")
        print(f"  still fail the slope test         : {n_dz - fitted}")

        print("\nClick-likeness vs the confirmed positives "
              f"({', '.join(_PLAUSIBILITY_KEYS)}):")
        got = None
        for pct in (0.0, 1.0, 5.0, 10.0):
            res = positive_envelope(_DATASET, pct)
            if res is None:
                break
            box, n_pos = res
            got = (box, n_pos, pct)
            n_like = count_clicklike(all_events, box)
            print(f"  within the [{pct:4.1f}, {100 - pct:5.1f}] percentile box of "
                  f"{n_pos} confirmed clicks : {n_like:4d}"
                  f"   ({100.0 * n_like / n_dz:.1f} %)")
        if got is not None:
            box, n_pos, _ = got
            print("\n  (widest box, p0-p100, i.e. the full observed range of the "
                  "confirmed clicks:)")
            full, _n = positive_envelope(_DATASET, 0.0)
            for key, (a, b) in full.items():
                print(f"     {key:16s} [{a:.4g}, {b:.4g}]")
        else:
            print("  dataset unavailable — plausibility box not computed")

        out = Path(args.out)
        cols = (["file", "frame_idx", "canonical_frame_idx", "peak_abs", "decay_len"]
                + cp_feature_names())
        with open(out, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(all_events)
        print(f"\nPer-event table -> {out}")
        print("(diagnostic output only — not a dataset, nothing consumes it, "
              "no labels assigned)")

    print(f"{'=' * 78}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
