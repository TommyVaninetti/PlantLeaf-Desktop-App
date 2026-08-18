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
v6_ek_plots.py — E[k] on real frames, clicks vs hard negatives
===============================================================

Phase 1 exit criterion: "every feature returns a verified-correct value on
synthetic signals of known spectral content, AND ON A HAND-CHECKED SAMPLE OF REAL
FRAMES." This produces that sample — E[k] for a handful of confirmed clicks and a
handful of confirmed hard negatives, so the subtraction can be eyeballed.

Events are chosen by (file, frame_idx) from the labelled master dataset. The CSV
is opened READ-ONLY and never written.

Each panel shows, on the 12-band grid and on the padded native grid:
    P_region        the click region's PSD           (already Tukey-tapered)
    P_noise·w²      Buffer 3, taper-corrected        (trap (a), §4.4)
    E = max(0, ·)   the excess spectrum

BOTH Buffer-3 modes are drawn, because they disagree by ~80× — see
AdaptiveNoiseEstimatorV5.p_noise_psd. That disagreement is the single most
important thing to look at in these plots.

    python3 scripts/v6_ek_plots.py [--n 4] [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
_DATASET = (Path.home() / "PlantLeaf_dev" / "Analisi" / "v5" / "SVM_Training"
            / "Dataset_20June2026.csv")
_AUDIO_ROOT = Path.home() / "PlantLeaf_dev" / "audio_tests"

#: Frames to replay before the target so Buffer 3 has a full W_NOISE window and
#: the warm-up has elapsed. 3000 frames = 7.7 s = 4x the 750-frame buffer.
#: ⚠️ This is a LOCAL noise estimate, not the from-frame-0 estimate the real
#: pipeline builds. For a noise floor that is arguably more appropriate (ambient
#: noise is non-stationary), but it IS a deviation and the plots say so.
_PRELUDE_FRAMES = 3000


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sa = _load("spectral_analysis", "src/core/spectral_analysis.py")
cp = _load("click_pipeline_v5", "src/core/click_pipeline_v5.py")
_diag = _load("v6_diag_deadzone", "scripts/v6_diag_deadzone.py")


def pick_events(n_each: int):
    """n_each confirmed clicks and n_each hard negatives, spread across files."""
    rows = []
    with open(_DATASET, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                lab = float(r.get("label") or "nan")
                fi = int(float(r["frame_idx"]))
            except (TypeError, ValueError, KeyError):
                continue
            if lab not in (0.0, 1.0) or not r.get("file"):
                continue
            try:
                snr = float(r["peak_SNR"])
            except (TypeError, ValueError):
                continue
            rows.append({"file": r["file"], "frame_idx": fi, "label": int(lab),
                         "peak_SNR": snr})

    def spread(pool):
        """Prefer one event per source file, strongest first."""
        pool = sorted(pool, key=lambda d: -d["peak_SNR"])
        seen, out = set(), []
        for d in pool:
            if d["file"] not in seen:
                seen.add(d["file"])
                out.append(d)
            if len(out) >= n_each:
                break
        return out

    return (spread([r for r in rows if r["label"] == 1]),
            spread([r for r in rows if r["label"] == 0]))


def find_paudio(stem: str):
    for p in _AUDIO_ROOT.rglob("*.paudio"):
        if p.stem == stem:
            return p
    return None


def analyse(path: Path, frame_idx: int):
    """Replay a prelude, then build P_region / P_noise / E for `frame_idx`."""
    start = max(0, frame_idx - _PRELUDE_FRAMES)
    mags, phases, header = _diag.read_paudio(path, max_frames=frame_idx + 2)
    if len(mags) <= frame_idx:
        return None
    fs = header.get("fs") or cp.FS
    fft_size = header.get("fft_size") or cp.FFT_SIZE
    band = slice(cp._BIN_START, cp._BIN_END + 1)

    est = cp.AdaptiveNoiseEstimatorV5()
    sigs = {}
    noise = None
    for i in range(start, frame_idx + 2):
        fr = cp.reconstruct_frame_v5(mags[i], phases[i], fs, fft_size, normalize=True)
        if fr is None:
            continue
        if i >= frame_idx - 1:
            sigs[i] = fr
        E_i = cp.compute_fft_energy(fr["fft_norm"][band])
        envelope = cp.compute_hilbert_envelope(fr["signal"])
        res = est.update(E_i, float(np.mean(envelope)), float(np.std(envelope)),
                         mags_norm=fr["fft_norm"][band], fs=fs, fft_size=fft_size)
        if i == frame_idx:
            noise = res

    if noise is None or frame_idx not in sigs:
        return None

    ctx = cp.build_click_context(sigs.get(frame_idx - 1, {}).get("signal"),
                                 sigs[frame_idx]["signal"],
                                 sigs.get(frame_idx + 1, {}).get("signal"))
    resolved = cp.resolve_click(ctx, noise["noise_floor"], noise["std_noise"])
    i0, i1 = resolved["onset"], resolved["decay_end"]
    if i1 - i0 < sa.MIN_SEGMENT_SAMPLES:
        return None

    region = ctx["signal"][i0:i1 + 1]
    region_spec = sa.compute_spectrum(region, fs, window="tukey", alpha=0.25,
                                      n_fft=4096, scaling="psd")

    taper = cp.analysis_band_taper()
    noise_f = sigs[frame_idx]["freq_axis"][band]
    out = {"n_seg": len(region), "resolved": resolved, "ctx": ctx,
           "noise_freqs": noise_f, "region_spec": region_spec,
           "decay_len": int(resolved["decay_end"] - resolved["decay_start"])}

    # ── TRAP (a), §4.4 ───────────────────────────────────────────────────────
    # P_region comes from the RECONSTRUCTED signal and already carries the
    # analysis-band Tukey taper. Buffer 3 is built from TRANSMITTED magnitudes and
    # does not. Multiply P_noise by taper SQUARED (squared because these are PSDs
    # and the taper acts on amplitude) before subtracting, or the subtraction
    # over-subtracts at the band edges where the region has been attenuated and
    # the noise estimate has not. The taper reaches exactly 0 at bins 51/204 —
    # NOT −6 dB as §4.4(a) and FFT_PHASE_TECHNICAL_SPECIFICATION.md §7.2 claim —
    # so never invert this by dividing E[k] by taper².
    for mode in ("mean", "min"):
        raw = est.p_noise_psd(mode=mode)
        if raw is None:
            continue
        p_noise = raw * taper ** 2
        # ── TRAP (b), §4.4 ───────────────────────────────────────────────────
        # Both sides must come from mic-NORMALISED data. B3 was fed
        # fr['fft_norm'] (mic-corrected) above and the region comes from
        # fr['signal'], reconstructed from the same normalised spectrum, so the
        # frequency-dependent 0.55x-1.49x gain divides out. Feeding B3 raw
        # magnitudes instead would corrupt every feature silently — measured at
        # −5.2 … +3.5 dB in verify_psd_convention_v6.py.
        out[mode] = {
            "p_noise": p_noise,
            "features": sa.v6_spectral_features(
                region_spec, p_noise, noise_f,
                env=ctx["envelope"], region=(i0, i1 + 1)),
        }
    return out


def plot_event(ax_t, ax_s, ev, res, edges):
    fs = cp.FS
    ctx, r = res["ctx"], res["resolved"]
    i0, i1 = r["onset"], r["decay_end"]
    lo = max(0, i0 - 120)
    hi = min(len(ctx["signal"]), i1 + 120)
    t = (np.arange(lo, hi) - i0) / fs * 1e3

    ax_t.plot(t, ctx["signal"][lo:hi] * 1e3, lw=0.7, color="#444")
    ax_t.plot(t, ctx["envelope"][lo:hi] * 1e3, lw=1.1, color="#c33")
    ax_t.axvspan(0, (i1 - i0) / fs * 1e3, color="#5a9", alpha=0.18)
    ax_t.set_xlabel("t from onset [ms]")
    ax_t.set_ylabel("mV")
    tag = "CLICK" if ev["label"] == 1 else "hard negative"
    ax_t.set_title(f"{tag}  ·  {ev['file'][:30]}  f{ev['frame_idx']}\n"
                   f"n_seg={res['n_seg']}  decay_len={res['decay_len']}  "
                   f"peak_SNR={ev['peak_SNR']:.1f}", fontsize=8)

    centers = sa.band_centers(edges) / 1e3
    spec = res["region_spec"]
    m = (spec.freqs >= 20e3) & (spec.freqs <= 80e3)
    ax_s.semilogy(spec.freqs[m] / 1e3, np.maximum(spec.mags[m], 1e-20),
                  lw=0.6, color="#999", label="P_region (native)")

    styles = {"min": ("#d62728", "--", "P_noise·w²  min (§2 as written, β=1.3)"),
              "mean": ("#1f77b4", "-", "P_noise·w²  mean mode")}
    for mode, (col, ls, lab) in styles.items():
        if mode not in res:
            continue
        f6 = res[mode]["features"]
        pn = f6["P_noise_bands"]
        e = f6["E_bands"]
        ax_s.step(centers, np.maximum(pn, 1e-20), where="mid", color=col,
                  ls=ls, lw=1.4, label=lab)
        ax_s.step(centers, np.maximum(e, 1e-20), where="mid", color=col,
                  lw=2.4, alpha=0.35,
                  label=f"E[k]  {mode}   H={f6['spectral_entropy']:.2f} "
                        f"nov={f6['shape_novelty']:.2f}")

    pr = res.get("mean", res.get("min"))["features"]["P_region_bands"]
    ax_s.step(centers, np.maximum(pr, 1e-20), where="mid", color="k", lw=1.6,
              label="P_region (12 bands)")
    ax_s.set_xlabel("f [kHz]")
    ax_s.set_ylabel("PSD [V²/Hz]")
    ax_s.grid(alpha=0.25, which="both")
    ax_s.legend(fontsize=5.5, loc="lower left", framealpha=0.9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="events per class (3-5)")
    ap.add_argument("--out", default=str(_REPO / "docs" / "fft_and_ifft" / "v6_plots"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    edges = sa.band_edges()

    clicks, negs = pick_events(args.n)
    print(f"selected {len(clicks)} clicks, {len(negs)} hard negatives")

    results = []
    for ev in clicks + negs:
        path = find_paudio(ev["file"])
        if path is None:
            print(f"  SKIP {ev['file']}: .paudio not found")
            continue
        try:
            res = analyse(path, ev["frame_idx"])
        except Exception as exc:                                  # noqa: BLE001
            print(f"  SKIP {ev['file']} f{ev['frame_idx']}: {exc}")
            continue
        if res is None:
            print(f"  SKIP {ev['file']} f{ev['frame_idx']}: unusable region")
            continue
        results.append((ev, res))
        f6 = res.get("mean", {}).get("features", {})
        print(f"  ok  {'CLICK' if ev['label'] else 'neg  '} {ev['file'][:34]:34s} "
              f"f{ev['frame_idx']:<9d} n_seg={res['n_seg']:3d} "
              f"H={f6.get('spectral_entropy', float('nan')):.3f} "
              f"nov={f6.get('shape_novelty', float('nan')):.3f} "
              f"tilt={f6.get('spectral_tilt', float('nan')):+.3f}")

    if not results:
        print("nothing to plot", file=sys.stderr)
        return 1

    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(13, 2.7 * n), squeeze=False)
    for row, (ev, res) in enumerate(results):
        plot_event(axes[row][0], axes[row][1], ev, res, edges)
    fig.suptitle("v6 excess spectrum E[k] on real frames — confirmed clicks vs "
                 "hard negatives\n"
                 "P_noise from Buffer 3, taper-corrected (trap a), mic-normalised "
                 "(trap b). Both B3 modes shown: they differ by ~80×.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    dest = out_dir / "v6_ek_real_frames.png"
    fig.savefig(dest, dpi=130)
    print(f"\nwrote {dest}")

    # A compact feature table, mean mode.
    print("\n{:6s} {:32s} {:>7s} {:>7s} {:>7s} {:>7s} {:>8s} {:>8s}".format(
        "label", "file", "n_seg", "H", "novelty", "tilt", "FPE_kHz", "t_conc"))
    for ev, res in results:
        f6 = res.get("mean", {}).get("features", {})
        print("{:6s} {:32s} {:7d} {:7.3f} {:7.3f} {:+7.3f} {:8.1f} {:8.3f}".format(
            "CLICK" if ev["label"] else "neg", ev["file"][:32], res["n_seg"],
            f6.get("spectral_entropy", float("nan")),
            f6.get("shape_novelty", float("nan")),
            f6.get("spectral_tilt", float("nan")),
            f6.get("FPE_hz", float("nan")) / 1e3,
            f6.get("temporal_concentration", float("nan"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
