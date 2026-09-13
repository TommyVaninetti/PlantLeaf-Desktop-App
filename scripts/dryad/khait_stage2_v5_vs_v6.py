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
Priority 1 of the Khait action plan: Stage 2 under v5's gates vs v6's gates.

WHY THIS IS PAIRED IN-RUN RATHER THAN AGAINST THE OLD REPORT
The transparency report's v5 figure (23.5 % Stage-2 loss) was measured on a bed
pool that no longer exists. Comparing a fresh v6 number against it would confound
the gate redesign with a change of noise beds. Instead both gate sets are
evaluated on THE SAME feature vectors from ONE injection pass, so the bed pool,
the injection gain and the feature code are identical in both arms and the
difference is attributable to the gates alone. The injection artefact the
transparency report measured (leak AUC 0.744) is likewise common to both arms and
cancels out of the difference.

WHAT IT DOES NOT CLAIM
Not an absolute pass rate for Khait clicks in the wild. The injector's fidelity
is still an open question (docs/fft_and_ifft/INJECTOR_FIDELITY_PROBLEM.md); what
survives that is the v5-vs-v6 DIFFERENCE, not either level on its own.

VALIDITY PRECONDITION, CHECKED AND PRINTED BEFORE ANY RATE
`_stage2_reason` treats NaN and missing keys as PASS. A v6 gate whose input never
arrives therefore inflates the v6 pass rate silently. Section 0 of the report
asserts every gate input is present before any rate is believable; if it fails,
the run is void and says so.

Examples
--------
    # the real thing: 120 clips per click class = 480, at PlantLeaf's median SNR
    python3 scripts/dryad/khait_stage2_v5_vs_v6.py --bed-roots "/Volumes/.../Noise"

    # smoke test with no external drive: synthetic Gaussian beds
    python3 scripts/dryad/khait_stage2_v5_vs_v6.py --synthetic-bed --limit-per-class 8
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hybrid import dryad_io as dio          # noqa: E402
from hybrid import frame_emulator as fe     # noqa: E402
from hybrid import injector as inj          # noqa: E402
from hybrid import noise_bed as nb          # noqa: E402
from hybrid.pipeline_loader import (        # noqa: E402
    FrameDataManager, compute_stage1_arrays, load_pipeline)

DEFAULT_DRYAD_ROOT = "/Users/tommy/PlantLeaf_dev/Analisi/DRYAD_database/PlantSounds"
CLICK_CLASSES = ("Tobacco Cut", "Tobacco Dry", "Tomato Cut", "Tomato Dry")

# Frames of slack beyond what the placements strictly need, matching
# dryad_build_plots so a bed accepted there is accepted here.
_BED_MARGIN_FRAMES = 400
# Gaussian sigma for --synthetic-bed, the order of a real indoor floor. Only ever
# a plumbing substitute: a synthetic bed makes no claim about pass rates.
_SYNTHETIC_SIGMA_V = 3.0e-3


def _bed_frames(n_clicks: int, spacing_s: float) -> int:
    return int(np.ceil(n_clicks * spacing_s * fe.FS / fe.FFT_SIZE)) + _BED_MARGIN_FRAMES


def _synthetic_bed(rng, n_clicks: int, spacing_s: float) -> nb.BedWindow:
    """A Gaussian stand-in, so the pipeline can be exercised with no drive mounted."""
    warmup = nb.WARMUP_FRAMES
    n_frames = warmup + _bed_frames(n_clicks, spacing_s)
    sig = rng.normal(0.0, _SYNTHETIC_SIGMA_V, n_frames * fe.FFT_SIZE)
    mags, phases = fe.frames_from_signal(sig)
    arrays = compute_stage1_arrays(mags, phases, fs=fe.FS, fft_size=fe.FFT_SIZE)
    return nb.BedWindow(
        bed_id="synthetic", source_path="<synthetic>", session_id="synthetic",
        room="synthetic", start_frame=0, n_frames=n_frames, warmup_frames=warmup,
        signal=sig,
        noise_floor=float(np.median(arrays["noise_floor_arr"][warmup:])),
        std_noise=float(np.median(arrays["std_noise_arr"][warmup:])),
        e_hat_floor=float(np.median(arrays["E_hat_floor_arr"][warmup:])),
        candidate_rate=0.0, regime="synthetic")


def _process_batch(job: dict) -> dict:
    """One batch, one bed, one pipeline pass. Runs in a worker process."""
    rng = np.random.default_rng(job["seed"])
    clips = [dio.DryadClip(**{**c, "path": Path(c["path"])}) for c in job["clips"]]
    audio = [dio.read_clip(c) for c in clips]

    if job["synthetic_bed"]:
        bed = _synthetic_bed(rng, len(clips), job["spacing_s"])
    else:
        sources = nb.discover_bed_sources(job["bed_roots"])
        if not sources:
            return {"batch_id": job["batch_id"], "rows": [],
                    "error": "no bed sources (external drive not mounted?)"}
        start = job["batch_index"] % len(sources)
        order = sources[start:] + sources[:start]
        needed = _bed_frames(len(clips), job["spacing_s"])
        bed = None
        for source in order:
            bed = nb.find_clean_window(source, rng, usable_frames=needed)
            if bed is not None:
                break
        if bed is None:
            return {"batch_id": job["batch_id"], "rows": [],
                    "error": f"no event-free bed of {needed} frames in any source"}

    _, _, results = inj.inject_batch(
        bed, audio, rng, amplitude_mode=job["amplitude_mode"],
        target_peak_snr=job["target_snr"], spacing_s=job["spacing_s"],
        keep_render_payload=False)

    rows = []
    for r in results:
        row = r.provenance()
        row["batch_id"] = job["batch_id"]
        rows.append(row)
    return {"batch_id": job["batch_id"], "rows": rows, "error": None}


def _v5_subcause(row: dict, min_fit_samples: int) -> str:
    """
    Which condition inside v5's fit gate rejected this row.

    _stage2_reason_v5 collapses four distinct conditions into one verdict string
    (Stage2_R2). The transparency report decomposed them, so reproduce that here
    from the exported features rather than by editing the pipeline.
    """
    def g(key, default=float("nan")):
        v = row.get(f"feat_{key}", default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    if g("decay_len_samples") < min_fit_samples:
        return "window too short"
    # Once decay_len >= MIN_FIT_SAMPLES, _fit_decay_segment sets fit_valid = 0
    # only when the OLS slope is not negative (the denom ~ 0 guard is unreachable
    # for n >= 2 distinct x). So this IS the report's "slope >= 0" category —
    # labelling it "fit_valid == 0" hid that and broke the comparison.
    if not int(g("fit_valid") or 0):
        return "slope >= 0 (no decay)"
    r2, tau = g("R2"), g("tau_ms")
    if r2 != r2 or tau != tau:
        return "R2/tau NaN"
    if r2 < 0.10:
        return "R2 < 0.10"
    return "other"


def _pct(num: int, den: int) -> str:
    return f"{100.0 * num / den:5.1f} %" if den else "    -  "


def _report(rows: list[dict], cp) -> None:
    det = [r for r in rows if int(r.get("detected", 0))]
    n, nd = len(rows), len(det)

    print("\n" + "=" * 76)
    print("0. VALIDITY — are v6's gate inputs actually present?")
    print("=" * 76)
    print("   _stage2_reason treats NaN and missing as PASS, so a gate with no")
    print("   input does not fail loudly: it stops gating and flatters v6.")
    void = False
    for key in ("peak_SNR", "n_seg", "local_crest"):
        bad = sum(1 for r in det
                  if r.get(f"feat_{key}") is None
                  or float(r[f"feat_{key}"]) != float(r[f"feat_{key}"]))
        void |= bad > 0
        print(f"   {key:<22}: {nd - bad}/{nd} finite"
              + ("" if bad == 0 else "   *** GATE INACTIVE — RUN IS VOID ***"))

    b3 = [float(r.get("feat_b3_frames", 0) or 0) for r in det]
    warm = sum(1 for v in b3 if v >= 0.9 * cp.W_NOISE)
    print(f"   {'b3_frames':<22}: {warm}/{nd} warm (>= 0.9 x W_NOISE = {cp.W_NOISE})"
          + ("" if warm == nd else "   <- cold B3 makes harmonic_confinement noisy"))

    # harmonic_confinement is legitimately undefined in two documented cases. A
    # blind NaN (f1_hz also NaN) would instead mean p_noise_psd never arrived.
    undef = [r for r in det
             if float(r.get("feat_harmonic_confinement", float("nan")))
             != float(r.get("feat_harmonic_confinement", float("nan")))]
    blind = [r for r in undef
             if float(r.get("feat_hc_f1_hz", float("nan")))
             != float(r.get("feat_hc_f1_hz", float("nan")))]
    void |= bool(blind)
    print(f"   {'harmonic_confinement':<22}: {nd - len(undef)}/{nd} defined; "
          f"{len(undef)} undefined, of which {len(blind)} blind"
          + ("" if not blind else "   *** p_noise_psd MISSING — RUN IS VOID ***"))
    if undef and not blind:
        print(f"   {'':<22}  the {len(undef)} undefined are the documented cases "
              f"(m == 0, or 2*f1 off-band);")
        print(f"   {'':<22}  the harmonic gate is INERT for them, which is a "
              f"finding about the gate,")
        print(f"   {'':<22}  not a defect in this run.")
    if void:
        print("\n   *** DO NOT BELIEVE THE RATES BELOW. ***")

    print("\n" + "=" * 76)
    print("1. HEADLINE — same clips, same beds, same features, two gate sets")
    print("=" * 76)
    p5 = sum(1 for r in det if int(r.get("stage2_pass_v5", 0)))
    p6 = sum(1 for r in det if int(r.get("stage2_pass_v6", 0)))
    print(f"   clips injected            : {n}")
    print(f"   tripped Stage 1           : {nd}  ({_pct(nd, n)})")
    print(f"   Stage 2 pass, v5 gates    : {p5}/{nd}  ({_pct(p5, nd)})   "
          f"loss {_pct(nd - p5, nd)}")
    print(f"   Stage 2 pass, v6 gates    : {p6}/{nd}  ({_pct(p6, nd)})   "
          f"loss {_pct(nd - p6, nd)}")
    print(f"   recovered by the redesign : {p6 - p5:+d} clicks "
          f"({100.0 * (p6 - p5) / nd:+.1f} points)" if nd else "")
    lost = [r for r in det if int(r.get("stage2_pass_v5", 0))
            and not int(r.get("stage2_pass_v6", 0))]
    print(f"   NEWLY lost under v6       : {len(lost)}  "
          f"(passed v5, fails v6 — a fresh Khait-specific cost if > 0)")

    print("\n" + "=" * 76)
    print("2. FAILURE DECOMPOSITION")
    print("=" * 76)
    print("   v5 gates, by the condition inside the fit gate:")
    c5 = Counter(_v5_subcause(r, cp.MIN_FIT_SAMPLES)
                 for r in det if not int(r.get("stage2_pass_v5", 0)))
    for reason, k in c5.most_common():
        print(f"      {reason:<28} {k:>5}  {_pct(k, nd)}")
    if not c5:
        print("      (none)")

    print("   v6 gates, by rejecting gate:")
    c6 = Counter(r.get("stage2_reason_v6", "") for r in det
                 if not int(r.get("stage2_pass_v6", 0)))
    for reason, k in c6.most_common():
        print(f"      {reason:<28} {k:>5}  {_pct(k, nd)}")
    if not c6:
        print("      (none)")

    print("\n" + "=" * 76)
    print("3. BY KHAIT CLASS — does v6 stay uniform, or concentrate the loss?")
    print("=" * 76)
    print("   v5 showed no class concentration (70.8-79.2 %). n_seg and")
    print("   local_crest were derived from indoor PlantLeaf distributions and")
    print("   never validated against Khait's shorter regions, so a new spread")
    print("   here is the thing to look for.")
    print(f"   {'class':<16}{'n':>5}{'det':>6}{'v5 pass':>10}{'v6 pass':>10}"
          f"{'delta':>8}{'n_seg med':>11}{'crest med':>11}")
    print("   " + "-" * 73)
    by_class = defaultdict(list)
    for r in det:
        by_class[r.get("class_name", "?")].append(r)
    for name in sorted(by_class):
        g = by_class[name]
        total = sum(1 for r in rows if r.get("class_name") == name)
        a = sum(1 for r in g if int(r.get("stage2_pass_v5", 0)))
        b = sum(1 for r in g if int(r.get("stage2_pass_v6", 0)))
        med = lambda k: float(np.median([float(r.get(f"feat_{k}", np.nan)) for r in g]))
        print(f"   {name:<16}{total:>5}{len(g):>6}{_pct(a, len(g)):>10}"
              f"{_pct(b, len(g)):>10}{100.0*(b-a)/len(g):>+7.1f}"
              f"{med('n_seg'):>11.0f}{med('local_crest'):>11.2f}")

    print("\n" + "=" * 76)
    print("4. GATE HEADROOM — how close the survivors sit to each threshold")
    print("=" * 76)
    print("   A gate that costs nothing today but sits one p10 away from the")
    print("   distribution is a different risk from one with real margin.")
    for key, thr, side in (("peak_SNR", cp.STAGE2_PEAK_SNR_MIN, "min"),
                           ("n_seg", cp.STAGE2_N_SEG_MIN, "min"),
                           ("local_crest", cp.STAGE2_LOCAL_CREST_MIN, "min"),
                           ("harmonic_confinement", cp.STAGE2_HC_MAX, "max")):
        vals = np.array([float(r.get(f"feat_{key}", np.nan)) for r in det])
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            print(f"   {key:<22} no finite values")
            continue
        p10, p50, p90 = np.percentile(vals, [10, 50, 90])
        print(f"   {key:<22} threshold {side} {thr:<8g} "
              f"p10 {p10:>9.2f}  med {p50:>9.2f}  p90 {p90:>9.2f}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dryad-root", default=DEFAULT_DRYAD_ROOT)
    parser.add_argument("--classes", nargs="+", default=list(CLICK_CLASSES),
                        help="default: the four confirmed-click classes")
    parser.add_argument("--limit-per-class", type=int, default=120,
                        help="120 x 4 classes = the 480 clips of the transparency report")
    parser.add_argument("--amplitude-mode", choices=inj.AMPLITUDE_MODES,
                        default=inj.AMPLITUDE_FIXED_SNR,
                        help="fixed-snr by default: Priority 1 compares gates at ONE "
                             "injection level, matching the transparency report. "
                             "peak_SNR is then a leak, which does not matter here "
                             "because nothing is trained.")
    parser.add_argument("--target-snr", type=float, default=inj.PLANTLEAF_MEDIAN_PEAK_SNR)
    parser.add_argument("--spacing-s", type=float, default=inj.DEFAULT_SPACING_S)
    parser.add_argument("--clicks-per-batch", type=int, default=30)
    parser.add_argument("--bed-roots", nargs="+", default=list(nb.DEFAULT_BED_ROOTS))
    parser.add_argument("--synthetic-bed", action="store_true",
                        help="Gaussian beds instead of real ones. Exercises the code "
                             "with no drive mounted; the RATES ARE MEANINGLESS.")
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    parser.add_argument("--out", type=Path, default=Path("out/khait_stage2_v5_vs_v6.csv"))
    args = parser.parse_args(argv)

    cp = load_pipeline()
    clips = dio.build_manifest(args.dryad_root, classes=args.classes)
    non_click = sorted({c.class_name for c in clips if not c.is_click})
    if non_click:
        print(f"NOTE: dropping non-click classes {non_click} — Priority 1 is about "
              f"clicks passing Stage 2.")
        clips = [c for c in clips if c.is_click]

    by_class: dict[str, list] = defaultdict(list)
    for c in clips:
        by_class[c.class_name].append(c)

    jobs = []
    for name in sorted(by_class):
        members = by_class[name]
        if args.limit_per_class:
            members = members[:args.limit_per_class]
        stem = name.replace(" ", "_")
        for i in range(0, len(members), args.clicks_per_batch):
            chunk = members[i:i + args.clicks_per_batch]
            jobs.append({
                "batch_id": f"{stem}_batch_{i // args.clicks_per_batch:03d}",
                "batch_index": len(jobs),
                "clips": [{"path": str(c.path), "class_name": c.class_name,
                           "plant_id": c.plant_id, "sound_id": c.sound_id,
                           "is_click": c.is_click, "species": c.species,
                           "condition": c.condition} for c in chunk],
                "bed_roots": list(args.bed_roots),
                "synthetic_bed": args.synthetic_bed,
                "amplitude_mode": args.amplitude_mode,
                "target_snr": args.target_snr,
                "spacing_s": args.spacing_s,
                "seed": args.seed + len(jobs) * 7919,
            })

    total = sum(len(j["clips"]) for j in jobs)
    print(f"{total} clips across {len(by_class)} classes, {len(jobs)} batches, "
          f"{args.workers} workers")
    print(f"amplitude mode {args.amplitude_mode}"
          + (f" @ peak_SNR {args.target_snr}" if args.amplitude_mode == inj.AMPLITUDE_FIXED_SNR else "")
          + ("   [SYNTHETIC BEDS — rates meaningless]" if args.synthetic_bed else ""))

    if args.workers > 1 and len(jobs) > 1:
        with mp.Pool(args.workers) as pool:
            out = pool.map(_process_batch, jobs)
    else:
        out = [_process_batch(j) for j in jobs]

    rows, errors = [], []
    for res in out:
        if res.get("error"):
            errors.append(f"{res['batch_id']}: {res['error']}")
        rows.extend(res["rows"])
    for e in errors:
        print(f"  BATCH FAILED  {e}")
    if not rows:
        print("\nNo rows produced. With real beds this usually means the drive is "
              "not mounted or --bed-roots is wrong; --synthetic-bed runs without one.")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    _report(rows, cp)
    print(f"\nper-clip rows: {args.out}  ({len(rows)} rows, {len(fields)} columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
