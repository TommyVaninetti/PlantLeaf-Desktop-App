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
Render every Khait/Dryad clip as a PlantLeaf-domain PNG, via noise-bed injection.

Pipeline per batch of clips:

    Dryad WAV -> resample 500->200 kHz -> colorize with the minimum-phase mic
    response -> inject into a screened, event-free PlantLeaf noise bed at a
    target peak_SNR -> re-frame -> .paudio -> full Stage 1-4 analysis -> PNG

Outputs under --out:

    png/<Class>/<clip_id>.png        one four-panel figure per clip
    contact/<Class>_sheet_NNN.png    tiled sheets for fast triage
    paudio/<Class>_batch_NNN.paudio  openable in PlantLeaf itself
    features_provisional.csv         17 v5 features + measured SNR per clip
    provenance.jsonl                 source, bed, t0, gain, target vs measured

features_provisional.csv is NOT training data. It has no train/validation split
and no augmentation-group or bed-level grouping (spec section 12), and a
meaningful fraction of rows carry tau = -1 (see --target-snr below). Treat it as
material for the cross-domain feature-distribution comparison of spec section 14.

Examples
--------
    # quick look: 20 clips per class
    python3 scripts/dryad_build_plots.py --limit-per-class 20

    # the full corpus, 8 workers
    python3 scripts/dryad_build_plots.py --workers 8

    # clicks only, higher SNR so the decay fit succeeds more often
    python3 scripts/dryad_build_plots.py --classes "Tomato Cut" "Tomato Dry" --target-snr 25
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hybrid import dryad_io as dio          # noqa: E402
from hybrid import frame_emulator as fe     # noqa: E402
from hybrid import injector as inj          # noqa: E402
from hybrid import noise_bed as nb          # noqa: E402
from hybrid import render as rd             # noqa: E402

DEFAULT_DRYAD_ROOT = "/Users/tommy/PlantLeaf_dev/Analisi/DRYAD_database/PlantSounds"
DEFAULT_OUT = "out/dryad_render"

# Frames of slack beyond what the placements strictly need, so plan_placements
# always has room for its jitter and the final click's tail.
_BED_MARGIN_FRAMES = 400


def _batch_bed_frames(n_clicks: int, spacing_s: float) -> int:
    """Usable frames a bed must provide to host `n_clicks` at `spacing_s`."""
    return int(np.ceil(n_clicks * spacing_s * fe.FS / fe.FFT_SIZE)) + _BED_MARGIN_FRAMES


def _process_batch(job: dict) -> dict:
    """
    Run one batch end to end. Executed in a worker process.

    Returns a small summary + provenance rows; the heavy arrays (bed signal,
    reconstructed contexts) never leave the worker.
    """
    out_dir = Path(job["out_dir"])
    rng = np.random.default_rng(job["seed"])

    clips = [dio.DryadClip(**{**c, "path": Path(c["path"])}) for c in job["clips"]]
    audio = [dio.read_clip(c) for c in clips]

    sources = nb.discover_bed_sources(job["bed_roots"])
    if not sources:
        return {"batch_id": job["batch_id"], "error": "no bed sources (external drive not mounted?)",
                "rows": [], "n_rendered": 0}

    # Rotate the starting session by batch index so consecutive batches do not
    # all draw from the same recording — bed diversity, spec section 8.3.
    order = sources[job["batch_index"] % len(sources):] + sources[:job["batch_index"] % len(sources)]

    needed = _batch_bed_frames(len(clips), job["spacing_s"])
    bed = None
    for source in order:
        bed = nb.find_clean_window(source, rng, usable_frames=needed)
        if bed is not None:
            break
    if bed is None:
        return {"batch_id": job["batch_id"],
                "error": f"no event-free bed of {needed} frames in any source",
                "rows": [], "n_rendered": 0}

    mags, phases, results = inj.inject_batch(
        bed, audio, rng,
        amplitude_mode=job["amplitude_mode"],
        target_peak_snr=job["target_snr"], spacing_s=job["spacing_s"],
        keep_render_payload=job["render_png"],
    )

    paudio_info = None
    if job["write_paudio"]:
        paudio_info = fe.write_paudio(
            out_dir / "paudio" / f"{job['batch_id']}.paudio", mags, phases,
            experiment_type="Dryad Hybrid",
        )

    png_paths = []
    if job["render_png"]:
        for result in results:
            path = out_dir / "png" / result.class_name.replace(" ", "_") / f"{result.clip_id}.png"
            rd.render_clip(result, path)
            png_paths.append(str(path))

    rows = []
    for result in results:
        row = result.provenance()
        row["batch_id"] = job["batch_id"]
        row["paudio"] = paudio_info["path"] if paudio_info else ""
        rows.append(row)

    return {
        "batch_id": job["batch_id"],
        "error": None,
        "rows": rows,
        "png_paths": png_paths,
        "n_rendered": len(png_paths),
        "bed": bed.provenance(),
        "paudio": paudio_info,
        "clck_repaired": (paudio_info or {}).get("clck_collisions_repaired", 0),
    }


def _build_jobs(clips: list[dio.DryadClip], args) -> list[dict]:
    """Group clips into per-class batches, each sharing one bed."""
    by_class: dict[str, list[dio.DryadClip]] = {}
    for clip in clips:
        by_class.setdefault(clip.class_name, []).append(clip)

    jobs = []
    for class_name in sorted(by_class):
        members = by_class[class_name]
        if args.limit_per_class:
            members = members[:args.limit_per_class]
        stem = class_name.replace(" ", "_")
        for i in range(0, len(members), args.clicks_per_batch):
            chunk = members[i:i + args.clicks_per_batch]
            index = i // args.clicks_per_batch
            jobs.append({
                "batch_id": f"{stem}_batch_{index:03d}",
                "batch_index": len(jobs),
                "clips": [{"path": str(c.path), "class_name": c.class_name,
                           "plant_id": c.plant_id, "sound_id": c.sound_id,
                           "is_click": c.is_click, "species": c.species,
                           "condition": c.condition} for c in chunk],
                "out_dir": str(args.out),
                "bed_roots": list(args.bed_roots),
                "amplitude_mode": args.amplitude_mode,
                "target_snr": args.target_snr,
                "spacing_s": args.spacing_s,
                "write_paudio": not args.no_paudio,
                "render_png": not args.no_png,
                "seed": args.seed + len(jobs) * 7919,
            })
    return jobs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dryad-root", default=DEFAULT_DRYAD_ROOT,
                        help="directory containing the Dryad class folders")
    parser.add_argument("--out", default=DEFAULT_OUT, type=Path, help="output directory")
    parser.add_argument("--classes", nargs="+", default=None,
                        help=f"subset of {sorted(dio.CLASS_INFO)}")
    parser.add_argument("--limit-per-class", type=int, default=0,
                        help="render at most N clips per class (0 = all)")
    parser.add_argument("--amplitude-mode", choices=inj.AMPLITUDE_MODES,
                        default=inj.AMPLITUDE_GLOBAL_G,
                        help="how loud each clip is injected. 'global-g' (default) applies one "
                             "constant to every clip, so peak_SNR inherits Khait's real "
                             "click-to-click amplitude spread and PlantLeaf's real bed floors "
                             "- use this for anything feeding training. 'fixed-snr' forces every "
                             "clip to --target-snr, which is right for the visualization PNGs "
                             "(one common scale) but makes peak_SNR a label leak, so never train "
                             "on it.")
    parser.add_argument("--target-snr", type=float, default=inj.PLANTLEAF_MEDIAN_PEAK_SNR,
                        help="target peak_SNR for --amplitude-mode fixed-snr. Default 12.79 is "
                             "PlantLeaf's own median. Do NOT raise it to improve the tau yield: "
                             "higher SNR lengthens the decay window by keeping the tail above "
                             "noise longer, which makes synthetic clicks less SNR-realistic than "
                             "real ones.")
    parser.add_argument("--spacing-s", type=float, default=inj.DEFAULT_SPACING_S,
                        help="seconds between injected clicks within a batch")
    parser.add_argument("--clicks-per-batch", type=int, default=40,
                        help="clicks sharing one bed and one .paudio file")
    parser.add_argument("--bed-roots", nargs="+", default=list(nb.DEFAULT_BED_ROOTS),
                        help="directories of event-free PlantLeaf .paudio recordings")
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    parser.add_argument("--no-paudio", action="store_true", help="skip .paudio output")
    parser.add_argument("--no-png", action="store_true", help="skip PNG rendering")
    parser.add_argument("--no-contact-sheets", action="store_true")
    parser.add_argument("--contact-cols", type=int, default=3)
    parser.add_argument("--contact-rows", type=int, default=4)
    args = parser.parse_args(argv)

    args.out = Path(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    clips = dio.build_manifest(args.dryad_root, args.classes)
    if not clips:
        print(f"No Dryad clips found under {args.dryad_root}", file=sys.stderr)
        return 1

    print(f"Dryad corpus: {len(clips)} clips")
    for class_name, info in sorted(dio.summarise_manifest(clips).items()):
        print(f"  {class_name:<20} n={info['n_clips']:>5}  plants={info['n_plants']:>3}  "
              f"{'click' if info['is_click'] else 'noise'}")

    sources = nb.discover_bed_sources(args.bed_roots)
    if not sources:
        print(f"\nNo noise beds found under {args.bed_roots}.\n"
              f"Is the external drive mounted?", file=sys.stderr)
        return 1
    rooms = sorted({s.room for s in sources})
    print(f"\nNoise beds: {len(sources)} recordings across {len(rooms)} room(s) {rooms}, "
          f"{sum(s.duration_s for s in sources) / 3600:.1f} h total")

    jobs = _build_jobs(clips, args)
    n_clips = sum(len(j["clips"]) for j in jobs)
    print(f"\n{n_clips} clips in {len(jobs)} batches "
          f"({args.clicks_per_batch}/batch, {args.spacing_s}s spacing, "
          f"target peak_SNR {args.target_snr})")
    print(f"Workers: {args.workers}\n")

    rows: list[dict] = []
    png_by_class: dict[str, list[str]] = {}
    failures: list[tuple[str, str]] = []
    repaired_total = 0
    start = time.time()

    def absorb(summary: dict, done: int):
        nonlocal repaired_total
        if summary["error"]:
            failures.append((summary["batch_id"], summary["error"]))
            print(f"  [{done}/{len(jobs)}] {summary['batch_id']}: FAILED - {summary['error']}")
            return
        rows.extend(summary["rows"])
        repaired_total += summary.get("clck_repaired", 0)
        for path in summary.get("png_paths", []):
            png_by_class.setdefault(Path(path).parent.name, []).append(path)
        detected = sum(1 for r in summary["rows"] if r["detected"])
        snrs = [r["measured_peak_snr"] for r in summary["rows"] if np.isfinite(r["measured_peak_snr"])]
        rate = len(rows) / max(time.time() - start, 1e-9)
        print(f"  [{done}/{len(jobs)}] {summary['batch_id']:<28} "
              f"{len(summary['rows']):>3} clips, {detected:>3} detected, "
              f"median SNR {np.median(snrs) if snrs else float('nan'):5.1f}, "
              f"bed {summary['bed']['session_id'][:28]} ({rate:.1f} clips/s)")

    if args.workers > 1 and len(jobs) > 1:
        with mp.Pool(args.workers) as pool:
            for done, summary in enumerate(pool.imap_unordered(_process_batch, jobs), 1):
                absorb(summary, done)
    else:
        for done, job in enumerate(jobs, 1):
            absorb(_process_batch(job), done)

    if not rows:
        print("\nNothing was produced.", file=sys.stderr)
        return 1

    # ── provenance + features ────────────────────────────────────────────────
    prov_path = args.out / "provenance.jsonl"
    with open(prov_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    csv_path = args.out / "features_provisional.csv"
    fieldnames = list(rows[0])
    for row in rows:                      # a failed fit can omit a key
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)

    # ── contact sheets ───────────────────────────────────────────────────────
    per_sheet = args.contact_cols * args.contact_rows
    n_sheets = 0
    if not args.no_contact_sheets and not args.no_png:
        for class_dir, paths in sorted(png_by_class.items()):
            paths = sorted(paths)
            for i in range(0, len(paths), per_sheet):
                sheet = args.out / "contact" / f"{class_dir}_sheet_{i // per_sheet:03d}.png"
                rd.render_contact_sheet(
                    paths[i:i + per_sheet], sheet,
                    cols=args.contact_cols, rows=args.contact_rows,
                    title=f"{class_dir}  {i + 1}-{min(i + per_sheet, len(paths))} of {len(paths)}",
                )
                n_sheets += 1

    # ── summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    snr = np.array([r["measured_peak_snr"] for r in rows], dtype=float)
    snr = snr[np.isfinite(snr)]
    tau = np.array([r.get("feat_tau_ms", np.nan) for r in rows], dtype=float)
    detected = sum(1 for r in rows if r["detected"])
    click_snr = np.array([r["measured_peak_snr"] for r in rows
                          if r["is_click"] and np.isfinite(r["measured_peak_snr"])])
    neg_snr = np.array([r["measured_peak_snr"] for r in rows
                        if not r["is_click"] and np.isfinite(r["measured_peak_snr"])])

    print(f"\n{'=' * 68}\nDone in {elapsed / 60:.1f} min - {len(rows)} clips\n{'=' * 68}")
    print(f"  amplitude mode      : {args.amplitude_mode}")
    print(f"  detected by Stage 1 : {detected}/{len(rows)} ({100 * detected / len(rows):.1f} %)")
    if len(snr):
        if args.amplitude_mode == inj.AMPLITUDE_FIXED_SNR:
            print(f"  measured peak_SNR   : median {np.median(snr):.2f} "
                  f"(target {args.target_snr}, ratio {np.median(snr) / args.target_snr:.3f})")
        else:
            print(f"  measured peak_SNR   : p10 {np.percentile(snr, 10):.2f}  "
                  f"median {np.median(snr):.2f}  p90 {np.percentile(snr, 90):.2f}   "
                  f"(PlantLeaf real: {inj.PLANTLEAF_PEAK_SNR_P10} / "
                  f"{inj.PLANTLEAF_MEDIAN_PEAK_SNR} / {inj.PLANTLEAF_PEAK_SNR_P90})")
    if len(click_snr) and len(neg_snr):
        # PlantLeaf's own data separates clicks from negatives by ~2.4x in median
        # peak_SNR. Whatever ratio appears here is KHAIT'S, inherited rather than
        # imposed -- report it rather than assume it transfers (spec B4).
        print(f"  click/negative sep  : {np.median(click_snr) / max(np.median(neg_snr), 1e-9):.2f}x "
              f"median peak_SNR (PlantLeaf's own data: 2.4x)")
    valid = tau[tau > 0]
    print(f"  tau_ms              : median {np.median(valid) if len(valid) else float('nan'):.3f} "
          f"over {len(valid)}/{len(tau)} successful fits")
    if repaired_total:
        print(f"  b'CLCK' collisions repaired in .paudio payloads: {repaired_total}")
    if failures:
        print(f"\n  {len(failures)} batch(es) FAILED:")
        for batch_id, error in failures[:10]:
            print(f"    {batch_id}: {error}")
    print(f"\n  provenance : {prov_path}")
    print(f"  features   : {csv_path}   (PROVISIONAL - not training data)")
    if n_sheets:
        print(f"  contact    : {n_sheets} sheets in {args.out / 'contact'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
