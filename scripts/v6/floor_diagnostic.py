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
floor_diagnostic.py — is the burst gate starving the adaptive noise floor?

    python3 scripts/v6/floor_diagnostic.py CORPUS_ROOT [...] -o floor.csv

THE QUESTION
------------
Outdoor recordings emit up to 344,773 candidates/hour — 24 % of ALL frames, and
half the structural cap at R = 1. Measured from the exported CSVs, the noise
floor's dynamic range across an ENTIRE recording is median 1.30x (p90 1.49x),
indoors and on a 2-hour balcony alike. A floor that static is not tracking
anything.

The suspect is the burst gate: a frame with `E_i > ALPHA * Ê_floor(i-1)` is
excluded from every noise buffer, and `ALPHA = 4.0` carries its own unfinished
note in click_pipeline_v5.py — "verify experimentally on outdoor recordings".
That verification is this script.

The decisive number is the **fraction of frames burst-gated**. A few percent and
the hypothesis is wrong. Tens of percent and the estimator is starved: it is
being fed only the quietest frames, so it settles below the true ambient level,
and Stage 1's `k * Ê_floor` threshold then sits under the noise.

WHY THIS IS CHEAP
-----------------
Nothing new is computed. `AdaptiveNoiseEstimatorV5.update()` already returns
`'is_burst'` per frame and NOTHING consumes it; `dm.fft_means` and
`dm.E_hat_floor_arr` already exist as per-frame arrays after the load. So the
gate decision is reconstructible vectorised as

    is_burst[i] = fft_means[i] > ALPHA * E_hat_floor_arr[i-1]

which also lets alternative ALPHA values be swept offline WITHOUT touching the
live estimator — no behaviour changes, no re-derivation, no risk to Stage 1.

The cost is one .paudio load per recording (~515 s per hour of audio; there is no
load cache). Resumable: rows append as each recording finishes and a re-run skips
what is already in the output.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

import windows  # noqa: E402,F401  (breaks a pre-existing circular import)

#: ALPHA values to sweep. The first is the shipped value.
ALPHA_SWEEP = (4.0, 6.0, 8.0, 12.0, 20.0)

COLUMNS = [
    "file", "total_frames", "duration_h",
    "gated_pct", "longest_gated_run", "stale_events",
    "floor_min", "floor_median", "floor_max", "floor_range",
    "cand_per_hour", "warm_up_frames",
    # ── is the floor BELOW the actual ambient, and is the noise IMPULSIVE? ──
    # Two different questions, and the answer to the second explains the first.
    #
    # `floor_pctile` is the percentile of the actual per-frame energy that the
    # median Ê_floor sits at. Minimum-statistics estimators are SUPPOSED to track
    # a low percentile — that is what makes them robust to speech/transients — so
    # a small number is not by itself a bug. It becomes the whole story when the
    # environment is impulsive: the quiet gaps between impulses are genuinely
    # quiet, the floor correctly settles there, and k * Ê_floor then sits UNDER
    # the impulses, so every impulse clears the threshold.
    #
    # `impulsivity` = p99/p50 of frame energy. Stationary noise is ~1-3; a train
    # of transients over a quiet background is large.
    "E_p50", "E_p99", "floor_pctile", "impulsivity", "frac_above_k_floor",
] + [f"gated_pct_alpha{a:g}" for a in ALPHA_SWEEP] + ["load_s", "note"]


def _find(roots):
    """Largest .paudio per stem — the corpus keeps short excerpts beside the
    full recordings, and censusing the excerpt understates by 1000x."""
    best = {}
    for root in roots:
        rp = Path(root)
        if not rp.exists():
            print(f"  ! missing root: {rp}", file=sys.stderr)
            continue
        try:
            found = list(rp.rglob("*.paudio"))
        except PermissionError:
            print(f"  ! permission denied: {rp}", file=sys.stderr)
            continue
        if not found:
            print(f"  ! no .paudio under {rp} (external drive? check Full Disk Access)",
                  file=sys.stderr)
        for p in found:
            if p.name.startswith("._"):
                continue
            try:
                sz = p.stat().st_size
            except OSError:
                continue
            if p.stem not in best or sz > best[p.stem][1]:
                best[p.stem] = (p, sz)
    return [p for p, _ in sorted(best.values(), key=lambda x: x[0].name.lower())]


def _runs_of_true(mask: np.ndarray) -> int:
    """Longest run of True. The floor cannot rise while every frame is gated, so
    this is 'how long could the estimator have been frozen'."""
    if not mask.any():
        return 0
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    return int((idx[1::2] - idx[0::2]).max())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("-o", "--output", type=Path, default=Path("floor_diagnostic.csv"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from components.data_collection_dialog_v5 import DataCollectionWorkerV5
    from core.click_pipeline_v5 import (
        ALPHA, WARM_UP_FRAMES, STALE_FLOOR_FRAMES, K_STAGE1_DEFAULT,
        has_precomputed_stage1_arrays, run_stage1_v5_precomputed)

    files = _find(args.roots)
    if args.limit:
        files = files[:args.limit]

    done = set()
    if args.output.exists():
        with open(args.output, newline="") as fh:
            done = {r["file"] for r in csv.DictReader(fh)}
        print(f"resuming: {len(done)} already done")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", newline="") as fh:
            csv.writer(fh).writerow(COLUMNS)

    print(f"\nALPHA = {ALPHA}   WARM_UP_FRAMES = {WARM_UP_FRAMES}   "
          f"STALE_FLOOR_FRAMES = {STALE_FLOOR_FRAMES}")
    print(f"{len(files)} recording(s)\n")

    class _Sig:
        def emit(self, *a, **k): pass
        def connect(self, *a, **k): pass

    for n, path in enumerate(files, 1):
        if path.stem in done:
            continue
        t0 = time.time()
        row = {c: "" for c in COLUMNS}
        row["file"] = path.stem
        row["warm_up_frames"] = WARM_UP_FRAMES
        try:
            w = DataCollectionWorkerV5.__new__(DataCollectionWorkerV5)
            w._current_audio_worker = None
            w._stop_requested = False
            w.load_progress = _Sig()
            w.error_occurred = _Sig()
            dm = DataCollectionWorkerV5._load_audio_file(w, path)
            if dm is None:
                row["note"] = "load failed"
            else:
                E = np.asarray(dm.fft_means, dtype=np.float64)
                F = np.asarray(dm.E_hat_floor_arr, dtype=np.float64)
                nf = np.asarray(getattr(dm, "noise_floor_arr", []), dtype=np.float64)
                fs = dm.header_info.get("fs", 200_000)
                nfft = dm.header_info.get("fft_size", 512)
                hrs = dm.total_frames * nfft / fs / 3600.0
                row["total_frames"] = dm.total_frames
                row["duration_h"] = round(hrs, 3)

                # The gate as the estimator applies it: E_i vs ALPHA * floor(i-1).
                # Warm-up frames are force-accepted, so they are excluded from the
                # statistic rather than counted as "not gated" — including them
                # would dilute the very number we are testing.
                prev = F[:-1]
                cur = E[1:]
                valid = np.arange(1, len(E)) >= WARM_UP_FRAMES
                valid &= np.isfinite(prev) & np.isfinite(cur) & (prev > 0)
                if valid.any():
                    gated = (cur > ALPHA * prev) & valid
                    row["gated_pct"] = round(100.0 * gated.sum() / valid.sum(), 2)
                    row["longest_gated_run"] = _runs_of_true(gated)
                    row["stale_events"] = int(
                        (_runs_of_true(gated) >= STALE_FLOOR_FRAMES))
                    for a in ALPHA_SWEEP:
                        g = (cur > a * prev) & valid
                        row[f"gated_pct_alpha{a:g}"] = round(
                            100.0 * g.sum() / valid.sum(), 2)

                # ── floor vs the actual energy distribution ──────────────────
                Ef = E[np.isfinite(E) & (E > 0)]
                Ff = F[np.isfinite(F) & (F > 0)]
                if Ef.size > 100 and Ff.size > 100:
                    e50, e99 = np.median(Ef), np.percentile(Ef, 99)
                    fmed = np.median(Ff)
                    row["E_p50"] = float(f"{e50:.6g}")
                    row["E_p99"] = float(f"{e99:.6g}")
                    # Where the floor sits within the energy distribution.
                    row["floor_pctile"] = round(
                        float(100.0 * np.mean(Ef < fmed)), 2)
                    row["impulsivity"] = round(float(e99 / max(e50, 1e-30)), 2)
                    row["frac_above_k_floor"] = round(
                        float(100.0 * np.mean(E[1:] > K_STAGE1_DEFAULT * F[:-1])), 2)

                src = nf if nf.size == F.size and np.isfinite(nf).any() else F
                pos = src[np.isfinite(src) & (src > 0)]
                if pos.size:
                    row["floor_min"] = float(f"{pos.min():.6g}")
                    row["floor_median"] = float(f"{np.median(pos):.6g}")
                    row["floor_max"] = float(f"{pos.max():.6g}")
                    row["floor_range"] = round(float(pos.max() / pos.min()), 3)

                if has_precomputed_stage1_arrays(dm):
                    n_cand = len(run_stage1_v5_precomputed(dm, k=K_STAGE1_DEFAULT))
                    row["cand_per_hour"] = round(n_cand / max(hrs, 1e-9))
        except Exception as exc:                              # noqa: BLE001
            row["note"] = f"{type(exc).__name__}: {exc}"[:160]

        row["load_s"] = round(time.time() - t0, 1)
        with open(args.output, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=COLUMNS).writerow(row)
        print(f"  [{n:>3}/{len(files)}] {path.stem[:40]:<40} "
              f"gated={str(row['gated_pct']):>6}%  longest={str(row['longest_gated_run']):>7}  "
              f"floorRange={str(row['floor_range']):>6}x  "
              f"cand/h={str(row['cand_per_hour']):>7}  {row['load_s']:>6}s {row['note']}",
              flush=True)

    _summarise(args.output, ALPHA)
    return 0


def _summarise(path: Path, alpha: float) -> None:
    with open(path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("gated_pct") not in ("", None)]
    if not rows:
        return
    g = np.array([float(r["gated_pct"]) for r in rows])
    rng = np.array([float(r["floor_range"]) for r in rows if r["floor_range"]])
    print("\n" + "=" * 72)
    print(f"{len(rows)} recordings   ALPHA = {alpha}")
    print("=" * 72)
    print(f"  frames burst-gated : p10={np.percentile(g,10):.2f}%  "
          f"median={np.median(g):.2f}%  p90={np.percentile(g,90):.2f}%  max={g.max():.2f}%")
    if rng.size:
        print(f"  floor dynamic range: median={np.median(rng):.2f}x  "
              f"p90={np.percentile(rng,90):.2f}x  max={rng.max():.2f}x")
    print("\n  ALPHA sweep (median % of frames gated):")
    for c in [c for c in rows[0] if c.startswith("gated_pct_alpha")]:
        v = np.array([float(r[c]) for r in rows if r[c]])
        if v.size:
            print(f"     {c.replace('gated_pct_alpha','ALPHA = '):>14}: "
                  f"{np.median(v):6.2f}%")
    print("\n  Read it as: a few percent => the burst gate is NOT the problem.")
    print("  Tens of percent => the estimator sees only the quietest frames and")
    print("  settles below the true ambient, so k*Ê_floor sits under the noise.")

    have = [r for r in rows if r.get("impulsivity")]
    if have:
        imp = np.array([float(r["impulsivity"]) for r in have])
        pct = np.array([float(r["floor_pctile"]) for r in have])
        print(f"\n  impulsivity (p99/p50 of frame energy): median={np.median(imp):.1f}  "
              f"max={imp.max():.1f}")
        print(f"  floor sits at percentile of energy   : median={np.median(pct):.1f}  "
              f"min={pct.min():.1f}")
        print("\n  A LOW floor percentile with HIGH impulsivity is not a broken")
        print("  estimator — it is minimum statistics correctly tracking the quiet")
        print("  gaps between transients. The threshold then sits under the")
        print("  transients, and every one of them becomes a candidate.\n")


if __name__ == "__main__":
    sys.exit(main())
