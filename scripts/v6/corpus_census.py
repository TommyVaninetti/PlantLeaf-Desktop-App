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
corpus_census.py — how big is the labelling job, per recording?

    python3 scripts/v6/corpus_census.py CORPUS_ROOT [CORPUS_ROOT ...] -o census.csv

WHY THIS EXISTS
---------------
Phase 3 is an EXHAUSTIVE pass over a SUBSET of recordings, so the subset choice
is the decision that sets the workload — and it cannot be made from filenames.
Measured on the v5-era exports, the census is p50 = 28 rows but p90 = 4997 and
max = 48574, and **95 % of all rows live in 18 of the 109 recordings** — the
plant-only and long-noise files, which carry almost no clicks. Picking by name
would either blow the budget on those or silently drop the negative class.

WHAT IT REPORTS, per recording
------------------------------
    n_unfiltered   Stage 1 candidates, EXPORT_UNFILTERED  (every candidate)
    n_export_all   Stage 1 candidates with fit_valid == 1 (what EXPORT_ALL writes)
    n_confirmed    survived Stages 2-4 (indicative only; needs a model)
    png_mb         screenshots at ~112 kB each, for the EXPORT_ALL count
    old_clicks     labels you already made on this recording, if --old-dir given
    old_noise
    group          the folder it sits in — a stand-in for the class

`n_export_all` vs `n_unfiltered` is roughly an 8x lever on labelling hours,
because fit_valid == 0 is ~76 % of the census. Both are reported so the export
mode is chosen with the number in hand rather than at the dialog.

COST — AND WHY YOU PROBABLY WANT --from-export
----------------------------------------------
Measured on a 2.57 M-frame recording (1 h 50 m of audio):

    load = 515.6 s        process = 0.7 s

The `.paudio` load is **99.9 %** of the cost, and it is the *same* load the Data
Collection export performs. So:

  * If you have NOT exported yet and want the numbers before committing to one,
    run the default mode. Budget ~9 minutes per hour-scale recording. It is
    RESUMABLE — rows append as each recording finishes and a re-run skips what is
    already in the output — so an interrupted pass costs one file.

  * If you HAVE exported, use `--from-export` instead. It reads the exported CSVs
    and is instant. Re-loading the .paudio files to count rows you already have on
    disk is pure waste, and at corpus scale that is hours.

`--stage1-only` exists but saves almost nothing, for the same reason: it skips the
0.7 s, not the 515 s.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

import windows  # noqa: E402,F401  (breaks a pre-existing circular import)

#: One exported screenshot, measured on the 1400x960 v6 render.
PNG_KB = 112

COLUMNS = [
    "file", "group", "path",
    "total_frames", "duration_s",
    "n_unfiltered", "n_export_all", "n_confirmed",
    "png_mb",
    "old_rows", "old_clicks", "old_noise",
    "load_s", "process_s", "note",
]


def _load_old_labels(old_dir: Path | None) -> dict:
    """{stem: (rows, clicks, noise, group)} from the v5-era per-recording CSVs."""
    out: dict = {}
    if old_dir is None or not old_dir.is_dir():
        return out
    for f in old_dir.rglob("*_candidates.csv"):
        stem = f.stem.replace("_candidates", "")
        try:
            with open(f, encoding="utf-8-sig") as fh:
                rows = list(csv.DictReader(fh))
        except OSError:
            continue
        if not rows:
            continue
        c = Counter((r.get("label") or "").strip() for r in rows)
        out[stem] = (len(rows),
                     c.get("1", 0) + c.get("1.0", 0),
                     c.get("0", 0) + c.get("0.0", 0),
                     f.parent.name)
    return out


def _find_paudio(roots) -> list:
    """Every .paudio under the given roots, de-duplicated by stem (largest wins).

    Largest wins because the corpus keeps short excerpt files ('..._forwe',
    6250 frames) beside the full recordings of the same name, and a census that
    silently measured the excerpt would understate the job by three orders of
    magnitude. This bit us once already while measuring the v5 -> v5.1 delta.
    """
    best: dict = {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            print(f"  ! root does not exist: {root}", file=sys.stderr)
            continue
        try:
            found = list(root.rglob("*.paudio"))
        except PermissionError:
            print(f"  ! permission denied: {root}", file=sys.stderr)
            continue
        if not found:
            # rglob swallows per-directory PermissionError and yields nothing, so
            # an empty result on a mounted volume is ambiguous. Say so.
            print(f"  ! no .paudio under {root} "
                  f"(if this is an external drive, check Full Disk Access)",
                  file=sys.stderr)
        for p in found:
            try:
                size = p.stat().st_size
            except OSError:
                continue
            cur = best.get(p.stem)
            if cur is None or size > cur[1]:
                best[p.stem] = (p, size)
    return [p for p, _ in sorted(best.values(), key=lambda x: x[0].name.lower())]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+",
                    help="directories to search for .paudio — or, with "
                         "--from-export, the export root(s) holding CSVs/<stem>/")
    ap.add_argument("--from-export", action="store_true",
                    help="count from ALREADY EXPORTED CSVs instead of loading the "
                         ".paudio files. Instant. Use this whenever the export "
                         "exists — the default mode re-does the export's own load, "
                         "which is 99.9%% of its runtime.")
    ap.add_argument("-o", "--output", type=Path, default=Path("corpus_census.csv"))
    ap.add_argument("--old-dir", type=Path,
                    default=Path.home() / "PlantLeaf_dev/Analisi/v5/Dataset",
                    help="v5-era per-recording CSVs, for the existing-label columns")
    ap.add_argument("--k", type=float, default=None,
                    help="Stage 1 multiplier (default: K_STAGE1_DEFAULT)")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N recordings (for a quick look)")
    ap.add_argument("--stage1-only", action="store_true",
                    help="skip the per-candidate fit: gives n_unfiltered but leaves "
                         "n_export_all blank. Faster, but see the timing note — the "
                         "LOAD dominates, so this saves less than it looks like it should")
    args = ap.parse_args()

    from components.data_collection_dialog_v5 import (
        DataCollectionWorkerV5, _process_file_for_collection, EXPORT_UNFILTERED)
    from core.click_pipeline_v5 import (
        K_STAGE1_DEFAULT, run_stage1_v5_precomputed, has_precomputed_stage1_arrays)

    k = args.k if args.k is not None else K_STAGE1_DEFAULT
    old = _load_old_labels(args.old_dir)

    if args.from_export:
        # Before _find_paudio: the export roots hold CSVs, not recordings, and
        # scanning them would warn about missing .paudio that were never expected.
        _census_from_export(args.roots, args.output, old)
        _summarise(args.output)
        return 0

    files = _find_paudio(args.roots)
    if args.limit:
        files = files[:args.limit]

    done = set()
    if args.output.exists():
        with open(args.output, newline="") as fh:
            done = {r["file"] for r in csv.DictReader(fh)}
        print(f"resuming: {len(done)} recording(s) already in {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", newline="") as fh:
            csv.writer(fh).writerow(COLUMNS)

    print(f"{len(files)} recording(s) to census, k = {k}\n")

    class _Sig:
        def emit(self, *a, **kw): pass
        def connect(self, *a, **kw): pass

    for n, path in enumerate(files, 1):
        if path.stem in done:
            continue
        t0 = time.time()
        row = {c: "" for c in COLUMNS}
        o = old.get(path.stem)
        row.update(file=path.stem, path=str(path),
                   group=(o[3] if o else path.parent.name),
                   old_rows=(o[0] if o else 0),
                   old_clicks=(o[1] if o else 0),
                   old_noise=(o[2] if o else 0))
        try:
            w = DataCollectionWorkerV5.__new__(DataCollectionWorkerV5)
            w._current_audio_worker = None
            w._stop_requested = False
            w.load_progress = _Sig()
            w.error_occurred = _Sig()
            dm = DataCollectionWorkerV5._load_audio_file(w, path)
            row["load_s"] = round(time.time() - t0, 1)
            t1 = time.time()
            if dm is None:
                row["note"] = "load failed"
            else:
                fs = dm.header_info.get("fs", 200_000)
                nfft = dm.header_info.get("fft_size", 512)
                row["total_frames"] = dm.total_frames
                row["duration_s"] = round(dm.total_frames * nfft / fs, 1)
                if args.stage1_only:
                    if has_precomputed_stage1_arrays(dm):
                        row["n_unfiltered"] = len(run_stage1_v5_precomputed(dm, k=k))
                    else:
                        row["note"] = "no precomputed arrays; needs a full pass"
                else:
                    # ONE pass, unfiltered: fit_valid then tells us what EXPORT_ALL
                    # would have written, so both numbers cost one load.
                    cands, _ = _process_file_for_collection(
                        dm, k=k, export_mode=EXPORT_UNFILTERED)
                    n_all = sum(1 for c in cands if c.fit_valid)
                    row["n_unfiltered"] = len(cands)
                    row["n_export_all"] = n_all
                    row["n_confirmed"] = sum(1 for c in cands if c.is_confirmed_click)
                    row["png_mb"] = round(n_all * PNG_KB / 1024.0, 1)
            row["process_s"] = round(time.time() - t1, 1)
        except Exception as exc:                              # noqa: BLE001
            row["note"] = f"{type(exc).__name__}: {exc}"[:160]
            if not row["load_s"]:
                row["load_s"] = round(time.time() - t0, 1)
        with open(args.output, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=COLUMNS).writerow(row)
        print(f"  [{n:>3}/{len(files)}] {path.stem[:42]:<42} "
              f"all={str(row['n_export_all']):>6} unf={str(row['n_unfiltered']):>7} "
              f"load={row['load_s']:>6}s proc={row['process_s']:>6}s {row['note']}",
              flush=True)

    _summarise(args.output)
    return 0


def _census_from_export(roots, out_path: Path, old: dict) -> None:
    """
    Count from exported *_candidates.csv instead of re-loading the recordings.

    The export already did the expensive work and wrote the answer to disk. This
    reads `fit_valid` to split EXPORT_ALL from EXPORT_UNFILTERED — an export made
    in EXPORT_ALL mode has fit_valid == 1 on every row, so the two columns come
    out equal, which is correct rather than a bug.
    """
    seen: dict = {}
    for root in roots:
        for f in Path(root).rglob("*_candidates.csv"):
            stem = f.stem.replace("_candidates", "")
            try:
                with open(f, encoding="utf-8-sig") as fh:
                    rows = list(csv.DictReader(fh))
            except OSError:
                continue
            if not rows:
                continue
            n_all = sum(1 for r in rows if (r.get("fit_valid") or "").strip() in ("1", "1.0"))
            lab = Counter((r.get("label") or "").strip() for r in rows)
            o = old.get(stem)
            seen[stem] = {
                "file": stem, "group": (o[3] if o else f.parent.parent.name),
                "path": str(f),
                "total_frames": "", "duration_s": "",
                "n_unfiltered": len(rows),
                "n_export_all": n_all or len(rows),
                "n_confirmed": sum(1 for r in rows
                                   if (r.get("stage_blocked") or "").strip() == ""
                                   and (r.get("svm_prediction") or "").strip() != ""),
                "png_mb": round((n_all or len(rows)) * PNG_KB / 1024.0, 1),
                "old_rows": (o[0] if o else 0),
                "old_clicks": (o[1] if o else 0),
                "old_noise": (o[2] if o else 0),
                "load_s": 0, "process_s": 0,
                # Labels ALREADY made in this export, which is what you want to see
                # while a labelling pass is in flight.
                "note": (f"labelled {lab.get('1',0)} click / {lab.get('0',0)} noise"
                         if (lab.get('1', 0) or lab.get('0', 0)) else ""),
            }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for stem in sorted(seen):
            w.writerow(seen[stem])
    print(f"read {len(seen)} exported recording(s) -> {out_path}")


def _summarise(path: Path) -> None:
    """The three numbers the subset decision actually turns on."""
    with open(path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("n_export_all") not in ("", None)]
    if not rows:
        return

    def _i(r, k_):
        try:
            return int(float(r[k_] or 0))
        except (TypeError, ValueError):
            return 0

    rows.sort(key=lambda r: -_i(r, "n_export_all"))
    tot = sum(_i(r, "n_export_all") for r in rows)
    print("\n" + "=" * 72)
    print(f"{len(rows)} recordings   {tot:,} rows at EXPORT_ALL   "
          f"{tot * PNG_KB / 1024 / 1024:.1f} GB of screenshots")
    print("=" * 72)
    n_top = min(12, len(rows))
    print(f"\nbiggest {n_top} — these are the ones to SAMPLE rather than sweep:")
    print(f"  {'EXPORT_ALL':>10} {'unfiltered':>10} {'old clicks':>10}  recording")
    cum = 0
    for r in rows[:n_top]:
        cum += _i(r, "n_export_all")
        print(f"  {_i(r,'n_export_all'):>10} {_i(r,'n_unfiltered'):>10} "
              f"{_i(r,'old_clicks'):>10}  {r['file'][:46]}")
    print(f"\n  those {n_top} are {100*cum/max(1,tot):.0f} % of the whole census")
    rest_n, rest = len(rows) - n_top, tot - cum
    if rest_n > 0:
        print(f"  the remaining {rest_n} recordings total {rest:,} rows "
              f"({rest * PNG_KB / 1024:.0f} MB) — the exhaustive-pass candidates")


if __name__ == "__main__":
    sys.exit(main())
