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
batch_export.py — re-export candidate CSVs from .paudio, headless and resumable

    python3 scripts/v6/batch_export.py CORPUS_ROOT [...] -o OUT_DIR \
        --model src/ml/v6/plantleaf_svm_v6_DEPLOYED.pkl

WHY THIS EXISTS
---------------
The Data Collection dialog already does batch export, but only with a GUI in
front of it and a human watching a progress bar for hours. A corpus-scale
re-export is an overnight job, so it needs three things the dialog cannot give:
it must run headless, it must survive being interrupted, and it must not render
screenshots.

NO SCREENSHOTS, AND WHY THAT IS FREE
------------------------------------
DataCollectionWorkerV5 writes the CSV first and only then EMITS each candidate on
`candidate_ready` for the main thread to render. Nothing here connects that
signal, so the emit goes nowhere and no PNG is produced. That is not a hack — it
is the existing separation between "the data" and "the pictures a human reviews".
At ~112 kB per candidate over ~200 000 candidates, skipping them saves ~22 GB and
the QPainter time that dominates the per-candidate cost.

⚠️ You need the screenshots if you intend to LABEL this export in the review
dialog. Use the GUI for that. This tool is for the recordings you want measured,
not reviewed.

MEMORY — THE LIMIT IS RAM, NOT CORES
------------------------------------
Measured peak memory footprint is ~4.25x the .paudio size: 5.6 GB for a 1.32 GB
recording (/usr/bin/time -l). The corpus runs to 2-4.5 GB per file, i.e. 9-19 GB
for ONE process. On a 16 GB machine, parallel shards that happen to load large
files together are killed by macOS jetsam — silently, with no traceback and no
FAILED line, the log just stops. That is what ended 4 of 6 shards on 16-09-2026
(JetsamEvent at 01:17: six processes at 5.0-10.6 GB).

So parallelise by SIZE CLASS: run small files in parallel with `--max-gb`, and
the large ones one at a time with `--min-gb`. run_full_export.sh does exactly
this. Budget ~4.5x file size per concurrent process, and leave ~5 GB for macOS.

RESUMABLE
---------
A recording whose CSV already exists is skipped, so an interrupted run costs one
file and a re-run picks up where it stopped. `--force` re-does them anyway.

EXPORT MODE — `unfiltered` IS THE ONLY CORRECT ONE FOR v6
----------------------------------------------------------
Every other mode drops `fit_valid == 0` candidates at BUILD time, before Stage 2
ever sees them (data_collection_dialog_v5.py, the `### FILTERS ###` block). That
filter predates v6 and is exactly the v5 fit gate — the thing v6's Stage 2
redesign exists to remove, because it "costs 12.2 % of confirmed clicks".

Measured on the recovered 27-08-2026 labelled set, exporting in any mode other
than `unfiltered` would silently discard:

        9.0 % of confirmed clicks   (17 of 189)
       22.2 % of ambiguous rows     (22 of 99)

They never reach the CSV, so no later stage can recover them and nothing in the
report says they are missing. A v6 recall figure computed on such an export is
wrong and looks fine.

    unfiltered  (default) EVERY Stage 1 candidate. ~204 k rows corpus-wide, but
                a row is ~640 B, so that is ~130 MB — the cost is disk-trivial and
                the load time dominates regardless.
    stage3      cleared Stage 2. Lean and fine for a v5-era question; for v6 it
                carries the fit-gate loss above.
    all         valid fit only. Same loss.
    confirmed   final clicks only. NOT suitable for analyze_dataset.py — with no
                negatives its candidate counts are meaningless.

SCREENSHOTS — DECOUPLED FROM THE EXPORT MODE, ON PURPOSE
--------------------------------------------------------
`--screenshots` says which candidates get a PNG; `--export-mode` says which get a
CSV ROW. They must be separate, because the two have opposite pressures: the CSV
wants `unfiltered` (or it loses 9 % of clicks), while an outdoor recording emits
up to 344 773 candidates/hour and rendering a picture for each is neither
possible nor useful.

    none       (default) measurement only. Nothing is rendered.
    confirmed  a PNG only for candidates that survived all four stages — the set
               a human would review to find the algorithm's FALSE POSITIVES.
               This is the one you want for labelling a noise/outdoor negative
               set: bounded by how often the detector fires, not by how noisy
               the recording is.
    stage3     a PNG for everything that reached the SVM. Much larger.
    all        a PNG per exported row. On outdoor recordings this is tens of
               thousands of files per hour; it exists for completeness.

A Stage-4 duplicate never gets its own PNG — it is the same physical click as
its confirmed sibling, so one picture serves both, exactly as the dialog does it.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

import windows  # noqa: E402,F401  (breaks a pre-existing circular import)

from PySide6.QtWidgets import QApplication          # noqa: E402
from components.data_collection_dialog_v5 import (  # noqa: E402
    DataCollectionWorkerV5, CSVS_FOLDER, SCREENSHOTS_FOLDER,
    _render_candidate_screenshot,
    EXPORT_ALL, EXPORT_CONFIRMED, EXPORT_UNFILTERED, EXPORT_STAGE3,
)
from core.click_pipeline_v5 import (   # noqa: E402
    load_svm_model, STAGE2_MODES, STAGE2_MODE, STAGE_OK, STAGE_BLOCKED_DEDUP,
)

MODES = {
    'stage3': EXPORT_STAGE3, 'all': EXPORT_ALL,
    'unfiltered': EXPORT_UNFILTERED, 'confirmed': EXPORT_CONFIRMED,
}


def find_paudio(roots: list[Path]) -> list[Path]:
    """Every .paudio under the roots, de-duplicated by stem, largest wins.

    Largest wins because the corpus keeps short excerpts beside the full
    recording of the same name ('..._forwebsite', 6250 frames vs 2.5 M). Exporting
    the excerpt instead would understate that recording by three orders of
    magnitude — corpus_census.py was bitten by exactly this.
    """
    best: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            print(f"  ! root does not exist: {root}", file=sys.stderr)
            continue
        # A root may be a single .paudio — handy for a timed trial or for
        # re-doing one recording — or a directory to walk.
        try:
            found = ([root] if root.is_file() and root.suffix == ".paudio"
                     else list(root.rglob("*.paudio")))
        except OSError as e:
            print(f"  ! cannot read {root}: {e.strerror}", file=sys.stderr)
            continue
        for p in found:
            if p.name.startswith("._"):
                continue                      # macOS AppleDouble sidecar
            cur = best.get(p.stem)
            try:
                if cur is None or p.stat().st_size > cur.stat().st_size:
                    best[p.stem] = p
            except OSError:
                continue
    return [best[k] for k in sorted(best)]


MANIFEST_HEADER = ["file", "source", "size_bytes", "rows", "seconds", "status"]


def _append_manifest(path: Path, row: list) -> None:
    """Append one row under an exclusive lock, writing the header only if empty.

    Shards run as separate processes against ONE manifest. The first version
    checked `exists()` at start-up and wrote a header if not — a check-then-act
    race: all shards start within a second, several see no file, and each writes
    a header, leaving stray header lines inside the data (measured: two of them in
    the 16-09-2026 run). Deciding "is it empty" INSIDE the lock, per row, makes
    the header exactly-once regardless of how many shards start together.
    """
    import fcntl
    with open(path, "a+", newline="", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.seek(0, os.SEEK_END)
            w = csv.writer(fh)
            if fh.tell() == 0:
                w.writerow(MANIFEST_HEADER)
            w.writerow(row)
            fh.flush()
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Headless, resumable candidate export from .paudio",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("-o", "--output-dir", required=True, type=Path)
    ap.add_argument("--model", type=Path,
                    default=_REPO / "src/ml/v6/plantleaf_svm_v6_DEPLOYED.pkl",
                    help="SVM .pkl. Pass --model none for a plain Stage 1 dump.")
    ap.add_argument("--export-mode", default="unfiltered", choices=sorted(MODES))
    ap.add_argument("--stage2-mode", default=STAGE2_MODE, choices=list(STAGE2_MODES))
    ap.add_argument("-k", type=float, default=1.5, help="Stage 1 multiplier")
    ap.add_argument("--screenshots", default="none",
                    choices=["none", "confirmed", "stage3", "all"],
                    help="which candidates get a PNG (default none). Needed only "
                         "if you intend to LABEL this export in the review dialog.")
    ap.add_argument("--force", action="store_true",
                    help="re-export recordings whose CSV already exists")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="process only shard I of N (1-based), e.g. 3/6. Export is "
                         "CPU-bound (~99 %% of one core on the .paudio decode), so N "
                         "processes on N cores is a near-linear speedup. Shards are "
                         "interleaved, not contiguous, so each gets a similar mix of "
                         "large and small recordings instead of one shard drawing "
                         "every 2 GB file.")
    ap.add_argument("--max-gb", type=float, default=None,
                    help="only recordings of at most this many GB. See MEMORY.")
    ap.add_argument("--min-gb", type=float, default=None,
                    help="only recordings LARGER than this many GB. See MEMORY.")
    ap.add_argument("--only-from", type=Path, default=None, metavar="CSV",
                    help="export only recordings named in this CSV's `file` column "
                         "(e.g. the labelled dataset), so the valuable ones go first")
    ap.add_argument("--order", default="name", choices=["name", "largest-first"],
                    help="processing order. largest-first front-loads the long "
                         "recordings, which must run one at a time anyway.")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N recordings (for a timed trial run)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be exported, touch nothing")
    a = ap.parse_args()

    app = QApplication.instance() or QApplication([])      # noqa: F841

    files = find_paudio(a.roots)
    if not files:
        print("No .paudio found.")
        return 2

    csvs_root = a.output_dir / CSVS_FOLDER
    todo, done = [], []
    for p in files:
        out_csv = csvs_root / p.stem / f"{p.stem}_candidates.csv"
        (done if (out_csv.exists() and not a.force) else todo).append(p)
    # Size filters come BEFORE sharding, so every shard of a pass draws from the
    # same size class and none of them can end up holding two of the big files.
    if a.only_from is not None:
        try:
            with open(a.only_from, newline="", encoding="utf-8-sig") as fh:
                wanted = {r["file"].strip() for r in csv.DictReader(fh) if r.get("file")}
        except (OSError, KeyError) as e:
            print(f"ERROR: cannot read a `file` column from {a.only_from}: {e}")
            return 2
        todo = [p for p in todo if p.stem in wanted]
        print(f"only-from  : {len(wanted)} named recording(s) in {a.only_from.name}; "
              f"{len(todo)} still to export")
    if a.max_gb is not None:
        todo = [p for p in todo if p.stat().st_size / 1e9 <= a.max_gb]
    if a.min_gb is not None:
        todo = [p for p in todo if p.stat().st_size / 1e9 > a.min_gb]
    if a.order == "largest-first":
        todo.sort(key=lambda p: p.stat().st_size, reverse=True)
    if a.shard:
        try:
            i, n = (int(x) for x in a.shard.split("/"))
            if not (1 <= i <= n):
                raise ValueError
        except ValueError:
            print(f"ERROR: --shard must be I/N with 1 <= I <= N, got {a.shard!r}")
            return 2
        todo = todo[i - 1::n]
        print(f"shard      : {i} of {n}")
    if a.limit:
        todo = todo[:a.limit]

    total_gb = sum(p.stat().st_size for p in todo) / 1e9
    print(f"found      : {len(files)} recording(s)")
    print(f"already done: {len(done)}  (skipped; --force to redo)")
    print(f"to export  : {len(todo)}   ({total_gb:.1f} GB to read)")
    print(f"mode       : {a.export_mode}   stage2={a.stage2_mode}   k={a.k}")
    print(f"screenshots: {a.screenshots}")
    if a.export_mode != "unfiltered":
        print("   ⚠️  this mode drops fit_valid == 0 BEFORE Stage 2 — a measured "
              "9.0 % of clicks\n       and 22.2 % of ambiguous rows. Use "
              "--export-mode unfiltered for v6 work.")
    print(f"output     : {a.output_dir}")
    if a.dry_run:
        for p in todo:
            print(f"   {p.stat().st_size/1e9:6.2f} GB  {p}")
        return 0
    if not todo:
        print("\nNothing to do.")
        return 0

    model = None
    if str(a.model).lower() != "none":
        if not a.model.exists():
            print(f"ERROR: model not found: {a.model}")
            return 2
        model = load_svm_model(a.model)
        print(f"model      : {a.model.name}  threshold={model['threshold']:.3f} "
              f"nan_policy={model.get('nan_policy','sentinel')}")
    else:
        print("model      : none — Stage 1 dump, verdict columns left empty")
    print()

    manifest = a.output_dir / "export_manifest.csv"
    a.output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    for i, p in enumerate(todo, 1):
        gb = p.stat().st_size / 1e9
        print(f"[{i}/{len(todo)}] {p.stem[:58]:<60} {gb:5.2f} GB ", end="", flush=True)
        result = {"rows": 0, "err": None}
        w = DataCollectionWorkerV5(
            [p], k=a.k, output_dir=a.output_dir, normalize_mode=True,
            svm_model=model, export_mode=MODES[a.export_mode],
            stage2_mode=a.stage2_mode)
        # candidate_ready is connected ONLY when screenshots were asked for. Left
        # unconnected the emit goes nowhere and no PNG is produced — that is the
        # whole reason this runs headless at corpus scale.
        if a.screenshots != "none":
            shots = {"n": 0}

            def _render(payload, _shots=shots):
                cand, path = payload
                if cand.stage_blocked == STAGE_BLOCKED_DEDUP:
                    return              # same physical click as its sibling
                if a.screenshots == "confirmed" and cand.stage_blocked != STAGE_OK:
                    return
                if a.screenshots == "stage3" and cand.svm_probability is None:
                    return              # never reached the SVM
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                if _render_candidate_screenshot(cand, Path(path)):
                    _shots["n"] += 1

            # DirectConnection by default and run() is called on THIS thread, so
            # the QPainter work happens on the main thread — which is what macOS
            # requires of any Qt paint device.
            w.candidate_ready.connect(_render)
            result["shots"] = shots
        w.file_complete.connect(lambda f, n, path: result.update({"rows": n}))
        t0 = time.time()
        try:
            w.run()
            status = "ok"
        except Exception as e:                                  # noqa: BLE001
            status, result["err"] = "FAILED", f"{type(e).__name__}: {e}"
        el = time.time() - t0
        n_shots = result.get("shots", {}).get("n", 0)
        print(f" {result['rows']:>7} rows  {el/60:5.1f} min  {status}"
              + (f"  {n_shots} png" if a.screenshots != "none" else "")
              + (f"  {result['err']}" if result["err"] else ""))
        _append_manifest(manifest, [p.stem, str(p), p.stat().st_size,
                                    result["rows"], round(el, 1), status])

        elapsed = time.time() - t_start
        if i < len(todo):
            eta = elapsed / i * (len(todo) - i)
            print(f"          elapsed {elapsed/3600:.2f} h   ETA {eta/3600:.2f} h")

    print(f"\nDone in {(time.time()-t_start)/3600:.2f} h — manifest: {manifest}")
    print(f"Next: python3 src/ml/analyze_dataset.py --model {a.model} "
          f"--paudio-dir <ROOT> {csvs_root} --output-dir <REPORT_DIR>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
