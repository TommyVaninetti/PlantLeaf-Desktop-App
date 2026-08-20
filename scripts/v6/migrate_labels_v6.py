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
migrate_labels_v6.py — carry v5 labels onto a v6 export, and build the review queue
===================================================================================

WHY THIS EXISTS
---------------
The v6 re-export recomputes every feature, so the rows are new objects. But the
labels attached to the old rows are human judgements — **1944** of them, spread
over 57 of the 117 candidate CSVs, of which 285 were curated into
Dataset_20June2026.csv. They are the scarcest resource in this project (v6 §12:
"the bottleneck is not features, it is 91 clicks"). Re-labelling from scratch
would throw all 1944 away and put ~204 000 unlabelled rows in front of a human.

This tool carries the settled labels forward and hands back a queue of ONLY the
rows that genuinely need a person: ambiguous migrations, and new rows.

SAFETY
------
Reads the old CSVs and the new export; writes a THIRD file. **Neither input is
ever modified.** That matters because `_write_csv_for_file` opens candidate CSVs
in 'w' mode and `click_review_dialog._save` writes labels back into the same
files — anything that edits in place can silently destroy labels. A bad run of
this tool costs nothing; just re-run it.

MATCHING
--------
Old rows carry only `(file, frame_idx)`, so that is the key — but it is **not
unique**: the fi and fi+1 Stage-1 candidates of one straddling click are separate
rows, which is how one event in the current dataset ended up labelled both 1 and
0. So a label is auto-carried ONLY when the match is unambiguous:

    exactly one labelled old row in the ±DEDUP_WINDOW_FRAMES neighbourhood
      AND exactly one new row it can attach to

Everything else is flagged for review, never guessed. New rows carry `peak_abs`
(exact sample arithmetic, identical for both candidates of a straddling click),
so future migrations will not have this problem.

OUTPUT
------
ONE FILE PER RECORDING (`<recording>_candidates.csv`), not a single merged CSV,
because that is how labelling actually happens — you open the file for the
recording you are reviewing. click_review_dialog also builds a table widget per
row and rewrites the whole CSV on every keystroke, so a 200k-row merged file
would hang it; per recording the counts stay in the thousands. A small
`migration_report.csv` summarises the queue across all of them.

Usage
-----
    python3 scripts/migrate_labels_v6.py NEW_EXPORT_DIR [--old-dir DIR]
                                         [--master CSV] [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

_DEFAULT_OLD_DIR = Path.home() / "PlantLeaf_dev" / "Analisi" / "v5" / "Dataset"
_DEFAULT_MASTER = (Path.home() / "PlantLeaf_dev" / "Analisi" / "v5" / "SVM_Training"
                   / "Dataset_20June2026.csv")

#: Same neighbourhood Stage 4 used to collapse duplicates. Not a new constant.
DEDUP_WINDOW_FRAMES = 3

#: The three features used to rank tier 3, from the confirmed positives.
_PLAUSIBILITY_KEYS = ("peak_SNR", "kurtosis", "rise_time_ms")

REVIEW_TIERS = {
    1: "ambiguous",        # migration could not be resolved — YOU must adjudicate
    2: "svm_conflict",     # model disagrees with a migrated label, or scores near threshold
    3: "clicklike",        # unlabelled, inside the confirmed-click envelope
    4: "new_fit_invalid",  # fit_valid == 0 — never reached a CSV before
}


def _f(row, key, default=float("nan")):
    try:
        v = str(row.get(key, "") or "").strip().replace(",", ".")
        return float(v) if v else default
    except (TypeError, ValueError):
        return default


def _label_of(row):
    """Normalise the '1.0' / '0.0' / '0' / '2' / '' label mix into 1, 0, 2 or None.

    2 is AMBIGUOUS — the reviewer looked and could not decide. It is carried
    through rather than dropped: returning None for it would make an ambiguous
    row indistinguishable from an unlabelled one, and the migration would then
    treat a deliberate judgement as a gap to be filled.
    """
    raw = str(row.get("label", "") or "").strip().replace(",", ".")
    if not raw:
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    return int(v) if v in (0.0, 1.0, 2.0) else None


# ─────────────────────────────────────────────────────────────────────────────
# Load the old labels
# ─────────────────────────────────────────────────────────────────────────────

def load_old_labels(old_dir: Path, master: Path):
    """
    Every labelled (file, frame_idx) the project has, from the per-recording CSVs
    and from the master training set.

    Returns {file_stem: {frame_idx: [labels...]}} — a LIST per frame, because the
    same coordinate can legitimately appear more than once and the contradictions
    are exactly what tier 1 has to surface.
    """
    out = defaultdict(lambda: defaultdict(list))
    n_files = 0

    sources = []
    if old_dir.exists():
        sources += sorted(old_dir.rglob("*_candidates.csv"))
    if master.exists():
        sources.append(master)

    for path in sources:
        try:
            with open(path, newline="") as fh:
                rows = list(csv.DictReader(fh))
        except Exception as exc:                                  # noqa: BLE001
            print(f"  skip {path.name}: {exc}", file=sys.stderr)
            continue
        hit = False
        for r in rows:
            lab = _label_of(r)
            if lab is None:
                continue
            stem = str(r.get("file", "") or "").strip()
            try:
                fi = int(float(r.get("frame_idx", "nan")))
            except (TypeError, ValueError):
                continue
            if not stem:
                continue
            out[stem][fi].append(lab)
            hit = True
        n_files += bool(hit)

    total = sum(len(v) for d in out.values() for v in d.values())
    print(f"old labels: {total} across {len(out)} recordings "
          f"({n_files} source files contributed)")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plausibility envelope for tier 3
# ─────────────────────────────────────────────────────────────────────────────

def positive_envelope(old_labels_rows, pct: float = 5.0):
    """[p, 100-p] box of the confirmed clicks on the three plausibility features."""
    import numpy as np
    vals = [v for v in old_labels_rows if all(math.isfinite(x) for x in v)]
    if len(vals) < 10:
        return None
    arr = np.asarray(vals)
    return {k: (float(np.percentile(arr[:, j], pct)),
                float(np.percentile(arr[:, j], 100.0 - pct)))
            for j, k in enumerate(_PLAUSIBILITY_KEYS)}, arr


def collect_positive_features(master: Path):
    if not master.exists():
        return []
    out = []
    with open(master, newline="") as fh:
        for r in csv.DictReader(fh):
            if _label_of(r) == 1:
                out.append([_f(r, k) for k in _PLAUSIBILITY_KEYS])
    return out


# ─────────────────────────────────────────────────────────────────────────────

def migrate(new_rows, old, envelope, arr_pos, threshold=0.2639, margin=0.10):
    import numpy as np

    by_file = defaultdict(list)
    for i, r in enumerate(new_rows):
        by_file[str(r.get("file", "") or "").strip()].append(i)

    centroid = arr_pos.mean(axis=0) if arr_pos is not None and len(arr_pos) else None
    scale = arr_pos.std(axis=0) if centroid is not None else None
    if scale is not None:
        scale[scale == 0] = 1.0

    stats = defaultdict(int)
    for r in new_rows:
        r["label"] = ""
        r["label_source"] = "new"
        r["needs_review"] = 0
        r["review_tier"] = ""
        r["migration_note"] = ""
        r["clicklike_rank"] = ""

    for stem, idxs in by_file.items():
        frames = old.get(stem)
        if not frames:
            continue
        for fi_old, labels in frames.items():
            # every new row whose frame_idx is within the dedup neighbourhood
            cands = [i for i in idxs
                     if abs(int(float(new_rows[i]["frame_idx"])) - fi_old)
                     <= DEDUP_WINDOW_FRAMES]
            uniq = set(labels)
            if len(cands) == 1 and len(labels) == 1:
                row = new_rows[cands[0]]
                row["label"] = str(labels[0])
                row["label_source"] = "migrated"
                row["migration_note"] = f"exact: old frame_idx {fi_old}"
                stats["migrated"] += 1
            elif not cands:
                stats["orphaned"] += 1
                # The old label has nowhere to land: the row it described is not in
                # the new export. Recorded on no row, but counted and reported —
                # a silent drop here would be a lost human judgement.
            else:
                for i in cands:
                    row = new_rows[i]
                    row["needs_review"] = 1
                    row["review_tier"] = 1
                    row["label_source"] = "ambiguous"
                    row["migration_note"] = (
                        f"old frame_idx {fi_old}: {len(labels)} label(s) "
                        f"{sorted(uniq)} -> {len(cands)} new row(s)"
                        + ("  CONTRADICTORY" if len(uniq) > 1 else ""))
                stats["ambiguous"] += 1

    # tiers 2-4, only for rows not already tier 1
    for r in new_rows:
        if r["review_tier"]:
            continue
        fit_valid = _f(r, "fit_valid", 1.0)
        prob = _f(r, "svm_probability")
        lab = _label_of(r)

        if lab is not None and math.isfinite(prob):
            pred = 1 if prob >= threshold else 0
            if pred != lab or abs(prob - threshold) < margin:
                r["needs_review"] = 1
                r["review_tier"] = 2
                r["migration_note"] = (f"svm p={prob:.4f} vs migrated label {lab}")
                stats["svm_conflict"] += 1
                continue

        if lab is None and envelope is not None:
            v = {k: _f(r, k) for k in _PLAUSIBILITY_KEYS}
            if all(math.isfinite(x) for x in v.values()) and \
               all(envelope[k][0] <= v[k] <= envelope[k][1] for k in envelope):
                r["needs_review"] = 1
                r["review_tier"] = 3
                if centroid is not None:
                    z = [(v[k] - centroid[j]) / scale[j]
                         for j, k in enumerate(_PLAUSIBILITY_KEYS)]
                    r["clicklike_rank"] = round(float(sum(x * x for x in z) ** 0.5), 4)
                r["migration_note"] = "inside the confirmed-click envelope"
                stats["clicklike"] += 1
                continue

        if lab is None and fit_valid == 0:
            r["needs_review"] = 1
            r["review_tier"] = 4
            r["migration_note"] = "fit_valid=0 — never exported before v6"
            stats["new_fit_invalid"] += 1

    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("new_export", help="directory of v6 *_candidates.csv files")
    ap.add_argument("--old-dir", default=str(_DEFAULT_OLD_DIR))
    ap.add_argument("--master", default=str(_DEFAULT_MASTER))
    ap.add_argument("--out", default=None,
                    help="output DIRECTORY (default: alongside the new export). "
                         "One <recording>_candidates.csv is written per recording, "
                         "plus migration_report.csv.")
    ap.add_argument("--threshold", type=float, default=0.2639)
    args = ap.parse_args()

    new_dir = Path(args.new_export).expanduser()
    out_path = Path(args.out) if args.out else (new_dir / "migrated")

    new_files = sorted(new_dir.rglob("*_candidates.csv"))
    if not new_files:
        print(f"no *_candidates.csv under {new_dir}", file=sys.stderr)
        return 1

    new_rows, fieldnames = [], None
    for f in new_files:
        with open(f, newline="") as fh:
            rd = csv.DictReader(fh)
            fieldnames = fieldnames or rd.fieldnames
            new_rows.extend(rd)
    print(f"new export: {len(new_rows)} rows across {len(new_files)} files")

    old = load_old_labels(Path(args.old_dir).expanduser(), Path(args.master).expanduser())
    pos = collect_positive_features(Path(args.master).expanduser())
    env = positive_envelope(pos)
    envelope, arr_pos = env if env else (None, None)
    if envelope:
        print(f"plausibility envelope from {len(arr_pos)} confirmed clicks (p5-p95)")

    stats = migrate(new_rows, old, envelope, arr_pos, threshold=args.threshold)

    extra = ["label_source", "needs_review", "review_tier", "clicklike_rank",
             "migration_note"]
    cols = list(fieldnames) + [c for c in extra if c not in fieldnames]
    # tier order, then click-likeness within tier 3
    new_rows.sort(key=lambda r: (
        int(r["review_tier"]) if r["review_tier"] else 99,
        float(r["clicklike_rank"]) if r["clicklike_rank"] else 0.0,
    ))
    # ── Write ONE FILE PER RECORDING, mirroring the export layout ────────────
    # Not one giant merged CSV. Labelling happens per recording — you open the
    # file for the recording you are reviewing — and click_review_dialog builds a
    # table widget per row and rewrites the whole CSV on every keystroke, so a
    # 200k-row file would hang it. Per recording the counts stay in the thousands,
    # where that code is perfectly fine. It also means a mistake costs one file.
    by_file = defaultdict(list)
    for r in new_rows:
        by_file[str(r.get("file", "") or "unknown").strip()].append(r)

    out_dir = out_path.parent if out_path.suffix else out_path
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for stem, rows_f in sorted(by_file.items()):
        dest = out_dir / f"{stem}_candidates.csv"
        with open(dest, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows_f)
        written.append((stem, dest, len(rows_f),
                        sum(1 for r in rows_f if r["needs_review"])))

    # ── One small report so the queue is still visible in one place ──────────
    report = out_dir / "migration_report.csv"
    with open(report, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "rows", "needs_review",
                    "tier1_ambiguous", "tier2_svm", "tier3_clicklike",
                    "tier4_new_fit_invalid", "migrated"])
        for stem, _dest, n_rows, n_rev in written:
            rf = by_file[stem]
            w.writerow([stem, n_rows, n_rev]
                       + [sum(1 for r in rf if str(r["review_tier"]) == str(t))
                          for t in (1, 2, 3, 4)]
                       + [sum(1 for r in rf if r["label_source"] == "migrated")])

    print(f"\n{'=' * 66}")
    print(f"labels migrated automatically : {stats['migrated']}")
    print(f"old labels with no new row    : {stats['orphaned']}"
          + ("   ⚠️  these judgements are LOST — investigate before labelling"
             if stats["orphaned"] else ""))
    print("\nreview queue, in the order the file is sorted:")
    for t in (1, 2, 3, 4):
        n = sum(1 for r in new_rows if str(r["review_tier"]) == str(t))
        print(f"  tier {t}  {REVIEW_TIERS[t]:16s} {n:7d}")
    print(f"\n  no review needed              {sum(1 for r in new_rows if not r['review_tier']):7d}")
    print(f"\nwrote {len(written)} per-recording CSV(s) to {out_dir}")
    for stem, dest, n_rows, n_rev in written[:8]:
        print(f"    {dest.name[:52]:52s} {n_rows:7d} rows, {n_rev:6d} to review")
    if len(written) > 8:
        print(f"    … and {len(written) - 8} more")
    print(f"  summary: {report}")
    print("⚠️  tier 2 (svm_conflict) orders rows by the CURRENT model's uncertainty,")
    print("    which biases new labels toward that model's blind spots. It is tagged")
    print("    so it can be excluded at analysis time; say so in any result that")
    print("    depends on tier-2-derived labels.")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
