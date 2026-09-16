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
apply_labels.py — carry recovered v6 labels onto a fresh v6 export, exactly

    python3 scripts/v6/apply_labels.py NEW_EXPORT_CSVS_DIR \
        --labels ~/PlantLeaf_dev/Analisi/v6/training_set_27082026_ambiguousincluded_v5evaluated.csv \
        --out LABELLED_DIR

WHY NOT migrate_labels_v6.py
----------------------------
That tool exists for v5 -> v6, where the old rows carry only `(file, frame_idx)`.
That key is NOT unique — the fi and fi+1 candidates of one straddling click are
separate rows — so it has to resolve ambiguity with a ±DEDUP_WINDOW neighbourhood
search and hand the doubtful cases to a human.

None of that applies here. Both sides are v6 and both carry `peak_abs`, which is
exact integer sample arithmetic and identical for both candidates of a
straddling click. The join is therefore exact, one-to-one, and needs no review
queue. Verified on stimolomeccanico_aloe_misurazione1: every feature on every
joined row is bit-identical between the 27-08 dataset and a fresh export, so the
detector has not moved under the labels.

SAFETY
------
Reads the export and the label file; writes a THIRD tree. Neither input is
modified. If a recording has no labels it is still copied, so the output is a
complete export that analyze_dataset.py can consume in one pass.

WHAT IT REPORTS
---------------
Unmatched labels are the number that matters: a label whose `peak_abs` is not in
the new export means that candidate was not exported, which for a v6 export means
the export mode dropped it. If this number is not ~0, you exported in the wrong
mode — see batch_export.py on `--export-mode unfiltered`.

GROUPED LAYOUTS  (--conditions)
-------------------------------
analyze_dataset.py rolls statistics up by folder, so the folder tree IS the
experimental design. With `--conditions recording_conditions.csv` (drafted by
draft_conditions.py, corrected by hand) the output becomes:

    OUT/_all/<rec>_candidates.csv                         every labelled CSV (real files)
    OUT/by_environment/<environment>/<condition>/<rec>    symlinks, include == 1 only
    OUT/by_condition/<condition>/<environment>/<rec>      symlinks, include == 1 only

Two nestings because the roll-up only totals along its own nesting: by_environment
answers "all outdoor", by_condition answers "all mechanical". Symlinks, so the second
layout costs no disk. Point analyze_dataset.py at ONE layout directory per run —
never at OUT itself, which would count every recording three times.

The layouts are rebuilt from scratch on every run, so correcting the table and
re-running is the whole workflow. A recording in the export but absent from the
table is not guessed into a group; it is listed and left out.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Exact peak_abs label join onto a fresh v6 export",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("export_dir", type=Path,
                    help="the CSVs/ directory of a batch_export.py run")
    ap.add_argument("--labels", required=True, type=Path,
                    help="recovered dataset CSV carrying file/peak_abs/label")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--conditions", type=Path, default=None,
                    help="recording_conditions.csv — build the grouped layouts")
    a = ap.parse_args()

    conditions = None
    if a.conditions is not None:
        if not a.conditions.exists():
            print(f"ERROR: conditions table not found: {a.conditions}")
            return 2
        conditions = pd.read_csv(a.conditions, dtype=str, keep_default_na=False,
                                 encoding="utf-8-sig")
        need = {"file", "environment", "condition", "include", "needs_review"}
        if not need.issubset(conditions.columns):
            print(f"ERROR: conditions table lacks {sorted(need - set(conditions.columns))}")
            return 2
        dups = conditions["file"][conditions["file"].duplicated()].tolist()
        if dups:
            print(f"ERROR: recordings listed twice in the conditions table: {dups}")
            return 2

    if not a.labels.exists():
        print(f"ERROR: label file not found: {a.labels}")
        return 2
    lab = pd.read_csv(a.labels, low_memory=False)
    for col in ("file", "peak_abs", "label"):
        if col not in lab.columns:
            print(f"ERROR: label file has no '{col}' column")
            return 2
    lab = lab[pd.to_numeric(lab["label"], errors="coerce").notna()].copy()
    lab["peak_abs"] = pd.to_numeric(lab["peak_abs"], errors="coerce").astype("Int64")
    lab = lab.dropna(subset=["peak_abs"])
    print(f"labels: {len(lab)} row(s) over {lab['file'].nunique()} recording(s)")

    csvs = sorted(p for p in a.export_dir.rglob("*_candidates.csv")
                  if not p.name.startswith("._"))
    if not csvs:
        print(f"ERROR: no *_candidates.csv under {a.export_dir}")
        return 2
    print(f"export: {len(csvs)} recording CSV(s)\n")

    a.out.mkdir(parents=True, exist_ok=True)
    tot_applied = tot_lab = 0
    per_file = []

    by_rec = {k: v for k, v in lab.groupby("file")}
    for src in csvs:
        stem = src.name.replace("_candidates.csv", "")
        df = pd.read_csv(src, low_memory=False)
        # Flat store when grouping, so the layouts can link to one copy.
        dst_dir = (a.out / "_all") if conditions is not None else (a.out / stem)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name

        sub = by_rec.get(stem)
        if sub is None or "peak_abs" not in df.columns:
            shutil.copyfile(src, dst)
            per_file.append((stem, len(df), 0, 0))
            continue

        df["peak_abs"] = pd.to_numeric(df["peak_abs"], errors="coerce").astype("Int64")
        mapping = dict(zip(sub["peak_abs"], pd.to_numeric(sub["label"], errors="coerce")))
        if "label" not in df.columns:
            df["label"] = pd.NA
        applied = df["peak_abs"].map(mapping)
        # Only write where a label exists; never blank an existing one.
        df.loc[applied.notna(), "label"] = applied[applied.notna()].astype(int)
        n_applied = int(applied.notna().sum())
        n_missing = len(sub) - n_applied
        df.to_csv(dst, index=False)

        tot_applied += n_applied
        tot_lab += len(sub)
        per_file.append((stem, len(df), n_applied, n_missing))

    print(f"{'recording':<58}{'rows':>8}{'applied':>9}{'UNMATCHED':>11}")
    for stem, n, ap_, miss in sorted(per_file, key=lambda x: -x[3]):
        if ap_ or miss:
            print(f"{stem[:56]:<58}{n:>8}{ap_:>9}{miss:>11}")
    print(f"\napplied {tot_applied}/{tot_lab} label(s); "
          f"{tot_lab - tot_applied} unmatched")
    if tot_lab - tot_applied:
        print("\n⚠️  Unmatched labels mean those candidates are NOT in the export.")
        print("   For a v6 export that almost always means the wrong --export-mode:")
        print("   anything but `unfiltered` drops fit_valid == 0 before Stage 2.")
    if conditions is not None:
        rc = build_layouts(a.out, [x[0] for x in per_file], conditions)
        if rc:
            return rc
    print(f"\nwrote: {a.out}")
    return 0


def _safe(name: str) -> str:
    name = (name or "").strip()
    if not name or name == "?":
        return "_unclassified"
    return name.replace("/", "-")


def _clear_layout(root: Path) -> bool:
    """Remove a layout tree we generated. Refuses if it holds any real file."""
    if not root.exists():
        return True
    real = [p for p in root.rglob("*") if p.is_file() and not p.is_symlink()]
    if real:
        print(f"ERROR: {root} contains {len(real)} real file(s), not only the symlinks "
              f"this tool creates — refusing to delete it. First: {real[0]}")
        return False
    shutil.rmtree(root)
    return True


def build_layouts(out: Path, exported: list[str], table: "pd.DataFrame") -> int:
    rows = {r["file"]: r for r in table.to_dict("records")}
    layouts = {
        "by_environment": lambda r: (_safe(r["environment"]), _safe(r["condition"])),
        "by_condition":   lambda r: (_safe(r["condition"]), _safe(r["environment"])),
    }
    for name in layouts:
        if not _clear_layout(out / name):
            return 2

    placed, excluded, untabled = 0, [], []
    for stem in sorted(exported):
        r = rows.get(stem)
        if r is None:
            untabled.append(stem)
            continue
        if str(r["include"]).strip() != "1":
            excluded.append((stem, r.get("exclude_reason", "")))
            continue
        target = out / "_all" / f"{stem}_candidates.csv"
        for name, key in layouts.items():
            d = out.joinpath(name, *key(r))
            d.mkdir(parents=True, exist_ok=True)
            link = d / target.name
            # Relative, so the whole OUT tree can be moved or copied as a unit.
            link.symlink_to(Path("../" * 3) / "_all" / target.name)
        placed += 1

    not_exported = sorted(set(rows) - set(exported))
    review = [f for f, r in rows.items()
              if str(r["needs_review"]).strip() == "1" and str(r["include"]).strip() == "1"
              and f in exported]

    print(f"\nlayouts: {placed} recording(s) placed, {len(excluded)} excluded by the table")
    from collections import Counter
    grid = Counter((rows[f]["environment"], rows[f]["condition"]) for f in exported
                   if f in rows and str(rows[f]["include"]).strip() == "1")
    for (env, cond), n in sorted(grid.items()):
        print(f"   {env:<8} {cond:<14} {n}")
    if untabled:
        print(f"\n⚠️  {len(untabled)} exported recording(s) are NOT in the conditions table "
              f"and were left out of both layouts — re-run draft_conditions.py:")
        for s_ in untabled[:20]:
            print(f"     {s_}")
    if not_exported:
        print(f"\n   {len(not_exported)} table row(s) have no export yet (not placed)")
    if review:
        print(f"\n⚠️  {len(review)} placed recording(s) still have needs_review = 1 — "
              f"their group may be wrong until you check them.")
    print("\n   Run analyze_dataset.py on by_environment/ and by_condition/ SEPARATELY,")
    print("   never on the output root (it would count each recording three times).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
