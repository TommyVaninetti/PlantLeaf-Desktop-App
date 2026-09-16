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
draft_conditions.py — a first-pass recording → condition table, for a human to correct

    python3 scripts/v6/draft_conditions.py \
        --manifest ~/PlantLeaf_dev/Analisi/v6/re-export/export_manifest.csv \
        -o ~/PlantLeaf_dev/Analisi/v6/recording_conditions.csv

WHY A TABLE AND NOT THE FOLDER TREE
-----------------------------------
analyze_dataset.py rolls statistics up by the folder a CSV sits in, so the
grouping IS the experimental design. The drive's folder tree cannot serve as that
design: measured on the 16-09-2026 export, 8 of 30 folders hold recordings of more
than one condition (Pomodori/Solo pianta holds watering sessions, Aloe Vera/meccanico
holds a cut), 39 of 111 names say nothing about indoor vs outdoor, and 16 carry no
stimulus marker at all. Any automatic grouping would be quietly impure.

So this script only DRAFTS. Every value records its basis — `name` (a marker in the
filename), `folder` (inferred from the folder), or `GUESS` — and any row resting on a
guess is marked needs_review = 1. Correct those in a spreadsheet; apply_labels.py
then builds the grouped layouts from the corrected table.

COLUMNS
-------
    file              recording stem — the join key, never edit
    source_folder     folder on the corpus drive, relative to Audio Experiments
    environment       indoor | outdoor
    environment_basis name | folder | GUESS
    condition         plant-only | mechanical | cut | detached | insects | water |
                      no-water | harvest+water | noise | insects-only
    condition_basis   name | folder | GUESS
    include           1 = counted in rates and time distributions, 0 = excluded
    exclude_reason    why include = 0
    needs_review      1 while any value above is a guess
    note              free text

`water`, `no-water` and `harvest+water` are deliberately separate classes rather
than folded into "stressed": whether being watered is a stress, a relief or neither
is a question for the analysis, not an assumption to bake into the grouping.

RE-RUNNING NEVER OVERWRITES YOUR CORRECTIONS
--------------------------------------------
If the output exists, every row already in it is kept byte-for-byte and only
recordings that are new to the manifest are appended (as drafts). Run it again after
the remaining recordings are exported; your edits survive.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

COLUMNS = [
    "file", "source_folder",
    "environment", "environment_basis",
    "condition", "condition_basis",
    "include", "exclude_reason", "needs_review", "note",
]

ANCHOR = "Audio Experiments/"

# ── Exclusions (decided 16-09-2026) ──────────────────────────────────────────
# Excerpts would count the same clicks twice; NOT_WORKING was the recordist's own
# verdict; the early tests predate the measurement protocol.
EXCLUDE_RULES = [
    (re.compile(r"_700s$|_forwebsite$"),                        "excerpt of another recording"),
    (re.compile(r"not_?working", re.I),                         "marked NOT_WORKING when recorded"),
    (re.compile(r"^test_|exemple|MARUMORE|^cactus_2026-01-00\d_mechanic", re.I),
                                                                "early test, before the protocol"),
]

# ── Environment ─────────────────────────────────────────────────────────────
# Outdoor is tested FIRST: 'outdoorteca' contains 'teca', which alone means indoor.
_OUT_NAME = re.compile(r"outdoor|esterno|balcon|taverna", re.I)
_IN_NAME = re.compile(r"indoor|indorteca|cameratom|camera_?ricy|camraricy|cameararicy|"
                      r"miacamera|teca|stanza", re.I)
_OUT_FOLDER = re.compile(r"balcone|cucina fuori|outdoor|taverna", re.I)
_IN_FOLDER = re.compile(r"stanza|camera", re.I)

# ── Condition, most specific first ──────────────────────────────────────────
_COND_NAME = [
    (r"solorumor|stanzavuota|solosuono|rumoristrada|esternocucina|^balcone|"
     r"ambienteesterno",                                      "noise"),
    (r"soloinsett",                                           "insects-only"),
    (r"soloteca",                                             "noise"),        # empty terrarium
    (r"noacqua|afternowater",                                 "no-water"),
    (r"raccolta",                                             "harvest+water"),
    (r"taglio",                                               "cut"),
    (r"staccato",                                             "detached"),
    (r"meccanic|mechanic|stimoloripetuto",                    "mechanical"),
    (r"insett|coccinell|cavallett",                           "insects"),
    (r"acqua|water",                                          "water"),
    (r"solopian|soloian|nostimoli|only_|tramonto",            "plant-only"),
]
_COND_FOLDER = [
    (r"^Noise/",                    "noise"),
    (r"^Bugs/",                     "insects-only"),
    (r"meccanico",                  "mechanical"),
    (r"staccato",                   "detached"),
    (r"insetti|bugs",               "insects"),
    (r"monitoraggio acqua|acqua",   "water"),
    (r"solo ?pianta",               "plant-only"),
]
_COND_NAME = [(re.compile(p, re.I), c) for p, c in _COND_NAME]
_COND_FOLDER = [(re.compile(p, re.I), c) for p, c in _COND_FOLDER]

# Names a human should look at even though a rule matched them.
_ALWAYS_REVIEW = [
    (re.compile(r"afternowater"), "'afternowater' read as no-water; confirm it is not 'after watering'"),
    (re.compile(r"STRANO"),       "name says 'solo pianta ma stimolato strano'"),
    (re.compile(r"soloteca"),     "empty terrarium, classed as noise"),
    (re.compile(r"_stimolo_outdoortaverna"), "stimulus not named"),
]


def _source_folder(source: str) -> str:
    s = source.replace("\\", "/")
    if ANCHOR in s:
        s = s.split(ANCHOR, 1)[1]
    return s.rsplit("/", 1)[0] if "/" in s else ""


def draft_row(stem: str, source: str) -> dict:
    folder = _source_folder(source)
    notes: list[str] = []

    if _OUT_NAME.search(stem):
        env, env_basis = "outdoor", "name"
    elif _IN_NAME.search(stem):
        env, env_basis = "indoor", "name"
    elif _OUT_FOLDER.search(folder):
        env, env_basis = "outdoor", "folder"
    elif _IN_FOLDER.search(folder):
        env, env_basis = "indoor", "folder"
    else:
        # The Jan-Apr 2026 plant sessions carry no location marker. They were
        # recorded indoors as far as the project records go, but that is exactly
        # the kind of assumption a table exists to make visible.
        env, env_basis = "indoor", "GUESS"

    cond, cond_basis = "", "GUESS"
    for rx, c in _COND_NAME:
        if rx.search(stem):
            cond, cond_basis = c, "name"
            break
    if not cond:
        for rx, c in _COND_FOLDER:
            if rx.search(folder):
                cond, cond_basis = c, "folder"
                break
    if not cond:
        cond = "?"

    include, reason = 1, ""
    for rx, why in EXCLUDE_RULES:
        if rx.search(stem):
            include, reason = 0, why
            break

    for rx, why in _ALWAYS_REVIEW:
        if rx.search(stem):
            notes.append(why)

    # A condition read off the FOLDER is flagged too: 8 of 30 folders mix
    # conditions, so the folder is weaker evidence than the name.
    needs_review = int(env_basis == "GUESS" or cond_basis in ("GUESS", "folder")
                       or cond == "?" or bool(notes))
    # An excluded recording's condition never reaches a statistic, so a guess there
    # is not worth a human's time.
    if include == 0:
        needs_review = 0

    return {
        "file": stem, "source_folder": folder,
        "environment": env, "environment_basis": env_basis,
        "condition": cond, "condition_basis": cond_basis,
        "include": include, "exclude_reason": reason,
        "needs_review": needs_review, "note": "; ".join(notes),
    }


def read_manifest(path: Path) -> dict[str, str]:
    """{stem: source}. Tolerates the stray header lines the pre-fix launcher wrote."""
    out: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row or row[:2] == ["file", "source"]:
                continue
            if len(row) >= 6 and row[5] == "ok":
                out[row[0]] = row[1]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Draft a recording -> condition table from an export manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("-o", "--output", required=True, type=Path)
    a = ap.parse_args()

    if not a.manifest.exists():
        print(f"ERROR: manifest not found: {a.manifest}")
        return 2
    recordings = read_manifest(a.manifest)
    if not recordings:
        print("ERROR: no successful exports in the manifest")
        return 2

    kept: list[dict] = []
    if a.output.exists():
        with open(a.output, newline="", encoding="utf-8-sig") as fh:
            kept = list(csv.DictReader(fh))
    known = {r["file"] for r in kept}

    new_rows = [draft_row(stem, src) for stem, src in sorted(recordings.items())
                if stem not in known]

    a.output.parent.mkdir(parents=True, exist_ok=True)
    with open(a.output, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in kept + new_rows:
            w.writerow(r)

    rows = kept + new_rows
    print(f"kept (untouched) : {len(kept)}")
    print(f"added as drafts  : {len(new_rows)}")
    print(f"total            : {len(rows)}")
    inc = [r for r in rows if str(r["include"]) == "1"]
    rev = [r for r in rows if str(r["needs_review"]) == "1"]
    print(f"included         : {len(inc)}   excluded: {len(rows) - len(inc)}")
    print(f"NEEDS REVIEW     : {len(rev)}")

    from collections import Counter
    grid = Counter((r["environment"], r["condition"]) for r in inc)
    print("\nincluded recordings by environment x condition:")
    for (env, cond), n in sorted(grid.items()):
        print(f"   {env:<8} {cond:<14} {n}")
    missing = set(recordings) - {r["file"] for r in rows}
    if missing:
        print(f"\n⚠️  {len(missing)} manifest recording(s) missing from the table")
    print(f"\nwrote: {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
