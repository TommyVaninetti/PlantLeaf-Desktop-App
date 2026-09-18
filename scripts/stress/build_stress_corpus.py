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
build_stress_corpus.py — the Stage 5 input table, built once from a v6 export

    python3 scripts/stress/build_stress_corpus.py \
        --export ~/PlantLeaf_dev/Analisi/v6/re-export \
        --conditions ~/PlantLeaf_dev/Analisi/v6/recording_conditions.csv \
        --training-set ~/PlantLeaf_dev/Analisi/v6/training_set_27082026_ambiguousincluded_v5evaluated.csv \
        -o ~/PlantLeaf_dev/Analisi/v6/stress

WHY A CACHE
-----------
The candidate CSVs hold ~2 M rows over 111 recordings, and Stage 5 needs three
columns of the ~0.2 % of them that survive Stages 2–4. Reading them is a minute;
every analysis downstream then runs in seconds and can be re-run freely.

WHAT AN "ACCEPTED EVENT" IS
---------------------------
`stage_blocked == ''` — survived Stage 2 gates, Stage 3 SVM and Stage 4 dedup, as
stamped by the export itself with the deployed model. Nothing here re-runs a
stage, so Stage 5 sees exactly the stream the exporter produced (§8.6: the
threshold is a property of THAT pipeline).

GROUPS ARE A DESIGN DECISION, WRITTEN DOWN ONCE
-----------------------------------------------
See GROUP_OF below. `include == 0` rows are `excluded` whatever their condition.
The condition itself comes only from recording_conditions.csv (§9.4) — never
from the filename here. A recording missing from that table is `unassigned` and
listed, not guessed.

DURATION
--------
From the .paudio via analyze_dataset.read_paudio_duration, using the exact path
the manifest recorded. The last candidate timestamp is kept as `t_last_candidate_s`
for a sanity check but is never used as a duration: a rate over a guessed
duration is not a rate.

OUTPUTS  (-o)
-------------
    sessions.csv   one row per recording
    events.csv     file, t, p                   accepted events only
    floor.csv      file, t, noise_floor_mV      all candidates, thinned to <= 4000
                                                per recording, for the §2.1 warm-up test
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / 'src' / 'ml'))
from analyze_dataset import read_paudio_duration   # noqa: E402

# ── The experimental design ───────────────────────────────────────────────────
#: (environment, condition) -> group. Anything outdoor is a probe (§12.4). The
#: indoor mapping was decided on 16-09-2026: water counts as a stimulus;
#: insects / detached / no-water are reported but kept out of the headline.
BASELINE, STIMULUS, PROBE, SECONDARY = 'baseline', 'stimulus', 'probe', 'secondary'
EXCLUDED, UNASSIGNED = 'excluded', 'unassigned'
ARTEFACT = 'artefact'

INDOOR_GROUP_OF = {
    'noise':        BASELINE,
    'plant-only':   BASELINE,
    'mechanical':   STIMULUS,
    'cut':          STIMULUS,
    'water':        STIMULUS,
    'insects-only': PROBE,       # insects, no plant: the §12.1 confound in isolation
    'insects':      SECONDARY,
    'detached':     SECONDARY,
    'no-water':     SECONDARY,
    # Decided 17-09-2026 from the lab records: the spatifillo_misurazioneN series
    # were monitoring days, with no watering during the recording. Not an acute
    # stimulus, so outside the headline comparison (spec §1.2).
    'water-monitoring': SECONDARY,
}

# ── Artefact rule: the ~66 kHz narrowband tone (spec §2.1) ────────────────────
#: DECLARED 17-09-2026, from the premises run and BEFORE any validation, so it is
#: an artefact criterion and not a result of tuning. Measured then:
#:
#:   accepted events   FPE median   FPE IQR   peak_SNR   SVM p
#:   pomodori_mis3/5   66-67 kHz    2 kHz     ~6         ~0.2     (baseline)
#:   stimulus          41-44 kHz    21-26 kHz 9-11       0.3-0.44
#:
#: 90 % of all indoor plant-only accepted events fell in 64-70 kHz, one session
#: held 72 % of the whole baseline, and the one stimulus "burst" longer than W
#: (spatifillo_misurazione5, 508 s) was 93 % tone. A fixed 2 kHz-wide line
#: repeating several times a second at low SNR is not a cavitation click. It
#: appeared only in plant sessions in camera Ricy (8-9 Apr, 4-8 May); the
#: empty-room recordings made there in the same weeks show none.
#:
#: A session is flagged when MORE THAN TONE_MIN_FRAC of its accepted events lie in
#: TONE_BAND_HZ and it has at least TONE_MIN_EVENTS of them — the same rule for
#: every group and environment. Flagged sessions get group = 'artefact' and are
#: kept out of calibration and validation. `--no-artefact-rule` turns it off, which
#: is the sensitivity analysis §9.3 asks for.
TONE_BAND_HZ = (64_000.0, 70_000.0)
TONE_MIN_FRAC = 0.5
TONE_MIN_EVENTS = 20

FLOOR_ROWS_PER_FILE = 4000
READ_COLS = ['timestamp_s', 'stage_blocked', 'svm_probability', 'noise_floor_mV', 'FPE_hz']


def group_of(env: str, condition: str, include: int) -> str:
    if include != 1:
        return EXCLUDED
    if env == 'outdoor':
        return PROBE
    return INDOOR_GROUP_OF.get(condition, SECONDARY)


def read_manifest(path: Path) -> dict[str, str]:
    """file -> .paudio path. The manifest is appended per run, so its header
    repeats mid-file and a file can appear twice; the last `ok` row wins."""
    out = {}
    with open(path, newline='', encoding='utf-8-sig') as f:
        for row in csv.reader(f):
            if len(row) < 6 or row[0] == 'file':
                continue
            if row[5] == 'ok':
                out[row[0]] = row[1]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--export', type=Path, required=True,
                    help='export root holding CSVs/ and export_manifest.csv')
    ap.add_argument('--conditions', type=Path, required=True)
    ap.add_argument('--training-set', type=Path, default=None,
                    help='SVM training CSV; marks recordings where Stage 3 is in-sample')
    ap.add_argument('-o', '--out', type=Path, required=True)
    ap.add_argument('--no-artefact-rule', action='store_true',
                    help='do not flag 66 kHz tone sessions (sensitivity analysis, §9.3)')
    a = ap.parse_args()

    manifest = read_manifest(a.export / 'export_manifest.csv')
    cond = pd.read_csv(a.conditions, dtype=str).fillna('')
    cond = cond.set_index('file')
    trained = set()
    if a.training_set:
        trained = set(pd.read_csv(a.training_set, usecols=['file'])['file'].astype(str))

    csvs = sorted(a.export.glob('CSVs/*/*_candidates.csv'))
    if not csvs:
        print(f"ERROR: no candidate CSVs under {a.export / 'CSVs'}", file=sys.stderr)
        return 2
    a.out.mkdir(parents=True, exist_ok=True)

    sessions, events, floors, problems = [], [], [], []
    for i, path in enumerate(csvs, 1):
        file = path.parent.name
        df = pd.read_csv(path, usecols=READ_COLS, dtype={'stage_blocked': str},
                         low_memory=False)
        ok = df[df['stage_blocked'].fillna('') == '']
        t = pd.to_numeric(ok['timestamp_s'], errors='coerce')
        p = pd.to_numeric(ok['svm_probability'], errors='coerce')
        if t.isna().any() or p.isna().any():
            problems.append(f"{file}: {int(t.isna().sum())} NaN t, {int(p.isna().sum())} NaN p "
                            f"among accepted rows (dropped)")
        keep = t.notna() & p.notna()
        fpe = pd.to_numeric(ok['FPE_hz'], errors='coerce')[keep]
        tone_n = int(((fpe >= TONE_BAND_HZ[0]) & (fpe <= TONE_BAND_HZ[1])).sum())
        tone_frac = tone_n / int(keep.sum()) if int(keep.sum()) else float('nan')
        is_tone = int(keep.sum()) >= TONE_MIN_EVENTS and tone_frac > TONE_MIN_FRAC
        events.append(pd.DataFrame({'file': file, 't': t[keep].to_numpy(),
                                    'p': p[keep].to_numpy()}))

        fl = df[['timestamp_s', 'noise_floor_mV']].dropna()
        if len(fl) > FLOOR_ROWS_PER_FILE:
            fl = fl.iloc[np.linspace(0, len(fl) - 1, FLOOR_ROWS_PER_FILE).astype(int)]
        floors.append(pd.DataFrame({'file': file, 't': fl['timestamp_s'].to_numpy(),
                                    'noise_floor_mV': fl['noise_floor_mV'].to_numpy()}))

        src = manifest.get(file, '')
        dur = read_paudio_duration(Path(src)) if src else None
        if dur is None:
            problems.append(f"{file}: no duration (manifest source {src!r})")

        if file in cond.index:
            c = cond.loc[file]
            env, condition = c['environment'], c['condition']
            include = int(c['include'] or 0)
            grp = group_of(env, condition, include)
            row = dict(environment_basis=c['environment_basis'],
                       condition_basis=c['condition_basis'], include=include,
                       needs_review=int(c['needs_review'] or 0),
                       exclude_reason=c['exclude_reason'], note=c['note'])
            if is_tone and grp != EXCLUDED and not a.no_artefact_rule:
                grp = ARTEFACT
                row['exclude_reason'] = (f"artefact: {tone_frac:.0%} of accepted events "
                                         f"at 64-70 kHz (declared 17-09-2026)")
        else:
            env, condition, grp = '', '', UNASSIGNED
            row = dict(environment_basis='', condition_basis='', include=0,
                       needs_review=1, exclude_reason='', note='')
            problems.append(f"{file}: not in {a.conditions.name}")

        sessions.append(dict(
            file=file, environment=env, condition=condition, group=grp,
            duration_s=np.nan if dur is None else round(dur, 3),
            n_candidates=len(df), n_accepted=int(keep.sum()),
            t_last_candidate_s=float(pd.to_numeric(df['timestamp_s'], errors='coerce').max()),
            stage3_in_sample=int(file in trained),
            tone66_n=tone_n, tone66_frac=round(tone_frac, 4) if tone_frac == tone_frac else np.nan,
            paudio=src, **row))
        print(f"\r  [{i}/{len(csvs)}] {file[:60]:60s}", end='', flush=True)
    print()

    ses = pd.DataFrame(sessions)
    ev = pd.concat(events, ignore_index=True).sort_values(['file', 't'], kind='stable')
    fl = pd.concat(floors, ignore_index=True)
    ses.to_csv(a.out / 'sessions.csv', index=False)
    ev.to_csv(a.out / 'events.csv', index=False)
    fl.to_csv(a.out / 'floor.csv', index=False)

    # Sanity: every event lies inside its recording.
    late = ev.merge(ses[['file', 'duration_s']], on='file')
    late = late[late['t'] > late['duration_s'] + 1.0]
    if len(late):
        problems.append(f"{len(late)} accepted events after the .paudio duration in "
                        f"{sorted(late['file'].unique())}")
    missing_train = sorted(trained - set(ses['file']))

    print(f"\n{len(ses)} recordings, {len(ev)} accepted events -> {a.out}")
    tab = ses.groupby(['group', 'environment', 'condition']).agg(
        n=('file', 'size'), hours=('duration_s', lambda s: s.sum() / 3600),
        events=('n_accepted', 'sum'), needs_review=('needs_review', 'sum'))
    print(tab.round(2).to_string())
    if missing_train:
        print(f"\n⚠️  {len(missing_train)} Stage 3 training recordings are NOT in this export "
              f"(export incomplete?):")
        for f in missing_train:
            print(f"     {f}")
    if problems:
        print("\n⚠️  problems:")
        for s in problems:
            print(f"     {s}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
