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
collect_training_set.py — gather labelled v6 events into one training CSV

    python3 src/ml/collect_training_set.py DATASET_DIR -o training_set.csv \
        --set-b Aloe_acqua50ml_misurazione1_11032026_09 test_aloe_1

Walks DATASET_DIR and every subfolder for `*_candidates.csv`, keeps the LABELLED
rows, and writes one CSV ordered:

    SET A   clicks         (label 1)
    SET A   ambiguous      (label 2)   --include-ambiguous
    SET A   noise          (label 0)
    SET B   clicks
    SET B   ambiguous                  --include-ambiguous
    SET B   noise
    stage-2 failures, A and B mixed    --include-stage2-failed

Set membership is written into a `set` column, so the split is defined once, in
the data. `train_svm.py --set-b-from-column` reads it back.

UNLABELLED ROWS ARE NEVER EMITTED, under any combination of flags. A blank label
is "not yet judged", which is not the same as "noise" — folding the two together
would silently inject every un-reviewed candidate into the negative class.

── THE THREE OPT-IN FLAGS, AND WHAT EACH ONE COSTS ──────────────────────────
--include-ambiguous       label=2 rows. Off by default because they are not a
                          class: `train_svm.py --ambiguous` decides what to do
                          with them, and it can only do that if they are here.
--include-stage2-failed   rows v6 Stage 2 would hard-reject. Off by default
                          because the deployed pipeline never shows them to the
                          SVM, so training on them models a population that does
                          not exist at inference. Turn on only to study the gate.
--include-partial         CSVs where some rows are still blank. Off by default
                          because a partially-labelled recording's label=0 rows
                          are NOT that session's noise population — they are the
                          rows that happened to get reviewed — so every "% of
                          noise" computed from them is inflated.

── WHY STAGE 2 IS RECOMPUTED AND NOT READ ───────────────────────────────────
The `stage_blocked` column in the v6 corpus holds *v5* verdicts (Stage2_R2 on
372 791 rows — the fit gate that v6 removed), and `stage2_mode` is empty on all
402 861 rows. Reading either would label rows by a rule that is no longer the
pipeline's. So `_stage2_reason` is called row-wise, in the mode named by
--stage2-mode, and the verdict is written to `stage2_pass` / `stage2_reason`.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

# `import core.click_pipeline_v5` executes src/core/__init__.py, which imports the
# Qt windows package and dies on a circular import outside the GUI. Load the one
# module by path instead.
_REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "click_pipeline_v5", _REPO / "src" / "core" / "click_pipeline_v5.py")
_cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cp)

STAGE2_MODE_V5 = _cp.STAGE2_MODE_V5
STAGE2_MODE_CONSERVATIVE = _cp.STAGE2_MODE_CONSERVATIVE
STAGE2_MODE_AGGRESSIVE = _cp.STAGE2_MODE_AGGRESSIVE
_ALL_MODES = (STAGE2_MODE_V5, STAGE2_MODE_CONSERVATIVE, STAGE2_MODE_AGGRESSIVE)

#: Columns Stage 2 reads, in either mode. Coerced to float before the gate runs
#: because `_stage2_reason`'s helpers do `float(v)` and treat a ValueError as
#: PASS — so handing it the raw string '12,73' from an Excel-edited cell would
#: silently pass a row that should fail.
_GATE_COLS = ('peak_SNR', 'n_seg', 'local_crest', 'harmonic_confinement',
              'SPR', 'fit_valid', 'R2', 'tau_ms')

#: Emitted before the source columns. `set` and `stage2_pass` are what the
#: consumers read; the other two are provenance, so a row in the training CSV can
#: always be traced back to the file and rule that produced it.
_ADDED = ('set', 'stage2_pass', 'stage2_reason', 'hard_negative', 'source_file')

#: Block order. Index 0-5 are the six set/class blocks; stage-2 failures are
#: appended after all of them as one block regardless of set.
_CLASS_ORDER = ('1', '2', '0')          # clicks, ambiguous, noise
_LABELS = {'0', '1', '2'}
_BLOCK_S2FAIL = 6


def _num(v):
    """CSV cell -> float, tolerating the decimal comma Excel writes under it-IT."""
    try:
        return float(str(v).replace(',', '.'))
    except (TypeError, ValueError):
        return math.nan


def _norm(name: str) -> str:
    """'x_candidates.csv', 'x.csv' and 'x' all name the same recording."""
    n = name.strip()
    for suffix in ('_candidates.csv', '.csv', '_candidates'):
        if n.endswith(suffix):
            n = n[: -len(suffix)]
            break
    return n


def discover(dataset_dir: Path):
    """Every readable *_candidates.csv under dataset_dir, as (path, stem).

    macOS AppleDouble forks ('._name.csv') are skipped: they are resource
    metadata, not data, and half the files in this corpus are those. A naive
    rglob reads them and dies on UnicodeDecodeError.
    """
    out = []
    for p in sorted(dataset_dir.rglob('*_candidates.csv')):
        if p.name.startswith('._'):
            continue
        out.append((p, p.stem.replace('_candidates', '')))
    return out


def read_set_b_file(path: Path) -> list[str]:
    """One recording name per line. Blank lines and '#' comments ignored."""
    names = []
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.split('#', 1)[0].strip()
        if line:
            names.append(line)
    return names


def resolve_set_b(names, files) -> set[str]:
    """Map user-supplied names onto discovered stems, or explain what went wrong.

    An unmatched name is a hard error, never a warning. Silently training on an
    empty Set B because a filename was mistyped produces a model that looks
    cross-validated and has no held-out test at all.
    """
    by_stem = {stem: stem for _, stem in files}
    lower = defaultdict(list)
    for _, stem in files:
        lower[stem.lower()].append(stem)

    resolved, unmatched = set(), []
    for raw in names:
        key = _norm(raw)
        if key in by_stem:
            resolved.add(key)
        elif len(lower.get(key.lower(), [])) == 1:
            resolved.add(lower[key.lower()][0])
        else:
            unmatched.append(raw)

    if unmatched:
        print('ERROR: these --set-b names matched no CSV in the dataset:',
              file=sys.stderr)
        for u in unmatched:
            print(f'    {u}', file=sys.stderr)
            near = [s for _, s in files if _norm(u).lower() in s.lower()
                    or s.lower() in _norm(u).lower()]
            for n in near[:3]:
                print(f'        did you mean: {n}', file=sys.stderr)
        print(f'\n  {len(files)} recordings are available; '
              f'run with --list to see them.', file=sys.stderr)
        sys.exit(2)
    return resolved


def score_hard_negatives(rows, model_path: Path):
    """
    Mark label-0 rows a trained v5 model calls CLICK.

    Returns {id(row_dict): 1} for the flagged rows, or {} if scoring is not
    possible. "Hard" is defined by the OLD model's mistakes: a noise event that
    survived v5's Stage 1 and Stage 2 and was then scored above threshold is a
    negative the previous generation could not separate, which makes it worth
    more to the next one than another easy row.

    ⚠️ Only rows that v5 would actually have SEEN are eligible. Scoring a row v5
    never reached says nothing about v5 being wrong on it.
    """
    try:
        import joblib
        import numpy as np
    except ImportError as exc:
        print(f"  ! --v5-model needs joblib/numpy ({exc}); no rows flagged",
              file=sys.stderr)
        return {}
    try:
        model = joblib.load(model_path)
    except Exception as exc:                                  # noqa: BLE001
        print(f"  ! could not load {model_path}: {exc}", file=sys.stderr)
        return {}

    feats, thr = model['features'], model['threshold']
    eligible = [r for r in rows
                if r['_label'] == '0'
                and _num(r['_raw'].get('would_pass_v5')) == 1
                and _cp._stage2_reason_v5(r['_gate']) == '']
    if not eligible:
        return {}
    X = np.array([[_num(r['_raw'].get(f)) for f in feats] for r in eligible],
                 dtype=float)
    ok = np.isfinite(X).all(axis=1)
    flagged = {}
    if ok.any():
        probs = model['pipeline'].predict_proba(X[ok])[:, 1]
        for r, pr in zip([e for e, k in zip(eligible, ok) if k], probs):
            if pr >= thr:
                flagged[id(r)] = 1
    print(f"\nHard negatives ({model_path.name}, threshold {thr:.3f}):")
    print(f"  label-0 rows v5 would have seen : {len(eligible)}")
    print(f"  of those, v5 called CLICK       : {len(flagged)}")
    if not ok.all():
        print(f"  skipped (non-finite features)   : {int((~ok).sum())}")
    return flagged


def main() -> int:
    ap = argparse.ArgumentParser(
        description='Collect labelled v6 events into one training CSV.',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('dataset_dir', type=Path,
                    help='Folder to search recursively for *_candidates.csv')
    ap.add_argument('-o', '--output', type=Path, default=Path('training_set.csv'),
                    help='Output CSV (default: training_set.csv)')
    ap.add_argument('--set-b', nargs='*', default=[], metavar='NAME',
                    help='Recordings to hold out as Set B. Accepts the bare stem '
                         'or the full *_candidates.csv filename.')
    ap.add_argument('--set-b-file', type=Path, default=None, metavar='PATH',
                    help='Text file of Set B names, one per line; # comments ok.')
    ap.add_argument('--include-ambiguous', action='store_true',
                    help='Include label=2 (ambiguous) rows. Default: off.')
    ap.add_argument('--include-stage2-failed', action='store_true',
                    help='Include rows v6 Stage 2 rejects. Default: off.')
    ap.add_argument('--include-partial', action='store_true',
                    help='Include CSVs that are not entirely labelled (still only '
                         'their labelled rows). Default: off.')
    ap.add_argument('--v5-model', type=Path, default=None, metavar='PATH',
                    help='Score label-0 rows with a trained v5 model and flag the '
                         'ones it calls CLICK as hard_negative=1. These are the '
                         'noise the previous generation got wrong, so they are the '
                         'negatives worth weighting. Off by default; needs joblib.')
    ap.add_argument('--stage2-mode', default=STAGE2_MODE_CONSERVATIVE,
                    choices=_ALL_MODES,
                    help=f'Stage 2 rule to evaluate (default: {STAGE2_MODE_CONSERVATIVE})')
    ap.add_argument('--list', action='store_true',
                    help='List discovered recordings with their label counts, then exit.')
    args = ap.parse_args()

    if not args.dataset_dir.is_dir():
        print(f'ERROR: not a directory: {args.dataset_dir}', file=sys.stderr)
        return 2

    files = discover(args.dataset_dir)
    if not files:
        print(f'ERROR: no *_candidates.csv found under {args.dataset_dir}',
              file=sys.stderr)
        return 2

    # ── pass 1: read, validate, count ───────────────────────────────────────
    header = None
    per_file = {}          # stem -> (path, rows, labelled_counts, is_partial)
    unreadable = []
    for path, stem in files:
        try:
            with open(path, encoding='utf-8-sig', newline='') as fh:
                rows = list(csv.DictReader(fh))
        except (OSError, UnicodeDecodeError) as exc:
            unreadable.append((stem, type(exc).__name__))
            continue
        if not rows:
            unreadable.append((stem, 'empty'))
            continue
        cols = list(rows[0].keys())
        if header is None:
            header = cols
        elif cols != header:
            print(f'ERROR: {path.name} has a different column set than the first '
                  f'CSV read. Mixing schemas would misalign the output.',
                  file=sys.stderr)
            missing = [c for c in header if c not in cols]
            extra = [c for c in cols if c not in header]
            if missing:
                print(f'    missing: {missing}', file=sys.stderr)
            if extra:
                print(f'    extra:   {extra}', file=sys.stderr)
            return 2
        counts = Counter((r.get('label') or '').strip() for r in rows)
        partial = counts.get('', 0) > 0
        per_file[stem] = (path, rows, counts, partial)

    if args.list:
        print(f'{len(per_file)} recordings under {args.dataset_dir}\n')
        print(f"  {'clicks':>7} {'ambig':>6} {'noise':>7} {'blank':>8}  recording")
        for stem in sorted(per_file):
            _, _, c, partial = per_file[stem]
            flag = '  (partial)' if partial else ''
            print(f"  {c.get('1',0):7d} {c.get('2',0):6d} {c.get('0',0):7d} "
                  f"{c.get('',0):8d}  {stem}{flag}")
        return 0

    set_b = resolve_set_b(list(args.set_b) +
                          (read_set_b_file(args.set_b_file) if args.set_b_file else []),
                          files)

    # ── pass 2: select rows ─────────────────────────────────────────────────
    keep_labels = _LABELS if args.include_ambiguous else {'0', '1'}
    blocks = defaultdict(list)
    skipped_partial, s2_dropped = [], Counter()
    tally = defaultdict(Counter)      # (set, block-kind) -> label counts

    for stem in sorted(per_file):
        path, rows, counts, partial = per_file[stem]
        if partial and not args.include_partial:
            skipped_partial.append((stem, counts))
            continue
        which = 'B' if stem in set_b else 'A'
        rel = str(path.relative_to(args.dataset_dir))
        for r in rows:
            lab = (r.get('label') or '').strip()
            if lab not in _LABELS:
                continue                      # blank or junk: NEVER emitted
            gate = dict(r)
            for c in _GATE_COLS:
                gate[c] = _num(r.get(c))
            reason = _cp._stage2_reason(gate, args.stage2_mode)
            passed = (reason == '')

            if not passed and not args.include_stage2_failed:
                s2_dropped[reason] += 1
                continue
            if lab not in keep_labels:
                continue

            block = _BLOCK_S2FAIL if not passed else (
                (0 if which == 'A' else 3) + _CLASS_ORDER.index(lab))
            out = {'set': which,
                   'stage2_pass': 1 if passed else 0,
                   'stage2_reason': reason,
                   'hard_negative': 0,
                   'source_file': rel}
            out.update(r)
            # Metadata for the optional hard-negative pass, stripped before write
            # (DictWriter uses extrasaction='ignore', so these never reach the CSV).
            out['_label'], out['_raw'], out['_gate'] = lab, r, gate
            blocks[block].append((stem, _num(r.get('frame_idx')), out))
            tally[(which, 'pass' if passed else 'fail')][lab] += 1

    # ── optional: flag the negatives the v5 model got wrong ─────────────────
    if args.v5_model is not None:
        every = [row for blk in blocks.values() for _, _, row in blk]
        flagged = score_hard_negatives(every, args.v5_model)
        for row in every:
            if id(row) in flagged:
                row['hard_negative'] = 1
        n_flagged = sum(1 for row in every if row['hard_negative'])
        in_survivors = sum(1 for row in every
                           if row['hard_negative'] and row['stage2_pass'] == 1)
        print(f"  inside the Stage 2 survivor set : {in_survivors}"
              f"  (the only ones training will see)")
        if n_flagged - in_survivors:
            print(f"  outside it (v6 blocks them)     : {n_flagged - in_survivors}")

    # ── write ───────────────────────────────────────────────────────────────
    out_cols = list(_ADDED) + header
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(args.output, 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=out_cols, extrasaction='ignore')
        w.writeheader()
        for block in sorted(blocks):
            # stable, reproducible order inside each block
            for _, _, row in sorted(blocks[block], key=lambda t: (t[0], t[1])):
                w.writerow(row)
                n_written += 1

    # ── report ──────────────────────────────────────────────────────────────
    print(f'\ndataset      {args.dataset_dir}')
    print(f'recordings   {len(per_file)} readable'
          + (f', {len(unreadable)} skipped' if unreadable else ''))
    for stem, why in unreadable:
        print(f'               ! {stem}  ({why})')
    print(f'Stage 2 rule {args.stage2_mode}')
    print(f'options      ambiguous={args.include_ambiguous}  '
          f'stage2_failed={args.include_stage2_failed}  '
          f'partial={args.include_partial}')

    if skipped_partial:
        n = sum(c.get('1', 0) + c.get('2', 0) + c.get('0', 0)
                for _, c in skipped_partial)
        print(f'\npartially-labelled recordings skipped: {len(skipped_partial)} '
              f'({n} labelled rows withheld) — --include-partial to keep them')
        for stem, c in skipped_partial[:10]:
            print(f"    {c.get('1',0):4d} clicks {c.get('2',0):4d} ambig "
                  f"{c.get('0',0):6d} noise {c.get('',0):7d} blank  {stem}")
        if len(skipped_partial) > 10:
            print(f'    ... and {len(skipped_partial) - 10} more')

    if s2_dropped:
        print(f'\nStage 2 rejects dropped: {sum(s2_dropped.values())}')
        for reason, n in s2_dropped.most_common():
            print(f'    {n:8d}  {reason}')

    print(f'\n{"":12s} {"clicks":>8} {"ambig":>8} {"noise":>8} {"total":>8}')
    for which in ('A', 'B'):
        c = tally[(which, 'pass')]
        print(f'  SET {which:8s} {c["1"]:8d} {c["2"]:8d} {c["0"]:8d} '
              f'{sum(c.values()):8d}')
    if args.include_stage2_failed:
        c = tally[('A', 'fail')] + tally[('B', 'fail')]
        print(f'  {"S2-failed":10s} {c["1"]:8d} {c["2"]:8d} {c["0"]:8d} '
              f'{sum(c.values()):8d}')
    print(f'  {"":10s} {"":8s} {"":8s} {"":8s} {"-"*8}')
    print(f'  {"written":10s} {"":8s} {"":8s} {"":8s} {n_written:8d}')

    if set_b:
        print(f'\nSet B: {len(set_b)} recording(s)')
        for s in sorted(set_b):
            print(f'    {s}')
    else:
        print('\n⚠ No Set B — the output has no held-out test set. '
              'Pass --set-b or --set-b-file.')
    if set_b and tally[('B', 'pass')]['1'] == 0:
        print('⚠ Set B contains NO clicks: recall on it is undefined.')
    if tally[('A', 'pass')]['1'] == 0:
        print('⚠ Set A contains NO clicks: nothing to train on.')

    print(f'\nwrote {args.output}  ({n_written} rows, {len(out_cols)} columns)')
    print(f'  train with:  python3 src/ml/train_svm.py --csv {args.output} '
          f'--set-b-from-column')
    return 0


if __name__ == '__main__':
    sys.exit(main())
