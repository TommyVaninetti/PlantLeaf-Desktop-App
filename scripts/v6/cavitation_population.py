"""
cavitation_population.py — the labelled clicks against the two cavitation models
================================================================================

For every labelled candidate in the v6 training set this script re-extracts the
click from its .paudio, runs chemical_simulators.click_model_comparison (the same
engine as the Chemical Simulator window) and writes:

    <out>/population_rows.csv     one row per candidate: identity, label, species,
                                  session type, the dataset's own v6 values, every
                                  model output, and the pre-registered categories
    <out>/summary.md              criteria (fixed BEFORE looking at results), counts,
                                  medians and the fraction compatible per group,
                                  clicks vs the noise null, and the verdict
    <out>/summary.csv             the same numbers, machine-readable
    <out>/fig_*.png               tau–f map, R–L map, bubble tau ratio, R², impulse factor

Models (see docs/physical_simulation/CHANGELOG_SIMULATOR.md, Steps B–B.2):
  vessel  Dutta et al. 2022 — xylem vessel as an organ pipe: R from tau, L from f
  bubble  free air bubble (Minnaert + radiation/viscous/thermal damping): tau PREDICTED from f

Usage (from the repository root):

    .venv/bin/python scripts/v6/cavitation_population.py \\
        --paudio-root "/Volumes/<drive>/PlantLeaf" \\
        --out ~/PlantLeaf_dev/Analisi/v6/cavitation_population

    options:
      --dataset PATH      labelled CSV (default: the 27-08-2026 training set)
      --paudio-root DIR   searched recursively for <file>.paudio; repeatable
      --labels 1,0,2      labels to analyse (1 click, 0 noise = null, 2 ambiguous)
      --anatomy JSON      vessel radius / element length ranges per species (optional)
      --workers N         parallel processes (default: CPU count - 1)
      --limit N           analyse only the first N rows (smoke test)

Anatomy JSON (radii in µm, lengths in mm; ranges as found in the literature):

    {"aloe":   {"R_um": [8, 25], "L_mm": [0.2, 1.5], "source": "..."},
     "cactus": {"R_um": [10, 20], "L_mm": [0.1, 0.5], "source": "..."}}

Without it the vessel model can only be judged on waveform, and the summary says so.

Only continuous (v3) recordings are supported: every labelled session in the
training set is one. Event recordings (v4) are skipped with a note.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in (SRC, SRC / "chemical_simulators"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DEFAULT_DATASET = Path.home() / "PlantLeaf_dev/Analisi/v6/training_set_27082026_ambiguousincluded_v5evaluated.csv"

# ─────────────────────────────────────────────────────────────────────────────
# PRE-REGISTERED CRITERIA — fixed before the population results are looked at.
# Changing them after seeing the numbers must be recorded in CHANGELOG_SIMULATOR.md.
# ─────────────────────────────────────────────────────────────────────────────
CRITERIA = {
    # A candidate is analysable when the v6 decay fit is valid and tau is above the
    # chain's resolution (measured in Step B: below ~0.06 ms tau is not resolved).
    'tau_min_ms': 0.06,
    # Waveform: the model (free amplitude and phase, v6 click region) explains at
    # least half of the click's variance.
    'r2_min': 0.5,
    # Free bubble: the chain-calibrated tau of the click is within a factor 1.5 of the
    # tau the bubble MUST have at that frequency (thermal damping included).
    'bubble_tau_ratio_range': (1 / 1.5, 1.5),
    # Vessel: R within the species' radius range; L within the length range widened
    # by the ±30 % the authors report for their L estimate.
    'vessel_L_tolerance': 0.30,
    # Impulse-dominated: the envelope peak is at least twice what the ring-down model
    # accounts for. Reported, not used to reject.
    'impulse_factor_high': 2.0,
    # Verdict on the clicks (label 1) that are analysable.
    'verdict_yes_min_fraction': 0.70,
    'verdict_no_max_fraction': 0.20,
    # ...and the test only discriminates if the noise null is clearly lower.
    'null_max_ratio': 0.5,
}

LABEL_NAMES = {'1': 'click', '0': 'noise', '2': 'ambiguous'}


def species_of(stem: str) -> str:
    s = stem.lower()
    if 'aloe' in s:
        return 'aloe'
    if 'cactus' in s:
        return 'cactus'
    if 'calancola' in s or 'kalanchoe' in s:
        return 'kalanchoe'
    if 'alocasia' in s:
        return 'alocasia'
    if 'stanzavuota' in s or 'solorumore' in s:
        return 'no_plant'
    return 'other'


def session_type_of(stem: str) -> str:
    s = stem.lower()
    if 'stanzavuota' in s or 'solorumore' in s:
        return 'empty_room'
    if 'meccanico' in s:
        return 'mechanical'
    if 'acqua' in s or 'water' in s:
        return 'watering'
    if 'solopianta' in s or 'soloianta' in s or 'nostimoli' in s:
        return 'plant_only'
    return 'other'


# ─────────────────────────────────────────────────────────────────────────────
# Worker (one process per core; the engine is Qt-free)
# ─────────────────────────────────────────────────────────────────────────────

_HEADERS: dict = {}


def _analyse_row(task):
    """task = (row_index, paudio_path, frame_idx, noise_floor_V, std_noise_V)."""
    import click_model_comparison as cmc
    from hybrid import frame_emulator as fe

    idx, path, fi, nf, sd = task
    try:
        header = _HEADERS.get(path) or fe.read_header(path)
        _HEADERS[path] = header
        total = header['total_frames']
        start = max(0, fi - 1)
        mags, phases = fe.read_frames(path, start, min(3, total - start), header=header)

        def frame(i):
            k = i - start
            return (mags[k], phases[k]) if 0 <= k < len(mags) else None

        frames = (frame(fi - 1), frame(fi), frame(fi + 1))
        if frames[1] is None:
            return idx, None, f'frame {fi} not in file ({total} frames)'
        res = cmc.analyse_click(frames, fi, nf, sd)
        row = cmc.result_to_row(res)
        v = res.get('vessel') or {}
        b = res.get('bubble') or {}
        row['_real_onset'] = (res.get('real') or {}).get('onset')
        row['_vessel_below_resolution'] = int(bool(v.get('below_resolution')))
        return idx, row, ''
    except Exception as e:  # noqa: BLE001 — one bad row must not stop the run
        return idx, None, f'{type(e).__name__}: {e}'


# ─────────────────────────────────────────────────────────────────────────────
# Categories
# ─────────────────────────────────────────────────────────────────────────────

def _f(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else float('nan')
    except (TypeError, ValueError):
        return float('nan')


def categorise(row, anatomy):
    """Adds the pre-registered category columns to one output row."""
    c = CRITERIA
    tau = _f(row.get('real_tau_ms'))
    analysable = (str(row.get('real_fit_valid')) == '1' and tau >= c['tau_min_ms']
                  and row.get('vessel_ok') == 1 and row.get('bubble_ok') == 1)
    row['analysable'] = int(analysable)

    r2v, r2b = _f(row.get('vessel_r2_wave')), _f(row.get('bubble_r2_wave'))
    ratio = _f(row.get('bubble_tau_ratio_true_over_pred'))
    lo, hi = c['bubble_tau_ratio_range']

    row['bubble_tau_consistent'] = int(lo <= ratio <= hi) if np.isfinite(ratio) else ''
    row['bubble_compatible'] = int(analysable and lo <= ratio <= hi and r2b >= c['r2_min']) \
        if analysable else ''

    geo = anatomy.get(row['species'])
    row['vessel_waveform_ok'] = int(r2v >= c['r2_min']) if analysable else ''
    if analysable and geo:
        R, L = _f(row.get('vessel_R_um')), _f(row.get('vessel_L_mm'))
        rlo, rhi = geo['R_um']
        llo, lhi = geo['L_mm']
        tol = c['vessel_L_tolerance']
        geo_ok = rlo <= R <= rhi and llo * (1 - tol) <= L <= lhi * (1 + tol)
        row['vessel_geometry_ok'] = int(geo_ok)
        row['vessel_compatible'] = int(geo_ok and r2v >= c['r2_min'])
    else:
        row['vessel_geometry_ok'] = ''
        # geometry unknown: only the waveform can be judged
        row['vessel_compatible'] = int(r2v >= c['r2_min']) if analysable else ''
    row['vessel_geometry_checked'] = int(bool(geo))

    if analysable:
        vc, bc = row['vessel_compatible'] == 1, row['bubble_compatible'] == 1
        row['model_category'] = ('both' if vc and bc else 'vessel_only' if vc
                                 else 'bubble_only' if bc else 'neither')
        imp = max(_f(row.get('vessel_impulse_factor')), _f(row.get('bubble_impulse_factor')))
        row['impulse_dominated'] = int(imp >= c['impulse_factor_high'])
    else:
        row['model_category'] = 'not_analysable'
        row['impulse_dominated'] = ''
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def _median_iqr(values):
    v = np.array([x for x in map(_f, values) if np.isfinite(x)])
    if len(v) == 0:
        return 'n/a'
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    return f'{q2:.3g} [{q1:.3g}–{q3:.3g}]'


def _frac(rows, key):
    vals = [r[key] for r in rows if r.get(key) in (0, 1)]
    return (sum(vals) / len(vals), len(vals)) if vals else (float('nan'), 0)


def summarise(rows, anatomy, out_dir, meta):
    groups = defaultdict(list)
    for r in rows:
        groups[('all', r['label_name'], 'all')].append(r)
        groups[(r['species'], r['label_name'], 'all')].append(r)
        groups[('all', r['label_name'], r['session_type'])].append(r)
        groups[(r['species'], r['label_name'], r['session_type'])].append(r)

    metrics = [
        ('f (Hz, click region)', 'real_f_hz'), ('tau calibrated (ms)', 'vessel_tau_true_ms'),
        ('Q', 'vessel_Q'), ('vessel R (µm)', 'vessel_R_um'), ('vessel L (mm)', 'vessel_L_mm'),
        ('bubble R0 (µm)', 'bubble_R0_um'), ('tau / tau bubble', 'bubble_tau_ratio_true_over_pred'),
        ('R² vessel', 'vessel_r2_wave'), ('R² bubble', 'bubble_r2_wave'),
        ('impulse factor (vessel)', 'vessel_impulse_factor'),
    ]
    fracs = [('vessel compatible', 'vessel_compatible'), ('bubble compatible', 'bubble_compatible'),
             ('bubble tau consistent', 'bubble_tau_consistent'), ('impulse dominated', 'impulse_dominated')]

    table = []
    for (sp, lab, st), g in sorted(groups.items()):
        ana = [r for r in g if r['analysable'] == 1]
        entry = {'species': sp, 'label': lab, 'session_type': st, 'n': len(g), 'n_analysable': len(ana)}
        for name, key in metrics:
            entry[name] = _median_iqr(r.get(key) for r in ana)
        for name, key in fracs:
            f, n = _frac(ana, key)
            entry[name] = f'{f:.2f} (n={n})' if n else 'n/a'
        any_comp = [int(r['vessel_compatible'] == 1 or r['bubble_compatible'] == 1) for r in ana]
        entry['compatible with ≥1 model'] = f'{np.mean(any_comp):.2f} (n={len(any_comp)})' if any_comp else 'n/a'
        entry['_any_frac'] = float(np.mean(any_comp)) if any_comp else float('nan')
        entry['categories'] = dict(Counter(r['model_category'] for r in g))
        table.append(entry)

    with open(out_dir / 'summary.csv', 'w', newline='') as fh:
        cols = [k for k in table[0] if not k.startswith('_')] if table else []
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for e in table:
            w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v)
                        for k, v in e.items() if not k.startswith('_')})

    def get(sp, lab, st='all'):
        for e in table:
            if (e['species'], e['label'], e['session_type']) == (sp, lab, st):
                return e
        return None

    clicks, noise = get('all', 'click'), get('all', 'noise')
    c = CRITERIA
    verdict = 'not computable (no analysable clicks)'
    if clicks and np.isfinite(clicks['_any_frac']):
        p_click = clicks['_any_frac']
        p_null = noise['_any_frac'] if noise and np.isfinite(noise['_any_frac']) else float('nan')
        discriminates = np.isfinite(p_null) and p_null <= c['null_max_ratio'] * p_click
        if p_click >= c['verdict_yes_min_fraction'] and discriminates:
            verdict = 'YES — the clicks are consistent with the cavitation models'
        elif p_click <= c['verdict_no_max_fraction'] or not discriminates:
            verdict = ('NO — the clicks are not consistent with the cavitation models'
                       if p_click <= c['verdict_no_max_fraction'] else
                       'NOT DISCRIMINATING — noise candidates are about as compatible as clicks')
        else:
            verdict = 'MIX — only part of the clicks is consistent (see the per-group table)'
        verdict += f'  [clicks compatible: {p_click:.2f}; noise null: {p_null:.2f}]'

    lines = [
        '# Cavitation models vs labelled clicks — population summary', '',
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} by scripts/v6/cavitation_population.py", '',
        f"Dataset: `{meta['dataset']}`  ", f"Rows analysed: {meta['n_done']} of {meta['n_rows']} "
        f"(missing .paudio: {meta['n_missing']}, errors: {meta['n_errors']})", '',
        '## Pre-registered criteria', '', '```', json.dumps(CRITERIA, indent=2), '```', '',
        'Vessel geometry checked for: ' + (', '.join(sorted(anatomy)) if anatomy else
                                            '**none — no anatomy file given; the vessel model is judged on waveform only**'),
        '', '## Verdict', '', f'**{verdict}**', '',
        '## Per group (analysable rows; medians [IQR])', '',
    ]
    keys = ['species', 'label', 'session_type', 'n', 'n_analysable', 'compatible with ≥1 model',
            'vessel compatible', 'bubble compatible', 'bubble tau consistent', 'impulse dominated'] \
        + [m[0] for m in metrics]
    lines.append('| ' + ' | '.join(keys) + ' |')
    lines.append('|' + '---|' * len(keys))
    for e in table:
        lines.append('| ' + ' | '.join(str(e[k]) for k in keys) + ' |')
    lines += ['', '## Categories per group', '']
    for e in table:
        lines.append(f"- {e['species']} / {e['label']} / {e['session_type']}: {e['categories']}")
    (out_dir / 'summary.md').write_text('\n'.join(lines))
    return verdict


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def figures(rows, anatomy, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import click_model_comparison as cmc
    import rayleigh_plesset as rp
    import vessel_resonance as vr

    ana = [r for r in rows if r['analysable'] == 1]
    style = {'click': dict(c='#e53935', s=18, label='clicks'),
             'ambiguous': dict(c='#fb8c00', s=12, label='ambiguous'),
             'noise': dict(c='#9e9e9e', s=8, label='noise (null)')}

    # tau–f map
    fig, ax = plt.subplots(figsize=(8, 5.5))
    fgrid = np.linspace(18_000, 90_000, 80)
    ax.plot(fgrid, [rp.bubble_linear_properties(cmc._bubble_R0_for_frequency(f))['tau'] * 1e3 for f in fgrid],
            color='#2196F3', lw=2, label='free bubble τ(f)')
    lo, hi = CRITERIA['bubble_tau_ratio_range']
    band = np.array([rp.bubble_linear_properties(cmc._bubble_R0_for_frequency(f))['tau'] * 1e3 for f in fgrid])
    ax.fill_between(fgrid, band * lo, band * hi, color='#2196F3', alpha=0.12, label='bubble-consistent band')
    for R in (10, 20, 30, 40, 50):
        t = vr.settling_time(R * 1e-6) * 1e3
        ax.axhline(t, color='#8d6e63', ls='--', lw=0.8)
        ax.text(18_500, t * 1.03, f'vessel R={R} µm', color='#6d4c41', fontsize=7)
    for lab in ('noise', 'ambiguous', 'click'):
        pts = [(_f(r['vessel_f_true_hz']), _f(r['vessel_tau_true_ms'])) for r in ana if r['label_name'] == lab]
        if pts:
            ax.scatter(*zip(*pts), alpha=0.7, **style[lab])
    ax.set_yscale('log')
    ax.set_xlabel('frequency (Hz, chain-calibrated)')
    ax.set_ylabel('τ (ms, chain-calibrated)')
    ax.set_title('Clicks vs cavitation models')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / 'fig_tau_f_map.png', dpi=130)
    plt.close(fig)

    # R–L map per species
    fig, ax = plt.subplots(figsize=(7, 5))
    for sp, col in (('aloe', '#43a047'), ('cactus', '#8e24aa'), ('kalanchoe', '#fb8c00'), ('other', '#757575')):
        pts = [(_f(r['vessel_R_um']), _f(r['vessel_L_mm'])) for r in ana
               if r['label_name'] == 'click' and r['species'] == sp]
        if pts:
            ax.scatter(*zip(*pts), s=18, color=col, alpha=0.7, label=f'{sp} clicks')
        if sp in anatomy:
            (r0, r1), (l0, l1) = anatomy[sp]['R_um'], anatomy[sp]['L_mm']
            ax.add_patch(plt.Rectangle((r0, l0), r1 - r0, l1 - l0, fill=False, ec=col, lw=1.5, ls='--'))
    ax.set_xlabel('vessel radius R (µm) — from τ')
    ax.set_ylabel('element length L (mm) — from f')
    ax.set_title('Vessel geometry implied by each click (dashed: anatomy, if given)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / 'fig_vessel_R_L.png', dpi=130)
    plt.close(fig)

    # distributions
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    bins_ratio = np.logspace(-1, 1.3, 30)
    for lab in ('noise', 'click'):
        sel = [r for r in ana if r['label_name'] == lab]
        kw = dict(alpha=0.55, color=style[lab]['c'], label=style[lab]['label'], density=True)
        vals = [_f(r['bubble_tau_ratio_true_over_pred']) for r in sel]
        axes[0].hist([v for v in vals if np.isfinite(v) and v > 0], bins=bins_ratio, **kw)
        axes[1].hist([_f(r['vessel_r2_wave']) for r in sel if np.isfinite(_f(r['vessel_r2_wave']))],
                     bins=np.linspace(-0.5, 1, 31), **kw)
        axes[2].hist([_f(r['vessel_impulse_factor']) for r in sel if np.isfinite(_f(r['vessel_impulse_factor']))],
                     bins=np.linspace(0.5, 8, 31), **kw)
    axes[0].set_xscale('log')
    axes[0].axvspan(*CRITERIA['bubble_tau_ratio_range'], color='#2196F3', alpha=0.12)
    axes[0].set_xlabel('τ calibrated / τ free bubble')
    axes[1].axvline(CRITERIA['r2_min'], color='k', ls='--', lw=1)
    axes[1].set_xlabel('waveform R² — vessel model (v6 region)')
    axes[2].axvline(CRITERIA['impulse_factor_high'], color='k', ls='--', lw=1)
    axes[2].set_xlabel('impulse factor')
    for a in axes:
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / 'fig_distributions.png', dpi=130)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    ap.add_argument('--paudio-root', type=Path, action='append', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--labels', default='1,0,2')
    ap.add_argument('--anatomy', type=Path)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    anatomy = json.loads(args.anatomy.read_text()) if args.anatomy else {}
    wanted = set(args.labels.split(','))

    with open(args.dataset, newline='') as fh:
        rows = [r for r in csv.DictReader(fh) if r['label'] in wanted]
    if args.limit:
        rows = rows[:args.limit]

    index = {}
    for root in args.paudio_root:
        for p in Path(root).expanduser().rglob('*.paudio'):
            index.setdefault(p.stem, p)

    tasks, missing = [], Counter()
    for i, r in enumerate(rows):
        path = index.get(r['file'])
        if path is None:
            missing[r['file']] += 1
            continue
        tasks.append((i, str(path), int(r['frame_idx']),
                      float(r['noise_floor_mV']) / 1e3, float(r['std_noise_mV']) / 1e3))

    print(f'{len(rows)} labelled rows, {len(tasks)} with a .paudio, '
          f'{sum(missing.values())} missing ({len(missing)} sessions)')
    for stem, n in missing.most_common():
        print(f'   missing: {stem} ({n} rows)')

    results, errors = {}, {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, (i, row, err) in enumerate(pool.map(_analyse_row, tasks, chunksize=4), 1):
            if row is None:
                errors[i] = err
            else:
                results[i] = row
            if k % 50 == 0 or k == len(tasks):
                print(f'   {k}/{len(tasks)}  ({time.time() - t0:.0f}s)', flush=True)

    out_rows = []
    keep_cols = ['session_id', 'file', 'frame_idx', 'peak_abs', 'timestamp_s', 'label', 'set',
                 'tau_ms', 'R2', 'fit_valid', 'FPE_hz', 'FPE_hz_region', 'peak_SNR', 'svm_probability']
    for i, r in enumerate(rows):
        base = {f'ds_{k}': r.get(k, '') for k in keep_cols}
        base.update({'label_name': LABEL_NAMES.get(r['label'], r['label']),
                     'species': species_of(r['file']), 'session_type': session_type_of(r['file'])})
        if i in results:
            base.update({k: v for k, v in results[i].items() if k not in ('file', 'frame_idx', 'timestamp_s')})
            tau_ds, tau_new = _f(r.get('tau_ms')), _f(results[i].get('real_tau_ms'))
            base['tau_matches_dataset'] = int(abs(tau_new - tau_ds) <= 0.02 * max(abs(tau_ds), 1e-9)) \
                if tau_ds > 0 and tau_new > 0 else ''
            base['error'] = ''
        else:
            base['error'] = errors.get(i, 'paudio not found')
        out_rows.append(categorise(base, anatomy) if i in results else
                        {**base, 'analysable': 0, 'model_category': 'not_analysable'})

    cols = []
    for r in out_rows:
        for k in r:
            if k not in cols and not k.startswith('_'):
                cols.append(k)
    with open(args.out / 'population_rows.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(out_rows)

    done = [r for r in out_rows if not r.get('error')]
    meta = {'dataset': str(args.dataset), 'n_rows': len(rows), 'n_done': len(done),
            'n_missing': sum(missing.values()), 'n_errors': len(errors)}
    verdict = summarise(done, anatomy, args.out, meta) if done else 'no rows analysed'
    if done:
        figures(done, anatomy, args.out)

    mism = [r for r in done if r.get('tau_matches_dataset') == 0]
    print(f'\nre-measured tau differs from the dataset (>2 %) on {len(mism)} of '
          f'{sum(1 for r in done if r.get("tau_matches_dataset") in (0, 1))} rows')
    if errors:
        print(f'{len(errors)} errors, e.g.: {next(iter(errors.values()))}')
    print(f'\nVERDICT: {verdict}\nOutputs in {args.out}')


if __name__ == '__main__':
    main()
