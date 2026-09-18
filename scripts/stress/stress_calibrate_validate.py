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
stress_calibrate_validate.py — calibrate h_high and validate Stage 5 (spec §8–§9)

    python3 scripts/stress/stress_calibrate_validate.py \
        --corpus ~/PlantLeaf_dev/Analisi/v6/stress --W 180 --R 300 --phi 0.1

W, R AND phi ARE INPUTS, NOT OUTPUTS
------------------------------------
W and R come from stress_premises.py (burst length, §4.2 / §6); phi_target is a
declared design choice (§8.1). This script never picks any of them. The sweeps
exist to show how much the result depends on them — around the declared point.

WHAT "CALIBRATE" MEANS HERE
---------------------------
    h_high(phi) = min { h : A(h) / T_base <= phi }                          (§8.1)

A(h) = total stress_onset events that core/stress_detector.py — the class that
ships, hysteresis and refractory included — emits over every baseline session.
Each session's onset count A_s(h) is computed once per configuration; a LOSO fold
is then Σ A_s(h) − A_heldout(h), so validation re-runs no detector.

THE UNIT IS THE SESSION  (§9.1)
-------------------------------
A session is predicted "stressed" iff the detector emits >= 1 onset in it. The
group comparison is Mann–Whitney U on per-session max Score. No per-event p-value
is computed anywhere in this script.

LOSO  (§9.2)
------------
For every headline session s: calibrate on all baseline sessions except s, run on
s. For a stimulus session the calibration set is the full baseline, so h equals
the full-corpus h — asserted, not assumed. The spread of h across baseline folds
is reported: a wide spread means one session dominates (§8.5).

OUTPUTS  (--out, default <corpus>/validation_W<W>_R<R>)
    report.md             everything below, in spec order
    calibration_curve.csv config, h, A, T_base_h, fa_h, fa_ub95, sens_insample, fpr_insample
    session_scores.csv    per headline session: max score (hard, soft), onsets at h
    loso_folds.csv        per fold: held-out session, h, onsets, prediction
    sweeps.csv            one row per (config, subset, mode, phi)
    others.csv            secondary + probe sessions at the full-baseline h
    fa_curve.png, session_max.png, timelines_*.png, sweep_W.png
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stress_common import (   # noqa: E402
    BASELINE, GRID, GROUP_COLOR, INK, INK_2, PROBE, SECONDARY, STIMULUS,
    fa_upper_bound, load_corpus, md_table, style_axes, times_by_file,
)
from stress_detector import (   # noqa: E402
    MODE_HARD, MODE_SOFT, StressDetectorParams, replay_session, score_trace,
)

SOFT_STEP = 0.05
W_SWEEP = (60, 120, 180, 300, 600, 900)
RHO_SWEEP = (0.3, 0.5, 0.7)
DEAD_SWEEP = (0, 60, 300)


@dataclass(frozen=True, eq=False)   # identity equality: holds numpy arrays
class Session:
    file: str
    group: str
    environment: str
    condition: str
    T: float
    t: np.ndarray
    p: np.ndarray
    needs_review: int
    in_sample: int


@dataclass(frozen=True)
class Config:
    W: float
    R: float
    rho: float
    mode: str
    dead_s: float = 0.0

    def label(self) -> str:
        return f"W={self.W:g} R={self.R:g} rho={self.rho:g} {self.mode} D={self.dead_s:g}"

    def params(self, h: float) -> StressDetectorParams:
        return StressDetectorParams(W=self.W, h_high=h, R=self.R, rho=self.rho, mode=self.mode)


def trimmed(s: Session, dead_s: float) -> Session | None:
    """Drop the first dead_s seconds (§2.1 handling dead time)."""
    if dead_s <= 0:
        return s
    if s.T <= dead_s:
        return None
    keep = s.t >= dead_s
    return replace(s, T=s.T - dead_s, t=s.t[keep], p=s.p[keep])


def max_score(s: Session, W: float, mode: str) -> float:
    if s.t.size == 0:
        return 0.0
    tr = score_trace(s.t, W, None if mode == MODE_HARD else s.p)
    return max(v for _, v in tr)


def h_grid(max_seen: float, mode: str) -> np.ndarray:
    if mode == MODE_HARD:
        return np.arange(1, int(math.floor(max_seen)) + 2, dtype=float)
    return np.round(np.arange(SOFT_STEP, max_seen + 2 * SOFT_STEP, SOFT_STEP), 10)


def onset_counts(s: Session, cfg: Config, grid: np.ndarray, smax: float) -> np.ndarray:
    """A_s(h) over the grid. An onset needs Score >= h, so h > max score gives 0."""
    out = np.zeros(grid.size, dtype=int)
    probs = None if cfg.mode == MODE_HARD else s.p
    for k, h in enumerate(grid):
        if h > smax + 1e-12:
            break
        out[k] = replay_session(s.t, cfg.params(float(h)), probs=probs, t_end=s.T).n_onsets
    return out


def calibrate(A: np.ndarray, T_h: float, grid: np.ndarray, phi: float) -> tuple[float, int]:
    """§8.1, literally: the smallest h with A(h)/T <= phi."""
    ok = np.flatnonzero(A / T_h <= phi)
    k = int(ok[0])            # always exists: past every max score A = 0
    return float(grid[k]), k


def evaluate(sessions: list[Session], cfg: Config, phis: tuple[float, ...]) -> dict:
    """Everything for one configuration on one headline subset."""
    ss = [x for x in (trimmed(s, cfg.dead_s) for s in sessions) if x is not None]
    base = [s for s in ss if s.group == BASELINE]
    stim = [s for s in ss if s.group == STIMULUS]
    smax = {s.file: max_score(s, cfg.W, cfg.mode) for s in ss}
    grid = h_grid(max(smax.values(), default=1.0), cfg.mode)
    A = {s.file: onset_counts(s, cfg, grid, smax[s.file]) for s in ss}
    T_base_h = sum(s.T for s in base) / 3600
    A_base = np.sum([A[s.file] for s in base], axis=0) if base else np.zeros(grid.size, int)

    curve = pd.DataFrame(dict(h=grid, A=A_base, T_base_h=T_base_h, fa_h=A_base / T_base_h,
                              fa_ub95=[fa_upper_bound(int(a), T_base_h) for a in A_base],
                              sens_insample=[np.mean([A[s.file][k] >= 1 for s in stim]) if stim else np.nan
                                             for k in range(grid.size)],
                              fpr_insample=[np.mean([A[s.file][k] >= 1 for s in base]) for k in range(grid.size)]))
    curve['monotone_A'] = bool(np.all(np.diff(A_base) <= 0))

    xb = np.array([smax[s.file] for s in base])
    xs = np.array([smax[s.file] for s in stim])
    if xb.size and xs.size:
        mw = mannwhitneyu(xs, xb, alternative='two-sided')
        auc = mw.statistic / (xs.size * xb.size)
        U, p_mw = float(mw.statistic), float(mw.pvalue)
    else:
        auc = U = p_mw = float('nan')

    per_phi, folds = [], []
    for phi in phis:
        h_full, k_full = calibrate(A_base, T_base_h, grid, phi)
        tp = tn = 0
        h_base_folds = []
        for s in base + stim:
            if s.group == BASELINE:
                Ab = A_base - A[s.file]
                Tb = T_base_h - s.T / 3600
                h, k = calibrate(Ab, Tb, grid, phi)
                h_base_folds.append(h)
            else:
                h, k = h_full, k_full
            n_on = int(A[s.file][k])
            pred = int(n_on >= 1)
            tp += int(s.group == STIMULUS and pred)
            tn += int(s.group == BASELINE and not pred)
            folds.append(dict(phi=phi, file=s.file, group=s.group, condition=s.condition,
                              h=h, onsets=n_on, predicted_stress=pred,
                              correct=int(pred == (s.group == STIMULUS)),
                              max_score=smax[s.file]))
        hb = np.array(h_base_folds) if h_base_folds else np.array([np.nan])
        A_full = int(A_base[k_full])
        per_phi.append(dict(
            phi=phi, h_full=h_full, A_full=A_full, fa_h=A_full / T_base_h,
            fa_ub95=fa_upper_bound(A_full, T_base_h),
            loso_sens=tp / len(stim) if stim else np.nan, tp=tp, n_stim=len(stim),
            loso_spec=tn / len(base) if base else np.nan, tn=tn, n_base=len(base),
            h_fold_median=float(np.median(hb)), h_fold_min=float(hb.min()),
            h_fold_max=float(hb.max()),
            h_fold_n_differ=int(np.sum(hb != h_full)),
            insample_sens=float(curve['sens_insample'].iloc[k_full]),
            insample_fpr=float(curve['fpr_insample'].iloc[k_full])))
    return dict(cfg=cfg, sessions=ss, base=base, stim=stim, smax=smax, grid=grid, A=A,
                T_base_h=T_base_h, curve=curve, auc=auc, U=U, p_mw=p_mw,
                per_phi=pd.DataFrame(per_phi), folds=pd.DataFrame(folds))


def run_others(others: list[Session], cfg: Config, h: float) -> pd.DataFrame:
    rows = []
    for s in others:
        s2 = trimmed(s, cfg.dead_s)
        if s2 is None:
            continue
        r = replay_session(s2.t, cfg.params(h), probs=None if cfg.mode == MODE_HARD else s2.p,
                           t_end=s2.T, tick_s=1.0)
        rows.append(dict(file=s.file, group=s.group, environment=s.environment,
                         condition=s.condition, hours=s2.T / 3600, events=int(s2.t.size),
                         h=h, onsets=r.n_onsets, onsets_per_h=r.n_onsets / (s2.T / 3600),
                         first_onset_s=r.onsets[0].t if r.onsets else np.nan,
                         alarm_frac=r.alarm_seconds / s2.T, max_occupancy=r.max_occupancy,
                         needs_review=s.needs_review))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--corpus', type=Path, required=True)
    ap.add_argument('--W', type=float, required=True, help='window [s], from premises')
    ap.add_argument('--R', type=float, required=True, help='refractory [s], from premises')
    ap.add_argument('--rho', type=float, default=0.5)
    ap.add_argument('--phi', type=float, required=True,
                    help='declared false alarms / hour (the headline operating point)')
    ap.add_argument('--phis', type=float, nargs='+', default=[0.05, 0.1, 0.2, 0.5])
    ap.add_argument('--R-sweep', type=float, nargs='+', default=None,
                    help='default: W, R/2, R, 2R, 1800')
    ap.add_argument('--no-plots', action='store_true')
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    phis = tuple(sorted(set(a.phis) | {a.phi}))
    out = a.out or a.corpus / f'validation_W{a.W:g}_R{a.R:g}'
    out.mkdir(parents=True, exist_ok=True)

    ses, ev = load_corpus(a.corpus)
    tf = times_by_file(ev)
    empty = (np.empty(0), np.empty(0))
    all_s = []
    for _, r in ses.iterrows():
        if r['group'] not in (BASELINE, STIMULUS, SECONDARY, PROBE) or not np.isfinite(r['duration_s']):
            continue
        t, p = tf.get(r['file'], empty)
        all_s.append(Session(r['file'], r['group'], r['environment'], r['condition'],
                             float(r['duration_s']), t, p, int(r['needs_review']),
                             int(r['stage3_in_sample'])))
    headline = [s for s in all_s if s.group in (BASELINE, STIMULUS) and s.environment == 'indoor']
    others = [s for s in all_s if s not in headline]
    excluded = ses[~ses['file'].isin([s.file for s in all_s])][['file', 'group', 'condition', 'exclude_reason']]

    op = Config(a.W, a.R, a.rho, MODE_HARD)
    print(f'operating point: {op.label()}  phi={a.phi}')
    main_hard = evaluate(headline, op, phis)
    main_soft = evaluate(headline, replace(op, mode=MODE_SOFT), phis)

    def pick(res, phi):
        return res['per_phi'].set_index('phi').loc[phi]

    h_op = float(pick(main_hard, a.phi)['h_full'])
    h_op_soft = float(pick(main_soft, a.phi)['h_full'])

    # invariant: stimulus folds use the full-baseline h
    f = main_hard['folds']
    for phi in phis:
        hf = float(pick(main_hard, phi)['h_full'])
        assert (f[(f['phi'] == phi) & (f['group'] == STIMULUS)]['h'] == hf).all()

    # ── sweeps ───────────────────────────────────────────────────────────────
    R_sweep = a.R_sweep or sorted({a.W, a.R / 2, a.R, 2 * a.R, 1800.0})
    configs = ([replace(op, W=w) for w in W_SWEEP] + [replace(op, R=r) for r in R_sweep]
               + [replace(op, rho=x) for x in RHO_SWEEP] + [replace(op, dead_s=d) for d in DEAD_SWEEP])
    configs = list(dict.fromkeys(configs))
    configs += [replace(c, mode=MODE_SOFT) for c in configs]

    top2 = sorted(main_hard['base'], key=lambda s: -main_hard['smax'][s.file])[:2]
    subsets = {
        'all': headline,
        'no_needs_review': [s for s in headline if s.needs_review == 0],
        'no_top2_baseline': [s for s in headline if s not in top2],
        'stage3_out_of_sample': [s for s in headline if s.in_sample == 0],
        'duration_ge_W': [s for s in headline if s.T >= a.W],
    }
    sweep_rows, cache = [], {}

    def add(res, subset):
        for _, r in res['per_phi'].iterrows():
            c = res['cfg']
            sweep_rows.append(dict(subset=subset, W=c.W, R=c.R, rho=c.rho, mode=c.mode,
                                   dead_s=c.dead_s, n_base=r['n_base'], n_stim=r['n_stim'],
                                   T_base_h=res['T_base_h'], auc=res['auc'], mw_p=res['p_mw'],
                                   **{k: r[k] for k in r.index if k not in ('n_base', 'n_stim')}))

    add(main_hard, 'all')
    add(main_soft, 'all')
    for c in configs:
        if c in (op, replace(op, mode=MODE_SOFT)):
            continue
        print(f'  sweep {c.label()}')
        cache[c] = evaluate(headline, c, phis)
        add(cache[c], 'all')
    for name, sub in subsets.items():
        if name == 'all':
            continue
        for mode in (MODE_HARD, MODE_SOFT):
            print(f'  subset {name} {mode}')
            add(evaluate(sub, replace(op, mode=mode), phis), name)
    sweeps = pd.DataFrame(sweep_rows)
    sweeps.to_csv(out / 'sweeps.csv', index=False)

    # ── outputs ──────────────────────────────────────────────────────────────
    curves = []
    for res in (main_hard, main_soft):
        c = res['curve'].copy()
        c.insert(0, 'config', res['cfg'].label())
        curves.append(c)
    pd.concat(curves).to_csv(out / 'calibration_curve.csv', index=False)

    folds = pd.concat([main_hard['folds'].assign(mode=MODE_HARD),
                       main_soft['folds'].assign(mode=MODE_SOFT)])
    folds.to_csv(out / 'loso_folds.csv', index=False)

    k_h = int(np.flatnonzero(main_hard['grid'] == h_op)[0])
    k_s = int(np.flatnonzero(np.isclose(main_soft['grid'], h_op_soft))[0])
    score_rows = []
    for s in main_hard['sessions']:
        score_rows.append(dict(file=s.file, group=s.group, condition=s.condition,
                               hours=s.T / 3600, events=int(s.t.size),
                               max_score_hard=main_hard['smax'][s.file],
                               max_score_soft=main_soft['smax'][s.file],
                               onsets_hard_at_h=int(main_hard['A'][s.file][k_h]),
                               onsets_soft_at_h=int(main_soft['A'][s.file][k_s]),
                               needs_review=s.needs_review, stage3_in_sample=s.in_sample))
    scores = pd.DataFrame(score_rows).sort_values(['group', 'max_score_hard'], ascending=[True, False])
    scores.to_csv(out / 'session_scores.csv', index=False)

    oth = pd.concat([run_others(others, op, h_op).assign(mode=MODE_HARD),
                     run_others(others, replace(op, mode=MODE_SOFT), h_op_soft).assign(mode=MODE_SOFT)])
    oth.to_csv(out / 'others.csv', index=False)

    # queue capacity (§7.1.5): largest occupancy seen anywhere indoor at the operating point
    occ = {s.file: replay_session(s.t, op.params(h_op), t_end=s.T).max_occupancy
           for s in all_s if s.environment == 'indoor'}
    worst_occ = max(occ.items(), key=lambda kv: kv[1])

    if not a.no_plots:
        plots(out, main_hard, headline, others, op, h_op, a.phi, sweeps)

    # ── report ───────────────────────────────────────────────────────────────
    ph = main_hard['per_phi']
    ps = main_soft['per_phi']
    T_b = main_hard['T_base_h']
    L = []
    L.append('# Stage 5 — calibration and validation\n')
    L.append('Spec: STRESS_DETECTION_ALGORITHM_v1.md. Detector: `src/core/stress_detector.py` '
             '(the replayed class is the deployable one).\n')
    L.append('## Declared inputs\n')
    L.append(f'| W | R | rho | phi_target | mode (primary) |\n|---|---|---|---|---|\n'
             f'| {a.W:g} s | {a.R:g} s | {a.rho:g} | **{a.phi:g} FA/h** | hard (Variant A) |\n')
    L.append('W and R were fixed from `premises/premises.md` before this script was run.\n')
    L.append('> **Caveats that go with every number below.** Stage 3 is in-sample on '
             f'{int(sum(s.in_sample for s in headline))}/{len(headline)} headline sessions. '
             f'{int(sum(s.needs_review for s in headline))} headline sessions have a condition that '
             'is still a guess (§9.4). Indoor only (§12.4). The export may be incomplete — check '
             '`build_stress_corpus.py` output.\n')
    L.append('## Corpus\n')
    L.append(f'Headline: {len(main_hard["base"])} baseline sessions ({T_b:.2f} h), '
             f'{len(main_hard["stim"])} stimulus sessions '
             f'({sum(s.T for s in main_hard["stim"]) / 3600:.2f} h).\n')
    L.append(md_table(pd.DataFrame([dict(group=s.group, condition=s.condition) for s in headline])
                      .value_counts().rename('sessions').reset_index()) + '\n')
    if len(excluded):
        L.append('Not analysed (`include = 0`, unassigned, or no duration):\n')
        L.append(md_table(excluded) + '\n')

    L.append('## §8 Calibration (full baseline)\n')
    L.append(f'Resolution limit: one false alarm = {1 / T_b:.3f} FA/h. Certifying phi with zero '
             f'alarms needs T_min = 3/phi hours of baseline (§8.4):\n')
    L.append(md_table(pd.DataFrame(dict(phi=phis, T_min_h=[3 / x for x in phis],
                                        available_h=T_b,
                                        certifiable=['yes' if T_b >= 3 / x else 'no' for x in phis]))) + '\n')
    for name, tab in (('Variant A — hard', ph), ('Variant B — soft', ps)):
        L.append(f'**{name}.** `fa_ub95` is the one-sided 95 % upper bound (§8.3).\n')
        L.append(md_table(tab[['phi', 'h_full', 'A_full', 'fa_h', 'fa_ub95',
                               'insample_sens', 'insample_fpr']], '.3g') + '\n')
    if not bool(main_hard['curve']['monotone_A'].iloc[0]):
        L.append('⚠️ A(h) is not monotone in h at this configuration: the §8.1 minimum was '
                 'applied literally.\n')

    L.append('## §9.1 Session-level discrimination\n')
    L.append(md_table(pd.DataFrame([
        dict(variant='A hard', AUC=main_hard['auc'], U=main_hard['U'], p=main_hard['p_mw'],
             n_stim=len(main_hard['stim']), n_base=len(main_hard['base'])),
        dict(variant='B soft', AUC=main_soft['auc'], U=main_soft['U'], p=main_soft['p_mw'],
             n_stim=len(main_soft['stim']), n_base=len(main_soft['base']))]), '.3g') + '\n')
    L.append('Mann–Whitney U on per-session max Score(t), two-sided; AUC = U / (n_stim · n_base). '
             'Variant B scores are a ranking, not probabilities (§5.2, no labels to test calibration).\n')

    L.append('## §9.2 Leave-one-session-out\n')
    cols = ['phi', 'loso_sens', 'tp', 'n_stim', 'loso_spec', 'tn', 'n_base', 'h_full',
            'h_fold_median', 'h_fold_min', 'h_fold_max', 'h_fold_n_differ']
    L.append('**Variant A — hard**\n')
    L.append(md_table(ph[cols], '.3g') + '\n')
    L.append('**Variant B — soft**\n')
    L.append(md_table(ps[cols], '.3g') + '\n')
    L.append('`h_fold_*` is over the BASELINE folds; stimulus folds use `h_full` by construction.\n')
    fh = main_hard['folds']
    wrong = fh[(fh['phi'] == a.phi) & (fh['correct'] == 0)]
    L.append(f'Misclassified at phi = {a.phi:g} (Variant A):\n')
    L.append(md_table(wrong[['file', 'group', 'condition', 'h', 'onsets', 'max_score']], '.4g') + '\n')

    L.append('## §9.3 Sensitivity analyses\n')
    L.append(f'All at phi = {a.phi:g}, LOSO. One parameter moved at a time from the declared point.\n')
    sw = sweeps[sweeps['phi'] == a.phi]
    show_cols = ['mode', 'W', 'R', 'rho', 'dead_s', 'n_stim', 'n_base', 'T_base_h', 'h_full',
                 'loso_sens', 'loso_spec', 'h_fold_min', 'h_fold_max', 'auc']
    for title, mask in (
            ('W sweep', (sw['subset'] == 'all') & (sw['R'] == a.R) & (sw['rho'] == a.rho) & (sw['dead_s'] == 0)),
            ('R sweep', (sw['subset'] == 'all') & (sw['W'] == a.W) & (sw['rho'] == a.rho) & (sw['dead_s'] == 0)),
            ('rho sweep', (sw['subset'] == 'all') & (sw['W'] == a.W) & (sw['R'] == a.R) & (sw['dead_s'] == 0)),
            ('dead time (first D seconds dropped, §2.1)',
             (sw['subset'] == 'all') & (sw['W'] == a.W) & (sw['R'] == a.R) & (sw['rho'] == a.rho))):
        L.append(f'**{title}**\n')
        L.append(md_table(sw[mask].sort_values(['mode', 'W', 'R', 'rho', 'dead_s'])[show_cols], '.3g') + '\n')
    L.append('**Subsets**\n')
    L.append(f'`no_top2_baseline` removes: {", ".join(s.file for s in top2)}.\n')
    sub = sw[(sw['W'] == a.W) & (sw['R'] == a.R) & (sw['rho'] == a.rho) & (sw['dead_s'] == 0)]
    L.append(md_table(sub.sort_values(['subset', 'mode'])[['subset'] + show_cols], '.3g') + '\n')

    L.append('## §12 Probes and secondary conditions (full-baseline h, never calibrated on)\n')
    L.append(md_table(oth[oth['mode'] == MODE_HARD].sort_values(['group', 'environment', 'condition'])
                      [['file', 'group', 'environment', 'condition', 'hours', 'events', 'h', 'onsets',
                        'onsets_per_h', 'first_onset_s', 'alarm_frac']], '.3g') + '\n')

    L.append('## §7.1.5 Queue capacity\n')
    L.append(f'Largest window occupancy indoor at the operating point: {worst_occ[1]} '
             f'(`{worst_occ[0]}`).\n')

    L.append('## Per-session scores\n')
    L.append(md_table(scores, '.4g') + '\n')
    L.append('## Figures\n')
    for p in ('fa_curve.png', 'session_max.png', 'sweep_W.png', 'timelines_stimulus.png',
              'timelines_controls.png'):
        L.append(f'![{p}]({p})\n')
    (out / 'report.md').write_text('\n'.join(L))
    print(f'wrote {out}/report.md')
    print(ph[cols].to_string(index=False))
    return 0


def continuous_score(t: np.ndarray, W: float, T: float) -> tuple[np.ndarray, np.ndarray]:
    """Hard-mode Score(t) as a step function over [0, T], for plots only.

    The detector evaluates Score only on arrival (§4.1), which is exact for onsets
    but, drawn as a step plot, holds the value flat until the next event. Here each
    event also steps DOWN at t_i + W — at that instant it is already outside the
    half-open window (t - W, t], so departures sort before arrivals at equal times.
    """
    ts = np.concatenate([t, t + W])
    dy = np.concatenate([np.ones(t.size), -np.ones(t.size)])
    o = np.lexsort((dy, ts))                   # -1 before +1 at equal times
    x = np.concatenate([[0.0], ts[o], [max(T, ts.max() if ts.size else T)]])
    y = np.concatenate([[0.0], np.cumsum(dy[o]), [0.0]])
    y[-1] = y[-2]
    return x, y


def plots(out, res, headline, others, op, h_op, phi, sweeps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # FA curve
    c = res['curve']
    fig, ax = plt.subplots(figsize=(6.5, 3.6), dpi=150)
    style_axes(ax)
    ax.plot(c['h'], c['fa_ub95'], color=INK_2, linewidth=1, linestyle='--', label='95 % upper bound')
    pos = c['A'] > 0
    ax.plot(c['h'][pos], c['fa_h'][pos], color=GROUP_COLOR[BASELINE], linewidth=2, marker='o',
            markersize=4, label='observed A(h) / T_base')
    ax.axhline(phi, color=GROUP_COLOR[STIMULUS], linewidth=1)
    ax.annotate(f'phi = {phi:g}/h', (c['h'].iloc[-1], phi), xytext=(0, 3), textcoords='offset points',
                ha='right', fontsize=7, color=INK_2)
    ax.axvline(h_op, color=INK_2, linewidth=0.8, linestyle=':')
    ax.set_yscale('log')
    ax.set_xscale('log')            # h >= one grid step, never 0
    ax.set_xlim(c['h'].iloc[0] * 0.9, c['h'].iloc[-1] * 1.1)
    ax.set_xlabel('h_high')
    ax.set_ylabel('false alarms per hour')
    ax.set_title(f'Calibration on {len(res["base"])} baseline sessions ({res["T_base_h"]:.1f} h), '
                 f'{op.label()}', fontsize=8, loc='left')
    ax.legend(frameon=False, fontsize=7, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out / 'fa_curve.png')
    plt.close(fig)

    # session max score strip
    rows = []
    for s in res['sessions']:
        rows.append((f'{s.group}: {s.condition}', s.group, res['smax'][s.file]))
    for s in others:
        if s.environment == 'indoor':
            rows.append((f'{s.group}: {s.condition}', s.group, max_score(s, op.W, MODE_HARD)))
    df = pd.DataFrame(rows, columns=['cat', 'group', 'm'])
    cats = sorted(df['cat'].unique(), key=lambda x: ({BASELINE: 0, STIMULUS: 1, SECONDARY: 2, PROBE: 3}
                                                     [x.split(':')[0]], x))
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(cats) + 1), dpi=150)
    style_axes(ax)
    rng = np.random.default_rng(0)
    folds = res['folds']
    hf = folds[(folds['phi'] == phi) & (folds['group'] == BASELINE)]['h']
    if len(hf):
        ax.axvspan(hf.min(), hf.max(), color=GRID, alpha=0.6, linewidth=0)
    ax.axvline(h_op, color=INK, linewidth=1)
    for i, cat in enumerate(cats):
        g = df[df['cat'] == cat]
        ax.scatter(g['m'], i + rng.uniform(-0.2, 0.2, len(g)), s=18,
                   color=GROUP_COLOR[g['group'].iloc[0]], edgecolors='white', linewidths=0.8, zorder=3)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats, fontsize=7, color=INK_2)
    # symlog with linthresh=1 keeps the many 0-count sessions visible at x = 0
    ax.set_xscale('symlog', linthresh=1, linscale=0.5)
    ax.set_xlim(-0.4, df['m'].max() * 1.5)
    ax.set_xlabel(f'max events in any {op.W:g} s window (per session)')
    ax.set_title(f'Per-session max Score; line = h_high at phi = {phi:g}/h, band = LOSO fold range',
                 fontsize=8, loc='left')
    fig.tight_layout()
    fig.savefig(out / 'session_max.png')
    plt.close(fig)

    # W sweep
    sw = sweeps[(sweeps['subset'] == 'all') & (sweeps['phi'] == phi) & (sweeps['R'] == op.R)
                & (sweeps['rho'] == op.rho) & (sweeps['dead_s'] == 0)]
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.0), dpi=150, sharey=True)
    for ax, mode in zip(axs, (MODE_HARD, MODE_SOFT)):
        style_axes(ax)
        g = sw[sw['mode'] == mode].sort_values('W')
        ax.plot(g['W'], g['loso_sens'], color=GROUP_COLOR[STIMULUS], linewidth=2, marker='o',
                markersize=4, label='sensitivity')
        ax.plot(g['W'], g['loso_spec'], color=GROUP_COLOR[BASELINE], linewidth=2, marker='o',
                markersize=4, label='specificity')
        ax.axvline(op.W, color=GRID, linewidth=1)
        ax.set_xscale('log')
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('W [s]')
        ax.set_title(f'LOSO, {mode}, phi = {phi:g}/h, R = {op.R:g} s', fontsize=8, loc='left')
    axs[0].legend(frameon=False, fontsize=7, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out / 'sweep_W.png')
    plt.close(fig)

    # timelines
    def timelines(sessions, fname, title):
        if not sessions:
            return
        ncol = 3
        nrow = math.ceil(len(sessions) / ncol)
        fig, axs = plt.subplots(nrow, ncol, figsize=(10, 1.6 * nrow + 0.6), dpi=130, squeeze=False)
        for ax in axs.flat[len(sessions):]:
            ax.axis('off')
        for ax, s in zip(axs.flat, sessions):
            style_axes(ax)
            x, y = continuous_score(s.t, op.W, s.T)
            ax.step(x / 60, y, where='post', color=GROUP_COLOR[s.group], linewidth=1.2)
            r = replay_session(s.t, op.params(h_op), t_end=s.T)
            for e in r.onsets:
                ax.axvline(e.t / 60, color=INK, linewidth=0.8)
            ax.axhline(h_op, color=INK_2, linewidth=0.8, linestyle='--')
            ax.axhline(op.rho * h_op, color=GRID, linewidth=0.8, linestyle=':')
            ax.set_xlim(0, s.T / 60)
            ax.set_ylim(0, max(h_op * 1.3, y.max() * 1.1))
            ax.set_title(f'{s.file[:42]}  ({s.condition}, {r.n_onsets} onset)', fontsize=6, loc='left')
            ax.tick_params(labelsize=6)
        fig.suptitle(title, fontsize=9, x=0.01, ha='left', color=INK)
        fig.supxlabel('time [min]', fontsize=7, color=INK_2)
        fig.tight_layout()
        fig.savefig(out / fname)
        plt.close(fig)

    stim = sorted(res['stim'], key=lambda s: -res['smax'][s.file])
    timelines(stim, 'timelines_stimulus.png',
              f'Score(t), stimulus sessions — dashed h_high = {h_op:g}, dotted h_low, vertical = onset')
    worst_base = sorted(res['base'], key=lambda s: -res['smax'][s.file])[:6]
    probes = [s for s in others if s.environment == 'indoor' and s.group == PROBE]
    timelines(worst_base + probes, 'timelines_controls.png',
              f'Score(t), 6 highest baseline sessions + indoor probes — h_high = {h_op:g}')


if __name__ == '__main__':
    sys.exit(main())
