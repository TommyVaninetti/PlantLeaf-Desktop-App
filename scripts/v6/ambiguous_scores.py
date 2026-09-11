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
ambiguous_scores.py — what does the model think the ambiguous rows are?

    python3 scripts/v6/ambiguous_scores.py \
        --model  ".../finals/6best-plusR2/plantleaf_svm_v6_..._28082026.pkl" \
        --csv    ".../training_set_27082026_ambiguousincluded_v5evaluated.csv" \
        -o ambiguous_scores.png

WHY THIS EXISTS
---------------
Every classifier here emits a number in [0, 1] before anything is decided: the
SVM's estimated probability that a row is a click. `predict_proba` IS that
number. Every ROC curve and every "Set B AUC" in the reports is a summary of it;
thresholding at 0.121 is the last step, applied after the number already exists.
It exists for every row, always, including rows nobody has labelled.

So there is a question the labels cannot answer but the model can: the reviewer
marked 99 rows label = 2, "could be either". A model trained with
`--ambiguous exclude` never saw them. Run its `predict_proba` on those 99 rows
and it applies exactly the arithmetic it applies to everything else, with no
knowledge that a human hesitated over them. Where the resulting numbers land is
independent evidence about what the ambiguous set actually contains.

Read the answer as a LOCATION, not a verdict. If the ambiguous scores sit on top
of the confirmed clicks, the model agrees they are clicks. If they sit on the
noise, the reviewer's hesitation was the model's certainty. If they sit between
— which is what this corpus actually shows — then the ambiguous set is genuinely
intermediate, and no hard relabelling to 0 or 1 is faithful to it.

WHAT IT GUARANTEES
------------------
Two things have to be true or the evidence is circular, and both are checked
before anything is plotted:

  1. The model was trained with `--ambiguous exclude`. Scoring ambiguous rows
     with a model that was TRAINED on them (`--ambiguous click/noise`) measures
     memorisation, not agreement. Refused unless --allow-trained-on-ambiguous.

  2. No ambiguous row reached training. The .pkl records n_train, n_train_clicks
     and n_train_noise; clicks + noise == n_train is arithmetic proof that no
     third class was present. Checked, not assumed.

SET A vs SET B, AND WHY BOTH ARE SHOWN
--------------------------------------
The split is taken from the MODEL's sessions_train / sessions_test, not from the
CSV's `set` column, so it is the split the model was actually held out from.

  Set B is the clean comparison. Those whole sessions were held out, so clicks,
  noise and ambiguous rows there are ALL unseen and score on equal terms.

  Set A is shown too, but its clicks and noise are in-sample and score
  optimistically high, while its ambiguous rows were never trained on. That
  asymmetry works AGAINST the hypothesis: it inflates the click reference the
  ambiguous rows are being compared to. If they still land click-side there, the
  in-sample advantage did not manufacture it.

Read Set B as the result and Set A as corroboration, never the reverse.

WHAT IT REPORTS
---------------
Per set and per class: n, median probability, IQR, and the fraction above the
model's own tuned threshold. Then two rank statistics that need no threshold:

    AUC(ambiguous vs noise)   1.0 = every ambiguous row outranks every noise row
    AUC(click vs ambiguous)   0.5 = ambiguous and clicks are indistinguishable

with Mann-Whitney U p-values. Rank statistics are used because they do not care
how the probabilities are calibrated, and an SVM's `predict_proba` (Platt
scaling) is not calibrated in any strong sense — its ORDERING is what the ROC
curves in the reports already rest on.

THE PLOT
--------
Every row is drawn as its own point. At n = 25 ambiguous rows in Set B a smoothed
density would invent structure that 25 points cannot support, so the strip plot
shows the actual sample and the ECDF below it shows the whole distribution
without a bandwidth choice. The threshold is drawn where it actually sits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402

from scipy.stats import mannwhitneyu                               # noqa: E402
from sklearn.metrics import roc_auc_score                          # noqa: E402

REPO = Path(__file__).resolve().parents[2]

#: Sentinel columns the v6 exporter writes on a failed decay fit. Kept in step
#: with train_svm.FIT_SENTINEL_COLS — imported from there when the module is
#: importable so there is one definition, with this as the fallback.
FIT_SENTINEL_COLS = ('tau_ms', 'R2')
try:
    sys.path.insert(0, str(REPO / 'src' / 'ml'))
    from train_svm import FIT_SENTINEL_COLS as _FSC                # noqa: E402  NORMAL WARNING: import outside top-level package
    FIT_SENTINEL_COLS = _FSC
except Exception:                                                  # noqa: BLE001
    pass

CLICK, NOISE, AMBIG = 1, 0, 2

#: Blue / amber / grey — separable in greyscale and for the common colour vision
#: deficiencies, which red-green is not. Amber is the accent because the
#: ambiguous set is the subject; the two reference classes stay quiet.
STYLE = {
    NOISE: dict(name='noise (label 0)',      color='#6b7a88', marker='o'),
    AMBIG: dict(name='ambiguous (label 2)',  color='#d98324', marker='D'),
    CLICK: dict(name='click (label 1)',      color='#1b6f8c', marker='o'),
}
ORDER = [NOISE, AMBIG, CLICK]


# ── loading ──────────────────────────────────────────────────────────────────

def load_scored(csv_path: Path, model: dict) -> pd.DataFrame:
    """
    Read the CSV, encode it exactly as the model was trained, and score it.

    "Exactly" is the whole point. nan_policy in particular is a train/serve pair:
    the trainer converts the decay-fit sentinels to NaN and lets the pipeline's
    imputer handle them, so a scorer that leaves tau_ms = -1.0 in place is
    feeding the model a value it was never fitted on. The policy is read back
    from the .pkl rather than assumed.
    """
    feats = list(model['features'])
    df = pd.read_csv(csv_path)

    for col in ('label', 'session_id'):
        if col not in df.columns:
            sys.exit(f"ERROR: {csv_path.name} has no '{col}' column.")

    df['label'] = pd.to_numeric(df['label'].astype(str).str.strip(), errors='coerce')
    df = df[df['label'].notna()].copy()
    df['label'] = df['label'].astype(int)

    missing = [f for f in feats if f not in df.columns]
    if missing:
        sys.exit(f"ERROR: the CSV is missing features the model needs: {missing}")

    # Same coercion as load_and_prepare — Italian decimal commas and stray space.
    for col in feats + ['fit_valid']:
        if col not in df.columns:
            continue
        if df[col].dtype == object:
            df[col] = (df[col].astype(str).str.strip()
                              .str.replace(',', '.', regex=False))
        df[col] = pd.to_numeric(df[col], errors='coerce')

    nan_policy = model.get('nan_policy', 'sentinel')
    if nan_policy == 'nan':
        if 'fit_valid' not in df.columns:
            sys.exit("ERROR: model nan_policy='nan' needs a fit_valid column.")
        failed = df['fit_valid'].fillna(0).astype(int) == 0
        present = [c for c in FIT_SENTINEL_COLS if c in df.columns]
        df.loc[failed, present] = np.nan
        print(f"  nan_policy 'nan':  {int(failed.sum())} row(s) with fit_valid == 0 "
              f"→ {', '.join(present)} set to NaN")
    elif nan_policy != 'sentinel':
        sys.exit(f"ERROR: unknown nan_policy {nan_policy!r} in the model.")
    else:
        print(f"  nan_policy 'sentinel':  decay-fit sentinels left as written")

    df['p'] = model['pipeline'].predict_proba(df[feats].to_numpy(float))[:, 1]

    # The split comes from the model, so it is the split the model was actually
    # held out from — not whatever a later CSV happens to say in its `set`
    # column. Those can disagree once a corpus is re-exported.
    train_s = set(model.get('sessions_train') or [])
    test_s  = set(model.get('sessions_test') or [])
    df['split'] = np.where(df['session_id'].isin(test_s), 'B',
                  np.where(df['session_id'].isin(train_s), 'A', 'unseen-session'))

    if 'set' in df.columns:
        csv_split = df['set'].astype(str).str.strip().str.upper()
        clash = int(((csv_split.isin(('A', 'B'))) & (csv_split != df['split'])).sum())
        if clash:
            print(f"  ⚠️  {clash} row(s) where the CSV's 'set' column disagrees with "
                  f"the model's session lists — the model's lists win.")

    n_new = int((df['split'] == 'unseen-session').sum())
    if n_new:
        print(f"  {n_new} row(s) sit in sessions the model has never seen at all; "
              f"reported separately as 'unseen-session'.")
    return df


def assert_not_circular(model: dict, allow: bool) -> None:
    """Refuse to present memorisation as agreement."""
    amb = model.get('ambiguous')
    if amb != 'exclude' and not allow:
        sys.exit(
            f"ERROR: this model was trained with --ambiguous {amb!r}, so the "
            f"ambiguous rows WERE in its training data.\n"
            f"  Scoring them with it measures what it memorised, not whether it "
            f"independently agrees.\n"
            f"  Use a model trained with --ambiguous exclude, or pass "
            f"--allow-trained-on-ambiguous to override and read the result as "
            f"a sanity check only."
        )

    n, nc, nn = (model.get('n_train'), model.get('n_train_clicks'),
                 model.get('n_train_noise'))
    if None in (n, nc, nn):
        print("  ⚠️  the model records no training row counts, so 'no ambiguous row "
              "reached training' cannot be verified from the file — trusting the "
              "--ambiguous flag alone.")
        return
    if nc + nn != n:
        sys.exit(
            f"ERROR: the model's own counts do not add up: {nc} clicks + {nn} "
            f"noise = {nc + nn}, but n_train = {n}. The {n - nc - nn} unaccounted "
            f"row(s) mean a third class was present in training."
        )
    print(f"  verified: {nc} clicks + {nn} noise = {n} training rows, "
          f"no third class present")


# ── statistics ───────────────────────────────────────────────────────────────

def describe(p: np.ndarray, thr: float) -> dict:
    return {'n': len(p), 'median': float(np.median(p)),
            'q1': float(np.quantile(p, .25)), 'q3': float(np.quantile(p, .75)),
            'frac_above': float((p >= thr).mean())}


def rank_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """AUC of a-over-b plus the Mann-Whitney p-value, or (nan, nan) if degenerate."""
    if len(a) < 1 or len(b) < 1:
        return float('nan'), float('nan')
    y = np.r_[np.ones(len(a), int), np.zeros(len(b), int)]
    auc = roc_auc_score(y, np.r_[a, b])
    try:
        pval = float(mannwhitneyu(a, b, alternative='two-sided').pvalue)
    except ValueError:
        pval = float('nan')
    return float(auc), pval


def report(df: pd.DataFrame, thr: float) -> dict:
    out = {}
    for split in ('B', 'A', 'unseen-session'):
        d = df[df['split'] == split]
        if d.empty:
            continue
        groups = {lab: d[d['label'] == lab]['p'].to_numpy() for lab in ORDER}
        head = {'B': 'Set B — whole sessions held out; every class here is unseen',
                'A': 'Set A — clicks/noise are IN-SAMPLE; only the ambiguous rows are not',
                'unseen-session': 'Sessions absent from both of the model\'s lists'}[split]
        print(f"\n  {head}")
        print(f"    {'class':<22} {'n':>5} {'median':>8} {'IQR':>17} {'≥ thr':>8}")
        for lab in ORDER:
            p = groups[lab]
            if not len(p):
                continue
            s = describe(p, thr)
            print(f"    {STYLE[lab]['name']:<22} {s['n']:>5} {s['median']:>8.3f} "
                  f"  [{s['q1']:.3f}, {s['q3']:.3f}]   {100*s['frac_above']:>6.1f}%")

        amb, noi, clk = groups[AMBIG], groups[NOISE], groups[CLICK]
        if len(amb) and len(noi):
            auc, pv = rank_stats(amb, noi)
            print(f"    AUC(ambiguous vs noise) = {auc:.3f}   "
                  f"(1.0 = every ambiguous row outranks every noise row, "
                  f"p = {pv:.2e})")
        if len(amb) and len(clk):
            auc, pv = rank_stats(clk, amb)
            print(f"    AUC(click vs ambiguous) = {auc:.3f}   "
                  f"(0.5 = indistinguishable from confirmed clicks, "
                  f"p = {pv:.2e})")
        out[split] = groups
    return out


# ── plot ─────────────────────────────────────────────────────────────────────

def _ecdf(p: np.ndarray):
    x = np.sort(p)
    return np.r_[0, x, 1], np.r_[0, np.arange(1, len(x) + 1) / len(x), 1]


def draw(df: pd.DataFrame, thr: float, model: dict, dest: Path,
         csv_name: str) -> None:
    splits = [s for s in ('B', 'A') if (df['split'] == s).any()]
    if not splits:
        sys.exit("ERROR: no rows fell into Set A or Set B.")

    fig, axes = plt.subplots(
        2, len(splits), figsize=(6.6 * len(splits), 8.4),
        gridspec_kw=dict(height_ratios=[1.35, 1], hspace=.34, wspace=.30),
        squeeze=False)

    rng = np.random.default_rng(0)      # jitter only; fixed so the plot is stable

    for col, split in enumerate(splits):
        d = df[df['split'] == split]
        ax, axe = axes[0][col], axes[1][col]
        ticklabels = []

        for row, lab in enumerate(ORDER):
            p = d[d['label'] == lab]['p'].to_numpy()
            st = STYLE[lab]
            if not len(p):
                ticklabels.append(STYLE[lab]['name'])
                continue
            y = row + rng.uniform(-.26, .26, len(p))
            ax.scatter(p, y, s=26 if lab == AMBIG else 16,
                       c=st['color'], marker=st['marker'],
                       alpha=.80 if lab == AMBIG else .45,
                       linewidths=.5 if lab == AMBIG else 0,
                       edgecolors='white' if lab == AMBIG else 'none',
                       zorder=3 if lab == AMBIG else 2)
            med = float(np.median(p))
            ax.plot([med, med], [row - .36, row + .36], color=st['color'],
                    lw=2.6, zorder=4, solid_capstyle='butt')
            ax.plot([np.quantile(p, .25), np.quantile(p, .75)], [row - .40] * 2,
                    color=st['color'], lw=3.4, alpha=.55, zorder=4,
                    solid_capstyle='butt')
            # The counts live in the tick label rather than floating beside the
            # axes: text placed outside the axes lands on the NEXT subplot.
            ticklabels.append(f"{st['name']}\nn={len(p)} · median {med:.2f}")

            xs, ys = _ecdf(p)
            axe.step(xs, ys, where='post', color=st['color'],
                     lw=2.4 if lab == AMBIG else 1.5,
                     alpha=1.0 if lab == AMBIG else .7, zorder=3 if lab == AMBIG else 2)

        for a in (ax, axe):
            a.axvline(thr, color='#a8323c', ls='--', lw=1.4, zorder=5)
            a.set_xlim(-0.02, 1.02)
            a.grid(axis='x', color='#dfe3e8', lw=.7, zorder=0)
            a.set_axisbelow(True)
            for sp in ('top', 'right'):
                a.spines[sp].set_visible(False)

        ax.set_yticks(range(len(ORDER)))
        ax.set_yticklabels(ticklabels, fontsize=9)
        for tick, lab in zip(ax.get_yticklabels(), ORDER):
            tick.set_color(STYLE[lab]['color'])
        ax.set_ylim(-.72, len(ORDER) - .28)
        ax.set_xticklabels([])
        sub = {'B': 'held out — every class unseen',
               'A': 'clicks/noise in-sample; ambiguous rows unseen'}[split]
        ax.set_title(f"Set {split}   ·   {sub}", fontsize=11, loc='left', pad=9)
        ax.text(thr, len(ORDER) - .34, f'  threshold {thr:.3f}', color='#a8323c',
                fontsize=8.5, va='top', ha='left')

        axe.set_ylim(0, 1.02)
        if col == 0:
            axe.set_ylabel('fraction of rows at or below', fontsize=9)
        axe.set_xlabel('model probability that the row is a click   '
                       '(predict_proba)', fontsize=9.5)
        axe.set_title('cumulative distribution', fontsize=9.5, loc='left', pad=6)

    handles = [Line2D([], [], marker=STYLE[l]['marker'], ls='', color=STYLE[l]['color'],
                      label=STYLE[l]['name'], markersize=6) for l in ORDER]
    handles.append(Line2D([], [], color='#a8323c', ls='--',
                          label=f"decision threshold ({thr:.3f})"))
    axes[0][0].legend(handles=handles, loc='upper left', bbox_to_anchor=(0, -0.10),
                      ncol=4, frameon=False, fontsize=9)

    amb_b = df[(df['split'] == 'B') & (df['label'] == AMBIG)]['p'].to_numpy()
    lead = ''
    if len(amb_b):
        lead = (f"In Set B the model puts {100*(amb_b >= thr).mean():.0f}% of the "
                f"ambiguous rows above its own threshold, at a median of "
                f"{np.median(amb_b):.2f}.")
    fig.suptitle(
        "What the model calls the rows the reviewer could not\n"
        f"{model.get('kernel','?')} · {len(model['features'])} features · trained "
        f"--ambiguous exclude, so these {int((df['label']==AMBIG).sum())} rows were "
        f"never in its training data\n{lead}",
        fontsize=12.5, ha='left', x=0.055, y=0.985, va='top', linespacing=1.5)
    fig.text(0.055, 0.012, f"model: {Path(model.get('_path','?')).name}   ·   "
             f"rows: {csv_name}", fontsize=7.5, color='#7a838c')

    fig.subplots_adjust(top=.845, bottom=.085, left=.155, right=.975)
    fig.savefig(dest, dpi=130)
    plt.close(fig)
    print(f"\n  Plot written: {dest}")


# ── entry point ──────────────────────────────────────────────────────────────

DEFAULT_MODEL = ("/Volumes/Lexar 1TB/PlantLeaf/Analysis v6/Training/Models and "
                 "Reports/finals/6best-plusR2/"
                 "plantleaf_svm_v6_6bestfeatures_plusR2_28082026.pkl")
DEFAULT_CSV = ("/Volumes/Lexar 1TB/PlantLeaf/Analysis v6/Training/"
               "training_set_27082026_ambiguousincluded_v5evaluated.csv")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score the ambiguous (label 2) rows with a model that never "
                    "saw them, and plot where they land.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', type=Path, default=Path(DEFAULT_MODEL),
                    help='Trained .pkl. Must be --ambiguous exclude.')
    ap.add_argument('--csv', type=Path, default=Path(DEFAULT_CSV),
                    help='Training CSV that still CONTAINS the label = 2 rows.')
    ap.add_argument('-o', '--output', type=Path, default=Path('ambiguous_scores.png'))
    ap.add_argument('--csv-out', type=Path, default=None,
                    help='Also write per-row scores here, so individual ambiguous '
                         'rows can be pulled up in the review dialog.')
    ap.add_argument('--allow-trained-on-ambiguous', action='store_true',
                    help='Score with a model that WAS trained on the ambiguous '
                         'rows. The result is then memorisation, not agreement.')
    args = ap.parse_args()

    if not args.model.exists():
        sys.exit(f"ERROR: no model at {args.model}")
    if not args.csv.exists():
        sys.exit(f"ERROR: no CSV at {args.csv}")

    print(f"\nModel:  {args.model}")
    model = joblib.load(args.model)
    model['_path'] = str(args.model)
    thr = float(model['threshold'])
    print(f"  {model.get('kernel')} · {len(model['features'])} features · "
          f"scale={model.get('scale')} · threshold={thr:.4f}")
    print(f"  features: {model['features']}")
    assert_not_circular(model, args.allow_trained_on_ambiguous)

    print(f"\nRows:   {args.csv}")
    df = load_scored(args.csv, model)
    n_amb = int((df['label'] == AMBIG).sum())
    if not n_amb:
        sys.exit("ERROR: this CSV has no label = 2 rows — nothing to score.\n"
                 "  Use the export that still contains the ambiguous labels.")
    counts = df['label'].value_counts().to_dict()
    print(f"  scored {len(df)} rows:  {counts.get(1,0)} click · {n_amb} ambiguous "
          f"· {counts.get(0,0)} noise")
    if 'fit_valid' in df.columns:
        bad = int((df[df['label'] == AMBIG]['fit_valid'].fillna(0) == 0).sum())
        print(f"  of the {n_amb} ambiguous rows, {bad} had a failed decay fit "
              f"(imputed, as in training)")

    report(df, thr)
    draw(df, thr, model, args.output, args.csv.name)

    if args.csv_out:
        # frame_idx + timestamp_s + file are what the review dialog locates a
        # candidate by, so they travel with the score or the CSV cannot be acted
        # on. peak_SNR rides along because it is the feature doing most of the
        # work, and seeing it next to p makes an odd score interpretable.
        cols = [c for c in ('session_id', 'file', 'frame_idx', 'timestamp_s',
                            'label', 'split', 'p', 'peak_SNR', 'fit_valid',
                            'note') if c in df.columns]
        out = df[cols].sort_values('p', ascending=False)
        out.to_csv(args.csv_out, index=False)
        print(f"  Per-row scores: {args.csv_out}")


if __name__ == '__main__':
    main()
