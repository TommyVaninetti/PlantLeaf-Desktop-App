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
render_confirmed_clicks.py — screenshots of the hand-confirmed clicks, without re-exporting

    QT_QPA_PLATFORM=offscreen python3 scripts/results/render_confirmed_clicks.py \
        --labelled ~/PlantLeaf_dev/Analisi/v6/labelled/_all \
        --manifest ~/PlantLeaf_dev/Analisi/v6/re-export/export_manifest.csv \
        -o ~/PlantLeaf_dev/Analisi/v6/results/click_images

WHY THIS IS CHEAP
-----------------
A v3 .paudio is a flat array of fixed-size frames: 128-byte header, then
154 bins x (float32 magnitude + int8 phase) = 770 bytes per frame. So frame i
starts at 128 + i*770 and can be read on its own. A click needs its own frame and
its two neighbours — about 2 KB — instead of the 0.5-2.5 GB a full export reads.
Measured: a re-export is ~8 min per recording; this is seconds for the whole
corpus.

EVERY NUMBER ON THE IMAGE IS THE EXPORTED ONE
---------------------------------------------
The waveform, envelope and spectra are rebuilt from the three frames; every
feature printed in the footer is copied from the candidate's CSV row. Nothing is
recomputed for display, so an image cannot disagree with the dataset.

The reconstruction is then CHECKED against the CSV: `--verify` recomputes the v5
features from the rebuilt frames and compares them to the stored values. On a test
click all six matched to 5 decimals. A mismatch means the reconstruction is wrong,
and the row is reported and skipped rather than rendered — a plausible-looking
wrong picture is worse than no picture.

THE MISSING CURVE (deliberate)
------------------------------
The FFT panel normally draws three curves; the third is Buffer 3's noise PSD,
which is ESTIMATOR STATE — a rolling 750-frame history with burst gating, not
stored per click. It cannot be recovered from three frames, so it is omitted and
the panel draws the region spectrum and the frame context only. Decided 20-09-2026.
Everything else, including every v6 number in the footer, is unaffected.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / 'src'))

import windows  # noqa: E402,F401  (breaks a pre-existing circular import)

from PySide6.QtWidgets import QApplication                      # noqa: E402
from components.data_collection_dialog_v5 import (              # noqa: E402
    CandidateData, _get_frame_envelope_pair, _region_display_spectrum,
    _render_candidate_screenshot,
)
from core.click_pipeline_v5 import (                            # noqa: E402
    build_click_context, resolve_click, compute_features_v5, FS, FFT_SIZE,
)

HEADER = 128
BINS = 154
FRAME_BYTES = BINS * 5
RECORD = np.dtype([('mag', '<f4'), ('ph', 'i1')])   # interleaved per bin

#: v5 features recomputed from the rebuilt frames and compared with the CSV.
VERIFY = ('peak_SNR', 'rise_time_ms', 'fall_time_ms', 'tau_ms', 'R2', 'FPE_hz')

#: Everything the footer prints, copied straight from the CSV row.
COPY_FLOAT = (
    'E_hat_floor', 'peak_SNR', 'pre_SNR', 'post_SNR', 'rise_time_ms', 'fall_time_ms',
    'asymmetry_integral', 'ZCR_pre', 'ZCR_click', 'ZCR_post', 'kurtosis',
    'centroid_shift_hz', 'tau_ms', 'R2', 'fit_coverage', 'SPR', 'R_spectral', 'FPE_hz',
    'spectral_entropy', 'shape_novelty', 'spectral_tilt', 'temporal_concentration',
    'FPE_hz_region', 'SPR_region', 'f_50_hz', 'IQR_f', 'local_crest',
    'harmonic_confinement', 'hc_f1_hz', 'hc_r_A', 'hc_r_B', 'run_crest', 'k_ratio',
)
COPY_INT = ('fit_valid', 'decay_len', 'n_seg', 'b3_frames', 'gibbs_fired',
            'run_id', 'run_length', 'pos_in_run', 'would_pass_v5', 'peak_abs')


class FrameFile:
    """Random access to one .paudio's frames. Reads only what is asked for."""

    def __init__(self, path: Path):
        self.path = path
        size = path.stat().st_size
        self.n_frames = max((size - HEADER) // FRAME_BYTES, 0)
        self._fh = open(path, 'rb')

    def __len__(self):
        return self.n_frames

    def frame(self, idx: int):
        if not (0 <= idx < self.n_frames):
            return None, None
        self._fh.seek(HEADER + idx * FRAME_BYTES)
        buf = self._fh.read(FRAME_BYTES)
        if len(buf) < FRAME_BYTES:
            return None, None
        a = np.frombuffer(buf, dtype=RECORD)
        return a['mag'].astype(np.float32), a['ph'].astype(np.int8)

    def close(self):
        self._fh.close()


class LazyFrames:
    """What _get_frame_envelope_pair expects of `dm`: fft_mags / phase_int8 / len.

    It indexes three frames and takes len(dm.fft_mags) for the bounds check, so a
    pair of thin sequence views over the file is enough — no array is materialised.
    """

    class _View:
        def __init__(self, ff: FrameFile, which: int):
            self._ff, self._which = ff, which

        def __len__(self):
            return len(self._ff)

        def __getitem__(self, idx):
            return self._ff.frame(idx)[self._which]

    def __init__(self, ff: FrameFile):
        self.fft_mags = self._View(ff, 0)
        self.phase_int8 = self._View(ff, 1)


def _f(row, key, default=float('nan')):
    v = str(row.get(key, '')).strip().replace(',', '.')
    if v in ('', 'None', 'nan'):
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _i(row, key, default=0):
    v = _f(row, key, float('nan'))
    return default if v != v else int(v)


def read_manifest(path: Path) -> dict[str, str]:
    out = {}
    with open(path, newline='', encoding='utf-8-sig') as fh:
        for r in csv.reader(fh):
            if len(r) >= 6 and r[0] != 'file' and r[5] == 'ok':
                out[r[0]] = r[1]
    return out


def build_candidate(row: dict, dm, stem: str, normalize: bool):
    """Rebuild one click from its three frames; returns (candidate, recomputed)."""
    frame_idx = _i(row, 'frame_idx', -1)
    (prev_env, curr_env, next_env, curr_sig, prev_sig, next_sig,
     curr_fft_norm, curr_freq_axis) = _get_frame_envelope_pair(dm, frame_idx, normalize=normalize)

    nf = _f(row, 'noise_floor_mV') * 1e-3
    sd = _f(row, 'std_noise_mV') * 1e-3
    ctx = build_click_context(prev_sig, curr_sig, next_sig)
    resolved = resolve_click(ctx, nf, sd)

    # p_noise_psd = None: Buffer 3 is estimator state and cannot be recovered from
    # three frames, so the noise curve is omitted (see the module docstring).
    disp_freqs, disp_region, _disp_noise = _region_display_spectrum(
        ctx, resolved, None, curr_freq_axis, FS)

    recomputed = compute_features_v5(ctx, resolved, curr_fft_norm, curr_freq_axis,
                                     nf, sd, FS)

    c = CandidateData(
        file=stem, frame_idx=frame_idx, timestamp_s=_f(row, 'timestamp_s'),
        noise_floor=nf, std_noise=sd,
        **{k: _f(row, k) for k in COPY_FLOAT if k != 'E_hat_floor'},
        E_hat_floor=_f(row, 'E_hat_floor'),
    )
    for k in COPY_INT:
        setattr(c, k, _i(row, k))
    c.session_id = str(row.get('session_id', stem))
    c.label = str(row.get('label', ''))
    c.note = str(row.get('note', ''))
    c.stage_blocked = str(row.get('stage_blocked', '') or '')
    p = _f(row, 'svm_probability')
    c.svm_probability = None if p != p else p
    pred = row.get('svm_prediction', '')
    c.svm_prediction = None if str(pred).strip() in ('', 'None', 'nan') else _i(row, 'svm_prediction')

    # ── Plot arrays — the same construction the exporter uses ────────────────
    # resolve_click returns indices, not curves: the render window, the marks and
    # the fit curve are built by the export loop, so they are rebuilt identically
    # here. tau and R2 come from the CSV, so the drawn fit is the exported fit.
    env_ctx, sig_ctx = ctx['envelope'], ctx['signal']
    peak, onset = resolved['peak'], resolved['onset']
    d_start, d_end = resolved['decay_start'], resolved['decay_end']

    half = FFT_SIZE // 2
    w0 = max(0, min(peak - half, onset))
    w1 = min(len(sig_ctx), max(peak + half, d_end + 1))
    c.render_signal = sig_ctx[w0:w1]
    c.render_envelope = env_ctx[w0:w1]
    c.render_t_ms = (np.arange(w0, w1, dtype=np.float64) - peak) / FS * 1000.0
    c.mark_onset_ms = (onset - peak) / FS * 1000.0
    c.mark_decay_end_ms = (d_end - peak) / FS * 1000.0
    c.seams_ms = [float((s - peak) / FS * 1000.0) for s in ctx['seams'] if w0 <= s < w1]

    tau_ms_val, R2_val = _f(row, 'tau_ms', -1.0), _f(row, 'R2', 0.0)
    c.fit_t_ms = c.fit_y = None
    if tau_ms_val > 0 and R2_val > 0.1 and d_end > d_start + 2:
        try:
            n_arr = np.arange(d_end - d_start, dtype=np.float64)
            rate = 1.0 / ((max(tau_ms_val, 0.01) / 1000.0) * FS)
            A0 = float(env_ctx[min(d_start, len(env_ctx) - 1)])
            c.fit_t_ms = (np.arange(d_start, d_end, dtype=np.float64) - peak) / FS * 1000.0
            c.fit_y = A0 * np.exp(-rate * n_arr)
        except Exception:                                        # noqa: BLE001
            pass

    c.fft_norm = curr_fft_norm
    c.freq_axis = curr_freq_axis
    c.region_freqs = disp_freqs
    c.region_amp = disp_region
    c.noise_amp = np.array([])                  # omitted on purpose
    return c, recomputed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--labelled', required=True, type=Path,
                    help='apply_labels.py _all/ directory of labelled candidate CSVs')
    ap.add_argument('--manifest', required=True, type=Path,
                    help='export_manifest.csv — gives each recording its .paudio path')
    ap.add_argument('-o', '--out', required=True, type=Path)
    ap.add_argument('--labels', default='1',
                    help='which label values to render (default 1 = confirmed clicks; '
                         '"1,2" adds the ambiguous ones)')
    ap.add_argument('--no-verify', action='store_true',
                    help='skip the feature check (not recommended)')
    ap.add_argument('--tol', type=float, default=1e-3,
                    help='relative tolerance for the feature check (default 1e-3)')
    ap.add_argument('--no-normalize', action='store_true',
                    help='disable the microphone correction (export used it ON)')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()

    app = QApplication.instance() or QApplication([])           # noqa: F841
    wanted = {s.strip() for s in a.labels.split(',') if s.strip()}
    manifest = read_manifest(a.manifest)
    csvs = sorted(p for p in a.labelled.rglob('*_candidates.csv')
                  if not p.name.startswith('._'))
    if not csvs:
        print(f'ERROR: no *_candidates.csv under {a.labelled}')
        return 2

    a.out.mkdir(parents=True, exist_ok=True)
    index_path = a.out / 'rendered_clicks.csv'
    index = open(index_path, 'w', newline='', encoding='utf-8')
    iw = csv.writer(index)
    iw.writerow(['file', 'frame_idx', 'timestamp_s', 'label', 'svm_probability',
                 'peak_SNR', 'verdict', 'image'])

    t0 = time.time()
    n_render = n_skip_file = n_bad = n_exists = 0
    problems: list[str] = []
    for path in csvs:
        stem = path.name.replace('_candidates.csv', '')
        with open(path, newline='', encoding='utf-8-sig') as fh:
            rows = [r for r in csv.DictReader(fh)
                    if str(r.get('label', '')).strip().rstrip('.0') in wanted
                    or str(r.get('label', '')).strip() in wanted]
        if not rows:
            continue
        src = manifest.get(stem)
        if not src or not Path(src).exists():
            n_skip_file += len(rows)
            problems.append(f'{stem}: .paudio not found ({src or "not in manifest"}) '
                            f'— {len(rows)} click(s) skipped')
            continue

        ff = FrameFile(Path(src))
        dm = LazyFrames(ff)
        out_dir = a.out / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        done = 0
        for row in rows:
            if a.limit is not None and n_render >= a.limit:
                break
            frame_idx = _i(row, 'frame_idx', -1)
            img = out_dir / f'{stem}_{frame_idx:06d}.png'
            if img.exists() and not a.overwrite:
                n_exists += 1
                continue
            try:
                c, recomputed = build_candidate(row, dm, stem, normalize=not a.no_normalize)
            except Exception as e:                               # noqa: BLE001
                n_bad += 1
                problems.append(f'{stem} frame {frame_idx}: {type(e).__name__}: {e}')
                continue

            if not a.no_verify:
                bad = []
                for k in VERIFY:
                    want, got = _f(row, k), float(recomputed.get(k, float('nan')))
                    if want != want and got != got:
                        continue
                    if abs(got - want) > max(a.tol, abs(want) * a.tol):
                        bad.append(f'{k} {got:.5g} vs CSV {want:.5g}')
                if bad:
                    n_bad += 1
                    problems.append(f'{stem} frame {frame_idx}: reconstruction differs — '
                                    + '; '.join(bad))
                    continue

            if _render_candidate_screenshot(c, img):
                n_render += 1
                done += 1
                iw.writerow([stem, frame_idx, row.get('timestamp_s', ''),
                             row.get('label', ''), row.get('svm_probability', ''),
                             row.get('peak_SNR', ''),
                             'confirmed' if c.is_confirmed_click else (c.stage_blocked or 'n/a'),
                             str(img)])
            else:
                n_bad += 1
                problems.append(f'{stem} frame {frame_idx}: renderer returned False')
        ff.close()
        if done:
            print(f'  {done:4} image(s)  {stem}')

    index.close()
    el = time.time() - t0
    print(f'\nrendered {n_render} image(s) in {el:.1f}s -> {a.out}')
    if n_exists:
        print(f'  {n_exists} already existed (use --overwrite to redo)')
    if n_skip_file:
        print(f'  {n_skip_file} click(s) skipped: .paudio not reachable')
    if n_bad:
        print(f'  ⚠️  {n_bad} click(s) NOT rendered')
    for p in problems[:25]:
        print(f'     {p}')
    if len(problems) > 25:
        print(f'     ... and {len(problems) - 25} more')
    print(f'  index: {index_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
