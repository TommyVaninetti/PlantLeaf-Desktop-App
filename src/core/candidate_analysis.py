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
ONE CANDIDATE, ONE COMPUTATION — the Stage 2/3 input for a single click.

Three places need exactly this: the offline detector walking a continuous
recording, the same detector walking an event recording, and the live pipeline
watching events arrive from the board. They must produce the SAME ROW for the
same click, because a click seen live, the same click re-analysed from its
.paudio file, and the same click exported for a retrain are supposed to be one
row with one set of numbers.

That is not a tidiness argument. `_stage1_select` in click_pipeline_v5 carries a
docstring recording what happened last time this computation existed in more
than one place: two Stage-1 paths disagreed about mic correction and produced
2762 vs 3879 candidates on the same recording. One rule, one place.

WHAT THIS DOES NOT DO

  * It does not find candidates. Stage 1 decides that — on the host for a
    continuous recording, on the MCU for an event one.
  * It does not run Stages 2-4. Those are list operations
    (`run_stages234_annotated`), because Stage 4's dedup compares candidates
    against each other.

So this is the middle: given a candidate frame and the frames either side of it,
produce the feature vector and the identity Stage 4 groups on.
"""

from typing import Optional

import numpy as np

from core.click_pipeline_v5 import (
    FFT_SIZE,
    FS,
    build_click_context,
    click_event_key,
    compute_features_v5,
    reconstruct_frame_v5,
    resolve_click,
)


def _reconstruct(frame, fs, fft_size):
    """One (mags, phases) pair to its reconstructed frame dict, or None."""
    if frame is None:
        return None
    mags, phases = frame
    if mags is None or phases is None:
        return None
    return reconstruct_frame_v5(
        np.asarray(mags), np.asarray(phases), fs, fft_size,
        # normalize=True is the whole pipeline's convention: the mic correction
        # is frequency-dependent (0.55x-1.49x across 20-80 kHz), so it does not
        # cancel out of anything and a frame reconstructed without it is not
        # comparable with a noise floor built from frames that had it.
        normalize=True,
    )


def analyse_candidate(prev_frame, curr_frame, next_frame, *,
                      frame_idx: int,
                      noise_floor: float,
                      std_noise: float,
                      fs: int = FS,
                      fft_size: int = FFT_SIZE,
                      p_noise_psd: Optional[np.ndarray] = None) -> Optional[dict]:
    """
    Features and identity for one Stage 1 candidate.

    Parameters
    ----------
    prev_frame, curr_frame, next_frame : (mags, phases) or None
        The candidate frame and its two TEMPORAL neighbours — the frames either
        side of it in the recording, not in whatever array holds them. `None` is
        legal for the neighbours and means "that frame is not available": the
        start or end of a recording, or, in event mode, a neighbour the board
        never transmitted. `build_click_context` stitches what it is given and
        moves `origin` accordingly, so the features are still computed, on two
        frames instead of three. `ctx_complete` in the result says which
        happened, because a click measured on a truncated context is worth
        less than one measured on a full one and nothing else records that.
    frame_idx : int
        The candidate's position in the RECORDING, not in the array holding the
        frames. `click_event_key` builds `peak_abs = frame_idx * FFT_SIZE + ...`
        and Stage 4 groups on that, so an array index here would make two clicks
        from different parts of an event recording collide.
    noise_floor, std_noise : float
        The adaptive noise state at this frame [V]. From the host estimator for
        a continuous recording; from the board for an event one, where the host
        cannot recompute it — it comes from a minimum-statistics estimator over
        the quiet frames, which event mode never transmits.
    p_noise_psd : np.ndarray or None
        Buffer 3's per-bin noise PSD. None switches the v6 spectral family to
        NaN rather than to a wrong number, which is what event mode needs: B3 is
        a rolling mean over 750 accepted frames and its input is exactly the
        quiet frames that were not sent.

    Returns
    -------
    dict, or None when the candidate frame cannot be reconstructed.
        Every `compute_features_v5` key, plus:
          peak_amp, peak_abs, canonical_frame_idx, decay_len, b3_frames,
          gibbs_fired          - what the CSV schema and Stage 4 need
          ctx_signal, ctx_origin, ctx_seams, ctx_region, ctx_complete
                               - the trace the features were measured on, so a
                                 plot can draw those samples instead of
                                 re-deriving a different set
    """
    curr = _reconstruct(curr_frame, fs, fft_size)
    if curr is None:
        return None

    prev = _reconstruct(prev_frame, fs, fft_size)
    nxt = _reconstruct(next_frame, fs, fft_size)

    ctx = build_click_context(
        prev['signal'] if prev else None,
        curr['signal'],
        nxt['signal'] if nxt else None,
    )
    resolved = resolve_click(ctx, noise_floor, std_noise)
    features = compute_features_v5(
        ctx, resolved,
        curr['fft_norm'], curr['freq_axis'],
        noise_floor, std_noise, fs,
        p_noise_psd=p_noise_psd,
    )
    peak_abs, canonical_frame_idx = click_event_key(ctx, resolved, frame_idx)

    d0 = int(resolved.get('decay_start', 0))
    d1 = int(resolved.get('decay_end', 0))

    out = dict(features)
    out.update({
        'peak_amp': resolved['peak_amp'],
        'peak_abs': peak_abs,
        'canonical_frame_idx': canonical_frame_idx,
        'decay_len': max(0, d1 - d0),
        # How many frames went into the B3 estimate behind p_noise_psd. Zero
        # when there is no B3 at all, which is what tells a reviewer that the
        # NaN v6 columns are structural and not a warm-up artefact.
        'b3_frames': 0,
        # suppress_edge_artifacts' signature: its fade's first coefficient is
        # exactly 0, and nothing else in the chain produces a hard zero there.
        'gibbs_fired': int(len(curr['signal']) > 0 and curr['signal'][0] == 0.0),

        # ── the trace, for whatever draws it ────────────────────────────────
        'ctx_signal': ctx['signal'],
        'ctx_origin': ctx['origin'],
        'ctx_seams': ctx['seams'],
        # The decay window as indices into ctx_signal - the SAME span
        # _feat_v6_spectral measures on. Handing a plot the span beats making it
        # re-derive one from peak_abs, which is how the picture and the numbers
        # end up describing different samples.
        'ctx_region': (int(resolved['onset']), int(resolved['decay_end']) + 1),
        'ctx_complete': bool(prev is not None and nxt is not None),
    })
    return out
