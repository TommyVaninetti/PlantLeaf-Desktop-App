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
LIVE EVENT PIPELINE — Stages 2-4 on the events the firmware sends, in real time.

Firmware v3 in event mode runs Stage 1 on the MCU and transmits ONLY click
candidates plus their two immediate neighbours. This worker turns that stream
back into the same annotated rows the offline pipeline produces, so a click seen
live and the same click re-analysed from its .paudio file are the same row.

That is not an aspiration, it is the design rule: every step below calls the
same function `click_detection_worker.py` calls, with the same arguments —
`reconstruct_frame_v5(normalize=True)`, `build_click_context`, `resolve_click`,
`compute_features_v5`, `click_event_key`, `run_stages234_annotated`. Nothing is
reimplemented here. What IS here is the part that has no offline counterpart:
reassembling prev|curr|next triples out of a stream that has gaps in it, and
deduplicating without being able to see the whole recording first.

WHAT EVENT MODE CANNOT PRODUCE, and why it is safe

  * The v6 spectral family (spectral_entropy, shape_novelty, spectral_tilt,
    FPE_hz_region, SPR_region, f_50_hz, IQR_f) and harmonic_confinement need
    Buffer 3 — a rolling MEAN per-bin noise PSD over 750 accepted frames. Its
    input is the quiet frames, which event mode never transmits, and its
    750x154 ring does not fit in the MCU's 128 kB. p_noise_psd is therefore
    None and those features come back NaN.
  * local_crest needs E[j] over j in [i-10, i+10]; the wire carries +/-1.

  Neither costs a detection. The DEPLOYED model reads seven features —
  peak_SNR, pre_SNR, post_SNR, fall_time_ms, rise_time_ms, fit_valid, R2 — and
  none of them is in the list above; five need only the two scalars
  noise_floor / std_noise, which the firmware ships per event and which were
  verified against the host reference to 7.7e-08 and 1.6e-07. In Stage 2,
  `_stage2_reason` treats NaN as "pass", so the local_crest and
  harmonic_confinement gates are simply not applied — both are documented as
  rejecting 0.0 % of labelled clicks, so recall is unchanged and only more
  noise reaches the SVM. The other four gates (peak_SNR floor, the
  non-physical ceiling, n_seg and SPR) do apply: n_seg is emitted by
  compute_features_v5 before its p_noise_psd guard.

THREADING

  submit() is a plain, lock-guarded method callable from any thread and never
  blocks its caller: the serial reader must not be stalled by feature
  extraction. Work is drained by a timer running in THIS object's thread.
  When the inbox is full the OLDEST pending frame is dropped and counted —
  losing the oldest keeps the newest triple assemblable, and a silent stall
  would be far worse than a counted loss.
"""

import threading
import traceback
from collections import deque

import numpy as np
from PySide6 import QtCore
from PySide6.QtCore import Signal, Slot

from core.click_pipeline_v5 import (
    FFT_SIZE,
    FS,
    K_STAGE1_DEFAULT,
    LOCAL_CREST_C,
    PEAK_MATCH_SAMPLES,
    PEAK_REFRACTORY_R,
    STAGE_BLOCKED_DEDUP,
    STAGE_OK,
    build_click_context,
    click_event_key,
    compute_features_v5,
    load_svm_model,
    reconstruct_frame_v5,
    resolve_click,
    run_stage4_v5,
    run_stages234_annotated,
)


class LiveEventWorker(QtCore.QObject):
    """Stages 2-4 on a live event stream. Lives in its own QThread."""

    #: One fully annotated row, ready for EventsTable.add_event().
    eventReady = Signal(dict)
    #: Counters for the status line: dropped, overflow, incomplete, backlog.
    statusChanged = Signal(dict)
    #: Non-fatal problem worth telling the user about once.
    error = Signal(str)

    #: Frames held before feature extraction. 512 frames is ~1.3 s of a
    #: continuous burst — far more than the firmware's own 12-slot USB FIFO
    #: can put on the wire, so this bound is reached only if the HOST stalls.
    MAX_INBOX = 512

    #: A resolved row is held until a candidate this many frames newer has
    #: been resolved, so the two candidates of one straddling click are
    #: deduplicated together rather than one at a time. 3 frames is 7.7 ms.
    FLUSH_LAG_FRAMES = 3

    #: Frames drained per timer tick. Bounded so the thread stays responsive
    #: to stop() and flush() even under a sustained burst.
    DRAIN_PER_TICK = 32

    def __init__(self, fs=FS, fft_size=FFT_SIZE, model_path=None,
                 threshold=None, stage2_mode=None, k=K_STAGE1_DEFAULT,
                 session_id='live', parent=None):
        super().__init__(parent)
        self.fs = int(fs)
        self.fft_size = int(fft_size)
        self.model_path = model_path
        self.threshold = threshold
        self.stage2_mode = stage2_mode
        self.k = float(k)
        self.session_id = session_id

        self._svm_model = None
        self._timer = None

        self._lock = threading.Lock()
        self._inbox = deque()

        self.reset()

    # ─────────────────────────────────────────────────────────────────────
    #  Lifecycle
    # ─────────────────────────────────────────────────────────────────────

    def reset(self):
        """Forget everything: a new recording starts at frame 0 again."""
        with self._lock:
            self._inbox.clear()
        self._frames = {}            # frame_idx -> event dict from the reader
        self._waiting = []           # candidate frame indices awaiting a verdict
        self._pending = []           # annotated rows awaiting the dedup flush
        self._newest_idx = -1
        self._last_kept_peak_abs = None
        self.n_events_in = 0
        self.n_candidates = 0
        self.n_rows_out = 0
        self.n_inbox_dropped = 0     # host could not keep up
        self.n_board_overflow = 0    # board's FIFO dropped events before us
        self.n_incomplete = 0        # candidate missing prev and/or next
        self.n_failed = 0            # a candidate that raised

    @Slot()
    def start(self):
        """Called once the object is inside its thread (thread.started signal)."""
        if self._timer is None:
            self._timer = QtCore.QTimer(self)
            self._timer.setInterval(5)
            self._timer.timeout.connect(self._drain)
        self._timer.start()

    @Slot()
    def stop(self):
        if self._timer is not None:
            self._timer.stop()

    @Slot()
    def flush(self):
        """
        End of recording: drain the inbox, resolve every candidate still
        waiting for a neighbour that will never arrive, and emit everything.
        """
        while True:
            with self._lock:
                empty = not self._inbox
            if empty:
                break
            self._drain()

        for idx in sorted(self._waiting):
            self._resolve(idx)
        self._waiting = []
        self._flush_pending(final=True)
        self._emit_status()

    # ─────────────────────────────────────────────────────────────────────
    #  Intake
    # ─────────────────────────────────────────────────────────────────────

    def submit(self, event):
        """
        Hand one event frame to the worker. Thread-safe, never blocks.

        Returns False when the inbox was full and the oldest frame had to be
        discarded — the caller can surface that, but must not retry.
        """
        dropped = False
        with self._lock:
            if len(self._inbox) >= self.MAX_INBOX:
                self._inbox.popleft()
                self.n_inbox_dropped += 1
                dropped = True
            self._inbox.append(event)
        return not dropped

    def backlog(self):
        with self._lock:
            return len(self._inbox)

    # ─────────────────────────────────────────────────────────────────────
    #  Draining
    # ─────────────────────────────────────────────────────────────────────

    @Slot()
    def _drain(self):
        for _ in range(self.DRAIN_PER_TICK):
            with self._lock:
                if not self._inbox:
                    return
                event = self._inbox.popleft()
            try:
                self._ingest(event)
            except Exception as e:                       # noqa: BLE001
                # One bad frame must not stop a recording.
                self.n_failed += 1
                print(f"⚠️ Live event: frame {event.get('frame_idx')} "
                      f"skipped ({e})")
        self._emit_status()

    def _ingest(self, event):
        idx = int(event['frame_idx'])
        self.n_events_in += 1
        if event.get('had_overflow'):
            # The board raises this on the frame AFTER the loss, so it means
            # "events were dropped before this one" — not "this one is bad".
            self.n_board_overflow += 1

        self._frames[idx] = event
        if idx > self._newest_idx:
            self._newest_idx = idx
        if event.get('is_candidate'):
            self.n_candidates += 1
            self._waiting.append(idx)

        # A candidate is decidable once its next neighbour has arrived, or once
        # a LATER frame has arrived — which proves the next neighbour is never
        # coming (the board sends strictly ascending frame indices).
        ready = [c for c in self._waiting
                 if (c + 1) in self._frames or self._newest_idx > c + 1]
        for c in sorted(ready):
            self._waiting.remove(c)
            self._resolve(c)

        self._prune_frames()
        self._flush_pending()

    def _prune_frames(self):
        """Keep only what an undecided candidate or a future one can still need."""
        keep_from = self._newest_idx - 2
        if self._waiting:
            keep_from = min(keep_from, min(self._waiting) - 1)
        for idx in [i for i in self._frames if i < keep_from]:
            del self._frames[idx]

    # ─────────────────────────────────────────────────────────────────────
    #  One candidate
    # ─────────────────────────────────────────────────────────────────────

    def _model(self):
        if self._svm_model is None and self.model_path is not None:
            # joblib, not pickle: the model carries raw numpy buffers.
            self._svm_model = load_svm_model(self.model_path)
        return self._svm_model

    def _signal_of(self, idx):
        event = self._frames.get(idx)
        if event is None:
            return None
        fd = reconstruct_frame_v5(
            np.asarray(event['fft_mags']), np.asarray(event['phases']),
            self.fs, self.fft_size, normalize=True,
        )
        return fd

    def _resolve(self, cand_idx):
        """Run Stages 2-4 for one candidate frame and queue the annotated row."""
        curr = self._signal_of(cand_idx)
        if curr is None:
            self.n_failed += 1
            return
        event = self._frames[cand_idx]

        prev = self._signal_of(cand_idx - 1)
        nxt = self._signal_of(cand_idx + 1)
        if prev is None or nxt is None:
            # Legal at the very start of a recording, and after a board FIFO
            # overflow. build_click_context handles it; the row is marked so
            # nobody later mistakes a truncated context for a clean one.
            self.n_incomplete += 1

        ctx = build_click_context(
            prev['signal'] if prev else None,
            curr['signal'],
            nxt['signal'] if nxt else None,
        )

        noise_floor = float(event['noise_floor'])
        std_noise = float(event['std_noise'])
        resolved = resolve_click(ctx, noise_floor, std_noise)

        # p_noise_psd is None on purpose — see the module docstring. It is the
        # one argument that differs from the offline call, and it is what turns
        # the v6 spectral family into an honest NaN instead of a wrong number.
        features = compute_features_v5(
            ctx, resolved,
            curr['fft_norm'], curr['freq_axis'],
            noise_floor, std_noise, self.fs,
            p_noise_psd=None,
        )
        peak_abs, canonical_frame_idx = click_event_key(ctx, resolved, cand_idx)

        E_i = float(event['E_i'])
        E_hat_floor = float(event['E_hat_floor'])
        d0 = int(resolved.get('decay_start', 0))
        d1 = int(resolved.get('decay_end', 0))

        cand = {
            'schema_version': 'v6',
            'session_id': self.session_id,
            'frame_idx': int(cand_idx),
            # Stage 1 ran on the MCU; record its configuration the same way the
            # offline exporter does, with the source made explicit.
            'stage1_params': (f'fw_v3_event;k={self.k:.2f};'
                              f'R={PEAK_REFRACTORY_R};C={LOCAL_CREST_C}'),
            'noise_floor': noise_floor,
            'std_noise': std_noise,
            'noise_floor_mV': round(noise_floor * 1e3, 4),
            'std_noise_mV': round(std_noise * 1e3, 4),
            'E_hat_floor': E_hat_floor,
            'E_i': E_i,
            'k_ratio': (E_i / E_hat_floor) if E_hat_floor else float('nan'),
            'timestamp_s': cand_idx * self.fft_size / float(self.fs),
        }
        cand.update(features)
        cand.update({
            'peak_amp': resolved['peak_amp'],
            'peak_abs': peak_abs,
            'canonical_frame_idx': canonical_frame_idx,
            'decay_len': max(0, d1 - d0),
            # No Buffer 3 in event mode, so no frames went into one.
            'b3_frames': 0,
            # suppress_edge_artifacts' signature: its fade's first coefficient
            # is exactly 0, and nothing else in the chain makes a hard zero.
            'gibbs_fired': int(len(curr['signal']) > 0
                               and curr['signal'][0] == 0.0),
            # Live-only provenance. Kept out of CSV_COLUMNS on purpose: they
            # describe the LINK, not the click.
            'ctx_signal': ctx['signal'],
            'ctx_origin': ctx['origin'],
            'ctx_seams': ctx['seams'],
            'fft_mags': np.asarray(event['fft_mags']),
            'phases': np.asarray(event['phases']),
            'ctx_complete': bool(prev is not None and nxt is not None),
            'board_overflow': bool(event.get('had_overflow')),
        })

        model = self._model()
        if model is None:
            # No model: Stage 2's verdict is still meaningful and worth showing.
            from core.click_pipeline_v5 import _stage2_reason
            cand['stage_blocked'] = _stage2_reason(cand, self.stage2_mode)
            cand['svm_probability'] = None
            cand['svm_prediction'] = None
            row = cand
        else:
            row = run_stages234_annotated(
                [cand], model, threshold=self.threshold,
                stage2_mode=self.stage2_mode,
            )[0]

        self._pending.append(row)

    # ─────────────────────────────────────────────────────────────────────
    #  Live Stage 4
    # ─────────────────────────────────────────────────────────────────────

    def _flush_pending(self, final=False):
        """
        Emit resolved rows once no future candidate can share their click.

        Offline, Stage 4 sees the whole recording at once. Live it cannot, so
        rows are held for FLUSH_LAG_FRAMES and deduplicated in batches. That is
        exact for the case Stage 4 exists for — a click straddling a frame
        boundary produces candidates in ADJACENT frames, which always land in
        the same batch. The one case a batch boundary could split is two
        candidates >= 3 frames apart whose peaks both sit on the shared frame
        edge; `_last_kept_peak_abs` closes it by carrying the previous batch's
        last kept peak across.
        """
        if not self._pending:
            return

        if final:
            ready, self._pending = self._pending, []
        else:
            cutoff = self._newest_idx - self.FLUSH_LAG_FRAMES
            ready = [r for r in self._pending if r['frame_idx'] <= cutoff]
            if not ready:
                return
            self._pending = [r for r in self._pending
                             if r['frame_idx'] > cutoff]

        ready.sort(key=lambda r: r.get('peak_abs', 0))

        confirmed = [r for r in ready if r.get('stage_blocked') == STAGE_OK]
        if len(confirmed) > 1:
            kept_ids = {id(r) for r in run_stage4_v5(confirmed)}
            for row in confirmed:
                if id(row) not in kept_ids:
                    row['stage_blocked'] = STAGE_BLOCKED_DEDUP

        for row in ready:
            if row.get('stage_blocked') == STAGE_OK:
                peak_abs = int(row.get('peak_abs', 0))
                if (self._last_kept_peak_abs is not None
                        and peak_abs - self._last_kept_peak_abs
                        <= PEAK_MATCH_SAMPLES):
                    row['stage_blocked'] = STAGE_BLOCKED_DEDUP
                else:
                    self._last_kept_peak_abs = peak_abs
            self.n_rows_out += 1
            self.eventReady.emit(row)

    # ─────────────────────────────────────────────────────────────────────

    def _emit_status(self):
        self.statusChanged.emit({
            'events_in': self.n_events_in,
            'candidates': self.n_candidates,
            'rows_out': self.n_rows_out,
            'inbox_dropped': self.n_inbox_dropped,
            'board_overflow': self.n_board_overflow,
            'incomplete': self.n_incomplete,
            'failed': self.n_failed,
            'backlog': self.backlog(),
        })
