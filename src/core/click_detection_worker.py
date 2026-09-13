"""
Background worker that runs the full v5 click detection pipeline on a recording.

Stage 1 → per-candidate features → Stages 2/3/4 (annotated), off the GUI thread.

TWO KINDS OF RECORDING

  * CONTINUOUS (.paudio v3): every frame is on disk, so Stage 1 runs here.
  * EVENT (.paudio v4): only click candidates and their neighbours were
    transmitted, and Stage 1 already ran ON THE MCU. Re-running it host-side is
    not merely wasteful, it is impossible — the criterion is E_i > k x Ê_floor(i)
    and Ê_floor comes from a minimum-statistics estimator over the QUIET frames,
    which are exactly the ones event mode does not send. So for an event
    recording the candidates are read from the EVNT footer's flags, and the
    board's own noise state travels with them.

  The per-candidate computation is identical either way: both branches call
  core.candidate_analysis.analyse_candidate, which is also what the live
  pipeline calls. That is deliberate — see that module's docstring for what
  happened the last time this computation existed in more than one place.
Every Stage 1 candidate comes back, each carrying its features and the verdict keys
written by run_stages234_annotated ('stage_blocked', 'svm_probability',
'svm_prediction'), so the caller can show either the confirmed clicks or the whole
census without re-running anything.

Threading notes (macOS):
  • QObject + moveToThread, not a QThread subclass — the pattern used elsewhere in
    this codebase (see main_window_chemical_simulator._launch_click_detector).
  • No Qt widgets are created here. Results are emitted and rendered by the caller
    on the GUI thread.
  • click_pipeline_v5 is pure numpy and SVC.predict_proba is libsvm (no BLAS), so
    all of this is safe off the main thread — see run_stage3_v5's docstring.
"""

import traceback

from PySide6.QtCore import QObject, Signal


class PaudioDataManagerAdapter:
    """
    Minimal duck-typed data manager for run_stage1_v5.

    run_stage1_v5 only needs .fft_data / .phase_data / .header_info / .total_frames,
    so a recording held as raw arrays can be fed to the pipeline without constructing
    a full AudioDataManager.
    """

    def __init__(self, fft_data, phase_data, fs, fft_size):
        self.fft_data     = fft_data
        self.phase_data   = phase_data
        self.header_info  = {'fs': fs, 'fft_size': fft_size}
        self.total_frames = len(fft_data)


class ClickDetectionWorker(QObject):
    """
    Run the v5 pipeline over one recording.

    Signals
    -------
    finished(list) : every Stage 1 candidate, annotated with features + verdict.
                     Empty list when Stage 1 found nothing.
    progress(int, int) : (frames_done, frames_total) during feature extraction.
    error(str) : the pipeline could not run at all (e.g. the model failed to load).
    """

    finished = Signal(list)
    progress = Signal(int, int)
    error    = Signal(str)

    def __init__(self, fft_data, phase_data, fs, fft_size, frame_duration_ms,
                 model_path=None, k=None, threshold=None, dm=None,
                 stage2_mode=None):
        super().__init__()
        self.fft_data          = fft_data
        self.phase_data        = phase_data
        self.fs                = fs
        self.fft_size          = fft_size
        self.frame_duration_ms = frame_duration_ms
        self.model_path        = model_path   # None → the model shipped with the app
        self.k                 = k            # None → K_STAGE1_DEFAULT
        self.threshold         = threshold    # None → the model's own threshold
        # The real data manager, when the caller has one. It carries the per-frame
        # arrays AudioLoadWorker computed at load time, which is BOTH much faster than
        # recomputing Stage 1 and the path every candidate CSV (and hence the trained
        # model) came from. Without it we fall back to run_stage1_v5.
        self.dm                = dm
        # None → the module default (conservative). The aggressive tier costs a
        # measured 2.1 % of clicks, so it is never chosen implicitly.
        self.stage2_mode       = stage2_mode
        self._stop_requested   = False

    def _frames_for(self, cand):
        """
        The candidate frame and its two TEMPORAL neighbours, as (mags, phases).

        For a continuous recording, temporal neighbours are array neighbours.
        For an event recording they are not, and the candidate carries the row
        indices resolved by _event_candidates — a `None` there means the board
        never transmitted that neighbour, at the start of a recording or after a
        USB FIFO overflow. analyse_candidate accepts None and measures on two
        frames instead of three rather than stitching an unrelated frame on.
        """
        row = cand.get('row_idx', cand['frame_idx'])
        n_frames = min(len(self.fft_data), len(self.phase_data))

        def frame(i):
            if i is None or not (0 <= i < n_frames):
                return None
            return (self.fft_data[i], self.phase_data[i])

        if 'row_idx' in cand:                       # event recording
            return frame(cand.get('prev_row')), frame(row), frame(cand.get('next_row'))
        return frame(row - 1), frame(row), frame(row + 1)

    def _event_candidates(self):
        """
        Stage 1's verdict for an event recording, read from the EVNT footer.

        The board flagged each transmitted frame as candidate and/or neighbour,
        and shipped the noise state it measured at that frame. Both travel here
        unchanged: the host cannot recompute either, and the values were verified
        against this pipeline's own reference to ~1e-7 on hardware.

        The `run_*` / `local_crest` diagnostics that run_stage1_v5 attaches are
        absent, because they need frames that were never sent — local_crest wants
        +/-10 frames of energy and the wire carries +/-1. They stay missing rather
        than being filled with a plausible number; Stage 2's gates treat a missing
        value as a pass, which is the honest outcome.
        """
        from core import paudio_format as pf

        dm = self.dm
        flags = getattr(dm, 'event_flags', None)
        if flags is None or len(flags) == 0:
            return []

        idx = dm.event_frame_idx
        # Which row holds a given recording frame, so a candidate's neighbours
        # can be found by TIME rather than by array adjacency.
        row_of = {int(f): r for r, f in enumerate(idx)}

        out = []
        for row, fl in enumerate(flags):
            if not (int(fl) & pf.FLAG_CANDIDATE):
                continue
            frame_idx = int(idx[row])
            out.append({
                'frame_idx':   frame_idx,
                'row_idx':     row,
                'prev_row':    row_of.get(frame_idx - 1),
                'next_row':    row_of.get(frame_idx + 1),
                'E_i':         float(dm.event_E_i[row]),
                'E_hat_floor': float(dm.event_E_hat_floor[row]),
                'noise_floor': float(dm.event_noise_floor[row]),
                'std_noise':   float(dm.event_std_noise[row]),
                'board_overflow': bool(int(fl) & pf.FLAG_OVERFLOW),
            })
        return out

    def request_stop(self):
        """Cooperative cancel — checked between candidates."""
        self._stop_requested = True

    def run(self):
        try:
            from core.candidate_analysis import analyse_candidate
            from core.click_pipeline_v5 import (
                run_stage1_v5, run_stage1_v5_precomputed,
                has_precomputed_stage1_arrays, run_stages234_annotated,
                load_svm_model, K_STAGE1_DEFAULT,
                p_noise_at, p_noise_frames_at,
            )
            from ml import default_model_path

            model_path = self.model_path or default_model_path()
            svm_model  = load_svm_model(model_path)

            k = self.k if self.k is not None else K_STAGE1_DEFAULT

            is_event = bool(getattr(self.dm, 'is_event_recording', False))

            if is_event:
                # The board already ran Stage 1, and the golden test matched its
                # output frame-for-frame against this pipeline on 2.7 M frames.
                # Reading the verdict is not a shortcut; it is the only correct
                # answer available, because Ê_floor cannot be rebuilt from the
                # frames that were transmitted.
                stage1 = self._event_candidates()
            elif self.dm is not None and has_precomputed_stage1_arrays(self.dm):
                # Prefer the precomputed arrays: same Stage 1 the Data Collection
                # export runs, so the two features can never report different
                # candidates for the same recording.
                stage1 = run_stage1_v5_precomputed(self.dm, k=k)
            else:
                dm = PaudioDataManagerAdapter(
                    self.fft_data, self.phase_data, self.fs, self.fft_size
                )
                stage1 = run_stage1_v5(dm, k=k)

            if not stage1:
                self.finished.emit([])
                return

            candidates = []
            total = len(stage1)

            for n, cand in enumerate(stage1):
                if self._stop_requested:
                    self.finished.emit([])
                    return

                # frame_idx is the position in the RECORDING; row is the position
                # in the arrays holding the frames. They are the same number for
                # a continuous recording and are not for an event one, and the
                # distinction matters twice: click_event_key builds peak_abs from
                # frame_idx (so Stage 4 groups correctly), while fft_data has to
                # be indexed by row.
                fi = cand['frame_idx']
                row = cand.get('row_idx', fi)
                try:
                    prev_frame, curr_frame, next_frame = self._frames_for(cand)
                    if curr_frame is None:
                        continue

                    # p_noise_psd is what SWITCHES THE v6 FAMILY ON. Without it
                    # _feat_v6_spectral returns its NaN skeleton, so every v6 feature
                    # was NaN here while the identical call in the Data Collection
                    # export produced real values — the two paths silently disagreed.
                    # It needs the data manager's Buffer-3 snapshots, so it is None
                    # (and the v6 features honestly NaN) when the caller had no dm,
                    # and always None for an event recording, where B3 cannot exist:
                    # it is a rolling mean over 750 accepted frames and its input is
                    # the quiet frames that were never transmitted.
                    p_noise = (None if is_event or self.dm is None
                               else p_noise_at(self.dm, row))

                    analysed = analyse_candidate(
                        prev_frame, curr_frame, next_frame,
                        frame_idx=fi,
                        noise_floor=cand['noise_floor'],
                        std_noise=cand['std_noise'],
                        fs=self.fs, fft_size=self.fft_size,
                        p_noise_psd=p_noise,
                    )
                    if analysed is None:
                        continue

                    E_i = float(cand.get('E_i', float('nan')))
                    floor = float(cand.get('E_hat_floor', float('nan')))
                    candidates.append({
                        **cand, **analysed,
                        'timestamp_s': fi * self.frame_duration_ms / 1000.0,
                        # Display/export columns that are not features. Filled
                        # here so a row from this worker and a row from the live
                        # pipeline carry the same keys — the two are meant to be
                        # comparable side by side, and a blank column would look
                        # like a missing measurement rather than a missing field.
                        'noise_floor_mV': round(cand['noise_floor'] * 1e3, 4),
                        'std_noise_mV': round(cand['std_noise'] * 1e3, 4),
                        'k_ratio': (E_i / floor) if floor else float('nan'),
                        # How many frames the B3 estimate behind p_noise averaged.
                        # Not a feature: it is what tells a reviewer a warm, full
                        # window from one built during warm-up — or, at zero, that
                        # there was no B3 at all.
                        'b3_frames': (0 if p_noise is None
                                      else p_noise_frames_at(self.dm, row)),
                    })

                except Exception as e:  # noqa: BLE001
                    # One bad frame must not abort a whole recording.
                    print(f"⚠️ Click detection: frame {fi} skipped ({e})")

                if n % 10 == 0:
                    self.progress.emit(n, total)

            self.progress.emit(total, total)

            annotated = run_stages234_annotated(
                candidates, svm_model, threshold=self.threshold,
                stage2_mode=self.stage2_mode,
            )
            self.finished.emit(annotated)

        except Exception as e:  # noqa: BLE001
            self.error.emit(f"{e}\n{traceback.format_exc()}")
