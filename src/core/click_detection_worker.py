"""
Background worker that runs the full v5 click detection pipeline on a recording.

Stage 1 → per-candidate features → Stages 2/3/4 (annotated), off the GUI thread.
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
                 model_path=None, k=None, threshold=None, dm=None):
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
        self._stop_requested   = False

    def request_stop(self):
        """Cooperative cancel — checked between candidates."""
        self._stop_requested = True

    def run(self):
        try:
            from core.click_pipeline_v5 import (
                run_stage1_v5, run_stage1_v5_precomputed,
                has_precomputed_stage1_arrays, run_stages234_annotated,
                reconstruct_frame_v5,
                build_click_context, resolve_click, click_event_key,
                compute_features_v5, load_svm_model, K_STAGE1_DEFAULT,
                p_noise_at, p_noise_frames_at,
            )
            from ml import default_model_path

            model_path = self.model_path or default_model_path()
            svm_model  = load_svm_model(model_path)

            k = self.k if self.k is not None else K_STAGE1_DEFAULT

            # Prefer the precomputed arrays: same Stage 1 the Data Collection export
            # runs, so the two features can never report different candidates for the
            # same recording.
            if self.dm is not None and has_precomputed_stage1_arrays(self.dm):
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

                fi = cand['frame_idx']
                try:
                    fd = reconstruct_frame_v5(
                        self.fft_data[fi], self.phase_data[fi],
                        self.fs, self.fft_size, normalize=True,
                    )
                    if fd is None:
                        continue

                    curr_sig = fd['signal']

                    # The previous and next frame SIGNALS build the stitched
                    # context: a click can straddle a frame boundary, and every
                    # feature is computed on that continuous trace.
                    prev_sig = None
                    if fi > 0 and fi - 1 < len(self.phase_data):
                        pf = reconstruct_frame_v5(
                            self.fft_data[fi - 1], self.phase_data[fi - 1],
                            self.fs, self.fft_size, normalize=True,
                        )
                        if pf is not None:
                            prev_sig = pf['signal']

                    next_sig = None
                    if fi + 1 < len(self.fft_data) and fi + 1 < len(self.phase_data):
                        nf = reconstruct_frame_v5(
                            self.fft_data[fi + 1], self.phase_data[fi + 1],
                            self.fs, self.fft_size, normalize=True,
                        )
                        if nf is not None:
                            next_sig = nf['signal']

                    ctx      = build_click_context(prev_sig, curr_sig, next_sig)
                    resolved = resolve_click(ctx, cand['noise_floor'], cand['std_noise'])
                    # p_noise_psd is what SWITCHES THE v6 FAMILY ON. Without it
                    # _feat_v6_spectral returns its NaN skeleton, so every v6 feature
                    # was NaN here while the identical call in the Data Collection
                    # export produced real values — the two paths silently disagreed.
                    # It needs the data manager's Buffer-3 snapshots, so it is None
                    # (and the v6 features honestly NaN) when the caller had no dm.
                    p_noise = p_noise_at(self.dm, fi) if self.dm is not None else None
                    features = compute_features_v5(
                        ctx, resolved,
                        fd['fft_norm'], fd['freq_axis'],
                        cand['noise_floor'], cand['std_noise'], self.fs,
                        p_noise_psd=p_noise,
                    )
                    peak_abs, canonical_frame_idx = click_event_key(ctx, resolved, fi)

                    # The three quality columns that are NOT features and therefore do
                    # not come out of compute_features_v5. Same definitions as
                    # data_collection_dialog_v5, deliberately: a row exported from the
                    # replay window and one exported from the collection dialog must
                    # be the same row.
                    d0 = int(resolved.get('decay_start', 0))
                    d1 = int(resolved.get('decay_end', 0))
                    candidates.append({
                        **cand, **features,
                        'peak_amp': resolved['peak_amp'],
                        'peak_abs': peak_abs,
                        'canonical_frame_idx': canonical_frame_idx,
                        'timestamp_s': fi * self.frame_duration_ms / 1000.0,
                        'decay_len': max(0, d1 - d0),
                        'b3_frames': (p_noise_frames_at(self.dm, fi)
                                      if self.dm is not None else 0),
                        # suppress_edge_artifacts' signature: its fade's first
                        # coefficient is exactly 0, and nothing else in the chain
                        # produces a hard zero at sample 0.
                        'gibbs_fired': int(len(curr_sig) > 0 and curr_sig[0] == 0.0),
                    })

                except Exception as e:  # noqa: BLE001
                    # One bad frame must not abort a whole recording.
                    print(f"⚠️ Click detection: frame {fi} skipped ({e})")

                if n % 10 == 0:
                    self.progress.emit(n, total)

            self.progress.emit(total, total)

            annotated = run_stages234_annotated(
                candidates, svm_model, threshold=self.threshold
            )
            self.finished.emit(annotated)

        except Exception as e:  # noqa: BLE001
            self.error.emit(f"{e}\n{traceback.format_exc()}")
