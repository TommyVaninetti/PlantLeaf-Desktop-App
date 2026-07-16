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
                reconstruct_frame_v5, compute_hilbert_envelope, find_peak,
                compute_features_v5, load_svm_model, K_STAGE1_DEFAULT,
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
                    curr_env = compute_hilbert_envelope(curr_sig)
                    peak_idx, peak_amp = find_peak(curr_env)

                    # The previous and next frames are needed for the pre/post
                    # windows: a click can straddle a frame boundary.
                    prev_env, prev_sig = None, None
                    if fi > 0 and fi - 1 < len(self.phase_data):
                        pf = reconstruct_frame_v5(
                            self.fft_data[fi - 1], self.phase_data[fi - 1],
                            self.fs, self.fft_size, normalize=True,
                        )
                        if pf is not None:
                            prev_sig = pf['signal']
                            prev_env = compute_hilbert_envelope(prev_sig)

                    next_env = None
                    if fi + 1 < len(self.fft_data) and fi + 1 < len(self.phase_data):
                        nf = reconstruct_frame_v5(
                            self.fft_data[fi + 1], self.phase_data[fi + 1],
                            self.fs, self.fft_size, normalize=True,
                        )
                        if nf is not None:
                            next_env = compute_hilbert_envelope(nf['signal'])

                    features = compute_features_v5(
                        signal=curr_sig, envelope=curr_env,
                        fft_norm=fd['fft_norm'], freq_axis=fd['freq_axis'],
                        noise_floor=cand['noise_floor'], std_noise=cand['std_noise'],
                        peak_idx=peak_idx, fs=self.fs,
                        next_frame_envelope=next_env,
                        prev_frame_envelope=prev_env, prev_frame_signal=prev_sig,
                    )

                    candidates.append({
                        **cand, **features,
                        'peak_amp': peak_amp,
                        'timestamp_s': fi * self.frame_duration_ms / 1000.0,
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
