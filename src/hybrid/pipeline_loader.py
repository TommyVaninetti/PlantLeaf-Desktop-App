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
Qt-free access to the analysis modules that live under src/core/.

Why this exists
---------------
`src/core/__init__.py` imports the application windows, so the obvious

    from core.click_pipeline_v5 import reconstruct_frame_v5

drags in PySide6. That is fatal here for three reasons: this engine runs under
`multiprocessing`, it runs headless in CI/batch contexts with no display, and
`click_pipeline_v5` itself avoids scipy specifically because BLAS calls segfault
inside QThreads on macOS — importing Qt would reintroduce exactly the
environment that motivated that workaround.

Loading the module files directly by path bypasses the package `__init__`
entirely, so no Qt is touched. The pattern is the one already proven in
`test_scripts/verify_ifft_scale_fix.py`; it is centralised here so the loading
trick appears exactly once in the codebase rather than being copy-pasted into
every script that needs the pipeline.

Usage
-----
    from hybrid.pipeline_loader import load_pipeline, load_spectral

    cp = load_pipeline()          # the click_pipeline_v5 module object
    sa = load_spectral()          # the spectral_analysis module object

    frame = cp.reconstruct_frame_v5(mags, phases, cp.FS, cp.FFT_SIZE)

Both loaders memoise, so repeated calls are free and every caller in a process
shares one module instance (which matters: `click_pipeline_v5` builds a Gaussian
kernel and several derived constants at import time).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

# src/hybrid/pipeline_loader.py  ->  src/
_SRC_DIR = Path(__file__).resolve().parent.parent
_CORE_DIR = _SRC_DIR / "core"

_CACHE: dict[str, ModuleType] = {}


def _load_by_path(module_name: str, file_path: Path) -> ModuleType:
    """
    Import a single .py file as a top-level module, without touching its package.

    The module is registered in `sys.modules` under `module_name` so that
    pickling (multiprocessing) and repeated imports resolve to the same object.
    """
    if module_name in _CACHE:
        return _CACHE[module_name]

    # Respect an already-imported instance (e.g. the app itself loaded it).
    existing = sys.modules.get(module_name)
    if existing is not None and getattr(existing, "__file__", None):
        if Path(existing.__file__).resolve() == file_path.resolve():
            _CACHE[module_name] = existing
            return existing

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Cannot load '{module_name}': {file_path} does not exist. "
            f"Expected the PlantLeaf source tree at {_SRC_DIR}."
        )

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build an import spec for {file_path}")

    module = importlib.util.module_from_spec(spec)
    # Register BEFORE exec so a self-referential import inside the module
    # resolves to this same partially-initialised object instead of recursing.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    _CACHE[module_name] = module
    return module


def load_pipeline() -> ModuleType:
    """
    Return the `click_pipeline_v5` module (the v5 detection/feature pipeline).

    Provides, among others: FS, FFT_SIZE, _BIN_START, _BIN_END, _K_BINS,
    reconstruct_frame_v5, build_click_context, resolve_click,
    compute_features_v5, compute_hilbert_envelope, compute_fft_energy,
    AdaptiveNoiseEstimatorV5, K_STAGE1_DEFAULT, W_NOISE.
    """
    return _load_by_path("click_pipeline_v5", _CORE_DIR / "click_pipeline_v5.py")


def load_spectral() -> ModuleType:
    """
    Return the `spectral_analysis` module (the Region-FFT maths).

    Provides: compute_spectrum, make_window, to_db, band_descriptors, nenbw,
    default_nfft, Spectrum.
    """
    return _load_by_path("spectral_analysis", _CORE_DIR / "spectral_analysis.py")


class FrameDataManager:
    """
    Minimal stand-in for the app's AudioDataManager, backed by in-memory frames.

    Stage 1 needs an object exposing `.fft_data`, `.phase_data`, `.header_info`
    and `.total_frames`. Providing that shim lets us call the real Stage 1 on
    synthetic and bed frames instead of reimplementing its criterion.

    When the per-frame arrays are attached (via `attach_stage1_arrays`, normally
    through `compute_stage1_arrays`), `run_stage1_v5_precomputed()` becomes
    available. Prefer it: its docstring notes it is the *authoritative* path —
    every candidate CSV and the trained SVM came from it, because
    AudioLoadWorker computes E_i from the mic-normalized spectrum whereas the
    non-precomputed `run_stage1_v5` uses raw magnitudes as a fallback, and the
    two can select slightly different candidate sets.
    """

    def __init__(self, mags, phases, fs: int = 200_000, fft_size: int = 512):
        if len(mags) != len(phases):
            raise ValueError(f"{len(mags)} magnitude frames vs {len(phases)} phase frames")
        self.fft_data = mags
        self.phase_data = phases
        self.header_info = {"fs": int(fs), "fft_size": int(fft_size)}
        self.total_frames = len(mags)
        self.fft_means = None
        self.E_hat_floor_arr = None
        self.noise_floor_arr = None
        self.std_noise_arr = None
        # v6 Buffer 3. Named exactly as AudioLoadWorker names them, because
        # click_pipeline_v5.p_noise_at()/p_noise_frames_at() read them off the
        # data manager by getattr — matching the names is what lets the hybrid
        # path reuse those helpers instead of growing a second implementation.
        self.p_noise_snapshots = None
        self.p_noise_stride = None
        self.p_noise_counts = None

    def attach_stage1_arrays(self, arrays: dict) -> "FrameDataManager":
        """Attach the per-frame arrays from `compute_stage1_arrays`. Chainable."""
        self.fft_means = arrays["fft_means"]
        self.E_hat_floor_arr = arrays["E_hat_floor_arr"]
        self.noise_floor_arr = arrays["noise_floor_arr"]
        self.std_noise_arr = arrays["std_noise_arr"]
        self.p_noise_snapshots = arrays.get("p_noise_snapshots")
        self.p_noise_stride = arrays.get("p_noise_stride")
        self.p_noise_counts = arrays.get("p_noise_counts")
        return self


def compute_stage1_arrays(mags, phases, fs: int = 200_000, fft_size: int = 512) -> dict:
    """
    Reproduce, in one pass, the per-frame arrays AudioLoadWorker builds at load time.

    Mirrors `src/saving/audio_load_progress.py` exactly: E_i is the energy of the
    **mic-normalized** in-band spectrum (not the raw magnitudes), and the
    estimator is fed the Hilbert envelope mean/std of the normalized
    reconstruction. Matching that is what makes downstream Stage 1 results
    comparable with every existing candidate CSV.

    Returns the four arrays plus `envelopes_mean`/`envelopes_std` for callers
    that want them, and the final estimator state under `final`.

    One pass, not two: the arrays feed `run_stage1_v5_precomputed` for the
    threshold test *and* carry the noise floor at every frame, so there is no
    need to re-walk the recording to read the floor at a particular point.

    Buffer 3 (v6)
    -------------
    `mags_norm` is passed to `est.update()` so the per-bin noise PSD actually
    fills. Without it `est.p_noise_psd()` returns None for the whole recording,
    `compute_features_v5` takes its `p_noise_psd is None` early return, and every
    v6 region feature — `harmonic_confinement` included — comes out NaN. Since
    `_stage2_reason` treats NaN as *pass*, that failure is silent: the v6 gates
    would simply never fire and the pass rate would look better than it is.

    The slice handed over is the mic-normalized analysis band, the same one E_i
    is computed from — trap (b) of v6 §4.4; raw magnitudes would corrupt every v6
    feature without any visible symptom.

    B3 is sampled on a stride rather than stored per frame, exactly as
    AudioLoadWorker does it (`SUBWINDOW_SIZE`, float32), so that
    `cp.p_noise_at(dm, i)` reads it back unchanged. Staleness is bounded: between
    two snapshots at most SUBWINDOW_SIZE of W_NOISE = 750 entries rotate.
    """
    import numpy as np

    cp = load_pipeline()
    n = len(mags)

    fft_means = np.zeros(n, dtype=np.float64)
    e_hat = np.zeros(n, dtype=np.float64)
    noise_floor = np.zeros(n, dtype=np.float64)
    std_noise = np.zeros(n, dtype=np.float64)
    env_mean = np.zeros(n, dtype=np.float64)
    env_std = np.zeros(n, dtype=np.float64)

    stride = int(cp.SUBWINDOW_SIZE)
    n_snap = (n + stride - 1) // stride
    p_noise_snapshots = np.full((n_snap, cp._K_BINS), np.nan, dtype=np.float32)
    p_noise_counts = np.zeros(n_snap, dtype=np.int32)

    est = cp.AdaptiveNoiseEstimatorV5()
    last = {"E_hat_floor": 0.0, "noise_floor": 0.0, "std_noise": 0.0}

    for i in range(n):
        frame = cp.reconstruct_frame_v5(mags[i], phases[i], fs, fft_size, normalize=True)
        if frame is None:
            fft_means[i] = cp.compute_fft_energy(mags[i])
            e_hat[i], noise_floor[i], std_noise[i] = (
                last["E_hat_floor"], last["noise_floor"], last["std_noise"])
            continue

        band = frame["fft_norm"][cp._BIN_START:cp._BIN_END + 1]
        e_i = cp.compute_fft_energy(band)
        envelope = cp.compute_hilbert_envelope(frame["signal"])
        m, s = float(np.mean(envelope)), float(np.std(envelope))

        last = est.update(e_i, m, s, mags_norm=band, fs=fs, fft_size=fft_size)
        fft_means[i] = e_i
        env_mean[i], env_std[i] = m, s
        e_hat[i] = last["E_hat_floor"]
        noise_floor[i] = last["noise_floor"]
        std_noise[i] = last["std_noise"]

        if i % stride == 0:
            pn = est.p_noise_psd()
            if pn is not None:
                p_noise_snapshots[i // stride] = pn
                p_noise_counts[i // stride] = est.b3_window_frames

    return {
        "fft_means": fft_means,
        "E_hat_floor_arr": e_hat,
        "noise_floor_arr": noise_floor,
        "std_noise_arr": std_noise,
        "envelopes_mean": env_mean,
        "envelopes_std": env_std,
        "p_noise_snapshots": p_noise_snapshots,
        "p_noise_stride": stride,
        "p_noise_counts": p_noise_counts,
        "final": last,
    }
