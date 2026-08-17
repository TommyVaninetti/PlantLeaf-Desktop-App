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
Noise bed extraction — event-free stretches of PlantLeaf's own recordings.

A Dryad clip is 2 ms of anechoic-chamber audio. PlantLeaf's adaptive estimator
needs ~750 frames (~1.92 s) of running context before `noise_floor` and
`std_noise` mean anything, and `peak_SNR` is literally undefined without a floor
to divide by. So a clip cannot be analysed on its own — it has to be transplanted
into a real recording. This module produces the host signal.

Where the events actually are: the recording edges
--------------------------------------------------
Measured by streaming ONE estimator across each file, the way the application
does, and counting post-`MAX_RUN` Stage-1 candidates at k = 1.5 in 2250-frame
windows:

    file        first 2250   last 2250   interior (20/40/60/80 % in)
    ricy mis5            0          14   0, 0, 0, 0
    ricy mis2           10           0   0, 0, 0, 0
    ricy mis6           13           4   201, 0, 0, 0
    tommy mis1           0           0   0, 0, 0, 0

The interiors of these recordings are clean. What trips Stage 1 is the first and
last few seconds — the operator starting and stopping the acquisition, plus the
estimator's own warm-up, during which `is_burst` is forced False
(click_pipeline_v5 §4.4) so genuine start transients are written straight into
the noise buffer.

Hence EDGE_GUARD_FRAMES: window selection excludes both ends outright. Screening
alone is not enough protection there, because `screen_window` only counts
candidates past its own warm-up prefix — a transient inside the first 750 frames
of a window placed at the file's start would escape the candidate count while
still contaminating the noise estimate the injection gain is scaled against.

Screening remains mandatory on top of the guard: ricy mis6 carries a genuine
event region ~20 % into the file, far from either edge. A bed with a real
transient would inject a phantom event next to the Dryad click and silently
poison whatever was trained on it.

Reconstruction uses `frame_emulator.inverse_raw()`, not
`reconstruct_frame_v5()` — see the frame_emulator module docstring for why that
distinction is load-bearing (reconstruct/re-FFT distorts band edges by 100 %).
The bed therefore stays in raw recorded space, which is also the space the
colorized Dryad click is delivered in, so the two can simply be added.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import frame_emulator as fe
from .pipeline_loader import FrameDataManager, compute_stage1_arrays, load_pipeline

# Bed source rooms.
DEFAULT_BED_ROOTS = (
    "/Volumes/Lexar 1TB/PlantLeaf/Audio Experiments/Noise/Stanza Ricy vuota",
    "/Volumes/Lexar 1TB/PlantLeaf/Audio Experiments/Noise/Stanza Tommy vuota",
)

# The estimator's circular buffer is W_NOISE = 750 frames. A bed window must
# carry at least that much lead-in before its first usable injection point,
# otherwise the floor the pipeline reports there is a warm-up artefact.
WARMUP_FRAMES = 750

# Default usable span per window, on top of the warm-up.
DEFAULT_USABLE_FRAMES = 1500        # ~3.84 s

# Frames excluded at BOTH ends of every recording before a window may be drawn.
# The measured contamination sits inside the first and last ~2250 frames (see the
# module docstring); 5000 frames = 12.8 s gives better than 2x margin and costs
# nothing against recordings that run 6-90 minutes.
EDGE_GUARD_FRAMES = 5000

# Spec section 7 regime boundary. Both bed rooms measure well below this; the
# high-floor regime (aloe_6jan, test_aloe_1 at 4.61-4.99 mV) has no empty-room
# counterpart and is not represented here.
TYPICAL_REGIME_MAX_NF_V = 4.0e-3

_ROOM_RE = re.compile(r"Stanza\s+(\w+)", re.IGNORECASE)


@dataclass
class BedWindow:
    """One screened, event-free stretch of PlantLeaf noise, ready for injection."""
    bed_id: str
    source_path: str
    session_id: str
    room: str
    start_frame: int
    n_frames: int
    warmup_frames: int
    signal: np.ndarray          # raw recorded space, warmup_frames*512 samples of lead-in
    noise_floor: float          # V, as the pipeline reports it at the usable region
    std_noise: float            # V
    e_hat_floor: float          # V^2
    candidate_rate: float       # fraction of usable frames tripping Stage 1 (must be 0.0)
    regime: str

    @property
    def fs(self) -> int:
        return fe.FS

    @property
    def usable_start_sample(self) -> int:
        """First sample at which a click may be injected (past estimator warm-up)."""
        return self.warmup_frames * fe.FFT_SIZE

    @property
    def duration_s(self) -> float:
        return len(self.signal) / fe.FS

    def provenance(self) -> dict:
        """Serialisable record for the split-leakage checks (spec section 12)."""
        return {
            "bed_id": self.bed_id,
            "source_path": self.source_path,
            "session_id": self.session_id,
            "room": self.room,
            "start_frame": self.start_frame,
            "n_frames": self.n_frames,
            "warmup_frames": self.warmup_frames,
            "noise_floor_V": self.noise_floor,
            "std_noise_V": self.std_noise,
            "e_hat_floor_V2": self.e_hat_floor,
            "candidate_rate": self.candidate_rate,
            "regime": self.regime,
        }


@dataclass(frozen=True)
class BedSource:
    """A candidate bed recording, before any window has been screened."""
    path: Path
    session_id: str
    room: str
    total_frames: int

    @property
    def duration_s(self) -> float:
        return self.total_frames * fe.FFT_SIZE / fe.FS


def discover_bed_sources(roots=DEFAULT_BED_ROOTS) -> list[BedSource]:
    """
    Enumerate usable .paudio bed recordings under the given directories.

    Skips macOS `._` resource forks and any file too short to yield even one
    window. Missing roots are skipped silently — the beds live on an external
    drive that may not be mounted, and the caller reports that more usefully.
    """
    sources: list[BedSource] = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        room_match = _ROOM_RE.search(root.name)
        room = room_match.group(1).lower() if room_match else root.name.lower()

        for path in sorted(root.glob("*.paudio")):
            if path.name.startswith("."):
                continue
            try:
                header = fe.read_header(path)
            except (ValueError, OSError):
                continue
            if header["version"] < 3.0:
                continue        # magnitude-only; no phase, so no resynthesis
            if header["total_frames"] < (WARMUP_FRAMES + DEFAULT_USABLE_FRAMES
                                         + 2 * EDGE_GUARD_FRAMES):
                continue
            sources.append(BedSource(
                path=path, session_id=path.stem, room=room,
                total_frames=header["total_frames"],
            ))
    return sources


def screen_window(mags: np.ndarray, phases: np.ndarray, *, warmup_frames: int = WARMUP_FRAMES,
                  k: float = None, arrays: dict | None = None) -> dict:
    """
    Run the real Stage 1 over a candidate window and report whether it is clean.

    Uses `click_pipeline_v5.run_stage1_v5_precomputed()` — the path the docstring
    marks authoritative, since every existing candidate CSV and the trained SVM
    came from it — rather than reimplementing the criterion. Screening therefore
    applies exactly the threshold live acquisition applies, run-length filter
    included.

    Candidates inside the warm-up prefix are ignored: before the estimator has
    filled its buffer Ê_floor is unreliable and flags there mean nothing.

    The reported `noise_floor`/`std_noise` are the *median over the usable
    region*, not the value at a single frame — the floor drifts slowly, and the
    median is what the injection gain should be scaled against.

    Pass `arrays` to reuse an existing `compute_stage1_arrays()` result.
    """
    cp = load_pipeline()
    k = cp.K_STAGE1_DEFAULT if k is None else k

    arrays = arrays or compute_stage1_arrays(mags, phases, fs=fe.FS, fft_size=fe.FFT_SIZE)
    dm = FrameDataManager(mags, phases, fs=fe.FS, fft_size=fe.FFT_SIZE)
    dm.attach_stage1_arrays(arrays)

    candidates = cp.run_stage1_v5_precomputed(dm, k=k)
    late = [c for c in candidates if c["frame_idx"] >= warmup_frames]
    n_usable = max(1, len(mags) - warmup_frames)

    usable = slice(warmup_frames, len(mags))
    return {
        "n_frames": len(mags),
        "n_usable": n_usable,
        "n_candidates": len(late),
        "candidate_rate": len(late) / n_usable,
        "noise_floor": float(np.median(arrays["noise_floor_arr"][usable])),
        "std_noise": float(np.median(arrays["std_noise_arr"][usable])),
        "e_hat_floor": float(np.median(arrays["E_hat_floor_arr"][usable])),
        "clean": len(late) == 0,
        "k": float(k),
        "arrays": arrays,
    }


def extract_window(source: BedSource, start_frame: int, *,
                   usable_frames: int = DEFAULT_USABLE_FRAMES,
                   warmup_frames: int = WARMUP_FRAMES,
                   k: float = None) -> BedWindow | None:
    """
    Screen one specific window and, if clean, reconstruct it to a waveform.

    Returns None when the window trips Stage 1 anywhere past warm-up, or when it
    runs off the end of the file. Rejection is the common case for some
    recordings and is not an error.
    """
    n_frames = warmup_frames + usable_frames
    if start_frame < 0 or start_frame + n_frames > source.total_frames:
        return None

    mags, phases = fe.read_frames(source.path, start_frame, n_frames)
    if len(mags) < n_frames:
        return None

    stats = screen_window(mags, phases, warmup_frames=warmup_frames, k=k)
    if not stats["clean"]:
        return None

    signal = fe.signal_from_frames(mags, phases)
    nf = stats["noise_floor"]

    return BedWindow(
        bed_id=f"{source.session_id}@{start_frame}",
        source_path=str(source.path),
        session_id=source.session_id,
        room=source.room,
        start_frame=start_frame,
        n_frames=n_frames,
        warmup_frames=warmup_frames,
        signal=signal,
        noise_floor=nf,
        std_noise=stats["std_noise"],
        e_hat_floor=stats["e_hat_floor"],
        candidate_rate=stats["candidate_rate"],
        regime="typical" if nf <= TYPICAL_REGIME_MAX_NF_V else "high_floor",
    )


def find_clean_window(source: BedSource, rng: np.random.Generator, *,
                      usable_frames: int = DEFAULT_USABLE_FRAMES,
                      warmup_frames: int = WARMUP_FRAMES,
                      max_attempts: int = 12, k: float = None,
                      edge_guard: int = EDGE_GUARD_FRAMES) -> BedWindow | None:
    """
    Draw random windows from one recording until a clean one is found.

    Random placement (rather than walking from the start) keeps beds spread
    across each session, so a bed's specific texture at one point in time does
    not dominate the synthetic set — the bed-level leakage concern of spec
    section 12.4, one level down from session splitting.

    Draws are confined to `[edge_guard, total - n_frames - edge_guard]`. The
    start and stop transients of a recording live outside that range and are the
    only place most of these files trip Stage 1 at all (module docstring).

    Returns None if every attempt trips Stage 1 — a real outcome for a session
    with a genuine event region, e.g. ricy mis6.
    """
    n_frames = warmup_frames + usable_frames
    lowest_start = edge_guard
    highest_start = source.total_frames - n_frames - edge_guard
    if highest_start < lowest_start:
        # Recording too short for the guard; fall back to the full range rather
        # than discarding it, and let screening do the work.
        lowest_start, highest_start = 0, source.total_frames - n_frames
    if highest_start < lowest_start:
        return None

    for _ in range(max_attempts):
        start = int(rng.integers(lowest_start, highest_start + 1))
        window = extract_window(source, start, usable_frames=usable_frames,
                                warmup_frames=warmup_frames, k=k)
        if window is not None:
            return window
    return None


def build_bed_pool(sources: list[BedSource], n_beds: int, rng: np.random.Generator, *,
                   usable_frames: int = DEFAULT_USABLE_FRAMES,
                   warmup_frames: int = WARMUP_FRAMES,
                   k: float = None, progress=None) -> list[BedWindow]:
    """
    Build a diverse pool of clean beds, round-robin across sessions.

    Round-robin rather than "fill from the first file" is deliberate: spec
    section 8.3 requires beds drawn from as many distinct sessions/rooms/times as
    available, because a single reused bed leaks its particular hum, resonance
    and transient pattern into every click injected on top of it.

    Sessions that repeatedly fail screening drop out of the rotation instead of
    stalling the pool.
    """
    if not sources:
        raise ValueError("no bed sources available (is the external drive mounted?)")

    pool: list[BedWindow] = []
    live = list(sources)
    while len(pool) < n_beds and live:
        for source in list(live):
            if len(pool) >= n_beds:
                break
            window = find_clean_window(source, rng, usable_frames=usable_frames,
                                       warmup_frames=warmup_frames, k=k)
            if window is None:
                live.remove(source)
                continue
            pool.append(window)
            if progress is not None:
                progress(window)

    if not pool:
        raise RuntimeError(
            "no event-free bed window found in any source. Every candidate window "
            "tripped Stage 1 — check the recordings or relax `usable_frames`."
        )
    return pool
