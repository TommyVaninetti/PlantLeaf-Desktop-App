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
Firmware frame convention: time domain <-> transmitted FFT frames, and .paudio I/O.

`forward()` reproduces exactly what the STM32 firmware does to a 512-sample
block, and `inverse_raw()` undoes it exactly. Together they let an arbitrary
time-domain waveform be written as a .paudio that the live pipeline cannot
distinguish from a genuine recording.


Why inverse_raw() and not reconstruct_frame_v5()
------------------------------------------------
`reconstruct_frame_v5()` is the right function for ANALYSIS — it is what the
detector and every feature use, and it deliberately does four things beyond the
inverse FFT: microphone normalization, a spectral Tukey taper on the band edges,
the x N/2 amplitude correction, and Gibbs edge suppression.

Every one of those is wrong for RESYNTHESIS. Reconstructing a bed with it and
then re-running `forward()` applies the taper twice and the normalization twice,
and the pipeline then normalizes a third time when it reads the file back.
Measured on 300 real bed frames, that path distorts magnitudes by 41 % in the
median and by 100 % at both band edges — the edges are annihilated.

`inverse_raw()` therefore does only the strictly invertible part: zero-pad the
transmitted bins into a 256-bin array, decode the int8 phase, undo the firmware's
2/N scaling, inverse FFT. Nothing else. Verified over 300 real bed frames:

    magnitude, max relative error : 4.6e-14
    phase, max circular distance  : 1 LSB (0.0245 rad), on 0.065 % of bins

That is an exact round trip on the same 512-sample grid; the 1-LSB residue is
int8 wraparound at +-pi, not loss. It gives a very sharp regression assertion:
a zero-gain injection must reproduce its source bed bin for bin.

Note the round trip is exact *per frame on a fixed grid*. Concatenated frames do
not form a continuous band-limited signal — each was independently band-limited,
so there are small discontinuities at the joins. That is fine and in fact
faithful: it is exactly the signal the real pipeline reconstructs and analyses.
What matters is that re-framing on the SAME grid recovers each frame exactly.
"""

from __future__ import annotations

import struct
import time
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Format constants — must match the firmware and src/saving/audio_load_progress.py
# ─────────────────────────────────────────────────────────────────────────────

FS = 200_000
FFT_SIZE = 512
FREQ_MIN_HZ = 20_000
FREQ_MAX_HZ = 80_000

BIN_FREQ = FS / FFT_SIZE                      # 390.625 Hz
BIN_START = int(FREQ_MIN_HZ / BIN_FREQ)       # 51
BIN_END = int(FREQ_MAX_HZ / BIN_FREQ)         # 204
N_BINS = BIN_END - BIN_START + 1              # 154

# The firmware keeps FFT_SIZE//2 = 256 bins, NOT the mathematical N/2+1 = 257.
# The Nyquist bin is simply never transmitted. np.fft.irfft accepts a 256-length
# input with n=512 because it only needs n//2+1 entries and treats the rest as
# absent, so this stays consistent end to end.
HALF_BINS = FFT_SIZE // 2                     # 256

# Phase is int8 with 128 counts per pi (firmware LUT: round(angle * 128/pi)).
# The divisor is 128, not 127.
PHASE_SCALE = 128.0

HEADER_SIZE = 128
RECORD_SIZE = 5                               # float32 magnitude + int8 phase
FRAME_BYTES = N_BINS * RECORD_SIZE            # 770
PAUDIO_MAGIC = b"PLANTAUDIO"
PAUDIO_VERSION = 3.0                          # 3.0 = magnitude + phase

_FRAME_DTYPE = np.dtype([("m", "<f4"), ("p", "i1")])


# ─────────────────────────────────────────────────────────────────────────────
# Forward: time domain -> transmitted frame
# ─────────────────────────────────────────────────────────────────────────────

def forward(signal: np.ndarray, fft_size: int = FFT_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """
    Emulate the firmware's per-frame processing of one 512-sample block.

        real FFT  ->  keep bins [0, 256)  ->  scale by 2/N  ->  quantise phase
        ->  keep the analysis band, bins 51..204

    The 2/N factor is the firmware's own normalization (FFT_NORMALIZATION_FACTOR_NORMAL
    in main_with_phase.c), applied after the ADC counts have already been
    converted to volts. Transmitted magnitudes are therefore amplitudes in volts,
    which is what makes `inverse_raw`'s x N/2 the correct counterpart.

    Parameters
    ----------
    signal : np.ndarray
        Exactly `fft_size` samples in volts. Shorter input is zero-padded, which
        is only correct at the very end of a recording.

    Returns
    -------
    (magnitudes, phases)
        float64 magnitudes [V] and int8 phases, both length 154.
    """
    x = np.asarray(signal, dtype=np.float64)
    if len(x) < fft_size:
        x = np.pad(x, (0, fft_size - len(x)))
    elif len(x) > fft_size:
        raise ValueError(f"frame is {len(x)} samples, expected at most {fft_size}")

    spectrum = np.fft.rfft(x)[:fft_size // 2]

    mags = np.abs(spectrum) * (2.0 / fft_size)
    phases = np.clip(np.round(np.angle(spectrum) * PHASE_SCALE / np.pi), -128, 127).astype(np.int8)

    return mags[BIN_START:BIN_END + 1], phases[BIN_START:BIN_END + 1]


def frames_from_signal(signal: np.ndarray, fft_size: int = FFT_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a continuous waveform into consecutive frames and forward-transform each.

    A trailing partial frame is dropped rather than zero-padded: a padded frame
    has a spectrum unlike anything the hardware produces, and the noise estimator
    would see it as an anomaly.

    Returns (magnitudes, phases) with shape (n_frames, 154).
    """
    x = np.asarray(signal, dtype=np.float64)
    n_frames = len(x) // fft_size
    if n_frames == 0:
        raise ValueError(f"signal of {len(x)} samples is shorter than one {fft_size}-sample frame")

    mags = np.empty((n_frames, N_BINS), dtype=np.float64)
    phases = np.empty((n_frames, N_BINS), dtype=np.int8)
    for i in range(n_frames):
        mags[i], phases[i] = forward(x[i * fft_size:(i + 1) * fft_size], fft_size)
    return mags, phases


# ─────────────────────────────────────────────────────────────────────────────
# Inverse: transmitted frame -> time domain (exact)
# ─────────────────────────────────────────────────────────────────────────────

def inverse_raw(mags: np.ndarray, phases: np.ndarray, fft_size: int = FFT_SIZE) -> np.ndarray:
    """
    Exact inverse of `forward()`. No taper, no normalization, no Gibbs suppression.

    This returns the signal in RAW RECORDED SPACE — the pre-normalization domain
    the firmware stored. Use it to resynthesise a waveform that will be written
    back out as .paudio. For anything analytical, use
    `click_pipeline_v5.reconstruct_frame_v5()` instead: this function
    deliberately omits the corrections that make a signal suitable for feature
    extraction.

    The x fft_size/2 factor undoes the firmware's 2/N scaling. Without it,
    `np.fft.irfft`'s own 1/N would be applied on top of the firmware's, leaving
    the signal N/2 = 256x too small (see IFFT_AMPLITUDE_SCALE_FIX.md).

    Returns `fft_size` samples in volts.
    """
    mags = np.asarray(mags, dtype=np.float64)
    phases = np.asarray(phases)
    n = min(len(mags), len(phases), N_BINS)

    full_mag = np.zeros(fft_size // 2, dtype=np.float64)
    full_phase = np.zeros(fft_size // 2, dtype=np.float64)
    full_mag[BIN_START:BIN_START + n] = mags[:n]
    full_phase[BIN_START:BIN_START + n] = phases[:n]

    spectrum = full_mag * np.exp(1j * (full_phase / PHASE_SCALE) * np.pi)
    spectrum *= (fft_size / 2.0)

    return np.fft.irfft(spectrum, n=fft_size)


def signal_from_frames(mags: np.ndarray, phases: np.ndarray, fft_size: int = FFT_SIZE) -> np.ndarray:
    """
    Inverse-transform consecutive frames and concatenate them in acquisition order.

    Produces one continuous waveform of `n_frames * fft_size` samples, aligned to
    the frame grid so that `frames_from_signal()` on the result recovers the
    input exactly.
    """
    mags = np.atleast_2d(np.asarray(mags, dtype=np.float64))
    phases = np.atleast_2d(np.asarray(phases))
    if len(mags) != len(phases):
        raise ValueError(f"{len(mags)} magnitude frames vs {len(phases)} phase frames")

    out = np.empty(len(mags) * fft_size, dtype=np.float64)
    for i in range(len(mags)):
        out[i * fft_size:(i + 1) * fft_size] = inverse_raw(mags[i], phases[i], fft_size)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# .paudio reading
# ─────────────────────────────────────────────────────────────────────────────

def read_header(path: str | Path) -> dict:
    """
    Parse the 128-byte .paudio header. Mirrors src/saving/audio_load_progress.py.
    """
    path = Path(path)
    with open(path, "rb") as f:
        raw = f.read(HEADER_SIZE)
    if len(raw) < HEADER_SIZE:
        raise ValueError(f"{path.name}: file shorter than a {HEADER_SIZE}-byte header")

    magic = raw[0:10].rstrip(b"\x00")
    if not magic.startswith(b"PLANTAUD"):
        raise ValueError(f"{path.name}: bad magic {magic!r}, not a .paudio file")

    fs = struct.unpack("<I", raw[34:38])[0]
    fft_size = struct.unpack("<I", raw[38:42])[0]
    freq_min = struct.unpack("<I", raw[42:46])[0]
    freq_max = struct.unpack("<I", raw[46:50])[0]

    bin_freq = fs / fft_size
    n_bins = int(freq_max / bin_freq) - int(freq_min / bin_freq) + 1

    return {
        "path": str(path),
        "magic": magic.decode("ascii", errors="replace"),
        "version": struct.unpack("<f", raw[10:14])[0],
        "experiment_type": raw[14:34].rstrip(b"\x00 ").decode("ascii", errors="replace"),
        "fs": fs,
        "fft_size": fft_size,
        "freq_min": freq_min,
        "freq_max": freq_max,
        "threshold": struct.unpack("<f", raw[50:54])[0],
        "start_time": struct.unpack("<d", raw[54:62])[0],
        "end_time": struct.unpack("<d", raw[62:70])[0],
        "data_points": struct.unpack("<I", raw[70:74])[0],
        "acquisition_count": struct.unpack("<I", raw[74:78])[0],
        "n_bins": n_bins,
        "frame_bytes": n_bins * RECORD_SIZE,
        "total_frames": (path.stat().st_size - HEADER_SIZE) // (n_bins * RECORD_SIZE),
        "file_size": path.stat().st_size,
    }


def read_frames(path: str | Path, start_frame: int = 0, count: int | None = None,
                header: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Read a contiguous span of frames without loading the whole file.

    Files here run to 1.6 GB, so seeking to the requested offset and reading only
    what is needed is not an optimisation but a requirement. Uses a structured
    dtype rather than per-sample `struct.unpack`, which is ~100x faster.

    Returns (magnitudes, phases), shapes (n, n_bins), dtypes float64 / int8.

    Only the current v3.0 layout (no inter-frame separators) is handled. The
    legacy NaN-separated v3.0 and magnitude-only v2.0 variants that
    audio_load_progress.py auto-detects are not supported — this module is for
    the beds and synthetic files we control.
    """
    header = header or read_header(path)
    n_bins = header["n_bins"]
    frame_bytes = n_bins * RECORD_SIZE

    if header["version"] < 3.0:
        raise ValueError(
            f"{Path(path).name}: version {header['version']} is magnitude-only; "
            "phase is required for resynthesis"
        )

    total = header["total_frames"]
    if start_frame < 0 or start_frame >= total:
        raise ValueError(f"start_frame {start_frame} out of range (file has {total} frames)")
    count = total - start_frame if count is None else min(count, total - start_frame)

    with open(path, "rb") as f:
        f.seek(HEADER_SIZE + start_frame * frame_bytes)
        buf = f.read(frame_bytes * count)

    n = len(buf) // frame_bytes
    arr = np.frombuffer(buf[:n * frame_bytes], dtype=_FRAME_DTYPE)
    mags = np.ascontiguousarray(arr["m"].reshape(n, n_bins), dtype=np.float64)
    phases = np.ascontiguousarray(arr["p"].reshape(n, n_bins), dtype=np.int8)
    return mags, phases


# ─────────────────────────────────────────────────────────────────────────────
# .paudio writing
# ─────────────────────────────────────────────────────────────────────────────

def build_header(n_frames: int, *, experiment_type: str = "Dryad Hybrid",
                 fs: int = FS, fft_size: int = FFT_SIZE,
                 freq_min: int = FREQ_MIN_HZ, freq_max: int = FREQ_MAX_HZ,
                 threshold: float = 0.03, start_time: float | None = None,
                 end_time: float | None = None, acquisition_count: int = 0) -> bytes:
    """
    Build the 128-byte header. Byte-for-byte identical layout to
    `MainWindowAudio._create_header()`.
    """
    start_time = time.time() if start_time is None else start_time
    end_time = (start_time + n_frames * fft_size / fs) if end_time is None else end_time

    out = bytearray()
    out.extend(PAUDIO_MAGIC[:10].ljust(10, b"\x00"))                        # [0:10]
    out.extend(struct.pack("<f", PAUDIO_VERSION))                           # [10:14]
    out.extend(experiment_type.encode("ascii", errors="replace")[:20].ljust(20, b"\x00"))  # [14:34]
    out.extend(struct.pack("<I", int(fs)))                                  # [34:38]
    out.extend(struct.pack("<I", int(fft_size)))                            # [38:42]
    out.extend(struct.pack("<I", int(freq_min)))                            # [42:46]
    out.extend(struct.pack("<I", int(freq_max)))                            # [46:50]
    out.extend(struct.pack("<f", float(threshold)))                         # [50:54]
    out.extend(struct.pack("<d", float(start_time)))                        # [54:62]
    out.extend(struct.pack("<d", float(end_time)))                          # [62:70]
    out.extend(struct.pack("<I", int(n_frames)))                            # [70:74]
    out.extend(struct.pack("<I", int(acquisition_count)))                   # [74:78]
    out.extend(b"\x00" * 50)                                                # [78:128]

    if len(out) != HEADER_SIZE:
        raise ValueError(f"header is {len(out)} bytes, expected {HEADER_SIZE}")
    return bytes(out)


def _sanitise_clck(payload: bytearray) -> int:
    """
    Ensure the frame payload contains no literal b'CLCK'.

    The loader locates the optional trailing click section with
    `remaining_data.find(b'CLCK')` and treats everything before the first hit as
    FFT data. A chance occurrence of those four bytes inside the magnitude/phase
    stream therefore truncates the recording at that point.

    This is not hypothetical at our scale: the odds are ~2^-32 per byte offset,
    and this run writes on the order of 3 GB, so roughly one collision is
    expected across the whole output. (The same latent hazard exists for genuine
    recordings; fixing the loader is out of scope here.)

    Repair: flip the lowest bit of a mantissa byte that lies INSIDE the matched
    span, which is what guarantees the pattern is broken. Of the four matched
    bytes, one at most is a phase byte (they occur every 5th), so a magnitude
    byte is always available. We take the one with the lowest index within its
    float, preferring byte 0 (a 1-ULP change, ~6e-8 relative). Only when the
    match starts at record offset 1 is byte 0 unavailable, and byte 1 is used
    instead: a 2^8-ULP change, still ~2.4e-5 relative, i.e. ~2e-8 V on a ~1e-3 V
    magnitude — orders of magnitude below the 3 mV noise floor and below
    anything downstream can resolve. Bytes 2 and 3 carry the exponent and are
    never touched.

    Returns the number of collisions repaired.
    """
    repaired = 0
    idx = payload.find(b"CLCK")
    while idx >= 0:
        # Among the four matched byte positions, find the one sitting earliest
        # within its float record (position % RECORD_SIZE == 4 is the phase byte).
        target, best_in_float = None, None
        for pos in range(idx, idx + 4):
            in_float = pos % RECORD_SIZE
            if in_float >= 4:
                continue                    # phase byte, not ours to perturb
            if best_in_float is None or in_float < best_in_float:
                target, best_in_float = pos, in_float

        if target is None:                  # unreachable: 4 consecutive bytes
            raise RuntimeError("no magnitude byte inside a 4-byte span")

        payload[target] ^= 0x01
        repaired += 1
        if repaired > 1000:
            raise RuntimeError("runaway b'CLCK' sanitisation; payload is likely malformed")

        # Re-scan from just before the repair: the flip cannot recreate the
        # pattern at `idx`, but could in principle form one overlapping it.
        idx = payload.find(b"CLCK", max(0, idx - 3))
    return repaired


def write_paudio(path: str | Path, mags: np.ndarray, phases: np.ndarray,
                 *, experiment_type: str = "Dryad Hybrid", threshold: float = 0.03,
                 start_time: float | None = None, fs: int = FS,
                 fft_size: int = FFT_SIZE) -> dict:
    """
    Write frames as a v3.0 .paudio the application can open like any recording.

    Layout: 128-byte header, then per bin a float32 magnitude followed by an int8
    phase, 154 bins per frame, 770 bytes per frame, no separators and no trailing
    click section.

    Returns a summary dict including `clck_collisions_repaired` (see
    `_sanitise_clck`), which should normally be 0.
    """
    path = Path(path)
    mags = np.atleast_2d(np.asarray(mags, dtype=np.float32))
    phases = np.atleast_2d(np.asarray(phases, dtype=np.int8))

    if mags.shape != phases.shape:
        raise ValueError(f"magnitude shape {mags.shape} != phase shape {phases.shape}")
    if mags.shape[1] != N_BINS:
        # audio_load_progress.py hardcodes samples_per_fft = 154; anything else
        # is silently misparsed rather than rejected.
        raise ValueError(f"expected {N_BINS} bins per frame, got {mags.shape[1]}")
    if not np.all(np.isfinite(mags)):
        # A NaN magnitude paired with phase -1 is the legacy frame separator, so
        # a stray NaN would flip the loader into separator mode.
        raise ValueError("magnitudes contain NaN or inf")

    n_frames = mags.shape[0]
    records = np.empty(mags.size, dtype=_FRAME_DTYPE)
    records["m"] = mags.reshape(-1)
    records["p"] = phases.reshape(-1)

    payload = bytearray(records.tobytes())
    repaired = _sanitise_clck(payload)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "wb") as f:
        f.write(build_header(n_frames, experiment_type=experiment_type, fs=fs,
                             fft_size=fft_size, threshold=threshold, start_time=start_time))
        f.write(payload)
    tmp.replace(path)   # atomic: a crashed run never leaves a half-written .paudio

    return {
        "path": str(path),
        "n_frames": n_frames,
        "duration_s": n_frames * fft_size / fs,
        "bytes": HEADER_SIZE + len(payload),
        "clck_collisions_repaired": repaired,
    }
