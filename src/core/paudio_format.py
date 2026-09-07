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
.paudio ON-DISK FORMAT — the one place the byte layout is written down.

    header   128 B                     PLANTAUDIO magic, version, metadata
    body     n_frames x 770 B          154 x (float32 magnitude + int8 phase)
    CLCK     b'CLCK' + uint32 + zlib   click events, JSON
    EVNT     b'EVNT' + uint32 + N*21   v4 only: per-frame event metadata

v3.0 is a CONTINUOUS recording: body frame i is the i-th frame of the signal,
so its time is i * fft_size / fs.

v4.0 is an EVENT recording made by firmware v3 in event mode. Only click
candidates and their immediate neighbours were transmitted, so body frame i is
NOT the i-th frame of the signal — its position in time is in the EVNT footer.

WHY THE BODY DID NOT CHANGE

The obvious design widens the record to carry its own metadata. It was not
chosen. The 154-bin / 5-byte / 770-byte frame is hand-duplicated in six
independent readers and writers across this repo (audio_load_progress,
audio_save_worker plus main_window_audio's inline re-reader, audio_trim_export,
hybrid/frame_emulator, main_window_chemical_simulator) and in three test-script
generators, none of them sharing code. Widening the record means changing all
of them together or silently corrupting whatever is missed.

Keeping the body identical and putting the new data in a footer means a v3-only
reader locates the end of the body exactly as it always did, with find(b'CLCK'),
and ignores everything after. That is why v4 ALWAYS writes a CLCK block, even
an empty one: without it those readers would run off the end of the body and
read EVNT bytes as magnitudes.

They would still lay a non-contiguous recording out as if it were continuous —
wrong in a way nobody would notice — so `check_version` exists for them to
refuse a v4 file outright rather than mis-time it.
"""

import struct
import zlib
import json

import numpy as np

MAGIC = b'PLANTAUDIO'
HEADER_SIZE = 128

VERSION_CONTINUOUS = 3.0     # every frame transmitted
VERSION_EVENT = 4.0          # click candidates and neighbours only

BINS_PER_FRAME = 154
RECORD_SIZE = 5              # float32 magnitude + int8 phase
FRAME_BYTES = BINS_PER_FRAME * RECORD_SIZE      # 770

CLCK_MARKER = b'CLCK'
EVNT_MARKER = b'EVNT'

#: One EVNT record. Same fields, same order and same types as the 21 event
#: bytes on the wire, so nothing is reinterpreted between the USB frame and
#: the file — see Audio_FIrmware_THIRD_realtime/Core/Src/fft_usb.c.
EVENT_RECORD_FORMAT = '<IBffff'
EVENT_RECORD_SIZE = struct.calcsize(EVENT_RECORD_FORMAT)    # 21

FLAG_CANDIDATE = 0x01
FLAG_NEIGHBOUR = 0x02
FLAG_OVERFLOW = 0x04

_HEADER_FIELDS = (
    # (key,           offset, struct format)
    ('version',           10, '<f'),
    ('fs',                34, '<I'),
    ('fft_size',          38, '<I'),
    ('freq_min',          42, '<I'),
    ('freq_max',          46, '<I'),
    ('threshold',         50, '<f'),
    ('start_time',        54, '<d'),
    ('end_time',          62, '<d'),
    ('data_points',       70, '<I'),
    ('acquisition_count', 74, '<I'),
)


# ── header ──────────────────────────────────────────────────────────────────

def build_header(header: dict) -> bytes:
    """Serialise the 128-byte header. Mirrors MainWindowAudio._create_header."""
    out = bytearray(HEADER_SIZE)
    magic = header.get('magic', MAGIC)[:10]
    out[0:len(magic)] = magic

    exp = header.get('experiment_type', 'Audio Test')
    if isinstance(exp, str):
        exp = exp.encode('ascii', errors='replace')
    exp = exp[:20]
    out[14:14 + len(exp)] = exp

    for key, offset, fmt in _HEADER_FIELDS:
        struct.pack_into(fmt, out, offset, header[key])
    return bytes(out)


def parse_header(blob: bytes) -> dict:
    """Read the 128-byte header. Raises ValueError on a foreign file."""
    if len(blob) < HEADER_SIZE:
        raise ValueError("header incompleto")
    magic = blob[0:10].rstrip(b'\x00')
    if magic != MAGIC:
        raise ValueError(f"magic number non valido: {magic!r}")
    info = {
        'magic': magic.decode('ascii'),
        'experiment': blob[14:34].rstrip(b'\x00').decode('ascii', errors='replace'),
    }
    for key, offset, fmt in _HEADER_FIELDS:
        info[key] = struct.unpack_from(fmt, blob, offset)[0]
    return info


def is_event_recording(version) -> bool:
    """True for a file whose body frames are NOT contiguous in time."""
    try:
        return float(version) >= VERSION_EVENT
    except (TypeError, ValueError):
        return False


def check_version(version, tool: str):
    """
    Raise unless this tool can read the file honestly.

    For every reader that assumes contiguous frames. A v4 file loaded by one of
    them is not corrupt — it parses perfectly — it is simply laid out in time as
    if the gaps were not there, and nothing about the result would look wrong.
    """
    if is_event_recording(version):
        raise ValueError(
            f"{tool}: questo e' un file a EVENTI (.paudio v{float(version):.1f}). "
            "I frame non sono contigui nel tempo: la posizione di ciascuno sta nel "
            "footer EVNT. Usa il caricatore principale (AudioLoadWorker), che lo "
            "legge, invece di trattarli come una registrazione continua.")


# ── footers ─────────────────────────────────────────────────────────────────

def pack_click_footer(click_events) -> bytes:
    """The CLCK block. Written even when empty in v4 — see the module docstring."""
    payload = zlib.compress(
        json.dumps(click_events or [], separators=(',', ':')).encode('utf-8'))
    return CLCK_MARKER + struct.pack('<I', len(payload)) + payload


def pack_event_footer(records) -> bytes:
    """
    The EVNT block: one record per body frame, in body order.

    `records` is an iterable of
        (frame_idx, flags, E_i, E_hat_floor, noise_floor, std_noise)

    POSITIONAL, deliberately: record i describes body frame i.

    frame_idx is a position in the FILE, not the board's raw counter. The board
    restarts its counter at 0 on every !start!, so the app phases each
    acquisition by the recording time already elapsed before writing it here —
    otherwise a second acquisition appended to the same file would sit on top
    of the first in time. The values are therefore monotonic across the whole
    file, and the phase counts recording time only: a stop-and-resume ten
    minutes later leaves no ten-minute hole, exactly as the elapsed-time clock
    in the UI shows it.
    """
    payload = bytearray()
    for rec in records:
        payload += struct.pack(EVENT_RECORD_FORMAT, int(rec[0]), int(rec[1]),
                               float(rec[2]), float(rec[3]),
                               float(rec[4]), float(rec[5]))
    return EVNT_MARKER + struct.pack('<I', len(payload)) + bytes(payload)


def find_block(blob: bytes, marker: bytes, start: int = 0):
    """
    Locate one footer block. Returns (marker_start, payload_start, payload_len)
    or None.
    """
    pos = blob.find(marker, start)
    while pos >= 0:
        if pos + 8 <= len(blob):
            length = struct.unpack_from('<I', blob, pos + 4)[0]
            if pos + 8 + length <= len(blob):
                return pos, pos + 8, length
        pos = blob.find(marker, pos + 1)
    return None


def split_sections(blob: bytes):
    """
    Cut everything after the header into (body, click_payload, event_payload).

    The body ends at the FIRST footer marker, whichever it is: a v4 file always
    writes CLCK first, but a file truncated between the two must not turn the
    remaining EVNT bytes into magnitudes.
    """
    clck = find_block(blob, CLCK_MARKER)
    evnt = find_block(blob, EVNT_MARKER)

    ends = [b[0] for b in (clck, evnt) if b is not None]
    body_end = min(ends) if ends else len(blob)

    click_payload = blob[clck[1]:clck[1] + clck[2]] if clck else b''
    event_payload = blob[evnt[1]:evnt[1] + evnt[2]] if evnt else b''
    return blob[:body_end], click_payload, event_payload


def parse_click_payload(payload: bytes):
    if not payload:
        return []
    try:
        return json.loads(zlib.decompress(payload).decode('utf-8'))
    except Exception:                                   # noqa: BLE001
        return []


def parse_event_payload(payload: bytes) -> dict:
    """
    Decode the EVNT block into parallel arrays, one entry per body frame.

    Returns empty arrays when the block is absent, so a caller can use the
    result unconditionally.
    """
    n = len(payload) // EVENT_RECORD_SIZE if payload else 0
    out = {
        'frame_idx':   np.zeros(n, dtype=np.uint32),
        'flags':       np.zeros(n, dtype=np.uint8),
        'E_i':         np.zeros(n, dtype=np.float32),
        'E_hat_floor': np.zeros(n, dtype=np.float32),
        'noise_floor': np.zeros(n, dtype=np.float32),
        'std_noise':   np.zeros(n, dtype=np.float32),
    }
    for i in range(n):
        (out['frame_idx'][i], out['flags'][i], out['E_i'][i],
         out['E_hat_floor'][i], out['noise_floor'][i],
         out['std_noise'][i]) = struct.unpack_from(
            EVENT_RECORD_FORMAT, payload, i * EVENT_RECORD_SIZE)
    return out


def event_timestamps(frame_idx, fs, fft_size) -> np.ndarray:
    """
    Where each transmitted frame actually sits in the recording, in seconds.

    ⚠️ This is the board's count of PROCESSED frames. A frame the ADC queue
    dropped never reached Stage 1, so it does not appear as a gap here — the
    count is simply one short, and every timestamp after it runs early by
    fft_size / fs. The firmware's !stats! drop counter is the only evidence
    that happened; the app records it at stop.
    """
    return np.asarray(frame_idx, dtype=np.float64) * (float(fft_size) / float(fs))
