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
Reading the Khait et al. (2023) Dryad corpus.

Dataset: doi:10.5061/dryad.jwstqjqf7, CC0. Cite Khait et al. 2023 + the Dryad
DOI, and disclose the dataset explicitly per EUCYS Rules Article 14.

Layout: one directory per class, files named `id_<plant_id>_sound_<sound_id>.wav`.
For non-plant classes (Empty Pot, Greenhouse Noises) `plant_id` identifies a
recording instance rather than a plant.

Measured properties of the corpus (all six classes, verified across random
samples — see `test_scripts/verify_channel_model.py`):

  * mono int16, 500 kHz, **exactly 1001 samples** (2.002 ms) in every file
  * the envelope peak sits at index 499 in the median for EVERY class — these
    are trigger-aligned extractions, not free-running captures. The click
    classes cluster tightly around it (Tobacco Dry spreads 6 samples, Tomato Dry
    17, Empty Pot 34); Greenhouse Noises spreads ~665 because a diffuse
    broadband noise clip has no well-defined envelope peak to align on.
  * peaks are 6.9k-11.6k of int16 full scale, 0 % clipped; absolute units are
    arbitrary (the Avisoft chain's scaling is not documented)
  * 79-91 % of clip energy already lies within PlantLeaf's 20-80 kHz band, and
    only 4.8-13.6 % lies above the 100 kHz that resampling discards

The near-constant peak position matters downstream: injection MUST randomise the
insertion time, otherwise every synthetic positive carries an almost identical
peak-within-clip offset that a classifier can exploit instead of learning the
click's shape.

The arbitrary amplitude units matter too, favourably: since absolute scale is
meaningless, amplitude is set entirely by the injection gain, and the unresolved
board-gain calibration question (spec section 3) does not block any of this work.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.io import wavfile

# Class directory -> (is_click, species, condition). The Dryad README describes
# each directory; `is_click` records whether Khait's dual-microphone coincidence
# test labelled these events as plant sounds.
CLASS_INFO: dict[str, tuple[bool, str, str]] = {
    "Tomato Cut":        (True,  "tomato",  "cut"),
    "Tomato Dry":        (True,  "tomato",  "dry"),
    "Tobacco Cut":       (True,  "tobacco", "cut"),
    "Tobacco Dry":       (True,  "tobacco", "dry"),
    "Empty Pot":         (False, "none",    "empty_pot"),
    "Greenhouse Noises": (False, "none",    "greenhouse"),
}

CLICK_CLASSES = tuple(k for k, v in CLASS_INFO.items() if v[0])
NOISE_CLASSES = tuple(k for k, v in CLASS_INFO.items() if not v[0])

EXPECTED_FS = 500_000
EXPECTED_SAMPLES = 1001

_NAME_RE = re.compile(r"^id_(\d+)_sound_(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class DryadClip:
    """One Dryad .wav file and its parsed provenance."""
    path: Path
    class_name: str
    plant_id: int
    sound_id: int
    is_click: bool
    species: str
    condition: str

    @property
    def clip_id(self) -> str:
        """Stable identifier, unique across the corpus."""
        return f"{self.class_name.replace(' ', '_')}__id_{self.plant_id}_sound_{self.sound_id}"


@dataclass
class ClipAudio:
    """Decoded audio plus the anomalies found while reading it."""
    clip: DryadClip
    fs: int
    samples: np.ndarray            # float64, native rate, DC-removed
    warnings: list[str] = field(default_factory=list)


def parse_filename(path: str | Path) -> tuple[int, int] | None:
    """
    Extract (plant_id, sound_id) from `id_<plant>_sound_<n>.wav`, or None.
    """
    m = _NAME_RE.match(Path(path).stem)
    return (int(m.group(1)), int(m.group(2))) if m else None


def build_manifest(root: str | Path, classes: list[str] | None = None) -> list[DryadClip]:
    """
    Enumerate the corpus under `root` (the directory containing the class folders).

    Files whose names do not match the documented pattern are skipped rather than
    guessed at — provenance that cannot be parsed cannot be tracked through the
    split-leakage checks, so such a file must not silently enter the dataset.

    Returns clips sorted by (class, plant_id, sound_id) so runs are reproducible.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dryad root not found: {root}")

    wanted = list(CLASS_INFO) if classes is None else list(classes)
    unknown = [c for c in wanted if c not in CLASS_INFO]
    if unknown:
        raise ValueError(f"unknown Dryad class(es): {unknown}. Known: {sorted(CLASS_INFO)}")

    clips: list[DryadClip] = []
    for class_name in wanted:
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        is_click, species, condition = CLASS_INFO[class_name]
        for wav in class_dir.glob("*.wav"):
            if wav.name.startswith("."):        # macOS resource forks (._foo.wav)
                continue
            parsed = parse_filename(wav)
            if parsed is None:
                continue
            plant_id, sound_id = parsed
            clips.append(DryadClip(
                path=wav, class_name=class_name, plant_id=plant_id, sound_id=sound_id,
                is_click=is_click, species=species, condition=condition,
            ))

    clips.sort(key=lambda c: (c.class_name, c.plant_id, c.sound_id))
    return clips


def read_clip(clip: DryadClip, *, strict: bool = False) -> ClipAudio:
    """
    Decode one clip to float64 at its native 500 kHz rate.

    Deviations from the documented format are collected in `warnings` rather than
    raised, so a single odd file cannot abort a 5477-clip batch; pass
    `strict=True` to turn them into errors instead.

    Conversions applied:
      * stereo -> first channel (the corpus is mono; this is a guard, and it is
        recorded as a warning if it ever fires)
      * int16/int32/uint8 -> float64 in the same arbitrary units, no rescaling to
        +-1. Absolute scale is meaningless here and the injection gain sets
        amplitude, so normalising would only invent a false calibration.
      * DC offset removed. Measured DC is ~0 for the plant classes but +8.8 LSB
        for Greenhouse Noises, and any DC survives resampling to become a bin-0
        artefact.
    """
    warnings: list[str] = []
    fs, data = wavfile.read(clip.path)

    if data.ndim > 1:
        warnings.append(f"{data.shape[1]} channels, using channel 0")
        data = data[:, 0]

    x = data.astype(np.float64)

    if fs != EXPECTED_FS:
        warnings.append(f"fs={fs} Hz, expected {EXPECTED_FS}")
    if len(x) != EXPECTED_SAMPLES:
        warnings.append(f"{len(x)} samples, expected {EXPECTED_SAMPLES}")
    if len(x) == 0:
        warnings.append("empty file")
    else:
        peak = float(np.max(np.abs(x)))
        if np.issubdtype(data.dtype, np.integer):
            full_scale = float(np.iinfo(data.dtype).max)
            if peak >= full_scale:
                warnings.append(f"clipped at full scale ({peak:.0f})")
        if peak == 0.0:
            warnings.append("all-zero signal")

    if strict and warnings:
        raise ValueError(f"{clip.path.name}: " + "; ".join(warnings))

    if len(x):
        x = x - float(np.mean(x))

    return ClipAudio(clip=clip, fs=int(fs), samples=x, warnings=warnings)


def summarise_manifest(clips: list[DryadClip]) -> dict[str, dict]:
    """
    Per-class counts and distinct plant counts, for logging before a long run.
    """
    out: dict[str, dict] = {}
    for c in clips:
        entry = out.setdefault(c.class_name, {
            "n_clips": 0, "plants": set(), "is_click": c.is_click,
            "species": c.species, "condition": c.condition,
        })
        entry["n_clips"] += 1
        entry["plants"].add(c.plant_id)
    for entry in out.values():
        entry["n_plants"] = len(entry.pop("plants"))
    return out
