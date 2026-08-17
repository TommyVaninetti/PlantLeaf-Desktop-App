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
Hybrid dataset construction — Khait/Dryad clicks injected into PlantLeaf noise beds.

Implements HYBRID_TRAINING_NOISE_INJECTION_SPEC.md. Every module here is
deliberately Qt-free so the whole engine runs headless, under multiprocessing,
and inside plain pytest-style scripts. Nothing in this package may import
PySide6, directly or transitively — see `pipeline_loader` for why that takes
care.

Module map
----------
pipeline_loader : Qt-free access to click_pipeline_v5 / spectral_analysis.
channel_model   : resampling + minimum-phase microphone colorization.
frame_emulator  : time domain <-> firmware frame convention, .paudio writer.
dryad_io        : Dryad WAV reading and manifest building.
noise_bed       : event-free noise bed extraction from PlantLeaf recordings.
injector        : the per-click injection procedure (spec section 9).
render          : batch PNG + contact-sheet rendering.
"""

__all__ = [
    "pipeline_loader",
    "channel_model",
    "frame_emulator",
    "dryad_io",
    "noise_bed",
    "injector",
    "render",
]
