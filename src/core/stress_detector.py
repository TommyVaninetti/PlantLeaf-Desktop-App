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
stress_detector.py — Stage 5: sliding-window stress-event detector
==================================================================

Specification: STRESS_DETECTION_ALGORITHM_v1.md (§ references below are to it).

Consumes the stream of ACCEPTED click events (Stage 4 survivors,
`stage_blocked == ''`) and emits `stress_onset` / `stress_cleared`.

    Score(t) = sum of w_i over events with t - W < t_i <= t        (§4)
    onset    : not active  and Score >= h_high  and t - t_last_onset >= R
    clear    : active      and Score <  h_low = rho * h_high      (§6)

ONE IMPLEMENTATION, ON PURPOSE
------------------------------
There is no vectorised "offline" version of this detector. Calibration and
validation replay THIS class event by event (`replay_session`), because the
false-alarm rate is a property of the complete detector — window, hysteresis and
refractory together (§8.2). A second implementation would be a second thing to
calibrate, and the first one is the thing that ships.

PURE PYTHON
-----------
No Qt, no numpy. Importable headless from the analysis scripts without pulling
in core/__init__ (which drags the window stack), and a line-by-line reference for
a firmware port.

QUEUE CAPACITY
--------------
`capacity=None` (the default) means unbounded — required for calibration, where a
clipped window would silently understate the statistic on exactly the largest
bursts. A deployment sets a finite capacity from the calibration corpus's
`max_occupancy`, with margin (§7.1.5). On overflow the OLDEST entry is evicted and
the next emitted event carries `overflow=True`; the newest event is never dropped.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

ONSET = 'stress_onset'
CLEARED = 'stress_cleared'

MODE_HARD = 'hard'   # Variant A: w_i = 1  (§5.1, primary)
MODE_SOFT = 'soft'   # Variant B: w_i = p_i (§5.2)
MODES = (MODE_HARD, MODE_SOFT)


@dataclass(frozen=True)
class StressDetectorParams:
    """Detector parameters. See spec §10 for how each one is set."""
    W: float                           # window length [s]
    h_high: float                      # onset threshold — a CALIBRATION OUTPUT (§8)
    R: float                           # refractory period [s], onset only (§6.1)
    rho: float = 0.5                   # h_low = rho * h_high
    mode: str = MODE_HARD
    capacity: Optional[int] = None     # None = unbounded (calibration)
    resum_every: int = 10_000          # soft mode: recompute the sum every N events

    def __post_init__(self):
        if self.W <= 0:
            raise ValueError(f"W must be > 0, got {self.W}")
        if self.R < 0:
            raise ValueError(f"R must be >= 0, got {self.R}")
        if not 0 < self.rho < 1:
            raise ValueError(f"rho must be in (0, 1), got {self.rho}")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.capacity is not None and self.capacity < 1:
            raise ValueError(f"capacity must be >= 1 or None, got {self.capacity}")

    @property
    def h_low(self) -> float:
        return self.rho * self.h_high


@dataclass(frozen=True)
class StressEvent:
    kind: str           # ONSET | CLEARED
    t: float            # timestamp of the arrival (or tick) that decided it [s]
    score: float        # Score(t) at the decision
    occupancy: int      # events inside the window at the decision
    overflow: bool      # the queue overflowed since the previous emitted event


class StressDetector:
    """Event-driven Stage 5 detector (spec §7). O(1) amortised per event."""

    def __init__(self, params: StressDetectorParams):
        self.params = params
        self.reset()

    def reset(self) -> None:
        self._queue: deque = deque()            # (t, w)
        self._sum = 0 if self.params.mode == MODE_HARD else 0.0
        self._n_since_resum = 0
        self._overflow_pending = False
        self.alarm_active = False
        self.t_last_alarm = -math.inf
        self.t_last = -math.inf
        self.overflow_count = 0

    # ── state ────────────────────────────────────────────────────────────────

    @property
    def score(self) -> float:
        return self._sum

    @property
    def occupancy(self) -> int:
        return len(self._queue)

    # ── internals ────────────────────────────────────────────────────────────

    def _check_time(self, t: float) -> None:
        if t < self.t_last:
            raise ValueError(f"timestamps must be non-decreasing: {t} after {self.t_last}")
        self.t_last = t

    def _evict(self, t: float) -> None:
        # `<=`: the window is half-open, (t - W, t]  (§7.1.1)
        edge = t - self.params.W
        q = self._queue
        while q and q[0][0] <= edge:
            self._sum -= q.popleft()[1]

    def _resum(self) -> None:
        self._sum = math.fsum(w for _, w in self._queue)
        self._n_since_resum = 0

    def _emit(self, kind: str, t: float) -> StressEvent:
        ev = StressEvent(kind, t, self._sum, len(self._queue), self._overflow_pending)
        self._overflow_pending = False
        return ev

    def _clear_check(self, t: float) -> List[StressEvent]:
        # No refractory gate on the clear (§6.1).
        if self.alarm_active and self._sum < self.params.h_low:
            self.alarm_active = False
            if self.params.mode == MODE_SOFT:
                self._resum()                   # §7.1.3
            return [self._emit(CLEARED, t)]
        return []

    # ── public API ───────────────────────────────────────────────────────────

    def on_event(self, t: float, p: float = 1.0) -> List[StressEvent]:
        """Feed one accepted click at time `t` [s] with Stage 3 score `p`."""
        self._check_time(t)
        prm = self.params

        # admit, THEN evict against the new timestamp (§7.1.2)
        w = 1 if prm.mode == MODE_HARD else float(p)
        self._queue.append((t, w))
        self._sum += w
        self._evict(t)

        if prm.capacity is not None and len(self._queue) > prm.capacity:
            self._sum -= self._queue.popleft()[1]      # oldest, never the newest
            self._overflow_pending = True
            self.overflow_count += 1

        if prm.mode == MODE_SOFT:
            self._n_since_resum += 1
            if self._n_since_resum >= prm.resum_every:
                self._resum()

        if (not self.alarm_active and self._sum >= prm.h_high
                and t - self.t_last_alarm >= prm.R):
            self.alarm_active = True
            self.t_last_alarm = t
            return [self._emit(ONSET, t)]
        return self._clear_check(t)

    def tick(self, t: float) -> List[StressEvent]:
        """Evict and clear-check with no new event (§7.1.4). Never declares an onset."""
        self._check_time(t)
        self._evict(t)
        return self._clear_check(t)


# ── offline replay ───────────────────────────────────────────────────────────

@dataclass
class SessionResult:
    onsets: List[StressEvent] = field(default_factory=list)
    clears: List[StressEvent] = field(default_factory=list)
    max_score: float = 0.0
    max_occupancy: int = 0
    overflow_count: int = 0
    alarm_intervals: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def n_onsets(self) -> int:
        return len(self.onsets)

    @property
    def alarm_seconds(self) -> float:
        return sum(b - a for a, b in self.alarm_intervals)


def replay_session(times: Sequence[float],
                   params: StressDetectorParams,
                   probs: Optional[Sequence[float]] = None,
                   t_end: Optional[float] = None,
                   tick_s: Optional[float] = None) -> SessionResult:
    """Run a fresh detector over one recording's accepted events.

    `times` need not be sorted. `probs` is required in soft mode. With `tick_s`,
    ticks run at k * tick_s up to `t_end` (or the last event), so clear times are
    accurate to one tick; onsets are identical either way. An alarm still active
    at the end is closed at `t_end` (or the last event) in `alarm_intervals`.
    """
    if probs is None:
        if params.mode == MODE_SOFT:
            raise ValueError("soft mode needs probs")
        order = sorted(range(len(times)), key=times.__getitem__)
        stream: Iterable[Tuple[float, float]] = ((float(times[i]), 1.0) for i in order)
    else:
        if len(probs) != len(times):
            raise ValueError("times and probs differ in length")
        order = sorted(range(len(times)), key=times.__getitem__)
        stream = ((float(times[i]), float(probs[i])) for i in order)

    det = StressDetector(params)
    res = SessionResult()
    t_on: Optional[float] = None

    def absorb(events: List[StressEvent]) -> None:
        nonlocal t_on
        for ev in events:
            if ev.kind == ONSET:
                res.onsets.append(ev)
                t_on = ev.t
            else:
                res.clears.append(ev)
                res.alarm_intervals.append((t_on, ev.t))
                t_on = None

    next_tick = 0.0 if tick_s else math.inf
    last_t = 0.0
    for t, p in stream:
        while next_tick <= t:
            absorb(det.tick(next_tick))
            next_tick += tick_s
        absorb(det.on_event(t, p))
        last_t = t
        if det.score > res.max_score:
            res.max_score = det.score
        if det.occupancy > res.max_occupancy:
            res.max_occupancy = det.occupancy

    end = last_t if t_end is None else max(float(t_end), last_t)
    if tick_s:
        while next_tick <= end:
            absorb(det.tick(next_tick))
            next_tick += tick_s
    if t_on is not None:
        res.alarm_intervals.append((t_on, end))
    res.overflow_count = det.overflow_count
    return res


def score_trace(times: Sequence[float], W: float,
                weights: Optional[Sequence[float]] = None) -> List[Tuple[float, float]]:
    """(t_i, Score(t_i)) at every arrival — for plots. Uses the same eviction rule."""
    prm = StressDetectorParams(W=W, h_high=math.inf, R=0.0,
                               mode=MODE_HARD if weights is None else MODE_SOFT)
    det = StressDetector(prm)
    order = sorted(range(len(times)), key=times.__getitem__)
    out = []
    for i in order:
        det.on_event(float(times[i]), 1.0 if weights is None else float(weights[i]))
        out.append((float(times[i]), det.score))
    return out
