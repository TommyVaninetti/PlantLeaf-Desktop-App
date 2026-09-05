"""
Data Collection Dialog v5 for PlantLeaf Click Detection

Phase 2: Exports Stage 1 candidates from .paudio files as CSV + screenshots.
- Runs Stage 1 adaptive threshold to find candidates (wide net, k=1.5 default)
- Computes all 17 features for each survivor
- Renders two-panel screenshots (FFT + iFFT + features)
- Exports CSV with schema ready for manual labeling
"""

import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import traceback
from dataclasses import dataclass, asdict, field

import numpy as np
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QSpinBox, QDoubleSpinBox, QLabel, QFileDialog,
    QProgressBar, QPlainTextEdit, QCheckBox, QAbstractItemView,
    QWidget, QSizePolicy, QApplication, QRadioButton, QButtonGroup,
    QMessageBox, QGridLayout, QComboBox,
)
from PySide6.QtCore import QThread, Signal, Qt, QPoint, QRect
from PySide6.QtGui import QColor, QPixmap, QPainter, QFont, QImage, QPen

from core.settings_manager import SettingsManager
# Stage 2 mode names. ⚠️ IMPORTED AT MODULE LEVEL ON PURPOSE. They were previously
# referenced only inside a conditional expression in the start handler, so the
# NameError fired ONLY when the user selected the non-default option — a crash
# that dialog-construction tests and every default-path run sail straight past.
# This module otherwise uses lazy imports to dodge a circular import, but
# click_pipeline_v5 imports nothing from components, so a top-level import of
# these three names is safe.
from core.click_pipeline_v5 import (            # noqa: E402
    STAGE2_MODE_V5, STAGE2_MODE_CONSERVATIVE, STAGE2_MODE_AGGRESSIVE,
)

# Import v5 signal processing pipeline (use lazy imports to avoid circular imports)
import importlib.util
import sys


# ── CONFIGURATION ──────────────────────────────────────────────────────────
"""
All tuneable parameters for data collection dialog.
"""

CSV_FILENAME = 'data_collection_export.csv'
SCREENSHOTS_FOLDER = 'screenshots'
CSVS_FOLDER = 'CSVs'

# Output layout — ONE FOLDER PER ANALYSED RECORDING, under each of the two trees:
#
#     <output_dir>/
#         CSVs/<stem>/<stem>_candidates.csv
#         screenshots/<stem>/<stem>_<frame:06d>.png
#
# Both used to be flat, so a session covering 20 recordings put 20 CSVs in one
# directory and several thousand PNGs in another. The per-recording folder is what
# makes a single file's review self-contained — which is how the labelling pass is
# actually done, one recording at a time, rather than by opening one giant export.
#
# click_review_dialog reads BOTH layouts: it walks up from the CSV looking for a
# `screenshots` tree, so exports made before this change still resolve.

# Screenshot rendering (pure QPainter + QImage — no pyqtgraph, no OpenGL, no segfaults)
#
# ⚠️ THESE ARE THE SINGLE SOURCE OF TRUTH FOR THE SCREENSHOT GEOMETRY.
# click_review_dialog crops the header and footer off before displaying the PNG,
# and it IMPORTS these constants to do it. They used to be duplicated there as
# hand-tuned literals with only a comment holding them in step, which meant any
# change to the footer height silently mis-cropped every screenshot. Do not
# re-introduce literals on the consumer side.
SCREENSHOT_WIDTH  = 1400
SCREENSHOT_HEIGHT = 960   # the v6 footer carries 11 groups at 11 pt, and both
                          # Quality and Stage 1 can wrap to a second line when they
                          # warn. At 900 the footer held 12 lines and the Verdict
                          # group — the last one drawn — fell off the bottom with no
                          # error; 960 gives 14. See the truncation marker in
                          # _draw_feature_footer, which now makes that visible.
PANEL_HEIGHT      = 480   # plot area height within each panel
SCREENSHOT_MARGIN = 14
SCREENSHOT_HEADER_H = 36  # title strip, cropped by the review dialog
SCREENSHOT_GAP    = 8
#: Footer height, derived — the review dialog crops exactly this much off the bottom.
SCREENSHOT_FOOTER_H = (SCREENSHOT_HEIGHT
                       - (SCREENSHOT_HEADER_H + SCREENSHOT_MARGIN + PANEL_HEIGHT
                          + SCREENSHOT_GAP)
                       - SCREENSHOT_MARGIN)

# K spinbox constraints
K_MIN = 0.5
K_MAX = 3.0
K_STEP = 0.1
K_DEFAULT = 1.5  # Default Stage 1 multiplier (from click_pipeline_v5.py)


# ── CSV SCHEMA ──────────────────────────────────────────────────────────────

SCHEMA_VERSION = 'v6'   # The v6 schema: 51 columns, Stage 1 v5.1 peak-picking.
                        # There is no v6.0/v6.1 split — the 45-column intermediate
                        # never left development, so v6 means these 51 columns and
                        # nothing else. Anything stamped 'v6.0' predates Stage 1 v5.1
                        # and is a development artefact, not a dataset.

# ── v6 schema (SPECTRAL_FEATURES_v6_PROPOSAL.md Phase 2) ────────────────────
# A SUPERSET, deliberately: it carries the v5 features that v6 proposes to remove
# (SPR, R_spectral, centroid_shift_hz) ALONGSIDE the v6 additions. §7.4 requires
# removals and additions to be measured as separate experiments (Run A / B / C),
# and with a superset each run is a column subset of one export instead of three
# separate exports and three chances for the candidate set to drift.
CSV_COLUMNS = [
    # ── identity & provenance ──
    'schema_version',
    'session_id',          # hard-required by train_svm.py; defaults to the file stem
    'file',
    'frame_idx',
    'peak_abs',            # absolute sample index of the peak — exact integer
                           # arithmetic, IDENTICAL for the fi / fi+1 candidates of
                           # one straddling click. This is the safe label-migration
                           # key; (file, frame_idx) is not unique.
                           # `canonical_frame_idx` used to sit here and was dropped
                           # in v6: it is exactly `peak_abs // FFT_SIZE`.
    'timestamp_s',
    'stage2_mode',         # which Stage 2 RULE produced this row:
                           #   'v5_fitgate'      the original (costs 12.2 % of clicks)
                           #   'v6_conservative' default, measured zero click loss
                           #   'v6_aggressive'   opt-in, 2.1 % click cost
                           # Same reasoning as stage1_params: two exports made with
                           # different tiers are otherwise indistinguishable after
                           # the fact, and the aggressive one has a measured 2.1 %
                           # recall cost, so it must never be invisible.
    'stage1_params',       # 'v51_peakpick;k=1.50;R=1;C=10' — one column instead of
                           # four (stage1_mode / k_used / R_used / C_used) that would
                           # repeat the same value on every row of an export.
    # ── noise state at detection ──
    'noise_floor_mV',
    'std_noise_mV',
    'E_hat_floor',
    'k_ratio',             # E_i / Ê_floor(i) — EXACTLY what Stage 1 thresholds on,
                           # since E_i > k·Ê_floor ⟺ k_ratio > k. Exporting it makes
                           # the k sweep a CSV filter instead of a corpus re-pass.
                           # E_i itself is recoverable as k_ratio × E_hat_floor.
    # ── v5 features (17), unchanged names and order ──
    'peak_SNR',
    'pre_SNR',
    'post_SNR',
    'rise_time_ms',
    'fall_time_ms',
    'asymmetry_integral',
    'ZCR_pre',
    'ZCR_click',
    'ZCR_post',
    'kurtosis',
    'centroid_shift_hz',
    'tau_ms',
    'R2',
    'fit_coverage',
    'SPR',
    'R_spectral',
    'FPE_hz',
    # ── v6 features (9) ──
    'spectral_entropy',       # D5
    'shape_novelty',          # D6
    'spectral_tilt',          # D7  (on P_region — see §5.3's clarification)
    'temporal_concentration', # D8
    'FPE_hz_region',          # D17 — beside the frame FPE_hz, so Run C is a subset
    'SPR_region',             # lets D16 be tested rather than assumed
    'f_50_hz',                # D18 — CSV only, not fed to the SVM
    'IQR_f',                  # D18 — CSV only
    # ── harmonic_confinement (HARMONIC_CONFINEMENT_FEATURE_SPEC.md) ──
    # FRAME domain, not region: the region window is ill-defined for sustained
    # oscillation and too coarse to resolve a 1-2 kHz transducer linewidth.
    'harmonic_confinement',   # log2(min(r_A, r_B)). 0 = excess spread uniformly.
                              # Bounded above by ~3.36 BY CONSTRUCTION (all the
                              # excess in the two bands, split 2/3 : 1/3).
    'hc_f1_hz',               # the located fundamental. Also disambiguates WHY a
                              # NaN happened: NaN here = no excess at all; finite
                              # and > 40843.75 Hz = the 2nd harmonic is off the
                              # transmitted range, so band B was clipped away.
    'hc_r_A',                 # confinement in the fundamental band, vs its null
    'hc_r_B',                 # ...and in the harmonic band. Kept because
                              # harmonic_confinement is a min() and therefore
                              # discards WHICH band was flat — the distinction §5
                              # of the spec needs to separate a lone hum from a
                              # true pair. Not recoverable after the fact.
    'local_crest',            # Stage 1 v5.1 §4.3 — E_i over the median of its
                              # ±C neighbourhood, the immediate neighbours excluded.
                              # NaN, never −1, when that median is 0 or non-finite;
                              # `local_crest_valid` is therefore redundant and was
                              # never added (it is `not isnan(local_crest)`).
    # ── validity & quality flags (5) ──
    'fit_valid',    # 0/1 — "fit failed" vs "fit succeeded and was terrible"
    'decay_len',    # the dead-zone coordinate; fit_coverage encodes it only indirectly
    'n_seg',        # §4.3 requires recording it beside every feature
                    # `n_seg_valid` is not a column — it is `n_seg >= V6_MIN_NSEG`.
    'b3_frames',    # frames in the Buffer-3 window; 0 ⇒ every v6 feature is NaN
    'gibbs_fired',  # suppress_edge_artifacts tripped ⇒ biased subtraction that frame
    # ── Stage 1 v5.1 diagnostics (5) — logged, NOT fed to the SVM (§4.4) ──
    # run_length and run_crest are threshold- and history-dependent, so they are
    # deliberately outside a feature set advertised as dimensionless and
    # scale-invariant. They are here because the v5 → v5.1 delta in the paper
    # wants them, and because they are unrecoverable after the fact.
    'run_id',        # index of the above-threshold run this frame belongs to.
                     # peak_rank_in_run / n_peaks_in_run were dropped: both are
                     # groupby(run_id) one-liners.
    'run_length',    # L — frames in that run. v5 rejected the whole run at L > 3.
    'run_crest',     # E_i / median(E over the run) — kept for the delta analysis,
                     # superseded as a detector statistic by local_crest (D5).
    'pos_in_run',    # i − run_start
    'would_pass_v5', # 0/1 — the v5 MAX_RUN verdict, evaluated inline. Filtering on
                     # it reproduces the v5 candidate set from a v5.1 export, which
                     # is what makes the delta computable from ONE pass (D6).
    # ── labels & verdicts ──
    'label',  # Manual labelling, written by click_review_dialog:
              #   1 = click   0 = noise   2 = AMBIGUOUS   '' = not yet judged
              # 2 means the reviewer looked and could not decide. It is a
              # DECISION, not a missing value: it counts as progress, never as a
              # class, and every consumer must handle it explicitly —
              # train_svm.py --ambiguous, and evaluate_candidates.print_summary,
              # both exclude it by default. Left implicit it reaches the SVM as a
              # third class and silently invalidates every metric.
    'note',   # Free text, written by you in the review dialog. Exported empty and
              # never read by the pipeline or the trainer — it exists so an
              # observation made WHILE labelling ("double event", "probe knock",
              # "floor stepped up here") survives to the analysis, instead of being
              # lost or crammed into the label. Kept beside `label` so both travel
              # together through migration.
    # Stage 2/3/4 verdict — written when classification is enabled, otherwise left
    # empty. Same three names, order and semantics as src/ml/evaluate_candidates.py,
    # so a CSV exported here and one produced by the offline CLI are interchangeable
    # and analyze_dataset.py reads either without modification.
    'svm_probability',
    'svm_prediction',
    'stage_blocked',
]

# The v5 features compute_features_v5 produces, in the order the docs list them.
# The deployed SVM reads only the 16 it was trained on (model['features'] is
# authoritative and excludes fit_coverage).
FEATURE_NAMES = [
    'peak_SNR', 'pre_SNR', 'post_SNR',
    'rise_time_ms', 'fall_time_ms', 'asymmetry_integral',
    'ZCR_pre', 'ZCR_click', 'ZCR_post',
    'kurtosis', 'centroid_shift_hz',
    'tau_ms', 'R2', 'fit_coverage',
    'SPR', 'R_spectral', 'FPE_hz',
]

# The v6 additions. Kept as a separate list so a consumer can ask for "v5 only",
# "v6 only" or both without re-deriving the split from CSV_COLUMNS.
FEATURE_NAMES_V6 = [
    'spectral_entropy', 'shape_novelty', 'spectral_tilt', 'temporal_concentration',
    'FPE_hz_region', 'SPR_region', 'f_50_hz', 'IQR_f',
    # Stage 1 v5.1. In the set so it CAN be evaluated in Phase 4, not because it has
    # earned its place — that ablation (O-3) is Phase 4 work. The deployed model is
    # unaffected either way: model['features'] is authoritative at inference, so an
    # extra CSV column changes nothing until a retrain picks it up. It is added now
    # because back-filling it later costs a full re-export.
    'local_crest',
    # CSV-only for now. The spec (§5) is explicit that no gate is added yet and
    # that whether this earns a place is decided from the labelled distribution —
    # §5.3's correlation check against spectral_entropy / FPE_hz. Listing it here
    # makes the value AVAILABLE to a future retrain without feeding the deployed
    # model, which reads model['features'] and not this list.
    'harmonic_confinement',
]

# Emitted for every row but NOT features: provenance, validity and quality flags.
QUALITY_COLUMNS = [
    'fit_valid', 'decay_len', 'n_seg', 'b3_frames', 'gibbs_fired',
]

# Stage 1 v5.1 provenance (§4.4). Emitted for every row, never features — see the
# note beside them in CSV_COLUMNS for why run_length / run_crest stay out.
STAGE1_COLUMNS = [
    'run_id', 'run_length', 'run_crest', 'pos_in_run', 'would_pass_v5',
]

# harmonic_confinement's supporting numbers. NAMED rather than left as literals
# because every exporter has to iterate SOME list to fill them, and the last three
# columns that belonged to no list were silently written empty by the replay
# window for an entire schema generation.
HARMONIC_COLUMNS = [
    'hc_f1_hz', 'hc_r_A', 'hc_r_B',
]

# Export modes for the classification section
EXPORT_ALL        = 'all'        # every Stage 1 candidate with a valid fit, annotated
EXPORT_CONFIRMED  = 'confirmed'  # only candidates that survived all four stages
EXPORT_UNFILTERED = 'unfiltered' # EVERY Stage 1 candidate, including fit_valid == 0
EXPORT_STAGE3     = 'stage3'     # only candidates that cleared Stage 2 (i.e. reached
                                  # the SVM), whatever Stage 3/4 then did with them
#: EXPORT_UNFILTERED exists because those rows have never reached a CSV in this
#: project's history: Stage 2 dropped tau <= 0 before export, so a decay window that
#: could not be fitted was invisible. They arrive with tau_ms / R2 / fit_coverage as
#: NaN, which is only safe now that fit_valid disambiguates "no fit" from "bad fit"
#: (v6 §7.5.3). Expect a lot of them and expect most to be noise — they are Stage 1
#: candidates, which are overwhelmingly noise, and this mode applies NO fit filter.
#:
#: EXPORT_STAGE3 exists for reviewing the SVM/dedup decisions themselves: a
#: candidate has svm_prediction is not None the moment it clears Stage 2 (see
#: run_stages234_annotated), regardless of whether Stage 3 then accepted or
#: rejected it, or Stage 4 deduplicated it. EXPORT_CONFIRMED throws all of that
#: away except the survivors; this mode keeps it.


def _sigfig(x, digits: int = 6):
    """
    Round to SIGNIFICANT FIGURES, for columns whose magnitude is not O(1).

    `round(x, n)` keeps n decimal places, which destroys any quantity far below
    1.0: Ê_floor is an energy in V² around 7e-8, and `round(x, 6)` made every
    single one of them exactly 0.0. Use this wherever the column's natural scale
    is unknown or very small; plain `round` is fine for ratios, mV and ms.

    NaN and inf pass through unchanged — they are meaningful here (v6 §7.5.3) and
    must not be coerced to a number.
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return x
    if v != v or v in (float('inf'), float('-inf')) or v == 0.0:
        return v
    return float(f'{v:.{digits}g}')


# ── CANDIDATE DATA STRUCTURE ────────────────────────────────────────────────

@dataclass
class CandidateData:
    """
    Represents one Stage 1 survivor candidate for export.
    
    Stores metadata, features, and signal data for CSV export + screenshot rendering.
    """
    file: str                           # Filename (.paudio)
    frame_idx: int                      # Frame index
    timestamp_s: float                  # Timestamp in seconds
    
    # Noise estimates (from AdaptiveNoiseEstimatorV5)
    noise_floor: float                  # noise_floor in V
    std_noise: float                    # std_noise in V
    E_hat_floor: float                  # Ê_floor energy threshold in V²
    
    # 17 Features (from compute_features_v5)
    peak_SNR: float
    pre_SNR: float
    post_SNR: float
    rise_time_ms: float
    fall_time_ms: float
    asymmetry_integral: float
    ZCR_pre: float
    ZCR_click: float
    ZCR_post: float
    kurtosis: float
    centroid_shift_hz: float
    tau_ms: float
    R2: float
    fit_coverage: float
    SPR: float
    R_spectral: float
    FPE_hz: float
    
    # Absolute, model-independent click identity (from click_event_key). peak_abs
    # is the peak's absolute sample in the recording; canonical_frame_idx is the
    # frame that owns it. Both Stage 1 candidates of a straddling click share
    # these, which is how Stage 4 collapses them.
    peak_abs: int = 0
    canonical_frame_idx: int = 0

    # ── v6 spectral features (SPECTRAL_FEATURES_v6_PROPOSAL.md §5) ───────────
    # NaN, never a sentinel, when there is no Buffer-3 estimate or the region is
    # too short to transform. A zero would make E[k] equal P_region and look like
    # a clean detection while measuring nothing (§7.5.3).
    spectral_entropy: float = float('nan')
    shape_novelty: float = float('nan')
    spectral_tilt: float = float('nan')
    temporal_concentration: float = float('nan')
    FPE_hz_region: float = float('nan')
    SPR_region: float = float('nan')
    f_50_hz: float = float('nan')
    IQR_f: float = float('nan')
    local_crest: float = float('nan')   # NaN when the ±C background median is 0

    # ── harmonic_confinement (frame domain) ──────────────────────────────────
    harmonic_confinement: float = float('nan')
    hc_f1_hz: float = float('nan')
    hc_r_A: float = float('nan')
    hc_r_B: float = float('nan')

    # ── validity & quality flags ─────────────────────────────────────────────
    fit_valid: int = 0        # 0/1 — "fit failed" vs "fit succeeded and was terrible"
    decay_len: int = 0        # decay_end − decay_start, the dead-zone coordinate
    n_seg: int = 0            # region length; §4.3 requires it beside every feature
    b3_frames: int = 0        # frames in the Buffer-3 window; 0 ⇒ v6 features are NaN
    gibbs_fired: int = 0      # suppress_edge_artifacts tripped on this frame

    # ── Stage 1 v5.1 provenance (§4.4) ───────────────────────────────────────
    # `stage1_params` is the whole selector configuration as one string, so a CSV
    # can always be attributed to the rule that produced it even after the module
    # constants move on. Everything here is diagnostic; none of it is a feature.
    stage1_params: str = ''
    stage2_mode: str = ''
    run_id: int = -1
    run_length: int = 0
    run_crest: float = float('nan')
    pos_in_run: int = 0
    would_pass_v5: int = 0
    k_ratio: float = float('nan')   # E_i / Ê_floor(i); > k by construction

    # Grouping key for StratifiedGroupKFold. train_svm.py exits without it.
    session_id: str = ''

    # Screenshot render window — a slice of the stitched context centred on the
    # peak (2.56 ms, widened if the click is longer). Time axis is peak-relative
    # milliseconds (peak at t = 0), so the picture is frame-grid independent.
    render_signal: np.ndarray = field(default_factory=lambda: np.array([]))
    render_envelope: np.ndarray = field(default_factory=lambda: np.array([]))
    render_t_ms: np.ndarray = field(default_factory=lambda: np.array([]))
    mark_onset_ms: float = 0.0          # onset marker, peak-relative ms (≤ 0)
    mark_decay_end_ms: float = 0.0      # decay-end marker, peak-relative ms (≥ 0)
    seams_ms: list = field(default_factory=list)  # frame-join times in the window
    fft_norm: np.ndarray = field(default_factory=lambda: np.array([]))
    freq_axis: np.ndarray = field(default_factory=lambda: np.array([]))

    # ── Screenshot FFT panel (v6) ────────────────────────────────────────────
    # The region's own amplitude spectrum, and Buffer 3's noise floor expressed on
    # the SAME amplitude axis. All three curves (frame / region / noise) share one
    # axis in mV; see _region_display_spectrum for why that is legitimate.
    region_freqs: np.ndarray = field(default_factory=lambda: np.array([]))
    region_amp:   np.ndarray = field(default_factory=lambda: np.array([]))
    noise_amp:    np.ndarray = field(default_factory=lambda: np.array([]))

    # Pre-computed exponential fit curve (worker thread) — avoids scipy on main
    # thread. Time axis is peak-relative ms, spanning the REAL decay window
    # [decay_start, decay_end] with no frame-edge clip.
    fit_t_ms: Optional[np.ndarray] = field(default=None)  # peak-relative time [ms]
    fit_y: Optional[np.ndarray] = field(default=None)     # fit amplitude [V]

    # Label (empty for Phase 2, filled manually later)
    label: str = ''
    note: str = ''      # free text, filled in the review dialog; never a feature

    # Stage 2/3/4 verdict — filled by the worker when classification is enabled.
    # None/'' means the candidate was never classified (classification switched off),
    # which is what leaves the three CSV columns empty.
    svm_probability: Optional[float] = None
    svm_prediction: Optional[int] = None
    stage_blocked: str = ''

    @property
    def is_confirmed_click(self) -> bool:
        """True once classified and it survived all four stages."""
        return self.svm_prediction is not None and self.stage_blocked == ''

    @property
    def reached_stage3(self) -> bool:
        """True once it cleared Stage 2, whatever Stage 3/4 did with it after."""
        return self.svm_prediction is not None

    def to_feature_dict(self) -> Dict:
        """
        The dict the pipeline expects: the 17 features + the identity keys Stage 4
        deduplicates on (peak_abs / canonical_frame_idx) and frame_idx.

        Values are taken unrounded from the dataclass, unlike to_csv_dict — the SVM
        must see the same precision the features were computed at.
        """
        d = {name: getattr(self, name) for name in FEATURE_NAMES}
        d.update({name: getattr(self, name) for name in FEATURE_NAMES_V6})
        d.update({name: getattr(self, name) for name in QUALITY_COLUMNS})
        d['frame_idx'] = self.frame_idx
        d['peak_abs'] = self.peak_abs
        d['canonical_frame_idx'] = self.canonical_frame_idx
        return d

    def to_csv_dict(self) -> Dict:
        """Convert to dictionary for CSV export (the 51 v6 columns)."""
        return {
            'schema_version': SCHEMA_VERSION,
            'session_id': self.session_id or self.file,
            'file': self.file,
            'frame_idx': self.frame_idx,
            'peak_abs': self.peak_abs,
            'timestamp_s': round(self.timestamp_s, 6),
            'stage1_params': self.stage1_params,
            'stage2_mode': self.stage2_mode,
            # mV since the iFFT amplitude-scale fix — the reconstructed signal is
            # now in true volts, so these land in the mV range. CSVs exported before
            # that fix carry uV columns 256x smaller; the 17 feature columns are
            # dimensionless ratios and are unaffected, so old and new training sets
            # remain comparable. See docs/fft_and_ifft/IFFT_AMPLITUDE_SCALE_FIX.md
            'noise_floor_mV': round(self.noise_floor * 1e3, 4),
            'std_noise_mV': round(self.std_noise * 1e3, 4),
            # ⚠️ SIGNIFICANT FIGURES, NOT DECIMAL PLACES — and the difference is the
            # whole column. Ê_floor is an ENERGY in V², typically ~7e-8, so the
            # `round(x, 6)` this used to be quantised every value to 0.0: measured
            # at 290 061 of 290 061 rows across the v6 corpus, i.e. 100 %. It also
            # silently broke the documented `E_i = k_ratio × E_hat_floor` recovery
            # path, and it had been doing so since v5.
            #
            # Every other float here is a ratio, a millivolt or a millisecond —
            # O(1e-3) or larger — so decimal-place rounding is fine for them and
            # only this one column needed changing. Verified by scanning all 56.
            'E_hat_floor': _sigfig(self.E_hat_floor, 6),
            'k_ratio': round(self.k_ratio, 4),
            'peak_SNR': round(self.peak_SNR, 3),
            'pre_SNR': round(self.pre_SNR, 3),
            'post_SNR': round(self.post_SNR, 3),
            'rise_time_ms': round(self.rise_time_ms, 6),
            'fall_time_ms': round(self.fall_time_ms, 6),
            'asymmetry_integral': round(self.asymmetry_integral, 6),
            'ZCR_pre': round(self.ZCR_pre, 3),
            'ZCR_click': round(self.ZCR_click, 3),
            'ZCR_post': round(self.ZCR_post, 3),
            'kurtosis': round(self.kurtosis, 3),
            'centroid_shift_hz': round(self.centroid_shift_hz, 2),
            'tau_ms': round(self.tau_ms, 6),
            'R2': round(self.R2, 4),
            'fit_coverage': round(self.fit_coverage, 4),
            'SPR': round(self.SPR, 3),
            'R_spectral': round(self.R_spectral, 3),
            'FPE_hz': round(self.FPE_hz, 2),
            # ── v6 (NaN stays NaN — csv writes 'nan', pandas reads it back as NaN,
            #     which is what SimpleImputer expects in Phase 4) ──
            'spectral_entropy': round(self.spectral_entropy, 6),
            'shape_novelty': round(self.shape_novelty, 6),
            'spectral_tilt': round(self.spectral_tilt, 6),
            'temporal_concentration': round(self.temporal_concentration, 6),
            'FPE_hz_region': round(self.FPE_hz_region, 2),
            'SPR_region': round(self.SPR_region, 3),
            'f_50_hz': round(self.f_50_hz, 2),
            'IQR_f': round(self.IQR_f, 2),
            'local_crest': round(self.local_crest, 4),
            'harmonic_confinement': round(self.harmonic_confinement, 4),
            'hc_f1_hz': round(self.hc_f1_hz, 1),
            'hc_r_A': round(self.hc_r_A, 3),
            'hc_r_B': round(self.hc_r_B, 3),
            # ── validity & quality ──
            'fit_valid': int(self.fit_valid),
            'decay_len': int(self.decay_len),
            'n_seg': int(self.n_seg),
            'b3_frames': int(self.b3_frames),
            'gibbs_fired': int(self.gibbs_fired),
            # ── Stage 1 v5.1 diagnostics ──
            'run_id': int(self.run_id),
            'run_length': int(self.run_length),
            'run_crest': round(self.run_crest, 4),
            'pos_in_run': int(self.pos_in_run),
            'would_pass_v5': int(self.would_pass_v5),
            'label': self.label,
            'note': self.note,
            # Rounded to 4 dp to match evaluate_candidates.py's output exactly.
            # None → '' via csv.DictWriter, which is how an unclassified candidate
            # and a Stage-2-blocked one (never scored) both read as blank.
            'svm_probability': (
                round(self.svm_probability, 4)
                if self.svm_probability is not None else ''
            ),
            'svm_prediction': (
                self.svm_prediction if self.svm_prediction is not None else ''
            ),
            'stage_blocked': self.stage_blocked,
        }


# ── SIGNAL PROCESSING PIPELINE (Stage 2) ────────────────────────────────────

def _get_frame_envelope_pair(
    dm,
    frame_idx: int,
    normalize: bool = True,
) -> Tuple:
    """
    Reconstruct current frame + its neighbours, return envelopes and signals.

    Centralises ALL iFFT/Hilbert work for a candidate so the caller never needs
    to call reconstruct_frame_v5 a second time for the same frame.

    Returns:
        prev_envelope : ndarray or None
        curr_envelope : ndarray
        next_envelope : ndarray or None
        curr_signal   : ndarray
        prev_signal   : ndarray or None
        next_signal   : ndarray or None  ← needed to build the stitched context
        curr_fft_norm : ndarray  ← mic-corrected FFT magnitudes (full half-spectrum)
        curr_freq_axis: ndarray  ← frequency axis [Hz] for curr_fft_norm
    """
    from core.click_pipeline_v5 import (
        reconstruct_frame_v5,
        compute_hilbert_envelope,
        FS, FFT_SIZE,
    )

    envelopes  = {}
    signals    = {}
    curr_fft_norm  = None
    curr_freq_axis = None

    frames_to_load = [frame_idx]
    if frame_idx > 0:
        frames_to_load.append(frame_idx - 1)
    if frame_idx < len(dm.fft_mags) - 1:
        frames_to_load.append(frame_idx + 1)

    for idx in frames_to_load:
        fd = reconstruct_frame_v5(
            dm.fft_mags[idx], dm.phase_int8[idx],
            FS, FFT_SIZE, normalize=normalize,
        )
        if fd is None:
            continue
        signals[idx]   = fd['signal']
        envelopes[idx] = compute_hilbert_envelope(fd['signal'])
        if idx == frame_idx:          # keep FFT data for the candidate frame
            curr_fft_norm  = fd['fft_norm']
            curr_freq_axis = fd['freq_axis']

    prev_envelope = envelopes.get(frame_idx - 1) if frame_idx > 0 else None
    curr_envelope = envelopes[frame_idx]
    next_envelope = envelopes.get(frame_idx + 1) if frame_idx < len(dm.fft_mags) - 1 else None
    curr_signal   = signals[frame_idx]
    prev_signal   = signals.get(frame_idx - 1) if frame_idx > 0 else None
    next_signal   = signals.get(frame_idx + 1) if frame_idx < len(dm.fft_mags) - 1 else None

    return (prev_envelope, curr_envelope, next_envelope,
            curr_signal, prev_signal, next_signal,
            curr_fft_norm, curr_freq_axis)


def _process_file_for_collection(
    dm,
    k: float = K_DEFAULT,
    normalize: bool = True,
    stop_check=None,
    progress_cb=None,
    export_mode: str = EXPORT_ALL,
    stage2_mode: str = None,
) -> Tuple[List[CandidateData], List[Dict]]:
    """
    Process single .paudio file: find Stage 1 survivors and compute all features.

    Args (additional):
        stop_check  : Optional callable() → bool. Called before each survivor.
                      Return True to abort early (partial results returned).
        progress_cb : Optional callable(i, total) emitted every 10 survivors so
                      the dialog can show that analysis is progressing.

    Fast path: if dm already carries the pre-computed arrays produced by
    AudioLoadWorker (fft_means, E_hat_floor_arr, noise_floor_arr, std_noise_arr),
    Stage 1 is a pure-numpy threshold + run-length filter — no per-frame iFFT
    or Hilbert transform needed (that was already done during file loading).
    This avoids re-processing every frame a second time, which was the main
    performance bottleneck.

    Slow-path fallback (run_stage1_v5) is used when those arrays are absent.

    Args:
        dm: AudioDataManager with loaded file
        k: Stage 1 multiplier (default 1.5)
        normalize: Use normalized FFT if True

    Returns:
        Tuple of:
        - candidates: List of CandidateData objects (one per Stage 1 survivor)
        - csv_rows: List of CSV-ready dicts (for convenience)
    """
    # Lazy import to avoid circular imports
    from core.click_pipeline_v5 import (
        run_stage1_v5,
        run_stage1_v5_precomputed,
        has_precomputed_stage1_arrays,
        build_click_context,
        resolve_click,
        click_event_key,
        p_noise_at,
        p_noise_frames_at,
        compute_features_v5,
        FS, FFT_SIZE,
        STAGE2_R2_MIN, STAGE2_TAU_MIN,
        STAGE1_MODE, PEAK_REFRACTORY_R, LOCAL_CREST_C, STAGE2_MODE,
    )

    # Stamped on every row. R and C are PROVISIONAL (Stage 1 v5.1 spec §4.1, O-1):
    # Phase 0 could not validate them, so recording the values used is the only
    # thing that lets a later study attribute results to them.
    stage1_params = (f'{STAGE1_MODE};k={k:.2f};'
                     f'R={PEAK_REFRACTORY_R};C={LOCAL_CREST_C}')
    # The Stage 2 tier is stamped even when Stages 2-4 do not run, because the
    # question a reader asks later is "which rule COULD have rejected this row",
    # and a blank there is indistinguishable from "the default happened to apply".
    stage2_mode_used = stage2_mode or STAGE2_MODE

    candidates = []
    csv_rows   = []

    # ── Stage 1: find above-threshold candidates ──────────────────────────────
    # Fast path: AudioLoadWorker already computed per-frame energy and adaptive
    # noise estimates.  Reuse them — no iFFT/Hilbert needed here.
    if has_precomputed_stage1_arrays(dm):
        candidates_raw = run_stage1_v5_precomputed(dm, k=k)
    else:
        # Slow path: recompute noise estimator from scratch (file was loaded
        # without pre-computed arrays, e.g. older .paudio format).
        candidates_raw = run_stage1_v5(dm, k=k)

    if not candidates_raw:
        return candidates, csv_rows

    n_survivors = len(candidates_raw)

    # Process each survivor
    for i, survivor in enumerate(candidates_raw):
        # Honour cancel request — return partial results already built
        if stop_check is not None and stop_check():
            break

        # Emit periodic progress so the dialog label doesn't appear frozen
        if progress_cb is not None and i % 10 == 0:
            progress_cb(i, n_survivors)

        frame_idx = survivor['frame_idx']

        try:
            # Single iFFT reconstruction pass — _get_frame_envelope_pair now
            # returns curr_fft_norm and curr_freq_axis so we never reconstruct
            # the same frame twice.
            (prev_env, curr_env, next_env,
             curr_sig, prev_sig, next_sig,
             curr_fft_norm, curr_freq_axis) = _get_frame_envelope_pair(
                dm, frame_idx, normalize=normalize
            )

            noise_floor = survivor['noise_floor']
            std_noise   = survivor['std_noise']

            # Stitch prev|curr|next and resolve the click on that continuous
            # trace, then compute all features in one coordinate system.
            ctx      = build_click_context(prev_sig, curr_sig, next_sig)
            resolved = resolve_click(ctx, noise_floor, std_noise)
            # v6: Buffer 3's per-bin noise PSD in effect at this frame. None when
            # the recording was loaded without the v6 arrays (anything loaded by an
            # older build) — compute_features_v5 then emits the v6 features as NaN
            # and every v5 value is unchanged.
            p_noise = p_noise_at(dm, frame_idx)

            # Did suppress_edge_artifacts (reconstruct_frame_v5 Step 3) fire on this
            # frame? When it does, the reconstruction is no longer the exact inverse
            # transform of taper·A[k], so P_region != taper²·P_noise and the v6
            # subtraction is biased for this event. The fade's first coefficient is
            # 0.5·(1−cos 0) = 0 exactly, so a hard zero at sample 0 is the signature;
            # nothing else in the chain produces one. Measured to fire on 0 of 300
            # dispersed-phase frames, so this should be rare — which is precisely why
            # it is worth recording rather than assuming.
            gibbs_fired = int(len(curr_sig) > 0 and curr_sig[0] == 0.0)

            # The three curves the screenshot's FFT panel draws, all on one
            # amplitude axis. Computed here because ctx / resolved / p_noise are
            # already in hand; see _region_display_spectrum for the maths.
            _disp_freqs, _disp_region, _disp_noise = _region_display_spectrum(
                ctx, resolved, p_noise, curr_freq_axis, FS)
            features = compute_features_v5(
                ctx, resolved,
                curr_fft_norm, curr_freq_axis,
                noise_floor, std_noise, FS,
                p_noise_psd=p_noise,
            )
            peak_abs, canonical_frame_idx = click_event_key(ctx, resolved, frame_idx)

            # Timestamp — taken from the loader's own array when it exists, so the
            # CSV and the app's time axis are the SAME numbers by construction and
            # cannot drift apart again.
            #
            # They had drifted: the app used a hardcoded `estimated_fft_rate = 390.0`
            # (2.564103 ms/frame) while this line computed fs/fft_size
            # (2.560000 ms/frame). 0.16 % apart, accumulating to ~+10 s at the end of
            # a 110-minute recording, with the CSV reading short. The constant is now
            # fixed at source; reading the array here removes the duplication that
            # allowed the two to disagree in the first place.
            #
            # The fallback is the same formula as before, for a dm without the array.
            _ts = getattr(dm, 'fft_timestamps', None)
            if _ts is not None and frame_idx < len(_ts):
                timestamp_s = float(_ts[frame_idx])
            else:
                timestamp_s = frame_idx / (FS / FFT_SIZE)

            env_ctx  = ctx['envelope']
            sig_ctx  = ctx['signal']
            peak     = resolved['peak']
            onset    = resolved['onset']
            d_start  = resolved['decay_start']
            d_end    = resolved['decay_end']

            # ── Render window: 2.56 ms centred on the peak, widened so the whole
            #    click (onset → decay_end) always fits. Time axis is peak-relative
            #    ms, so the picture is identical regardless of frame alignment. ──
            half = FFT_SIZE // 2
            w0   = max(0, min(peak - half, onset))
            w1   = min(len(sig_ctx), max(peak + half, d_end + 1))
            render_signal   = sig_ctx[w0:w1]
            render_envelope = env_ctx[w0:w1]
            render_t_ms     = (np.arange(w0, w1, dtype=np.float64) - peak) / FS * 1000.0
            mark_onset_ms     = (onset - peak) / FS * 1000.0
            mark_decay_end_ms = (d_end - peak) / FS * 1000.0
            seams_ms = [float((s - peak) / FS * 1000.0)
                        for s in ctx['seams'] if w0 <= s < w1]

            # Pre-compute the exponential fit curve for the renderer, spanning the
            # REAL decay window on the stitched envelope — no frame-edge clip.
            fit_t_ms_arr = None
            fit_y_arr    = None
            tau_ms_val   = features.get('tau_ms', -1.0)
            R2_val       = features.get('R2', 0.0)
            if tau_ms_val > 0 and R2_val > 0.1 and d_end > d_start + 2:
                try:
                    n_arr        = np.arange(d_end - d_start, dtype=np.float64)
                    tau_s        = max(tau_ms_val, 0.01) / 1000.0
                    rate         = 1.0 / (tau_s * FS)
                    A0           = float(env_ctx[min(d_start, len(env_ctx) - 1)])
                    fit_t_ms_arr = (np.arange(d_start, d_end, dtype=np.float64) - peak) / FS * 1000.0
                    fit_y_arr    = A0 * np.exp(-rate * n_arr)
                except Exception:
                    pass

            ### FILTERS ### (also skips screenshot rendering)

            # Drop candidates with no exponential fit at all: R² is exactly 0 (the
            # degenerate-window result) or τ is the −1 sentinel. There is nothing to
            # see in their screenshot and nothing to learn from their features, which
            # are artefacts of the fallback decay window.
            #
            # Deliberately NOT the full Stage 2 fit gate (R² < STAGE2_R2_MIN): rows
            # with a low-but-nonzero R² are still exported and come back tagged
            # 'Stage2_R2'. They are a large, legitimate part of the existing training
            # CSVs (~23% of the current dataset), so filtering them here would silently
            # change what future exports contain. Stage 2 is what rejects them, and it
            # says so in the CSV.
            #
            # v6: the two sentinel tests above became one flag. `fit_valid` is EXACTLY
            # equivalent — tau_ms <= 0 ⟺ tau_ms == −1 ⟺ fit_valid == 0, since a
            # converged fit has slope < 0 and therefore tau > 0 always (verified over
            # 3000 random windows in test_scripts/verify_v6_buffer3.py §6). So this
            # swap does not change which rows are exported.
            #
            # EXPORT_UNFILTERED skips the filter entirely. Those rows are the ones
            # that have never reached a CSV in the project's history, and they arrive
            # with tau_ms / R2 / fit_coverage as NaN — which is why the sentinels had
            # to go before this mode could exist.
            if export_mode != EXPORT_UNFILTERED and not features.get('fit_valid', 0):
                continue

            ### ------- ###

            # Create candidate object
            candidate = CandidateData(
                file=dm.filename,
                frame_idx=frame_idx,
                timestamp_s=timestamp_s,
                noise_floor=survivor['noise_floor'],
                std_noise=survivor['std_noise'],
                E_hat_floor=survivor['E_hat_floor'],
                peak_SNR=features.get('peak_SNR', 0.0),
                pre_SNR=features.get('pre_SNR', 0.0),
                post_SNR=features.get('post_SNR', 0.0),
                rise_time_ms=features.get('rise_time_ms', 0.0),
                fall_time_ms=features.get('fall_time_ms', 0.0),
                asymmetry_integral=features.get('asymmetry_integral', 0.0),
                ZCR_pre=features.get('ZCR_pre', 0.0),
                ZCR_click=features.get('ZCR_click', 0.0),
                ZCR_post=features.get('ZCR_post', 0.0),
                kurtosis=features.get('kurtosis', 0.0),
                centroid_shift_hz=features.get('centroid_shift_hz', 0.0),
                tau_ms=features.get('tau_ms', 0.0),
                R2=features.get('R2', 0.0),
                fit_coverage=features.get('fit_coverage', 0.0),
                SPR=features.get('SPR', 0.0),
                R_spectral=features.get('R_spectral', 0.0),
                FPE_hz=features.get('FPE_hz', 0.0),
                peak_abs=peak_abs,
                canonical_frame_idx=canonical_frame_idx,
                session_id=dm.filename,
                # ── v6 features. NaN default, so a recording loaded without the
                #    Buffer-3 arrays yields NaN rather than a plausible-looking 0. ──
                spectral_entropy=features.get('spectral_entropy', float('nan')),
                shape_novelty=features.get('shape_novelty', float('nan')),
                spectral_tilt=features.get('spectral_tilt', float('nan')),
                temporal_concentration=features.get('temporal_concentration', float('nan')),
                FPE_hz_region=features.get('FPE_hz_region', float('nan')),
                SPR_region=features.get('SPR_region', float('nan')),
                f_50_hz=features.get('f_50_hz', float('nan')),
                IQR_f=features.get('IQR_f', float('nan')),
                # ── validity & quality ──
                fit_valid=int(features.get('fit_valid', 0)),
                decay_len=int(d_end - d_start),
                n_seg=int(features.get('n_seg', 0)),
                b3_frames=p_noise_frames_at(dm, frame_idx),
                gibbs_fired=gibbs_fired,
                # ── Stage 1 v5.1 (§4.3, §4.4) ──
                # local_crest and the run diagnostics are produced by
                # _stage1_select and ride along on the survivor dict; they cannot
                # be recomputed here, because that would need the whole per-frame
                # energy series and would be a second implementation of the rule.
                local_crest=float(survivor.get('local_crest', float('nan'))),
                harmonic_confinement=features.get('harmonic_confinement', float('nan')),
                hc_f1_hz=features.get('hc_f1_hz', float('nan')),
                hc_r_A=features.get('hc_r_A', float('nan')),
                hc_r_B=features.get('hc_r_B', float('nan')),
                stage1_params=stage1_params,
                stage2_mode=stage2_mode_used,
                run_id=int(survivor.get('run_id', -1)),
                run_length=int(survivor.get('run_length', 0)),
                run_crest=float(survivor.get('run_crest', float('nan'))),
                pos_in_run=int(survivor.get('pos_in_run', 0)),
                would_pass_v5=int(survivor.get('would_pass_v5', 0)),
                k_ratio=(float(survivor['E_i']) / float(survivor['E_hat_floor'])
                         if survivor.get('E_hat_floor', 0.0) > 0 else float('nan')),
                render_signal=render_signal,
                render_envelope=render_envelope,
                render_t_ms=render_t_ms,
                mark_onset_ms=mark_onset_ms,
                mark_decay_end_ms=mark_decay_end_ms,
                seams_ms=seams_ms,
                fft_norm=curr_fft_norm,
                freq_axis=curr_freq_axis,
                region_freqs=_disp_freqs,
                region_amp=_disp_region,
                noise_amp=_disp_noise,
                fit_t_ms=fit_t_ms_arr,
                fit_y=fit_y_arr,
                label='',
            )
            
            candidates.append(candidate)
            csv_rows.append(candidate.to_csv_dict())
        
        except Exception as e:
            print(f"Warning: Failed to process frame {frame_idx}: {e}")
            traceback.print_exc()
            continue
    
    return candidates, csv_rows


# ── SCREENSHOT RENDERING (Stage 3) ──────────────────────────────────────────
# Pure QPainter + QImage — no pyqtgraph, no OpenGL context, no segfaults.
# All numpy/scipy work is pre-computed in the worker thread; this section
# only does coordinate mapping and QPainter draw calls on the main thread.

# Inner padding (px) reserved inside each panel rect for axes / labels / title
_PL = 62   # left  — y-axis ticks + label
_PR = 8    # right
_PT = 26   # top   — panel title
_PB = 40   # bottom — x-axis ticks + label


def _px_rect(panel: QRect) -> QRect:
    """Return the inner plot area (inside axis padding) for a panel QRect."""
    return QRect(
        panel.left()   + _PL,
        panel.top()    + _PT,
        panel.width()  - _PL - _PR,
        panel.height() - _PT - _PB,
    )


def _draw_axes(
    p: QPainter, panel: QRect,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    title: str, x_label: str, y_label: str,
    c_grid: QColor, c_axis: QColor, c_title: QColor,
    n_x: int = 6, n_y: int = 5,
):
    """Draw grid lines, tick labels, axis labels and panel title."""
    r     = _px_rect(panel)
    x_rng = max(x_max - x_min, 1e-30)
    y_rng = max(y_max - y_min, 1e-30)

    # Grid lines and tick labels
    p.setFont(QFont("Arial", 7))
    for i in range(n_x + 1):
        xv = x_min + i * x_rng / n_x
        px = int(r.left() + i / n_x * r.width())
        p.setPen(QPen(c_grid, 1))
        p.drawLine(px, r.top(), px, r.bottom())
        p.setPen(QPen(c_axis, 1))
        p.drawText(QRect(px - 22, r.bottom() + 3, 44, 15), Qt.AlignCenter, f"{xv:.3g}")

    for i in range(n_y + 1):
        yv = y_min + i * y_rng / n_y
        py = int(r.bottom() - i / n_y * r.height())
        p.setPen(QPen(c_grid, 1))
        p.drawLine(r.left(), py, r.right(), py)
        p.setPen(QPen(c_axis, 1))
        p.drawText(QRect(panel.left() + 2, py - 8, _PL - 5, 16),
                   Qt.AlignRight | Qt.AlignVCenter, f"{yv:.3g}")

    # Axis border
    p.setPen(QPen(c_axis, 1))
    p.drawRect(r)

    # Panel title
    p.setFont(QFont("Arial", 9, QFont.Bold))
    p.setPen(c_title)
    p.drawText(QRect(panel.left(), panel.top() + 3, panel.width(), _PT - 3),
               Qt.AlignCenter, title)

    # X-axis label
    p.setFont(QFont("Arial", 8))
    p.setPen(c_axis)
    p.drawText(QRect(panel.left(), r.bottom() + 21, panel.width(), 16),
               Qt.AlignCenter, x_label)

    # Y-axis label (rotated 90°)
    p.save()
    p.translate(panel.left() + 11, r.top() + r.height() // 2)
    p.rotate(-90)
    p.drawText(QRect(-55, -8, 110, 16), Qt.AlignCenter, y_label)
    p.restore()


def _draw_line(
    p: QPainter, panel: QRect,
    x_arr: np.ndarray, y_arr: np.ndarray,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    color: QColor, lw: int = 1, dashed: bool = False,
):
    """Map x_arr/y_arr from data coords to pixels within panel and draw a polyline.

    Skips any point where x or y is non-finite (inf / nan) — these arise from
    corrupted FFT frames whose energy overflowed. Pixel coordinates are also
    clamped to ±30 000 so Qt's signed-int32 limit is never exceeded.
    """
    if len(x_arr) < 2:
        return
    r   = _px_rect(panel)
    xr  = max(x_max - x_min, 1e-30)
    yr  = max(y_max - y_min, 1e-30)
    rw  = r.width()
    rh  = r.height()
    rl  = r.left()
    rb  = r.bottom()
    # Qt uses signed 32-bit integers for coordinates; clamp well inside that range.
    _MAX_COORD = 30_000
    style = Qt.DashLine if dashed else Qt.SolidLine
    p.setPen(QPen(color, lw, style))

    prev_px: Optional[int] = None
    prev_py: Optional[int] = None

    for i in range(len(x_arr)):
        xv = float(x_arr[i])
        yv = float(y_arr[i])
        if not (np.isfinite(xv) and np.isfinite(yv)):
            # Non-finite point → break the polyline here (don't connect across gap)
            prev_px = prev_py = None
            continue
        cx = int(np.clip(rl + (xv - x_min) / xr * rw, -_MAX_COORD, _MAX_COORD))
        cy = int(np.clip(rb  - (yv - y_min) / yr * rh, -_MAX_COORD, _MAX_COORD))
        if prev_px is not None:
            p.drawLine(prev_px, prev_py, cx, cy)
        prev_px, prev_py = cx, cy


def _draw_hline(
    p: QPainter, panel: QRect,
    y_val: float, y_min: float, y_max: float,
    color: QColor, lw: int = 1,
):
    """Draw a horizontal reference line at y_val across the inner plot area."""
    if not (y_min <= y_val <= y_max):
        return
    r  = _px_rect(panel)
    yr = max(y_max - y_min, 1e-30)
    py = int(r.bottom() - (y_val - y_min) / yr * r.height())
    p.setPen(QPen(color, lw, Qt.DashLine))
    p.drawLine(r.left(), py, r.right(), py)


def _draw_vline(
    p: QPainter, panel: QRect,
    x_val: float, x_min: float, x_max: float,
    color: QColor, lw: int = 1, dashed: bool = True, label: str = None,
):
    """Draw a vertical reference line at x_val across the inner plot area."""
    if not (x_min <= x_val <= x_max):
        return
    r  = _px_rect(panel)
    xr = max(x_max - x_min, 1e-30)
    px = int(r.left() + (x_val - x_min) / xr * r.width())
    p.setPen(QPen(color, lw, Qt.DashLine if dashed else Qt.SolidLine))
    p.drawLine(px, r.top(), px, r.bottom())
    if label:
        p.setFont(QFont("Arial", 8))
        p.setPen(color)
        p.drawText(px + 2, r.top() + 10, label)


def _region_display_spectrum(ctx, resolved, p_noise_psd, freq_axis, fs):
    """
    Build the three curves the screenshot's FFT panel draws, all on ONE amplitude
    axis in volts: the region spectrum, and Buffer 3's noise floor.

    Returns (freqs, region_amp, noise_amp); any of them empty when unavailable.

    ── WHAT IS AND IS NOT COMPARABLE ON THIS AXIS ───────────────────────────────
    Read this before trusting the picture. No single scaling makes all three curves
    quantitatively comparable — that is structural, not an implementation choice
    (SPECTRAL_FEATURES_v6_PROPOSAL.md §3.2): each signal class scales differently
    with segment length N under amplitude scaling.

        coherent tone spanning the window   invariant
        stationary noise                    ∝ N^(-1/2)
        finite transient (the click)        ∝ N^(-1)

    ✅ REGION vs NOISE is quantitative, and it is the comparison that matters. The
       noise line is converted below to the amplitude that Buffer 3's power density
       produces WHEN MEASURED WITH THIS REGION'S OWN WINDOW, so the vertical gap
       between the region curve and the noise line IS the per-frequency excess the
       v6 features are computed from.

    ⚠️ REGION vs FRAME is NOT quantitative. A click is a finite transient, so its
       amplitude spectrum falls as N^(-1): the same click measured over a
       60-sample region and over the 512-sample frame differs by ~8x in amplitude
       for reasons that have nothing to do with the signal. The frame curve is
       drawn as CONTEXT — where the frame's energy sits in frequency — and is
       labelled as such in the legend. Do not read a ratio off it.

    The noise conversion:

        A_rms(noise) = sqrt( 2 · P_noise · Δf )        Δf = spec.enbw_hz

    Derivation: amplitude A = 2|X|/Σw and psd P = 2|X|²/(fs·Σw²) give
    A² = 2·P·fs·Σw²/(Σw)², and NENBW = n·Σw²/(Σw)², so A² = 2·P·(NENBW·fs/n) =
    2·P·Δf. Verified against measured spectra at n_seg 30/58/90/150/512: ratio
    0.9993-1.0033, i.e. within 0.3 % at every region length.

    With that conversion the vertical gap between the region curve and the noise
    line IS the per-frequency excess the v6 features measure, so the picture and
    the numbers agree. Drawing the raw PSD on an mV axis instead would look
    plausible and mean nothing.
    """
    empty = (np.array([]), np.array([]), np.array([]))
    try:
        from core.click_pipeline_v5 import (_spectral, analysis_band_taper,
                                            _BIN_START, _BIN_END, _K_BINS,
                                            REGION_NFFT)
        sa = _spectral()

        signal = ctx['signal']
        i0 = max(0, int(resolved['onset']))
        i1 = min(len(signal), int(resolved['decay_end']) + 1)
        if i1 - i0 < sa.MIN_SEGMENT_SAMPLES:
            return empty

        spec = sa.compute_spectrum(signal[i0:i1], fs,
                                   window=sa.DEFAULT_WINDOW, alpha=sa.DEFAULT_ALPHA,
                                   n_fft=REGION_NFFT, scaling='amplitude')
        freqs = spec.freqs
        region_amp = spec.mags

        noise_amp = np.array([])
        if p_noise_psd is not None:
            pn = np.asarray(p_noise_psd, dtype=np.float64)
            if pn.shape == (_K_BINS,) and np.all(np.isfinite(pn)):
                # Trap (a): B3 is untapered, the region is not. Match them first.
                pn = pn * analysis_band_taper(_K_BINS) ** 2
                nf = np.asarray(freq_axis, dtype=np.float64)[_BIN_START:_BIN_END + 1]
                pn_on_region = np.interp(freqs, nf, pn, left=np.nan, right=np.nan)
                noise_amp = np.sqrt(np.maximum(2.0 * pn_on_region * spec.enbw_hz, 0.0))

        return freqs, region_amp, noise_amp
    except Exception:                                          # noqa: BLE001
        # A screenshot must never be the reason an export fails.
        return empty


def _draw_feature_footer(
    p: QPainter, c: 'CandidateData', rect: QRect,
    c_text: QColor, c_key: QColor,
):
    """
    Draw every feature + noise info as readable multi-line text below the plots.

    Each group is one line: a bold label on the left, then values separated by │.

    Covers the 17 v5 features, the 8 v6 features, and a Quality line carrying the
    validity flags. The Quality line is the important one — fit_valid, b3_frames,
    n_seg_valid and gibbs_fired are what tell a reviewer that the other numbers on
    the page are not to be trusted, and without it a failed row looks identical to
    a good one.

    Font is 11 pt (was 9). At 9 pt the review dialog considered this footer
    unreadable and cropped it away entirely, re-rendering the numbers itself.
    """
    LINE_H = 28          # was 22 — the 9 pt footer was unreadable once rescaled

    def _num(v, fmt, bad="n/a"):
        """NaN-safe formatting. A NaN here is a real state, not a glitch: it means
        the quantity could not be measured, and printing 'nan' hides that."""
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return bad
        return bad if fv != fv else format(fv, fmt)

    tau_str = f"{c.tau_ms:.4f} ms" if c.tau_ms == c.tau_ms and c.tau_ms > 0 else "N/A"

    # (label, [value strings]) — one line per group
    groups = [
        ("Noise",    [f"floor = {c.noise_floor * 1e3:.4f} mV",
                      f"std = {c.std_noise * 1e3:.4f} mV",
                      f"Ê_floor = {c.E_hat_floor:.3e}"]),
        ("SNR",      [f"peak = {c.peak_SNR:.2f}",
                      f"pre = {c.pre_SNR:.2f}",
                      f"post = {c.post_SNR:.2f}"]),
        ("Shape",    [f"rise = {c.rise_time_ms:.4f} ms",
                      f"fall = {c.fall_time_ms:.4f} ms",
                      f"asymmetry = {c.asymmetry_integral:.4f}",
                      f"kurtosis = {c.kurtosis:.2f}"]),
        ("ZCR",      [f"pre = {c.ZCR_pre:.3f}",
                      f"click = {c.ZCR_click:.3f}",
                      f"post = {c.ZCR_post:.3f}"]),
        ("Spec v5",  [f"centroid_shift = {c.centroid_shift_hz:.0f} Hz",
                      f"SPR = {c.SPR:.2f}",
                      f"R_spectral = {c.R_spectral:.3f}",
                      f"FPE = {c.FPE_hz:.0f} Hz"]),
        # ── v6, computed on the excess spectrum E[k] = max(0, P_region − P_noise) ──
        ("Spec v6",  [f"entropy = {_num(c.spectral_entropy, '.3f')}",
                      f"novelty = {_num(c.shape_novelty, '.3f')}",
                      f"tilt = {_num(c.spectral_tilt, '+.3f')} dB/kHz",
                      f"t_conc = {_num(c.temporal_concentration, '.3f')}"]),
        ("Location", [f"FPE_region = {_num(c.FPE_hz_region, '.0f')} Hz",
                      f"SPR_region = {_num(c.SPR_region, '.2f')}",
                      f"f_50 = {_num(c.f_50_hz, '.0f')} Hz",
                      f"IQR_f = {_num(c.IQR_f, '.0f')} Hz"]),
        # Frame-domain, unlike every other v6 line above — see its spec §1.
        ("Harmonic", [f"confinement = {_num(c.harmonic_confinement, '+.2f')}",
                      f"f₁ = {_num(c.hc_f1_hz, '.0f')} Hz",
                      f"r_A = {_num(c.hc_r_A, '.2f')}",
                      f"r_B = {_num(c.hc_r_B, '.2f')}"]),
        ("Fit",      [f"τ = {tau_str}",
                      f"R² = {_num(c.R2, '.4f')}",
                      f"coverage = {_num(c.fit_coverage, '.3f')}",
                      f"decay_len = {c.decay_len}"]),
    ]

    # ── Quality — the group that says whether the numbers above mean anything ──
    # fit_valid = 0 makes τ / R² / coverage meaningless; b3_frames = 0 makes every
    # v6 feature meaningless; n_seg < 45 biases entropy toward 1 (§4.3). Without
    # this line a failed row shows a full table of plausible-looking garbage.
    q = []
    q.append("fit_valid = 1" if c.fit_valid else "⚠ FIT INVALID — τ/R²/coverage not meaningful")
    if c.b3_frames:
        q.append(f"b3_frames = {c.b3_frames}")
    else:
        q.append("⚠ NO B3 ESTIMATE — v6 features unavailable")
    # Derived, not stored: v6 dropped the redundant n_seg_valid column.
    from core.spectral_analysis import V6_MIN_NSEG
    q.append(f"n_seg = {c.n_seg}" if c.n_seg >= V6_MIN_NSEG
             else f"⚠ n_seg = {c.n_seg} too short — bands correlated, entropy biased high")
    if c.gibbs_fired:
        q.append("⚠ Gibbs fade fired — subtraction biased on this frame")
    groups.append(("Quality", q))

    # ── Stage 1 v5.1 — why this frame is on the page at all ──────────────────
    # would_pass_v5 = 0 marks a candidate v5 would have DELETED for sitting in a
    # run longer than MAX_RUN. Those rows have never been labelled by anyone, so
    # a reviewer has to be able to see it on the picture rather than infer it.
    s1 = [f"local_crest = {_num(c.local_crest, '.2f')}",
          f"k_ratio = {_num(c.k_ratio, '.2f')}",
          f"run L = {c.run_length} (pos {c.pos_in_run})",
          f"run_crest = {_num(c.run_crest, '.2f')}"]
    if not c.would_pass_v5:
        s1.append("NEW under v5.1 — v5 deleted this run (L > MAX_RUN)")
    groups.append(("Stage 1", s1))

    # Verdict line — only when the candidate went through Stages 2-4. Without it a
    # PNG shows what the frame looked like but not what the algorithm made of it.
    if c.svm_prediction is not None or c.stage_blocked:
        if c.svm_probability is not None:
            prob_str = f"P(click) = {c.svm_probability:.3f}"
        else:
            prob_str = "P(click) = n/a (never reached the SVM)"
        verdict = "CONFIRMED CLICK" if c.is_confirmed_click else f"blocked at {c.stage_blocked}"
        groups.append(("Verdict", [verdict, prob_str]))

    f_key = QFont("Arial", 11, QFont.Bold)      # was 9 pt
    f_val = QFont("Courier New", 11)            # was 9 pt
    KEY_W = 92  # pixels reserved for group label column (wider labels at 11 pt)

    # Wrap a group across several lines rather than letting it run off the panel.
    # The Quality group in particular can carry four warnings at once, and a
    # truncated warning is worse than none — it looks like the row is fine.
    from PySide6.QtGui import QFontMetrics
    SEP = "   │   "
    avail = rect.width() - KEY_W
    fm = QFontMetrics(f_val)

    def _wrap(values):
        lines, cur = [], []
        for v in values:
            trial = SEP.join(cur + [v])
            if cur and fm.horizontalAdvance(trial) > avail:
                lines.append(SEP.join(cur))
                cur = [v]
            else:
                cur.append(v)
        if cur:
            lines.append(SEP.join(cur))
        return lines

    # Flattened first, so an overflow can be REPORTED rather than silently
    # dropping whatever happened to be drawn last. It once ate the Verdict group
    # and the only symptom was a missing line on a 1400 px PNG.
    flat = [(label, n, line)
            for label, values in groups
            for n, line in enumerate(_wrap(values))]
    n_fit = max(0, (rect.height() - 6) // LINE_H)
    truncated = len(flat) - n_fit
    if truncated > 0:
        flat = flat[:max(0, n_fit - 1)]

    y = rect.top() + 6
    for label, n, line in flat:
        if n == 0:
            p.setFont(f_key)
            p.setPen(c_key)
            p.drawText(QRect(rect.left(), y, KEY_W, LINE_H),
                       Qt.AlignLeft | Qt.AlignVCenter, label + ":")
        p.setFont(f_val)
        p.setPen(c_text)
        p.drawText(QRect(rect.left() + KEY_W, y, avail, LINE_H),
                   Qt.AlignLeft | Qt.AlignVCenter, line)
        y += LINE_H

    if truncated > 0:
        p.setFont(f_key)
        p.setPen(QColor(200, 60, 60))
        p.drawText(QRect(rect.left(), y, rect.width(), LINE_H),
                   Qt.AlignLeft | Qt.AlignVCenter,
                   f"⚠ {truncated + 1} more footer line(s) did not fit — "
                   f"raise SCREENSHOT_HEIGHT")


def _render_candidate_screenshot(
    candidate: CandidateData,
    output_path: Path,
) -> bool:
    """
    Render two-panel screenshot using QPainter + QImage.

    Completely avoids pyqtgraph and OpenGL — the previous pyqtgraph approach
    caused segfaults on macOS because container.render() cannot capture
    GL-backed PlotWidget content, and QApplication.processEvents() inside a
    queued signal handler creates dangerous re-entrancy.

    Layout (1400 × 800 px):
      ┌──────────────────────── title bar (36 px) ──────────────────────────┐
      │  Frame XXXXXX  │  t = XX.XXXX s  │  filename                       │
      ├──── FFT panel (40 %) ──────┬─── iFFT panel (60 %) ─────────────────┤
      │  Frequency (kHz)           │  Time (ms)                            │
      │  Amplitude (mV / µV / V)   │  Amplitude (mV / µV / V)             │
      │  linear scale              │  signal (teal) + envelope (red)       │
      │                            │  fit (green dashed, if R²>0.1)        │
      │                            │  noise floor (cyan) + ±std (purple)   │
      ├──────────────── feature footer (6 lines) ───────────────────────────┤
      │  Noise:    floor = … mV  │  std = … mV  │  Ê_floor = …            │
      │  SNR:      peak = …  │  pre = …  │  post = …                       │
      │  Shape:    rise = … ms  │  fall = … ms  │  asymmetry = …           │
      │  ZCR:      pre = …  │  click = …  │  post = …                      │
      │  Spectral: centroid_shift = … Hz  │  SPR = …  │  FPE = … Hz        │
      │  Fit:      τ = … ms  │  R² = …  │  coverage = …                   │
      └─────────────────────────────────────────────────────────────────────┘

    Args:
        candidate: CandidateData with signal, envelope, fft data, pre-computed fit
        output_path: Path to save PNG file

    Returns:
        True if saved successfully, False otherwise
    """
    try:
        from core.click_pipeline_v5 import FS

        # ── Layout constants ──────────────────────────────────────────────────
        W, H       = SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT
        MARGIN     = SCREENSHOT_MARGIN
        HEADER_H   = SCREENSHOT_HEADER_H
        PLOT_H     = PANEL_HEIGHT       # height of each plot panel
        GAP        = SCREENSHOT_GAP     # gap between header/panels/footer
        FOOTER_Y   = HEADER_H + MARGIN + PLOT_H + GAP
        FOOTER_H   = SCREENSHOT_FOOTER_H

        FFT_W  = int((W - 3 * MARGIN) * 0.40)
        IFFT_W = W - 3 * MARGIN - FFT_W

        FFT_panel   = QRect(MARGIN,            HEADER_H + MARGIN, FFT_W,  PLOT_H)
        IFFT_panel  = QRect(2*MARGIN + FFT_W,  HEADER_H + MARGIN, IFFT_W, PLOT_H)
        FOOTER_rect = QRect(MARGIN, FOOTER_Y, W - 2*MARGIN, FOOTER_H)

        # ── Colour palette ────────────────────────────────────────────────────
        C_BG     = QColor('#1e1e2e')
        C_PANEL  = QColor('#12121f')
        C_BORDER = QColor('#3a3a55')
        C_TITLE  = QColor('#e8e8e8')
        C_AXIS   = QColor('#8888aa')
        C_GRID   = QColor(36, 36, 56)
        C_FFT    = QColor('#4ea6dc')   # FFT trace — steel blue
        C_SIG    = QColor('#80cbc4')   # raw signal — teal
        C_ENV    = QColor('#ef5350')   # envelope — red
        C_FIT    = QColor('#00E676')   # fit curve — green
        C_FLOOR  = QColor('#00CED1')   # noise floor line — cyan
        C_FRAME  = QColor('#42506b')   # whole-frame FFT — muted, sits behind
        C_STD    = QColor('#9370DB')   # noise+std line — purple
        C_TEXT   = QColor('#cccccc')
        C_KEY    = QColor('#7aadcc')

        # ── Create image ──────────────────────────────────────────────────────
        img = QImage(W, H, QImage.Format_RGB32)
        img.fill(C_BG)
        p = QPainter(img)
        # try/finally guarantees p.end() even if any draw call raises — without
        # this Qt prints "Cannot destroy paint device that is being painted".
        try:
            p.setRenderHint(QPainter.Antialiasing, False)
            p.setRenderHint(QPainter.TextAntialiasing, True)

            # ── Header ───────────────────────────────────────────────────────
            p.setFont(QFont("Arial", 11, QFont.Bold))
            p.setPen(C_TITLE)
            hdr = (f"Frame {candidate.frame_idx:06d}   │   "
                   f"t = {candidate.timestamp_s:.4f} s   │   {candidate.file}")
            p.drawText(QRect(0, 4, W, HEADER_H - 4), Qt.AlignCenter, hdr)

            # ── Panel backgrounds + borders ───────────────────────────────────
            for panel in (FFT_panel, IFFT_panel):
                p.fillRect(panel, C_PANEL)
                p.setPen(QPen(C_BORDER, 1))
                p.drawRect(panel)

            # ── FFT Panel — three curves, ONE amplitude axis ──────────────────
            # Foreground: the REGION spectrum (onset→decay_end) — the click itself,
            #             which is what every v6 feature is computed on.
            # Dashed:     Buffer 3's noise floor, converted to the amplitude it
            #             would have when measured with THIS region's window
            #             (A_rms = sqrt(2·P·Δf)). Region vs noise IS quantitative:
            #             the gap between the two is the per-frequency excess the
            #             v6 features measure.
            # Background: the whole-frame 512-point spectrum, de-emphasised and
            #             labelled "context". It is NOT quantitatively comparable
            #             to the region curve — a finite transient's amplitude
            #             spectrum falls as N^(-1), so the same click reads ~8x
            #             lower over 512 samples than over a 60-sample region.
            #             It shows WHERE the frame's energy sits, nothing more.
            #             See _region_display_spectrum.
            freq_khz = candidate.freq_axis / 1000.0
            mask     = (freq_khz >= 20) & (freq_khz <= 80)
            fq       = freq_khz[mask]
            frame_vals = candidate.fft_norm[mask].copy() if len(candidate.fft_norm) else np.array([])
            frame_vals = np.where(np.isfinite(frame_vals), frame_vals, 0.0)

            rq = candidate.region_freqs / 1000.0 if len(candidate.region_freqs) else np.array([])
            rmask = (rq >= 20) & (rq <= 80) if len(rq) else np.array([], dtype=bool)
            rq_v   = rq[rmask] if len(rq) else np.array([])
            reg_vals = (np.where(np.isfinite(candidate.region_amp[rmask]),
                                 candidate.region_amp[rmask], 0.0)
                        if len(candidate.region_amp) == len(rq) and len(rq) else np.array([]))
            noi_vals = (candidate.noise_amp[rmask]
                        if len(candidate.noise_amp) == len(rq) and len(rq) else np.array([]))

            if len(fq) == 0:
                fq, frame_vals = np.array([20.0, 80.0]), np.array([0.0, 0.0])

            # One auto-scale for all three, so they stay comparable.
            _all = [v for v in (frame_vals, reg_vals, noi_vals) if len(v)]
            peak = max((float(np.nanmax(np.abs(v))) for v in _all
                        if np.any(np.isfinite(v))), default=0.0)
            if peak >= 0.5:
                sc, fft_unit = 1.0, 'V'
            elif peak >= 5e-4:
                sc, fft_unit = 1e3, 'mV'
            else:
                sc, fft_unit = 1e6, 'µV'

            fft_y_min = 0.0
            fft_y_max = peak * sc * 1.08
            if not np.isfinite(fft_y_max) or fft_y_max <= 0:
                fft_y_max = 1.0

            _title = "Region FFT (onset→decay end)" if len(reg_vals) else "FFT Spectrum"
            _draw_axes(p, FFT_panel,
                       20.0, 80.0, fft_y_min, fft_y_max,
                       title=_title,
                       x_label="Frequency (kHz)", y_label=f"Amplitude ({fft_unit})",
                       c_grid=C_GRID, c_axis=C_AXIS, c_title=C_TITLE)

            # Background first, so the region curve sits on top of it.
            _draw_line(p, FFT_panel, fq, frame_vals * sc,
                       20.0, 80.0, fft_y_min, fft_y_max, C_FRAME, lw=1)
            if len(noi_vals) and np.any(np.isfinite(noi_vals)):
                _draw_line(p, FFT_panel, rq_v, np.nan_to_num(noi_vals) * sc,
                           20.0, 80.0, fft_y_min, fft_y_max,
                           C_FLOOR, lw=1, dashed=True)
            if len(reg_vals):
                _draw_line(p, FFT_panel, rq_v, reg_vals * sc,
                           20.0, 80.0, fft_y_min, fft_y_max, C_FFT, lw=2)

            # Legend — without it three curves in one panel are unreadable.
            _lg = [("region", C_FFT), ("frame (context)", C_FRAME)]
            if len(noi_vals) and np.any(np.isfinite(noi_vals)):
                _lg.append(("B3 noise", C_FLOOR))
            else:
                _lg.append(("no B3 estimate", C_AXIS))
            p.setFont(QFont("Arial", 9))
            _lx = FFT_panel.right() - 150
            _ly = FFT_panel.top() + 26
            for _name, _col in _lg:
                p.setPen(QPen(_col, 2))
                p.drawLine(_lx, _ly + 5, _lx + 18, _ly + 5)
                p.setPen(C_TEXT)
                p.drawText(QRect(_lx + 24, _ly - 3, 104, 16),
                           Qt.AlignLeft | Qt.AlignVCenter, _name)
                _ly += 16

            # ── iFFT Panel ────────────────────────────────────────────────────
            # The render window is a slice of the stitched prev|curr|next context
            # centred on the peak; time is peak-relative ms (peak at 0), so the
            # picture is frame-grid independent and the whole click is visible.
            time_ms = candidate.render_t_ms
            n_samp  = len(time_ms)

            sig_raw = np.where(np.isfinite(candidate.render_signal),   candidate.render_signal,   0.0)
            env_raw = np.where(np.isfinite(candidate.render_envelope), candidate.render_envelope, 0.0)

            # Auto-scale: pick unit from peak envelope amplitude
            env_max = float(np.max(np.abs(env_raw))) if n_samp > 0 else 1e-10
            if env_max >= 0.5:
                i_sc, i_unit = 1.0, 'V'
            elif env_max >= 5e-4:
                i_sc, i_unit = 1e3, 'mV'
            else:
                i_sc, i_unit = 1e6, 'µV'

            sig_sc  = sig_raw * i_sc
            env_sc  = env_raw * i_sc
            floor_v = candidate.noise_floor * i_sc
            std_v   = (candidate.noise_floor + candidate.std_noise) * i_sc

            all_y   = np.concatenate([sig_sc, env_sc])
            i_y_min = float(np.min(all_y)) * 1.05
            i_y_max = float(np.max(all_y)) * 1.10
            if not (np.isfinite(i_y_min) and np.isfinite(i_y_max)) or i_y_max <= i_y_min:
                i_y_min, i_y_max = -1.0, 1.0
            # Always keep noise lines within view
            if np.isfinite(floor_v) and floor_v > 0:
                i_y_min = min(i_y_min, floor_v * 0.9)

            t0, t1 = (float(time_ms[0]), float(time_ms[-1])) if n_samp else (-1.28, 1.28)
            span_ms = t1 - t0

            _draw_axes(p, IFFT_panel,
                       t0, t1, i_y_min, i_y_max,
                       title=f"iFFT Signal + Envelope  (span {span_ms:.2f} ms)",
                       x_label="Time (ms, peak at 0)", y_label=f"Amplitude ({i_unit})",
                       c_grid=C_GRID, c_axis=C_AXIS, c_title=C_TITLE)
            _draw_line(p, IFFT_panel, time_ms, sig_sc,  t0, t1, i_y_min, i_y_max, C_SIG, lw=1)
            _draw_line(p, IFFT_panel, time_ms, env_sc,  t0, t1, i_y_min, i_y_max, C_ENV, lw=2)
            _draw_hline(p, IFFT_panel, floor_v, i_y_min, i_y_max, C_FLOOR, lw=1)
            _draw_hline(p, IFFT_panel, std_v,   i_y_min, i_y_max, C_STD,   lw=1)

            # Frame joins (seams) — drawn honestly since each frame carries its
            # own Tukey taper. Faint, so they don't compete with the click.
            for s_ms in candidate.seams_ms:
                _draw_vline(p, IFFT_panel, s_ms, t0, t1, C_GRID, lw=1, dashed=True)

            # Click markers: onset, peak (t=0), decay end.
            _draw_vline(p, IFFT_panel, candidate.mark_onset_ms, t0, t1,
                        C_AXIS, lw=1, dashed=True, label="onset")
            _draw_vline(p, IFFT_panel, 0.0, t0, t1,
                        C_ENV, lw=1, dashed=False, label="peak")
            _draw_vline(p, IFFT_panel, candidate.mark_decay_end_ms, t0, t1,
                        C_AXIS, lw=1, dashed=True, label="decay end")

            # Pre-computed fit curve (no scipy on main thread) — spans the real
            # decay window, no frame-edge clip.
            if candidate.fit_t_ms is not None and len(candidate.fit_t_ms) > 1:
                fit_sc = candidate.fit_y * i_sc
                _draw_line(p, IFFT_panel, candidate.fit_t_ms, fit_sc,
                           t0, t1, i_y_min, i_y_max, C_FIT, lw=2, dashed=True)

            # ── Feature footer ────────────────────────────────────────────────
            _draw_feature_footer(p, candidate, FOOTER_rect, C_TEXT, C_KEY)

        finally:
            p.end()   # always release the painter, even on exception

        output_path.parent.mkdir(parents=True, exist_ok=True)
        pixmap = QPixmap.fromImage(img)
        return pixmap.save(str(output_path), 'PNG')

    except Exception as e:
        print(f"Error rendering screenshot: {e}")
        traceback.print_exc()
        return False


# ── CSV EXPORT ──────────────────────────────────────────────────────────────

def csv_has_labels(path: Path) -> bool:
    """
    True if `path` is an existing candidates CSV with at least one non-empty label.

    Used to refuse a destructive overwrite. Any error reading the file returns
    True — if we cannot prove a file is unlabelled, we must not overwrite it.
    """
    try:
        if not path.exists():
            return False
        import csv as _csv
        with open(path, newline='') as fh:
            for row in _csv.DictReader(fh):
                if str(row.get('label', '') or '').strip():
                    return True
        return False
    except Exception:                                          # noqa: BLE001
        return True


def _write_csv_for_file(
    candidates: List[CandidateData],
    output_path: Path,
    force: bool = False,
) -> bool:
    """
    Write CSV file with all candidates for a single audio file.

    Args:
        candidates: List of CandidateData objects
        output_path: Path to write CSV
        force: overwrite even if the target already carries labels

    Returns:
        True if successful, False otherwise

    ⚠️ REFUSES TO DESTROY LABELS.
    This function opens the target in 'w' mode, and click_review_dialog._save
    writes labels back into this very file. So re-exporting over a directory that
    has already been labelled silently blanks every label — no prompt, no backup,
    no way to tell afterwards. That is a real, previously-unguarded footgun: 1944
    manual labels across 57 candidate CSVs live in files with exactly this name
    pattern.

    An export into a labelled file is therefore refused unless `force=True`.
    Point new exports at a fresh directory instead.
    """
    try:
        import csv

        if not force and csv_has_labels(output_path):
            print(f"REFUSING to overwrite labelled CSV: {output_path}\n"
                  f"  It contains manual labels that 'w' mode would destroy.\n"
                  f"  Export to a fresh directory, or pass force=True if you are "
                  f"certain.")
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            
            for candidate in candidates:
                csv_dict = candidate.to_csv_dict()
                writer.writerow(csv_dict)
        
        return True
    
    except Exception as e:
        print(f"Error writing CSV: {e}")
        traceback.print_exc()
        return False


# ── PLACEHOLDER STUBS (to be filled in next stages) ───────────────────────

class DataCollectionWorkerV5(QThread):
    """
    Non-blocking worker thread for batch data collection processing.
    
    Processes a list of .paudio files:
    1. Loads each file into AudioDataManager
    2. Runs Stage 1 to find candidates
    3. Computes all 17 features for each candidate
    4. Renders screenshots
    5. Accumulates CSV rows
    
    Emits signals to main thread for progress updates and results.
    """
    
    # Signals
    #: (file_idx, total_files, done_in_file, n_in_file, total_candidates)
    #: `done_in_file` / `n_in_file` are what let the bar advance DURING the
    #: screenshot phase. It previously carried the file's candidate count and the
    #: running total, so the handler had no within-file fraction and pinned the bar
    #: to the midpoint of the file's share for that entire phase.
    progress_updated = Signal(int, int, int, int, int)
    load_progress    = Signal(int, int, int)        # (file_idx, total_files, pct_0_100) — fired during file loading
    candidate_ready  = Signal(object)              # (CandidateData, str) — rendered on main thread
    file_complete    = Signal(str, int, str)       # (filename, candidate_count, csv_path)
    error_occurred   = Signal(str)
    finished         = Signal(int, str)            # (total_candidates, output_dir)
    
    def __init__(
        self,
        file_list: List[Path],
        k: float = K_DEFAULT,
        output_dir: Path = None,
        normalize_mode: bool = True,
        svm_model: dict = None,
        export_mode: str = EXPORT_ALL,
        threshold: float = None,
        stage2_mode: str = None,
    ):
        """
        Initialize worker thread.

        Args:
            file_list: List of .paudio file paths to process
            k: Stage 1 multiplier (default 1.5)
            output_dir: Directory to save CSVs and screenshots
            normalize_mode: Use normalized FFT if True (default True)
            svm_model: Loaded model dict (from load_svm_model). None disables
                       Stages 2-4 entirely — the export is then a plain Stage 1
                       dump with the three verdict columns left empty.
                       Loaded on the MAIN thread by the dialog so a bad .pkl is
                       reported in a message box instead of killing the worker.
            export_mode: EXPORT_ALL (every candidate, annotated),
                       EXPORT_CONFIRMED (only clicks that survived all 4 stages),
                       EXPORT_UNFILTERED (every Stage 1 candidate, fit_valid == 0
                       included), or EXPORT_STAGE3 (only candidates that cleared
                       Stage 2, i.e. were scored by the SVM, regardless of the
                       Stage 3/4 verdict). Ignored when svm_model is None.
            threshold: Override the model's own decision threshold. None = use it.
        """
        super().__init__()
        self.file_list = file_list
        self.k = k
        self.output_dir = Path(output_dir) if output_dir else Path.home() / 'plantleaf_data_collection'
        self.normalize_mode = normalize_mode
        self.svm_model = svm_model
        self.export_mode = export_mode
        # None → the module default (conservative). Never defaulted to aggressive:
        # that tier has a measured 2.1 % click cost.
        self.stage2_mode = stage2_mode
        self.threshold = threshold
        self._stop_requested = False
        # Holds the active AudioLoadWorker so request_stop() can cancel it mid-load.
        self._current_audio_worker = None
        # Funnel counts accumulated across every processed file, for the final summary.
        self.totals = {}
    
    def run(self):
        """Main worker loop — pure numpy/IO work only. No Qt widgets created here."""
        total_candidates = 0

        try:
            # Use wakepy to prevent sleep during long processing (especially on macOS)
            from core.wake_lock_manager import WakeLockManager
            waker = WakeLockManager()
            waker.acquire()


            self.output_dir.mkdir(parents=True, exist_ok=True)
            # The per-recording sub-folders are created inside the loop, once the
            # stem is known; only the two roots are made here.
            screenshots_root = self.output_dir / SCREENSHOTS_FOLDER
            screenshots_root.mkdir(parents=True, exist_ok=True)
            csvs_root = self.output_dir / CSVS_FOLDER
            csvs_root.mkdir(parents=True, exist_ok=True)

            for file_idx, paudio_file in enumerate(self.file_list):
                if self._stop_requested:
                    self.error_occurred.emit("Data collection cancelled by user")
                    return

                try:
                    self.error_occurred.emit(f"Loading {paudio_file.name}...")
                    dm = self._load_audio_file(paudio_file, file_idx, len(self.file_list))
                    if dm is None:
                        self.error_occurred.emit(f"Failed to load {paudio_file.name}, skipping")
                        continue

                    # Loading done — tell the user we're now in the analysis phase.
                    # Without this, the label stays frozen at the last "Loading: X%"
                    # value for the entire (potentially long) feature-extraction phase.
                    self.error_occurred.emit(
                        f"Analysing {paudio_file.name} ({dm.total_frames} frames)…"
                    )

                    candidates, csv_rows = _process_file_for_collection(
                        dm, k=self.k, normalize=self.normalize_mode,
                        stop_check=lambda: self._stop_requested,
                        progress_cb=lambda done, total: self.error_occurred.emit(
                            f"  → {paudio_file.name}: {done}/{total} survivors processed…"
                        ),
                        export_mode=self.export_mode,
                        stage2_mode=self.stage2_mode,
                    )

                    ## STAGES 2-4 ##
                    # Classification runs here, in the worker: SVC.predict_proba is
                    # libsvm + pure numpy (no BLAS), so it is safe off the main thread
                    # on macOS — see run_stage3_v5's docstring.
                    # Candidates are classified one file at a time, which is exactly
                    # what Stage 4 dedup requires: frame indices must never be compared
                    # across recordings.
                    if self.svm_model is not None:
                        candidates = self._classify(candidates, paudio_file.name)

                    ## CSV ##
                    # Write CSV immediately — pure file I/O, safe in background thread.
                    # Does NOT depend on screenshots, so we don't need to wait for rendering.
                    csv_dir = csvs_root / paudio_file.stem
                    csv_dir.mkdir(parents=True, exist_ok=True)
                    csv_path = csv_dir / f"{paudio_file.stem}_candidates.csv"
                    _write_csv_for_file(candidates, csv_path)
                    self.file_complete.emit(paudio_file.name, len(candidates), str(csv_path))

                    ## SCREENSHOTS ##
                    # Emit each candidate to the main thread for screenshot rendering.
                    # Qt.QueuedConnection (set in dialog) guarantees _on_candidate_ready
                    # runs on the main thread — mandatory on macOS for any Qt widget.
                    #
                    # A borderline click's Stage 4 duplicate (tagged Stage4_dedup) is
                    # the SAME physical click as its confirmed sibling — same peak,
                    # same picture — so we skip rendering it: one screenshot per
                    # click. Its CSV row is still exported (census intact); the
                    # review dialog just shows it without a screenshot.
                    from core.click_pipeline_v5 import STAGE_BLOCKED_DEDUP
                    shots_dir = screenshots_root / paudio_file.stem
                    shots_dir.mkdir(parents=True, exist_ok=True)
                    n_in_file = len(candidates)
                    for done_in_file, candidate in enumerate(candidates, start=1):
                        if self._stop_requested:
                            return

                        if candidate.stage_blocked == STAGE_BLOCKED_DEDUP:
                            continue

                        # Name by the candidate's own frame_idx so it matches this
                        # row's frame_idx in the CSV (how click_review_dialog looks
                        # the screenshot up). For a confirmed click the kept
                        # candidate owns the peak, so frame_idx == canonical anyway.
                        screenshot_filename = f"{paudio_file.stem}_{candidate.frame_idx:06d}.png"
                        screenshot_path = shots_dir / screenshot_filename

                        # Hand off to main thread — never create widgets here
                        self.candidate_ready.emit((candidate, str(screenshot_path)))

                        total_candidates += 1
                        self.progress_updated.emit(
                            file_idx, len(self.file_list),
                            done_in_file, n_in_file, total_candidates
                        )

                except Exception as e:
                    self.error_occurred.emit(f"Error processing {paudio_file.name}: {str(e)}")
                    traceback.print_exc()
                    continue

            self.finished.emit(total_candidates, str(self.output_dir))

            waker.release() # ensure wake lock is released when done

        except Exception as e:
            self.error_occurred.emit(f"Critical error in worker: {str(e)}")
            traceback.print_exc()

    def _classify(self, candidates: List[CandidateData], filename: str) -> List[CandidateData]:
        """
        Run Stages 2-4 over one file's candidates and attach the verdicts.

        Returns the list to actually export: every candidate in EXPORT_ALL mode,
        only the confirmed clicks in EXPORT_CONFIRMED mode, only the candidates
        that reached Stage 3 in EXPORT_STAGE3 mode. Filtering here (rather than
        at CSV-write time) means the discarded candidates also skip screenshot
        rendering, which is where nearly all the time goes — typically only ~5-10%
        of candidates survive, so confirmed-only exports are dramatically faster.
        """
        if not candidates:
            return candidates

        from core.click_pipeline_v5 import run_stages234_annotated, stage_summary

        annotated = run_stages234_annotated(
            [c.to_feature_dict() for c in candidates],
            self.svm_model,
            threshold=self.threshold,
            stage2_mode=self.stage2_mode,
        )

        # run_stages234_annotated preserves input order, so zip is a safe pairing.
        for cand, row in zip(candidates, annotated):
            cand.svm_probability = row['svm_probability']
            cand.svm_prediction  = row['svm_prediction']
            cand.stage_blocked   = row['stage_blocked']

        counts = stage_summary(annotated)
        for key, value in counts.items():
            self.totals[key] = self.totals.get(key, 0) + value

        self.error_occurred.emit(
            f"  → {filename}: {counts['total']} candidates → "
            f"{counts['confirmed']} confirmed click(s)"
        )

        if self.export_mode == EXPORT_CONFIRMED:
            return [c for c in candidates if c.is_confirmed_click]
        if self.export_mode == EXPORT_STAGE3:
            return [c for c in candidates if c.reached_stage3]
        return candidates

    def _load_audio_file(self, paudio_path: Path, file_idx: int = 0, total_files: int = 1):
        """
        Load a .paudio file into AudioDataManager.

        Uses AudioLoadWorker.run() synchronously (no extra QThread needed —
        we are already inside DataCollectionWorkerV5.run()).
        Mirrors file_handler_mixin._on_finished for consistent field population.

        Forwards AudioLoadWorker's internal progress (0-100 %) through the
        load_progress signal so the dialog can animate its progress bar during
        the loading phase — otherwise the UI appears frozen for large files.

        Args:
            paudio_path : Path to .paudio file
            file_idx    : Zero-based index of this file in the batch (for progress scaling)
            total_files : Total number of files in the batch (for progress scaling)

        Returns:
            AudioDataManager if successful, None otherwise
        """
        try:
            from saving.audio_load_progress import AudioLoadWorker
            from windows.replay_window_audio import AudioDataManager

            # Capture worker output via signal lambdas
            result = {'data': None, 'error': None}

            worker = AudioLoadWorker(str(paudio_path))
            worker.finished.connect(lambda d: result.update({'data': d}))
            worker.error.connect(lambda msg: result.update({'error': msg}))

            # Forward AudioLoadWorker's internal progress to the dialog.
            # worker.progress emits 0-100 within this file; load_progress
            # carries (file_idx, total_files, pct) so the dialog can scale it
            # correctly across the full batch.
            worker.progress.connect(
                lambda pct: self.load_progress.emit(file_idx, total_files, pct)
            )

            # Register so request_stop() can cancel this worker immediately.
            self._current_audio_worker = worker

            # Run synchronously — blocks until file is fully loaded.
            # Signals are emitted synchronously (same-thread DirectConnection).
            worker.run()

            # Deregister — the worker has finished (or was cancelled).
            self._current_audio_worker = None

            if self._stop_requested:
                return None   # cancelled during load — skip this file

            if result['error']:
                print(f"Load error for {paudio_path.name}: {result['error']}")
                return None
            if result['data'] is None:
                print(f"No data returned for {paudio_path.name}")
                return None

            data = result['data']

            # Populate AudioDataManager (mirrors file_handler_mixin._on_finished)
            dm = AudioDataManager()
            dm.header_info          = data['header_info']
            dm.fft_data             = data['fft_data']
            dm.phase_data           = data.get('phase_data', [])
            dm.frequency_axis       = np.array(data['frequency_axis'])
            dm.total_frames         = data['total_frames']
            dm.frame_duration_ms    = data['frame_duration_ms']
            dm.total_duration_sec   = data['total_duration_sec']
            dm.click_events         = data['click_events']
            dm.overview_x           = np.array(data['overview_x'])
            dm.overview_y           = np.array(data['overview_y'])
            dm.overview_loaded      = True
            dm.streaming_x          = np.array(data['streaming_x'])
            dm.streaming_y          = np.array(data['streaming_y'])
            dm.streaming_start_time = data['streaming_start_time']
            dm.streaming_end_time   = data['streaming_end_time']
            dm.filename             = paudio_path.stem

            # Aliases expected by click_pipeline_v5 (run_stage1_v5,
            # reconstruct_frame_v5) which use fft_mags/phase_int8 naming.
            dm.fft_mags   = dm.fft_data    # same list, pipeline-compatible alias
            dm.phase_int8 = dm.phase_data  # same list, pipeline-compatible alias

            # Use pre-computed noise arrays from worker if available;
            # otherwise compute them now (required by run_stage1_v5).
            fft_means = data.get('fft_means')
            if fft_means is not None and len(fft_means) > 0:
                dm.fft_means       = data['fft_means']
                dm.fft_timestamps  = data['fft_timestamps']
                dm.E_hat_floor_arr = data['E_hat_floor_arr']
                dm.noise_floor_arr = data['noise_floor_arr']
                dm.std_noise_arr   = data['std_noise_arr']
                # v6 Buffer 3. Without these three, p_noise_at() returns None for
                # every frame and all eight v6 features export as NaN — which is
                # exactly what happened on the first real export. .get() keeps a
                # dict from an older worker loadable.
                dm.p_noise_snapshots = data.get('p_noise_snapshots')
                dm.p_noise_stride    = data.get('p_noise_stride')
                dm.p_noise_counts    = data.get('p_noise_counts')
            else:
                # Fallback for files the worker did not pre-compute. It builds no
                # Buffer 3, so the v6 features are genuinely unavailable here and
                # must stay NaN — b3_frames = 0 records that honestly rather than
                # letting a stale estimate leak in.
                dm.precompute_fft_means()
                dm.p_noise_snapshots = None
                dm.p_noise_stride    = None
                dm.p_noise_counts    = None

            return dm

        except Exception as e:
            print(f"Failed to load audio file: {e}")
            traceback.print_exc()
            return None
        

    def request_stop(self):
        """
        Request the worker to stop gracefully.

        Also cancels any AudioLoadWorker that is currently running synchronously
        inside _load_audio_file, so the cancel takes effect immediately instead
        of waiting for the full file-load loop to complete.
        """
        self._stop_requested = True
        if self._current_audio_worker is not None:
            self._current_audio_worker.cancel_load()  # sets AudioLoadWorker._cancelled


class DataCollectionDialogV5(QDialog):
    """
    Dialog for batch data collection from multiple .paudio files.

    Provides UI for:
    - Selecting .paudio files to process
    - Configuring Stage 1 multiplier (k)
    - Running Stages 2-4 (SVM classification) and choosing what to export
    - Selecting output folder
    - Running batch export (with progress feedback)
    - Viewing results
    """

    # The app's QSS themes target QMainWindow, not QDialog, so under a light theme a
    # dialog inherits the *dark* palette's pale text on a white background — i.e. the
    # group boxes, check boxes and radios become unreadable. Every widget class used
    # here has to be given an explicit colour. (Same approach as RegionFFTDialog.)
    _LIGHT_CSS = """
        QDialog, QWidget { background-color: white; color: black; }
        QGroupBox { color: black; border: 1px solid #ccc; border-radius: 4px;
                    margin-top: 8px; padding-top: 6px; font-weight: bold; }
        QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px;
                           color: black; }
        QLabel { color: black; background: transparent; }
        QCheckBox, QRadioButton { color: black; background: transparent; }
        QPushButton { background-color: #f0f0f0; color: black;
                      border: 1px solid #ccc; padding: 4px 8px; border-radius: 3px; }
        QPushButton:hover { background-color: #e0e0e0; }
        QPushButton:disabled { color: #999; }
        QDoubleSpinBox, QSpinBox, QComboBox {
            background-color: white; color: black;
            border: 1px solid #ccc; padding: 2px 4px; border-radius: 3px; }
        QComboBox QAbstractItemView { background-color: white; color: black;
                                      selection-background-color: #d0e4ff;
                                      selection-color: black; }
        QListWidget, QPlainTextEdit { background-color: white; color: black;
                                      border: 1px solid #ccc; }
        QProgressBar { border: 1px solid #bbb; border-radius: 5px;
                       background-color: #eee; color: black; }
        QProgressBar::chunk { background-color: #1f77b4; }
    """


    def __init__(self, parent=None):
        """Initialize data collection dialog."""
        super().__init__(parent)
        self.setWindowTitle("Data Collection — Stage 1 Batch Export (v5)")
        self.resize(820, 600)

        self.settings_manager = SettingsManager()
        self.file_list = []
        self.output_dir = Path(self.settings_manager.get_last_directory("data_collection_output"))
        self.worker = None
        self.is_processing = False
        self.last_csv_path = None   # most recent exported CSV — opened by 'Review & Label'

        self._setup_ui()
        self._setup_connections()

        # Put dialog in the middle of the screen
        screen_geometry = self.screen().geometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 4
        self.move(x, y)

        # Correct theme for light mode
        if self.parent() and hasattr(self.parent(), 'theme_manager'):
            try:
                theme = self.parent().theme_manager.load_saved_theme()
                self.parent().theme_manager.apply_theme(self, theme)
                if 'light' in theme.lower():  # known bug fix for light themes
                    self.setStyleSheet(self._LIGHT_CSS)
            except Exception:
                pass

        # Fai in modo che con ESC si chiuda la finestra
        self.setWindowFlag(Qt.WindowCloseButtonHint, True)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, False)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, False)
        self.setModal(True)

    def _setup_ui(self):
        """Build the dialog UI layout."""
        layout = QVBoxLayout(self)
        # Six stacked group boxes make this dialog tall; trim the default padding so
        # it fits on a laptop screen without scrolling.
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ── File List Section ──
        layout.addWidget(self._create_file_list_section())
        
        # ── Parameters Section ──
        layout.addWidget(self._create_parameters_section())

        # ── Classification Section (Stages 2-4) ──
        layout.addWidget(self._create_classification_section())

        # ── Output Folder Section ──
        layout.addWidget(self._create_output_section())
        
        # ── Progress Section ──
        layout.addWidget(self._create_progress_section())
        
        # ── Button Section ──
        layout.addWidget(self._create_button_section())
        
        # ── Log Area ──
        layout.addWidget(self._create_log_section())
        
        self.setLayout(layout)
    
    def _create_file_list_section(self):
        """Create file selection section (add/remove/clear buttons + list)."""
        from PySide6.QtWidgets import QGroupBox
        
        group = QGroupBox("Audio Files (.paudio)")
        layout = QVBoxLayout()
        
        # File list widget
        self.file_list_widget = QListWidget()
        self.file_list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.file_list_widget.setMaximumHeight(90)
        layout.addWidget(self.file_list_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.btn_add = QPushButton("Add Files")
        self.btn_add.clicked.connect(self._on_add_files)
        button_layout.addWidget(self.btn_add)
        
        self.btn_remove = QPushButton("Remove Selected")
        self.btn_remove.clicked.connect(self._on_remove_file)
        button_layout.addWidget(self.btn_remove)
        
        self.btn_clear = QPushButton("Clear All")
        self.btn_clear.clicked.connect(self._on_clear_files)
        button_layout.addWidget(self.btn_clear)
        
        layout.addLayout(button_layout)
        group.setLayout(layout)
        return group
    
    def _create_parameters_section(self):
        """Create parameters section (k multiplier, normalize checkbox)."""
        from PySide6.QtWidgets import QGroupBox
        
        group = QGroupBox("Parameters")
        layout = QHBoxLayout()
        
        # K multiplier
        layout.addWidget(QLabel(" k (Stage 1 multiplier):"))
        self.spinbox_k = QDoubleSpinBox()
        self.spinbox_k.setMinimum(K_MIN)
        self.spinbox_k.setMaximum(K_MAX)
        self.spinbox_k.setSingleStep(K_STEP)
        self.spinbox_k.setValue(K_DEFAULT)
        self.spinbox_k.setDecimals(1)
        self.spinbox_k.setToolTip("Higher k = stricter threshold (fewer candidates)")
        layout.addWidget(self.spinbox_k)
        
        layout.addSpacing(30)
        
        # Note: data is always normalized (replace checkbox)
        note_lbl = QLabel("Note: data is always normalized (not configurable)")
        note_lbl.setStyleSheet("QLabel { color: gray; font-style: italic; }")
        layout.addWidget(note_lbl)

        layout.addStretch()
        group.setLayout(layout)

        return group
    
    def _create_classification_section(self):
        """
        Create the Stages 2-4 section: model choice, export mode, threshold.

        Everything here is optional. With 'Run Stages 2-4' unticked the dialog
        behaves exactly as it always has — a plain Stage 1 dump — and no model is
        touched, so the training-data collection workflow is never blocked by a
        missing or broken .pkl.
        """
        from PySide6.QtWidgets import QGroupBox

        group = QGroupBox("Classification (Stages 2-4)")
        layout = QVBoxLayout()

        # ── Master switch ──
        self.chk_classify = QCheckBox("Run Stages 2-4 (SVM classification)")
        self.chk_classify.setChecked(True)
        self.chk_classify.setToolTip(
            "Run the hard gates, the SVM and deduplication on every Stage 1 candidate.\n"
            "Unticked: export raw Stage 1 candidates only (no model needed)."
        )
        self.chk_classify.toggled.connect(self._on_classify_toggled)
        layout.addWidget(self.chk_classify)

        # ── Export mode ──
        mode_layout = QHBoxLayout()
        mode_layout.addSpacing(20)
        mode_layout.addWidget(QLabel("Export:"))

        self.radio_all = QRadioButton("All candidates + stats")
        self.radio_all.setChecked(True)
        self.radio_all.setToolTip(
            "Every Stage 1 candidate, each tagged with the stage that rejected it\n"
            "and its SVM probability. Best for analysis and for labelling."
        )

        self.radio_confirmed = QRadioButton("Confirmed clicks only")
        self.radio_confirmed.setToolTip(
            "Only candidates that survived all four stages.\n"
            "Much faster: rejected candidates skip screenshot rendering."
        )

        self.radio_unfiltered = QRadioButton("All + unfittable")
        self.radio_unfiltered.setToolTip(
            "Every Stage 1 candidate, including those whose decay window could not\n"
            "be fitted at all (fit_valid = 0). Those rows have NEVER reached a CSV\n"
            "before — Stage 2 dropped them ahead of export — and they arrive with\n"
            "tau_ms / R2 / fit_coverage as NaN.\n\n"
            "Expect many, and expect most to be noise: Stage 1 candidates are\n"
            "overwhelmingly noise and this mode applies no fit filter at all."
        )

        self.radio_stage3 = QRadioButton("Reached Stage 3")
        self.radio_stage3.setToolTip(
            "Only candidates that cleared Stage 2 — i.e. were scored by the SVM —\n"
            "whatever Stage 3/4 then did with them. Includes SVM-rejected and\n"
            "deduplicated candidates, not just confirmed clicks.\n\n"
            "For reviewing the SVM's own decisions (threshold tuning, error\n"
            "analysis) without the Stage 2 noise majority in the way."
        )

        self.export_mode_group = QButtonGroup(self)
        self.export_mode_group.addButton(self.radio_all)
        self.export_mode_group.addButton(self.radio_confirmed)
        self.export_mode_group.addButton(self.radio_unfiltered)
        self.export_mode_group.addButton(self.radio_stage3)

        mode_layout.addWidget(self.radio_all)
        mode_layout.addWidget(self.radio_confirmed)
        mode_layout.addWidget(self.radio_unfiltered)
        mode_layout.addWidget(self.radio_stage3)

        # ── Stage 2 rule ──
        # Three RULES, not three strictness levels: v5 is the original gate, kept
        # so a v5 result stays reproducible. Ordered by click cost so the default
        # is in the middle and the expensive one cannot be picked by accident.
        mode_layout.addSpacing(16)
        mode_layout.addWidget(QLabel("Stage 2:"))
        self.combo_stage2 = QComboBox()
        #: (label, STAGE2_MODE_* value) — index order is the ONLY thing
        #: _selected_stage2_mode() depends on, so append, never reorder.
        self._stage2_choices = [
            ("v6 conservative (default)", STAGE2_MODE_CONSERVATIVE),
            ("v6 aggressive",             STAGE2_MODE_AGGRESSIVE),
            ("v5 fit gate (legacy)",      STAGE2_MODE_V5),
        ]
        self.combo_stage2.addItems([n for n, _ in self._stage2_choices])
        self.combo_stage2.setToolTip(
            "Which Stage 2 rule to apply. Measured on 32 exhaustively-labelled\n"
            "recordings (189 clicks / 99 ambiguous / 5786 noise):\n\n"
            "  v6 conservative :  0.0 % clicks lost,  83.6 % of noise removed\n"
            "  v6 aggressive   :  2.1 % clicks lost,  87.0 % of noise removed\n"
            "  v5 fit gate     : 12.2 % clicks lost,  91.4 % of noise removed\n\n"
            "v6 aggressive raises the peak_SNR floor from 4.5 to 5.0. It is for\n"
            "sessions whose candidate rate makes review impossible — outdoor\n"
            "recordings have been measured at 344,000 candidates/hour — and its\n"
            "recall cost is real, not a rounding error.\n\n"
            "v5 rejects any candidate whose decay fit failed. That is why it loses\n"
            "12 % of clicks: 8-9 % of confirmed clicks have fit_valid = 0. Kept so a\n"
            "v5 result can be reproduced exactly, not because it is recommended.\n\n"
            "The rule used is written into each row's stage2_mode column, so an\n"
            "export can always be attributed to what produced it."
        )
        mode_layout.addWidget(self.combo_stage2)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        # ── Model picker ──
        model_layout = QHBoxLayout()
        model_layout.addSpacing(20)
        model_layout.addWidget(QLabel("Model:"))

        self.label_model = QLabel()
        self.label_model.setStyleSheet("QLabel { padding: 4px; }")
        model_layout.addWidget(self.label_model, stretch=1)

        self.btn_browse_model = QPushButton("Browse...")
        self.btn_browse_model.clicked.connect(self._on_browse_model)
        model_layout.addWidget(self.btn_browse_model)

        self.btn_reset_model = QPushButton("Reset")
        self.btn_reset_model.setToolTip("Go back to the model shipped with the app")
        self.btn_reset_model.clicked.connect(self._on_reset_model)
        model_layout.addWidget(self.btn_reset_model)

        layout.addLayout(model_layout)

        # ── Model info + threshold override ──
        thr_layout = QHBoxLayout()
        thr_layout.addSpacing(20)

        self.label_model_info = QLabel()
        self.label_model_info.setStyleSheet("QLabel { color: gray; font-style: italic; }")
        thr_layout.addWidget(self.label_model_info, stretch=1)

        self.chk_use_model_threshold = QCheckBox("Use model threshold")
        self.chk_use_model_threshold.setChecked(True)
        self.chk_use_model_threshold.setToolTip(
            "The threshold chosen during training to hit the target recall.\n"
            "Untick to override it: lower = more clicks detected but more false\n"
            "positives; higher = stricter."
        )
        self.chk_use_model_threshold.toggled.connect(
            lambda on: self.spin_threshold.setEnabled(not on)
        )
        thr_layout.addWidget(self.chk_use_model_threshold)

        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.0, 1.0)
        self.spin_threshold.setSingleStep(0.01)
        self.spin_threshold.setDecimals(3)
        self.spin_threshold.setEnabled(False)
        thr_layout.addWidget(self.spin_threshold)

        layout.addLayout(thr_layout)

        group.setLayout(layout)

        # Load the default model so the info line and threshold are populated up front.
        self._set_model_path(self._default_model_path(), announce=False)

        return group

    def _default_model_path(self) -> Optional[Path]:
        """The model shipped with the app, or None if ml/ cannot be imported."""
        try:
            from ml import default_model_path
            return default_model_path()
        except Exception:
            return None

    def _set_model_path(self, path: Optional[Path], announce: bool = True):
        """
        Point the dialog at a model file and read its metadata.

        The model is loaded here on the main thread — not in the worker — so a
        missing or corrupt .pkl surfaces immediately, next to the control that
        chose it, instead of aborting an export ten minutes in.
        """
        self.model_path = path
        self.svm_model = None

        if path is None:
            self.label_model.setText("(no model found)")
            self.label_model_info.setText("")
            return

        self.label_model.setText(path.name)
        self.label_model.setToolTip(str(path))

        try:
            from core.click_pipeline_v5 import load_svm_model
            self.svm_model = load_svm_model(path)
        except Exception as e:
            self.label_model_info.setText(f"could not load: {e}")
            if announce:
                QMessageBox.warning(
                    self, "Model not loaded",
                    f"Could not load the model:\n\n{path}\n\n{e}",
                )
            return

        threshold = float(self.svm_model['threshold'])
        n_feat    = len(self.svm_model['features'])
        kernel    = self.svm_model.get('kernel', '?')

        # AUC is informational, written by train_svm.py — absent on older models.
        auc = None
        all_results = self.svm_model.get('all_results') or {}
        if isinstance(all_results, dict) and kernel in all_results:
            auc = (all_results[kernel] or {}).get('auc')

        info = f"kernel={kernel}  threshold={threshold:.3f}  features={n_feat}"
        if auc is not None:
            info += f"  AUC={auc:.3f}"
        self.label_model_info.setText(info)

        # Pre-fill the override with the model's own value, so unticking the box
        # starts from the trained threshold rather than from zero.
        self.spin_threshold.setValue(threshold)

        if announce:
            self._log(f"Model loaded: {path.name}  ({info})")

    def _on_browse_model(self):
        """Pick a different .pkl (other trained variants live in docs/autoclick/v5/pkl/)."""
        start_dir = str(self.model_path.parent) if self.model_path else self.settings_manager.get_last_directory("svm_model")
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select SVM Model", start_dir, "SVM model (*.pkl)"
        )
        if filepath:
            self._set_model_path(Path(filepath))
            self.settings_manager.set_last_directory("svm_model", filepath)

    def _on_reset_model(self):
        """Go back to the model shipped with the app."""
        self._set_model_path(self._default_model_path())

    def _on_classify_toggled(self, enabled: bool):
        """Grey out the whole classification sub-section when Stages 2-4 are off."""
        for w in (
            self.radio_all, self.radio_confirmed,
            self.label_model, self.btn_browse_model, self.btn_reset_model,
            self.label_model_info, self.chk_use_model_threshold,
        ):
            w.setEnabled(enabled)
        self.spin_threshold.setEnabled(
            enabled and not self.chk_use_model_threshold.isChecked()
        )

    def _create_output_section(self):
        """Create output folder selection section."""
        from PySide6.QtWidgets import QGroupBox

        group = QGroupBox("Output Folder")
        layout = QHBoxLayout()
        
        layout.addWidget(QLabel(" Folder:"))
        self.label_output = QLabel(str(self.output_dir))
        self.label_output.setStyleSheet("QLabel { padding: 5px; }")
        layout.addWidget(self.label_output)
        
        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self._on_browse_output)
        layout.addWidget(self.btn_browse)
        
        group.setLayout(layout)
        return group
    
    def _create_progress_section(self):
        """Create progress bar section."""
        from PySide6.QtWidgets import QGroupBox
        
        group = QGroupBox("Progress")
        layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        self.label_progress = QLabel(" Ready")
        layout.addWidget(self.label_progress)
        
        group.setLayout(layout)
        return group
    
    def _create_log_section(self):
        """Create log output section."""
        from PySide6.QtWidgets import QGroupBox
        
        group = QGroupBox("Log")
        layout = QVBoxLayout()
        
        self.log_area = QPlainTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(120)
        layout.addWidget(self.log_area)
        
        group.setLayout(layout)
        return group
    
    def _create_button_section(self):
        """Create action buttons (Export, Cancel, Close)."""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.addStretch()
        
        # Enabled once an export finishes — the natural next step is to label it.
        self.btn_review = QPushButton("Review && Label...")
        self.btn_review.setMinimumWidth(120)
        self.btn_review.setEnabled(False)
        self.btn_review.setToolTip("Open the exported candidates in the labelling dialog")
        self.btn_review.clicked.connect(self._on_review)
        layout.addWidget(self.btn_review)

        self.btn_export = QPushButton("Export")
        self.btn_export.setMinimumWidth(100)
        self.btn_export.clicked.connect(self._on_export)
        layout.addWidget(self.btn_export)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setMinimumWidth(100)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel)
        layout.addWidget(self.btn_cancel)
        
        self.btn_close = QPushButton("Close")
        self.btn_close.setMinimumWidth(100)
        self.btn_close.clicked.connect(self.close)
        layout.addWidget(self.btn_close)
        
        container.setLayout(layout)
        return container
    
    def _setup_connections(self):
        """Connect signals/slots (most connected in _create_* methods)."""
        pass
    
    # ── Event Handlers ──
    
    def _on_add_files(self):
        """Browse and add .paudio files."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select .paudio files",
            self.settings_manager.get_last_directory("add_paudio_files"),
            "Audio Files (*.paudio);;All Files (*)",
        )

        for filepath in files:
            path = Path(filepath)
            if path not in [Path(self.file_list_widget.item(i).text()) for i in range(self.file_list_widget.count())]:
                self.file_list_widget.addItem(str(path))
                self.file_list.append(path)

        if files:
            self.settings_manager.set_last_directory("add_paudio_files", files[0])

        self._update_export_button()
    
    def _on_remove_file(self):
        """Remove selected file from list."""
        row = self.file_list_widget.currentRow()
        if row >= 0:
            item = self.file_list_widget.takeItem(row)
            self.file_list = [Path(self.file_list_widget.item(i).text()) for i in range(self.file_list_widget.count())]
        
        self._update_export_button()
    
    def _on_clear_files(self):
        """Clear all files from list."""
        self.file_list_widget.clear()
        self.file_list = []
        self._update_export_button()
    
    def _on_browse_output(self):
        """Browse for output directory."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
            str(self.output_dir),
        )
        
        if folder:
            self.output_dir = Path(folder)
            self.label_output.setText(str(self.output_dir))
            self.settings_manager.set_last_directory("data_collection_output", folder)
    
    def _on_export(self):
        """Start data collection worker."""
        if not self.file_list:
            self._log("No files selected!")
            return
        
        # Update file list from widget
        self.file_list = [Path(self.file_list_widget.item(i).text()) for i in range(self.file_list_widget.count())]

        # ── Resolve classification settings ──
        # The model is validated BEFORE the worker starts: an unusable .pkl must
        # stop the export here, not after minutes of feature extraction.
        classify = self.chk_classify.isChecked()
        svm_model = None
        threshold = None
        export_mode = EXPORT_ALL

        if classify:
            if self.svm_model is None:
                QMessageBox.warning(
                    self, "No model",
                    "Stages 2-4 are enabled but no usable SVM model is loaded.\n\n"
                    "Choose a valid .pkl, or untick 'Run Stages 2-4' to export "
                    "raw Stage 1 candidates.",
                )
                return
            svm_model = self.svm_model
            if self.radio_confirmed.isChecked():
                export_mode = EXPORT_CONFIRMED
            elif self.radio_unfiltered.isChecked():
                export_mode = EXPORT_UNFILTERED
            elif self.radio_stage3.isChecked():
                export_mode = EXPORT_STAGE3
            else:
                export_mode = EXPORT_ALL
            if not self.chk_use_model_threshold.isChecked():
                threshold = self.spin_threshold.value()

        # Disable/enable buttons
        self.btn_export.setEnabled(False)
        self.btn_add.setEnabled(False)
        self.btn_remove.setEnabled(False)
        self.btn_clear.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.is_processing = True
        
        # Clear log
        self.log_area.clear()
        self._log(f"Starting data collection for {len(self.file_list)} file(s)...")
        self._log(f"  k = {self.spinbox_k.value()}")
        self._log(f"  Normalize: True (fixed)")
        self._log(f"  Stage 2 rule: {self._selected_stage2_mode()}")
        if classify:
            eff_thr = threshold if threshold is not None else float(svm_model['threshold'])
            mode_txt = ("every candidate incl. unfittable" if export_mode == EXPORT_UNFILTERED
                        else "confirmed clicks only" if export_mode == EXPORT_CONFIRMED
                        else "reached Stage 3 only" if export_mode == EXPORT_STAGE3
                        else "all candidates + stats")
            self._log(f"  Stages 2-4: {self.model_path.name}  (threshold = {eff_thr:.3f})")
            self._log(f"  Export: {mode_txt}")
        else:
            self._log(f"  Stages 2-4: off (Stage 1 candidates only)")
        self._log(f"  Output: {self.output_dir}")
        self._log("")

        # Create and start worker
        self.worker = DataCollectionWorkerV5(
            file_list=self.file_list,
            k=self.spinbox_k.value(),
            output_dir=self.output_dir,
            normalize_mode=True,
            svm_model=svm_model,
            export_mode=export_mode,
            threshold=threshold,
            stage2_mode=self._selected_stage2_mode(),
        )
        
        # QueuedConnection → slot always executes on the main thread.
        # This is mandatory on macOS: Qt widgets / NSWindow must be created
        # on the main thread. The worker emits the signal from its thread;
        # Qt queues it and delivers it here on the GUI thread.
        self.worker.candidate_ready.connect(
            self._on_candidate_ready, Qt.QueuedConnection
        )
        self.worker.progress_updated.connect(
            self._on_progress_updated, Qt.QueuedConnection
        )
        self.worker.load_progress.connect(
            self._on_load_progress, Qt.QueuedConnection
        )
        self.worker.file_complete.connect(
            self._on_file_complete, Qt.QueuedConnection
        )
        self.worker.error_occurred.connect(
            self._on_error, Qt.QueuedConnection
        )
        self.worker.finished.connect(
            self._on_worker_finished, Qt.QueuedConnection
        )

        self.worker.start()
    
    def _on_candidate_ready(self, args):
        """
        Render screenshot on the main thread.

        Called via Qt.QueuedConnection from DataCollectionWorkerV5 — this
        method ALWAYS runs on the main (GUI) thread, satisfying the macOS
        requirement that NSWindow/QWidget are only instantiated on the main thread.
        """
        candidate, screenshot_path = args
        _render_candidate_screenshot(candidate, Path(screenshot_path))

    def _on_cancel(self):
        """
        Cancel worker thread without blocking the main thread.

        NEVER call self.worker.wait() here — it would block the main thread
        while DataCollectionWorkerV5 is stuck inside AudioLoadWorker.run(),
        which cannot respect _stop_requested until the current file finishes
        loading.  Instead we:
          1. Signal both workers to stop (request_stop also cancels the inner
             AudioLoadWorker via _current_audio_worker.cancel_load())
          2. Disconnect all signals so no stale callbacks reach the UI
          3. Reset the UI immediately
        The background QThread finishes on its own; Python GC cleans it up.
        """
        if self.worker:
            self._log("Cancelling...")
            self.worker.request_stop()
            # Disconnect all signals so in-flight queued events are harmless.
            try:
                self.worker.candidate_ready.disconnect()
                self.worker.progress_updated.disconnect()
                self.worker.load_progress.disconnect()
                self.worker.file_complete.disconnect()
                self.worker.error_occurred.disconnect()
                self.worker.finished.disconnect()
            except RuntimeError:
                pass
            self.worker = None   # allow GC; thread finishes in background
            self._log("Cancelled.")
        self._reset_ui()
    
    def _on_progress_updated(self, file_idx: int, total_files: int,
                             done_in_file: int, n_in_file: int,
                             total_candidates: int):
        """
        Update progress bar and label during the screenshot phase.

        Each file owns an equal share of 0-100 %: loading fills the first half of
        that share, screenshot rendering the second. This used to set the share's
        MIDPOINT on every emit regardless of how many candidates had been rendered,
        so on a single file the bar sat at 50 % for the whole phase and then jumped
        to 100 — which is what made the number read as half the real progress.
        """
        if total_files <= 0:
            return
        file_share = 100.0 / total_files
        frac = (done_in_file / n_in_file) if n_in_file > 0 else 1.0
        pct = int(file_idx * file_share + file_share * (0.5 + 0.5 * frac))
        self.progress_bar.setValue(min(100, pct))
        self.label_progress.setText(
            f"Rendering file {file_idx + 1}/{total_files}: "
            f"{done_in_file}/{n_in_file}  |  Candidates so far: {total_candidates}"
        )

    def _on_load_progress(self, file_idx: int, total_files: int, pct_in_file: int):
        """
        Update progress bar during file loading phase.

        AudioLoadWorker reports 0-100 % internally as it reads and processes
        the .paudio file. We scale that into each file's share of the total
        progress bar (first half of that share), so the bar moves
        smoothly during what would otherwise be a frozen UI.

        The bar therefore reads HALF of pct_in_file while a single file loads, and
        that is correct rather than the bug it resembles: loading is half the work
        and _on_progress_updated fills the rest. The label names the phase so the
        two numbers cannot be mistaken for one another.
        """
        if total_files > 0:
            file_share = 100.0 / total_files
            # Map pct_in_file (0-100) to the first half of this file's share
            pct = int(file_idx * file_share + (pct_in_file / 100.0) * file_share * 0.5)
        else:
            pct = pct_in_file // 2
       # print(f"Load progress for file {file_idx + 1}/{total_files}: {pct_in_file}% → overall {pct}%")
        self.progress_bar.setValue(pct)
        self.label_progress.setText(
            f"Loading file {file_idx + 1}/{total_files}: {pct_in_file}% loaded "
        )

    def _on_screenshot_saved(self, filepath: str):
        """Log screenshot saved (optional, keep concise)."""
        pass  # Too verbose to log every screenshot
    
    def _on_file_complete(self, filename: str, candidate_count: int, csv_path: str):
        """Log file completion."""
        self._log(f"✓ {filename}: {candidate_count} candidates → {Path(csv_path).name}")
        # Remember the most recent CSV so 'Review & Label' can open it directly.
        self.last_csv_path = Path(csv_path)

    def _on_review(self):
        """Open the labelling dialog, on the last exported CSV if there is one."""
        try:
            from components.click_review_dialog import ClickReviewDialog
            theme_manager = getattr(self.parent(), 'theme_manager', None)
            dialog = ClickReviewDialog(
                csv_path=getattr(self, 'last_csv_path', None),
                parent=self,
                theme_manager=theme_manager,
            )
            dialog.exec()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open click review dialog: {e}")
    
    def _on_error(self, error_msg: str):
        """Log error message."""
        self._log(f"⚠ {error_msg}")
    
    def _on_worker_finished(self, total_candidates: int, output_dir: str):
        """Worker finished — reset UI and show results."""
        self._log("")
        self._log(f"✅ Collection complete: {total_candidates} total candidates exported")

        self._log_funnel()

        self._log(f"📁 Output: {output_dir}")
        self.progress_bar.setValue(100)

        if self.last_csv_path is not None:
            self.btn_review.setEnabled(True)
            self._log("")
            self._log("→ 'Review & Label...' opens these candidates for labelling.")

        self._reset_ui()

    def _log_funnel(self):
        """
        Log the per-stage funnel accumulated over every processed file.

        Note this counts what the pipeline *saw*, which in 'confirmed clicks only'
        mode is more than what was written to disk — that is the point: it shows
        what was discarded and why.
        """
        totals = getattr(self.worker, 'totals', None)
        if not totals or not totals.get('total'):
            return

        from core.click_pipeline_v5 import (
            STAGE_BLOCKED_R2, STAGE_BLOCKED_SPR,
            STAGE_BLOCKED_SVM, STAGE_BLOCKED_DEDUP,
        )

        total     = totals['total']
        confirmed = totals['confirmed']
        pct       = (100.0 * confirmed / total) if total else 0.0

        self._log("")
        self._log(f"  Stage 1 candidates : {total}")
        self._log(f"    Stage2_R2        : {totals.get(STAGE_BLOCKED_R2, 0)}")
        self._log(f"    Stage2_SPR       : {totals.get(STAGE_BLOCKED_SPR, 0)}")
        self._log(f"    Stage3_SVM       : {totals.get(STAGE_BLOCKED_SVM, 0)}")
        self._log(f"    Stage4_dedup     : {totals.get(STAGE_BLOCKED_DEDUP, 0)}")
        self._log(f"  Confirmed clicks   : {confirmed}  ({pct:.1f}%)")
    
    def _reset_ui(self):
        """Reset UI after export completes or cancels."""
        self.btn_export.setEnabled(True)
        self.btn_add.setEnabled(True)
        self.btn_remove.setEnabled(True)
        self.btn_clear.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.is_processing = False
        self._update_export_button()
    
    def _update_export_button(self):
        """Enable/disable Export button based on file list."""
        self.btn_export.setEnabled(self.file_list_widget.count() > 0 and not self.is_processing)
    
    def _selected_stage2_mode(self) -> str:
        """
        The STAGE2_MODE_* value the user picked.

        Reads self._stage2_choices by INDEX, so the combo's items may be appended
        to but never reordered — same append-only rule the review dialog's filter
        combo follows, and for the same reason: the index is the contract.
        """
        i = self.combo_stage2.currentIndex()
        if 0 <= i < len(self._stage2_choices):
            return self._stage2_choices[i][1]
        return STAGE2_MODE_CONSERVATIVE

    def _log(self, message: str):
        """Append message to log area."""
        self.log_area.appendPlainText(message)
        # Auto-scroll to bottom
        self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())
