"""
Click Review Dialog — label candidates and compare them against the algorithm.

Replaces the Excel round-trip described in docs/autoclick/v5/SVM_TRAINING_DATA_GUIDE.md:
open a *_candidates.csv exported by DataCollectionDialogV5, step through the
candidates, and mark each one click (1) or noise (0) with a single keystroke. The
`label` column is written straight back to the same CSV, ready for train_svm.py.

The screenshots the export already rendered are reused as-is — the PNG for a row is
`screenshots/{file}_{frame_idx:06d}.png` next to the CSV — so nothing is recomputed
and opening a file is instant.

When the CSV carries the Stage 2/3/4 verdict columns (svm_probability, svm_prediction,
stage_blocked), the dialog also shows a live confusion matrix of your labels against
the pipeline's decisions, which is the fastest way to see where the model is wrong.
"""

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QFileDialog, QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView,
    QSplitter, QScrollArea, QWidget, QMessageBox, QGroupBox, QSizePolicy,
    QLineEdit, QFrame,
)
from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import QPixmap, QColor, QFont

from components.wide_combo_box import WideComboBox
from core.settings_manager import SettingsManager


SCREENSHOTS_FOLDER = 'screenshots'

# Table columns. The frame index is deliberately not shown: it costs width and tells
# the user nothing they can act on.
_COL_TIME, _COL_PROB, _COL_VERDICT, _COL_LABEL = range(4)
_N_COLS = 4

# The exported PNG is 1400x800: a title strip on top, the FFT/iFFT plots in the middle,
# a feature footer at the bottom. Scaled down to fit a pane, that footer text is far too
# small to read — so it is cropped away and the same numbers are shown in a real table
# underneath instead (see _build_features_group).
# IMPORTED from the renderer, never duplicated. These were hand-tuned literals (40 /
# 260) with only a comment asking future editors to keep them in step — so raising
# the footer to fit the v6 features would have silently mis-cropped every screenshot,
# with no error and no visible cause. Importing makes that impossible.
from components.data_collection_dialog_v5 import (      # noqa: E402
    SCREENSHOT_HEADER_H as _CROP_TOP,
    SCREENSHOT_FOOTER_H as _CROP_BOTTOM,
)
# Same reasoning: the "region too short" banner is a threshold that lives in one
# place, and a copy here would drift the day V6_MIN_NSEG moves.
from core.spectral_analysis import V6_MIN_NSEG as _V6_MIN_NSEG   # noqa: E402

# Every feature the CSV may carry, each with a display precision. A name that is
# absent from the CSV renders as "—" (see _update_features), so this list is safe
# against both v5 (24-column) and v6 (51-column) files.
_FEATURE_FMT = [
    # ── v5, in the order the docs list them ──
    ('peak_SNR',     3), ('pre_SNR',    3), ('post_SNR',           3),
    ('rise_time_ms', 4), ('fall_time_ms', 4), ('asymmetry_integral', 4),
    ('ZCR_pre',      3), ('ZCR_click',  3), ('ZCR_post',           3),
    ('kurtosis',     2), ('centroid_shift_hz', 0),
    ('tau_ms',       4), ('R2',         4), ('fit_coverage',       3),
    ('SPR',          2), ('R_spectral', 3), ('FPE_hz',             0),
    # ── v6 spectral family, computed on E[k] = max(0, P_region − P_noise) ──
    ('spectral_entropy',       3), ('shape_novelty', 3),
    ('spectral_tilt',          3), ('temporal_concentration', 3),
    ('FPE_hz_region',          0), ('SPR_region',    2),
    ('f_50_hz',                0), ('IQR_f',         0),
    # ── Stage 1 v5.1 ──
    # local_crest is a feature; the rest are diagnostics, shown because they are
    # what a reviewer needs to judge a row that v5 would have deleted outright.
    ('local_crest',            3), ('k_ratio',       2),
    # harmonic_confinement: 0 = excess spread uniformly, >0 = confined to BOTH the
    # fundamental and its second harmonic (an artificial ranging sensor / alarm),
    # <0 = one of the two bands is empty. hc_f1_hz says which fundamental.
    ('harmonic_confinement',   2), ('hc_f1_hz',      0),
    ('hc_r_A',                 2), ('hc_r_B',        2),
    ('run_length',             0), ('run_crest',     3),
    ('pos_in_run',             0), ('would_pass_v5', 0),
]

# ── Validity flags — shown SEPARATELY and first ──────────────────────────────
# These are not features, they are the columns that say whether the features above
# mean anything: a row with fit_valid = 0 has NaN τ/R²/coverage, and one with
# b3_frames = 0 has no v6 features at all. Mixed in among 25 numbers they would be
# missed, and a reviewer would judge a row on values that are not measurements.
# `n_seg_valid` is NOT here: v6 has no such column, because it is exactly
# `n_seg >= V6_MIN_NSEG`, and the banner below derives it rather than reading it.
_QUALITY_FMT = [
    ('fit_valid',   0), ('b3_frames',  0), ('n_seg',       0),
    ('decay_len',   0), ('gibbs_fired', 0),
]
_FEATURE_COLS = 3    # name/value pairs side by side. History: 3 -> 5 -> 6 -> 4 -> 3.
                     # Qt elides the NAME, not the number, so an over-wide grid fails
                     # SILENTLY — a row reads "would_pass_v...: 1" and nothing says it
                     # was cut. The grid must therefore fit at the dialog's MINIMUM
                     # width, not at whatever size it happens to open at.
                     #
                     # Measured need for the 35-entry v6 table (widest name + a
                     # "-48779.30" value cell, per column, + spacing):
                     #     4 cols @ 11 pt -> ~1417 px   (what was cutting text)
                     #     4 cols @  9 pt -> ~1239 px
                     #     3 cols @ 11 pt -> ~1187 px
                     #     3 cols @ 10 pt -> ~1094 px   <- fits under the 1100 minimum
                     # 3 x 10 pt is the only combination that fits without the window
                     # having to be widened, and it costs ~60 px of height (318 vs
                     # 290) — spent against the height budget, not the width.
_FEATURE_PT = 10     # explicit point size for the grid. The platform default (~13 on
                     # macOS) is what made the names elide in the first place; the
                     # numbers are what must stay readable and 10 pt keeps them so.


def _to_float(value, default: float = float('nan')) -> float:
    """
    Parse a CSV cell that may use an Italian-locale decimal comma ("0,5888").

    Candidate CSVs are routinely opened and re-saved in Excel, which rewrites every
    float with a comma on an Italian system. evaluate_candidates.load_csvs already
    defends against this; the dialog has to as well, or it dies on the user's own
    saved files.
    """
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text.replace(',', '.'))
    except ValueError:
        return default

# Label values as they appear in the CSV
LABEL_CLICK = '1'
LABEL_NOISE = '0'
LABEL_NONE  = ''

# Row tint per verdict — a blocked candidate should be recognisable without reading.
_VERDICT_TINT = {
    '':             QColor(46, 125, 50, 60),    # confirmed click — green
    'Stage2_R2':    QColor(120, 120, 120, 45),  # invalid fit — grey, barely worth a look
    'Stage2_SPR':   QColor(120, 120, 120, 45),
    'Stage3_SVM':   QColor(211, 47, 47, 45),    # the SVM said noise — red
    'Stage4_dedup': QColor(255, 152, 0, 45),    # duplicate of a stronger detection — amber
}


class ClickReviewDialog(QDialog):
    """
    Review and label the candidates in a *_candidates.csv.

    Parameters
    ----------
    csv_path : Path, optional
        Opened immediately. None → the user picks a file with the Open button.
    theme_manager : passed explicitly — a child dialog cannot reliably reach the
        main window's theme_manager through its parent chain.
    """

    _LIGHT_CSS = """
        QDialog, QWidget { background-color: white; color: black; }
        QLabel { color: black; background: transparent; }
        QPushButton { background-color: #f0f0f0; color: black;
                      border: 1px solid #ccc; padding: 4px 8px; border-radius: 3px; }
        QPushButton:hover { background-color: #e0e0e0; }
        QPushButton:disabled { color: #999; }
        QComboBox { background-color: white; color: black;
                    border: 1px solid #ccc; padding: 2px 4px; border-radius: 3px; }
        QComboBox QAbstractItemView { background-color: white; color: black;
                                      selection-background-color: #d0e4ff;
                                      selection-color: black; }
        QTableWidget { background-color: white; color: black;
                       gridline-color: #ddd; }
        QHeaderView::section { background-color: #f0f0f0; color: black;
                               border: 1px solid #ddd; padding: 3px; }
        QGroupBox { color: black; }
        QSplitter::handle { background-color: #ddd; }
    """

    def __init__(self, csv_path: Optional[Path] = None, parent=None, theme_manager=None):
        super().__init__(parent)

        self.settings_manager = SettingsManager()
        self.theme_manager = theme_manager
        self.csv_path: Optional[Path] = None
        self.df = None                # pandas DataFrame, all columns kept as text
        self.visible_rows: list = []  # df row indices, in the order the table shows them
        self._pixmap = None           # the cropped screenshot (scaled on show/resize)
        self._png_index = None        # lower-cased name → path, built lazily
        self._click_names = None      # df row → 'click<sec>[_n].png', built lazily
        self._note_row = None         # df row the note box is currently editing
        # Must exist BEFORE _build_ui: the table's event filter is installed early
        # in _build_ui, and any event arriving before the note box is constructed
        # runs eventFilter, where a missing attribute raises inside a Qt virtual
        # callback — which segfaults instead of raising.
        self.note_edit = None
        self.screenshots_dir: Optional[Path] = None   # None → look beside the CSV

        self.setWindowTitle("Click Review — label candidates")
        # Minimum width is what the feature grid is sized against (see
        # _FEATURE_COLS): the grid must not elide at the smallest the window can be,
        # because elision is silent. Height covers the taller 3-column grid.
        self.setMinimumSize(1100, 760)
        self.resize(1280, 920)

        self._build_ui()
        self._apply_theme()
        self._center_on_parent(parent)

        if csv_path is not None:
            self._load_csv(Path(csv_path))

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)

        # ── File row ──
        file_row = QHBoxLayout()
        self.btn_open = QPushButton("Open CSV...")
        self.btn_open.clicked.connect(self._on_open)
        file_row.addWidget(self.btn_open)

        self.label_file = QLabel("(no file loaded)")
        file_row.addWidget(self.label_file, stretch=1)

        self.btn_shots_dir = QPushButton("Screenshots folder...")
        self.btn_shots_dir.setToolTip(
            "By default screenshots are looked up next to the CSV (sub-folders included).\n"
            "Point this somewhere else if you keep them apart."
        )
        self.btn_shots_dir.clicked.connect(self._on_choose_screenshots_dir)
        file_row.addWidget(self.btn_shots_dir)

        file_row.addWidget(QLabel("Show:"))
        self.combo_filter = WideComboBox()
        self.combo_filter.addItems([
            "All candidates",
            "Unlabelled only",
            "Confirmed clicks",
            "Rejected by SVM (Stage3_SVM)",
            "Blocked by gates (Stage 2)",
            # APPEND ONLY — _refresh_table branches on the raw index below.
            "Needs review (v6 queue)",
        ])
        self.combo_filter.currentIndexChanged.connect(self._refresh_table)
        file_row.addWidget(self.combo_filter)

        file_row.addWidget(QLabel("Sort:"))
        self.combo_sort = WideComboBox()
        self.combo_sort.addItems([
            "P(click) — highest first",
            "P(click) — lowest first",
            "Frame order",
            # APPEND ONLY — _refresh_table branches on the raw index below.
            "Review queue (tier order)",
        ])
        self.combo_sort.currentIndexChanged.connect(self._refresh_table)
        file_row.addWidget(self.combo_sort)

        root.addLayout(file_row)

        # ── Splitter: table | screenshot ──
        splitter = QSplitter(Qt.Horizontal)

        self.table = QTableWidget()
        self.table.setColumnCount(_N_COLS)
        self.table.setHorizontalHeaderLabels(
            ["Time (s)", "P(click)", "Verdict", "Label"]
        )
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.currentCellChanged.connect(lambda *_: self._show_current())
        # The table keeps focus while labelling, so it must forward our keys (see eventFilter).
        self.table.installEventFilter(self)
        hdr = self.table.horizontalHeader()
        # Label absorbs the slack. It is the column being edited and the one the eye
        # returns to after every keystroke, so it gets the room; Verdict is
        # content-sized instead. (This was the other way round, on the reasoning that
        # Verdict holds the longest text — but 'Stage4_dedup' is a fixed vocabulary
        # that ResizeToContents fits exactly, so stretching it only padded whitespace.)
        for c in (_COL_TIME, _COL_PROB, _COL_VERDICT):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(_COL_LABEL, QHeaderView.Stretch)
        hdr.setStretchLastSection(False)   # otherwise it overrides the mode above
        # NOTE: no setMinimumSectionSize here. It is a HEADER-wide floor, not a
        # per-column one, so raising it to widen Label would pad Time / P(click) /
        # Verdict to the same width and take back the slack Stretch just gave Label.
        self.table.setMinimumWidth(360)
        splitter.addWidget(self.table)

        # Right-hand side: the cropped screenshot on top, its features underneath.
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = QLabel("Select a candidate")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(1, 1)   # let it shrink; _rescale_pixmap sizes it
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.image_scroll = QScrollArea()
        self.image_scroll.setWidget(self.image_label)
        self.image_scroll.setWidgetResizable(True)
        right_layout.addWidget(self.image_scroll, stretch=1)

        right_layout.addWidget(self._build_features_group())

        splitter.addWidget(right)

        # The screenshot is the thing being judged — give it the room.
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([330, 1000])
        root.addWidget(splitter, stretch=1)

        # ── Metrics ──
        root.addWidget(self._build_metrics_group())

        # ── Note editor ──
        # Free text per row, saved with the labels. Deliberately a single line and
        # deliberately NOT in the table: it must be reachable without breaking the
        # 1/0/Space rhythm, and a multi-line box would swallow Enter.
        note_row = QHBoxLayout()
        note_lbl = QLabel("Note:")
        note_lbl.setStyleSheet("QLabel { font-weight: bold; }")
        note_row.addWidget(note_lbl)
        self.note_edit = QLineEdit()
        self.note_edit.setPlaceholderText(
            "free text for this row — Enter or Tab saves, Esc returns to the list"
        )
        self.note_edit.setClearButtonEnabled(True)
        # Saved on Enter and on focus loss. Focus loss covers the case that actually
        # loses work: typing a note and then clicking the next row without pressing
        # Enter. _show_row also flushes before it moves, so keyboard navigation
        # cannot drop one either.
        self.note_edit.editingFinished.connect(self._commit_note)
        self.note_edit.installEventFilter(self)
        note_row.addWidget(self.note_edit, stretch=1)
        root.addLayout(note_row)

        # ── Bottom bar ──
        bottom = QHBoxLayout()
        hint = QLabel(
            "Keys:  1 = click   0 = noise   Backspace = clear   "
            "Space / ↓ = next   ↑ = previous   N = note"
        )
        hint.setStyleSheet("QLabel { color: gray; font-style: italic; }")
        bottom.addWidget(hint)
        bottom.addStretch()

        self.label_progress = QLabel("")
        bottom.addWidget(self.label_progress)

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)
        bottom.addWidget(self.btn_close)
        root.addLayout(bottom)

        # NO default button anywhere in this dialog. Qt promotes the first
        # autoDefault QPushButton it finds, so Return pressed in any line edit was
        # activating whatever that happened to be — 'Close' in the note box's case,
        # which shut the window mid-note, and 'Open CSV...' from anywhere else.
        # Clearing the flag on every button kills the whole class of bug rather than
        # the one instance of it. This dialog saves continuously and has no
        # confirm-and-dismiss action, so it has nothing a default button is for.
        for _btn in self.findChildren(QPushButton):
            _btn.setAutoDefault(False)
            _btn.setDefault(False)

    def _build_features_group(self):
        """
        The 17 features of the selected candidate, laid out to actually be readable.

        This replaces the footer that used to be baked into the PNG: at the scale the
        screenshot is displayed, that text was illegible. Here the values are live
        widgets, so they stay sharp at any window size and can be selected and copied.
        """
        group = QGroupBox("Features")
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(2)

        name_font = QFont()
        name_font.setBold(True)
        name_font.setPointSize(_FEATURE_PT)
        value_font = QFont("Courier New")
        value_font.setPointSize(_FEATURE_PT)

        self._feature_values = {}   # feature name → its value QLabel

        # ── Validity banner — spans the grid, above everything ───────────────
        # Plain text, red when anything is wrong. A reviewer must see "this row's
        # numbers are not measurements" before reading the numbers, not after.
        self.validity_lbl = QLabel("")
        self.validity_lbl.setWordWrap(True)
        vf = QFont(); vf.setBold(True)
        self.validity_lbl.setFont(vf)
        grid.addWidget(self.validity_lbl, 0, 0, 1, _FEATURE_COLS * 2)

        row0 = 1
        n_rows = -(-len(_FEATURE_FMT) // _FEATURE_COLS)   # ceil
        for idx, (name, _prec) in enumerate(_FEATURE_FMT):
            r = row0 + idx % n_rows
            c = idx // n_rows

            name_lbl = QLabel(f"{name}:")
            name_lbl.setFont(name_font)

            value_lbl = QLabel("—")
            value_lbl.setFont(value_font)
            value_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)

            grid.addWidget(name_lbl,  r, c * 2)
            grid.addWidget(value_lbl, r, c * 2 + 1)
            self._feature_values[name] = value_lbl

        # ── Quality flags below the features, visually separated ──────────────
        # WRAPPED at _FEATURE_COLS like the features above. They used to be laid on
        # a single row, which was fine at 6 columns and silently wider than the whole
        # feature grid at 4 — one over-long row stretches the QGroupBox and undoes
        # the horizontal saving the narrower grid was made for.
        # A real rule between the two, because the comment above used to claim
        # "visually separated" while nothing separated them: the flags read as four
        # more features, which is the opposite of the point. They are what say
        # whether the numbers above are measurements at all.
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        grid.addWidget(sep, row0 + n_rows, 0, 1, _FEATURE_COLS * 2)

        qrow0 = row0 + n_rows + 1
        for idx, (name, _prec) in enumerate(_QUALITY_FMT):
            r = qrow0 + idx // _FEATURE_COLS
            c = idx % _FEATURE_COLS
            name_lbl = QLabel(f"{name}:")
            name_lbl.setFont(name_font)
            value_lbl = QLabel("—")
            value_lbl.setFont(value_font)
            value_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            grid.addWidget(name_lbl,  r, c * 2)
            grid.addWidget(value_lbl, r, c * 2 + 1)
            self._feature_values[name] = value_lbl

        # Let the value columns take the slack, so the names stay tight against them.
        for c in range(_FEATURE_COLS):
            grid.setColumnStretch(c * 2 + 1, 1)

        group.setLayout(grid)
        return group

    def _update_features(self, i: Optional[int]):
        """Refresh the feature panel for df row `i` (None → clear it)."""
        fmt = dict(_FEATURE_FMT + _QUALITY_FMT)
        for name, _prec in _FEATURE_FMT + _QUALITY_FMT:
            label = self._feature_values[name]

            if i is None or self.df is None or name not in self.df.columns:
                label.setText("—")
                continue

            value = _to_float(self.df.at[i, name])
            if value != value:          # NaN → the column exists but the cell is empty
                label.setText("—")
                continue

            label.setText(f"{value:.{fmt[name]}f}")

        self._load_note(i)
        self._update_validity(i)

    def _update_validity(self, i: Optional[int]):
        """
        Say plainly when the numbers above are not measurements.

        A row with fit_valid = 0 carries NaN for τ / R² / fit_coverage; one with
        b3_frames = 0 has no Buffer-3 estimate so every v6 feature is NaN; one with
        n_seg below V6_MIN_NSEG has a region too short for the 12-band grid, which
        biases entropy toward 1 (§4.3). Those rows are exactly review tier 4 — the
        population that has never been labelled by anyone — so the warning has to
        be impossible to miss rather than one "—" among twenty-five numbers.
        """
        if i is None or self.df is None:
            self.validity_lbl.setText("")
            return

        def _flag(col, default=1.0):
            if col not in self.df.columns:
                return None
            v = _to_float(self.df.at[i, col])
            return default if v != v else v

        warn = []
        if _flag('fit_valid') == 0:
            warn.append("FIT INVALID — τ / R² / fit_coverage are not meaningful")
        if _flag('b3_frames') == 0:
            warn.append("NO BUFFER-3 ESTIMATE — every v6 spectral feature is unavailable")
        # Derived, not read: v6 has no n_seg_valid column. Development-era
        # and older CSVs still carry it, so honour it when it is there and fall
        # back to the definition when it is not.
        n_seg_ok = _flag('n_seg_valid')
        if n_seg_ok is None:
            n_seg = _flag('n_seg', float('nan'))
            n_seg_ok = 1 if (n_seg != n_seg or n_seg >= _V6_MIN_NSEG) else 0
        if n_seg_ok == 0:
            warn.append("REGION TOO SHORT — bands correlated, entropy biased high (§4.3)")
        if _flag('gibbs_fired', 0.0) == 1:
            warn.append("Gibbs fade fired — the noise subtraction is biased on this frame")

        note = str(self.df.at[i, 'migration_note']) if 'migration_note' in self.df.columns else ''
        tier = _flag('review_tier', float('nan'))

        parts = []
        if warn:
            parts.append("⚠  " + "   ·   ".join(warn))
        if tier == tier and tier:
            parts.append(f"review tier {int(tier)}" + (f" — {note}" if note else ""))
        elif note:
            parts.append(note)

        self.validity_lbl.setText("\n".join(parts))
        self.validity_lbl.setStyleSheet(
            "color: #d84343;" if warn else "color: #8a8a8a;")

    def _build_metrics_group(self):
        group = QGroupBox("Your labels vs. the algorithm")
        layout = QHBoxLayout()

        self.label_confusion = QLabel("—")
        self.label_confusion.setFont(QFont("Courier New", 10))
        layout.addWidget(self.label_confusion)

        layout.addStretch()

        self.label_metrics = QLabel("")
        self.label_metrics.setFont(QFont("Courier New", 10))
        layout.addWidget(self.label_metrics)

        group.setLayout(layout)
        return group

    # ── Loading ───────────────────────────────────────────────────────────────

    def _on_open(self):
        start = str(self.csv_path.parent) if self.csv_path else self.settings_manager.get_last_directory("review_csv")
        path, _ = QFileDialog.getOpenFileName(
            self, "Open candidates CSV", start, "Candidate CSV (*.csv)"
        )
        if path:
            self._load_csv(Path(path))
            self.settings_manager.set_last_directory("review_csv", path)

    def _on_choose_screenshots_dir(self):
        """Look for screenshots somewhere other than beside the CSV."""
        start = str(self._search_root()) if self.csv_path else self.settings_manager.get_last_directory("review_screenshots")
        chosen = QFileDialog.getExistingDirectory(self, "Screenshots folder", start)
        if not chosen:
            return

        self.screenshots_dir = Path(chosen)
        self._png_index = None          # different tree → the cached index is stale
        self.btn_shots_dir.setToolTip(f"Screenshots folder: {chosen}")
        self.settings_manager.set_last_directory("review_screenshots", chosen)
        self._show_current()

    def _load_csv(self, path: Path):
        """
        Load the CSV as text.

        dtype=str + keep_default_na=False is deliberate: every column is round-tripped
        back to disk byte-for-byte and only `label` is ever rewritten. Parsing the
        floats would silently reformat the feature columns of the user's training data.
        """
        import pandas as pd

        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
        except Exception as e:
            QMessageBox.warning(self, "Could not open CSV", f"{path}\n\n{e}")
            return

        if 'frame_idx' not in df.columns:
            QMessageBox.warning(
                self, "Not a candidates CSV",
                f"{path.name} has no 'frame_idx' column — it does not look like a "
                f"file exported by the Data Collection dialog."
            )
            return

        if 'label' not in df.columns:
            df['label'] = LABEL_NONE

        # Normalise the label spelling. The existing corpus is a documented mix of
        # '1' / '1.0' / '0' / '0.0' / '' (migrate_labels_v6.py:100). A '1.0' here
        # fails three ways at once and silently: the table cell renders blank, the
        # row hides under "Unlabelled only", and the metrics count it as neither
        # class — so a labelled click looks unlabelled and is labelled again.
        def _norm_label(v):
            s = str(v).strip().replace(',', '.')
            if not s:
                return LABEL_NONE
            try:
                f = float(s)
            except ValueError:
                return LABEL_NONE
            if f == 1.0:
                return LABEL_CLICK
            if f == 0.0:
                return LABEL_NOISE
            return LABEL_NONE

        df['label'] = df['label'].map(_norm_label)
        # Older CSVs predate the note column. Create it rather than disabling the
        # editor, so a file exported before notes existed can still be annotated —
        # the column is simply written on the next save.
        if 'note' not in df.columns:
            df['note'] = ''

        self._note_row = None     # the previous file's row index means nothing here
        self.df = df
        self.csv_path = path
        self._png_index = None    # new folder → rebuild the screenshot index lazily
        self._click_names = None  # CLICK<sec> names are derived from this file's labels
        self.label_file.setText(str(path))
        self.table.setFocus()    # so the labelling keys work without clicking first
        self._refresh_table()

    # ── Table ─────────────────────────────────────────────────────────────────

    def _prob(self, i: int) -> float:
        """svm_probability of df row i as a float; -1 when absent or unscored."""
        if self.df is None or 'svm_probability' not in self.df.columns:
            return -1.0
        return _to_float(self.df.at[i, 'svm_probability'], -1.0)

    def _verdict(self, i: int) -> str:
        if 'stage_blocked' not in self.df.columns:
            return ''
        return str(self.df.at[i, 'stage_blocked'])

    def _is_classified(self) -> bool:
        return self.df is not None and 'stage_blocked' in self.df.columns

    def _refresh_table(self):
        if self.df is None:
            return

        # ── Filter ──
        mode = self.combo_filter.currentIndex()
        rows = list(range(len(self.df)))

        if mode == 1:
            rows = [i for i in rows if self.df.at[i, 'label'] == LABEL_NONE]
        elif mode == 2 and self._is_classified():
            rows = [i for i in rows if self._verdict(i) == '']
        elif mode == 3 and self._is_classified():
            rows = [i for i in rows if self._verdict(i) == 'Stage3_SVM']
        elif mode == 4 and self._is_classified():
            rows = [i for i in rows if self._verdict(i).startswith('Stage2')]
        elif mode == 5 and 'needs_review' in self.df.columns:
            # The v6 review queue produced by scripts/migrate_labels_v6.py. Rows
            # whose label migrated cleanly are already settled and are not here.
            rows = [i for i in rows
                    if _to_float(self.df.at[i, 'needs_review'], 0.0) == 1]

        # ── Sort ──
        sort_mode = self.combo_sort.currentIndex()
        if sort_mode == 0:
            rows.sort(key=self._prob, reverse=True)
        elif sort_mode == 1:
            rows.sort(key=self._prob)
        elif sort_mode == 3 and 'review_tier' in self.df.columns:
            # Tier order, then click-likeness within a tier. Tier 1 (ambiguous
            # migrations, including outright contradictions) must be adjudicated
            # first because everything downstream inherits those decisions.
            # Unflagged rows sort last rather than being hidden.
            rows.sort(key=lambda i: (
                _to_float(self.df.at[i, 'review_tier'], 99.0) or 99.0,
                -_to_float(self.df.at[i, 'clicklike_rank'], 0.0)
                if 'clicklike_rank' in self.df.columns else 0.0,
                _to_float(self.df.at[i, 'frame_idx'], 0.0),
            ))
        else:
            rows.sort(key=lambda i: _to_float(self.df.at[i, 'frame_idx'], 0.0))

        self.visible_rows = rows

        # ── Populate ──
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))

        for r, i in enumerate(rows):
            prob = self._prob(i)
            verdict = self._verdict(i) if self._is_classified() else ''

            ts = (_to_float(self.df.at[i, 'timestamp_s'])
                  if 'timestamp_s' in self.df.columns else float('nan'))

            cells = [
                f"{ts:.3f}" if ts == ts else '—',   # NaN-safe
                f"{prob:.3f}" if prob >= 0 else '—',
                ('CLICK' if verdict == '' else verdict) if self._is_classified() else '—',
                {LABEL_CLICK: 'click', LABEL_NOISE: 'noise'}.get(
                    self.df.at[i, 'label'], ''),
            ]

            tint = _VERDICT_TINT.get(verdict) if self._is_classified() else None
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if tint is not None:
                    item.setBackground(tint)
                self.table.setItem(r, c, item)

        self.table.blockSignals(False)

        if rows:
            self.table.setCurrentCell(0, 0)
        else:
            self.image_label.setText("No candidates match this filter")

        self._update_metrics()

    def _current_df_row(self) -> Optional[int]:
        r = self.table.currentRow()
        if r < 0 or r >= len(self.visible_rows):
            return None
        return self.visible_rows[r]

    def _search_root(self) -> Path:
        """The single best root, kept for callers that want one path to show."""
        roots = self._search_roots()
        return roots[0] if roots else self.csv_path.parent

    def _search_roots(self) -> list:
        """
        Every directory worth searching for this CSV's screenshots, best first.

        The export writes ONE FOLDER PER RECORDING under two sibling trees:

            <out>/CSVs/<stem>/<stem>_candidates.csv
            <out>/screenshots/<stem>/<stem>_<frame>.png

        so the screenshots are NOT under the CSV's own folder, nor under its
        parent — searching `csv_path.parent` alone (which is what this did) finds
        nothing at all under that layout. Walk up instead, and take any
        `screenshots` tree found on the way, so the older flat layout
        (`<out>/<stem>_candidates.csv` beside `<out>/screenshots/`) keeps working
        without a migration.

        An explicitly chosen folder always wins and is used alone: if the user
        picked a directory, second-guessing them with inferred ones would show a
        screenshot from somewhere they did not ask for.
        """
        if self.screenshots_dir:
            return [self.screenshots_dir]
        if self.csv_path is None:
            return []

        roots, seen = [], set()

        def _add(p: Path):
            try:
                rp = p.resolve()
            except OSError:
                return
            if rp not in seen and p.is_dir():
                seen.add(rp)
                roots.append(p)

        here = self.csv_path.parent
        # Up to three levels: <stem>/ -> CSVs/ -> <out>/. Bounded deliberately —
        # an unbounded walk to / would index the whole disk on a stray CSV.
        for up in (here, here.parent, here.parent.parent):
            shots = up / SCREENSHOTS_FOLDER
            if shots.is_dir():
                # Prefer this recording's own sub-folder when it exists.
                _add(shots / self.csv_path.stem.replace('_candidates', ''))
                _add(shots)
        _add(here)
        return roots

    def _png_lookup(self):
        """
        Lazily index every PNG under the search root, by lower-cased filename.

        One recursive pass, reused for every candidate: rescanning per candidate would
        make stepping through a big export crawl. Sub-folders are included because the
        user files screenshots into them (screenshots/stage 2/…).
        """
        if self._png_index is None:
            self._png_index = {}
            for root in self._search_roots():
                for png in root.rglob('*.png'):
                    # setdefault, not assignment: _search_roots is ordered
                    # best-first, so the first root to supply a name wins.
                    self._png_index.setdefault(png.name.lower(), png)
        return self._png_index

    def _click_ordinals(self) -> dict:
        """
        Map each labelled-click row to its name under the CLICK<second> convention.

        The user's older analyses name a click's screenshot after its whole-second
        timestamp — a click at 58.29888 s is CLICK58.png — and when several clicks land
        in the same second they are numbered in time order: CLICK58, CLICK58_2, CLICK58_3.

        The ordinal counts CLICKS, not candidates. In the DIONEA recording, second 58
        holds 11 candidates but only 3 labelled clicks, and those 3 are what _2 and _3
        refer to. So the ranking is over label==1 rows only, which also means this
        convention can only resolve for rows the user has already labelled as clicks —
        exactly the rows those screenshots were made for.

        Returns
        -------
        dict : df row index → filename (lower-cased, e.g. 'click58_2.png')
        """
        if self._click_names is not None:
            return self._click_names

        self._click_names = {}
        if self.df is None or 'timestamp_s' not in self.df.columns:
            return self._click_names

        # Gather the labelled clicks, grouped by whole second, ordered in time.
        by_second: dict = {}
        for i in range(len(self.df)):
            if self.df.at[i, 'label'] != LABEL_CLICK:
                continue
            ts = _to_float(self.df.at[i, 'timestamp_s'])
            if ts != ts:
                continue
            by_second.setdefault(int(ts), []).append((ts, i))

        for second, entries in by_second.items():
            entries.sort()
            for n, (_ts, i) in enumerate(entries, start=1):
                suffix = '' if n == 1 else f'_{n}'
                self._click_names[i] = f"click{second}{suffix}.png"

        return self._click_names

    def _find_screenshot(self, row: int, stem: str, frame: int):
        """
        Locate the PNG for df row `row`, trying both naming conventions.

        1. The export's own name: {file}_{frame:06d}.png — checked directly first, so
           the common case costs one stat() and never builds the index.
        2. The user's CLICK<second> convention (see _click_ordinals), for screenshots
           kept from an earlier analysis.
        """
        standard = f"{stem}_{frame:06d}.png"
        # Fast path: try the exact places the export writes to, so the common case
        # costs a couple of stat() calls and never builds the whole index.
        for root in self._search_roots():
            for direct in (root / standard,
                           root / stem / standard,
                           root / SCREENSHOTS_FOLDER / standard,
                           root / SCREENSHOTS_FOLDER / stem / standard):
                if direct.exists():
                    return direct

        index = self._png_lookup()

        hit = index.get(standard.lower())
        if hit is not None:
            return hit

        click_name = self._click_ordinals().get(row)
        if click_name is not None:
            return index.get(click_name)

        return None

    def _show_current(self):
        """Display the screenshot the export already rendered for this candidate."""
        i = self._current_df_row()
        self._update_features(i)

        if i is None:
            return

        stem  = str(self.df.at[i, 'file'])
        frame = int(_to_float(self.df.at[i, 'frame_idx'], -1))

        png = self._find_screenshot(i, stem, frame)

        if png is None:
            self._pixmap = None
            self.image_label.setPixmap(QPixmap())
            tried = f"{stem}_{frame:06d}.png"
            alt = self._click_ordinals().get(i)
            if alt:
                tried += f"  or  {alt}"
            self.image_label.setText(
                f"No screenshot found for frame {frame}\n\n"
                f"Looked for {tried}\n"
                # Every root, not just the first: with the per-recording layout
                # there are several, and naming one of them would send the user
                # to check a directory that was never the problem.
                + "under:\n" + "\n".join(f"  {r}" for r in self._search_roots())
                + "\n(including sub-folders)"
            )
            return

        pix = QPixmap(str(png))
        if pix.isNull():
            self._pixmap = None
            self.image_label.setText(f"Could not read {png.name}")
            return

        # Crop the title strip and the feature footer: the footer is unreadable once
        # scaled, and its numbers are shown properly in the Features panel below.
        h = pix.height() - _CROP_TOP - _CROP_BOTTOM
        if h > 0:
            pix = pix.copy(0, _CROP_TOP, pix.width(), h)

        self._pixmap = pix
        self._rescale_pixmap()

    def _rescale_pixmap(self):
        """Fit the screenshot to the whole pane, preserving aspect ratio."""
        if getattr(self, '_pixmap', None) is None:
            return

        area = self.image_scroll.viewport().size()
        if area.width() < 10 or area.height() < 10:
            return

        self.image_label.setPixmap(
            self._pixmap.scaled(area, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def resizeEvent(self, event):
        """Keep the screenshot filling the pane as the dialog is resized."""
        super().resizeEvent(event)
        self._rescale_pixmap()

    # ── Labelling ─────────────────────────────────────────────────────────────

    def eventFilter(self, obj, event):
        """
        Route the labelling keys even while the table has focus.

        The table is what the user clicks, so it holds focus — and QTableWidget eats
        key presses (digits drive its type-to-search, Space toggles selection), so a
        keyPressEvent on the dialog alone never fires. Filtering the table's events is
        the only way 1/0 reach us.
        """
        # While the note box has focus, EVERY key belongs to it — '1' and '0' are
        # ordinary characters in a note, and routing them to the labeller would
        # silently relabel the row the user is annotating. Only Escape is taken,
        # to hand focus back to the list.
        if self.note_edit is not None and obj is self.note_edit \
                and event.type() == QEvent.KeyPress:
            # Return/Enter in a QLineEdit propagates to the dialog's default button,
            # which closed the whole review window mid-note. Commit and swallow it:
            # in this dialog Enter means "save this note", never "I am done here".
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self._commit_note()
                self.table.setFocus()
                return True
            if event.key() == Qt.Key_Escape:
                self._commit_note()
                self.table.setFocus()
                return True
            return False

        if obj is self.table and event.type() == QEvent.KeyPress:
            if self._handle_label_key(event.key()):
                return True   # consumed — don't let the table also act on it
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        if not self._handle_label_key(event.key()):
            super().keyPressEvent(event)

    def _handle_label_key(self, key) -> bool:
        """Apply a labelling keystroke. Returns True if the key was one of ours."""
        if key == Qt.Key_1:
            self._set_label(LABEL_CLICK, advance=True)
        elif key == Qt.Key_0:
            self._set_label(LABEL_NOISE, advance=True)
        elif key in (Qt.Key_Backspace, Qt.Key_Delete):
            self._set_label(LABEL_NONE, advance=False)
        elif key == Qt.Key_Space:
            self._advance()
        elif key == Qt.Key_N:
            self.note_edit.setFocus()
            self.note_edit.selectAll()
        else:
            return False
        return True

    def _set_label(self, value: str, advance: bool):
        i = self._current_df_row()
        if i is None:
            return

        self.df.at[i, 'label'] = value

        # The CLICK<sec> ordinals are derived from which rows are labelled clicks, so
        # changing a label invalidates them.
        self._click_names = None

        # Reflect it in the table without rebuilding (which would move the cursor).
        r = self.table.currentRow()
        text = {LABEL_CLICK: 'click', LABEL_NOISE: 'noise'}.get(value, '')
        self.table.item(r, _COL_LABEL).setText(text)

        self._save()
        self._update_metrics()

        if advance:
            self._advance()

    def _commit_note(self):
        """
        Write the note box back to the row it was opened on.

        Keyed to `self._note_row`, NOT to the current selection: editingFinished
        fires on focus loss, which can arrive after the user has already clicked a
        different row, and writing to the current row would put the note on the
        wrong candidate.
        """
        i = self._note_row
        if i is None or self.df is None or 'note' not in self.df.columns:
            return
        new = self.note_edit.text()
        if str(self.df.at[i, 'note']) == new:
            return                      # nothing changed — don't rewrite the CSV
        self.df.at[i, 'note'] = new
        self._save()

    def _load_note(self, i):
        """Point the note box at row i, flushing whatever was in it first."""
        if self._note_row is not None and self._note_row != i:
            self._commit_note()
        self._note_row = i
        if self.df is None or i is None or 'note' not in self.df.columns:
            self.note_edit.clear()
            self.note_edit.setEnabled(False)
            return
        self.note_edit.setEnabled(True)
        v = self.df.at[i, 'note']
        self.note_edit.setText('' if v != v else str(v))   # NaN -> ''

    def _advance(self):
        r = self.table.currentRow()
        if r + 1 < self.table.rowCount():
            self.table.setCurrentCell(r + 1, 0)

    def _save(self):
        """
        Write the CSV back after every label.

        Candidate CSVs are small (thousands of rows at most) and a lost labelling
        session is far more expensive than a rewrite, so there is no explicit Save.
        """
        try:
            # Write to a sibling temp file and rename over the original. to_csv()
            # truncates in place, so a crash or a full disk mid-write used to leave
            # a half-written CSV and lose every label in it — and this runs after
            # EVERY keystroke. os.replace is atomic on the same filesystem.
            import os
            import tempfile
            fd, tmp = tempfile.mkstemp(dir=str(self.csv_path.parent),
                                       prefix='.' + self.csv_path.name + '.',
                                       suffix='.tmp')
            os.close(fd)
            try:
                self.df.to_csv(tmp, index=False)
                os.replace(tmp, self.csv_path)
            except BaseException:
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise
        except Exception as e:
            QMessageBox.warning(
                self, "Could not save",
                f"Labels were NOT written to:\n{self.csv_path}\n\n{e}"
            )

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _update_metrics(self):
        if self.df is None:
            return

        total = len(self.df)
        n_click = int((self.df['label'] == LABEL_CLICK).sum())
        n_noise = int((self.df['label'] == LABEL_NOISE).sum())
        labelled = n_click + n_noise

        # Your own tally, from the label column alone. It must never depend on the
        # verdict column: these are your labels, and they are just as real on a CSV
        # exported without Stages 2-4 as on one exported with them.
        self.label_progress.setText(
            f"{labelled} / {total} labelled   —   {n_click} click, {n_noise} noise"
        )

        if not self._is_classified():
            # No verdicts to compare against, but the counts above are still yours to see.
            self.label_confusion.setText(
                f"You marked {n_click} click / {n_noise} noise of {total} candidates.\n"
                f"No algorithm verdicts in this CSV — re-export with 'Run Stages 2-4' "
                f"enabled to compare them."
            )
            self.label_metrics.setText("")
            return

        # Confusion of the user's labels against the pipeline's confirmed clicks.
        tp = fp = fn = tn = 0
        for i in range(total):
            lab = self.df.at[i, 'label']
            if lab == LABEL_NONE:
                continue
            predicted_click = self._verdict(i) == ''
            if lab == LABEL_CLICK:
                tp += predicted_click
                fn += not predicted_click
            elif lab == LABEL_NOISE:
                fp += predicted_click
                tn += not predicted_click

        self.label_confusion.setText(
            f"TP={tp}   FP={fp}   FN={fn}   TN={tn}"
        )

        if tp + fn == 0 and tp + fp == 0:
            self.label_metrics.setText("(label some candidates to see metrics)")
            return

        def ratio(num, den):
            return f"{num / den:.3f}" if den else "—"

        self.label_metrics.setText(
            f"recall={ratio(tp, tp + fn)}   "
            f"precision={ratio(tp, tp + fp)}   "
            f"specificity={ratio(tn, tn + fp)}"
        )

    # ── Theme / geometry (same approach as RegionFFTDialog) ───────────────────

    def _apply_theme(self):
        tm = self.theme_manager
        if tm is None:
            return
        try:
            saved = tm.load_saved_theme()
            tm.apply_theme(self, saved)
            if 'light' in saved.lower():
                self.setStyleSheet(self._LIGHT_CSS)
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ ClickReviewDialog: theme not applied ({e})")

    def _center_on_parent(self, parent):
        if not parent:
            return
        r = parent.geometry()
        self.resize(int(r.width() * 0.9), int(r.height() * 0.9))
        self.move(r.x() + (r.width() - self.width()) // 2,
                  r.y() + (r.height() - self.height()) // 2)
