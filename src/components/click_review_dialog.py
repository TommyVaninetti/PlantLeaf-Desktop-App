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
)
from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import QPixmap, QColor, QFont

from components.wide_combo_box import WideComboBox


SCREENSHOTS_FOLDER = 'screenshots'

# Table columns. The frame index is deliberately not shown: it costs width and tells
# the user nothing they can act on.
_COL_TIME, _COL_PROB, _COL_VERDICT, _COL_LABEL = range(4)
_N_COLS = 4

# The exported PNG is 1400x800: a title strip on top, the FFT/iFFT plots in the middle,
# a feature footer at the bottom. Scaled down to fit a pane, that footer text is far too
# small to read — so it is cropped away and the same numbers are shown in a real table
# underneath instead (see _build_features_group).
# Keep in step with the layout in data_collection_dialog_v5._render_candidate_screenshot.
_CROP_TOP    = 40    # title strip
_CROP_BOTTOM = 260   # feature footer

# The 17 features in the order the v5 docs list them, each with a display precision.
_FEATURE_FMT = [
    ('peak_SNR',     3), ('pre_SNR',    3), ('post_SNR',           3),
    ('rise_time_ms', 4), ('fall_time_ms', 4), ('asymmetry_integral', 4),
    ('ZCR_pre',      3), ('ZCR_click',  3), ('ZCR_post',           3),
    ('kurtosis',     2), ('centroid_shift_hz', 0),
    ('tau_ms',       4), ('R2',         4), ('fit_coverage',       3),
    ('SPR',          2), ('R_spectral', 3), ('FPE_hz',             0),
]
_FEATURE_COLS = 3    # number of name/value pairs laid side by side


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

        self.theme_manager = theme_manager
        self.csv_path: Optional[Path] = None
        self.df = None                # pandas DataFrame, all columns kept as text
        self.visible_rows: list = []  # df row indices, in the order the table shows them
        self._pixmap = None           # the cropped screenshot (scaled on show/resize)
        self._png_index = None        # lower-cased name → path, built lazily
        self._click_names = None      # df row → 'click<sec>[_n].png', built lazily
        self.screenshots_dir: Optional[Path] = None   # None → look beside the CSV

        self.setWindowTitle("Click Review — label candidates")
        self.setMinimumSize(1100, 700)

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
        ])
        self.combo_filter.currentIndexChanged.connect(self._refresh_table)
        file_row.addWidget(self.combo_filter)

        file_row.addWidget(QLabel("Sort:"))
        self.combo_sort = WideComboBox()
        self.combo_sort.addItems([
            "P(click) — highest first",
            "P(click) — lowest first",
            "Frame order",
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
        # Label is content-sized, never stretched: it is the column being edited, so it
        # must always be fully visible. Verdict absorbs the slack instead — it holds the
        # longest and most variable text ('Stage4_dedup'), and it is the one that can be
        # clipped without costing the user anything.
        for c in (_COL_TIME, _COL_PROB, _COL_LABEL):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(_COL_VERDICT, QHeaderView.Stretch)
        hdr.setStretchLastSection(False)   # otherwise it overrides the mode above
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

        # ── Bottom bar ──
        bottom = QHBoxLayout()
        hint = QLabel(
            "Keys:  1 = click   0 = noise   Backspace = clear   "
            "Space / ↓ = next   ↑ = previous"
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
        value_font = QFont("Courier New")

        self._feature_values = {}   # feature name → its value QLabel

        n_rows = -(-len(_FEATURE_FMT) // _FEATURE_COLS)   # ceil
        for idx, (name, _prec) in enumerate(_FEATURE_FMT):
            r = idx % n_rows
            c = idx // n_rows

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
        for name, _prec in _FEATURE_FMT:
            label = self._feature_values[name]

            if i is None or self.df is None or name not in self.df.columns:
                label.setText("—")
                continue

            value = _to_float(self.df.at[i, name])
            if value != value:          # NaN → the column exists but the cell is empty
                label.setText("—")
                continue

            prec = dict(_FEATURE_FMT)[name]
            label.setText(f"{value:.{prec}f}")

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
        start = str(self.csv_path.parent) if self.csv_path else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Open candidates CSV", start, "Candidate CSV (*.csv)"
        )
        if path:
            self._load_csv(Path(path))

    def _on_choose_screenshots_dir(self):
        """Look for screenshots somewhere other than beside the CSV."""
        start = str(self._search_root()) if self.csv_path else str(Path.home())
        chosen = QFileDialog.getExistingDirectory(self, "Screenshots folder", start)
        if not chosen:
            return

        self.screenshots_dir = Path(chosen)
        self._png_index = None          # different tree → the cached index is stale
        self.btn_shots_dir.setToolTip(f"Screenshots folder: {chosen}")
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

        # ── Sort ──
        sort_mode = self.combo_sort.currentIndex()
        if sort_mode == 0:
            rows.sort(key=self._prob, reverse=True)
        elif sort_mode == 1:
            rows.sort(key=self._prob)
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
        """Where to look for screenshots: the chosen folder, else beside the CSV."""
        return self.screenshots_dir or self.csv_path.parent

    def _png_lookup(self):
        """
        Lazily index every PNG under the search root, by lower-cased filename.

        One recursive pass, reused for every candidate: rescanning per candidate would
        make stepping through a big export crawl. Sub-folders are included because the
        user files screenshots into them (screenshots/stage 2/…).
        """
        if self._png_index is None:
            self._png_index = {}
            for png in self._search_root().rglob('*.png'):
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
        root = self._search_root()

        standard = f"{stem}_{frame:06d}.png"
        direct = root / SCREENSHOTS_FOLDER / standard
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
                f"under {self._search_root()} (including sub-folders)"
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
            self.df.to_csv(self.csv_path, index=False)
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
