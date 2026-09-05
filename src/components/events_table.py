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
EVENTS TABLE — one row per event transmitted by the firmware.

Replaces the old 5-column DataTable in the audio main window. The firmware being
written now sends only the frames that survive Stage 1 (and possibly some Stage 2
hard gates) instead of every frame, so a row here is an EVENT, not a threshold
crossing, and it carries the whole v6 feature vector that was computed for it.

⚠️ THE COLUMN SET IS NOT DEFINED HERE. It is `CSV_COLUMNS` from
data_collection_dialog_v5 — the 51-column v6 export schema — imported rather than
copied, so this table, the offline export and the review dialog cannot drift apart.
The only thing this module decides is the ORDER (the five interactive columns are
hoisted to the front) and the formatting.
"""

import math

from PySide6.QtWidgets import (QTableWidget, QTableWidgetItem, QLineEdit, QComboBox,
                               QHeaderView, QStyledItemDelegate, QMenu)
from PySide6.QtCore import Qt, Signal

# Module-level despite the import graph: click_review_dialog already imports this
# same module at module level (with the same noqa), and data_collection_dialog_v5
# imports nothing from components, so there is no cycle to fall into.
from components.data_collection_dialog_v5 import (          # noqa: E402
    CSV_COLUMNS,
    FEATURE_NAMES,          # the 17 v5 features
    FEATURE_NAMES_V6,       # the 10 v6 additions (incl. local_crest, harmonic_confinement)
    QUALITY_COLUMNS,        # fit_valid, decay_len, n_seg, b3_frames, gibbs_fired
    STAGE1_COLUMNS,         # run_id, run_length, run_crest, pos_in_run, would_pass_v5
    HARMONIC_COLUMNS,       # hc_f1_hz, hc_r_A, hc_r_B
)

# The 0/1/2 label vocabulary. IMPORTED, never re-declared: click_review_dialog is
# what writes labels into the CSVs that train the SVM, and a table that disagreed
# with it about what '2' means would silently corrupt a training set.
from components.click_review_dialog import (                # noqa: E402
    LABEL_CLICK, LABEL_NOISE, LABEL_AMBIG, LABEL_NONE, LABELS_DECIDED,
)


# ── COLUMN LAYOUT ───────────────────────────────────────────────────────────

#: The five columns the user actually interacts with, hoisted to the front.
#: Their schema keys, in display order.
CORE_KEYS = ['timestamp_s', 'stage_blocked', 'svm_probability', 'label', 'note']

CORE_HEADERS = {
    'timestamp_s':     'Timestamp',
    'stage_blocked':   'Verdict',
    'svm_probability': 'P(click)',
    'label':           'Label',
    'note':            'Notes',
}

COL_TIMESTAMP = 0
COL_VERDICT   = 1
COL_PROB      = 2
COL_LABEL     = 3
COL_NOTE      = 4
N_CORE        = len(CORE_KEYS)

#: Display order: the five interactive columns, then everything else in the exact
#: order CSV_COLUMNS declares it, so reading the table left-to-right past column 4
#: is reading the export schema.
COLUMN_KEYS = CORE_KEYS + [c for c in CSV_COLUMNS if c not in CORE_KEYS]

#: Everything in CSV_COLUMNS that no named sub-list claims: identity, provenance,
#: the noise state at detection, and the SVM's own prediction.
_GROUPED = set(CORE_KEYS) | set(FEATURE_NAMES) | set(FEATURE_NAMES_V6) \
           | set(QUALITY_COLUMNS) | set(STAGE1_COLUMNS) | set(HARMONIC_COLUMNS)
PROVENANCE_COLUMNS = [c for c in CSV_COLUMNS if c not in _GROUPED]

#: Header right-click menu: label -> the schema keys it shows/hides. Core is not
#: here on purpose — those five are never hideable.
COLUMN_GROUPS = [
    ('v5 features',            FEATURE_NAMES),
    ('v6 features',            FEATURE_NAMES_V6),
    ('Harmonic confinement',   HARMONIC_COLUMNS),
    ('Quality flags',          QUALITY_COLUMNS),
    ('Stage 1 diagnostics',    STAGE1_COLUMNS),
    ('Provenance & verdicts',  PROVENANCE_COLUMNS),
]

#: Visible on a fresh install, beside the five core columns. One representative
#: from each stage of the pipeline rather than a whole group, so the default view
#: fits on screen and still says something about every stage.
HEADLINE_KEYS = [
    'frame_idx',
    'peak_SNR', 'tau_ms', 'R2', 'FPE_hz',
    'n_seg',
    'spectral_entropy', 'harmonic_confinement', 'local_crest',
]

#: settings_manager key for the persisted visible-column set.
SETTINGS_KEY = 'audio_events_columns'

#: Schema columns that are integers. Formatting them as %.4g would print
#: `frame_idx = 1.234e+04`, which is unreadable and un-greppable.
_INT_KEYS = {
    'frame_idx', 'peak_abs', 'decay_len', 'n_seg', 'b3_frames', 'gibbs_fired',
    'fit_valid', 'run_id', 'run_length', 'pos_in_run', 'would_pass_v5',
    'svm_prediction',
}

_NOTE_MAX_CHARS = 120   # the CSV `note` column is free text; the old 20-char cap
                        # belonged to the legacy CLCK block, not to the schema.


def _is_blank(value) -> bool:
    """True for the three ways this schema spells 'no value': None, '' and NaN."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ''
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def format_cell(key: str, value) -> str:
    """Render one schema value for display. Never raises: an unexpected type
    falls through to str(), because a malformed event must show up as a wrong
    looking row, not as a crash in the GUI thread."""
    if _is_blank(value):
        # An empty stage_blocked is not missing data — it is the verdict.
        if key == 'stage_blocked':
            return 'CLICK'
        # "not classified" is worth seeing in the P(click) column; a blank cell
        # reads as a zero-width feature value everywhere else.
        if key == 'svm_probability':
            return '—'
        return ''

    if key == 'timestamp_s':
        try:
            return f"{float(value):.2f} s"
        except (TypeError, ValueError):
            return str(value)
    if key == 'svm_probability':
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return str(value)
    if key in _INT_KEYS:
        try:
            return str(int(float(value)))
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, str):
        return value
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


# ── DELEGATES ───────────────────────────────────────────────────────────────

class EventEditDelegate(QStyledItemDelegate):
    """Editors for the only two editable columns: Label (0/1/2) and Notes."""

    def createEditor(self, parent, option, index):
        if index.column() == COL_LABEL:
            editor = QComboBox(parent)
            editor.addItems([LABEL_NONE, LABEL_NOISE, LABEL_CLICK, LABEL_AMBIG])
            editor.setToolTip("0 = noise   1 = click   2 = ambiguous   empty = not yet judged")
            return editor
        if index.column() == COL_NOTE:
            editor = QLineEdit(parent)
            editor.setMaxLength(_NOTE_MAX_CHARS)
            editor.setStyleSheet("""
                QLineEdit {
                    padding: 6px;
                    font-size: 14px;
                    border: 2px solid #4CAF50;
                    border-radius: 4px;
                    background-color: white;
                    color: black;
                }
            """)
            return editor
        return super().createEditor(parent, option, index)

    def setEditorData(self, editor, index):
        text = index.model().data(index, Qt.ItemDataRole.EditRole) or ""
        if index.column() == COL_LABEL:
            pos = editor.findText(text)
            editor.setCurrentIndex(pos if pos >= 0 else 0)
            return
        if index.column() == COL_NOTE:
            editor.setText(text)
            return
        super().setEditorData(editor, index)

    def setModelData(self, editor, model, index):
        if index.column() == COL_LABEL:
            model.setData(index, editor.currentText(), Qt.ItemDataRole.EditRole)
            return
        if index.column() == COL_NOTE:
            model.setData(index, editor.text()[:_NOTE_MAX_CHARS], Qt.ItemDataRole.EditRole)
            return
        super().setModelData(editor, model, index)

    def updateEditorGeometry(self, editor, option, index):
        if index.column() in (COL_LABEL, COL_NOTE):
            rect = option.rect
            rect.setHeight(max(rect.height(), 34))
            if index.column() == COL_NOTE:
                rect.setWidth(max(rect.width(), 260))
            editor.setGeometry(rect)
            return
        super().updateEditorGeometry(editor, option, index)


# ── THE TABLE ───────────────────────────────────────────────────────────────

class EventsTable(QTableWidget):
    """
    One row per transmitted event, carrying the full v6 schema.

    The raw (unformatted) event dict is kept on column 0 of each row under
    Qt.UserRole, so consumers get the numbers back rather than the strings —
    and so the arrays the future event-based reader will attach
    (`fft_mags`, `phases`) survive the round trip untouched.
    """

    #: Row index of the newly selected event, or -1 when the selection is cleared.
    eventSelected = Signal(int)
    #: (row, label) whenever the manual label changes. '' means "cleared".
    labelChanged = Signal(int, str)

    def __init__(self, theme_manager, parent=None, settings_manager=None):
        super().__init__(parent)
        self.theme_manager = theme_manager
        self.settings_manager = settings_manager
        self._events = []            # raw dicts, parallel to the rows
        self._visible_keys = set()
        self._suppress_label_signal = False
        self.setup_table()

    # ── construction ────────────────────────────────────────────────────────

    def setup_table(self):
        self.setColumnCount(len(COLUMN_KEYS))
        self.setHorizontalHeaderLabels(
            [CORE_HEADERS.get(k, k) for k in COLUMN_KEYS]
        )

        self.event_delegate = EventEditDelegate()
        self.setItemDelegate(self.event_delegate)

        header = self.horizontalHeader()
        header.setStretchLastSection(False)
        for col in range(len(COLUMN_KEYS)):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_NOTE, QHeaderView.ResizeMode.Interactive)
        self.setColumnWidth(COL_NOTE, 180)

        # Column visibility is chosen from the header, not from a settings dialog:
        # the 51 columns are the point of the table, and hunting for them in a
        # menu bar would defeat it.
        header.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        header.customContextMenuRequested.connect(self._show_header_menu)

        self.verticalHeader().setDefaultSectionSize(25)
        self.verticalHeader().setVisible(False)

        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.setAlternatingRowColors(True)
        # Deliberately off: events arrive in time order and that order is the
        # only one in which the iFFT sequence makes sense to step through.
        self.setSortingEnabled(False)

        self.setToolTip(
            "Click a row to show that event in the iFFT and FFT plots.\n"
            "Label: 0 = noise, 1 = click, 2 = ambiguous (keys 0/1/2, Del to clear).\n"
            "Right-click the header to show or hide feature columns."
        )

        self.itemSelectionChanged.connect(self._on_selection_changed)
        self.itemChanged.connect(self._on_item_changed)

        self._restore_visible_keys()

    # ── data ────────────────────────────────────────────────────────────────

    def add_event(self, event: dict) -> int:
        """
        Append one event. `event` is keyed by CSV_COLUMNS names; anything absent
        renders empty, so a partially populated event (all this window can build
        until the new firmware lands) is a valid argument.

        Extra keys are kept, not dropped — that is how `fft_mags` / `phases` and
        the legacy `peak_amplitude_v` ride along without becoming columns.
        """
        row = self.rowCount()
        self._suppress_label_signal = True
        try:
            self.insertRow(row)
            self._events.append(dict(event))
            self._fill_row(row, event)
        finally:
            self._suppress_label_signal = False

        # Follow the stream only while the user is not inspecting something:
        # auto-scrolling out from under a selected row makes the table unusable
        # during an acquisition, which is exactly when events arrive.
        if self.currentRow() < 0:
            self.scrollToBottom()
        return row

    def set_events(self, events):
        self.clear_events()
        for ev in events:
            self.add_event(ev)

    def clear_events(self):
        self._suppress_label_signal = True
        try:
            self.setRowCount(0)
            self._events = []
        finally:
            self._suppress_label_signal = False

    def event_at(self, row):
        if 0 <= row < len(self._events):
            return self._events[row]
        return None

    def current_event(self):
        return self.event_at(self.currentRow())

    def _fill_row(self, row, event: dict):
        for col, key in enumerate(COLUMN_KEYS):
            item = QTableWidgetItem(format_cell(key, event.get(key)))
            if col in (COL_LABEL, COL_NOTE):
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            else:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if col == COL_PROB or col in _NUMERIC_ALIGNED:
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                      | Qt.AlignmentFlag.AlignVCenter)
            self.setItem(row, col, item)
        # The raw dict lives on column 0 so a consumer never has to parse the
        # formatted strings back into numbers.
        self.item(row, 0).setData(Qt.ItemDataRole.UserRole, event)

    # ── interaction ─────────────────────────────────────────────────────────

    def _on_selection_changed(self):
        self.eventSelected.emit(self.currentRow())

    def _on_item_changed(self, item):
        if self._suppress_label_signal or item is None:
            return
        row = item.row()
        if not (0 <= row < len(self._events)):
            return
        if item.column() == COL_LABEL:
            value = item.text().strip()
            if value not in LABELS_DECIDED:
                value = LABEL_NONE
                self._set_label_text(row, value)
            self._events[row]['label'] = value
            self.labelChanged.emit(row, value)
        elif item.column() == COL_NOTE:
            self._events[row]['note'] = item.text()[:_NOTE_MAX_CHARS]

    def _set_label_text(self, row, value):
        was = self._suppress_label_signal
        self._suppress_label_signal = True
        try:
            self.item(row, COL_LABEL).setText(value)
        finally:
            self._suppress_label_signal = was

    def keyPressEvent(self, event):
        """0 / 1 / 2 label the selected row, Del/Backspace clears it — the same
        keys as click_review_dialog, so the labelling reflex transfers. Only
        reached when no cell editor is open, so it never eats note typing."""
        row = self.currentRow()
        if 0 <= row < len(self._events):
            text = event.text()
            if text in LABELS_DECIDED:
                self._set_label_text(row, text)
                self._events[row]['label'] = text
                self.labelChanged.emit(row, text)
                event.accept()
                return
            if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
                self._set_label_text(row, LABEL_NONE)
                self._events[row]['label'] = LABEL_NONE
                self.labelChanged.emit(row, LABEL_NONE)
                event.accept()
                return
        super().keyPressEvent(event)

    # ── column visibility ───────────────────────────────────────────────────

    def _restore_visible_keys(self):
        saved = None
        if self.settings_manager is not None:
            try:
                saved = self.settings_manager.get_value(SETTINGS_KEY, None)
            except Exception as e:
                print(f"⚠️ Impossibile leggere {SETTINGS_KEY}: {e}")
        if isinstance(saved, (list, tuple)) and saved:
            # Intersect with the current schema: a saved set from an older
            # schema must not resurrect columns that no longer exist.
            self._visible_keys = {k for k in saved if k in COLUMN_KEYS}
        else:
            self._visible_keys = set(HEADLINE_KEYS)
        self._apply_visibility()

    def _persist_visible_keys(self):
        if self.settings_manager is None:
            return
        try:
            self.settings_manager.set_value(SETTINGS_KEY, sorted(self._visible_keys))
        except Exception as e:
            print(f"⚠️ Impossibile salvare {SETTINGS_KEY}: {e}")

    def _apply_visibility(self):
        for col, key in enumerate(COLUMN_KEYS):
            if col < N_CORE:
                self.setColumnHidden(col, False)     # core is never hideable
            else:
                self.setColumnHidden(col, key not in self._visible_keys)

    def set_visible_keys(self, keys):
        self._visible_keys = {k for k in keys if k in COLUMN_KEYS}
        self._apply_visibility()
        self._persist_visible_keys()

    def _show_header_menu(self, pos):
        menu = QMenu(self)

        all_optional = [k for k in COLUMN_KEYS[N_CORE:]]
        menu.addAction("Show all columns",
                       lambda: self.set_visible_keys(all_optional))
        menu.addAction("Headline features only",
                       lambda: self.set_visible_keys(HEADLINE_KEYS))
        menu.addAction("Hide all feature columns",
                       lambda: self.set_visible_keys([]))
        menu.addSeparator()

        for title, keys in COLUMN_GROUPS:
            action = menu.addAction(title)
            action.setCheckable(True)
            shown = [k for k in keys if k in self._visible_keys]
            action.setChecked(len(shown) == len(keys))
            action.triggered.connect(
                lambda checked, ks=tuple(keys): self._toggle_group(ks, checked)
            )

        menu.addSeparator()
        single = menu.addMenu("Individual columns")
        for title, keys in COLUMN_GROUPS:
            sub = single.addMenu(title)
            for key in keys:
                a = sub.addAction(key)
                a.setCheckable(True)
                a.setChecked(key in self._visible_keys)
                a.triggered.connect(
                    lambda checked, k=key: self._toggle_key(k, checked)
                )

        menu.exec(self.horizontalHeader().mapToGlobal(pos))

    def _toggle_group(self, keys, checked):
        if checked:
            self._visible_keys |= set(keys)
        else:
            self._visible_keys -= set(keys)
        self._apply_visibility()
        self._persist_visible_keys()

    def _toggle_key(self, key, checked):
        if checked:
            self._visible_keys.add(key)
        else:
            self._visible_keys.discard(key)
        self._apply_visibility()
        self._persist_visible_keys()

    # ── export ──────────────────────────────────────────────────────────────

    def export_rows(self):
        """Every row as a full CSV_COLUMNS dict, ready for
        `csv.DictWriter(f, fieldnames=CSV_COLUMNS)` — the same call the offline
        exporter and the replay window make, so the files are interchangeable."""
        rows = []
        for event in self._events:
            rows.append({key: event.get(key, '') for key in CSV_COLUMNS})
        return rows

    def export_click_data(self):
        """
        ⚠️ LEGACY CONTRACT — DO NOT CHANGE THESE FIVE KEYS.

        The dicts returned here are JSON-compressed into the `CLCK` block of the
        .paudio file by MainWindowAudio.save_click_data, and read back by
        saving/audio_load_progress.py, core/audio_trim_export.py and
        windows/main_window_chemical_simulator.py. Adding keys is safe; renaming
        or removing timestamp / frequency / amplitude / duration_us / notes
        breaks every recording ever saved.
        """
        click_data = []
        for event in self._events:
            def _f(key, default=0.0):
                value = event.get(key)
                if _is_blank(value):
                    return default
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return default

            click_data.append({
                "timestamp":   _f('timestamp_s'),
                "frequency":   _f('FPE_hz'),
                # Not a schema column: the peak amplitude in volts has no place in
                # a dimensionless feature set, so the producer attaches it as an
                # extra key purely to keep this legacy field populated.
                "amplitude":   _f('peak_amplitude_v'),
                # One event = one frame = 2560 µs, unless the producer says otherwise.
                "duration_us": int(_f('duration_us', 2560)),
                "notes":       str(event.get('note', ''))[:20],
            })
        return click_data


#: Columns rendered right-aligned because they are numbers, resolved once at
#: import rather than per cell. Defined after the helpers it depends on.
_NUMERIC_ALIGNED = {
    col for col, key in enumerate(COLUMN_KEYS)
    if key not in ('note', 'stage_blocked', 'label', 'file', 'session_id',
                   'schema_version', 'stage2_mode', 'stage1_params')
}
