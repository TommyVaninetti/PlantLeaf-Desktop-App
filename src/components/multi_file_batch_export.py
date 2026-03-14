"""
Multi-File Batch Screenshot Export
====================================
Accessible from the Home window File menu.
Allows selecting multiple .paudio files and running the full headless
pipeline for each file sequentially:
    AudioLoadWorker  →  AudioDataManager  →  estimate_noise_offline
    →  BatchExportWorker  →  render_click_screenshot (main thread)

Output layout:
    {chosen_root}/
        {file1_stem}_screenshots/
            frame_006010_t15.4230s.png        ← isolated frame
            t=15.4230s/
                frame_006010_t15.4230s.png    ← group member
                frame_006011_t15.4256s.png
        {file2_stem}_screenshots/
            ...

Memory safety:
  • Files are processed strictly one at a time.
  • After each file the data dict and AudioDataManager are explicitly deleted
    and gc.collect() is called before loading the next file.
  • Even for 2 GB files (≈ 600 MB of FFT + phase arrays) peak resident memory
    stays bounded to one file at a time.

Threshold: μ + SIGMA_MULTIPLIER × σ  (per-file, computed from precompute_fft_means)
"""

import os
import gc
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from PySide6.QtCore import (
    QObject, QThread, Signal, Qt, QTimer
)
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QProgressBar, QListWidget, QListWidgetItem,
    QApplication, QMessageBox, QGroupBox, QSplitter, QWidget,
    QSizePolicy, QPlainTextEdit,
)
from PySide6.QtGui import QFont, QColor, QIcon

# ── local imports ─────────────────────────────────────────────────────────────
from saving.audio_load_progress import AudioLoadWorker
from windows.replay_window_audio import (
    AudioDataManager,
    estimate_noise_offline,
)
from components.batch_export_screenshots import (
    BatchExportWorker,
    render_click_screenshot,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SIGMA_MULTIPLIER = 5   # threshold = fft_mean + SIGMA_MULTIPLIER * fft_std


# ═══════════════════════════════════════════════════════════════════════════════
# STATUS ITEMS  — list widget row data
# ═══════════════════════════════════════════════════════════════════════════════
STATUS_WAITING   = "⏸ waiting"
STATUS_LOADING   = "⏳ loading…"
STATUS_DETECTING = "🔍 noise est.…"
STATUS_EXPORTING = "📸 exporting…"
STATUS_DONE      = "✅ done"
STATUS_ERROR     = "❌ error"
STATUS_SKIPPED   = "⏭ skipped"


class MultiFileBatchExportDialog(QDialog):
    """
    Modal dialog for multi-file batch screenshot export.

    UI layout:
    ┌─ File list ────────────────────────────────────────────────────────────┐
    │  ⏸ file1.paudio                                                        │
    │  ⏸ file2.paudio                                                        │
    │  …                                                                     │
    ├─ Output folder ────────────────────────────────────────────────────────┤
    │  [Browse…]  /path/to/output                                            │
    ├─ Overall progress ─────────────────────────────────────────────────────┤
    │  ████████░░  3 / 7 files                                               │
    ├─ Current file progress ────────────────────────────────────────────────┤
    │  ██████████  Processing frame 1234/5678                               │
    ├─ Log ──────────────────────────────────────────────────────────────────┤
    │  [scrollable plain text log]                                           │
    ├───────────────────────────────────────────────────────────────────────┤
    │              [Start]   [Cancel]                                        │
    └────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Screenshot Export — Multiple Files")
        self.setMinimumSize(700, 600)
        self.resize(820, 870)

        # Internal state
        self._file_paths: list[str] = []
        self._output_dir: str = ""
        self._current_file_idx: int = 0   # index into _file_paths
        self._running: bool = False
        self._cancelled: bool = False

        # Qt objects kept alive for the current file pipeline
        self._load_thread: QThread | None = None
        self._load_worker: AudioLoadWorker | None = None
        self._batch_worker: BatchExportWorker | None = None
        self._current_dm: AudioDataManager | None = None

        # Render-side state for the current file
        self._render_state: dict = {}

        # Accent color (copied from parent's theme_manager if available)
        self._accent_color = '#007bff'
        if parent and hasattr(parent, 'theme_manager'):
            try:
                self._accent_color = parent.theme_manager.get_accent_color()
            except Exception:
                pass

        self._build_ui()

        # Apply parent theme
        if parent and hasattr(parent, 'theme_manager'):
            try:
                theme = parent.theme_manager.load_saved_theme()
                parent.theme_manager.apply_theme(self, theme)
                if 'light' in theme.lower(): #correzione bug noto per tema lights
                    self.setStyleSheet("QDialog { background-color: white; }")

            except Exception:
                pass

        

    # ─────────────────────────────────────────────────────────────────────────
    # UI BUILD
    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)

        # Title and description
        title = QLabel("Batch Screenshot Export")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        root.addWidget(title)
        description = QLabel("Select the files you want to take screenshots from. ")
        description.setObjectName("descriptionLabel")
        description_threshold = QLabel("The threshold is set to μ + 5σ (computed from each file's FFT means/std).")
        description_threshold.setObjectName("descriptionThresholdLabel")
        #centra il testo di description e description_threshold
        description.setAlignment(Qt.AlignCenter)
        description_threshold.setAlignment(Qt.AlignCenter)
        root.addWidget(description)
        root.addWidget(description_threshold)

        # ── File list group ──────────────────────────────────────────────────
        grp_files = QGroupBox("Files to process")
        grp_files_lay = QVBoxLayout(grp_files)

        btn_row = QHBoxLayout()
        self._btn_add_files = QPushButton("Add files…")
        self._btn_add_files.clicked.connect(self._add_files)
        self._btn_clear = QPushButton("Clear list")
        self._btn_clear.clicked.connect(self._clear_files)
        btn_row.addWidget(self._btn_add_files)
        btn_row.addWidget(self._btn_clear)
        btn_row.addStretch()
        grp_files_lay.addLayout(btn_row)

        self._file_list = QListWidget()
        self._file_list.setMinimumHeight(120)
        grp_files_lay.addWidget(self._file_list)
        root.addWidget(grp_files)

        # ── Output folder ────────────────────────────────────────────────────
        grp_out = QGroupBox("Output root folder")
        grp_out_lay = QHBoxLayout(grp_out)
        self._lbl_output = QLabel("(not selected)")
        self._lbl_output.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._lbl_output.setWordWrap(True)
        btn_browse = QPushButton("Browse…")
        btn_browse.setFixedWidth(95)
        btn_browse.clicked.connect(self._browse_output)
        grp_out_lay.addWidget(self._lbl_output)
        grp_out_lay.addWidget(btn_browse)
        root.addWidget(grp_out)

        # ── Progress ─────────────────────────────────────────────────────────
        grp_prog = QGroupBox("Progress")
        grp_prog_lay = QVBoxLayout(grp_prog)

        self._lbl_overall = QLabel("Overall: 0 / 0 files")
        self._prog_overall = QProgressBar()
        self._prog_overall.setRange(0, 100)
        self._prog_overall.setValue(0)
        grp_prog_lay.addWidget(self._lbl_overall)
        grp_prog_lay.addWidget(self._prog_overall)

        self._lbl_current = QLabel("Current file: –")
        self._prog_current = QProgressBar()
        self._prog_current.setRange(0, 100)
        self._prog_current.setValue(0)
        grp_prog_lay.addWidget(self._lbl_current)
        grp_prog_lay.addWidget(self._prog_current)
        root.addWidget(grp_prog)

        # ── Log ──────────────────────────────────────────────────────────────
        grp_log = QGroupBox("Log")
        grp_log_lay = QVBoxLayout(grp_log)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)

        self._log.setMaximumBlockCount(2000)
        self._log.setMinimumHeight(100)
        #imposta il colore del testo dentro il plaintext come lo stesso del tema
    
        grp_log_lay.addWidget(self._log)
        root.addWidget(grp_log)

        # ── Buttons ──────────────────────────────────────────────────────────
        btn_row2 = QHBoxLayout()
        btn_row2.addStretch()
        self._btn_start = QPushButton("Start")
        self._btn_start.setMinimumWidth(100)
        self._btn_start.clicked.connect(self._start)
        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.setMinimumWidth(100)
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel)
        self._btn_close = QPushButton("Close")
        self._btn_close.setMinimumWidth(100)
        self._btn_close.clicked.connect(self.reject)
        btn_row2.addWidget(self._btn_start)
        btn_row2.addWidget(self._btn_cancel)
        btn_row2.addWidget(self._btn_close)
        root.addLayout(btn_row2)

    # ─────────────────────────────────────────────────────────────────────────
    # FILE / FOLDER MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────
    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select .paudio files", "",
            "PlantLeaf Audio (*.paudio);;All files (*.*)"
        )
        for p in paths:
            if p not in self._file_paths:
                self._file_paths.append(p)
                item = QListWidgetItem(f"{STATUS_WAITING}  {os.path.basename(p)}")
                item.setToolTip(p)
                self._file_list.addItem(item)
        self._log_msg(f"Added {len(paths)} file(s). Total: {len(self._file_paths)}")

    def _clear_files(self):
        if self._running:
            return
        self._file_paths.clear()
        self._file_list.clear()
        self._log_msg("File list cleared.")

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select Output Root Folder", ""
        )
        if d:
            self._output_dir = d
            self._lbl_output.setText(d)
            self._log_msg(f"Output folder: {d}")

    # ─────────────────────────────────────────────────────────────────────────
    # START / CANCEL
    # ─────────────────────────────────────────────────────────────────────────
    def _start(self):
        if not self._file_paths:
            QMessageBox.warning(self, "No files", "Please add at least one .paudio file.")
            return
        if not self._output_dir:
            QMessageBox.warning(self, "No output folder", "Please select an output folder.")
            return

        # Reset all items to 'waiting'
        for i in range(self._file_list.count()):
            item = self._file_list.item(i)
            fname = os.path.basename(self._file_paths[i])
            item.setText(f"{STATUS_WAITING}  {fname}")

        self._running = True
        self._cancelled = False
        self._current_file_idx = 0
        self._btn_start.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._btn_add_files.setEnabled(False)
        self._btn_clear.setEnabled(False)

        self._prog_overall.setRange(0, len(self._file_paths))
        self._prog_overall.setValue(0)
        self._lbl_overall.setText(f"Overall: 0 / {len(self._file_paths)} files")

        self._log_msg("=" * 60)
        self._log_msg(f"Starting batch export: {len(self._file_paths)} files → {self._output_dir}")
        self._log_msg("=" * 60)

        # Kick off the first file
        self._process_next_file()

    def _cancel(self):
        self._cancelled = True
        self._log_msg("⚠️  Cancel requested — stopping after current file…")

        # Stop batch worker if running
        if self._batch_worker is not None and self._batch_worker.isRunning():
            self._batch_worker.cancel()

        # Stop load worker if running
        if self._load_worker is not None:
            self._load_worker.cancel_load()

        self._btn_cancel.setEnabled(False)

    # ─────────────────────────────────────────────────────────────────────────
    # PER-FILE PIPELINE
    # ─────────────────────────────────────────────────────────────────────────
    def _process_next_file(self):
        """Entry point: load the file at _current_file_idx."""
        if self._cancelled or self._current_file_idx >= len(self._file_paths):
            self._finish_all()
            return

        idx = self._current_file_idx
        file_path = self._file_paths[idx]
        fname = os.path.basename(file_path)

        # Update list item status
        self._set_item_status(idx, STATUS_LOADING)

        self._prog_current.setValue(0)
        self._lbl_current.setText(f"[{idx+1}/{len(self._file_paths)}] Loading: {fname}")
        self._log_msg(f"\n▶ File {idx+1}/{len(self._file_paths)}: {fname}")

        # ── AudioLoadWorker on a QThread ────────────────────────────────────
        self._load_worker = AudioLoadWorker(file_path)
        self._load_thread = QThread(self)
        self._load_worker.moveToThread(self._load_thread)

        # Connect worker signals
        self._load_worker.progress.connect(self._on_load_progress, Qt.QueuedConnection)
        self._load_worker.finished.connect(self._on_file_loaded, Qt.QueuedConnection)
        self._load_worker.error.connect(self._on_load_error, Qt.QueuedConnection)

        # Start worker when thread starts
        self._load_thread.started.connect(self._load_worker.run)

        self._load_thread.start()

    def _on_load_progress(self, pct: int):
        self._prog_current.setValue(min(pct, 50))   # loading = 0–50%
        fname = os.path.basename(self._file_paths[self._current_file_idx])
        self._lbl_current.setText(
            f"[{self._current_file_idx+1}/{len(self._file_paths)}] Loading {fname}: {pct}%"
        )

    def _on_load_error(self, msg: str):
        idx = self._current_file_idx
        self._set_item_status(idx, STATUS_ERROR)
        self._log_msg(f"  ❌ Load error: {msg}")
        self._cleanup_load()
        self._advance_to_next_file()

    def _on_file_loaded(self, data_dict: dict):
        """Called in main thread after AudioLoadWorker.finished."""
        idx = self._current_file_idx
        fname = os.path.basename(self._file_paths[idx])

        self._log_msg(f"  ✅ Loaded: {data_dict.get('total_frames', '?')} frames, "
                      f"{data_dict.get('total_duration_sec', 0):.1f} s")

        # ── Cleanup load thread (no longer needed) ───────────────────────────
        self._cleanup_load()

        # ── Populate AudioDataManager ────────────────────────────────────────
        dm = AudioDataManager()
        dm.header_info          = data_dict['header_info']
        dm.fft_data             = data_dict['fft_data']
        dm.phase_data           = data_dict.get('phase_data', [])
        dm.frequency_axis       = np.array(data_dict['frequency_axis'])
        dm.total_frames         = data_dict['total_frames']
        dm.frame_duration_ms    = data_dict['frame_duration_ms']
        dm.total_duration_sec   = data_dict['total_duration_sec']
        dm.click_events         = data_dict['click_events']
        dm.overview_x           = np.array(data_dict['overview_x'])
        dm.overview_y           = np.array(data_dict['overview_y'])
        dm.overview_loaded      = True
        dm.streaming_x          = np.array(data_dict['streaming_x'])
        dm.streaming_y          = np.array(data_dict['streaming_y'])
        dm.streaming_start_time = data_dict['streaming_start_time']
        dm.streaming_end_time   = data_dict['streaming_end_time']

        # Explicit del to free data_dict memory (fft_data is now owned by dm)
        del data_dict
        gc.collect()

        # Precompute means/std (required by threshold + estimate_noise_offline)
        self._log_msg("  🔄 Precomputing FFT means…")
        dm.precompute_fft_means()

        # ── Threshold: μ + SIGMA_MULTIPLIER × σ ────────────────────────────
        threshold_v = dm.fft_mean + SIGMA_MULTIPLIER * dm.fft_std
        self._log_msg(
            f"  📊 Threshold (μ+{SIGMA_MULTIPLIER}σ): {threshold_v*1000:.3f} mV  "
            f"(μ={dm.fft_mean*1000:.3f}, σ={dm.fft_std*1000:.3f})"
        )

        # ── Noise estimation (offline, for C1–C5 in footer) ─────────────────
        self._set_item_status(idx, STATUS_DETECTING)
        self._lbl_current.setText(
            f"[{idx+1}/{len(self._file_paths)}] Noise estimation: {fname}"
        )
        self._prog_current.setValue(55)
        QApplication.processEvents()

        try:
            noise_result = estimate_noise_offline(dm, energy_threshold_multiplier=4.0,
                                                  max_samples=500)
            dm._cached_noise_rms = noise_result['noise_rms']
            self._log_msg(
                f"  🔍 Noise RMS: {noise_result['noise_rms']*1000:.4f} mV  "
                f"({noise_result['n_samples']} samples)"
            )
        except Exception as e:
            dm._cached_noise_rms = None
            self._log_msg(f"  ⚠️  Noise estimation failed ({e}); C1–C5 will be N/A")

        self._prog_current.setValue(60)

        # Keep dm alive during export
        self._current_dm = dm

        # ── Build output folder for this file ───────────────────────────────
        stem = os.path.splitext(fname)[0]
        root_dir = os.path.join(self._output_dir, f"{stem}_screenshots")
        os.makedirs(root_dir, exist_ok=True)
        self._log_msg(f"  📁 Output: {root_dir}")

        # ── Launch BatchExportWorker ─────────────────────────────────────────
        self._set_item_status(idx, STATUS_EXPORTING)
        self._lbl_current.setText(
            f"[{idx+1}/{len(self._file_paths)}] Exporting: {fname}"
        )

        self._render_state = {
            'n_saved': 0,
            'n_failed': 0,
            'done': False,
            'n_emitted': 0,
            'n_rendered': 0,
            'root_dir': root_dir,
        }

        self._batch_worker = BatchExportWorker(
            data_manager=dm,
            threshold_v=threshold_v,
            use_normalization=True,
            parent=None,   # no Qt parent: we manage lifetime manually
        )

        self._batch_worker.frame_data_ready.connect(
            self._on_frame_ready, Qt.QueuedConnection
        )
        self._batch_worker.progress_updated.connect(
            self._on_batch_progress, Qt.QueuedConnection
        )
        self._batch_worker.finished_export.connect(
            self._on_batch_finished, Qt.QueuedConnection
        )
        self._batch_worker.error_occurred.connect(
            self._on_batch_error, Qt.QueuedConnection
        )

        self._batch_worker.start()

    # ── Batch worker callbacks ────────────────────────────────────────────────
    def _on_frame_ready(self, frame_data: dict):
        state    = self._render_state
        root_dir = state['root_dir']

        frame_idx      = frame_data['frame_idx']
        ts             = frame_data['timestamp']
        group_size     = frame_data.get('group_size', 1)
        group_first_ts = frame_data.get('group_first_ts', ts)

        fname = f"frame_{frame_idx:06d}_t{ts:.4f}s.png"
        if group_size > 1:
            subdir = os.path.join(root_dir, f"t={group_first_ts:.4f}s")
            os.makedirs(subdir, exist_ok=True)
            out_path = os.path.join(subdir, fname)
        else:
            out_path = os.path.join(root_dir, fname)

        try:
            ok = render_click_screenshot(frame_data, out_path, self._accent_color)
            if ok:
                state['n_saved'] += 1
            else:
                state['n_failed'] += 1
        except Exception as e:
            self._log_msg(f"  ⚠️  Render error frame {frame_idx}: {e}")
            state['n_failed'] += 1

        state['n_rendered'] += 1

        # If worker already finished and all renders are done → advance
        if state['done'] and state['n_rendered'] >= state['n_emitted']:
            self._finalize_current_file()

    def _on_batch_progress(self, pct: int, msg: str):
        # Batch worker uses 0–100; map to 60–100 of the current-file bar
        mapped = 60 + int(pct * 0.40)
        self._prog_current.setValue(min(mapped, 100))
        self._lbl_current.setText(
            f"[{self._current_file_idx+1}/{len(self._file_paths)}] {msg}"
        )

    def _on_batch_finished(self, n_exported: int):
        state = self._render_state
        state['done'] = True
        state['n_emitted'] = n_exported
        # If all renders already completed → advance immediately
        if state['n_rendered'] >= state['n_emitted']:
            self._finalize_current_file()

    def _on_batch_error(self, msg: str):
        idx = self._current_file_idx
        self._set_item_status(idx, STATUS_ERROR)
        self._log_msg(f"  ❌ Batch worker error: {msg}")
        self._cleanup_batch()
        self._advance_to_next_file()

    # ─────────────────────────────────────────────────────────────────────────
    # FINALIZE CURRENT FILE
    # ─────────────────────────────────────────────────────────────────────────
    def _finalize_current_file(self):
        idx   = self._current_file_idx
        state = self._render_state

        self._set_item_status(idx, STATUS_DONE)
        self._log_msg(
            f"  ✅ Done: {state['n_saved']} saved, {state['n_failed']} failed → {state['root_dir']}"
        )
        self._prog_current.setValue(100)

        self._cleanup_batch()
        self._advance_to_next_file()

    def _advance_to_next_file(self):
        """Release current file's memory and move to the next file."""
        # ── Explicit memory release ──────────────────────────────────────────
        if self._current_dm is not None:
            # Drop the large fft_data / phase_data lists
            self._current_dm.fft_data  = []
            self._current_dm.phase_data = []
            self._current_dm = None
        gc.collect()

        self._current_file_idx += 1
        n_done = self._current_file_idx
        n_total = len(self._file_paths)

        self._prog_overall.setValue(n_done)
        self._lbl_overall.setText(f"Overall: {n_done} / {n_total} files")

        if self._cancelled:
            # Mark remaining files as skipped
            for i in range(n_done, n_total):
                self._set_item_status(i, STATUS_SKIPPED)
            self._finish_all()
            return

        # Process next file on the next event-loop tick so the UI can breathe
        QTimer.singleShot(0, self._process_next_file)

    # ─────────────────────────────────────────────────────────────────────────
    # FINISH ALL
    # ─────────────────────────────────────────────────────────────────────────
    def _finish_all(self):
        self._running = False
        self._btn_start.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._btn_add_files.setEnabled(True)
        self._btn_clear.setEnabled(True)

        n_done   = self._current_file_idx
        n_ok     = sum(
            1 for i in range(self._file_list.count())
            if STATUS_DONE in (self._file_list.item(i).text() or "")
        )
        n_errors = sum(
            1 for i in range(self._file_list.count())
            if STATUS_ERROR in (self._file_list.item(i).text() or "")
        )

        if self._cancelled:
            msg = (f"Batch export cancelled.\n\n"
                   f"Completed: {n_ok} files\n"
                   f"Errors: {n_errors}\n"
                   f"Skipped: {len(self._file_paths) - n_done}\n\n"
                   f"Output: {self._output_dir}")
        else:
            msg = (f"Batch export complete!\n\n"
                   f"Files processed: {n_ok}\n"
                   f"Errors: {n_errors}\n\n"
                   f"Output: {self._output_dir}")

        self._log_msg("\n" + "=" * 60)
        self._log_msg(msg.replace("\n", "  "))

        QMessageBox.information(self, "Batch Export Complete", msg)

    # ─────────────────────────────────────────────────────────────────────────
    # CLEANUP HELPERS
    # ─────────────────────────────────────────────────────────────────────────
    def _cleanup_load(self):
        """Stop and tear down the load thread/worker."""
        if self._load_thread is not None:
            try:
                if self._load_thread.isRunning():
                    self._load_thread.quit()
                    self._load_thread.wait(5000)
            except Exception:
                pass
            self._load_thread = None
        self._load_worker = None

    def _cleanup_batch(self):
        """Stop and tear down the batch worker thread."""
        if self._batch_worker is not None:
            try:
                if self._batch_worker.isRunning():
                    self._batch_worker.cancel()
                    self._batch_worker.wait(10000)
            except Exception:
                pass
            self._batch_worker = None

    # ─────────────────────────────────────────────────────────────────────────
    # UI HELPERS
    # ─────────────────────────────────────────────────────────────────────────
    def _set_item_status(self, idx: int, status: str):
        if idx < self._file_list.count():
            fname = os.path.basename(self._file_paths[idx])
            item  = self._file_list.item(idx)
            item.setText(f"{status}  {fname}")

    def _log_msg(self, text: str):
        self._log.appendPlainText(text)
        # Scroll to bottom
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())
        QApplication.processEvents()

    # ─────────────────────────────────────────────────────────────────────────
    # CLOSE / REJECT GUARD
    # ─────────────────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        if self._running:
            result = QMessageBox.question(
                self, "Export in progress",
                "An export is in progress. Cancel and close?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if result != QMessageBox.Yes:
                event.ignore()
                return
            self._cancel()
        # Give workers a moment to stop
        self._cleanup_load()
        self._cleanup_batch()
        super().closeEvent(event)


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def launch_multi_file_batch_export(parent_window):
    """
    Called from the Home window File menu (or any window).
    Opens the MultiFileBatchExportDialog.
    """
    dlg = MultiFileBatchExportDialog(parent=parent_window)
    dlg.exec()
