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
"FFT of Region" dialog: the spectrum of a user-selectable portion of a signal.

DELIBERATELY GENERIC. It receives `(time_s, signal, fs)` and nothing else that is
domain-specific: it does not know about data_manager, does not import
click_pipeline_v5, and has no concept of a frame. All the maths lives in
core/spectral_analysis.py.

As written it is already reusable for voltage: pass the concatenated samples,
self.sampling_rate, and band=None.

Layout:

    ┌─ time trace ────────────────────────────────────┐
    │   ~~~~/\/\------~~~~~~   [draggable region]     │
    ├─ spectrum of the selection ─────────────────────┤
    │        /\_/\___                                 │
    ├─────────────────────────────────────────────────┤
    │ presets | start/end | window | n_fft | dB       │
    │ Δf = … · N = … samples · n_fft = …              │
    └─────────────────────────────────────────────────┘
"""

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                               QHBoxLayout, QLabel, QListView, QPushButton,
                               QSizePolicy, QSplitter, QStyle, QVBoxLayout,
                               QWidget)

from core.spectral_analysis import (DEFAULT_ALPHA, DEFAULT_WINDOW,
                                    MIN_SEGMENT_SAMPLES, WINDOWS,
                                    band_descriptors, compute_spectrum,
                                    default_nfft, to_db)
from plotting.plot_manager import BasePlotWidget

#: Transform lengths offered in the UI.
#:   None → auto (default_nfft: enough padding for a smooth curve)
#:   NATIVE → no padding at all, n_fft = n_seg
NATIVE = -1
NFFT_CHOICES = [(None, "Auto"), (NATIVE, "Native (no padding)"),
                (512, "512"), (1024, "1024"), (2048, "2048"),
                (4096, "4096"), (8192, "8192")]

_DEBOUNCE_MS = 30


# Moved to its own module so every dialog with a combo box gets the same fix.
from components.wide_combo_box import WideComboBox  # noqa: E402,F401


def _fmt_hz(hz):
    if hz >= 1000:
        return f"{hz / 1000:.2f} kHz"
    return f"{hz:.0f} Hz"


def _fmt_unit(value, unit):
    """Auto-scale a value carrying a base SI unit of volts."""
    if unit == 'V':
        v = abs(value)
        if v >= 0.5:
            return f"{value:.3f} V"
        if v >= 5e-4:
            return f"{value * 1e3:.3f} mV"
        return f"{value * 1e6:.2f} µV"
    return f"{value:.4g} {unit}".strip()


class RegionFFTDialog(QDialog):
    """
    Spectrum of a user-selectable region of a signal.

    Parameters
    ----------
    time_s, signal : the trace. `time_s` is absolute time in seconds.
    fs : sampling rate [Hz].
    theme_manager : passed explicitly — a grandchild dialog cannot reliably reach
        the main window's theme_manager through its parent chain.
    unit_y : base unit of `signal` ('V' enables mV/µV auto-scaling).
    band : (lo, hi) Hz — analysis band for the descriptors and the default view.
        None → whole spectrum (this is what voltage would pass).
    initial_region : (t0, t1) in seconds. None → whole trace.
    markers : {label: t_seconds} → labelled vertical lines on the time plot.
    seams : [t_seconds] → dashed lines where independently reconstructed frames
        were stitched together (be honest about the discontinuity).
    presets : [(label, (t0, t1)), ...] → one-click region presets.
    reference : (freqs, mags, label) → an extra curve on the spectrum plot,
        e.g. the frame's transmitted FFT.
    banner : a note shown above the plots (e.g. "no fitted region available").
    """

    def __init__(self, time_s, signal, fs, *,
                 parent=None,
                 theme_manager=None,
                 unit_y='V',
                 title='Region FFT',
                 band=(20e3, 80e3),
                 initial_region=None,
                 markers=None,
                 seams=None,
                 presets=None,
                 reference=None,
                 banner=None):
        super().__init__(parent)

        self.time_s = np.asarray(time_s, dtype=np.float64)
        self.signal = np.asarray(signal, dtype=np.float64)
        self.fs = float(fs)
        self.theme_manager = theme_manager
        self.unit_y = unit_y
        self.band = band
        self.reference = reference
        self._spec = None

        self.setWindowTitle(title)
        self.setMinimumSize(900, 640)

        root = QVBoxLayout(self)

        if banner:
            self.banner = QLabel(banner)
            self.banner.setWordWrap(True)
            self.banner.setStyleSheet(
                "color: #FFA726; font-weight: bold; font-size: 10pt; padding: 4px;")
            root.addWidget(self.banner)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self._build_time_plot(markers, seams))
        splitter.addWidget(self._build_spectrum_plot())
        splitter.setSizes([260, 380])
        root.addWidget(splitter, 1)

        root.addWidget(self._build_controls(presets))

        self.readout = QLabel()
        self.readout.setWordWrap(True)
        self.readout.setStyleSheet("font-family: monospace; font-size: 10pt;")
        root.addWidget(self.readout)

        # Debounce so dragging the region stays smooth.
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(_DEBOUNCE_MS)
        self._timer.timeout.connect(self._recompute)

        # Initial region
        if initial_region is None:
            initial_region = (float(self.time_s[0]), float(self.time_s[-1])) \
                if len(self.time_s) else (0.0, 1.0)
        self.region.setRegion(list(initial_region))
        self._on_region_changed()          # clamps + syncs the spin boxes
        self._recompute()

        self._apply_theme()
        self._center_on_parent(parent)

    # ── construction ────────────────────────────────────────────────────────

    def _build_time_plot(self, markers, seams):
        y = self.signal
        peak = float(np.max(np.abs(y))) if len(y) else 1.0
        if not np.isfinite(peak) or peak <= 0:
            peak = 1.0
        x0 = float(self.time_s[0]) if len(self.time_s) else 0.0
        x1 = float(self.time_s[-1]) if len(self.time_s) else 1.0

        self.time_plot = BasePlotWidget(
            x_label="Time", y_label="Amplitude",
            x_range=(x0, x1), y_range=(-peak * 1.2, peak * 1.2),
            x_min=x0, x_max=x1, y_min=-peak * 5, y_max=peak * 5,
            unit_x="s", unit_y=self.unit_y, parent=self,
        )
        pw = self.time_plot.plot_widget
        pw.showGrid(x=True, y=True)
        self.trace = pw.plot(self.time_s, y, pen={'width': 1.4}, name='Signal')

        # Seams first, so they sit under everything else.
        for t in (seams or []):
            pw.addLine(x=float(t), pen=pg.mkPen('#777', width=1,
                                                style=Qt.DashDotLine))

        # Marker lines carry NO text: click start / peak / decay start / decay end
        # land within a few samples of each other, so their labels overlapped into
        # an unreadable smear. The lines alone are legible, and the exact positions
        # are one preset click away.
        for _label, t in (markers or {}).items():
            pw.addLine(x=float(t),
                       pen=pg.mkPen('#FFD600', width=1.2, style=Qt.DotLine))

        self.region = pg.LinearRegionItem(movable=True, brush=(80, 140, 255, 55))
        self.region.setZValue(10)
        pw.addItem(self.region)
        self.region.sigRegionChanged.connect(self._on_region_changed)
        return self.time_plot

    def _build_spectrum_plot(self):
        # Clamp the x-axis to the analysis band (20–80 kHz for audio). Outside it
        # the transmitted spectrum is zero by construction, so there is nothing to
        # see, and an unclamped axis only invites misreading the taper skirts.
        # band=None (voltage) → the whole spectrum.
        lo, hi = self.band if self.band else (0.0, self.fs / 2)

        self.spec_plot = BasePlotWidget(
            x_label="Frequency", y_label="Amplitude",
            x_range=(lo, hi), y_range=(0, 1),
            x_min=lo, x_max=hi,
            unit_x="Hz", unit_y=self.unit_y, parent=self,
        )
        self.spec_plot.plot_widget.showGrid(x=True, y=True)
        self.spec_plot.plot_widget.addLegend(offset=(-10, 10))

        self.ref_curve = None
        if self.reference is not None:
            rf, rm, rlabel = self.reference
            self.ref_curve = self.spec_plot.plot_widget.plot(
                np.asarray(rf), np.asarray(rm),
                pen=pg.mkPen('#9E9E9E', width=1.2, style=Qt.DashLine),
                name=rlabel)

        self.spec_curve = self.spec_plot.plot_widget.plot(
            pen={'width': 2}, name='Region')
        return self.spec_plot

    @staticmethod
    def _pair(label_text, widget):
        """A label glued to its widget. QGridLayout spreads its columns evenly,
        which pushes labels far away from the field they name — so each pair gets
        its own tight QHBoxLayout instead, and only the trailing stretch absorbs
        the extra width."""
        h = QHBoxLayout()
        h.setSpacing(4)
        h.setContentsMargins(0, 0, 0, 0)
        lab = QLabel(label_text)
        lab.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        h.addWidget(lab)
        h.addWidget(widget)
        return h

    def _build_controls(self, presets):
        box = QWidget()
        outer = QVBoxLayout(box)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.setSpacing(4)

        # ── Row 0 — presets ─────────────────────────────────────────────────
        row0 = QHBoxLayout()
        row0.setSpacing(6)
        row0.addWidget(QLabel("<b>Preset:</b>"))
        for label, span in (presets or []):
            b = QPushButton(label)
            b.setToolTip(f"{span[0]*1e3:.3f} – {span[1]*1e3:.3f} ms")
            b.clicked.connect(lambda _=False, s=span: self._set_region(s))
            row0.addWidget(b)
        row0.addStretch()
        outer.addLayout(row0)

        # ── Row 1 — bounds + window ─────────────────────────────────────────
        row1 = QHBoxLayout()
        row1.setSpacing(14)

        self.spin_start = self._make_spin()
        self.spin_end = self._make_spin()
        for s in (self.spin_start, self.spin_end):
            s.valueChanged.connect(self._on_spin_changed)
        row1.addLayout(self._pair("Start:", self.spin_start))
        row1.addLayout(self._pair("End:", self.spin_end))

        self.combo_window = WideComboBox()
        self.combo_window.addItems(WINDOWS)
        self.combo_window.setCurrentText(DEFAULT_WINDOW)
        self.combo_window.setToolTip(
            "Tukey α=0.25 is the default.\n\n"
            "The decay is truncated at decay_end, where the envelope meets the\n"
            "noise floor — which for a low-SNR click is still 35–60% of the peak.\n"
            "That leaves a large step at the trailing edge, and a rectangular\n"
            "window would leak badly off it.\n\n"
            "Rectangular only wins once the region is long enough to contain the\n"
            "WHOLE decay (edge → 0); for real click regions (30–80 samples) Tukey\n"
            "reproduces the true spectrum more accurately, at only 10% worse\n"
            "resolution (NENBW 1.10 vs 1.00). Hann is far worse on both counts.")
        self.combo_window.currentTextChanged.connect(self._on_window_changed)
        row1.addLayout(self._pair("Window:", self.combo_window))

        self.spin_alpha = QDoubleSpinBox()
        self.spin_alpha.setRange(0.0, 1.0)
        self.spin_alpha.setSingleStep(0.05)
        self.spin_alpha.setDecimals(2)
        self.spin_alpha.setValue(DEFAULT_ALPHA)
        self.spin_alpha.setToolTip("Tukey taper fraction (total, split across both ends).")
        self.spin_alpha.valueChanged.connect(self._queue)
        row1.addLayout(self._pair("α:", self.spin_alpha))

        row1.addStretch()
        outer.addLayout(row1)

        # ── Row 2 — transform + display ─────────────────────────────────────
        row2 = QHBoxLayout()
        row2.setSpacing(14)

        self.combo_nfft = WideComboBox()
        for v, label in NFFT_CHOICES:
            self.combo_nfft.addItem(label, v)
        self.combo_nfft.setCurrentIndex(0)
        self.combo_nfft.setToolTip(
            "Zero-pad length. This is INTERPOLATION ONLY — it makes the curve\n"
            "smooth and refines the peak-frequency readout, but it does NOT\n"
            "improve resolution. The real resolution is Δf, shown below, and it\n"
            "is set by the region length. n_fft=512 lands on the frame's native\n"
            "390.625 Hz grid.")
        self.combo_nfft.currentIndexChanged.connect(self._queue)
        row2.addLayout(self._pair("n_fft:", self.combo_nfft))

        self.combo_scaling = WideComboBox()
        self.combo_scaling.addItem("Amplitude (V)", 'amplitude')
        self.combo_scaling.addItem("PSD (V²/Hz)", 'psd')
        self.combo_scaling.addItem("Magnitude (raw |X|)", 'magnitude')
        self.combo_scaling.currentIndexChanged.connect(self._queue)
        row2.addLayout(self._pair("Scaling:", self.combo_scaling))

        self.chk_db = QCheckBox("dB scale")
        self.chk_db.setToolTip("20·log10(X / peak), floored at −120 dB.")
        self.chk_db.toggled.connect(self._queue)
        row2.addWidget(self.chk_db)

        self.chk_ref = QCheckBox("Show reference")
        self.chk_ref.setChecked(True)
        self.chk_ref.setEnabled(self.ref_curve is not None)
        self.chk_ref.toggled.connect(self._toggle_ref)
        row2.addWidget(self.chk_ref)

        row2.addStretch()
        outer.addLayout(row2)

        return box

    def _make_spin(self):
        s = QDoubleSpinBox()
        s.setDecimals(4)
        s.setSingleStep(0.01)
        s.setSuffix(" ms")
        if len(self.time_s):
            s.setRange(self.time_s[0] * 1e3, self.time_s[-1] * 1e3)
        return s

    # ── region handling ─────────────────────────────────────────────────────

    def _on_window_changed(self, name):
        # α only means anything for Tukey.
        self.spin_alpha.setEnabled(name == 'tukey')
        self._queue()

    def _set_region(self, span):
        self.region.setRegion([float(span[0]), float(span[1])])

    def _on_region_changed(self):
        """Clamp the region to the trace and to a usable minimum length, then
        mirror it into the spin boxes. Follows the _region_check() pattern in
        replay_base_window: mutate with signals blocked to avoid recursion."""
        if len(self.time_s) < 2:
            return
        t0, t1 = self.region.getRegion()
        lo, hi = float(self.time_s[0]), float(self.time_s[-1])
        min_len = MIN_SEGMENT_SAMPLES / self.fs

        t0, t1 = (t1, t0) if t0 > t1 else (t0, t1)
        t0 = max(lo, min(t0, hi - min_len))
        t1 = min(hi, max(t1, t0 + min_len))

        cur = self.region.getRegion()
        if abs(cur[0] - t0) > 1e-12 or abs(cur[1] - t1) > 1e-12:
            self.region.blockSignals(True)
            self.region.setRegion([t0, t1])
            self.region.blockSignals(False)

        for spin, val in ((self.spin_start, t0), (self.spin_end, t1)):
            spin.blockSignals(True)
            spin.setValue(val * 1e3)
            spin.blockSignals(False)

        self._queue()

    def _on_spin_changed(self):
        self.region.blockSignals(True)
        self.region.setRegion([self.spin_start.value() / 1e3,
                               self.spin_end.value() / 1e3])
        self.region.blockSignals(False)
        self._on_region_changed()

    def _queue(self):
        self._timer.start()

    # ── computation + display ───────────────────────────────────────────────

    def _band_mask(self, freqs):
        """Bins inside the analysis band (all of them when band is None)."""
        if not self.band or len(freqs) == 0:
            return np.ones(len(freqs), dtype=bool)
        m = (freqs >= self.band[0]) & (freqs <= self.band[1])
        return m if np.any(m) else np.ones(len(freqs), dtype=bool)

    def _selected_slice(self):
        t0, t1 = self.region.getRegion()
        i0 = int(np.searchsorted(self.time_s, t0, side='left'))
        i1 = int(np.searchsorted(self.time_s, t1, side='right'))
        i0 = max(0, min(i0, len(self.signal) - 1))
        i1 = max(i0 + 1, min(i1, len(self.signal)))
        return i0, i1

    def _recompute(self):
        i0, i1 = self._selected_slice()
        seg = self.signal[i0:i1]

        if len(seg) < MIN_SEGMENT_SAMPLES:
            self.readout.setText(
                f"⚠️  Region too short: {len(seg)} samples "
                f"(minimum {MIN_SEGMENT_SAMPLES}).")
            self.spec_curve.setData([], [])
            return

        n_fft = self.combo_nfft.currentData()
        if n_fft == NATIVE:
            n_fft = len(seg)              # truly no padding

        spec = compute_spectrum(
            seg, self.fs,
            window=self.combo_window.currentText(),
            alpha=self.spin_alpha.value(),
            # No one-sided taper: with an ONSET-based region the larger
            # discontinuity is at the TRAILING edge (the decay is cut off at the
            # noise floor, still at 35–60% of peak), so a leading-edge-only taper
            # would be backwards. spectral_analysis still supports left_only for
            # other callers.
            n_fft=n_fft,
            scaling=self.combo_scaling.currentData(),
        )
        self._spec = spec

        use_db = self.chk_db.isChecked()
        # dB is relative to the IN-BAND peak, so the visible curve tops out at 0 dB.
        ref = float(np.max(spec.mags[self._band_mask(spec.freqs)])) if use_db else None
        y = to_db(spec.mags, ref=ref) if use_db else spec.mags
        self.spec_curve.setData(spec.freqs, y)

        # BasePlotWidget.set_y_range() ALSO hard-clamps the ViewBox, so a dB
        # curve (negative values) would be clipped out of existence by the
        # linear-mode limits. Set the limits explicitly on every switch.
        pw = self.spec_plot.plot_widget
        if use_db:
            pw.getViewBox().setLimits(yMin=-130, yMax=10)
            pw.setYRange(-80, 3, padding=0)
            self.spec_plot.set_y_label("Magnitude", "dB")
        else:
            # Scale to what is actually VISIBLE: the x-axis is clamped to the
            # analysis band, so out-of-band bins must not drive the y-range.
            in_band = y[self._band_mask(spec.freqs)]
            ymax = float(np.max(in_band)) if len(in_band) else 1.0
            ymax = ymax * 1.1 if np.isfinite(ymax) and ymax > 0 else 1.0
            pw.getViewBox().setLimits(yMin=-0.05 * ymax, yMax=ymax * 10)
            pw.setYRange(0, ymax, padding=0)
            self.spec_plot.set_y_label(
                "Amplitude" if spec.scaling == 'amplitude'
                else ('PSD' if spec.scaling == 'psd' else 'Magnitude'),
                spec.unit or None)

        if self.ref_curve is not None:
            self.ref_curve.setVisible(
                self.chk_ref.isChecked()
                and spec.scaling == 'amplitude' and not use_db)

        self._update_readout(spec)

    def _update_readout(self, spec):
        d = band_descriptors(spec, self.band)
        pad = "native" if spec.n_fft == spec.n_seg else f"{spec.n_fft} (zero-padded)"

        line1 = (f"Δf = <b>{_fmt_hz(spec.enbw_hz)}</b>  ·  "
                 f"N = {spec.n_seg} samples ({spec.duration_s*1e3:.3f} ms)  ·  "
                 f"n_fft = {pad}  ·  "
                 f"plotted every {_fmt_hz(spec.bin_spacing_hz)}")

        in_band = spec.mags[self._band_mask(spec.freqs)]
        peak_amp = float(np.max(in_band)) if len(in_band) else 0.0

        line2 = (f"peak {_fmt_hz(d['peak_freq_hz'])}"
                 f" @ {_fmt_unit(peak_amp, spec.unit)}"
                 f"  ·  centroid {_fmt_hz(d['centroid_hz'])}"
                 f"  ·  RMS bandwidth {_fmt_hz(d['bandwidth_rms_hz'])}"
                 f"  ·  −3 dB {_fmt_hz(d['bw_3db_hz'])}"
                 f"  ·  SPR {d['spr']:.1f}"
                 f"  ·  R_spectral {d['r_spectral']:.2f}")

        self.readout.setText(
            f"{line1}<br>{line2}"
            "<br><span style='color:#888;'>Δf is the TRUE resolution "
            "(set by the region length); n_fft only interpolates.</span>")

    def _toggle_ref(self):
        if self.ref_curve is not None:
            self._recompute()

    # ── chrome ──────────────────────────────────────────────────────────────

    #: Light-theme override. The theme CSS styles QMainWindow, not QDialog, so a
    #: dialog falls through to a dark default and the text becomes unreadable on
    #: light themes. Same workaround already used by the Decay Analysis dialog.
    _LIGHT_CSS = """
        QDialog, QWidget { background-color: white; color: black; }
        QLabel { color: black; background: transparent; }
        QPushButton { background-color: #f0f0f0; color: black;
                      border: 1px solid #ccc; padding: 4px 8px; border-radius: 3px; }
        QPushButton:hover { background-color: #e0e0e0; }
        QPushButton:disabled { color: #999; }
        QComboBox, QDoubleSpinBox { background-color: white; color: black;
                                    border: 1px solid #ccc; padding: 2px 4px;
                                    border-radius: 3px; }
        QComboBox QAbstractItemView { background-color: white; color: black;
                                      selection-background-color: #d0e4ff;
                                      selection-color: black; }
        QCheckBox { color: black; background: transparent; }
        QSplitter::handle { background-color: #ddd; }
    """

    def _apply_theme(self):
        tm = self.theme_manager
        if tm is None:
            return
        try:
            saved = tm.load_saved_theme()
            tm.apply_theme(self, saved)

            if 'light' in saved.lower():
                self.setStyleSheet(self._LIGHT_CSS)
                # The banner sets its own colour and must survive the override.
                if hasattr(self, 'banner'):
                    self.banner.setStyleSheet(
                        "color: #B26A00; font-weight: bold; font-size: 10pt; padding: 4px;")

            tm.apply_theme_to_plot(plot_widget_name=self.time_plot.plot_widget,
                                   plot_instance=self.trace)
            tm.apply_theme_to_plot(plot_widget_name=self.spec_plot.plot_widget,
                                   plot_instance=self.spec_curve)
        except Exception as e:                                   # noqa: BLE001
            print(f"⚠️ RegionFFTDialog: theme not applied ({e})")

    def _center_on_parent(self, parent):
        if not parent:
            return
        r = parent.geometry()
        self.resize(int(r.width() * 0.85), int(r.height() * 0.85))
        self.move(r.x() + (r.width() - self.width()) // 2,
                  r.y() + (r.height() - self.height()) // 2)
