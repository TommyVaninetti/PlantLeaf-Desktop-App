"""
WideComboBox — a QComboBox whose drop-down is wide enough to read.

Qt lays the popup out to the *combo's* width at the moment it is shown, and elides
anything longer — so widening the view once at construction time has no effect (it
gets overwritten). The width has to be forced from inside showPopup(), immediately
before the popup is laid out.

Elision is also disabled outright, so a long entry can never be cut to "…".

Use this instead of QComboBox anywhere the entries can be longer than the collapsed
combo: it costs nothing when they are not.
"""

from PySide6.QtWidgets import QComboBox, QListView, QStyle
from PySide6.QtCore import Qt


class WideComboBox(QComboBox):
    """A QComboBox that never truncates its drop-down entries."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # A plain QListView is required: the default popup view ignores
        # setTextElideMode().
        view = QListView(self)
        view.setTextElideMode(Qt.ElideNone)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setView(view)

    def _widest_item_px(self) -> int:
        fm = self.view().fontMetrics()
        return max((fm.horizontalAdvance(self.itemText(i))
                    for i in range(self.count())), default=0)

    def showPopup(self):
        extra = self.style().pixelMetric(QStyle.PM_ScrollBarExtent) + 24  # bar + padding
        want = self._widest_item_px() + extra
        view = self.view()
        view.setMinimumWidth(max(want, self.width()))
        # The popup lives in the view's window (a QFrame); widen it too, or it
        # will clip the view back to the combo's width.
        popup = view.window()
        if popup is not None:
            popup.setMinimumWidth(max(want, self.width()))
        super().showPopup()
