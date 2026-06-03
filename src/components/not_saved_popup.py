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

from PySide6.QtWidgets import QMessageBox
from PySide6.QtGui import QIcon, QFont
from config.app_config import AppConfig

def show_not_saved_popup(parent=None):
    msg = QMessageBox(parent)
    msg.setIcon(QMessageBox.Warning)
    msg.setWindowTitle("Attention - Unsaved Changes")
    msg.setWindowIcon(QIcon(AppConfig.LOGO_DIR))
    msg.setText("The document has unsaved changes.\nDo you want to save before leaving?")
    msg.setStandardButtons(QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
    msg.setDefaultButton(QMessageBox.Save)

    default_font = QFont()
    msg.setFont(default_font)
    # Forza stile chiaro
    msg.setStyleSheet("""
        QMessageBox {
            background-color: white;
            color: black;
        }
        QLabel {
            color: black;
        }
        QPushButton {
            color: black;
            background-color: #f0f0f0;
        }
    """)

    result = msg.exec()
    if result == QMessageBox.Save:
        return "save"
    elif result == QMessageBox.Discard:
        return "dont_save"
    elif result == QMessageBox.Cancel:
        return "cancel"