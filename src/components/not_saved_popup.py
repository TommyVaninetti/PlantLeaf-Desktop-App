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