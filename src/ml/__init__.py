# ML module for PlantLeaf v5
# Contains SVM training and evaluation for click detection

import sys
from pathlib import Path

DEFAULT_MODEL_FILENAME = 'plantleaf_svm_v5_0.835.pkl'


def default_model_path() -> Path:
    """
    Absolute path to the SVM model shipped with the app.

    Resolved from this file's own location rather than from a working directory
    or AppConfig.BASE_DIR, so it is correct however the app is launched. Under
    PyInstaller the source tree is unpacked to sys._MEIPASS, and the .pkl is
    placed at src/ml/ inside the bundle by the `datas` entry in PlantLeaf.spec —
    keep the two in sync if this ever moves.

    The path is not guaranteed to exist; callers should let load_svm_model raise
    FileNotFoundError and report it, rather than checking here.
    """
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS) / 'src' / 'ml' / DEFAULT_MODEL_FILENAME
    return Path(__file__).resolve().parent / DEFAULT_MODEL_FILENAME
