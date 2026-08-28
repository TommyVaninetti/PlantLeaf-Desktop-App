# ML module for PlantLeaf v6 (current) and v5 (legacy).
# Contains SVM training and evaluation for click detection.

import sys
from pathlib import Path

DEFAULT_MODEL_FILENAME_V5 = 'plantleaf_svm_v5_0.835.pkl'
DEFAULT_MODEL_FILENAME_V6 = 'plantleaf_svm_v6_DEPLOYED.pkl'

#: version name -> filename, and the set of versions this app ships. Both models
#: are bundled: v6 is the default, and v5 stays loadable so the Data Collection
#: dialog's model browser can still select it for comparison runs.
MODEL_FILENAMES = {
    'v5': DEFAULT_MODEL_FILENAME_V5,
    'v6': DEFAULT_MODEL_FILENAME_V6,
}

DEFAULT_MODEL_VERSION = 'v6'


def default_model_path(version_name: str = DEFAULT_MODEL_VERSION) -> Path:
    """
    Absolute path to a SVM model shipped with the app.

    Resolved from this file's own location rather than from a working directory
    or AppConfig.BASE_DIR, so it is correct however the app is launched. Under
    PyInstaller the source tree is unpacked to sys._MEIPASS and this file lands
    at src/ml/ inside it, so the same relative join works frozen and unfrozen —
    which is why there is no separate frozen branch here.

    The models live in PER-VERSION subdirectories (src/ml/v5/, src/ml/v6/), not
    directly under src/ml/. The `datas` entries in PlantLeaf.spec and
    PlantLeaf_windows.spec must place them at those same subpaths — keep the two
    in sync if this ever moves.

    Parameters
    ----------
    version_name : str
        'v6' (default) or 'v5'. Validated, because the previous version of this
        function fell through to the v6 filename for any unrecognised string:
        a typo produced a plausible-looking path under a directory that does not
        exist, and the failure only surfaced later as a FileNotFoundError naming
        the wrong file.

    Returns
    -------
    Path
        Absolute path. NOT guaranteed to exist — callers should let
        load_svm_model raise FileNotFoundError and report it, rather than
        checking here.

    Raises
    ------
    ValueError
        If version_name is not a known model version.
    """
    try:
        filename = MODEL_FILENAMES[version_name]
    except KeyError:
        raise ValueError(
            f"unknown model version {version_name!r}; "
            f"expected one of {sorted(MODEL_FILENAMES)}"
        ) from None

    base = Path(sys._MEIPASS) if getattr(sys, 'frozen', False) else None
    if base is not None:
        return (base / 'src' / 'ml' / version_name / filename).resolve()
    return (Path(__file__).resolve().parent / version_name / filename).resolve()
