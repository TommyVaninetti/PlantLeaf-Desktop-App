"""
Wake Lock Manager - Previene lo sleep del sistema durante l'acquisizione.
Compatibile con Windows, Linux e macOS tramite la libreria wakepy.
"""

class WakeLockManager:
    """
    Gestisce il wake lock del sistema per prevenire lo sleep durante l'acquisizione.
    Usa wakepy (cross-platform: Windows, Linux, macOS).
    """

    def __init__(self):
        self._wake = None
        self._active = False
        self._available = False

        try:
            import wakepy  # noqa: F401
            self._available = True
        except ImportError:
            print("⚠️ wakepy non disponibile: il sistema potrebbe andare in sleep durante l'acquisizione.")
            print("   Installa con: pip install wakepy")

    def acquire(self):
        """Attiva il wake lock: impedisce al sistema di andare in sleep."""
        if not self._available or self._active:
            return

        try:
            from wakepy import keep
            self._wake = keep.running()
            self._wake.__enter__()
            self._active = True
            print("☀️ Wake lock attivato: il sistema non andrà in sleep durante l'acquisizione.")
        except Exception as e:
            print(f"⚠️ Impossibile attivare wake lock: {e}")

    def release(self):
        """Disattiva il wake lock: il sistema può tornare a spegnersi normalmente."""
        if not self._active or self._wake is None:
            return

        try:
            self._wake.__exit__(None, None, None)
            self._wake = None
            self._active = False
            print("🌙 Wake lock rilasciato: gestione sleep normale ripristinata.")
        except Exception as e:
            print(f"⚠️ Errore durante il rilascio del wake lock: {e}")

    @property
    def is_active(self):
        return self._active
