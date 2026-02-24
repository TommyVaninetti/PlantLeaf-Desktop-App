"""
Gestione centralizzata delle impostazioni dell'applicazione
"""

from PySide6.QtCore import QSettings
from config.app_config import AppConfig

class SettingsManager:
    """Gestisce il salvataggio e caricamento delle impostazioni dell'applicazione"""
    
    def __init__(self):
        self.settings = QSettings(AppConfig.ORGANIZATION, AppConfig.APPLICATION_NAME)
        self._load_default_settings()
    

    # carica le impostazioni predefinite se non esistono (inizializzazione)
    def _load_default_settings(self):
        """Carica le impostazioni predefinite se non esistono"""
        try:
            if not self.settings.contains("font_scale"):
                self.settings.setValue("font_scale", AppConfig.DEFAULT_FONT_SCALE)
            
            if not self.settings.contains("theme"):
                self.settings.setValue("theme", AppConfig.DEFAULT_THEME)
            
            if not self.settings.contains("window_geometry"):
                self.settings.setValue("window_geometry", None)
        except Exception as e:
            print(f"⚠️ Errore nel caricamento delle impostazioni predefinite: {e}")
            self.settings.clear()
            self._load_default_settings()  # Ricarica le impostazioni predefinite


    # Ottiene un valore dalle impostazioni
    def get_value(self, key: str, default_value=None):
        """Ottiene un valore dalle impostazioni"""
        return self.settings.value(key, default_value)
    
    # Imposta un valore nelle impostazioni
    def set_value(self, key: str, value):
        """Imposta un valore nelle impostazioni"""
        self.settings.setValue(key, value)
        self.settings.sync()  # Sincronizza le impostazioni

    
    # Reset alle impostazioni predefinite
    def reset_to_defaults(self):
        """Reset alle impostazioni predefinite"""
        self.settings.clear()
        self._load_default_settings()
    



    # QUESTE SONO SULLE FUNZIONI PER LA GESTIONE DELLA FINESTRA
    # Salva la geometria della finestra
    def save_window_geometry(self, window):
        """Salva la geometria della finestra"""
        self.settings.setValue("window_geometry", window.saveGeometry())
    
    # Ripristina la geometria della finestra
    def restore_window_geometry(self, window):
        """Ripristina la geometria della finestra"""
        geometry = self.settings.value("window_geometry")
        if geometry:
            window.restoreGeometry(geometry)
        return geometry is not None
    


    
###   # QUESTE SONO SULLE FUNZIONI PER LA GESTIONE DEI FILE RECENTI non ancora utilizzate

    # Ottiene la lista dei file recenti
    def get_recent_files(self):
        """Ottiene la lista dei file recenti"""
        return self.settings.value("recent_files", [])
    
    # Aggiunge un file alla lista dei recenti
    def add_recent_file(self, file_path):
        """Aggiunge un file alla lista dei recenti"""
        recent_files = self.get_recent_files()
        if file_path in recent_files:
            recent_files.remove(file_path)
        recent_files.insert(0, file_path)
        # Mantieni solo gli ultimi 10 file
        recent_files = recent_files[:10]
        self.settings.setValue("recent_files", recent_files)