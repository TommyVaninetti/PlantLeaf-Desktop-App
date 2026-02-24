"""
Configurazioni globali dell'applicazione PlantLeaf
"""

import os
import sys

def resource_path(relative_path):
    """
    Restituisce il percorso assoluto della risorsa.
    Funziona sia in modalità Python che in modalità PyInstaller.
    """
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class AppConfig:
    """Configurazioni principali dell'applicazione"""
    
    # Informazioni applicazione
    APPLICATION_NAME = "PlantLeaf"
    ORGANIZATION = "Tommy Vaninetti"
    VERSION = "1.0.0"
    WEBSITE_URL = "https://www.plantleaf.it"
    
    # Impostazioni predefinite
    DEFAULT_FONT_SCALE = 1.35
    DEFAULT_THEME = "light_green.css"
    
    # Font per piattaforma
    PLATFORM_FONTS = {
        "Windows": "Segoe UI",
        "Darwin": "SF Pro Display",  # macOS
        "Linux": "Ubuntu"
    }
    
    # Dimensioni finestra (audio)
    MIN_WINDOW_WIDTH = 780
    MIN_WINDOW_HEIGHT = 576
    
    # Dimensioni finestra (voltage)
    VOLTAGE_MIN_WIDTH = 1060
    VOLTAGE_MIN_HEIGHT = 576


    #Fattori di scaling per width. (da aggiornare in base al font scale desiderato)
    WINDOW_SCALE_FACTORS = {
        'voltage': {
            'small': 0.95,    # Font scale <= 0.9
            'normal': 1.0,    # Font scale 1.0
            'large': 1.08,    # Font scale 1.1
            'xlarge': 1.16    # Font scale >= 1.2 (scegli tu il valore)
        },
        'audio': {
            'small': 0.96,    # Font scale <= 0.9
            'normal': 1.0,    # Font scale 1.0  
            'large': 1.10,    # Font scale 1.1
            'xlarge': 1.18    # Font scale >= 1.2 (scegli tu il valore)
        }
    }

    
    # Dimensioni finestra Home (fissa)
    HOME_WINDOW_WIDTH = 500
    HOME_WINDOW_HEIGHT = 650
    

    # Percorsi
    # CORREGGI: Percorsi - 3 livelli per arrivare alla root del progetto
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  #(project root) (/Users/tommy/PlantLeaf Development/plantleaf-app)
    THEMES_DIR = resource_path("themes")
    ASSETS_DIR = resource_path("assets")
    LOGO_DIR = resource_path("assets/logo.png")
    LOGO_WHITEBG_DIR = resource_path("assets/logo_whitebg.png")
    ICON_DIR = resource_path("assets/icons")
    README_PATH = resource_path("README.txt")
    LICENSES_PATH = resource_path("licenses.txt")

    
    # Scale font disponibili
    AVAILABLE_FONT_SCALES = [
        ("Small (1.15x)", 1.15),
        ("Normal (1.35x)", 1.35),
        ("Large (1.4x)", 1.4),
        ("Very Large (1.45x)", 1.45)
    ]
    
    # Temi disponibili
    AVAILABLE_THEMES = [
        ('dark.css', 'Dark'),
        ('dark_green.css', 'Dark Green'),
        ('dark_blue.css', 'Dark Blue'),
        ('dark_amber.css', 'Dark Amber'),
        ('light.css', 'Light'),
        ('light_green.css', 'Light Green'),
        ('light_blue.css', 'Light Blue'),
        ('light_amber.css', 'Light Amber'),
    ]
    
    # Configurazioni componenti
    COMPONENT_DEFAULTS = {
        'threshold_control': {
            'min_value': 0.0,
            'max_value': 1.0,
            'decimals': 3,
            'step': 0.001,
            'default_value': 0.05
        },
        'data_table': {
            'max_rows': 100,
            'alternating_colors': True,
            'grid_style': True
        },
        'start_stop_button': {
            'min_height': 40,
            'max_height': 50,
            'default_start_text': "START",
            'default_stop_text': "STOP"
        },
        'info_panel': {
            'min_height': 120,
            'font_scale_factor': 1.0
        }
    }


    
    # Configurazioni specifiche per piattaforma
    @staticmethod 
    def get_platform_specific_settings():
        """Restituisce impostazioni specifiche per la piattaforma corrente"""
        import platform
        system = platform.system()
        
        settings = {
            "font_family": AppConfig.PLATFORM_FONTS.get(system, "Arial"),
            "uses_native_menubar": system == "Darwin",
            "top_margin": 28 if system == "Darwin" else 0
        }
        
        return settings
    
    # Configurazioni per i componenti, da chiamare quando si vuole accedere centralmente alle info sopra definite
    @staticmethod
    def get_component_config(component_name):
        """Ottiene la configurazione per un componente specifico"""
        return AppConfig.COMPONENT_DEFAULTS.get(component_name, {})


    @staticmethod
    def load_application_logo(logo_label, target_width=250):
        """Carica il logo dell'applicazione con fallback automatico"""
        from PySide6.QtGui import QPixmap
        from PySide6.QtCore import Qt
        
        logo_loaded = False

        pixmap = QPixmap(AppConfig.LOGO_DIR)

        if not pixmap.isNull():
            # FIX: Oversampling per gestire schermi misti (Laptop vs Monitor Esterno)
            # Invece di calcolare il DPR dello schermo corrente (che può cambiare se sposti la finestra),
            # forziamo una risoluzione interna molto alta (es. 3.0x).
            # In questo modo l'immagine ha sempre abbastanza pixel per essere nitida anche su schermi 4K/Retina.
            
            target_dpr = 3.0  # Fattore di qualità elevato fisso
            physical_width = int(target_width * target_dpr)
            
            # Scaliamo l'immagine originale alla dimensione "fisica" grande
            scaled_pixmap = pixmap.scaledToWidth(physical_width, Qt.SmoothTransformation)
            
            # Diciamo a Qt che questa immagine ha una densità di pixel di 3.0
            # Qt la disegnerà occupando 'target_width' pixel logici, ma usando tutti i dettagli
            scaled_pixmap.setDevicePixelRatio(target_dpr)
            
            logo_label.setPixmap(scaled_pixmap)
            logo_loaded = True
            print(f"✅ Logo loaded successfully (High DPI Oversampling)")
  

        if not logo_loaded:
            # Fallback unificato con styling coerente
            logo_label.setText("🌿 PlantLeaf")
            logo_label.setStyleSheet("""
                font-size: 48px;
                font-weight: bold;
                color: #4CAF50;
                padding: 40px;
            """)
            print("📝 Using text fallback for logo")
        
        logo_label.setAlignment(Qt.AlignCenter)
        return logo_loaded