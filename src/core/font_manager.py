"""
Gestione dei font scalabili per l'applicazione PlantLeaf
"""

import platform
from PySide6.QtGui import QFont
from PySide6.QtCore import QSettings
from config.app_config import AppConfig

class FontManager:
    """Gestisce la scalatura e applicazione dei font in modo cross-platform"""
    
    def __init__(self, settings: QSettings):
        self.settings = settings
        self.current_font_scale = 1.35
        self.font_family = self._get_platform_font()

    #SCELTA DEL FONT IN BASE A OS
    def _get_platform_font(self) -> str:
        """Restituisce il font ottimale per la piattaforma corrente"""
        try:
            platform_settings = AppConfig.get_platform_specific_settings()
            return platform_settings["font_family"]
        except AttributeError:
            # Fallback se il metodo non esiste
            import platform
            system = platform.system()
            return AppConfig.PLATFORM_FONTS.get(system, "Arial")
        



    #APPLICA FONT DINAMICO A TUTTI I WIDGETS (PUNTO DI ACCESSO)
    def apply_fonts_and_css(self, window):
        """Applica font e CSS dinamico a tutti i widget della finestra"""
        self.apply_fonts_to_widgets(window)
        css = self._customize_css_for_fonts("")
        window.setStyleSheet(css)

        # Aggiorna layout dopo il cambio font
        if hasattr(window, 'layout_manager'):
            window.layout_manager.apply_all_layout_updates(window)




    #DIMENSIONI FONT SCALATI (vedi sotto per applicazioni)
    def get_scaled_font_sizes(self) -> dict:
        """Restituisce le dimensioni dei font scalate"""
        base_sizes = {
            'main': 16,
            'header': 18,
            'button': 11, # vengono cmq ingranditi dopo
            'label': 15,
            'info_label': 14,
            'small_label': 12,
            'spinbox': 14,
            'table_content': 13,
            'table_header': 12,
            'group': 15,
            'title': 16,
            'combobox': 14,
        }
        
        return {key: int(size * self.current_font_scale) 
                for key, size in base_sizes.items()}
    





    #CREAZIONE FONT
    def create_fonts(self) -> dict:
        """Crea oggetti QFont con le dimensioni scalate"""
        sizes = self.get_scaled_font_sizes()
        
        return {
            'main': QFont(self.font_family, sizes['main']),
            'header': QFont(self.font_family, sizes['header'], QFont.Weight.Bold),
            'button': QFont(self.font_family, sizes['button'], QFont.Weight.Medium),
            'label': QFont(self.font_family, 18, QFont.Weight.Medium), 
            'info_label': QFont(self.font_family, sizes['info_label']),
            'small_label': QFont(self.font_family, sizes['small_label']),
            'spinbox': QFont(self.font_family, sizes['spinbox']),
            'table_content': QFont(self.font_family, sizes['table_content']),
            'table_header': QFont(self.font_family, sizes['table_header'], QFont.Weight.Bold),
            'group': QFont(self.font_family, sizes['group'], QFont.Weight.Medium),
            'combobox': QFont(self.font_family, sizes['combobox'], QFont.Weight.Medium),

        }



    # APPLICA I FONT SCALATI
    def _customize_css_for_fonts(self, css_content: str) -> str:
        """Personalizza il CSS con le dimensioni dei font dinamiche - SOLO FONT, NO PULSANTI"""
        sizes = self.get_scaled_font_sizes()
        font_family = self.font_family
        
        # Calcola altezze dinamiche per altri widget (NON pulsanti)
        tab_height = max(28, int(30 * self.current_font_scale))
        spinbox_min_height = max(20, int(22 * self.current_font_scale))
        spinbox_max_height = max(26, int(28 * self.current_font_scale))
        
        dynamic_css = f"""
        /* SOLO FONT SCALING - NIENTE DIMENSIONI PULSANTI */
        
        /* Widget generici con font scaling */
        QMainWindow {{
            font-size: {sizes['main']}px;
            font-family: "{font_family}";
        }}
        
        
        QTabBar {{
            font-size: {sizes['button']}px;
            font-family: "{font_family}";
        }}
        
        QTabBar::tab {{
            height: {tab_height}px;
            font-size: {sizes['button']}px;
            font-family: "{font_family}";
        }}
        
        QLabel {{
            font-size: {sizes['info_label']}px;
            font-family: "{font_family}";
        }}
        
        QGroupBox {{
            font-size: {sizes['group']}px;
            font-family: "{font_family}";
        }}
        
        QGroupBox::title {{
            font-size: {sizes['title']}px;
            font-family: "{font_family}";
        }}
        
        QTableWidget {{
            font-size: {sizes['table_content']}px;
            font-family: "{font_family}";
        }}
        
        QHeaderView::section {{
            font-size: {sizes['table_header']}px;
            font-family: "{font_family}";
        }}
        
        QSpinBox, QDoubleSpinBox {{
            font-size: {sizes['spinbox']}px;
            font-family: "{font_family}";
            min-height: {spinbox_min_height}px;
            max-height: {spinbox_max_height}px;
        }}

        QComboBox {{
            font-size: {sizes['combobox']}px;
            font-family: "{font_family}";
        }}

        /* ✅ FIX WINDOWS: Stile per pulsanti disabilitati */
        QPushButton:disabled {{
            color: rgba(128, 128, 128, 0.5);
            background-color: rgba(200, 200, 200, 0.3);
            border-color: rgba(128, 128, 128, 0.3);
        }}
        
        /* ✅ FIX WINDOWS: Icone semitrasparenti quando disabilitate */
        QToolButton:disabled {{
            opacity: 0.4;
        }}
        
        QAction:disabled {{
            opacity: 0.4;
        }}

        /* Solo MainWindowHome buttons - lascia gli altri in pace */
        #titleLabel {{
            font-size: {int(26 * self.current_font_scale)}px;
            font-family: "{font_family}";
            font-weight: bold;
        }}
        
        #subtitleLabel {{
            font-size: {int(14 * self.current_font_scale)}px;
            font-family: "{font_family}";
        }}
        
        #footerLabel {{
            font-size: {int(12 * self.current_font_scale)}px;
            font-family: "{font_family}";
        }}
        
        #mainButton {{
            font-size: {int(16 * self.current_font_scale)}px;
            font-family: "{font_family}";
            font-weight: 600;
            min-height: {int(35 * self.current_font_scale)}px;
            max-height: {int(35 * self.current_font_scale)}px;
            padding: 8px 20px;
        }}
        """
        
        return css_content + dynamic_css




    # APPLICA FONT A NOME FINESTRA, CREA MAPPA DEI FONT E APPLICA FONT ALLE BARRE E POI A TUTTI I WIDGETS TRAMITE _APPLY_WIDGET_FONTS
    #METODO PUBBLICO CHE CHIAMA QUELLI SOTTO
    def apply_fonts_to_widgets(self, window):
        fonts = self.create_fonts()
        
        # Font principale della finestra
        window.setFont(fonts['main'])
        
       


    # APPLICA FONT A WIDGETS COMUNI (vedi sotto per vedere quali)
    def _apply_widget_fonts(self, window, fonts):
        """Applica font ai widget comuni - USA CHIAVI CORRETTE"""
        from PySide6.QtWidgets import (QLabel, QPushButton, QLineEdit, QTextEdit, 
                                    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, 
                                    QRadioButton, QGroupBox, QTabWidget)
        
        try:
            # DEBUG: Mostra chiavi disponibili
            print(f"🔍 Font keys disponibili: {list(fonts.keys())}")
            
            # CORREGGI: Applica font alle label - USA 'button' come fallback
            for label in window.findChildren(QLabel):
                if label.objectName():
                    # Usa 'button' se 'label' non esiste
                    font_key = 'label' if 'label' in fonts else 'button'
                    label.setFont(fonts[font_key])
            
            # CORREGGI: Applica font ai pulsanti
            for button in window.findChildren(QPushButton):
                if button.objectName():
                    button.setFont(fonts['button'])
            
            # CORREGGI: Applica font agli input
            for widget in window.findChildren(QLineEdit):
                if widget.objectName():
                    # Usa 'input' se esiste, altrimenti 'button'
                    font_key = 'input' if 'input' in fonts else 'button'
                    widget.setFont(fonts[font_key])
            
            for widget in window.findChildren(QTextEdit):
                if widget.objectName():
                    font_key = 'input' if 'input' in fonts else 'button'
                    widget.setFont(fonts[font_key])
            
            # CORREGGI: Applica font agli spinbox
            for widget in window.findChildren(QSpinBox):
                if widget.objectName():
                    font_key = 'spinbox' if 'spinbox' in fonts else 'button'
                    widget.setFont(fonts[font_key])
            
            for widget in window.findChildren(QDoubleSpinBox):
                if widget.objectName():
                    font_key = 'spinbox' if 'spinbox' in fonts else 'button'
                    widget.setFont(fonts[font_key])
            
            # CORREGGI: Applica font ai combobox
            for widget in window.findChildren(QComboBox):
                if widget.objectName():
                    font_key = 'combobox' if 'combobox' in fonts else 'button'
                    widget.setFont(fonts[font_key])
            
            # CORREGGI: Applica font ai checkbox
            for widget in window.findChildren(QCheckBox):
                if widget.objectName():
                    font_key = 'checkbox' if 'checkbox' in fonts else 'button'
                    widget.setFont(fonts[font_key])
            
            for widget in window.findChildren(QRadioButton):
                if widget.objectName():
                    font_key = 'checkbox' if 'checkbox' in fonts else 'button'
                    widget.setFont(fonts[font_key])
            
            # CORREGGI: Applica font ai groupbox
            for widget in window.findChildren(QGroupBox):
                if widget.objectName():
                    font_key = 'groupbox' if 'groupbox' in fonts else 'button'
                    widget.setFont(fonts[font_key])
            
            print("✅ Font applicati a tutti i widget")
            
        except Exception as e:
            print(f"⚠️ Errore nell'applicazione font widget: {e}")
            import traceback
            traceback.print_exc()




    #ALTRE FUNZIONI UTILI E SPECIFICHE PER TROVARE VALORI NEI FONT
    #SCALATURA FONT
    def load_font_scale(self) -> float:
        """Ritorna la scala del font dalle impostazioni, con validazione"""
        try:
            font_scale = float(self.settings.value("font_scale", 1.0))
            if font_scale < 0.8 or font_scale > 1.5:
                font_scale = 1.0
            self.current_font_scale = font_scale
            return font_scale
        except (ValueError, TypeError):
            self.current_font_scale = 1.0
            return 1.0


    #ALTEZZE PULSANTI
    def get_button_heights(self) -> dict:
        """Restituisce altezze ottimizzate per i diversi tipi di pulsanti"""
        scale = self.current_font_scale
        
        return {
            'main_button': max(35, int(35 * scale)),      # pulsante principale
            'action_button': max(24, int(27 * scale)),    #pulsanti di azione     
            'small_button': max(20, int(23 * scale)),         
            'toolbar_button': max(18, int(20 * scale))        
        }

        #SALVA FONT SCALE
    def save_font_scale(self, scale: float):
        """Salva la scala font nelle impostazioni"""
        self.current_font_scale = scale
        self.settings.setValue("font_scale", scale)
        self.settings.sync()  # Sincronizza le impostazioni