"""
Gestione dei temi CSS per l'applicazione PlantLeaf
"""

import os
import re
import pyqtgraph as pg
from PySide6.QtGui import QColor
from config.app_config import AppConfig

class ThemeManager:
    def __init__(self, settings, font_manager):
        self.settings = settings
        self.font_manager = font_manager
        self.current_theme = None
        self.themes_dir = AppConfig.THEMES_DIR

        # Carica tema salvato o default
        saved_theme = self.settings.value("theme", "light_blue.css", type=str)
        self.current_theme = saved_theme



    

    #ALTRE FUNZIONI UTILI
    #Carica il tema salvato nelle impostazioni
    def load_saved_theme(self) -> str:
        """Carica il tema salvato dalle impostazioni"""
        try:
            saved_theme = self.settings.value("theme", "light_blue.css", type=str)  # ✅ CORRETTO
            print(f"📂 Tema salvato caricato: {saved_theme}")
            return saved_theme
        except Exception as e:
            print(f"⚠️ Errore caricamento tema salvato: {e}")
            return "light_blue.css"  # Default fallback

    #Salva il tema
    def save_theme(self, theme_name: str):
        """Salva il tema corrente nelle impostazioni"""
        self.settings.setValue("theme", theme_name)
        self.current_theme = theme_name
        self.settings.sync()  # Sincronizza le impostazioni

        print(f"💾 Tema salvato: {theme_name}")
        return True


    #Ottieni i colori del tema corrente
    def get_theme_colors(self, theme_name=None):
        """Estrae i colori principali dal CSS del tema"""
        if theme_name is None:
            theme_name = self.current_theme
        theme_path = os.path.join(self.themes_dir, theme_name)
        colors = {}
        try:
            with open(theme_path, 'r', encoding='utf-8') as f:
                css = f.read()
            # Regex per trovare background e color
            bg = re.search(r'QMainWindow\s*{[^}]*background-color:\s*([^;]+);', css)
            fg = re.search(r'QMainWindow\s*{[^}]*color:\s*([^;]+);', css)
            label_fg = re.search(r'QLabel\s*{[^}]*color:\s*([^;]+);', css)
            axis_fg = fg or label_fg
            colors['background'] = bg.group(1).strip() if bg else "#ffffff"
            colors['foreground'] = (axis_fg.group(1).strip() if axis_fg else "#000000")
            # Puoi aggiungere qui altre estrazioni (es: border, accent, ecc.)
        except Exception as e:
            print(f"Errore lettura tema: {e}")
            colors = {'background': "#ffffff", 'foreground': "#000000"}
        return colors
    
    def get_accent_color(self, theme_name=None):
        if theme_name is None:
            theme_name = self.current_theme
        theme_path = os.path.join(self.themes_dir, theme_name)
        try:
            with open(theme_path, 'r', encoding='utf-8') as f:
                css = f.read()
            # Cerca un colore "accent" (es: background-color di QPushButton:checked)
            accent = re.search(r'QPushButton:checked\s*{[^}]*background-color:\s*([^;]+);', css)
            if accent:
                return accent.group(1).strip()
        except Exception as e:
            print(f"Errore lettura accent color: {e}")
        return "#007bff"  # fallback
    


    #GESTIONE TEMA DEL GRAFICO
    def apply_theme_to_plot(self, plot_widget_name, plot_instance=None):
        from PySide6.QtGui import QColor

        theme_colors = self.get_theme_colors()
        theme_bg = theme_colors['background']
        theme_fg = theme_colors['foreground']
        accent_color = QColor(self.get_accent_color())

        # Cambia sfondo
        plot_widget_name.setBackground(theme_bg)

        # Colore delle label degli assi
        plot_widget_name.getAxis('bottom').setTextPen(QColor(theme_fg))
        plot_widget_name.getAxis('left').setTextPen(QColor(theme_fg))

        # Cambia colore della curva principale
        if plot_instance is not None:
           plot_instance.setPen(pg.mkPen(accent_color, width=2))  # imposta anche width!


#PUNTO DI ACCESSO PER CAMBIARE IL TEMA
    #APPLICA TEMA E FONT DINAMICO
    def apply_theme(self, window, theme_name: str):
        # Imposta il tema corrente e lo salva
        self.current_theme = theme_name
        self.save_theme(theme_name)

        theme_path = os.path.join(AppConfig.THEMES_DIR, theme_name)
        print(f"🎨 Loading theme: {theme_path}")
        print(f"📁 Theme exists: {os.path.exists(theme_path)}")

        if not os.path.exists(theme_path):
            print(f"❌ Tema non trovato: {theme_path}")
            return False

        with open(theme_path, 'r', encoding='utf-8') as theme_file:
            css_content = theme_file.read()

        # Applica font personalizzati
        self.font_manager.apply_fonts_to_widgets(window)
        css_content = self.font_manager._customize_css_for_fonts(css_content)

        # Aggiungi il CSS per le ComboBox
        css_content += self.get_combobox_css()

        # Applica il CSS completo alla finestra
        window.setStyleSheet(css_content)

        return True

    # APPLICA TEMA E FONT DINAMICO PER WIDGETS (usa anche per i temi chiari il tema corrispondente scuro)
    def apply_theme_to_widgets(self, widget, *, background="#fafafa", foreground="#222", border_radius=8, padding=8):
        """
        Applica uno stile CSS di base e leggibile al widget passato.
        Non dipende dal tema globale e garantisce contrasto.
        """
        css = f"""
        QWidget {{
            background-color: {background};
            color: {foreground};
            border-radius: {border_radius}px;
            padding: {padding}px;
        }}
        QLabel, QCheckBox, QRadioButton {{
            color: {foreground};
            font-size: 15px;
        }}
        QLabel#titleLabel {{
            font-size: 18px;
            font-weight: bold;
        }}
        QPushButton {{
            background-color: #e0e0e0;
            color: #222;
            border: 1px solid #bbb;
            border-radius: 6px;
            padding: 6px 12px;
        }}
        QPushButton:hover {{
            background-color: #d5d5d5;
        }}
        QComboBox {{
            background-color: #f5f5f5;
            color: #222;
            border: 1px solid #bbb;
            border-radius: 6px;
        }}
        QLineEdit, QTextEdit {{
            background-color: #fff;
            color: #222;
            border: 1px solid #bbb;
            border-radius: 6px;
        }}
        """
        widget.setStyleSheet(css)

    

#COLORI SPECIFICI PER PULSANTI E COMPONENTI SPECIALI

    # Colori specifici per ogni tema
    def get_theme_specific_colors(self):
        """Colori specifici per ogni tema, sia attivi che inattivi"""
        theme_map = {
            'dark_green.css': {
                'active_bg': '#388e3c',
                'active_text': '#ffffff',
                'active_border': '#256d27',
                'active_hover': '#256d27',
                'active_pressed': '#1b4d1a',
                'inactive_bg': '#e0e0e0',
                'inactive_text': '#333333',
                'inactive_border': '#bdbdbd',
                'inactive_hover': '#cccccc',
                'inactive_pressed': '#bdbdbd',
            },
            'dark_blue.css': {
                'active_bg': '#1d4ed8',
                'active_text': '#ffffff',
                'active_border': '#1e40af',
                'active_hover': '#1e40af',
                'active_pressed': '#172554',
                'inactive_bg': '#e0e0e0',
                'inactive_text': '#333333',
                'inactive_border': '#bdbdbd',
                'inactive_hover': '#cccccc',
                'inactive_pressed': '#bdbdbd',
            },
            'dark_amber.css': {
                'active_bg': '#ff8f00',
                'active_text': '#3e2723',
                'active_border': '#ff6f00',
                'active_hover': '#ff6f00',
                'active_pressed': '#ff8f00',
                'inactive_bg': '#fff8e1',
                'inactive_text': '#333333',
                'inactive_border': '#ffe082',
                'inactive_hover': '#ffe082',
                'inactive_pressed': '#ffd54f',
            },
            'light_green.css': {
                'active_bg': '#5a8859',
                'active_text': '#ffffff',
                'active_border': '#4a7348',
                'active_hover': '#4a7348',
                'active_pressed': '#3b5c39',
                'inactive_bg': '#f5f5f5',
                'inactive_text': '#333333',
                'inactive_border': '#bdbdbd',
                'inactive_hover': '#e0e0e0',
                'inactive_pressed': '#bdbdbd',
            },
            'light_blue.css': {
                'active_bg': '#578bb5',
                'active_text': '#ffffff',
                'active_border': '#467aa3',
                'active_hover': '#467aa3',
                'active_pressed': '#345a7a',
                'inactive_bg': '#f5f5f5',
                'inactive_text': '#333333',
                'inactive_border': '#bdbdbd',
                'inactive_hover': '#e0e0e0',
                'inactive_pressed': '#bdbdbd',
            },
            'light_amber.css': {
                'active_bg': '#B8860B',
                'active_text': '#ffffff',
                'active_border': '#DAA520',
                'active_hover': '#DAA520',
                'active_pressed': '#9a7209',
                'inactive_bg': '#fff8e1',
                'inactive_text': '#333333',
                'inactive_border': '#ffe082',
                'inactive_hover': '#ffe082',
                'inactive_pressed': '#ffd54f',
            },
            'dark.css': {
                'active_bg': '#333333',
                'active_text': '#ffffff',
                'active_border': '#222222',
                'active_hover': '#222222',
                'active_pressed': '#111111',
                'inactive_bg': '#bbbbbb',
                'inactive_text': '#222222',
                'inactive_border': '#888888',
                'inactive_hover': '#cccccc',
                'inactive_pressed': '#aaaaaa',
            },
            'light.css': {
                'active_bg': '#1976d2',
                'active_text': '#ffffff',
                'active_border': '#1565c0',
                'active_hover': '#1565c0',
                'active_pressed': '#0d47a1',
                'inactive_bg': '#f5f5f5',
                'inactive_text': '#333333',
                'inactive_border': '#bdbdbd',
                'inactive_hover': '#e0e0e0',
                'inactive_pressed': '#bdbdbd',
            },
        }
        return theme_map.get(self.current_theme, theme_map['light.css'])


    #COLORI PULSANTI TOGGLE
    def get_toggle_button_style(self, is_active=True):
        """Genera CSS completo per toggle buttons, includendo lo stato disabilitato."""
        colors = self.get_theme_specific_colors()
        theme_colors = self.get_theme_colors() # Per i colori base
        
        # Calcola altezza dinamica come StartStopButton
        font = self.font_manager.create_fonts().get('button', None)
        font_pixel_size = font.pixelSize() if font and font.pixelSize() > 0 else 16
        button_height = max(30, int(font_pixel_size * 1.75))

        # Colori per lo stato disabilitato
        disabled_bg = theme_colors.get('background', '#222')
        disabled_fg = theme_colors.get('foreground', '#888')

        base_style = f"""
            QPushButton:disabled {{
                background-color: {disabled_bg};
                color: {disabled_fg};
                border: 1px solid {disabled_fg};
                opacity: 0.5;
            }}
        """

        if is_active:
            return base_style + f"""
            QPushButton {{
                min-height: {button_height}px;
                max-height: {button_height}px;
                background-color: {colors['active_bg']};
                color: {colors['active_text']};
                border: 1px solid {colors['active_border']};
                padding: 5px;
            }}
            QPushButton:hover {{
                background-color: {colors['active_hover']};
            }}
            QPushButton:pressed {{
                background-color: {colors['active_pressed']};
            }}
            """
        else: # Inactive
            return base_style + f"""
            QPushButton {{
                min-height: {button_height}px;
                max-height: {button_height}px;
                background-color: {colors['inactive_bg']};
                color: {colors['inactive_text']};
                border: 1px solid {colors['inactive_border']};
                padding: 5px;
            }}
            QPushButton:hover {{
                background-color: {colors['inactive_hover']};
            }}
            QPushButton:pressed {{
                background-color: {colors['inactive_pressed']};
            }}
            """
        


    #COLORI E CSS PULSANTI START/STOP
    def get_start_stop_css(self, is_running: bool):
        colors = self._get_start_stop_colors()
        font = self.font_manager.create_fonts().get('button', None)
        font_pixel_size = font.pixelSize() if font and font.pixelSize() > 0 else 16

        button_height = max(30, int(font_pixel_size * 1.75))

        button_width = max(80, int(font_pixel_size * 5))

        if hasattr(self.font_manager, "get_button_heights"):
            button_height = max(button_height, self.font_manager.get_button_heights().get('button', 30))

        css = f"""
        QPushButton {{
            background-color: {colors['stop_bg'] if is_running else colors['start_bg']};
            color: {colors['stop_text'] if is_running else colors['start_text']};
            border: 2px solid {colors['stop_border'] if is_running else colors['start_border']};
            min-height: {button_height}px;
            max-height: {button_height}px;
            min-width: {button_width}px;
            max-width: {button_width}px;
            font-size: {font_pixel_size}px;
        }}
        QPushButton:hover {{
            background-color: {colors['stop_hover'] if is_running else colors['start_hover']};
        }}
        QPushButton:pressed {{
            background-color: {colors['stop_pressed'] if is_running else colors['start_pressed']};
        }}
        QPushButton:disabled {{
            background-color: #cccccc;  /* Colore disabilitato */
            color: #666666;  /* Testo disabilitato */
            border: 1px solid #999999;  /* Bordo disabilitato */
        }}
        """
        return css

    def _get_start_stop_colors(self):
        """Restituisce i colori del tema corrente per pulsanti START/STOP"""
        start_stop_colors = {
            # Colori per pulsante START
            'start_bg': '#4CAF50',              # Verde base
            'start_text': '#ffffff',
            'start_border': '#45a049',
            'start_hover': '#45a049',
            'start_pressed': '#3d8b40',
            
            # Colori per pulsante STOP
            'stop_bg': '#f44336',               # Rosso base
            'stop_text': '#ffffff', 
            'stop_border': '#da190b',
            'stop_hover': '#da190b',
            'stop_pressed': '#c62828'
        }
 
        return start_stop_colors
    

    #CSS PER COMBOBOX
    def get_combobox_css(self):
        colors = self.get_theme_specific_colors()
        return f"""
        QComboBox {{
            background-color: {colors['inactive_bg']};
            color: {colors['inactive_text']};
            border: 1px solid {colors['inactive_border']};
        }}
        QComboBox:hover {{
            background-color: {colors['inactive_hover']};
            color: {colors['inactive_text']};
            border: 1px solid {colors['inactive_border']};
        }}
        QComboBox:focus {{
            background-color: {colors['active_bg']};
            color: {colors['active_text']};
            border: 1px solid {colors['active_border']};
        }}
        QComboBox QAbstractItemView {{
            background-color: {colors['active_bg']};
            color: {colors['active_text']};
            selection-background-color: {colors['active_hover']};
            selection-color: {colors['active_text']};
        }}
        QComboBox QAbstractItemView::item:hover {{
            background-color: {colors['active_hover']};
            color: {colors['active_text']};
        }}
        QComboBox QAbstractItemView::indicator {{
            width: 0px;
            height: 0px;
            padding: 0px;
            margin: 0px;
            border: none;
            background: transparent;
        }}
        QComboBox QAbstractItemView::item {{
            padding-left: 8px; /* opzionale: sposta il testo a sinistra */
        }}
        """


    def get_toolbar_bg(self, theme_name=None):
        """Estrae il background-color della QToolBar dal CSS del tema"""
        if theme_name is None:
            theme_name = self.current_theme
        theme_path = os.path.join(self.themes_dir, theme_name)
        try:
            with open(theme_path, 'r', encoding='utf-8') as f:
                css = f.read()
            tb_bg = re.search(r'QToolBar\s*{[^}]*background(?:-color)?:\s*([^;]+);', css)
            if tb_bg:
                return tb_bg.group(1).strip()
        except Exception as e:
            print(f"Errore lettura toolbar bg: {e}")
        return "#222"  # fallback