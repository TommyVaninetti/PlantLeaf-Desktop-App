"""
Gestione del layout e ridimensionamento per l'applicazione PlantLeaf
"""

from PySide6.QtCore import QTimer
from config.app_config import AppConfig

class LayoutManager:
    """Gestisce il ridimensionamento e l'allineamento del layout"""
    
    def __init__(self, font_manager):
        self.font_manager = font_manager
        self.platform_settings = AppConfig.get_platform_specific_settings()
    

    #METODO PUBBLICO CHE AGGIUSTA IL LAYOUT A TUTTO ()
    def apply_all_layout_updates(self, window):
        """Applica tutti gli aggiornamenti necessari al layout della finestra"""
        
        # Aggiorna font delle tabelle se presenti
        if hasattr(window, 'font_manager'):
            # Controlla se la window ha una tabella chiamata 'dataTable'
            if hasattr(window, 'dataTable'):
                window.font_manager.update_table_fonts(window.dataTable)
            # Puoi aggiungere altri controlli per altri nomi di tabella, ad esempio:
            elif hasattr(window, 'tableWidget'):
                window.font_manager.update_table_fonts(window.tableWidget)
        #AGGIUNGERE LAYOUT PERSONALIZZATI


        # Margine superiore per piattaforma (macOS)
        self.setup_platform_specific_layout(window)
        # Adatta dimensione della tabella (se presente)
        if hasattr(window, 'adjust_table_layout'):
            window.adjust_table_layout()
        # Adatta dimensione della finestra
        self.adjust_window_size_for_content(window)
        # Forza il repaint
        window.update()
        window.repaint()
    
    #METODO PUBBLICO PER EVITARE PROBLEMI DI RENDERING
    def schedule_layout_updates(self, window):
        """Programma aggiornamenti multipli del layout per garantire il rendering corretto"""
        def update():
            print("🔧 Aggiornamento layout programmato...")
            self.apply_all_layout_updates(window)
        QTimer.singleShot(10, update)
        QTimer.singleShot(100, update)
        QTimer.singleShot(300, update)








    # Configura il layout iniziale della finestra
    def setup_platform_specific_layout(self, window):
        """Configura il layout specifico per la piattaforma"""
        if self.platform_settings["top_margin"] > 0:
            # Margine superiore per macOS per evitare sovrapposizione con pulsanti finestra
            window.setContentsMargins(0, self.platform_settings["top_margin"], 0, 0)




    # Adatta la DIMENSIONE DELLA FINESTRA in base a grandezza del font (MOLTO SPECIFICO)
    def adjust_window_size_for_content(self, window):
        """Adatta la dimensione delle finestre in base al contenuto e font scale"""
        scale = self.font_manager.current_font_scale

        # Calcola la larghezza minima in base al font scale
        if hasattr(window, '__class__'):
            window_class = str(window.__class__.__name__) 
            
            if window_class == 'MainWindowVoltage':
                # VOLTAGE: Calcolo CORRETTO per layout voltage
                base_width = 1060  # CORRETTO: Valore reale necessario
                # Adatta in base al font scale
                if scale <= 0.9:
                    adjusted_width = int(base_width * 0.95)  # 1007px
                elif scale == 1.1:
                    adjusted_width = int(base_width * 1.08)  # 1145px
                elif scale >= 1.2:
                    adjusted_width = int(base_width * 1.15)  # 1219px
                else:
                    adjusted_width = base_width  # 1060px (con x1)
                    
            elif window_class == 'MainWindowAudio':
                # AUDIO: Mantieni calcolo esistente
                base_width = 850 ##CONTROLLARE IL VALORE REALE necessario
                if scale <= 0.9:
                    adjusted_width = int(base_width * 0.96)  # 816px
                elif scale == 1.1:
                    adjusted_width = int(base_width * 1.10)  # 935px
                elif scale >= 1.2:
                    adjusted_width = int(base_width * 1.20)  # 1020px
                else:
                    adjusted_width = base_width  # 850px (con x1)
            else:
                return
            
            # Applica la nuova width minima
            current_min_size = window.minimumSize()
            window.setMinimumSize(adjusted_width, current_min_size.height())
            print(f"✅ Dimensione minima finestra {window_class} impostata a {adjusted_width}px")



    #ALTRI METODI UTILI PER IL LAYOUT

    # Centra la finestra sullo schermo
    def center_window_on_screen(self, window):
        """Centra la finestra sullo schermo"""
        from PySide6.QtWidgets import QApplication
        
        screen_geometry = QApplication.primaryScreen().geometry()
        window_geometry = window.geometry()
        
        x = (screen_geometry.width() - window_geometry.width()) // 2
        y = (screen_geometry.height() - window_geometry.height()) // 2
        
        window.move(x, y)
    

    # Metodo specifico per layout della tabella
    def adjust_table_layout(self, table_widget):
        """Adatta il layout della tabella in base al font e al contenuto"""
        from PySide6.QtWidgets import QHeaderView
        
        # Aggiorna la larghezza delle colonne in base al contenuto
        h_header = table_widget.horizontalHeader()
        for col in range(table_widget.columnCount()):
            h_header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
            h_header.resizeSection(col, h_header.sectionSizeHint(col))
        h_header.setStretchLastSection(True)

        # Aggiorna la dimensione minima/massima della tabella in base al font scale
        scale = getattr(self, 'current_font_scale', 1.0)
        min_width = max(400, int(600 * scale))
        min_height = max(200, int(250 * scale))
        self.setMinimumSize(min_width, min_height)
        self.setMaximumHeight(800)  # opzionale: limita l'altezza massima

        # Aggiorna la geometria e forza il repaint
        self.updateGeometry()
        self.viewport().update()
        self.repaint()