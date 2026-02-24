"""
PlantLeaf Analysis Tool - Main Entry Point

Applicazione per l'analisi di segnali audio e voltage per il monitoraggio delle piante.
Supporta analisi in tempo reale e caricamento di file salvati precedentemente.
"""

import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon


# Aggiungi il percorso del progetto al PYTHONPATH per permettere gli import. (questo lasciamolo cosi senza usare AppConfig perche' e' necessario per gli import e nel caso di errori ci avvisa)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)

# Import delle classi necessarie
try:
    from windows.main_window_home import MainWindowHome
    from config.app_config import AppConfig
except ImportError as e:
    print(f"❌ Errore nell'importazione dei moduli: {e}")
    print(f"🔍 Percorso src: {src_path}")
    print(f"🔍 Percorso progetto: {project_root}")
    sys.exit(1)


# Classe per intercettare QFileOpenEvents
class PlantLeafApplication(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        self.file_to_open = None
        self.home_window = None
    
    def event(self, event):
        """Intercetta l'evento di apertura file da macOS"""
        from PySide6.QtCore import QEvent, QTimer
        if event.type() == QEvent.FileOpen:
            file_open_event = event
            self.file_to_open = file_open_event.file()
            print(f"📂 [FileOpenEvent] Ricevuto file da macOS: {self.file_to_open}")
            
            # Se la home window è già stata creata, apri il file
            if self.home_window is not None:
                print(f"✅ Home window già presente, apertura file...")
                QTimer.singleShot(100, lambda: self.home_window.open_file_action(self.file_to_open))
            
            return True
        
        return super().event(event)



def setup_application():
    """Configura l'applicazione Qt con le impostazioni base"""
    # PRIMA: Abilita High DPI scaling PRIMA di creare QApplication
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    
    # DOPO: Crea l'applicazione Qt USANDO LA CLASSE PERSONALIZZATA
    app = PlantLeafApplication(sys.argv)
    
    # Configura le proprietà dell'applicazione
    app.setApplicationName(AppConfig.APPLICATION_NAME)
    app.setOrganizationName(AppConfig.ORGANIZATION)
    app.setApplicationVersion(AppConfig.VERSION)
    app.setApplicationDisplayName("PlantLeaf")
    
    # Setup icona (corretto come sopra)
    #if AppConfig.LOGO_DIR:
     #   app.setWindowIcon(QIcon(AppConfig.LOGO_DIR))
    #else:
     #   print("⚠️ Icona non trovata in assets/logo.png, utilizzo icona predefinita")
      #  app.setWindowIcon(QIcon()) # Imposta un'icona vuota se non trovata
    #LASCIA COMMENTATO, ALMENO IN QUELLO FINALE È GESTITO AUTOMATICAMENTE DAL SISTEMA E USA IL .ICNS O .ICO

    return app


def main():
    """Funzione principale dell'applicazione"""
    print("🚀 Avvio PlantLeaf Analysis Tool...")
    print(f"📁 Directory progetto: {project_root}")
    print(f"📁 Directory src: {src_path}")
    
    try:
        # Setup dell'applicazione Qt
        app = setup_application()

        # Controlla sia sys.argv che l'evento FileOpen
        file_to_open = None
        
        # 1. Controlla sys.argv (funziona da terminale)
        if len(sys.argv) > 1:
            file_to_open = sys.argv[1]
            print(f"🟢 [sys.argv] File da aprire: {file_to_open}")
        
        # 2. Controlla se c'è un file dall'evento macOS (funziona da Finder)
        if app.file_to_open:
            file_to_open = app.file_to_open
            print(f"🟢 [FileOpenEvent] File da aprire: {file_to_open}")

        #test
        #file_to_open = "/Users/tommy/PlantLeaf_dev/audio_tests/venus_scuola_1.paudio"

        # Crea e configura la finestra principale
        print("🏠 Creazione finestra Home...")
        home_window = MainWindowHome()
        app.home_window = home_window  # ✅ Salva il riferimento per eventi futuri

        # Se c'è un file da aprire, caricalo
        if file_to_open:
            #apertura file diretta
            home_window.open_file_action(file_to_open)
        else:
            print("ℹ️ Nessun file da aprire, avvio in modalità vuota")

            # Centra la finestra sullo schermo
            # Dopo aver creato home_window
            from core.layout_manager import LayoutManager
            layout_manager = LayoutManager(home_window.font_manager)
            layout_manager.center_window_on_screen(home_window)
            
            # Mostra la finestra
            home_window.show()
            print("✅ Finestra Home mostrata")
            
            # Informazioni di debug
            print(f"📐 Dimensioni finestra: {home_window.size()}")
            print(f"📍 Posizione finestra: {home_window.pos()}")
            print(f"🎨 Tema corrente: {getattr(home_window, 'current_theme', 'Sconosciuto')}")

        print("🔄 Avvio loop eventi Qt...")
            
        # Avvia il loop degli eventi Qt
        exit_code = app.exec() # Questo blocca l'esecuzione fino alla chiusura della finestra
        
        print(f"🏁 Applicazione terminata con codice: {exit_code}")
        return exit_code
        
    except Exception as e:
        print(f"❌ Errore critico nell'applicazione: {e}")
        import traceback
        traceback.print_exc()
        return 1
    

if __name__ == "__main__":
    print("=" * 60)
    print("🌿 PLANTLEAF - debug")
    print("=" * 60)
    
    # Avvia l'applicazione
    sys.exit(main())