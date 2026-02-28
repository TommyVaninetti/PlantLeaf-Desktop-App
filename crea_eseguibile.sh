echo Programma per creare ESEGUIBILE .EXE con pyinstaller lanciato
echo ------------------------------------------------------

# Rimuovi eventuali build precedenti
rm -rf build dist __pycache__

# Crea l'eseguibile con pyinstaller
#prova prima con il file .spec generato

# chiedi il nome del file .spec
echo Inserisci il nome del file .spec (senza estensione):
read specfile

pyinstaller "$specfile.spec"

echo ------------------------------------------------------
echo ESEGUIBILE creato nella cartella dist/
echo ------------------------------------------------------
echo Premere un tasto per uscire...