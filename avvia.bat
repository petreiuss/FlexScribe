@echo off
rem Avvia FlexScribe (apre il browser su http://127.0.0.1:8765)
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Prima installazione: creo l'ambiente Python...
  python -m venv .venv || goto :errore
  ".venv\Scripts\python.exe" -m pip install -q --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt || goto :errore
)
if not exist "models\whisper-large-v3-turbo-int8-ov\openvino_encoder_model.xml" (
  echo Scarico il modello Whisper per la scheda grafica, circa 800 MB...
  ".venv\Scripts\python.exe" -c "from huggingface_hub import snapshot_download; snapshot_download('OpenVINO/whisper-large-v3-turbo-int8-ov', local_dir=r'models\whisper-large-v3-turbo-int8-ov')" || goto :errore
)
rem llama.cpp + Gemma for the summaries: downloaded only if missing (about 5 GB the first time)
".venv\Scripts\python.exe" tools\setup_llm.py || echo Attenzione: Gemma non installato, le sintesi non saranno disponibili. Riprova al prossimo avvio.
".venv\Scripts\python.exe" app.py
pause
exit /b

:errore
echo.
echo Installazione non riuscita: controlla la connessione e che sia installato Python 3.12, 3.13 o 3.14.
pause
exit /b 1
