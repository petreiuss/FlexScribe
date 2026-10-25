@echo off
rem Crea sul Desktop il collegamento "FlexScribe" (con icona) che avvia l'app.
rem La finestra del server si apre ridotta a icona: chiuderla spegne l'app.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$app = (Get-Location).Path;" ^
  "$lnk = Join-Path ([Environment]::GetFolderPath('Desktop')) 'FlexScribe.lnk';" ^
  "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk);" ^
  "$s.TargetPath = Join-Path $app 'avvia.bat';" ^
  "$s.WorkingDirectory = $app;" ^
  "$s.IconLocation = (Join-Path $app 'static\flexscribe.ico') + ',0';" ^
  "$s.WindowStyle = 7;" ^
  "$s.Description = 'FlexScribe - trascrizioni e sintesi delle sedute';" ^
  "$s.Save();" ^
  "Write-Host ('Collegamento creato: ' + $lnk)"
pause
