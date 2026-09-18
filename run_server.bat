@echo off
echo Server wordt gestart... browser opent automatisch.
echo Sluit dit venster om de server te stoppen.
"%~dp0.venv\Scripts\python.exe" "%~dp0webapp\app.py"
pause
