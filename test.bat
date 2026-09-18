@echo off
"%~dp0.venv\Scripts\python.exe" -m pytest "%~dp0tests" -v
pause
