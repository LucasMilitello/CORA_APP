@echo off
setlocal
cd /d "%~dp0"
python -m pip install -r requirements_robotizado.txt
exit /b %ERRORLEVEL%
