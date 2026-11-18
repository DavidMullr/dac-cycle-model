@echo off
setlocal
python -B "%~dp0render_guides.py"
exit /b %ERRORLEVEL%
