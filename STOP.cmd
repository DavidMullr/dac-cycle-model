@echo off
setlocal
set "ROOT=%~dp0model\"
set "PYTHON_EXE="
if defined CARBON_TEAM_PYTHON if exist "%CARBON_TEAM_PYTHON%" set "PYTHON_EXE=%CARBON_TEAM_PYTHON%"
if not defined PYTHON_EXE if exist "%ROOT%.venv\Scripts\python.exe" set "PYTHON_EXE=%ROOT%.venv\Scripts\python.exe"
pushd "%ROOT%"
if defined PYTHON_EXE (
  "%PYTHON_EXE%" -m launcher stop %*
) else (
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3.12 -m launcher stop %*
  ) else (
    python -m launcher stop %*
  )
)
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
