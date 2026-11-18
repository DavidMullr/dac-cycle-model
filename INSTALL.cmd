@echo off
setlocal
set "ROOT=%~dp0"
set "PYTHON_EXE="
if defined CARBON_TEAM_PYTHON if exist "%CARBON_TEAM_PYTHON%" set "PYTHON_EXE=%CARBON_TEAM_PYTHON%"
if defined PYTHON_EXE goto run
where py >nul 2>nul
if not errorlevel 1 (
  py -3.12 -c "pass" >nul 2>nul
  if errorlevel 1 (
    set "PYTHON_WORDS=py -3"
  ) else (
    set "PYTHON_WORDS=py -3.12"
  )
  goto run
)
where python >nul 2>nul
if not errorlevel 1 (
  set "PYTHON_WORDS=python"
  goto run
)
echo Carbon Team setup error: CPython 3.12 or newer was not found. 1>&2
exit /b 1

:run
if defined PYTHON_EXE (
  set "PYTHON_WORDS="%PYTHON_EXE%""
)
pushd "%ROOT%"
%PYTHON_WORDS% -B tools\bootstrap.py %*
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
