@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem ===========================================================================
rem  Dragon Ball Powers - start the app
rem
rem    run.bat                     start normally
rem    run.bat --renderer cpu      use the CPU renderer (if the GPU path fails)
rem    run.bat --help              list all options of the app
rem
rem  Needs a one-time "setup.bat" first. Works from any current folder, with
rem  spaces or special characters in the path and when double-clicked in
rem  Explorer. DBP_NO_PAUSE=1 disables the "press any key" pauses.
rem  Keep this file CRLF + plain ASCII (see the notes in setup.bat).
rem ===========================================================================

cd /d "%~dp0" 2>nul || pushd "%~dp0"
set "ROOT=%~dp0"
set "VENV_PY=%ROOT%.venv\Scripts\python.exe"
rem A user-defined ERRORLEVEL variable would hide the real exit codes, and a
rem PYTHONHOME left behind by another tool breaks every virtual environment.
set "ERRORLEVEL="
set "PYTHONHOME="
set "NO_PAUSE="
if "%DBP_NO_PAUSE%"=="1" set "NO_PAUSE=1"
if /i "%DBP_NO_PAUSE%"=="true" set "NO_PAUSE=1"

rem --------------------------------------------- 1. environment from setup.bat
if exist "%VENV_PY%" goto :check_packages
echo(
echo [ERROR] Dragon Ball Powers is not set up yet.
echo         The Python environment .venv\Scripts\python.exe was not found in
for %%I in ("%ROOT%.") do echo         %%~fI
echo(
echo         Run setup.bat first (double-click it). It installs everything the
echo         app needs into this folder; an internet connection is needed once.
set "RC=1"
goto :finish

rem ------------------------------------------------- 2. fast dependency check
rem find_spec only locates the packages (no heavy imports), so this check
rem takes a fraction of a second.
:check_packages
"%VENV_PY%" -c "import importlib.util, sys; missing = [name for name in ('mediapipe', 'cv2', 'pygame', 'moderngl', 'numpy') if importlib.util.find_spec(name) is None]; print('[ERROR] Python packages missing in .venv: ' + ', '.join(missing)) if missing else None; sys.exit(3 if missing else 0)"
set "RC=%errorlevel%"
if "%RC%"=="0" goto :check_models
if "%RC%"=="3" goto :packages_missing
rem 1260 / 4551: Windows refused to start python.exe (App Control policy).
if "%RC%"=="1260" goto :python_blocked
if "%RC%"=="4551" goto :python_blocked
echo(
echo [ERROR] The Python environment in .venv does not work (exit code %RC%).
echo         It may have been created by a Python that was uninstalled or moved.
echo         Recreate it with:  setup.bat --force
set "RC=1"
goto :finish

:packages_missing
echo         Run setup.bat to install them (it repairs the existing .venv).
set "RC=1"
goto :finish

:python_blocked
echo(
echo [ERROR] Windows blocked .venv\Scripts\python.exe (App Control / group policy).
echo         This PC only runs programs approved by its administrator. Ask your
echo         IT department to allow Python 3.12, or use a PC without this policy.
set "RC=1"
goto :finish

rem ------------------------------------------------------------ 3. models
rem The app still starts without them; tracking then reports the problem.
:check_models
set "MODELS_DIR=%ROOT%assets\models"
if defined DBP_MODELS_DIR set "MODELS_DIR=%DBP_MODELS_DIR%"
set "MISSING_MODELS="
if not exist "%MODELS_DIR%\hand_landmarker.task" set "MISSING_MODELS= hand_landmarker.task"
if not exist "%MODELS_DIR%\pose_landmarker_lite.task" set "MISSING_MODELS=%MISSING_MODELS% pose_landmarker_lite.task"
if not defined MISSING_MODELS goto :launch
echo [WARNING] Model file(s) missing:%MISSING_MODELS%
for %%I in ("%MODELS_DIR%") do echo           in %%~fI
echo           Hand/body tracking will not work until they are downloaded.
echo           Fix: run setup.bat, or  .venv\Scripts\python.exe scripts\download_models.py
echo           Starting the app anyway ...
echo(

rem ------------------------------------------------------------- 4. start
:launch
rem No trailing ";" when PYTHONPATH was empty: an empty entry would put the
rem current folder on sys.path.
if defined PYTHONPATH goto :pythonpath_append
set "PYTHONPATH=%ROOT%src"
goto :activate
:pythonpath_append
set "PYTHONPATH=%ROOT%src;%PYTHONPATH%"
:activate
rem Relative path (the current folder is the project folder): "call" would
rem expand percent signs and double carets in the full path a second time.
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
echo Starting Dragon Ball Powers ...
"%VENV_PY%" -m dragon_ball_powers %*
set "RC=%errorlevel%"
if "%RC%"=="0" exit /b 0
rem Ctrl+C in the console is a normal way to quit: the app returns 130 for it
rem (KeyboardInterrupt), a hard interrupt ends with STATUS_CONTROL_C_EXIT.
if "%RC%"=="130" exit /b %RC%
if "%RC%"=="-1073741510" exit /b %RC%
echo(
echo [ERROR] Dragon Ball Powers exited with code %RC%. See the messages above and logs\dragon_ball_powers.log
echo         If the window stayed black or closed at once, try:  run.bat --renderer cpu
goto :finish

:finish
if defined NO_PAUSE exit /b %RC%
echo(
pause
exit /b %RC%
