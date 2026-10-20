@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem ===========================================================================
rem  Dragon Ball Powers - one-time setup for Windows 10/11 (64-bit)
rem
rem    [1/6] find Python 3.11 or 3.12 (64-bit) - or download a private copy
rem          into .python (python-build-standalone, verified by SHA256)
rem    [2/6] create the virtual environment .venv
rem    [3/6] update pip
rem    [4/6] pip install -r requirements.txt
rem    [5/6] download + verify the MediaPipe models (scripts\download_models.py)
rem    [6/6] self-check: python -m dragon_ball_powers --check
rem
rem  "setup.bat --help" lists the options and exit codes. Running it again is
rem  safe: a healthy .venv is reused and only missing pieces are installed.
rem
rem  Notes for editing this file: keep CRLF line endings and plain ASCII.
rem  Delayed expansion stays OFF so "!" in folder names is harmless. Paths are
rem  always quoted, and no message echoes a raw path inside a ( ) block, so
rem  folders such as "C:\Games (x86)\A & B" work.
rem ===========================================================================

cd /d "%~dp0" 2>nul || pushd "%~dp0"
set "ROOT=%~dp0"

set "VENV_DIR=%ROOT%.venv"
set "VENV_PY=%ROOT%.venv\Scripts\python.exe"
set "LOCAL_PY_DIR=%ROOT%.python"
set "LOCAL_PY=%ROOT%.python\python.exe"
set "REQ_FILE=%ROOT%requirements.txt"
set "REQ_DEV_FILE=%ROOT%requirements-dev.txt"
set "MODELS_SCRIPT=%ROOT%scripts\download_models.py"
set "SYS32=%SystemRoot%\System32"
if not defined SystemRoot set "SYS32=C:\Windows\System32"

rem Private Python fallback: python-build-standalone, the CPython build used by
rem uv. "%%2B" is the URL-encoded "+" of the release name (doubled for batch).
set "PBS_VERSION=3.12.15+20261013"
set "PBS_URL=https://github.com/astral-sh/python-build-standalone/releases/download/20261013/cpython-3.12.15%%2B20261013-x86_64-pc-windows-msvc-install_only.tar.gz"
set "PBS_SHA256=bbc902d4d187ac4914915bf538196a064b49109a33a104965b5164ced5b80a1e"

if not defined TEMP set "TEMP=%ROOT%logs"
set "PBS_ARCHIVE=%TEMP%\dbp-cpython-3.12.15-20261013-win64.tar.gz"
set "PROBE_OUT=%TEMP%\dbp-setup-%RANDOM%%RANDOM%.out"
set "PROBE_ERR=%TEMP%\dbp-setup-%RANDOM%%RANDOM%.err"
set "HASH_OUT=%TEMP%\dbp-setup-%RANDOM%%RANDOM%.sha"

rem Prints "Python X.Y.Z (64-bit, win-amd64)" and exits with
rem   0 = usable, 3 = wrong version/architecture, 4 = is a virtual environment
rem   (rejected as a base interpreter), 5 = no venv/ensurepip (embeddable Python).
set "PROBE_CODE=import os, struct, sys, sysconfig, importlib.util as u; v = sys.version_info; b = struct.calcsize('P') * 8; p = sysconfig.get_platform(); e = sys.prefix != sys.base_prefix; base = os.environ.get('DBP_REJECT_VENV') == '1'; print('Python {}.{}.{} ({}-bit, {})'.format(v[0], v[1], v[2], b, p) + (', virtual environment' if e else '')); sys.exit(3 if not (v[:2] in ((3, 11), (3, 12)) and b == 64 and p == 'win-amd64') else 4 if base and e else 5 if base and (u.find_spec('venv') is None or u.find_spec('ensurepip') is None) else 0)"

rem Environment hygiene: a PYTHONHOME/PYTHONPATH left behind by another tool
rem breaks venv creation and pip; a user-defined ERRORLEVEL variable would hide
rem the real exit codes. pip must never wait for input (e.g. proxy passwords).
set "ERRORLEVEL="
set "PYTHONHOME="
set "PYTHONPATH="
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PIP_NO_INPUT=1"
rem A global pip setting "user = true" (pip.ini or PIP_USER=1) makes every pip
rem install inside a virtual environment fail; environment variables win.
set "PIP_USER=0"
rem The Python install manager (py.exe of python.org 3.14+) may offer to
rem install missing runtimes, and the older py.exe starts winget when
rem PYLAUNCHER_ALLOW_INSTALL is set; our read-only probes must never do that.
set "PYTHON_MANAGER_AUTOMATIC_INSTALL=false"
set "PYLAUNCHER_ALLOW_INSTALL="

set "DRY_RUN="
set "FORCE="
set "DEV="
set "NO_PAUSE="
if "%DBP_NO_PAUSE%"=="1" set "NO_PAUSE=1"
if /i "%DBP_NO_PAUSE%"=="true" set "NO_PAUSE=1"

:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="--help" goto :usage
if /i "%~1"=="-h" goto :usage
if "%~1"=="/?" goto :usage
if /i "%~1"=="--dry-run" (set "DRY_RUN=1" & shift & goto :parse_args)
if /i "%~1"=="--force" (set "FORCE=1" & shift & goto :parse_args)
if /i "%~1"=="--dev" (set "DEV=1" & shift & goto :parse_args)
if /i "%~1"=="--no-pause" (set "NO_PAUSE=1" & shift & goto :parse_args)
echo [ERROR] Unknown option "%~1". Run "setup.bat --help" to see the options.
exit /b 2

:usage
echo Dragon Ball Powers - setup
echo(
echo Usage: setup.bat [--dry-run] [--force] [--dev] [--no-pause] [--help]
echo(
echo Prepares everything the app needs, inside this folder only:
echo   [1/6] find Python 3.11 or 3.12 (64-bit): an existing .venv, "py -3.12",
echo         "py -3.11", "python" on PATH or .python\python.exe. If none is
echo         found, a private Python %PBS_VERSION% (python-build-standalone)
echo         is downloaded, verified (SHA256) and unpacked into .python
echo   [2/6] create the virtual environment .venv
echo   [3/6] update pip
echo   [4/6] install the pinned packages from requirements.txt
echo   [5/6] download and verify the MediaPipe models (assets\models)
echo   [6/6] run the app self-check (python -m dragon_ball_powers --check)
echo Running it again is safe: a working .venv is reused and repaired.
echo(
echo Options:
echo   --dry-run    show what would be done; nothing is downloaded or changed
echo   --force      delete and recreate the virtual environment .venv
echo   --dev        also install the developer tools from requirements-dev.txt
echo                (pytest, used by the VS Code test explorer)
echo   --no-pause   do not wait for a key press at the end
echo                (the same as setting the environment variable DBP_NO_PAUSE=1)
echo   --help       show this help
echo(
echo Exit codes:
echo   0  success (also when the self-check only reported warnings)
echo   1  no usable 64-bit Python 3.11/3.12 (missing, download failed, or
echo      blocked by Windows App Control / group policy)
echo   2  unknown command line option
echo   3  the virtual environment .venv could not be created or replaced
echo   4  pip or the package installation failed
echo   5  the MediaPipe models could not be downloaded or verified
echo(
echo Behind a proxy? Run "set HTTPS_PROXY=http://proxy.example.com:8080" in the
echo same window first (used by curl, pip and the model download).
exit /b 0

:args_done
echo(
echo ==============================================================
echo   Dragon Ball Powers - setup
echo ==============================================================
for %%I in ("%ROOT%.") do echo   Folder: %%~fI
if defined DRY_RUN echo   DRY RUN: only showing what would be done. Nothing is changed.
if defined FORCE echo   --force: the virtual environment .venv will be recreated.
if defined DEV echo   --dev: the developer tools will be installed as well.
set "DRY_PROBLEM="
set "PY_CMD="
set "PY_CALL="
set "CAND_EXE="
set "REUSE_VENV="

rem =========================================================== [1/6] Python
echo(
echo [1/6] Looking for Python 3.11 or 3.12 (64-bit) ...
if defined FORCE goto :find_base_python
if not exist "%VENV_DIR%" goto :find_base_python
if not exist "%VENV_PY%" goto :venv_unusable
set CAND="%VENV_PY%"
set "CAND_EXE="
set "CAND_LABEL=existing .venv"
set "DBP_REJECT_VENV="
call :probe
if not "%PROBE_RC%"=="0" goto :venv_unusable
set "REUSE_VENV=1"
set "VENV_DESC=%PROBE_DESC%"
goto :step2

:venv_unusable
echo       The existing .venv cannot be used; it will be recreated.

:find_base_python
set "DBP_REJECT_VENV=1"
set "CAND=py -3.12"
set "CAND_EXE=py"
set "CAND_LABEL=py -3.12"
call :probe
if "%PROBE_RC%"=="0" goto :python_found
rem 9009 = the "py" launcher itself is not installed, 112 = it knows no Python
rem at all: no need to ask for 3.11.
if "%PROBE_RC%"=="9009" goto :try_path_python
if "%PROBE_RC%"=="112" goto :try_path_python
set "CAND=py -3.11"
set "CAND_EXE=py"
set "CAND_LABEL=py -3.11"
call :probe
if "%PROBE_RC%"=="0" goto :python_found

:try_path_python
set "CAND=python"
set "CAND_EXE=python"
set "CAND_LABEL=python on PATH"
call :probe
if "%PROBE_RC%"=="0" goto :python_found
if not exist "%LOCAL_PY%" goto :get_local_python
set CAND="%LOCAL_PY%"
set "CAND_EXE="
set "CAND_LABEL=.python\python.exe"
call :probe
if "%PROBE_RC%"=="0" goto :python_found
if defined PROBE_BLOCKED goto :python_blocked
echo       The private Python in .python is damaged; it will be downloaded again.

:get_local_python
if defined DRY_RUN goto :get_local_python_dry
echo(
echo       No compatible Python found. Downloading a private Python %PBS_VERSION%
echo       for this project (python-build-standalone). It is unpacked into the
echo       folder .python - nothing is installed system-wide.
call :download_pbs
if not "%errorlevel%"=="0" goto :download_failed
call :extract_pbs
if not "%errorlevel%"=="0" goto :extract_failed
set CAND="%LOCAL_PY%"
set "CAND_EXE="
set "CAND_LABEL=.python\python.exe"
call :probe
if "%PROBE_RC%"=="0" goto :python_found
if defined PROBE_BLOCKED goto :python_blocked
echo(
echo [ERROR] The downloaded Python does not start (exit code %PROBE_RC%):
for /f "usebackq delims=" %%L in ("%PROBE_ERR%") do echo         %%L
goto :no_python_help

:get_local_python_dry
echo       [dry-run] No compatible Python found. A real run would:
if exist "%LOCAL_PY_DIR%" echo       [dry-run]   delete the damaged .python folder
echo       [dry-run]   download %PBS_URL%
echo       [dry-run]   (curl.exe, PowerShell as fallback) to "%PBS_ARCHIVE%"
echo       [dry-run]   check its SHA256 = %PBS_SHA256%
echo       [dry-run]   unpack it with tar.exe into .python and test-run .python\python.exe
set PY_CMD="%LOCAL_PY%"
goto :step2

:python_found
set PY_CMD=%CAND%
if defined CAND_EXE set "PY_CALL=1"
echo       Using %CAND_LABEL% (%PROBE_DESC%)
goto :step2

:python_blocked
call :explain_blocked
if defined DRY_RUN goto :python_blocked_dry
set "RC=1"
goto :finish

:python_blocked_dry
echo       [dry-run] A real run would stop here with exit code 1.
if not defined DRY_PROBLEM set "DRY_PROBLEM=step 1 - no usable Python"
set PY_CMD="%LOCAL_PY%"
goto :step2

:download_failed
echo(
echo [ERROR] Could not download the private Python runtime.
echo         Check the internet connection. Behind a proxy, run this first in the
echo         same window:  set HTTPS_PROXY=http://proxy.example.com:8080
goto :no_python_help

:extract_failed
echo(
echo [ERROR] Could not unpack the private Python runtime into .python
echo         Make sure this folder is writable and has about 200 MB of free space.
goto :no_python_help

:no_python_help
echo(
echo         Alternatively install Python 3.12 (64-bit) yourself from
echo         https://www.python.org/downloads/windows/ and run setup.bat again.
set "RC=1"
goto :finish

rem ====================================================== [2/6] virtual env
:step2
echo(
if not defined REUSE_VENV goto :step2_create
echo [2/6] Reusing the existing virtual environment .venv (%VENV_DESC%)
goto :step3

:step2_create
echo [2/6] Creating the virtual environment .venv ...
if not exist "%VENV_DIR%" goto :step2_venv
set "RM_TARGET=%VENV_DIR%"
call :remove_project_dir
if not "%errorlevel%"=="0" goto :venv_failed
:step2_venv
if defined DRY_RUN goto :step2_dry
if defined PY_CALL goto :step2_venv_call
%PY_CMD% -m venv "%VENV_DIR%"
goto :step2_venv_check
:step2_venv_call
rem A command name such as "py -3.12" is started with "call" (see :probe). The
rem relative folder name keeps "call" from expanding percent signs in the path.
call %PY_CMD% -m venv ".venv"
:step2_venv_check
if not "%errorlevel%"=="0" goto :venv_failed
if not exist "%VENV_PY%" goto :venv_failed
set CAND="%VENV_PY%"
set "CAND_EXE="
set "CAND_LABEL=new .venv"
set "DBP_REJECT_VENV="
call :probe
if not "%PROBE_RC%"=="0" goto :venv_failed
goto :step3

:step2_dry
echo       [dry-run] would run: %PY_CMD% -m venv ".venv"
goto :step3

:venv_failed
echo(
echo [ERROR] The virtual environment .venv could not be created.
echo         Make sure this folder is writable, close programs that use .venv
echo         (the app, VS Code, terminals) and run setup.bat again. If an
echo         antivirus tool quarantined python.exe, allow it and retry.
set "RC=3"
goto :finish

rem ============================================================ [3/6] pip
:step3
echo(
echo [3/6] Updating pip (the Python package installer) ...
if defined DRY_RUN goto :step3_dry
"%VENV_PY%" -m pip --version >nul 2>&1
if "%errorlevel%"=="0" goto :step3_upgrade
echo       pip is missing in .venv - installing it with ensurepip ...
"%VENV_PY%" -m ensurepip --upgrade --default-pip
if not "%errorlevel%"=="0" goto :pip_missing
:step3_upgrade
"%VENV_PY%" -m pip install --upgrade pip
if not "%errorlevel%"=="0" echo [WARNING] pip could not be updated - continuing with the installed version.
goto :step4

:step3_dry
echo       [dry-run] would run: ".venv\Scripts\python.exe" -m pip install --upgrade pip
goto :step4

:pip_missing
echo(
echo [ERROR] pip is missing in .venv and could not be installed.
echo         Recreate the environment with:  setup.bat --force
set "RC=4"
goto :finish

rem ======================================================= [4/6] packages
:step4
echo(
set "REQ_TO_INSTALL=%REQ_FILE%"
set "REQ_LABEL=requirements.txt"
if defined DEV set "REQ_TO_INSTALL=%REQ_DEV_FILE%"
if defined DEV set "REQ_LABEL=requirements-dev.txt"
echo [4/6] Installing the Python packages from %REQ_LABEL% ...
echo       (mediapipe, opencv, numpy, pygame, moderngl - this can take a few minutes)
if not exist "%REQ_TO_INSTALL%" goto :req_missing
if defined DRY_RUN goto :step4_dry
"%VENV_PY%" -m pip install -r "%REQ_TO_INSTALL%"
if not "%errorlevel%"=="0" goto :pip_failed
goto :step5

:step4_dry
echo       [dry-run] would run: ".venv\Scripts\python.exe" -m pip install -r %REQ_LABEL%
goto :step5

:req_missing
echo [ERROR] %REQ_LABEL% is missing - the project folder is incomplete.
echo         Copy or download the complete project again.
if defined DRY_RUN goto :req_missing_dry
set "RC=4"
goto :finish

:req_missing_dry
if not defined DRY_PROBLEM set "DRY_PROBLEM=step 4 - %REQ_LABEL% is missing"
goto :step5

:pip_failed
echo(
echo [ERROR] Installing the Python packages failed (see the pip messages above).
echo         - Check the internet connection. Behind a proxy, run this first:
echo             set HTTPS_PROXY=http://proxy.example.com:8080
echo         - Close Dragon Ball Powers, VS Code and terminals that use .venv.
echo         - A very long folder path can exceed the Windows path length limit:
echo           move the project to a short path such as C:\DragonBallPowers.
echo         - Still failing? Rebuild the environment:  setup.bat --force
set "RC=4"
goto :finish

rem ========================================================= [5/6] models
:step5
echo(
echo [5/6] Downloading and verifying the MediaPipe models ...
if not exist "%MODELS_SCRIPT%" goto :models_script_missing
if defined DRY_RUN goto :step5_dry
"%VENV_PY%" "%MODELS_SCRIPT%"
if not "%errorlevel%"=="0" goto :models_failed
goto :step6

:step5_dry
echo       [dry-run] would run: ".venv\Scripts\python.exe" scripts\download_models.py
goto :step6

:models_script_missing
echo [ERROR] scripts\download_models.py is missing - the project folder is incomplete.
if defined DRY_RUN goto :models_script_missing_dry
set "RC=5"
goto :finish

:models_script_missing_dry
if not defined DRY_PROBLEM set "DRY_PROBLEM=step 5 - scripts\download_models.py is missing"
goto :step6

:models_failed
echo(
echo [ERROR] The MediaPipe models could not be downloaded or verified (see above).
echo         Check the internet connection (proxy: set HTTPS_PROXY=...) and run
echo         setup.bat again, or fetch only the models with:
echo           .venv\Scripts\python.exe scripts\download_models.py
set "RC=5"
goto :finish

rem ===================================================== [6/6] self-check
:step6
echo(
echo [6/6] Running the self-check ...
set "PYTHONPATH=%ROOT%src"
set "CHECK_WARN="
if defined DRY_RUN goto :step6_dry
"%VENV_PY%" -m dragon_ball_powers --check
if not "%errorlevel%"=="0" set "CHECK_WARN=1"
goto :success

:step6_dry
echo       [dry-run] would run: ".venv\Scripts\python.exe" -m dragon_ball_powers --check
echo       [dry-run]            (with PYTHONPATH set to the src folder)
goto :dry_summary

rem ================================================================ done
:success
echo(
echo ==============================================================
if defined CHECK_WARN goto :success_with_warnings
echo   Setup complete!
goto :success_banner
:success_with_warnings
echo   [WARNING] Setup finished, but the self-check reported problems.
echo   Read the messages above; the app may still start.
:success_banner
echo(
echo   Start the app with run.bat
echo   (double-click it in Explorer, or type run.bat in this folder).
echo(
echo   VS Code: open this folder (File - Open Folder). The interpreter
echo   .venv\Scripts\python.exe is preselected; press F5 and pick
echo   "Dragon Ball Powers". For the test explorer run setup.bat --dev once.
echo ==============================================================
set "RC=0"
goto :finish

:dry_summary
echo(
if defined DRY_PROBLEM echo Dry run finished: a real run would stop at %DRY_PROBLEM% - see above.
if not defined DRY_PROBLEM echo Dry run finished: a real run would perform the steps listed above.
echo Nothing was downloaded, installed or changed.
set "RC=0"
goto :finish

:finish
del /f /q "%PROBE_OUT%" "%PROBE_ERR%" "%HASH_OUT%" >nul 2>&1
if defined NO_PAUSE exit /b %RC%
echo(
pause
exit /b %RC%

rem ===========================================================================
rem  Subroutines (called with "call :name"; they communicate through
rem  variables instead of arguments because "call" would expand percent signs in
rem  paths and URLs a second time).
rem ===========================================================================

rem ---------------------------------------------------------------------------
rem :probe - run the candidate interpreter CAND (command prefix, paths quoted)
rem   with PROBE_CODE and report the result.
rem   out: PROBE_RC (exit code), PROBE_DESC (first output line),
rem        PROBE_BLOCKED=1 if Windows refused to start it (App Control),
rem        PROBE_STUB=1 for the Microsoft Store "python" placeholder.
rem ---------------------------------------------------------------------------
:probe
set "PROBE_DESC="
set "PROBE_BLOCKED="
set "PROBE_STUB="
del /f /q "%PROBE_OUT%" "%PROBE_ERR%" >nul 2>&1
if not defined CAND_EXE goto :probe_run_path
if not exist "%SYS32%\where.exe" goto :probe_run_path
rem A command name ("py", "python") can be a batch file shim, e.g. the
rem python.bat of pyenv-win: started without "call", cmd.exe would continue in
rem the shim and never return here. "call" reports a missing command as exit
rem code 1 instead of 9009, so where.exe (same search as cmd.exe) checks first.
"%SYS32%\where.exe" /q %CAND_EXE% >nul 2>&1
if errorlevel 1 goto :probe_not_on_path
call %CAND% -I -c "%PROBE_CODE%" 1>"%PROBE_OUT%" 2>"%PROBE_ERR%" <nul
goto :probe_ran
:probe_not_on_path
set "PROBE_RC=9009"
goto :probe_missing
:probe_run_path
%CAND% -I -c "%PROBE_CODE%" 1>"%PROBE_OUT%" 2>"%PROBE_ERR%" <nul
:probe_ran
set "PROBE_RC=%errorlevel%"
if exist "%PROBE_OUT%" set /p PROBE_DESC=<"%PROBE_OUT%"
if "%PROBE_RC%"=="0" goto :probe_ok
if "%PROBE_RC%"=="3" goto :probe_unsupported
if "%PROBE_RC%"=="4" goto :probe_is_venv
if "%PROBE_RC%"=="5" goto :probe_no_venv_module
rem 1260 = ERROR_ACCESS_DISABLED_BY_POLICY ("blocked by group policy"),
rem 4551 = ERROR_SYSTEM_INTEGRITY_POLICY_VIOLATION (App Control / WDAC).
if "%PROBE_RC%"=="1260" set "PROBE_BLOCKED=1"
if "%PROBE_RC%"=="4551" set "PROBE_BLOCKED=1"
"%SYS32%\findstr.exe" /i /c:"group policy" /c:"Gruppenrichtlinie" /c:"App Control" /c:"Application Control" /c:"blocked by your" "%PROBE_ERR%" >nul 2>&1 && set "PROBE_BLOCKED=1"
if defined PROBE_BLOCKED goto :probe_blocked
"%SYS32%\findstr.exe" /i /c:"Microsoft Store" /c:"Python was not found" "%PROBE_ERR%" >nul 2>&1 && set "PROBE_STUB=1"
if defined PROBE_STUB goto :probe_stub
if "%PROBE_RC%"=="9009" goto :probe_missing
rem 103 = the py launcher / venv launcher found no matching Python installation,
rem 112 = the py launcher found no Python installation at all.
if "%PROBE_RC%"=="103" goto :probe_not_installed
if "%PROBE_RC%"=="112" goto :probe_not_installed
echo       - %CAND_LABEL%: does not start (exit code %PROBE_RC%)
exit /b 0
:probe_ok
echo       - %CAND_LABEL%: %PROBE_DESC% - OK
exit /b 0
:probe_unsupported
echo       - %CAND_LABEL%: %PROBE_DESC% - not supported (needs 64-bit Python 3.11 or 3.12)
exit /b 0
:probe_is_venv
echo       - %CAND_LABEL%: %PROBE_DESC% - skipped (a virtual environment, not an installation)
exit /b 0
:probe_no_venv_module
echo       - %CAND_LABEL%: %PROBE_DESC% - skipped (no venv/ensurepip module, e.g. embeddable Python)
exit /b 0
:probe_blocked
echo       - %CAND_LABEL%: BLOCKED by Windows (App Control / group policy)
exit /b 0
:probe_stub
echo       - %CAND_LABEL%: only the Microsoft Store placeholder ("App execution alias"), not a real Python
exit /b 0
:probe_missing
echo       - %CAND_LABEL%: not found
exit /b 0
:probe_not_installed
echo       - %CAND_LABEL%: the required Python is not installed (exit code %PROBE_RC%)
exit /b 0

rem ---------------------------------------------------------------------------
rem :explain_blocked - Windows refused to run CAND_LABEL (company policy).
rem ---------------------------------------------------------------------------
:explain_blocked
echo(
echo [ERROR] Windows refused to start %CAND_LABEL%:
for /f "usebackq delims=" %%L in ("%PROBE_ERR%") do echo           %%L
echo(
echo         This PC only runs programs approved by its administrator (Windows
echo         App Control / AppLocker policy - typical for company or school PCs),
echo         so the private Python of this project cannot run here, and no other
echo         Python 3.11/3.12 (64-bit) was found. What you can do:
echo           1. Ask your IT department to allow Python 3.12 (64-bit), or
echo           2. install Python 3.12 (64-bit) from https://www.python.org/downloads/windows/
echo              if this PC allows it, then run setup.bat again.
echo         On a private PC without such a policy setup.bat works as it is.
exit /b 0

rem ---------------------------------------------------------------------------
rem :download_pbs - download PBS_URL to PBS_ARCHIVE and verify PBS_SHA256.
rem   curl.exe (part of Windows 10 1803+) first, PowerShell as the fallback
rem   (curl.exe can be missing or blocked; PowerShell also uses the Windows
rem   proxy settings). An archive left over from an earlier run is reused if
rem   its checksum matches. Returns errorlevel 0 on success.
rem ---------------------------------------------------------------------------
:download_pbs
if not exist "%PBS_ARCHIVE%" goto :download_pbs_fetch
call :verify_pbs_archive
if not "%errorlevel%"=="0" goto :download_pbs_stale
echo       Using the archive downloaded earlier.
exit /b 0
:download_pbs_stale
del /f /q "%PBS_ARCHIVE%" >nul 2>&1
:download_pbs_fetch
del /f /q "%PBS_ARCHIVE%.part" >nul 2>&1
if not exist "%SYS32%\curl.exe" goto :download_pbs_powershell
echo       Downloading with curl.exe ...
"%SYS32%\curl.exe" --fail --location --retry 3 --retry-delay 2 --connect-timeout 30 --progress-bar --output "%PBS_ARCHIVE%.part" "%PBS_URL%"
if "%errorlevel%"=="0" goto :download_pbs_downloaded
echo       curl.exe did not succeed (exit code %errorlevel%) - trying PowerShell ...
del /f /q "%PBS_ARCHIVE%.part" >nul 2>&1
:download_pbs_powershell
echo       Downloading with PowerShell (no progress display, please wait) ...
rem URL and file are passed through the environment: no quoting problems.
set "DBP_DL_URL=%PBS_URL%"
set "DBP_DL_FILE=%PBS_ARCHIVE%.part"
"%SYS32%\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "$ProgressPreference = 'SilentlyContinue'; try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12 } catch { }; try { Invoke-WebRequest -UseBasicParsing -Uri $env:DBP_DL_URL -OutFile $env:DBP_DL_FILE; exit 0 } catch { Write-Host ('      PowerShell download failed: ' + $_.Exception.Message); exit 1 }"
if not "%errorlevel%"=="0" goto :download_pbs_failed
:download_pbs_downloaded
if not exist "%PBS_ARCHIVE%.part" goto :download_pbs_failed
move /y "%PBS_ARCHIVE%.part" "%PBS_ARCHIVE%" >nul
if not "%errorlevel%"=="0" goto :download_pbs_failed
call :verify_pbs_archive
if not "%errorlevel%"=="0" goto :download_pbs_bad_hash
exit /b 0
:download_pbs_bad_hash
echo [ERROR] The downloaded file is damaged or not the expected release
echo         (SHA256 mismatch). It was deleted; run setup.bat again to retry.
del /f /q "%PBS_ARCHIVE%" >nul 2>&1
exit /b 1
:download_pbs_failed
del /f /q "%PBS_ARCHIVE%.part" >nul 2>&1
exit /b 1

rem ---------------------------------------------------------------------------
rem :verify_pbs_archive - errorlevel 0 if SHA256(PBS_ARCHIVE) == PBS_SHA256.
rem   certutil prints the hash on its own line; the other lines are localized,
rem   and old Windows versions put spaces between the bytes. So every output
rem   line is compared with spaces removed, case-insensitively.
rem ---------------------------------------------------------------------------
:verify_pbs_archive
set "HASH_OK="
"%SYS32%\certutil.exe" -hashfile "%PBS_ARCHIVE%" SHA256 >"%HASH_OUT%" 2>&1
if not "%errorlevel%"=="0" exit /b 1
for /f "usebackq delims=" %%H in ("%HASH_OUT%") do (
    set "HASH_LINE=%%H"
    call :check_hash_line
)
if defined HASH_OK exit /b 0
exit /b 1

:check_hash_line
if not defined HASH_LINE exit /b 0
set "HASH_LINE=%HASH_LINE: =%"
if /i "%HASH_LINE%"=="%PBS_SHA256%" set "HASH_OK=1"
exit /b 0

rem ---------------------------------------------------------------------------
rem :extract_pbs - unpack PBS_ARCHIVE into .python (top-level "python/" folder
rem   stripped). Uses the tar.exe of Windows explicitly: a GNU tar from Git for
rem   Windows earlier in PATH would read "C:\..." as a remote host name.
rem ---------------------------------------------------------------------------
:extract_pbs
if not exist "%LOCAL_PY_DIR%" goto :extract_pbs_mkdir
set "RM_TARGET=%LOCAL_PY_DIR%"
call :remove_project_dir
if not "%errorlevel%"=="0" exit /b 1
:extract_pbs_mkdir
mkdir "%LOCAL_PY_DIR%"
if not exist "%LOCAL_PY_DIR%" exit /b 1
echo       Unpacking into .python ...
"%SYS32%\tar.exe" -xzf "%PBS_ARCHIVE%" -C "%LOCAL_PY_DIR%" --strip-components=1
if not "%errorlevel%"=="0" exit /b 1
if not exist "%LOCAL_PY%" exit /b 1
del /f /q "%PBS_ARCHIVE%" >nul 2>&1
exit /b 0

rem ---------------------------------------------------------------------------
rem :remove_project_dir - recursively delete RM_TARGET, but only if it is the
rem   .venv or .python folder directly inside this project and a real folder
rem   (not a junction/symlink that could point somewhere else).
rem   With --dry-run it only reports. Returns errorlevel 0 on success.
rem ---------------------------------------------------------------------------
:remove_project_dir
set "RM_PARENT="
set "RM_NAME="
set "RM_ATTR="
for %%I in ("%RM_TARGET%") do set "RM_PARENT=%%~dpI"
for %%I in ("%RM_TARGET%") do set "RM_NAME=%%~nxI"
for %%I in ("%RM_TARGET%") do set "RM_ATTR=%%~aI"
if /i not "%RM_PARENT%"=="%ROOT%" goto :remove_refused
if /i "%RM_NAME%"==".venv" goto :remove_name_ok
if /i "%RM_NAME%"==".python" goto :remove_name_ok
goto :remove_refused
:remove_name_ok
if not defined RM_ATTR exit /b 0
if /i not "%RM_ATTR:~0,1%"=="d" goto :remove_refused
if not "%RM_ATTR:l=%"=="%RM_ATTR%" goto :remove_refused
if defined DRY_RUN goto :remove_dry
echo       Deleting the old %RM_NAME% folder ...
rmdir /s /q "%RM_TARGET%"
if exist "%RM_TARGET%" goto :remove_failed
exit /b 0
:remove_dry
echo       [dry-run] would delete the folder %RM_NAME% (it is inside the project)
exit /b 0
:remove_failed
echo [ERROR] %RM_NAME% could not be deleted completely. Close the app, VS Code
echo         and terminals that use it, then run setup.bat again.
exit /b 1
:remove_refused
echo [ERROR] Refusing to delete "%RM_TARGET%" automatically: it is not a
echo         normal folder directly inside this project (junction, link or file?).
echo         Delete or rename it yourself, then run setup.bat again.
exit /b 1
