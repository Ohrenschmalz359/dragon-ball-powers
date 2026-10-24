@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem ===========================================================================
rem  Dragon Ball Powers - Windows 11 virtual camera (also for WhatsApp Desktop)
rem
rem  Installs the camera "Dragon Ball Powers (Windows 11)": a Media Foundation
rem  media source (built from native\win11_camera by GitHub Actions) that the
rem  Windows camera frame server offers to every app - also to apps such as
rem  WhatsApp Desktop and the Windows Camera app, which ignore older virtual
rem  cameras. The app (key V) sends its picture to it.
rem
rem    [1/3] check for Windows 11 (build 22000 or newer)
rem    [2/3] check the driver file in drivers\win11 (SHA256)
rem    [3/3] as administrator: copy it to Program Files and register the camera
rem
rem  "install_win11_camera.bat --uninstall" removes the camera again. Keep CRLF
rem  line endings and plain ASCII; delayed expansion stays OFF and paths are
rem  always quoted, so folders such as "C:\Games (x86)\A & B" work.
rem ===========================================================================

cd /d "%~dp0" 2>nul || pushd "%~dp0"
set "ROOT=%~dp0"
set "SELF=%~f0"
set "SYS32=%SystemRoot%\System32"
if not defined SystemRoot set "SYS32=C:\Windows\System32"
set "ERRORLEVEL="

set "CAM_NAME=Dragon Ball Powers (Windows 11)"
set "CAM_ARCH=x64"
if /i "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "CAM_ARCH=arm64"
if /i "%PROCESSOR_ARCHITEW6432%"=="ARM64" set "CAM_ARCH=arm64"
set "CAM_SHA_X64=aea89611be2cd6aea69cfa224e7127f68e1685b7b29d26005be8902117a3ca9b"
set "CAM_SHA_ARM64=628f93d2787cfc4ea878f7725035ddba39813096fb42c6bfad259ddf286464d8"
set "CAM_SHA=%CAM_SHA_X64%"
if "%CAM_ARCH%"=="arm64" set "CAM_SHA=%CAM_SHA_ARM64%"
set "CAM_SOURCE=%ROOT%drivers\win11\%CAM_ARCH%\DragonBallPowersCamera.dll"
set "CAM_DIR=%ProgramFiles%\Dragon Ball Powers"
set "CAM_DLL=%ProgramFiles%\Dragon Ball Powers\DragonBallPowersCamera.dll"
set "LOG_DIR=%ProgramData%\Dragon Ball Powers"
if not defined TEMP set "TEMP=%ROOT%logs"
set "HASH_OUT=%TEMP%\dbp-win11cam-%RANDOM%%RANDOM%.sha"

set "ACTION=install"
set "DRY_RUN="
set "NO_PAUSE="
set "RC=0"
if "%DBP_NO_PAUSE%"=="1" set "NO_PAUSE=1"
if /i "%DBP_NO_PAUSE%"=="true" set "NO_PAUSE=1"

:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="--help" goto :usage
if /i "%~1"=="-h" goto :usage
if "%~1"=="/?" goto :usage
if /i "%~1"=="--uninstall" (set "ACTION=uninstall" & shift & goto :parse_args)
if /i "%~1"=="--dry-run" (set "DRY_RUN=1" & shift & goto :parse_args)
if /i "%~1"=="--no-pause" (set "NO_PAUSE=1" & shift & goto :parse_args)
rem Internal options of the elevated copy started by :elevate.
if /i "%~1"=="--elevated-install" goto :elevated_install
if /i "%~1"=="--elevated-uninstall" goto :elevated_uninstall
echo [ERROR] Unknown option "%~1". Run "install_win11_camera.bat --help" to see the options.
exit /b 2

:usage
echo Dragon Ball Powers - Windows 11 virtual camera (also for WhatsApp Desktop)
echo(
echo Usage: install_win11_camera.bat [--uninstall] [--dry-run] [--no-pause] [--help]
echo(
echo Installs the camera "Dragon Ball Powers (Windows 11)" once. Then press V in
echo the app and choose that camera in WhatsApp, Discord, Zoom, Teams, the Windows
echo Camera app ... Needs Windows 11; Windows asks once for administrator permission.
echo(
echo   --uninstall  remove the camera again
echo   --dry-run    only show what would happen
echo   --no-pause   do not wait for a key at the end (also: set DBP_NO_PAUSE=1)
echo(
echo Exit codes: 0 = done, 1 = failed, 2 = invalid option,
echo             3 = administrator permission refused, 4 = not Windows 11
exit /b 0

:args_done
echo ===========================================================================
if "%ACTION%"=="uninstall" goto :header_uninstall
echo  Dragon Ball Powers - install the camera "%CAM_NAME%"
echo ===========================================================================
goto :do_install
:header_uninstall
echo  Dragon Ball Powers - remove the camera "%CAM_NAME%"
echo ===========================================================================
goto :do_uninstall

rem ---------------------------------------------------------------------------
:do_install
set "WIN_BUILD=0"
rem (System32 has no spaces: the command must not start with a quote inside for /f.)
for /f "tokens=3" %%B in ('%SYS32%\reg.exe query "HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion" /v CurrentBuildNumber 2^>nul ^| %SYS32%\find.exe "CurrentBuildNumber"') do set "WIN_BUILD=%%B"
echo [1/3] Windows build %WIN_BUILD% (Windows 11 = 22000 or newer), %CAM_ARCH%
if defined DRY_RUN goto :install_check_file
if %WIN_BUILD% LSS 22000 goto :not_windows11

:install_check_file
echo [2/3] Checking the camera driver file ...
if not exist "%CAM_SOURCE%" goto :missing_file
set "DL_FILE=%CAM_SOURCE%"
set "DL_SHA=%CAM_SHA%"
call :verify
if not "%errorlevel%"=="0" goto :bad_file
echo       OK - drivers\win11\%CAM_ARCH%\DragonBallPowersCamera.dll (SHA256 verified)
if defined DRY_RUN goto :dry_install
echo [3/3] Installing - please confirm the Windows administrator prompt ...
call :elevate --elevated-install
set "RC=%errorlevel%"
if "%RC%"=="3" goto :refused
if not "%RC%"=="0" goto :install_failed
echo(
echo [OK] The camera "%CAM_NAME%" is installed.
echo      Start the app (run.bat), press V, then choose "%CAM_NAME%"
echo      as the camera in WhatsApp, Discord, Zoom, Teams ... Restart call programs
echo      that were already open. To remove it: install_win11_camera.bat --uninstall
goto :end

:dry_install
echo [3/3] [dry-run] as administrator: copy the file to "%CAM_DIR%",
echo       [dry-run]   regsvr32 /s /n /i DragonBallPowersCamera.dll (creates the camera)
echo(
echo [OK] Dry run finished - nothing was copied or registered.
goto :end

rem ---------------------------------------------------------------------------
:do_uninstall
if defined DRY_RUN goto :dry_uninstall
if not exist "%CAM_DLL%" goto :nothing_to_remove
echo Removing the camera - please confirm the Windows administrator prompt ...
call :elevate --elevated-uninstall
set "RC=%errorlevel%"
if "%RC%"=="3" goto :refused
if not "%RC%"=="0" goto :uninstall_failed
echo [OK] The camera "%CAM_NAME%" was removed.
goto :end

:dry_uninstall
echo [dry-run] as administrator: regsvr32 /s /n /u /i DragonBallPowersCamera.dll, then
echo [dry-run]   delete it from "%CAM_DIR%"
echo [OK] Dry run finished - nothing was changed.
goto :end

:nothing_to_remove
echo [OK] Nothing to remove: the camera is not installed.
goto :end

rem ---------------------------------------------------------------------------
:not_windows11
set "RC=4"
echo [ERROR] This camera needs Windows 11 (this PC: build %WIN_BUILD%).
echo         Use install_virtual_camera.bat instead (works in Discord, Zoom, Teams ...).
goto :end
:missing_file
set "RC=1"
echo [ERROR] The driver file is missing: drivers\win11\%CAM_ARCH%\DragonBallPowersCamera.dll
echo         Get the complete project again (git pull or a new download).
goto :end
:bad_file
set "RC=1"
echo [ERROR] The driver file is damaged or not the expected version (SHA256 mismatch).
echo         Get the complete project again (git pull or a new download).
goto :end
:refused
set "RC=3"
echo [ERROR] Administrator permission was not given, so nothing was changed.
goto :end
:install_failed
echo [ERROR] Installing the camera failed (code %RC%). Details may be in
echo         "%LOG_DIR%\win11-camera.log".
set "RC=1"
goto :end
:uninstall_failed
echo [ERROR] Removing the camera failed (code %RC%).
set "RC=1"
goto :end

:end
del /f /q "%HASH_OUT%" >nul 2>&1
echo(
if not defined NO_PAUSE pause
exit /b %RC%

rem ---------------------------------------------------------------------------
rem :verify - errorlevel 0 if SHA256(DL_FILE) == DL_SHA (certutil, spaces removed).
:verify
set "HASH_OK="
"%SYS32%\certutil.exe" -hashfile "%DL_FILE%" SHA256 >"%HASH_OUT%" 2>&1
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
if /i "%HASH_LINE%"=="%DL_SHA%" set "HASH_OK=1"
exit /b 0

rem ---------------------------------------------------------------------------
rem :elevate - run this file again as administrator with argument 1 (one UAC
rem   prompt; the elevated copy runs hidden). Returns its exit code, 3 when the
rem   permission is refused.
rem ---------------------------------------------------------------------------
:elevate
"%SYS32%\fltmc.exe" >nul 2>&1
if "%errorlevel%"=="0" goto :elevate_here
set "DBP_SELF=%SELF%"
set "DBP_SELF_ARG=%~1"
"%SYS32%\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "try { $p = Start-Process -FilePath $env:DBP_SELF -ArgumentList $env:DBP_SELF_ARG -Verb RunAs -WindowStyle Hidden -Wait -PassThru; exit $p.ExitCode } catch { exit 1223 }"
set "ELEVATED_RC=%errorlevel%"
if "%ELEVATED_RC%"=="1223" exit /b 3
exit /b %ELEVATED_RC%
:elevate_here
if /i "%~1"=="--elevated-uninstall" goto :elevated_uninstall
goto :elevated_install

rem :elevated_install / :elevated_uninstall - the administrator part. The log
rem   folder is writable for the camera services (Local Service) and users.
:elevated_install
if not exist "%LOG_DIR%\" mkdir "%LOG_DIR%" 2>nul
"%SYS32%\icacls.exe" "%LOG_DIR%" /grant *S-1-5-19:(OI)(CI)M *S-1-5-32-545:(OI)(CI)M >nul 2>&1
if not exist "%CAM_DIR%\" mkdir "%CAM_DIR%" 2>nul
if exist "%CAM_DLL%" "%SYS32%\regsvr32.exe" /s /n /u /i "%CAM_DLL%"
call :copy_driver
if not "%errorlevel%"=="0" exit /b 5
set "DBP_CAM_DLL=%CAM_DLL%"
"%SYS32%\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -NonInteractive -Command "Unblock-File -LiteralPath $env:DBP_CAM_DLL" >nul 2>&1
"%SYS32%\regsvr32.exe" /s /n /i "%CAM_DLL%"
if not "%errorlevel%"=="0" exit /b 6
exit /b 0

:elevated_uninstall
"%SYS32%\regsvr32.exe" /s /n /u /i "%CAM_DLL%"
set "UNREG_RC=%errorlevel%"
del /f /q "%CAM_DLL%" >nul 2>&1
if exist "%CAM_DLL%" call :stop_camera_services
if exist "%CAM_DLL%" del /f /q "%CAM_DLL%" >nul 2>&1
rmdir "%CAM_DIR%" >nul 2>&1
if not "%UNREG_RC%"=="0" exit /b 4
exit /b 0

rem :copy_driver - copy the DLL; if the camera services still hold the old one,
rem   stop them (they start again on demand) and retry.
:copy_driver
copy /y "%CAM_SOURCE%" "%CAM_DLL%" >nul 2>&1
if not errorlevel 1 exit /b 0
call :stop_camera_services
copy /y "%CAM_SOURCE%" "%CAM_DLL%" >nul 2>&1
if errorlevel 1 exit /b 1
exit /b 0

:stop_camera_services
"%SYS32%\net.exe" stop FrameServerMonitor /y >nul 2>&1
"%SYS32%\net.exe" stop FrameServer /y >nul 2>&1
exit /b 0
