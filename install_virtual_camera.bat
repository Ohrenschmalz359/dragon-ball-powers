@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem ===========================================================================
rem  Dragon Ball Powers - virtual camera for video calls, no OBS needed
rem
rem  Registers the small Unity Capture DirectShow filter (MIT license, from
rem  github.com/schellingb/UnityCapture) as the camera "Dragon Ball Powers".
rem  Discord, Zoom, Teams, browsers ... list it, and the app sends its picture
rem  to it when the virtual camera is on (key V).
rem
rem    [1/2] download + verify (SHA256) the 64-bit and 32-bit filter DLLs into
rem          the folder virtualcam of this project
rem    [2/2] register them with regsvr32 - Windows asks once for administrator
rem          permission (only this step runs as administrator)
rem
rem  "install_virtual_camera.bat --uninstall" removes the camera again. The
rem  DLLs stay registered from this folder: uninstall before moving or deleting
rem  it. Keep CRLF line endings and plain ASCII; delayed expansion stays OFF and
rem  paths are always quoted, so folders such as "C:\Games (x86)\A & B" work.
rem ===========================================================================

cd /d "%~dp0" 2>nul || pushd "%~dp0"
set "ROOT=%~dp0"
set "SELF=%~f0"
set "SYS32=%SystemRoot%\System32"
if not defined SystemRoot set "SYS32=C:\Windows\System32"
set "WOW64=%SystemRoot%\SysWOW64"
set "ERRORLEVEL="

set "VC_NAME=Dragon Ball Powers"
set "VC_DIR=%ROOT%virtualcam"
set "VC_BASE=https://raw.githubusercontent.com/schellingb/UnityCapture/3ed54c325e0ad71afcf4f246c07e5e17b3d7f2d2/Install"
set "VC_DLL64=%ROOT%virtualcam\UnityCaptureFilter64.dll"
set "VC_SHA64=72812f5363d8ecb45632253f8c8c888844b1b62e27616f3c8cc21064ccde25e5"
set "VC_DLL32=%ROOT%virtualcam\UnityCaptureFilter32.dll"
set "VC_SHA32=aa3ebdf03dea7f3aab3dd7b724751f49ed71672256b57c6a19aa6809cabf30ba"
if not defined TEMP set "TEMP=%ROOT%logs"
set "HASH_OUT=%TEMP%\dbp-vcam-%RANDOM%%RANDOM%.sha"

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
rem Internal options of the elevated copy started by :elevate (registry part only).
if /i "%~1"=="--elevated-install" goto :register
if /i "%~1"=="--elevated-uninstall" goto :unregister
echo [ERROR] Unknown option "%~1". Run "install_virtual_camera.bat --help" to see the options.
exit /b 2

:usage
echo Dragon Ball Powers - virtual camera for video calls (no OBS needed)
echo(
echo Usage: install_virtual_camera.bat [--uninstall] [--dry-run] [--no-pause] [--help]
echo(
echo Installs the camera "Dragon Ball Powers" (the Unity Capture DirectShow
echo filter, MIT license) once. Then press V in the app and choose
echo "Dragon Ball Powers" as the camera in Discord, Zoom, Teams, Google Meet ...
echo Windows asks once for administrator permission.
echo(
echo   --uninstall  remove the camera again (before moving or deleting this folder)
echo   --dry-run    only show what would happen
echo   --no-pause   do not wait for a key at the end (also: set DBP_NO_PAUSE=1)
echo(
echo Exit codes: 0 = done, 1 = download or registration failed,
echo             2 = invalid option, 3 = administrator permission refused
exit /b 0

:args_done
echo ===========================================================================
if "%ACTION%"=="uninstall" goto :header_uninstall
echo  Dragon Ball Powers - install the virtual camera "%VC_NAME%"
echo ===========================================================================
goto :do_install
:header_uninstall
echo  Dragon Ball Powers - remove the virtual camera "%VC_NAME%"
echo ===========================================================================
goto :do_uninstall

rem ---------------------------------------------------------------------------
:do_install
echo [1/2] Downloading the camera driver (Unity Capture filter, about 330 KB) ...
if defined DRY_RUN goto :dry_install
if not exist "%VC_DIR%\" mkdir "%VC_DIR%" 2>nul
set "DL_FILE=%VC_DLL64%"
set "DL_URL=%VC_BASE%/UnityCaptureFilter64.dll"
set "DL_SHA=%VC_SHA64%"
call :fetch
if not "%errorlevel%"=="0" goto :download_failed
set "DL_FILE=%VC_DLL32%"
set "DL_URL=%VC_BASE%/UnityCaptureFilter32.dll"
set "DL_SHA=%VC_SHA32%"
call :fetch
if not "%errorlevel%"=="0" goto :download_failed
echo       OK - both files verified (SHA256).
echo [2/2] Registering the camera - please confirm the Windows administrator prompt ...
call :elevate --elevated-install
set "RC=%errorlevel%"
if "%RC%"=="3" goto :refused
if not "%RC%"=="0" goto :register_failed
echo(
echo [OK] The camera "%VC_NAME%" is installed.
echo      Start the app (run.bat), press V, then choose "%VC_NAME%" as the camera
echo      in Discord, Zoom, Teams, Google Meet ... Restart call programs that were
echo      already open. To remove it: install_virtual_camera.bat --uninstall
goto :end

:dry_install
echo       [dry-run] download "%VC_BASE%/UnityCaptureFilter64.dll"
echo       [dry-run]   and "%VC_BASE%/UnityCaptureFilter32.dll"
echo       [dry-run]   into the folder virtualcam, check SHA256 %VC_SHA64%
echo       [dry-run]   and %VC_SHA32%
echo [2/2] [dry-run] as administrator: regsvr32 /s UnityCaptureFilter64.dll and (SysWOW64)
echo       [dry-run]   UnityCaptureFilter32.dll with "/i:UnityCaptureName=%VC_NAME%"
echo(
echo [OK] Dry run finished - nothing was downloaded or registered.
goto :end

rem ---------------------------------------------------------------------------
:do_uninstall
if defined DRY_RUN goto :dry_uninstall
if not exist "%VC_DLL64%" goto :nothing_to_remove
echo Removing the camera - please confirm the Windows administrator prompt ...
call :elevate --elevated-uninstall
set "RC=%errorlevel%"
if "%RC%"=="3" goto :refused
if not "%RC%"=="0" goto :unregister_failed
echo [OK] The camera "%VC_NAME%" was removed. The folder virtualcam can be deleted now.
goto :end

:dry_uninstall
echo [dry-run] as administrator: regsvr32 /u /s UnityCaptureFilter64.dll and (SysWOW64)
echo [dry-run]   UnityCaptureFilter32.dll from the folder virtualcam
echo [OK] Dry run finished - nothing was changed.
goto :end

:nothing_to_remove
echo [OK] Nothing to remove: the camera driver files are not in the folder virtualcam.
goto :end

rem ---------------------------------------------------------------------------
:download_failed
set "RC=1"
echo [ERROR] The download failed. Check the internet connection (behind a proxy:
echo         set HTTPS_PROXY=http://proxy:port in this window first) and try again.
goto :end
:refused
set "RC=3"
echo [ERROR] Administrator permission was not given, so nothing was changed.
echo         Run install_virtual_camera.bat again and confirm the Windows prompt.
goto :end
:register_failed
echo [ERROR] Registering the camera failed (code %RC%). An antivirus program may have
echo         blocked regsvr32; try again, or use OBS Studio's "OBS Virtual Camera".
set "RC=1"
goto :end
:unregister_failed
echo [ERROR] Removing the camera failed (code %RC%).
set "RC=1"
goto :end

:end
del /f /q "%HASH_OUT%" >nul 2>&1
echo(
if not defined NO_PAUSE pause
exit /b %RC%

rem ---------------------------------------------------------------------------
rem :fetch - download DL_URL to DL_FILE (curl.exe first, PowerShell as the
rem   fallback) unless a file with the expected SHA256 is already there, and
rem   verify DL_SHA. Returns errorlevel 0 on success.
rem ---------------------------------------------------------------------------
:fetch
if not exist "%DL_FILE%" goto :fetch_download
call :verify
if "%errorlevel%"=="0" exit /b 0
del /f /q "%DL_FILE%" >nul 2>&1
:fetch_download
del /f /q "%DL_FILE%.part" >nul 2>&1
if not exist "%SYS32%\curl.exe" goto :fetch_powershell
"%SYS32%\curl.exe" --fail --location --silent --show-error --retry 3 --retry-delay 2 --connect-timeout 30 --output "%DL_FILE%.part" "%DL_URL%"
if "%errorlevel%"=="0" goto :fetch_downloaded
del /f /q "%DL_FILE%.part" >nul 2>&1
:fetch_powershell
rem URL and file are passed through the environment: no quoting problems.
set "DBP_DL_URL=%DL_URL%"
set "DBP_DL_FILE=%DL_FILE%.part"
"%SYS32%\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "$ProgressPreference = 'SilentlyContinue'; try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12 } catch { }; try { Invoke-WebRequest -UseBasicParsing -Uri $env:DBP_DL_URL -OutFile $env:DBP_DL_FILE; exit 0 } catch { Write-Host ('      PowerShell download failed: ' + $_.Exception.Message); exit 1 }"
if not "%errorlevel%"=="0" goto :fetch_failed
:fetch_downloaded
if not exist "%DL_FILE%.part" goto :fetch_failed
move /y "%DL_FILE%.part" "%DL_FILE%" >nul
if not "%errorlevel%"=="0" goto :fetch_failed
call :verify
if "%errorlevel%"=="0" exit /b 0
echo [ERROR] A downloaded file is damaged or not the expected version (SHA256 mismatch).
del /f /q "%DL_FILE%" >nul 2>&1
exit /b 1
:fetch_failed
del /f /q "%DL_FILE%.part" >nul 2>&1
exit /b 1

rem :verify - errorlevel 0 if SHA256(DL_FILE) == DL_SHA. certutil prints the
rem   hash on its own line (old Windows versions with spaces between the bytes).
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
rem :elevate - run the registry step (argument 1) as administrator: directly if
rem   this window already has the rights, otherwise through one UAC prompt (the
rem   elevated copy runs hidden and only calls regsvr32). Returns its exit code,
rem   3 when the permission is refused.
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
if /i "%~1"=="--elevated-uninstall" goto :unregister
goto :register

rem :register / :unregister - the administrator part. "/i:UnityCaptureName=..."
rem   gives the camera its name (DllInstall), as in Unity Capture's own installer.
:register
"%SYS32%\regsvr32.exe" /s "%VC_DLL64%" "/i:UnityCaptureName=%VC_NAME%"
if not "%errorlevel%"=="0" exit /b 4
if not exist "%WOW64%\regsvr32.exe" exit /b 0
"%WOW64%\regsvr32.exe" /s "%VC_DLL32%" "/i:UnityCaptureName=%VC_NAME%"
if not "%errorlevel%"=="0" exit /b 5
exit /b 0

:unregister
"%SYS32%\regsvr32.exe" /u /s "%VC_DLL64%"
set "UNREG_RC=%errorlevel%"
if exist "%WOW64%\regsvr32.exe" if exist "%VC_DLL32%" "%WOW64%\regsvr32.exe" /u /s "%VC_DLL32%"
if not "%UNREG_RC%"=="0" exit /b 4
exit /b 0
