@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem ===========================================================================
rem  Builds the Dragon Ball Powers Windows 11 camera with the Visual Studio C++
rem  compiler: out\DragonBallPowersCamera.dll (the media source) and
rem  out\CameraTest.exe (its test program). Run it in a Visual Studio
rem  "Developer Command Prompt" for x64 or ARM64. GitHub builds it the same way
rem  (.github\workflows\win11-camera.yml).
rem
rem  Needs WIL (header only) in deps\wil:
rem    git clone --depth 1 --branch v1.0.250325.1 https://github.com/microsoft/wil deps\wil
rem  C++/WinRT comes with the Windows SDK.
rem ===========================================================================
cd /d "%~dp0"
set "WIL_DIR=%~dp0deps\wil\include"
if not exist "%WIL_DIR%\wil\result.h" goto :no_wil
if not defined WindowsSdkDir goto :no_vs
set "CPPWINRT_DIR=%WindowsSdkDir%Include\%WindowsSDKVersion%cppwinrt"
if not exist out mkdir out

cl /nologo /std:c++20 /EHsc /W4 /O2 /MT /Zi /permissive- /DNDEBUG /DUNICODE /D_UNICODE /I "%WIL_DIR%" /I "%CPPWINRT_DIR%" /LD /Fo:out\ /Fd:out\DragonBallPowersCamera.pdb src\*.cpp /Fe:out\DragonBallPowersCamera.dll /link /DEF:src\DragonBallPowersCamera.def /DEBUG /OPT:REF /OPT:ICF
if errorlevel 1 exit /b 1
cl /nologo /std:c++20 /EHsc /W4 /O2 /MT /Zi /permissive- /DNDEBUG /DUNICODE /D_UNICODE /I "%WIL_DIR%" /I "%CPPWINRT_DIR%" /Fo:out\ /Fd:out\CameraTest.pdb test\CameraTest.cpp /Fe:out\CameraTest.exe
if errorlevel 1 exit /b 1
echo Built out\DragonBallPowersCamera.dll and out\CameraTest.exe
exit /b 0

:no_wil
echo [ERROR] WIL is missing. Get it with:
echo         git clone --depth 1 --branch v1.0.250325.1 https://github.com/microsoft/wil deps\wil
exit /b 1

:no_vs
echo [ERROR] Run this in a Visual Studio Developer Command Prompt (x64 or ARM64).
exit /b 1
