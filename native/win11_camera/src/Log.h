#pragma once

#include "framework.h"

// Appends a line to %ProgramData%\Dragon Ball Powers\win11-camera.log (if the folder
// exists - install_win11_camera.bat creates it). The media source runs inside several
// processes (frame server, frame server monitor, the app that uses the camera), so this
// file is the one place to see what happened. At most a few hundred lines per process.
void DbpLog(_Printf_format_string_ PCWSTR format, ...);
