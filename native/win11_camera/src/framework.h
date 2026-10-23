// Dragon Ball Powers - Windows 11 virtual camera (a Media Foundation media source).
// Adapted from VCamSample by Simon Mourier (MIT license, see ../LICENSE-VCamSample.txt).
#pragma once

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include <windows.h>
#include <sddl.h>
#include <strsafe.h>
#include <initguid.h>
#include <propvarutil.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfvirtualcamera.h>
#include <mferror.h>
#include <mfcaptureengine.h>
#include <ks.h>
#include <ksproxy.h>
#include <ksmedia.h>
#include <uuids.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

// WIL (header only, https://github.com/microsoft/wil) and C++/WinRT (Windows SDK).
#include <wil/com.h>
#include <wil/resource.h>
#include <wil/result.h>
#include <winrt/base.h>

#pragma comment(lib, "advapi32")
#pragma comment(lib, "mfplat")
#pragma comment(lib, "mfuuid")
#pragma comment(lib, "mfsensorgroup")
#pragma comment(lib, "ole32")
#pragma comment(lib, "windowsapp")
