#pragma once

#include "framework.h"

// {D61C8D41-AD63-4319-AE6F-EB46E5FEB187} - the COM class of the media source.
extern const GUID CLSID_DbpCamera;

// Name of the camera in every app's camera list.
constexpr wchar_t DBP_CAMERA_NAME[] = L"Dragon Ball Powers (Windows 11)";
constexpr UINT32 DBP_CAMERA_WIDTH = 1280;
constexpr UINT32 DBP_CAMERA_HEIGHT = 720;
constexpr UINT32 DBP_CAMERA_FPS = 30;

// Undocumented frame server attribute: process id of the app that opens the camera.
DEFINE_GUID(MF_FRAMESERVER_CLIENTCONTEXT_CLIENTPID, 0x5f8d322e, 0x0fe4, 0x43e4, 0x9e, 0x50, 0xd8, 0x3e, 0xcd, 0x9f, 0xc2, 0xb8);

// "{D61C8D41-...}" (registry / MFCreateVirtualCamera format).
std::wstring GuidToString(const GUID& guid);

// C++/WinRT answers QueryInterface only for the interfaces listed in implements<>:
// the base interfaces of these derived ones must be added explicitly.
namespace winrt
{
    template<> inline bool is_guid_of<IMFMediaSourceEx>(guid const& id) noexcept
    {
        return is_guid_of<IMFMediaSourceEx, IMFMediaSource, IMFMediaEventGenerator>(id);
    }

    template<> inline bool is_guid_of<IMFMediaStream2>(guid const& id) noexcept
    {
        return is_guid_of<IMFMediaStream2, IMFMediaStream, IMFMediaEventGenerator>(id);
    }

    template<> inline bool is_guid_of<IMFActivate>(guid const& id) noexcept
    {
        return is_guid_of<IMFActivate, IMFAttributes>(id);
    }
}
