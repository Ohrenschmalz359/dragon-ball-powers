#pragma once

#include "Attributes.h"
#include "Camera.h"
#include "MediaStream.h"

// The media source the Windows camera frame server loads for "Dragon Ball Powers
// (Windows 11)". Adapted from VCamSample's MediaSource (one stream, no camera controls).
struct MediaSource : winrt::implements<MediaSource, CBaseAttributes<IMFAttributes>, IMFMediaSourceEx, IMFGetService, IKsControl, IMFSampleAllocatorControl>
{
public:
    // IMFMediaEventGenerator
    STDMETHOD(BeginGetEvent)(IMFAsyncCallback* callback, IUnknown* state) override;
    STDMETHOD(EndGetEvent)(IMFAsyncResult* result, IMFMediaEvent** event) override;
    STDMETHOD(GetEvent)(DWORD flags, IMFMediaEvent** event) override;
    STDMETHOD(QueueEvent)(MediaEventType type, REFGUID extendedType, HRESULT status, const PROPVARIANT* value) override;

    // IMFMediaSource
    STDMETHOD(CreatePresentationDescriptor)(IMFPresentationDescriptor** descriptor) override;
    STDMETHOD(GetCharacteristics)(DWORD* characteristics) override;
    STDMETHOD(Pause)() override;
    STDMETHOD(Shutdown)() override;
    STDMETHOD(Start)(IMFPresentationDescriptor* descriptor, const GUID* timeFormat, const PROPVARIANT* startPosition) override;
    STDMETHOD(Stop)() override;

    // IMFMediaSourceEx
    STDMETHOD(GetSourceAttributes)(IMFAttributes** attributes) override;
    STDMETHOD(GetStreamAttributes)(DWORD streamIdentifier, IMFAttributes** attributes) override;
    STDMETHOD(SetD3DManager)(IUnknown* manager) override;

    // IMFGetService
    STDMETHOD(GetService)(REFGUID service, REFIID riid, LPVOID* object) override;

    // IMFSampleAllocatorControl
    STDMETHOD(SetDefaultAllocator)(DWORD outputStreamId, IUnknown* allocator) override;
    STDMETHOD(GetAllocatorUsage)(DWORD outputStreamId, DWORD* inputStreamId, MFSampleAllocatorUsage* usage) override;

    // IKsControl
    STDMETHOD_(NTSTATUS, KsProperty)(PKSPROPERTY property, ULONG length, LPVOID data, ULONG dataLength, ULONG* returned) override;
    STDMETHOD_(NTSTATUS, KsMethod)(PKSMETHOD method, ULONG length, LPVOID data, ULONG dataLength, ULONG* returned) override;
    STDMETHOD_(NTSTATUS, KsEvent)(PKSEVENT event, ULONG length, LPVOID data, ULONG dataLength, ULONG* returned) override;

public:
    HRESULT Initialize(IMFAttributes* attributes);

private:
    int StreamIndexById(DWORD id);

    winrt::slim_mutex _lock;
    std::vector<winrt::com_ptr<MediaStream>> _streams;
    wil::com_ptr_nothrow<IMFMediaEventQueue> _queue;
    wil::com_ptr_nothrow<IMFPresentationDescriptor> _descriptor;
};
