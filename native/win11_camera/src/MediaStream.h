#pragma once

#include "Attributes.h"
#include "Camera.h"
#include "SharedFrame.h"

// The camera's one video stream: 1280x720 @ 30 fps as RGB32 or NV12, filled with the
// frames of the Dragon Ball Powers app (or a dark placeholder). CPU only: a Direct3D
// manager offered by the frame server is ignored (the app's frames are in system memory).
struct MediaStream : winrt::implements<MediaStream, CBaseAttributes<IMFAttributes>, IMFMediaStream2, IKsControl>
{
public:
    // IMFMediaEventGenerator
    STDMETHOD(BeginGetEvent)(IMFAsyncCallback* callback, IUnknown* state) override;
    STDMETHOD(EndGetEvent)(IMFAsyncResult* result, IMFMediaEvent** event) override;
    STDMETHOD(GetEvent)(DWORD flags, IMFMediaEvent** event) override;
    STDMETHOD(QueueEvent)(MediaEventType type, REFGUID extendedType, HRESULT status, const PROPVARIANT* value) override;

    // IMFMediaStream
    STDMETHOD(GetMediaSource)(IMFMediaSource** source) override;
    STDMETHOD(GetStreamDescriptor)(IMFStreamDescriptor** descriptor) override;
    STDMETHOD(RequestSample)(IUnknown* token) override;

    // IMFMediaStream2
    STDMETHOD(SetStreamState)(MF_STREAM_STATE value) override;
    STDMETHOD(GetStreamState)(MF_STREAM_STATE* value) override;

    // IKsControl
    STDMETHOD_(NTSTATUS, KsProperty)(PKSPROPERTY property, ULONG length, LPVOID data, ULONG dataLength, ULONG* returned) override;
    STDMETHOD_(NTSTATUS, KsMethod)(PKSMETHOD method, ULONG length, LPVOID data, ULONG dataLength, ULONG* returned) override;
    STDMETHOD_(NTSTATUS, KsEvent)(PKSEVENT event, ULONG length, LPVOID data, ULONG dataLength, ULONG* returned) override;

public:
    HRESULT Initialize(IMFMediaSource* source, DWORD index);
    HRESULT SetAllocator(IUnknown* allocator);
    HRESULT Start(IMFMediaType* type);
    HRESULT Stop();
    void Shutdown();

private:
    HRESULT CurrentMediaType(IMFMediaType** type);
    HRESULT EnsureAllocator();
    HRESULT CreateSample(IMFSample** sample);
    HRESULT FillSample(IMFSample* sample);
    void Pace();

    winrt::slim_mutex _lock;
    MF_STREAM_STATE _state = MF_STREAM_STATE_STOPPED;
    GUID _format = GUID_NULL;
    UINT32 _width = DBP_CAMERA_WIDTH;
    UINT32 _height = DBP_CAMERA_HEIGHT;
    DWORD _index = 0;
    MFTIME _lastSampleTime = 0;
    ULONGLONG _frames = 0;
    SharedFrameReader _reader;
    std::vector<BYTE> _bgra;            // scratch picture for the NV12 conversion
    wil::com_ptr_nothrow<IMFStreamDescriptor> _descriptor;
    wil::com_ptr_nothrow<IMFMediaEventQueue> _queue;
    wil::com_ptr_nothrow<IMFMediaSource> _source;
    wil::com_ptr_nothrow<IMFVideoSampleAllocatorEx> _allocator;
    bool _allocatorReady = false;
};
