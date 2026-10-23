#include "MediaStream.h"
#include "Log.h"

namespace
{
    HRESULT CreateVideoType(REFGUID subtype, IMFMediaType** result)
    {
        wil::com_ptr_nothrow<IMFMediaType> type;
        RETURN_IF_FAILED(MFCreateMediaType(&type));
        const bool rgb = subtype == MFVideoFormat_RGB32;
        RETURN_IF_FAILED(type->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video));
        RETURN_IF_FAILED(type->SetGUID(MF_MT_SUBTYPE, subtype));
        RETURN_IF_FAILED(MFSetAttributeSize(type.get(), MF_MT_FRAME_SIZE, DBP_CAMERA_WIDTH, DBP_CAMERA_HEIGHT));
        // Positive stride: top row first. For NV12 it is the stride of the Y plane.
        RETURN_IF_FAILED(type->SetUINT32(MF_MT_DEFAULT_STRIDE, rgb ? DBP_CAMERA_WIDTH * 4 : DBP_CAMERA_WIDTH));
        RETURN_IF_FAILED(type->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive));
        RETURN_IF_FAILED(type->SetUINT32(MF_MT_ALL_SAMPLES_INDEPENDENT, TRUE));
        RETURN_IF_FAILED(MFSetAttributeRatio(type.get(), MF_MT_FRAME_RATE, DBP_CAMERA_FPS, 1));
        RETURN_IF_FAILED(MFSetAttributeRatio(type.get(), MF_MT_PIXEL_ASPECT_RATIO, 1, 1));
        RETURN_IF_FAILED(type->SetUINT32(MF_MT_AVG_BITRATE, DBP_CAMERA_WIDTH * DBP_CAMERA_HEIGHT * (rgb ? 32 : 12) * DBP_CAMERA_FPS));
        *result = type.detach();
        return S_OK;
    }

    PCWSTR FormatName(REFGUID format)
    {
        return format == MFVideoFormat_NV12 ? L"NV12" : format == MFVideoFormat_RGB32 ? L"RGB32" : L"?";
    }
}

HRESULT MediaStream::Initialize(IMFMediaSource* source, DWORD index)
{
    RETURN_HR_IF_NULL(E_POINTER, source);
    _source = source;
    _index = index;

    RETURN_IF_FAILED(SetGUID(MF_DEVICESTREAM_STREAM_CATEGORY, PINNAME_VIDEO_CAPTURE));
    RETURN_IF_FAILED(SetUINT32(MF_DEVICESTREAM_STREAM_ID, index));
    RETURN_IF_FAILED(SetUINT32(MF_DEVICESTREAM_FRAMESERVER_SHARED, 1));
    RETURN_IF_FAILED(SetUINT32(MF_DEVICESTREAM_ATTRIBUTE_FRAMESOURCE_TYPES, MFFrameSourceTypes_Color));
    RETURN_IF_FAILED(MFCreateEventQueue(&_queue));

    // RGB32 first as in VCamSample; NV12 is what most clients (browsers, Teams ...) pick.
    wil::com_ptr_nothrow<IMFMediaType> rgb;
    wil::com_ptr_nothrow<IMFMediaType> nv12;
    RETURN_IF_FAILED(CreateVideoType(MFVideoFormat_RGB32, &rgb));
    RETURN_IF_FAILED(CreateVideoType(MFVideoFormat_NV12, &nv12));
    IMFMediaType* types[] = { rgb.get(), nv12.get() };
    RETURN_IF_FAILED(MFCreateStreamDescriptor(_index, ARRAYSIZE(types), types, &_descriptor));

    wil::com_ptr_nothrow<IMFMediaTypeHandler> handler;
    RETURN_IF_FAILED(_descriptor->GetMediaTypeHandler(&handler));
    RETURN_IF_FAILED(handler->SetCurrentMediaType(types[0]));
    return S_OK;
}

HRESULT MediaStream::CurrentMediaType(IMFMediaType** type)
{
    RETURN_HR_IF(MF_E_SHUTDOWN, !_descriptor);
    wil::com_ptr_nothrow<IMFMediaTypeHandler> handler;
    RETURN_IF_FAILED(_descriptor->GetMediaTypeHandler(&handler));
    RETURN_IF_FAILED(handler->GetCurrentMediaType(type));
    return S_OK;
}

HRESULT MediaStream::SetAllocator(IUnknown* allocator)
{
    RETURN_HR_IF_NULL(E_POINTER, allocator);
    winrt::slim_lock_guard lock(_lock);
    _allocator.reset();
    _allocatorReady = false;
    RETURN_IF_FAILED(allocator->QueryInterface(IID_PPV_ARGS(&_allocator)));
    DbpLog(L"Frame server sample allocator received");
    return S_OK;
}

HRESULT MediaStream::EnsureAllocator()
{
    if (!_allocator || _allocatorReady)
        return S_OK;
    wil::com_ptr_nothrow<IMFMediaType> type;
    RETURN_IF_FAILED(CurrentMediaType(&type));
    RETURN_IF_FAILED(_allocator->InitializeSampleAllocator(10, type.get()));
    _allocatorReady = true;
    return S_OK;
}

HRESULT MediaStream::Start(IMFMediaType* type)
{
    winrt::slim_lock_guard lock(_lock);
    RETURN_HR_IF(MF_E_SHUTDOWN, !_queue);

    wil::com_ptr_nothrow<IMFMediaType> current;
    if (type)
        current = type;
    else
        RETURN_IF_FAILED(CurrentMediaType(&current));

    _format = GUID_NULL;
    current->GetGUID(MF_MT_SUBTYPE, &_format);
    if (_format != MFVideoFormat_NV12)
        _format = MFVideoFormat_RGB32;
    UINT32 width = 0, height = 0;
    if (SUCCEEDED(MFGetAttributeSize(current.get(), MF_MT_FRAME_SIZE, &width, &height)) && width && height && !(width & 1) && !(height & 1))
    {
        _width = width;
        _height = height;
    }

    if (_allocator)
    {
        if (_allocatorReady)
            LOG_IF_FAILED(_allocator->UninitializeSampleAllocator());
        _allocatorReady = false;
        RETURN_IF_FAILED(_allocator->InitializeSampleAllocator(10, current.get()));
        _allocatorReady = true;
    }

    RETURN_IF_FAILED(_queue->QueueEventParamVar(MEStreamStarted, GUID_NULL, S_OK, nullptr));
    _state = MF_STREAM_STATE_RUNNING;
    _frames = 0;
    DbpLog(L"Stream started: %s %ux%u (%s)", FormatName(_format), _width, _height,
        _allocator ? L"frame server allocator" : L"own samples");
    return S_OK;
}

HRESULT MediaStream::Stop()
{
    winrt::slim_lock_guard lock(_lock);
    RETURN_HR_IF(MF_E_SHUTDOWN, !_queue);

    if (_allocator && _allocatorReady)
        LOG_IF_FAILED(_allocator->UninitializeSampleAllocator());
    _allocatorReady = false;
    RETURN_IF_FAILED(_queue->QueueEventParamVar(MEStreamStopped, GUID_NULL, S_OK, nullptr));
    _state = MF_STREAM_STATE_STOPPED;
    DbpLog(L"Stream stopped after %llu frames", _frames);
    return S_OK;
}

void MediaStream::Shutdown()
{
    winrt::slim_lock_guard lock(_lock);
    if (_queue)
    {
        LOG_IF_FAILED(_queue->Shutdown());
        _queue.reset();
    }
    if (_allocator && _allocatorReady)
        LOG_IF_FAILED(_allocator->UninitializeSampleAllocator());
    _allocatorReady = false;
    _allocator.reset();
    _descriptor.reset();
    _source.reset();
    _attributes.reset();
}

void MediaStream::Pace()
{
    // A real camera delivers a frame every 1/30 s; asked in a tight loop we would burn a CPU core.
    constexpr MFTIME interval = 10'000'000 / DBP_CAMERA_FPS;
    auto now = MFGetSystemTime();
    auto due = _lastSampleTime + interval;
    if (_lastSampleTime && now < due)
    {
        auto waitMs = (DWORD)((due - now) / 10'000);
        if (waitMs > 0 && waitMs <= 40)
            Sleep(waitMs);
    }
    _lastSampleTime = MFGetSystemTime();
}

HRESULT MediaStream::CreateSample(IMFSample** result)
{
    if (_allocator)
    {
        RETURN_IF_FAILED(EnsureAllocator());
        RETURN_IF_FAILED(_allocator->AllocateSample(result));
        return S_OK;
    }

    // No frame server around (e.g. a source reader in the same process): own samples.
    wil::com_ptr_nothrow<IMFSample> sample;
    wil::com_ptr_nothrow<IMFMediaBuffer> buffer;
    RETURN_IF_FAILED(MFCreateSample(&sample));
    RETURN_IF_FAILED(MFCreate2DMediaBuffer(_width, _height, _format.Data1, FALSE, &buffer));
    RETURN_IF_FAILED(sample->AddBuffer(buffer.get()));
    *result = sample.detach();
    return S_OK;
}

HRESULT MediaStream::FillSample(IMFSample* sample)
{
    wil::com_ptr_nothrow<IMFMediaBuffer> buffer;
    RETURN_IF_FAILED(sample->GetBufferByIndex(0, &buffer));
    wil::com_ptr_nothrow<IMF2DBuffer2> buffer2d;
    RETURN_IF_FAILED(buffer->QueryInterface(IID_PPV_ARGS(&buffer2d)));

    BYTE* scanline = nullptr;
    BYTE* start = nullptr;
    LONG pitch = 0;
    DWORD length = 0;
    RETURN_IF_FAILED(buffer2d->Lock2DSize(MF2DBuffer_LockFlags_Write, &scanline, &pitch, &start, &length));
    auto unlock = wil::scope_exit([&] { buffer2d->Unlock2D(); });

    if (_format == MFVideoFormat_NV12)
    {
        _bgra.resize((SIZE_T)_width * _height * 4);
        if (!_reader.Read(_bgra.data(), (LONG)(_width * 4), _width, _height))
            FillPlaceholder(_bgra.data(), (LONG)(_width * 4), _width, _height);
        RETURN_IF_FAILED(Bgra32ToNv12(_bgra.data(), (LONG)(_width * 4), _width, _height, scanline, pitch));
    }
    else if (!_reader.Read(scanline, pitch, _width, _height))
    {
        FillPlaceholder(scanline, pitch, _width, _height);
    }
    unlock.reset();
    RETURN_IF_FAILED(buffer->SetCurrentLength(length));
    return S_OK;
}

// IMFMediaStream
STDMETHODIMP MediaStream::RequestSample(IUnknown* token)
{
    winrt::slim_lock_guard lock(_lock);
    RETURN_HR_IF(MF_E_SHUTDOWN, !_queue);

    Pace();
    wil::com_ptr_nothrow<IMFSample> sample;
    RETURN_IF_FAILED(CreateSample(&sample));
    RETURN_IF_FAILED(sample->SetSampleTime(MFGetSystemTime()));
    RETURN_IF_FAILED(sample->SetSampleDuration(10'000'000 / DBP_CAMERA_FPS));
    RETURN_IF_FAILED(FillSample(sample.get()));
    if (token)
        RETURN_IF_FAILED(sample->SetUnknown(MFSampleExtension_Token, token));
    RETURN_IF_FAILED(_queue->QueueEventParamUnk(MEMediaSample, GUID_NULL, S_OK, sample.get()));
    if (++_frames == 1)
        DbpLog(L"First frame delivered");
    return S_OK;
}

STDMETHODIMP MediaStream::GetMediaSource(IMFMediaSource** source)
{
    RETURN_HR_IF_NULL(E_POINTER, source);
    *source = nullptr;
    RETURN_HR_IF(MF_E_SHUTDOWN, !_source);
    RETURN_IF_FAILED(_source.copy_to(source));
    return S_OK;
}

STDMETHODIMP MediaStream::GetStreamDescriptor(IMFStreamDescriptor** descriptor)
{
    RETURN_HR_IF_NULL(E_POINTER, descriptor);
    *descriptor = nullptr;
    RETURN_HR_IF(MF_E_SHUTDOWN, !_descriptor);
    RETURN_IF_FAILED(_descriptor.copy_to(descriptor));
    return S_OK;
}

// IMFMediaStream2
STDMETHODIMP MediaStream::SetStreamState(MF_STREAM_STATE value)
{
    // Not under _lock: Stop() takes it. Like VCamSample, only "stopped" does real work;
    // samples for a re-started stream come from a lazily re-initialized allocator.
    if (value == _state)
        return S_OK;
    switch (value)
    {
    case MF_STREAM_STATE_STOPPED:
        return Stop();
    case MF_STREAM_STATE_PAUSED:
    case MF_STREAM_STATE_RUNNING:
        _state = value;
        return S_OK;
    default:
        RETURN_HR(MF_E_INVALID_STATE_TRANSITION);
    }
}

STDMETHODIMP MediaStream::GetStreamState(MF_STREAM_STATE* value)
{
    RETURN_HR_IF_NULL(E_POINTER, value);
    *value = _state;
    return S_OK;
}

// IMFMediaEventGenerator
STDMETHODIMP MediaStream::BeginGetEvent(IMFAsyncCallback* callback, IUnknown* state)
{
    winrt::slim_lock_guard lock(_lock);
    RETURN_HR_IF(MF_E_SHUTDOWN, !_queue);
    RETURN_IF_FAILED(_queue->BeginGetEvent(callback, state));
    return S_OK;
}

STDMETHODIMP MediaStream::EndGetEvent(IMFAsyncResult* result, IMFMediaEvent** event)
{
    RETURN_HR_IF_NULL(E_POINTER, event);
    *event = nullptr;
    winrt::slim_lock_guard lock(_lock);
    RETURN_HR_IF(MF_E_SHUTDOWN, !_queue);
    RETURN_IF_FAILED(_queue->EndGetEvent(result, event));
    return S_OK;
}

STDMETHODIMP MediaStream::GetEvent(DWORD flags, IMFMediaEvent** event)
{
    RETURN_HR_IF_NULL(E_POINTER, event);
    *event = nullptr;
    wil::com_ptr_nothrow<IMFMediaEventQueue> queue;
    {
        winrt::slim_lock_guard lock(_lock);
        RETURN_HR_IF(MF_E_SHUTDOWN, !_queue);
        queue = _queue;
    }
    // Outside the lock: a blocking GetEvent must not stop RequestSample from queueing.
    RETURN_IF_FAILED(queue->GetEvent(flags, event));
    return S_OK;
}

STDMETHODIMP MediaStream::QueueEvent(MediaEventType type, REFGUID extendedType, HRESULT status, const PROPVARIANT* value)
{
    winrt::slim_lock_guard lock(_lock);
    RETURN_HR_IF(MF_E_SHUTDOWN, !_queue);
    RETURN_IF_FAILED(_queue->QueueEventParamVar(type, extendedType, status, value));
    return S_OK;
}

// IKsControl: no camera controls (exposure, zoom ...) are offered.
STDMETHODIMP_(NTSTATUS) MediaStream::KsProperty(PKSPROPERTY property, ULONG, LPVOID, ULONG, ULONG* returned)
{
    RETURN_HR_IF_NULL(E_POINTER, property);
    RETURN_HR_IF_NULL(E_POINTER, returned);
    return HRESULT_FROM_WIN32(ERROR_SET_NOT_FOUND);
}

STDMETHODIMP_(NTSTATUS) MediaStream::KsMethod(PKSMETHOD method, ULONG, LPVOID, ULONG, ULONG* returned)
{
    RETURN_HR_IF_NULL(E_POINTER, method);
    RETURN_HR_IF_NULL(E_POINTER, returned);
    return HRESULT_FROM_WIN32(ERROR_SET_NOT_FOUND);
}

STDMETHODIMP_(NTSTATUS) MediaStream::KsEvent(PKSEVENT, ULONG, LPVOID, ULONG, ULONG* returned)
{
    RETURN_HR_IF_NULL(E_POINTER, returned);
    return HRESULT_FROM_WIN32(ERROR_SET_NOT_FOUND);
}
