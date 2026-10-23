#include "MediaSource.h"
#include "MediaStream.h"
#include "Log.h"

HRESULT MediaSource::Initialize(IMFAttributes* attributes)
{
    if (attributes)
        RETURN_IF_FAILED(attributes->CopyAllItems(this));

    // Camera profiles (as in VCamSample): the legacy one (<= 30 fps) and a high frame rate one.
    wil::com_ptr_nothrow<IMFSensorProfileCollection> collection;
    RETURN_IF_FAILED(MFCreateSensorProfileCollection(&collection));
    DWORD streamId = 0;
    wil::com_ptr_nothrow<IMFSensorProfile> profile;
    RETURN_IF_FAILED(MFCreateSensorProfile(KSCAMERAPROFILE_Legacy, 0, nullptr, &profile));
    RETURN_IF_FAILED(profile->AddProfileFilter(streamId, L"((RES==;FRT<=30,1;SUT==))"));
    RETURN_IF_FAILED(collection->AddProfile(profile.get()));
    RETURN_IF_FAILED(MFCreateSensorProfile(KSCAMERAPROFILE_HighFrameRate, 0, nullptr, &profile));
    RETURN_IF_FAILED(profile->AddProfileFilter(streamId, L"((RES==;FRT>=60,1;SUT==))"));
    RETURN_IF_FAILED(collection->AddProfile(profile.get()));
    RETURN_IF_FAILED(SetUnknown(MF_DEVICEMFT_SENSORPROFILE_COLLECTION, collection.get()));

    auto stream = winrt::make_self<MediaStream>();
    RETURN_IF_FAILED(stream->Initialize(this, 0));
    _streams.push_back(stream);

    std::vector<IMFStreamDescriptor*> descriptors;
    std::vector<wil::com_ptr_nothrow<IMFStreamDescriptor>> owners;
    for (auto& s : _streams)
    {
        wil::com_ptr_nothrow<IMFStreamDescriptor> descriptor;
        RETURN_IF_FAILED(s->GetStreamDescriptor(&descriptor));
        descriptors.push_back(descriptor.get());
        owners.push_back(descriptor);
    }
    RETURN_IF_FAILED(MFCreatePresentationDescriptor((DWORD)descriptors.size(), descriptors.data(), &_descriptor));
    RETURN_IF_FAILED(MFCreateEventQueue(&_queue));
    return S_OK;
}

int MediaSource::StreamIndexById(DWORD id)
{
    for (size_t i = 0; i < _streams.size(); i++)
    {
        wil::com_ptr_nothrow<IMFStreamDescriptor> descriptor;
        DWORD sid = 0;
        if (SUCCEEDED(_streams[i]->GetStreamDescriptor(&descriptor)) && SUCCEEDED(descriptor->GetStreamIdentifier(&sid)) && sid == id)
            return (int)i;
    }
    return -1;
}

// IMFMediaEventGenerator
STDMETHODIMP MediaSource::BeginGetEvent(IMFAsyncCallback* callback, IUnknown* state)
{
    winrt::slim_lock_guard lock(_lock);
    RETURN_HR_IF(MF_E_SHUTDOWN, !_queue);
    RETURN_IF_FAILED(_queue->BeginGetEvent(callback, state));
    return S_OK;
}

STDMETHODIMP MediaSource::EndGetEvent(IMFAsyncResult* result, IMFMediaEvent** event)
{
    RETURN_HR_IF_NULL(E_POINTER, event);
    *event = nullptr;
    winrt::slim_lock_guard lock(_lock);
    RETURN_HR_IF(MF_E_SHUTDOWN, !_queue);
    RETURN_IF_FAILED(_queue->EndGetEvent(result, event));
    return S_OK;
}

STDMETHODIMP MediaSource::GetEvent(DWORD flags, IMFMediaEvent** event)
{
    RETURN_HR_IF_NULL(E_POINTER, event);
    *event = nullptr;
    wil::com_ptr_nothrow<IMFMediaEventQueue> queue;
    {
        winrt::slim_lock_guard lock(_lock);
        RETURN_HR_IF(MF_E_SHUTDOWN, !_queue);
        queue = _queue;
    }
    RETURN_IF_FAILED(queue->GetEvent(flags, event));
    return S_OK;
}

STDMETHODIMP MediaSource::QueueEvent(MediaEventType type, REFGUID extendedType, HRESULT status, const PROPVARIANT* value)
{
    winrt::slim_lock_guard lock(_lock);
    RETURN_HR_IF(MF_E_SHUTDOWN, !_queue);
    RETURN_IF_FAILED(_queue->QueueEventParamVar(type, extendedType, status, value));
    return S_OK;
}

// IMFMediaSource
STDMETHODIMP MediaSource::CreatePresentationDescriptor(IMFPresentationDescriptor** descriptor)
{
    RETURN_HR_IF_NULL(E_POINTER, descriptor);
    *descriptor = nullptr;
    winrt::slim_lock_guard lock(_lock);
    RETURN_HR_IF(MF_E_SHUTDOWN, !_descriptor);
    RETURN_IF_FAILED(_descriptor->Clone(descriptor));
    return S_OK;
}

STDMETHODIMP MediaSource::GetCharacteristics(DWORD* characteristics)
{
    RETURN_HR_IF_NULL(E_POINTER, characteristics);
    *characteristics = MFMEDIASOURCE_IS_LIVE;
    return S_OK;
}

STDMETHODIMP MediaSource::Pause()
{
    RETURN_HR(MF_E_INVALID_STATE_TRANSITION);
}

STDMETHODIMP MediaSource::Shutdown()
{
    winrt::slim_lock_guard lock(_lock);
    RETURN_HR_IF(MF_E_SHUTDOWN, !_queue);
    LOG_IF_FAILED(_queue->Shutdown());
    _queue.reset();
    for (auto& stream : _streams)
        stream->Shutdown();
    _streams.clear();
    _descriptor.reset();
    _attributes.reset();
    DbpLog(L"Media source shut down");
    return S_OK;
}

STDMETHODIMP MediaSource::Start(IMFPresentationDescriptor* descriptor, const GUID* timeFormat, const PROPVARIANT* startPosition)
{
    RETURN_HR_IF_NULL(E_POINTER, descriptor);
    RETURN_HR_IF_NULL(E_POINTER, startPosition);
    RETURN_HR_IF(E_INVALIDARG, timeFormat && *timeFormat != GUID_NULL);
    winrt::slim_lock_guard lock(_lock);
    RETURN_HR_IF(MF_E_SHUTDOWN, !_queue || !_descriptor);

    DWORD count = 0;
    RETURN_IF_FAILED(descriptor->GetStreamDescriptorCount(&count));
    RETURN_HR_IF(E_INVALIDARG, count != (DWORD)_streams.size());

    wil::unique_prop_variant time;
    RETURN_IF_FAILED(InitPropVariantFromInt64(MFGetSystemTime(), &time));

    for (DWORD i = 0; i < count; i++)
    {
        wil::com_ptr_nothrow<IMFStreamDescriptor> streamDescriptor;
        BOOL selected = FALSE;
        RETURN_IF_FAILED(descriptor->GetStreamDescriptorByIndex(i, &selected, &streamDescriptor));
        DWORD id = 0;
        RETURN_IF_FAILED(streamDescriptor->GetStreamIdentifier(&id));
        auto index = StreamIndexById(id);
        RETURN_HR_IF(E_FAIL, index < 0);

        BOOL wasSelected = FALSE;
        wil::com_ptr_nothrow<IMFStreamDescriptor> ours;
        RETURN_IF_FAILED(_descriptor->GetStreamDescriptorByIndex(index, &wasSelected, &ours));
        MF_STREAM_STATE state;
        RETURN_IF_FAILED(_streams[index]->GetStreamState(&state));
        if (wasSelected && state == MF_STREAM_STATE_STOPPED)
            wasSelected = FALSE;
        else if (!wasSelected && state != MF_STREAM_STATE_STOPPED)
            wasSelected = TRUE;

        if (selected == wasSelected)
            continue;
        if (selected)
        {
            RETURN_IF_FAILED(_descriptor->SelectStream(index));
            wil::com_ptr_nothrow<IUnknown> unknown;
            RETURN_IF_FAILED(_streams[index]->QueryInterface(IID_PPV_ARGS(&unknown)));
            RETURN_IF_FAILED(_queue->QueueEventParamUnk(MENewStream, GUID_NULL, S_OK, unknown.get()));

            wil::com_ptr_nothrow<IMFMediaTypeHandler> handler;
            wil::com_ptr_nothrow<IMFMediaType> type;
            RETURN_IF_FAILED(streamDescriptor->GetMediaTypeHandler(&handler));
            RETURN_IF_FAILED(handler->GetCurrentMediaType(&type));
            RETURN_IF_FAILED(_streams[index]->Start(type.get()));
        }
        else
        {
            RETURN_IF_FAILED(_descriptor->DeselectStream(index));
            RETURN_IF_FAILED(_streams[index]->Stop());
        }
    }

    RETURN_IF_FAILED(_queue->QueueEventParamVar(MESourceStarted, GUID_NULL, S_OK, &time));
    return S_OK;
}

STDMETHODIMP MediaSource::Stop()
{
    winrt::slim_lock_guard lock(_lock);
    RETURN_HR_IF(MF_E_SHUTDOWN, !_queue || !_descriptor);

    wil::unique_prop_variant time;
    RETURN_IF_FAILED(InitPropVariantFromInt64(MFGetSystemTime(), &time));
    for (DWORD i = 0; i < (DWORD)_streams.size(); i++)
    {
        MF_STREAM_STATE state;
        if (SUCCEEDED(_streams[i]->GetStreamState(&state)) && state != MF_STREAM_STATE_STOPPED)
            RETURN_IF_FAILED(_streams[i]->Stop());
        RETURN_IF_FAILED(_descriptor->DeselectStream(i));
    }
    RETURN_IF_FAILED(_queue->QueueEventParamVar(MESourceStopped, GUID_NULL, S_OK, &time));
    return S_OK;
}

// IMFMediaSourceEx
STDMETHODIMP MediaSource::GetSourceAttributes(IMFAttributes** attributes)
{
    RETURN_HR_IF_NULL(E_POINTER, attributes);
    RETURN_IF_FAILED(QueryInterface(IID_PPV_ARGS(attributes)));
    return S_OK;
}

STDMETHODIMP MediaSource::GetStreamAttributes(DWORD streamIdentifier, IMFAttributes** attributes)
{
    RETURN_HR_IF_NULL(E_POINTER, attributes);
    *attributes = nullptr;
    winrt::slim_lock_guard lock(_lock);
    auto index = StreamIndexById(streamIdentifier);
    RETURN_HR_IF(E_FAIL, index < 0);
    RETURN_IF_FAILED(_streams[index]->QueryInterface(IID_PPV_ARGS(attributes)));
    return S_OK;
}

STDMETHODIMP MediaSource::SetD3DManager(IUnknown* manager)
{
    // CPU only on purpose: the app's frames arrive in system memory anyway.
    RETURN_HR_IF_NULL(E_POINTER, manager);
    return S_OK;
}

// IMFGetService
STDMETHODIMP MediaSource::GetService(REFGUID, REFIID, LPVOID* object)
{
    if (object)
        *object = nullptr;
    return MF_E_UNSUPPORTED_SERVICE;
}

// IMFSampleAllocatorControl
STDMETHODIMP MediaSource::SetDefaultAllocator(DWORD outputStreamId, IUnknown* allocator)
{
    RETURN_HR_IF_NULL(E_POINTER, allocator);
    winrt::slim_lock_guard lock(_lock);
    auto index = StreamIndexById(outputStreamId);
    RETURN_HR_IF(E_FAIL, index < 0);
    RETURN_IF_FAILED(_streams[index]->SetAllocator(allocator));
    return S_OK;
}

STDMETHODIMP MediaSource::GetAllocatorUsage(DWORD outputStreamId, DWORD* inputStreamId, MFSampleAllocatorUsage* usage)
{
    RETURN_HR_IF_NULL(E_POINTER, inputStreamId);
    RETURN_HR_IF_NULL(E_POINTER, usage);
    winrt::slim_lock_guard lock(_lock);
    RETURN_HR_IF(E_FAIL, StreamIndexById(outputStreamId) < 0);
    *inputStreamId = outputStreamId;
    *usage = MFSampleAllocatorUsage_UsesProvidedAllocator;
    return S_OK;
}

// IKsControl: no camera controls.
STDMETHODIMP_(NTSTATUS) MediaSource::KsProperty(PKSPROPERTY property, ULONG, LPVOID, ULONG, ULONG* returned)
{
    RETURN_HR_IF_NULL(E_POINTER, property);
    RETURN_HR_IF_NULL(E_POINTER, returned);
    return HRESULT_FROM_WIN32(ERROR_SET_NOT_FOUND);
}

STDMETHODIMP_(NTSTATUS) MediaSource::KsMethod(PKSMETHOD method, ULONG, LPVOID, ULONG, ULONG* returned)
{
    RETURN_HR_IF_NULL(E_POINTER, method);
    RETURN_HR_IF_NULL(E_POINTER, returned);
    return HRESULT_FROM_WIN32(ERROR_SET_NOT_FOUND);
}

STDMETHODIMP_(NTSTATUS) MediaSource::KsEvent(PKSEVENT, ULONG, LPVOID, ULONG, ULONG* returned)
{
    RETURN_HR_IF_NULL(E_POINTER, returned);
    return HRESULT_FROM_WIN32(ERROR_SET_NOT_FOUND);
}
