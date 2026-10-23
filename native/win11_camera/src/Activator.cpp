#include "Activator.h"
#include "Log.h"

HRESULT Activator::Initialize()
{
    _source = winrt::make_self<MediaSource>();
    RETURN_IF_FAILED(SetUINT32(MF_VIRTUALCAMERA_PROVIDE_ASSOCIATED_CAMERA_SOURCES, 1));
    RETURN_IF_FAILED(SetGUID(MFT_TRANSFORM_CLSID_Attribute, CLSID_DbpCamera));
    RETURN_IF_FAILED(_source->Initialize(this));
    return S_OK;
}

STDMETHODIMP Activator::ActivateObject(REFIID riid, void** object)
{
    RETURN_HR_IF_NULL(E_POINTER, object);
    *object = nullptr;
    RETURN_HR_IF(MF_E_SHUTDOWN, !_source);

    UINT32 pid = 0;
    if (SUCCEEDED(GetUINT32(MF_FRAMESERVER_CLIENTCONTEXT_CLIENTPID, &pid)) && pid)
    {
        wchar_t path[MAX_PATH] = L"?";
        DWORD size = ARRAYSIZE(path);
        if (auto process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid))
        {
            QueryFullProcessImageNameW(process, 0, path, &size);
            CloseHandle(process);
        }
        DbpLog(L"Camera opened by process %u (%s)", pid, path);
    }
    RETURN_IF_FAILED(_source->QueryInterface(riid, object));
    return S_OK;
}

STDMETHODIMP Activator::ShutdownObject()
{
    return S_OK;
}

STDMETHODIMP Activator::DetachObject()
{
    _source = nullptr;
    return S_OK;
}
