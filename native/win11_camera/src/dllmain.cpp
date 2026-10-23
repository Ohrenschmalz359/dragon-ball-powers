#include "Activator.h"
#include "Log.h"

// {D61C8D41-AD63-4319-AE6F-EB46E5FEB187}
const GUID CLSID_DbpCamera = { 0xd61c8d41, 0xad63, 0x4319, { 0xae, 0x6f, 0xeb, 0x46, 0xe5, 0xfe, 0xb1, 0x87 } };

namespace
{
    HMODULE g_module = nullptr;

    std::wstring ClsidKey()
    {
        return L"Software\\Classes\\CLSID\\" + GuidToString(CLSID_DbpCamera);
    }

    LSTATUS SetString(const std::wstring& key, PCWSTR name, PCWSTR value)
    {
        return RegSetKeyValueW(HKEY_LOCAL_MACHINE, key.c_str(), name, REG_SZ, value, (DWORD)((wcslen(value) + 1) * sizeof(wchar_t)));
    }

    // install: create the system-wide camera "Dragon Ball Powers (Windows 11)" and start it
    // (it stays until removed, also after a restart); otherwise remove it.
    HRESULT CreateOrRemoveCamera(bool install)
    {
        auto com = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        auto uninitialize = wil::scope_exit([&] { if (SUCCEEDED(com)) CoUninitialize(); });
        RETURN_IF_FAILED(MFStartup(MF_VERSION));
        auto shutdown = wil::scope_exit([] { MFShutdown(); });

        auto clsid = GuidToString(CLSID_DbpCamera);
        wil::com_ptr_nothrow<IMFVirtualCamera> camera;
        auto hr = MFCreateVirtualCamera(MFVirtualCameraType_SoftwareCameraSource, MFVirtualCameraLifetime_System,
            MFVirtualCameraAccess_AllUsers, DBP_CAMERA_NAME, clsid.c_str(), nullptr, 0, &camera);
        if (FAILED(hr))
        {
            DbpLog(L"MFCreateVirtualCamera failed: 0x%08X", hr);
            return hr;
        }
        // Like VCamSample: no IMFVirtualCamera::Shutdown here, it would shut the media source down twice.
        hr = install ? camera->Start(nullptr) : camera->Remove();
        DbpLog(L"Virtual camera %s: 0x%08X", install ? L"start" : L"remove", hr);
        return hr;
    }
}

std::wstring GuidToString(const GUID& guid)
{
    wchar_t text[64];
    return StringFromGUID2(guid, text, ARRAYSIZE(text)) ? text : L"";
}

BOOL APIENTRY DllMain(HMODULE module, DWORD reason, LPVOID)
{
    if (reason == DLL_PROCESS_ATTACH)
    {
        g_module = module;
        DisableThreadLibraryCalls(module);
        wil::SetResultLoggingCallback([](wil::FailureInfo const& failure) noexcept
            {
                wchar_t text[1024];
                if (SUCCEEDED(wil::GetFailureLogString(text, ARRAYSIZE(text), failure)))
                    DbpLog(L"%s", text);
            });
    }
    return TRUE;
}

struct ClassFactory : winrt::implements<ClassFactory, IClassFactory>
{
    STDMETHODIMP CreateInstance(IUnknown* outer, GUID const& riid, void** result) noexcept final
    {
        RETURN_HR_IF_NULL(E_POINTER, result);
        *result = nullptr;
        if (outer)
            RETURN_HR(CLASS_E_NOAGGREGATION);
        try
        {
            auto activator = winrt::make_self<Activator>();
            RETURN_IF_FAILED(activator->Initialize());
            return activator->QueryInterface(riid, result);
        }
        catch (...)
        {
            RETURN_CAUGHT_EXCEPTION();
        }
    }

    STDMETHODIMP LockServer(BOOL) noexcept final
    {
        return S_OK;
    }
};

STDAPI DllCanUnloadNow()
{
    if (winrt::get_module_lock())
        return S_FALSE;
    winrt::clear_factory_cache();
    return S_OK;
}

STDAPI DllGetClassObject(REFCLSID clsid, REFIID riid, LPVOID* object)
{
    RETURN_HR_IF_NULL(E_POINTER, object);
    *object = nullptr;
    if (clsid != CLSID_DbpCamera)
        return CLASS_E_CLASSNOTAVAILABLE;
    try
    {
        return winrt::make_self<ClassFactory>()->QueryInterface(riid, object);
    }
    catch (...)
    {
        RETURN_CAUGHT_EXCEPTION();
    }
}

// The frame server reads the registration from HKEY_LOCAL_MACHINE only (administrator).
STDAPI DllRegisterServer()
{
    wchar_t path[MAX_PATH * 2];
    auto length = GetModuleFileNameW(g_module, path, ARRAYSIZE(path));
    RETURN_LAST_ERROR_IF(length == 0 || length >= ARRAYSIZE(path));
    auto key = ClsidKey();
    RETURN_IF_WIN32_ERROR(SetString(key, nullptr, DBP_CAMERA_NAME));
    RETURN_IF_WIN32_ERROR(SetString(key + L"\\InprocServer32", nullptr, path));
    RETURN_IF_WIN32_ERROR(SetString(key + L"\\InprocServer32", L"ThreadingModel", L"Both"));
    DbpLog(L"Registered %s", path);
    return S_OK;
}

STDAPI DllUnregisterServer()
{
    auto status = RegDeleteTreeW(HKEY_LOCAL_MACHINE, ClsidKey().c_str());
    if (status != ERROR_SUCCESS && status != ERROR_FILE_NOT_FOUND)
        RETURN_WIN32(status);
    DbpLog(L"Unregistered");
    return S_OK;
}

// "regsvr32 /n /i <dll>": register + create the camera; "regsvr32 /n /u /i <dll>": remove both.
STDAPI DllInstall(BOOL install, PCWSTR)
{
    if (install)
    {
        RETURN_IF_FAILED(DllRegisterServer());
        return CreateOrRemoveCamera(true);
    }
    auto removed = CreateOrRemoveCamera(false);
    auto unregistered = DllUnregisterServer();
    return FAILED(removed) ? removed : unregistered;
}
