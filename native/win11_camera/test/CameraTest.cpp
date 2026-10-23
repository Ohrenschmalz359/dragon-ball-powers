// Tests for the Dragon Ball Powers Windows 11 virtual camera (run by .github/workflows/win11-camera.yml).
//
//   CameraTest.exe inproc <path to DragonBallPowersCamera.dll>
//       loads the media source into this process (no registration needed) and reads RGB32
//       and NV12 frames through a Media Foundation source reader
//   CameraTest.exe device
//       finds the registered camera "Dragon Ball Powers (Windows 11)" through the Windows
//       camera frame server (Windows 11 only) and reads frames from it
//
// Both write a test picture (left half red, right half green) into the shared frame memory
// exactly like the app does and check that it comes out of the camera.
#include "../src/SharedFrame.h"
#include <mfreadwrite.h>
#include <wrl/client.h>
#include <cstdio>
#include <cstdlib>

#pragma comment(lib, "mfreadwrite")
#pragma comment(lib, "mf")

using Microsoft::WRL::ComPtr;

namespace
{
    const GUID kCameraClsid = { 0xd61c8d41, 0xad63, 0x4319, { 0xae, 0x6f, 0xeb, 0x46, 0xe5, 0xfe, 0xb1, 0x87 } };
    constexpr wchar_t kCameraName[] = L"Dragon Ball Powers (Windows 11)";

#define CHECK(expr) do { HRESULT hr_ = (expr); if (FAILED(hr_)) { printf("FAILED 0x%08X: %s (line %d)\n", (unsigned)hr_, #expr, __LINE__); return 1; } } while (0)

    struct FrameWriter
    {
        HANDLE mapping = nullptr;
        BYTE* view = nullptr;

        ~FrameWriter()
        {
            if (view) UnmapViewOfFile(view);
            if (mapping) CloseHandle(mapping);
        }

        bool Open(bool create)
        {
            if (view)
                return true;
            mapping = OpenFileMappingW(FILE_MAP_WRITE, FALSE, DBP_SHARED_NAME);
            if (!mapping && create)
            {
                PSECURITY_DESCRIPTOR descriptor = nullptr;
                if (ConvertStringSecurityDescriptorToSecurityDescriptorW(DBP_SHARED_SDDL, SDDL_REVISION_1, &descriptor, nullptr))
                {
                    SECURITY_ATTRIBUTES security{ sizeof(security), descriptor, FALSE };
                    mapping = CreateFileMappingW(INVALID_HANDLE_VALUE, &security, PAGE_READWRITE, 0, (DWORD)DBP_SHARED_SIZE, DBP_SHARED_NAME);
                    LocalFree(descriptor);
                }
            }
            if (!mapping)
                return false;
            view = static_cast<BYTE*>(MapViewOfFile(mapping, FILE_MAP_WRITE, 0, 0, DBP_SHARED_SIZE));
            if (!view)
            {
                CloseHandle(mapping);
                mapping = nullptr;
                return false;
            }
            printf("frame memory opened\n");
            return true;
        }

        void Write(UINT width, UINT height)
        {
            auto header = reinterpret_cast<DbpFrameHeader*>(view);
            LONG64 writing = (header->sequence + 1) | 1;              // odd: a frame is being written
            InterlockedExchange64(&header->sequence, writing);
            header->magic = DBP_SHARED_MAGIC;
            header->version = DBP_SHARED_VERSION;
            header->width = width;
            header->height = height;
            header->stride = width * 4;
            header->format = DBP_FORMAT_BGRA;
            header->writerPid = GetCurrentProcessId();
            auto pixels = view + DBP_SHARED_HEADER_SIZE;
            for (UINT y = 0; y < height; y++)
            {
                auto row = reinterpret_cast<UINT32*>(pixels + (SIZE_T)y * width * 4);
                for (UINT x = 0; x < width; x++)
                    row[x] = x < width / 2 ? 0xFFFF0000 : 0xFF00FF00;    // BGRA bytes: red / green
            }
            InterlockedExchange64(&header->sequence, writing + 1);    // even: complete
        }
    };

    bool CheckRgb32(const BYTE* data, LONG stride, UINT width, UINT height)
    {
        auto left = data + (LONG_PTR)(height / 2) * stride + (width / 4) * 4;
        auto right = data + (LONG_PTR)(height / 2) * stride + (3 * width / 4) * 4;
        printf("  RGB32 left BGR=(%u,%u,%u) right BGR=(%u,%u,%u)\n", left[0], left[1], left[2], right[0], right[1], right[2]);
        return left[2] > 200 && left[1] < 60 && right[1] > 200 && right[2] < 60;
    }

    bool CheckNv12(const BYTE* data, LONG stride, UINT width, UINT height)
    {
        int left = data[(LONG_PTR)(height / 2) * stride + width / 4];
        int right = data[(LONG_PTR)(height / 2) * stride + 3 * width / 4];
        printf("  NV12 luma left=%d (red: 82) right=%d (green: 144)\n", left, right);
        return abs(left - 82) <= 8 && abs(right - 144) <= 8;
    }

    int ReadAndCheck(IMFSourceReader* reader, REFGUID subtype, FrameWriter& writer, bool createMemory, int maxFrames)
    {
        ComPtr<IMFMediaType> type;
        for (DWORD i = 0;; i++)
        {
            ComPtr<IMFMediaType> native;
            if (FAILED(reader->GetNativeMediaType((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, i, &native)))
                break;
            GUID nativeSubtype{};
            native->GetGUID(MF_MT_SUBTYPE, &nativeSubtype);
            UINT32 w = 0, h = 0;
            MFGetAttributeSize(native.Get(), MF_MT_FRAME_SIZE, &w, &h);
            printf("  native type %u: %s %ux%u\n", i, nativeSubtype == MFVideoFormat_NV12 ? "NV12" : nativeSubtype == MFVideoFormat_RGB32 ? "RGB32" : "other", w, h);
            if (nativeSubtype == subtype && !type)
                type = native;
        }
        if (!type)
        {
            printf("  the camera does not offer this format\n");
            return 1;
        }
        CHECK(reader->SetStreamSelection((DWORD)MF_SOURCE_READER_ALL_STREAMS, FALSE));
        CHECK(reader->SetStreamSelection((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, TRUE));
        CHECK(reader->SetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, nullptr, type.Get()));
        UINT32 width = 0, height = 0;
        CHECK(MFGetAttributeSize(type.Get(), MF_MT_FRAME_SIZE, &width, &height));

        for (int i = 0; i < maxFrames; i++)
        {
            if (writer.Open(createMemory))
                writer.Write(width, height);
            DWORD stream = 0, flags = 0;
            LONGLONG timestamp = 0;
            ComPtr<IMFSample> sample;
            CHECK(reader->ReadSample((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0, &stream, &flags, &timestamp, &sample));
            if (flags & MF_SOURCE_READERF_ERROR)
            {
                printf("  stream error\n");
                return 1;
            }
            if (!sample)
                continue;
            ComPtr<IMFMediaBuffer> buffer;
            CHECK(sample->GetBufferByIndex(0, &buffer));
            ComPtr<IMF2DBuffer> buffer2d;
            BYTE* data = nullptr;
            LONG stride = 0;
            DWORD maxLength = 0, length = 0;
            bool locked2d = SUCCEEDED(buffer.As(&buffer2d)) && SUCCEEDED(buffer2d->Lock2D(&data, &stride));
            if (!locked2d)
            {
                CHECK(buffer->Lock(&data, &maxLength, &length));
                stride = subtype == MFVideoFormat_RGB32 ? (LONG)width * 4 : (LONG)width;
            }
            bool ok = subtype == MFVideoFormat_RGB32 ? CheckRgb32(data, stride, width, height) : CheckNv12(data, stride, width, height);
            if (locked2d)
                buffer2d->Unlock2D();
            else
                buffer->Unlock();
            if (ok)
            {
                printf("  test picture received in frame %d\n", i);
                return 0;
            }
            Sleep(20);
        }
        printf("  the test picture never arrived\n");
        return 1;
    }

    int InProcess(const wchar_t* dllPath, REFGUID subtype)
    {
        auto dll = LoadLibraryW(dllPath);
        if (!dll)
        {
            printf("LoadLibrary failed: %lu\n", GetLastError());
            return 1;
        }
        using GetClassObject = HRESULT(STDAPICALLTYPE*)(REFCLSID, REFIID, LPVOID*);
        auto getClassObject = reinterpret_cast<GetClassObject>(GetProcAddress(dll, "DllGetClassObject"));
        if (!getClassObject)
        {
            printf("DllGetClassObject not exported\n");
            return 1;
        }
        ComPtr<IClassFactory> factory;
        CHECK(getClassObject(kCameraClsid, IID_PPV_ARGS(&factory)));
        ComPtr<IMFActivate> activate;
        CHECK(factory->CreateInstance(nullptr, IID_PPV_ARGS(&activate)));
        ComPtr<IMFMediaSource> source;
        CHECK(activate->ActivateObject(IID_PPV_ARGS(&source)));

        FrameWriter writer;
        if (!writer.Open(true))
        {
            printf("cannot create the frame memory: %lu\n", GetLastError());
            return 1;
        }
        writer.Write(1280, 720);
        ComPtr<IMFSourceReader> reader;
        CHECK(MFCreateSourceReaderFromMediaSource(source.Get(), nullptr, &reader));
        auto rc = ReadAndCheck(reader.Get(), subtype, writer, true, 90);
        reader.Reset();             // shuts the source down as well
        return rc;
    }

    int Device(REFGUID subtype)
    {
        ComPtr<IMFAttributes> attributes;
        CHECK(MFCreateAttributes(&attributes, 1));
        CHECK(attributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID));
        ComPtr<IMFActivate> camera;
        for (int attempt = 0; attempt < 30 && !camera; attempt++)
        {
            IMFActivate** devices = nullptr;
            UINT32 count = 0;
            CHECK(MFEnumDeviceSources(attributes.Get(), &devices, &count));
            for (UINT32 i = 0; i < count; i++)
            {
                wchar_t* name = nullptr;
                UINT32 length = 0;
                if (SUCCEEDED(devices[i]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &name, &length)))
                {
                    if (attempt == 0)
                        printf("  camera: %ls\n", name);
                    if (!camera && wcscmp(name, kCameraName) == 0)
                        camera = devices[i];
                    CoTaskMemFree(name);
                }
                devices[i]->Release();
            }
            CoTaskMemFree(devices);
            if (!camera)
                Sleep(500);
        }
        if (!camera)
        {
            printf("camera '%ls' not found\n", kCameraName);
            return 1;
        }
        ComPtr<IMFMediaSource> source;
        CHECK(camera->ActivateObject(IID_PPV_ARGS(&source)));
        ComPtr<IMFSourceReader> reader;
        CHECK(MFCreateSourceReaderFromMediaSource(source.Get(), nullptr, &reader));
        FrameWriter writer;          // the frame server creates the memory once it streams
        auto rc = ReadAndCheck(reader.Get(), subtype, writer, false, 300);
        reader.Reset();
        camera->ShutdownObject();
        return rc;
    }
}

int wmain(int argc, wchar_t** argv)
{
    if (argc < 2 || (wcscmp(argv[1], L"inproc") == 0 && argc < 3))
    {
        printf("usage: CameraTest.exe inproc <dll> | device\n");
        return 2;
    }
    CHECK(CoInitializeEx(nullptr, COINIT_MULTITHREADED));
    CHECK(MFStartup(MF_VERSION));
    int rc = 0;
    for (auto subtype : { MFVideoFormat_RGB32, MFVideoFormat_NV12 })
    {
        printf("%s %s:\n", wcscmp(argv[1], L"inproc") == 0 ? "in-process" : "frame server", subtype == MFVideoFormat_NV12 ? "NV12" : "RGB32");
        rc |= wcscmp(argv[1], L"inproc") == 0 ? InProcess(argv[2], subtype) : Device(subtype);
    }
    MFShutdown();
    CoUninitialize();
    printf(rc ? "FAIL\n" : "PASS\n");
    return rc;
}
