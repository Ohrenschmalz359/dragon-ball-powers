#include "SharedFrame.h"
#include "Log.h"

namespace
{
    constexpr ULONGLONG RETRY_MS = 1000;     // how often a missing section is looked for again
    constexpr ULONGLONG STALE_MS = 3000;     // no new frame for this long: the app stopped sending

    // Writer and reader both run on x64 / ARM64: an acquire load is enough for the sequence.
    LONG64 LoadSequence(const DbpFrameHeader* header)
    {
        return ReadAcquire64(&header->sequence);
    }

    void CopyScaled(const BYTE* src, UINT srcWidth, UINT srcHeight, UINT srcStride,
                    BYTE* dst, LONG dstStride, UINT width, UINT height, std::vector<UINT>& xmap)
    {
        if (srcWidth == width && srcHeight == height)
        {
            for (UINT y = 0; y < height; y++)
                memcpy(dst + (LONG_PTR)y * dstStride, src + (SIZE_T)y * srcStride, (SIZE_T)width * 4);
            return;
        }
        // Nearest neighbour: the app sends 16:9 like this camera, usually even the same size.
        xmap.resize(width);
        for (UINT x = 0; x < width; x++)
            xmap[x] = (UINT)((ULONGLONG)x * srcWidth / width);
        for (UINT y = 0; y < height; y++)
        {
            auto srcRow = reinterpret_cast<const UINT32*>(src + (SIZE_T)((ULONGLONG)y * srcHeight / height) * srcStride);
            auto dstRow = reinterpret_cast<UINT32*>(dst + (LONG_PTR)y * dstStride);
            for (UINT x = 0; x < width; x++)
                dstRow[x] = srcRow[xmap[x]];
        }
    }

    inline BYTE ClampByte(int value)
    {
        return (BYTE)(value < 0 ? 0 : value > 255 ? 255 : value);
    }
}

SharedFrameReader::~SharedFrameReader()
{
    Close();
}

void SharedFrameReader::Close()
{
    if (_view)
    {
        UnmapViewOfFile(_view);
        _view = nullptr;
    }
    if (_mapping)
    {
        CloseHandle(_mapping);
        _mapping = nullptr;
    }
}

bool SharedFrameReader::EnsureMapped()
{
    if (_view)
        return true;

    auto now = GetTickCount64();
    if (now < _nextAttempt)
        return false;
    _nextAttempt = now + RETRY_MS;

    _mapping = OpenFileMappingW(FILE_MAP_READ, FALSE, DBP_SHARED_NAME);
    if (!_mapping)
    {
        PSECURITY_DESCRIPTOR descriptor = nullptr;
        if (ConvertStringSecurityDescriptorToSecurityDescriptorW(DBP_SHARED_SDDL, SDDL_REVISION_1, &descriptor, nullptr))
        {
            SECURITY_ATTRIBUTES security{ sizeof(security), descriptor, FALSE };
            _mapping = CreateFileMappingW(INVALID_HANDLE_VALUE, &security, PAGE_READWRITE,
                (DWORD)((ULONGLONG)DBP_SHARED_SIZE >> 32), (DWORD)(DBP_SHARED_SIZE & 0xFFFFFFFF), DBP_SHARED_NAME);
            auto error = GetLastError();
            LocalFree(descriptor);
            SetLastError(error);
        }
    }
    if (!_mapping)
    {
        if (!_failureLogged)
        {
            _failureLogged = true;
            DbpLog(L"Frame memory not available yet (error %lu) - showing the placeholder", GetLastError());
        }
        return false;
    }

    _view = static_cast<const BYTE*>(MapViewOfFile(_mapping, FILE_MAP_READ, 0, 0, DBP_SHARED_SIZE));
    if (!_view)
    {
        DbpLog(L"MapViewOfFile failed (error %lu)", GetLastError());
        CloseHandle(_mapping);
        _mapping = nullptr;
        return false;
    }
    DbpLog(L"Frame memory connected");
    _failureLogged = false;
    return true;
}

bool SharedFrameReader::Read(BYTE* dst, LONG dstStride, UINT width, UINT height)
{
    if (!dst || !width || !height || !EnsureMapped())
        return false;

    auto header = reinterpret_cast<const DbpFrameHeader*>(_view);
    auto now = GetTickCount64();
    auto sequence = LoadSequence(header);
    if (header->magic == DBP_SHARED_MAGIC && header->version == DBP_SHARED_VERSION && !(sequence & 1) && sequence != _sequence)
    {
        UINT w = header->width;
        UINT h = header->height;
        UINT stride = header->stride;
        bool valid = header->format == DBP_FORMAT_BGRA && w && h && w <= DBP_SHARED_MAX_WIDTH && h <= DBP_SHARED_MAX_HEIGHT &&
            stride >= w * 4 && (SIZE_T)stride * h <= DBP_SHARED_SIZE - DBP_SHARED_HEADER_SIZE;
        if (valid)
        {
            _copy.resize((SIZE_T)stride * h);
            memcpy(_copy.data(), _view + DBP_SHARED_HEADER_SIZE, _copy.size());
            if (LoadSequence(header) == sequence)       // not overwritten while copying
            {
                _width = w;
                _height = h;
                _stride = stride;
                _sequence = sequence;
                _lastChange = now;
                if (!_frameLogged)
                {
                    _frameLogged = true;
                    DbpLog(L"First frame from the app: %ux%u", w, h);
                }
            }
        }
    }

    if (_sequence < 0 || now - _lastChange > STALE_MS)
        return false;

    CopyScaled(_copy.data(), _width, _height, _stride, dst, dstStride, width, height, _xmap);
    return true;
}

void FillPlaceholder(BYTE* dst, LONG dstStride, UINT width, UINT height)
{
    constexpr UINT32 color = 0xFF101830;     // A R G B in memory as B G R A: dark blue
    for (UINT y = 0; y < height; y++)
    {
        auto row = reinterpret_cast<UINT32*>(dst + (LONG_PTR)y * dstStride);
        for (UINT x = 0; x < width; x++)
            row[x] = color;
    }
}

HRESULT Bgra32ToNv12(const BYTE* input, LONG inputStride, UINT width, UINT height, BYTE* output, LONG outputStride)
{
    RETURN_HR_IF_NULL(E_INVALIDARG, input);
    RETURN_HR_IF_NULL(E_INVALIDARG, output);
    RETURN_HR_IF(E_INVALIDARG, (width & 1) || (height & 1) || !width || !height);

    auto uvPlane = output + (LONG_PTR)outputStride * height;
    for (UINT y = 0; y < height; y += 2)
    {
        auto row0 = input + (LONG_PTR)y * inputStride;
        auto row1 = row0 + inputStride;
        auto y0 = output + (LONG_PTR)y * outputStride;
        auto y1 = y0 + outputStride;
        auto uv = uvPlane + (LONG_PTR)(y / 2) * outputStride;
        for (UINT x = 0; x < width; x += 2)
        {
            int sumR = 0, sumG = 0, sumB = 0;
            const BYTE* pixels[4] = { row0 + x * 4, row0 + x * 4 + 4, row1 + x * 4, row1 + x * 4 + 4 };
            BYTE* luma[4] = { y0 + x, y0 + x + 1, y1 + x, y1 + x + 1 };
            for (int i = 0; i < 4; i++)
            {
                int b = pixels[i][0], g = pixels[i][1], r = pixels[i][2];
                *luma[i] = ClampByte(((66 * r + 129 * g + 25 * b + 128) >> 8) + 16);
                sumR += r;
                sumG += g;
                sumB += b;
            }
            int r = sumR / 4, g = sumG / 4, b = sumB / 4;
            uv[x] = ClampByte(((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128);
            uv[x + 1] = ClampByte(((112 * r - 94 * g - 18 * b + 128) >> 8) + 128);
        }
    }
    return S_OK;
}
