#pragma once

#include "framework.h"

// Frames travel from the Dragon Ball Powers app (the writer,
// src/dragon_ball_powers/win11_camera.py) to this media source through a named section.
// Both sides must agree on everything in this block (tests/test_win11_camera.py checks it).
//
// Layout: a 64-byte header, then the pixels: BGRA (B, G, R, A bytes), top row first.
// The writer makes "sequence" odd while it copies a frame and even when the frame is
// complete; the reader copies the pixels and keeps them only if the sequence did not change.
//
// The frame server (Local Service, session 0) creates the section when an app starts the
// camera: an ordinary user process may not create objects in the Global namespace, but it
// may open them - the security descriptor below grants signed-in users read/write access.
constexpr wchar_t DBP_SHARED_NAME[] = L"Global\\DragonBallPowersVirtualCamera";
constexpr UINT32 DBP_SHARED_MAGIC = 0x56504244;   // "DBPV" little-endian
constexpr UINT32 DBP_SHARED_VERSION = 1;
constexpr UINT32 DBP_SHARED_MAX_WIDTH = 1920;
constexpr UINT32 DBP_SHARED_MAX_HEIGHT = 1080;
constexpr UINT32 DBP_SHARED_HEADER_SIZE = 64;
constexpr SIZE_T DBP_SHARED_SIZE = DBP_SHARED_HEADER_SIZE + (SIZE_T)DBP_SHARED_MAX_WIDTH * DBP_SHARED_MAX_HEIGHT * 4;
constexpr UINT32 DBP_FORMAT_BGRA = 1;
constexpr wchar_t DBP_SHARED_SDDL[] = L"D:(A;;GA;;;SY)(A;;GA;;;LS)(A;;GA;;;BA)(A;;GRGW;;;AU)(A;;GRGW;;;AC)S:(ML;;NW;;;LW)";

#pragma pack(push, 1)
struct DbpFrameHeader
{
    UINT32 magic;                 // offset 0
    UINT32 version;               // 4
    UINT32 width;                 // 8
    UINT32 height;                // 12
    UINT32 stride;                // 16: bytes per row (>= width * 4)
    UINT32 format;                // 20: DBP_FORMAT_BGRA
    volatile LONG64 sequence;     // 24: odd while the writer copies a frame
    LONG64 timestamp;             // 32: writer clock in 100 ns units (informational)
    UINT32 writerPid;             // 40
    UINT32 reserved[5];           // 44..63
};
#pragma pack(pop)
static_assert(sizeof(DbpFrameHeader) == DBP_SHARED_HEADER_SIZE, "shared header layout");

class SharedFrameReader
{
public:
    ~SharedFrameReader();

    // Copies the newest frame of the app, scaled to width x height, into dst (BGRA, top
    // row first, rows dstStride bytes apart). False when there is none (app not sending).
    bool Read(BYTE* dst, LONG dstStride, UINT width, UINT height);

private:
    bool EnsureMapped();
    void Close();

    HANDLE _mapping = nullptr;
    const BYTE* _view = nullptr;
    ULONGLONG _nextAttempt = 0;
    bool _failureLogged = false;
    bool _frameLogged = false;
    std::vector<BYTE> _copy;
    UINT _width = 0;
    UINT _height = 0;
    UINT _stride = 0;
    LONG64 _sequence = -1;
    ULONGLONG _lastChange = 0;
    std::vector<UINT> _xmap;
};

// Dark blue picture shown while the app does not send.
void FillPlaceholder(BYTE* dst, LONG dstStride, UINT width, UINT height);

// BGRA -> NV12 (BT.601, limited range). width and height must be even.
HRESULT Bgra32ToNv12(const BYTE* input, LONG inputStride, UINT width, UINT height, BYTE* output, LONG outputStride);
