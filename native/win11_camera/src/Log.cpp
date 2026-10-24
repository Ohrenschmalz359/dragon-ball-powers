#include "Log.h"

namespace
{
    constexpr int MAX_LINES_PER_PROCESS = 400;
    constexpr LONGLONG MAX_LOG_BYTES = 1024 * 1024;

    std::mutex g_lock;
    std::atomic<int> g_lines{ 0 };

    std::wstring LogPath()
    {
        wchar_t path[MAX_PATH];
        auto length = ExpandEnvironmentStringsW(L"%ProgramData%\\Dragon Ball Powers\\win11-camera.log", path, ARRAYSIZE(path));
        if (!length || length > ARRAYSIZE(path) || wcschr(path, L'%'))
            return {};
        return path;
    }

    std::wstring ProcessName()
    {
        wchar_t path[MAX_PATH]{};
        GetModuleFileNameW(nullptr, path, ARRAYSIZE(path));
        auto name = wcsrchr(path, L'\\');
        return name ? name + 1 : path;
    }
}

void DbpLog(PCWSTR format, ...)
{
    if (g_lines.fetch_add(1) >= MAX_LINES_PER_PROCESS)
        return;

    wchar_t message[1024];
    va_list args;
    va_start(args, format);
    StringCchVPrintfW(message, ARRAYSIZE(message), format, args);
    va_end(args);

    SYSTEMTIME time;
    GetLocalTime(&time);
    wchar_t line[1400];
    StringCchPrintfW(line, ARRAYSIZE(line), L"%04u-%02u-%02u %02u:%02u:%02u.%03u [%lu %s] %s\r\n",
        time.wYear, time.wMonth, time.wDay, time.wHour, time.wMinute, time.wSecond, time.wMilliseconds,
        GetCurrentProcessId(), ProcessName().c_str(), message);

    char utf8[4096];
    auto length = WideCharToMultiByte(CP_UTF8, 0, line, -1, utf8, sizeof(utf8), nullptr, nullptr);
    if (length <= 1)
        return;

    std::lock_guard guard(g_lock);
    auto path = LogPath();
    if (path.empty())
        return;

    constexpr DWORD share = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
    auto file = CreateFileW(path.c_str(), FILE_APPEND_DATA | FILE_READ_ATTRIBUTES, share, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE)
        return;

    LARGE_INTEGER size{};
    if (GetFileSizeEx(file, &size) && size.QuadPart > MAX_LOG_BYTES)
    {
        CloseHandle(file);    // start over instead of growing forever
        file = CreateFileW(path.c_str(), GENERIC_WRITE, share, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE)
            return;
    }

    DWORD written = 0;
    WriteFile(file, utf8, (DWORD)(length - 1), &written, nullptr);
    CloseHandle(file);
}
