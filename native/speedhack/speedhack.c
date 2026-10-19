/*
 * speedhack.c —— 32 位"时间缩放"钩子 DLL
 *
 * 作用: 钩住目标进程 (东方正作, 32 位) 的 timeGetTime / GetTickCount,
 * 让游戏主循环按 factor 倍速运行:
 *   factor > 1  加速 —— 真机训练时显著提高每秒帧数 (训练吞吐);
 *   factor < 1  减速 —— 便于观察, 或给慢速 Agent 留出决策时间。
 *
 * 速度因子从 %TEMP%\touhou_speedhack.cfg 读取 (纯文本, 例如 "4.0"),
 * 由外部 Python 写入; DLL 每 200ms 重新读取, 所以可以运行中动态改速度。
 * 诊断信息写到 %TEMP%\touhou_speedhack.log。
 *
 * 钩子实现: 5 字节热补丁 (hotpatch) 序言 `8B FF 55 8B EC` + trampoline。
 * 若某个导出是转发器 (prologue 不匹配) 则跳过并尝试下一个模块 —— 绝不硬改。
 */

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PROLOGUE_LEN 5

static const BYTE HOTPATCH[PROLOGUE_LEN] = { 0x8B, 0xFF, 0x55, 0x8B, 0xEC };

typedef DWORD (WINAPI *tDwordFn)(void);

/* factor * 1000, 用 Interlocked 原子读写 */
static volatile LONG g_speed_milli = 1000;
static char g_cfg_path[MAX_PATH] = { 0 };
static char g_log_path[MAX_PATH] = { 0 };

typedef struct {
    tDwordFn          original;   /* trampoline: 原函数前 5 字节 + 跳回 */
    DWORD             last_real;
    DWORD             scaled;
    CRITICAL_SECTION  cs;
    int               ready;
} Scaler;

static Scaler g_tick;   /* GetTickCount  */
static Scaler g_time;   /* timeGetTime   */

static DWORD WINAPI hk_GetTickCount(void);
static DWORD WINAPI hk_timeGetTime(void);

static void log_fmt(const char *fmt, ...)
{
    char line[512];
    va_list args;
    HANDLE f;
    DWORD written;

    if (!g_log_path[0])
        return;
    va_start(args, fmt);
    vsnprintf(line, sizeof(line) - 3, fmt, args);
    va_end(args);
    strcat(line, "\r\n");

    f = CreateFileA(g_log_path, FILE_APPEND_DATA, FILE_SHARE_READ, NULL,
                    OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (f == INVALID_HANDLE_VALUE)
        return;
    SetFilePointer(f, 0, NULL, FILE_END);
    WriteFile(f, line, (DWORD)strlen(line), &written, NULL);
    CloseHandle(f);
}

static BOOL install_hook(void *target, void *detour, void **original_out)
{
    BYTE *p = (BYTE *)target;
    BYTE *tramp;
    DWORD old_protect;

    log_fmt("  target=0x%08lX bytes=%02X %02X %02X %02X %02X %02X %02X %02X",
            (unsigned long)(ULONG_PTR)target, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);

    /* 情形 A: 转发器桩 `jmp dword ptr [disp32]` (FF 25 xx xx xx xx)
     * 直接改它跳转的 IAT 槽, 原函数指针存起来当 trampoline 用。 */
    if (p[0] == 0xFF && p[1] == 0x25) {
        void **slot = (void **)(ULONG_PTR)(*(DWORD *)(p + 2));
        void *old;
        DWORD old_protect;
        old = *slot;
        if (!VirtualProtect(slot, sizeof(void *), PAGE_EXECUTE_READWRITE, &old_protect)) {
            log_fmt("  -> forwarder slot VirtualProtect failed");
            return FALSE;
        }
        *slot = detour;
        VirtualProtect(slot, sizeof(void *), old_protect, &old_protect);
        *original_out = old;
        log_fmt("  -> forwarder patched, original=0x%08lX", (unsigned long)(ULONG_PTR)old);
        return TRUE;
    }

    /* 情形 B: 标准热补丁序言, 用 trampoline。 */
    if (memcmp(p, HOTPATCH, PROLOGUE_LEN) != 0) {
        log_fmt("  -> prologue mismatch, skip");
        return FALSE;
    }

    tramp = (BYTE *)VirtualAlloc(NULL, 16, MEM_COMMIT | MEM_RESERVE,
                                 PAGE_EXECUTE_READWRITE);
    if (!tramp) {
        log_fmt("  -> VirtualAlloc failed");
        return FALSE;
    }

    memcpy(tramp, p, PROLOGUE_LEN);
    tramp[PROLOGUE_LEN] = 0xE9;  /* jmp target+5 */
    *(DWORD *)(tramp + PROLOGUE_LEN + 1) =
        (DWORD)((p + PROLOGUE_LEN) - (tramp + PROLOGUE_LEN + 5));

    if (!VirtualProtect(p, PROLOGUE_LEN, PAGE_EXECUTE_READWRITE, &old_protect)) {
        VirtualFree(tramp, 0, MEM_RELEASE);
        log_fmt("  -> VirtualProtect failed");
        return FALSE;
    }
    p[0] = 0xE9;
    *(DWORD *)(p + 1) = (DWORD)((BYTE *)detour - p - 5);
    VirtualProtect(p, PROLOGUE_LEN, old_protect, &old_protect);
    FlushInstructionCache(GetCurrentProcess(), p, PROLOGUE_LEN);

    *original_out = tramp;
    log_fmt("  -> hooked, trampoline=0x%08lX", (unsigned long)(ULONG_PTR)tramp);
    return TRUE;
}

static BOOL hook_export(const char *module, const char *name,
                        void *detour, void **original_out)
{
    HMODULE h = GetModuleHandleA(module);
    FARPROC fn;
    if (!h) {
        log_fmt("  module %s not loaded", module);
        return FALSE;
    }
    fn = GetProcAddress(h, name);
    if (!fn) {
        log_fmt("  %s!%s not found", module, name);
        return FALSE;
    }
    log_fmt("  try %s!%s @0x%08lX", module, name, (unsigned long)(ULONG_PTR)fn);
    return install_hook((void *)fn, detour, original_out);
}

static BOOL hook_first(const char *const *modules, int count, const char *name,
                       void *detour, void **original_out)
{
    int i;
    for (i = 0; i < count; ++i) {
        if (hook_export(modules[i], name, detour, original_out))
            return TRUE;
    }
    return FALSE;
}

static DWORD scale_time(Scaler *s)
{
    DWORD real = s->original();
    DWORD delta;
    LONG speed;
    DWORD out;

    EnterCriticalSection(&s->cs);
    delta = real - s->last_real;
    s->last_real = real;
    speed = InterlockedCompareExchange(&g_speed_milli, 0, 0);
    if (speed == 1000) {
        s->scaled = real;  /* 不缩放时保持原值, 完全透明 */
    } else {
        s->scaled += (DWORD)((double)delta * (double)speed / 1000.0);
    }
    out = s->scaled;
    LeaveCriticalSection(&s->cs);
    return out;
}

static DWORD WINAPI hk_GetTickCount(void)
{
    return scale_time(&g_tick);
}

static DWORD WINAPI hk_timeGetTime(void)
{
    return scale_time(&g_time);
}

static void install_all(void)
{
    static const char *tick_modules[] = { "kernel32.dll", "kernelbase.dll" };
    static const char *time_modules[] = { "winmm.dll", "winmmbase.dll" };
    int i;

    InitializeCriticalSection(&g_tick.cs);
    InitializeCriticalSection(&g_time.cs);

    log_fmt("[speedhack] install_all start, cfg=%s", g_cfg_path);

    /* 等 winmm 被加载 (游戏初始化声音/计时器时会加载) */
    for (i = 0; i < 300 && !GetModuleHandleA("winmm.dll"); ++i)
        Sleep(10);
    log_fmt("[speedhack] winmm loaded=%d (waited %d ms)", GetModuleHandleA("winmm.dll") != NULL, i * 10);

    /* 只钩一个时钟, 避免 timeGetTime 内部又走 GetTickCount 导致叠加缩放。
     * 优先 timeGetTime (东方主循环用它); 不可用再退到 GetTickCount。 */
    if (hook_first(time_modules, 2, "timeGetTime", (void *)hk_timeGetTime,
                   (void **)&g_time.original)) {
        g_time.last_real = g_time.original();
        g_time.scaled = g_time.last_real;
        g_time.ready = 1;
        log_fmt("[speedhack] timeGetTime hooked");
    } else {
        log_fmt("[speedhack] timeGetTime NOT hooked");
        if (hook_first(tick_modules, 2, "GetTickCount", (void *)hk_GetTickCount,
                       (void **)&g_tick.original)) {
            g_tick.last_real = g_tick.original();
            g_tick.scaled = g_tick.last_real;
            g_tick.ready = 1;
            log_fmt("[speedhack] GetTickCount hooked (fallback)");
        } else {
            log_fmt("[speedhack] GetTickCount NOT hooked");
        }
    }
    log_fmt("[speedhack] install_all done");
}

static void read_factor(void)
{
    HANDLE f;
    char buf[32];
    DWORD read = 0;
    double factor;
    LONG current;

    if (!g_cfg_path[0])
        return;
    f = CreateFileA(g_cfg_path, GENERIC_READ,
                    FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (f == INVALID_HANDLE_VALUE)
        return;
    ZeroMemory(buf, sizeof(buf));
    ReadFile(f, buf, sizeof(buf) - 1, &read, NULL);
    CloseHandle(f);

    factor = atof(buf);
    if (factor > 0.01 && factor < 100.0) {
        LONG milli = (LONG)(factor * 1000.0);
        current = InterlockedCompareExchange(&g_speed_milli, 0, 0);
        if (milli != current) {
            InterlockedExchange(&g_speed_milli, milli);
            log_fmt("[speedhack] factor -> %.3f", factor);
        }
    }
}

static DWORD WINAPI worker(LPVOID param)
{
    (void)param;
    install_all();
    for (;;) {
        read_factor();
        Sleep(200);
    }
    return 0;
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD reason, LPVOID reserved)
{
    (void)reserved;
    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(hModule);
        GetTempPathA(MAX_PATH, g_cfg_path);
        strcat(g_cfg_path, "touhou_speedhack.cfg");
        GetTempPathA(MAX_PATH, g_log_path);
        strcat(g_log_path, "touhou_speedhack.log");
        CreateThread(NULL, 0, worker, NULL, 0, NULL);
    }
    return TRUE;
}

__declspec(dllexport) void WINAPI SetSpeed(float factor)
{
    if (factor > 0.01f && factor < 100.0f)
        InterlockedExchange(&g_speed_milli, (LONG)(factor * 1000.0f));
}
