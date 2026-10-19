/*
 * timer_test.c —— 时间缩放钩子的自检靶子 (32 位)
 *
 * 在 ~3 秒真实时间里循环调用 timeGetTime, 打印"游戏时间"经过了多少毫秒。
 * 真实时间用 QueryPerformanceCounter 测 (不被钩子影响), 游戏时间用
 * timeGetTime 测 (会被 speedhack32.dll 缩放)。
 *
 *   game_ms ≈ 3000 * factor
 *
 * 用法: timer_test32.exe [毫秒]
 */

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <mmsystem.h>

int main(int argc, char **argv)
{
    LARGE_INTEGER freq, begin, now;
    double real_ms;
    double target_ms = 3000.0;
    DWORD start_game, game_ms;

    if (argc > 1)
        target_ms = atof(argv[1]);
    if (target_ms <= 0)
        target_ms = 3000.0;

    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&begin);
    start_game = timeGetTime();

    do {
        QueryPerformanceCounter(&now);
        real_ms = (double)(now.QuadPart - begin.QuadPart) * 1000.0 / (double)freq.QuadPart;
        Sleep(1);
    } while (real_ms < target_ms);

    game_ms = timeGetTime() - start_game;
    printf("real_ms=%.0f game_ms=%lu ratio=%.3f\n", real_ms, (unsigned long)game_ms,
           real_ms > 0 ? (double)game_ms / real_ms : 0.0);
    return 0;
}
