/*
 * injector.c —— 32 位 DLL 注入器 (injector32.exe)
 *
 * 用法: injector32.exe <pid> <dll 绝对路径>
 *
 * 为什么单独做一个 32 位 exe: 东方正作是 32 位程序, 从 64 位 Python 直接
 * 注入 32 位进程需要处理 WOW64 的模块地址差异; 用一个同位数(32 位)的
 * 小注入器最稳 —— kernel32 的基址在同类进程间一致, LoadLibraryA 地址通用。
 *
 * 退出码: 0 成功; 1 目标 LoadLibrary 返回 0 (加载失败); 2 参数/句柄错误。
 */

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    DWORD pid;
    const char *dll_path;
    HANDLE process, thread;
    SIZE_T len;
    LPVOID remote_mem;
    HMODULE kernel32;
    FARPROC load_library;
    DWORD exit_code = 0;

    if (argc < 3) {
        fprintf(stderr, "usage: injector32.exe <pid> <dll-path>\n");
        return 2;
    }
    pid = (DWORD)strtoul(argv[1], NULL, 0);
    dll_path = argv[2];
    len = strlen(dll_path) + 1;

    process = OpenProcess(PROCESS_CREATE_THREAD | PROCESS_QUERY_INFORMATION |
                              PROCESS_VM_OPERATION | PROCESS_VM_WRITE | PROCESS_VM_READ,
                          FALSE, pid);
    if (!process) {
        fprintf(stderr, "OpenProcess(%lu) failed: %lu\n", (unsigned long)pid,
                (unsigned long)GetLastError());
        return 2;
    }

    remote_mem = VirtualAllocEx(process, NULL, len, MEM_COMMIT | MEM_RESERVE,
                                PAGE_READWRITE);
    if (!remote_mem) {
        fprintf(stderr, "VirtualAllocEx failed: %lu\n", (unsigned long)GetLastError());
        CloseHandle(process);
        return 2;
    }
    if (!WriteProcessMemory(process, remote_mem, dll_path, len, NULL)) {
        fprintf(stderr, "WriteProcessMemory failed: %lu\n", (unsigned long)GetLastError());
        VirtualFreeEx(process, remote_mem, 0, MEM_RELEASE);
        CloseHandle(process);
        return 2;
    }

    kernel32 = GetModuleHandleA("kernel32.dll");
    load_library = GetProcAddress(kernel32, "LoadLibraryA");

    thread = CreateRemoteThread(process, NULL, 0,
                                (LPTHREAD_START_ROUTINE)load_library,
                                remote_mem, 0, NULL);
    if (!thread) {
        fprintf(stderr, "CreateRemoteThread failed: %lu\n", (unsigned long)GetLastError());
        VirtualFreeEx(process, remote_mem, 0, MEM_RELEASE);
        CloseHandle(process);
        return 2;
    }

    WaitForSingleObject(thread, 10000);
    GetExitCodeThread(thread, &exit_code);

    CloseHandle(thread);
    VirtualFreeEx(process, remote_mem, 0, MEM_RELEASE);
    CloseHandle(process);

    if (exit_code == 0) {
        fprintf(stderr, "LoadLibraryA returned NULL in target (injection failed)\n");
        return 1;
    }
    printf("injected ok (module handle = 0x%lX)\n", (unsigned long)exit_code);
    return 0;
}
