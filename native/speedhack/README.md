# speedhack —— 真机时间缩放钩子

东方正作是 32 位程序，主循环用 `timeGetTime()`（winmm）取时间。本目录的 DLL 钩住
它，让游戏按 `factor` 倍速运行：

| factor | 效果 | 用途 |
|---|---|---|
| `> 1` | **加速**（例如 4x） | 真机训练时提高每秒帧数 → 提高训练吞吐 |
| `= 1` | 原速（透明，不做任何缩放） | 默认 |
| `< 1` | **减速**（例如 0.25x） | 便于观察，或给慢速 Agent 留出决策时间 |

> 注意方向：本地策略本身很快，瓶颈是游戏只有 60fps。**加速**才能提高真机训练吞吐；
> 减速是给"人 / 慢速 Agent 看"用的。

## 构建

需要 Visual Studio 的 32 位工具链：

```powershell
powershell -ExecutionPolicy Bypass -File native\speedhack\build.ps1
```

产物（`native/speedhack/bin/`）：

- `speedhack32.dll` —— 钩子本体
- `injector32.exe` —— 32 位注入器（东方是 32 位，用同位数注入器最稳）
- `timer_test32.exe` —— 自检靶子（测 `timeGetTime` 是否被缩放）

## 原理与自检

- 速度因子写在 `%TEMP%\touhou_speedhack.cfg`（纯文本），DLL 每 200ms 重读，
  所以**运行中可动态改速度**。
- 钩子优先钩 `timeGetTime`；它不可用（例如非 winmm 游戏）才退到 `GetTickCount`。
  **只钩一个**，避免 `timeGetTime` 内部又走 `GetTickCount` 造成叠加缩放。
- 兼容转发器：`winmm!timeGetTime` 常是 `jmp [IAT]` 桩，本 DLL 会直接改它跳转的
  IAT 槽；标准热补丁序言 `8B FF 55 8B EC` 则用 trampoline。序言不认识就跳过，
  绝不硬改。
- 诊断日志：`%TEMP%\touhou_speedhack.log`。

自检（不需要游戏）：

```powershell
$bin = "native\speedhack\bin"
Set-Content "$env:TEMP\touhou_speedhack.cfg" "4.0"
& "$bin\timer_test32.exe" 3000                 # 基线: ratio ≈ 1.0
$p = Start-Process "$bin\timer_test32.exe" -ArgumentList 3000 -PassThru -NoNewWindow
Start-Sleep -Milliseconds 400
& "$bin\injector32.exe" $p.Id "$bin\speedhack32.dll"
$p.WaitForExit()
```

`timer_test32.exe` 用 `QueryPerformanceCounter` 测真实时间、用 `timeGetTime` 测"游戏
时间"，打印 `ratio = game_ms / real_ms`，应接近 factor（注入前那几百毫秒未缩放，
会把 ratio 略微拉低，属正常）。

## 在 MCP 里使用

```jsonc
// 方式一: 配置 (下次 attach 生效)
{ "slowdown": { "enabled": true, "factor": 4.0 } }

// 方式二: 运行中直接调用 (立即注入并生效)
th_set_game_speed({ "factor": 4.0 })
```

## 限制

- 仅 Windows + 32 位游戏；DLL 依赖 `timeGetTime` / `GetTickCount` 中至少一个可用。
- 极端 factor（>10x）可能让游戏逻辑/音频不稳；建议 2x~6x。
- 未在真机（th06~th14）上逐一验证 —— 自检只证明钩子机制正确。若某作品不响应，
  先看 `%TEMP%\touhou_speedhack.log` 是否显示 "hooked"。
