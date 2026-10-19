# 内存偏移标定手册

MCP 能观测到多少东西，完全取决于 `touhou_rl_mcp/games/offsets/thXX.json` 里
`fields` 标定了多少。本文是从零开始标定的完整流程（用 MCP 自带工具，
不需要装 Cheat Engine）。

## 0. 先搞清楚"三种定位方式"

东方的内存数据分两类，用不同方式描述：

| 数据 | 存放位置 | JSON 写法 |
|---|---|---|
| 全局游戏数据（分数、残机、灵力、关卡、符卡编号、自机选择） | 模块的 `.data` 段，**固定偏移** | `{"offset": "0xA56E4", "type": "i32"}` |
| 自机坐标、Boss 血量、弹幕链表、敌机链表（th10+ 都在堆上） | 堆，**指针链** | `{"pointer": "0xA56C0", "offsets": ["0x18", "0x24"], "type": "f32"}` |

指针链的解析规则（就是 Cheat Engine 的记法）：

```
"pointer": "0xA56C0", "offsets": ["0x18", "0x24"]
   ==  f32( read_u32( read_u32( base + 0xA56C0 ) + 0x18 ) + 0x24 )
   ==  CE:  [[th11c.exe+A56C0]+18]+24
```

规则：`pointer` 处读一个 4 字节指针 → 加上 `offsets[0]` 后再读一个指针 →
…… → 最后加上 `offsets[-1]`（**最后一项只相加，不再解引用**）。
只有一层指针（`[mod+ptr]+18`）就写 `"offsets": ["0x18"]`。

也可以写绝对地址：东方正作是**无 ASLR** 的 32 位程序，基址固定（通常
`0x400000`），所以绝对地址跨进程也稳定：`{"address": "0x1234ABCD"}`。
不过优先用 `offset` / `pointer`，更可移植。

类型：`i8 u8 i16 u16 i32 u32 f32 f64`。
变换：`mul10`（th11 分数内存值是 1/10）、`div20`（th11 灵力内存值是 20 倍）等。

## 1. 准备

```
th_attach({"backend": "memory", "game": "th11"})     # 会启动/连接游戏
th_env_status({})                                     # 看 pid / 窗口 / module_base
th_start_new_game({"difficulty": 1})                  # 进入游戏
```

如果 `th_env_status` 里 `module_base` 为 `null`，说明还没连上进程。

> 游戏请**窗口化**运行。全屏下按键注入和截图都不可靠。

## 2. 标定"整数型"字段（最简单，建议先练手）

以 **擦弹数 `graze`** 为例：

```
1. 让擦弹数停在 0
2. th_memory_scan({"value": 0, "type": "i32", "limit": 500})
   -> 候选很多 (0 到处都是)
3. 去擦几颗弹, 擦弹变成 11
4. th_memory_filter({"value": 11, "type": "i32"})
   -> 候选骤减
5. 再擦几颗, 变成 23
6. th_memory_filter({"value": 23, "type": "i32"})
7. th_memory_candidates({})   # 候选应该只剩个位数
8. th_memory_read({"address": "0x1234ABCD", "type": "i32"})
   -> 反复擦弹, 看数值是否跟着涨; 是则命中
9. 把候选里的 "as_offset" (形如 th11c.exe+0xA1234) 写进 offsets json
```

**技巧**

- 用"变化明显 + 只增不减"的量最容易：分数、擦弹、已收集的符卡数。
- `limit` 别设太大（默认 500 够用），否则候选列表 JSON 很长，浪费上下文。
- 若首次扫描 0 候选太多，可以先让值 > 0 再扫（例如擦到 30 再扫 30）。
- `include_images: true` 会连模块映像区一起扫 —— **全局游戏数据就在映像区里**，
  如果默认扫描找不到你要的值，把它打开重扫一次。
- 候选列表保存在 `.runtime/scanner/pid<pid>.json`，按进程分开存。

## 3. 标定"指针型"字段（自机坐标 / Boss 血量）

`player_x` 是 32 位浮点。**别直接扫 192.0**（f32 精度下你很难凑对数值），
用下面两条更可靠的路子：

### 路线 A：用"坐标整数化"或"分数的邻近字段"定位玩家结构

th10+ 的玩家结构里，很多字段挨着放。先用**指针扫描法**：

1. 用 `th_visualize_state` 或目测记下自机 x（比如 `192`），并精确记下
   一个**稳定不变**的量（比如自机所在 x 不动时）。
2. 在 CE 里对已知坐标地址做 `Find out what accesses this address`，看
   `base register` 的偏移 —— 这是最省事的办法，本 MCP 没有实现"谁访问了
   这个地址"（需要调试器），**这一步建议用 Cheat Engine 或
   `E:\thManager\TouHouManager\overlay\scanner.py` 配合完成**。
3. 拿到形如 `[[th11c.exe+A56C0]+18]+24` 的链后，用
   `th_memory_read({"address": "th11c.exe+A56C0", "offsets": ["0x18", "0x24"], "type": "f32"})`
   验证：移动自机，读数应该跟着变。
4. 写进 offsets json：

```json
"player_x": { "pointer": "0xA56C0", "offsets": ["0x18", "0x24"], "type": "f32", "label": "自机X" },
"player_y": { "pointer": "0xA56C0", "offsets": ["0x18", "0x28"], "type": "f32", "label": "自机Y" }
```

### 路线 B：扫描"指向已知坐标的指针"（纯 MCP 工具）

1. 记下自机 x 的**原始 f32 位模式**：例如 `192.0` 的位模式是 `0x43400000`
   （= 十进制 `1128267776`）。
2. `th_memory_scan({"value": 1128267776, "type": "u32"})` —— 这会找到那 4 个
   字节 = 192.0f 的地址 `A`。
3. 再用 `A`（作为 u32 数值）去扫一次：`th_memory_scan({"value": <A 的十进制值>, "type": "u32"})`
   —— 这一步会找到**指向 A 的指针变量**，它通常在模块 `.data` 里，
   位置相对模块基址固定 → 就是你要的 `pointer`。
4. `th_memory_read({"address": "0x<指针变量地址>", "offsets": [...]})` 验证。

### Boss 血量

Boss 血量是**只减不增**的 i32，且变化明显：

1. 进 Boss 战，不射击，等血量稳定 → 记不住精确值也没关系。
2. 打两下让血条掉一格 → `th_memory_scan` 扫当前值（需要一个精确数字，
   可以先用 `th_memory_read` 试读几个候选）。
3. 更实用的做法：**扫描"变化了的地址"** ——
   `th_memory_filter_changed({"expect": "changed"})` 配合"打/不打"的节奏，
   从一批候选里筛出只在你射击时减少的地址。

标准流程：

```
th_memory_scan({"value": 0, "type": "i32", "include_images": true})   # 占位, 建立候选
... 进入 Boss 战, 让血量从 X 变成 Y ...
th_memory_filter({"value": Y, "type": "i32"})
th_memory_filter({"value": Z, "type": "i32"})   # 反复
```

## 4. 标定 `frame`（帧计数）—— 强烈推荐

有了 `frame` 偏移，`advance()` 会从"墙钟 sleep"切换成**严格等待帧计数递增**，
帧同步会稳得多（宏观动作的精度直接取决于它）。

- 帧计数是一个**每帧 +1 的 i32**（也可能每帧 +2）。
- 用 `th_memory_filter_changed` 很难扫（它每帧都变）。更实际的办法：
  扫一个"每帧递增"的量的思路不可行，**建议在 CE 里对已知的帧计数器做
  `Find out what accesses`**，或直接扫"当前帧数的大致值"（比如开始游戏 10 秒后
  约 600）+ 用 `th_memory_read` 连续读两次看是否 +1/**+2**。
- 标定后写：`"frame": {"offset": "0x...", "type": "i32"}`，
  并在 `config.json` 里把 `control.tick_hz` 设成 **30**（若计数器每帧 +2）。

## 5. 标定完怎么验证

```
th_env_status({})          # calibrated_fields / uncalibrated_fields 应该更新
th_start_new_game({})
th_get_state({})           # player 不再是 null
th_step({"action": "move_to", "x": 100, "y": 200, "frames": 200})
th_get_state({})           # player.x ≈ 100
th_visualize_state({})     # 图上应该出现自机、弹幕、Boss 血条
```

## 6. 各作品现状

| 作品 | 已标定 | 备注 |
|---|---|---|
| th11 东方地灵殿 | `score` `lives` `power` `stage` `spell_id` `char_main` `char_sub` | 来自 `D:\东方AI实时分析\th11.py`；其余待标定 |
| th10 东方风神录 | — | 与 th11 内存布局接近，`th11` 的偏移**可能只差一个常量**，值得先试 |
| th06 / th07 / th08 | — | 老引擎（DX8 时代），布局与 th10+ 完全不同 |
| th12 / th13 / th14 | — | 与 th10/th11 同系，可先试 th11 偏移 ± 常量 |

标定 th10/th12 时的高效做法：先假设 `score` 偏移与 th11 相同，用
`th_memory_read({"address": "th10c.exe+0xA56E4", "type": "i32"})` 试读，
再在附近 ±0x400 范围内批量试读（用 `th_memory_read` 逐个试），
找到读数像"分数"的那个（打敌人会涨）。

## 7. 把标定结果固化下来

标定完后直接编辑 `touhou_rl_mcp/games/offsets/thXX.json`：

```json
"graze": { "offset": "0xA5740", "type": "i32", "label": "擦弹数" },
"player_x": { "pointer": "0xA56C0", "offsets": ["0x18", "0x24"], "type": "f32" },
"frame": { "offset": "0xA5744", "type": "i32", "label": "帧计数" }
```

改完不需要重启 MCP —— `th_attach` 会重新加载 `GameRegistry`：

```
th_attach({"backend": "memory", "game": "th11"})
```

（如果用的是同一个文件句柄，直接 `th_env_status` 看 `calibrated_fields` 是否已更新；
未更新就调一次 `th_close_env({})` 再 `th_attach`。）
