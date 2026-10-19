# Touhou RL MCP

把**东方 Project 正作**（th06 红魔乡 ~ th14 辉针城）包装成一个标准强化学习环境，
再以 **MCP 服务器**的形式暴露出来，供 LLM Agent / Agent 平台驱动，从而实现
**agentic RL**：让 Agent 一边打东方，一边根据回报调整自己的行为与训练参数。

```
                      ┌──────────────────────────────┐
   LLM Agent  ──MCP──▶│  touhou-rl-mcp (HTTP-less)   │
   / RL 训练循环       │  reset / step / get_state    │
                      │  screenshot / visualize      │
                      │  configure(reward 权重)      │
                      └───────────┬──────────────────┘
                                  │
                 ┌────────────────┴─────────────────┐
                 ▼                                  ▼
        backend = "mock"                   backend = "memory"
        纯 Python 弹幕模拟器               真实游戏: 读进程内存 + 注入按键
        可复现、无需游戏本体                + 窗口截图
```

---

## 1. 快速开始

实机观测更新（2026-10-17）：TH11 1.00a 已接入普通弹幕、敌人坐标和擦弹，
统一为场地坐标；读取失败将明确报错，不再静默用全零弹幕训练。
适用范围、验证结果与真机启动命令见 [TH11 实体观测说明](docs/TH11_ENTITIES.md)。
激光及其它作品实体布局仍未支持。

新增 Boss 血量/符卡 Bonus 奖励、空场对话射击点按及六机体独立继续训练，
用法和验证边界见 [战斗训练说明](docs/COMBAT_TRAINING.md)。

```powershell
# 1) 环境已经建好 (E:\TouHou-Ai\.venv)，如需重建:
C:\Python313\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2) 自检: 用内置模拟器跑一局 (不需要游戏)
.\.venv\Scripts\python.exe -m touhou_rl_mcp selftest --dodge --steps 200
#   -> 输出每步状态、总回报、统计, 并在 .runtime\viz\ 下生成一张状态可视化 PNG

# 3) 单元 + 集成测试 (含实体读取 / 真实 stdio MCP 会话 / 训练控制面 / 标定工具)
.\.venv\Scripts\python.exe -m unittest discover -s tests

# 4) 直接训练本地智能体 (不经过 MCP; 与 th_train_start 用的是同一个 Trainer)
.\.venv\Scripts\python.exe examples\train.py --backend mock --episodes 300
```

## 2. 接入 OpenCode / Agent 平台

本目录下已经放好 `opencode.jsonc`（OpenCode V2 的 `mcp.servers` 写法）：

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "servers": {
      "touhou": {
        "type": "local",
        "command": ["E:\\TouHou-Ai\\.venv\\Scripts\\python.exe", "-m", "touhou_rl_mcp"],
        "cwd": "E:\\TouHou-Ai"
      }
    }
  }
}
```

其它客户端（Claude Desktop / Cherry Studio / Cursor 等）通用 `mcpServers` 写法：

```json
{
  "mcpServers": {
    "touhou": {
      "command": "E:\\TouHou-Ai\\.venv\\Scripts\\python.exe",
      "args": ["-m", "touhou_rl_mcp"],
      "cwd": "E:\\TouHou-Ai"
    }
  }
}
```

也可以直接跑起来看它说话：

```powershell
.\.venv\Scripts\python.exe -m touhou_rl_mcp --backend mock --game th11    # stdio
```

---

## 3. 工具一览（30 个）

| 工具 | 作用 |
|---|---|
| `th_list_games` | 支持的作品 / 本机是否安装 / 各作品偏移标定情况 |
| `th_attach` | 切换后端与作品（`mock` / `memory`）并连接 |
| `th_env_status` | 当前后端、进程、窗口、已标定字段、本局进度 |
| **`th_start_new_game`** | **开始新一局（reset）**，返回初始状态 |
| **`th_step`** | **执行动作 → `{state, reward, done, truncated, info}`** |
| **`th_get_state`** | **查询当前状态（不推进时间）** |
| **`th_screenshot`** | **截图（真实后端抓游戏窗口，模拟器渲染合成画面）→ 返回图片** |
| **`th_visualize_state`** | **状态可视化（坐标网格 + 判定点 + 弹幕 + Boss 血条 + 数值 HUD）→ 返回图片** |
| `th_send_keys` | 发送原始按键序列（菜单导航 / 调试） |
| `th_get_config` / `th_configure` | 读取 / **在线修改配置与奖励权重**（agentic RL 调参入口） |
| `th_episode_stats` | 跨局滚动统计：平均回报、到达面数、通关率、掉残、擦弹 |
| `th_memory_scan` / `th_memory_filter` / `th_memory_filter_changed` | 内存首扫 / 再扫（标定偏移量，类似 Cheat Engine） |
| `th_memory_read` / `th_memory_candidates` / `th_memory_clear_candidates` | 读地址 / 查看 / 清空候选 |
| **`th_memory_find_player`** | **全自动定位自机坐标（撞墙法）→ 可一键写入 offsets** |
| **`th_memory_find_frame`** | **全自动定位帧计数器 frame（用于严格帧同步）** |
| **`th_offsets_write`** | **把标定结果写回 offsets json（手动标定后的收尾）** |
| **`th_set_game_speed`** | **真机时间缩放：加速提高训练吞吐 / 减速观察** |
| `th_close_env` | 松开所有按键、断开连接（不关游戏进程） |
| **`th_train_start`** | **启动/继续本地智能体训练（后台线程，不占用 MCP 往返）** |
| **`th_train_status`** | **观测训练进度：运行状态、学习曲线摘要、最近若干局、策略结构** |
| **`th_train_dashboard`** | **训练仪表盘（当前画面 + 学习曲线 + 实时统计）→ 返回图片** |
| **`th_train_configure`** | **在线改训练超参（lr/gamma/隐藏层/批量…），可同时改奖励** |
| `th_train_pause` | 暂停 / 恢复训练（不释放线程） |
| `th_train_stop` | 停止训练并保存策略 |
| `th_train_policy` | 策略的保存 / 载入 / 查看 |

### 一次典型的 Agent 交互

```
th_attach({"backend": "mock", "game": "th11"})
th_start_new_game({"seed": 42})
th_step({"action": "move_to", "x": 192, "y": 400, "frames": 180})
th_step({"action": "dodge", "frames": 60})
th_get_state({"include_bullets": true})
th_visualize_state({})            # -> 图片
th_configure({"reward": {"miss_penalty": -40, "graze_scale": 0.05}})
th_episode_stats({})
```

### 让 Agent 当"教练"训练本地智能体（推荐玩法）

LLM 反应慢，不适合逐帧亲自打游戏。正确的分工是：**本地策略（`MLPPolicy`）在后台线程
以最高速度打，LLM 只负责启动 / 观测 / 调参**，用奖励与超参加速收敛：

```
th_train_start({"config": {"batch_episodes": 16, "lr": 1e-3}})
th_train_status({"tail": 5})      # 看学习曲线摘要、平均面数
th_train_dashboard({})            # -> 仪表盘图片 (画面 + 曲线 + 统计)
th_train_configure({"patch": {"reward": {"danger_scale": 1.5, "alive_bonus": 0.3}}})
th_train_pause({"paused": true})  # 想细看时暂停
th_train_policy({"action": "save"})
th_train_stop({"save": true})
```

## 4. 动作空间（`th_step` 的 `action`）

三种形式可以混用：

| 形式 | 例子 |
|---|---|
| 按键组合 | `"right"`、`"right+shoot"`、`"up,left,focus"`（`+` `,` 空格 `;` 都可作分隔） |
| 简单动词 | `noop` `shoot` `focus` `focus_shoot` `bomb` `confirm` `cancel` `pause` |
| **宏动作** | `move_to` `dodge` `wait_in_game` |

宏动作是关键：**LLM 只能以 1~10 Hz 决策，而游戏跑在 60fps**。
`move_to`（闭环开到指定坐标）、`dodge`（基于最近弹幕斥力的躲避辅助）由控制器
在内部以 60Hz 逐帧执行，Agent 只需给出高层意图。

```jsonc
// move_to: 把自机开到场地坐标 (x, y)；场地 384x448，原点左上
{ "action": "move_to", "x": 192, "y": 400, "frames": 240, "timeout_frames": 180 }

// dodge: 内置避弹辅助，radius 是感知半径
{ "action": "dodge", "frames": 120, "radius": 40 }
```

> `move_to` / `dodge` 需要 `player_x` / `player_y` 偏移（`dodge` 还需要弹幕）。
> 模拟器自带这些字段；真实游戏请按第 7 节标定。

## 5. 状态字段

```jsonc
{
  "frame": 1234,            // 帧计数 (真实游戏需标定 frame 偏移, 否则为 0)
  "in_game": true,          // 是否在局内 (启发式判定)
  "stage": 2, "lives": 2, "power": 3.45, "score": 123456, "graze": 88,
  "spell_id": 43, "spell_name": "罠符「捕捉之网」",
  "char_main": 0, "char_sub": 0, "char_name": "灵梦A",
  "player": {"x": 192.0, "y": 400.0},
  "boss": {"hp": 422, "max_hp": 1040, "name": "模拟Boss 2面"},
  "enemy_positions": [[x, y], ...],
  "bullet_count": 143,
  "bullets": [[x, y], ...],   // include_bullets=true 时返回, 最多 256 个
  "raw": { "score": 12345, ... }   // 所有偏移的原始读数, 标定偏移量时对照用
}
```

## 6. 奖励函数（可在线调参）

`reward` 段每一项都有独立权重，默认值见 `config.json`：

| 权重 | 含义 |
|---|---|
| `score_scale` `graze_scale` `power_scale` | Δ分数 / Δ擦弹 / Δ灵敏度 的缩放 |
| `life_bonus` / `miss_penalty` / `death_penalty` | 加残 / 掉残 / Game Over |
| `stage_clear_bonus` / `game_clear_bonus` | 过面 / 通关 |
| `boss_damage_scale` | 对 Boss 造成伤害（需 `boss_hp` 偏移） |
| `spell_capture_bonus` | 真机确认收取符卡 Bonus 的一次性奖励 |
| `spell_clear_bonus` | 击破一张符卡（mock 由 `spell_cleared` 事件触发） |
| `invincible_output_scale` | **无敌帧期间 分数/Boss伤害/存活奖励 的缩放（默认 0 = 封堵"被弹刷输出"）** |
| `danger_scale` / `danger_radius` | 危险度惩罚（越靠近弹幕扣越多；教 RL 躲弹幕的密集信号） |
| `alive_bonus` | 每个 step 的存活奖励（无敌期不给） |
| `time_penalty` | 每 step 固定惩罚，鼓励速通 |
| `clip` | 单步奖励裁剪 |
| `shaping_scale` | **整体缩放 —— Agent 在线调参的总开关** |

`th_step` 返回的 `info.reward_parts` 会给出**每个分量的贡献**，
这样 Agent 能直接看到"我这次死掉扣了 100 分"，据此调整权重或策略：

```json
{"reward": -139.83, "info": {"reward_parts": {"score": 0.0009, "graze": 0.06, "miss": -50.0, "death": -100.0, "time": -0.002}}}
```

## 7. 真实游戏（`backend = "memory"`）

### 7.1 准备工作

1. **窗口化运行游戏**（游戏内 Options 或 `thXXconfig.ini`）。独占全屏会同时
   让截图和按键注入不可靠。
2. 确认游戏安装目录。`config.json` 的 `search_roots` 默认为 `F:\ E:\ D:\`，
   `th_list_games` 会自动在这些盘的下一级目录里找 `thXXc.exe`。
   也可以显式指定：`th_attach({"backend": "memory", "game": "th11", "game_dir": "F:\\[th11] 东方地灵殿 (汉化版+日文版)"})`。
3. 若进程没在跑，`th_attach(backend="memory")` 会替你启动它。

### 7.2 偏移量现状

| 作品 | 状态 |
|---|---|
| **th11 东方地灵殿 1.00a** | ✅ 标量：`score` `lives` `power` `stage` `spell_id` `char_main` `char_sub` `graze`<br>✅ 自机坐标、普通弹幕数组、敌人链表，统一场地坐标，版本签名检查<br>⚠️ 未接入：激光、`boss_hp`、`frame`、`invincible`；TH11 Bomb 消耗 power，无独立炸弹数量 |
| th06 / th07 / th08 / th10 / th12 / th13 / th14 | ⚠️ 仅有菜单导航模板，内存偏移待标定 |

标定后的字段决定了**能观测到什么**：

| 已标定字段 | 解锁的能力 |
|---|---|
| `score` `lives` `power` `stage` | 完整奖励信号、`in_game` 判定、过面/掉残事件 |
| `graze` | 擦弹奖励（鼓励贴弹，是东方打法的核心） |
| `player_x` `player_y` | `move_to` 宏、坐标可视化、自机轨迹 |
| 弹幕相关 | `dodge` 宏、弹幕可视化 |
| `boss_hp` | `boss_damage_scale` 伤害奖励、Boss 血条 |
| `frame` | 严格的帧同步（默认用 60Hz 墙钟同步） |

### 7.3 用 MCP 自己标定偏移（不需要 Cheat Engine）

```
1. th_attach({"backend": "memory", "game": "th11"})
2. th_start_new_game({})                     # 进入游戏
3. 记住自机当前 x 坐标 (用 th_visualize_state 或目测)
4. th_memory_scan({"value": 192.0, "type": "f32", "align": 4})
5. 让自机移动一点, 读新坐标 (如 180.5)
6. th_memory_filter({"value": 180.5, "type": "f32"})     # 反复 4~6 直到候选很少
7. th_memory_read({"address": "0x1234ABCD", "type": "f32"})   # 验证: 移动自机看数值是否跟着变
8. 把结果的 "as_offset" (形如 th11c.exe+0xA1234) 写进
   touhou_rl_mcp/games/offsets/th11.json 的 fields.player_x
```

扫描对**整数型**的量（分数、擦弹）更容易：它们变化明显、候选收敛快。
详细流程与技巧见 [`docs/OFFSET_CALIBRATION.md`](docs/OFFSET_CALIBRATION.md)。

**更省事：自动标定 + 一键写回**（`player_x/y` 与 `frame` 可全自动）

```
th_attach({"backend": "memory", "game": "th11"})
th_start_new_game({})
th_memory_find_player({"write": true})   # 撞墙法找自机坐标, 直接写进 offsets
th_memory_find_frame({"write": true})    # 找帧计数器, 解锁严格帧同步
# 手动标定别的量后收尾 (写回是原子的, 默认留 .json.bak 备份):
th_offsets_write({"graze": {"offset": "0xA571C", "type": "i32"}})
```

命令行等价物：`python tools/probe.py find-coord --write`、`python tools/probe.py find-frame --write`、
`python tools/probe.py write-field graze 0xA571C --as-offset`。

### 7.4 按键注入方式

东方正作用 **DirectInput** 读键盘，所以 `PostMessage(WM_KEYDOWN)` 无效。
本 MCP 使用 `SendInput`（`inputs/keyboard.py`）注入，并在每次动作前用
`AttachThreadInput` + `SetForegroundWindow` 把游戏窗口置前。
**游戏窗口必须在前台**，否则按键会被系统送到别的窗口。

### 7.5 真机时间缩放（提高训练吞吐）

真机只有 60fps —— 想练得更快就得让游戏**加速**跑。`native/speedhack/` 提供一个
32 位时间缩放钩子（钩 `timeGetTime`），先构建：

```powershell
powershell -ExecutionPolicy Bypass -File native\speedhack\build.ps1
```

然后在 MCP 里直接调速（立即生效）：

```
th_set_game_speed({"factor": 4.0})   # 4 倍速, 真机训练吞吐 x4
th_set_game_speed({"factor": 1.0})   # 恢复原速
```

也可在 `config.json` 里设 `"slowdown": {"enabled": true, "factor": 4.0}`（下次 attach 生效）。
速度因子写在 `%TEMP%\touhou_speedhack.cfg`，DLL 每 200ms 重读，可运行中动态改；
诊断日志在 `%TEMP%\touhou_speedhack.log`。原理、自检与限制见
[`native/speedhack/README.md`](native/speedhack/README.md)。

---

## 8. 目录结构

```
E:\TouHou-Ai\
├── config.json                      # 运行配置 (奖励权重/控制/可视化/搜索路径)
├── opencode.jsonc                   # OpenCode V2 的 MCP 接入配置
├── requirements.txt / pyproject.toml
├── examples\rl_loop.py              # 不经过 MCP, 直接在 Python 里跑 RL 循环
├── examples\train.py                # 训练入口 (与 th_train_start 同一个 Trainer)
├── native\speedhack\                # 32 位时间缩放钩子 (DLL + 注入器 + 自检; build.ps1 构建)
├── tools\probe.py                   # 真机探针 / 内存偏移标定 CLI
├── tools\watch_training.py          # 训练实时监视窗口
├── docs\OFFSET_CALIBRATION.md       # 内存偏移标定手册
├── tests\                           # 51 个用例 (MCP 会话 / 训练 / 标定 / 时间缩放)
└── touhou_rl_mcp\
    ├── server.py                    # MCP 工具定义 (26 个)
    ├── env.py                       # TouhouEnv: reset / step / observe + 奖励 + 统计
    ├── config.py  types.py  errors.py
    ├── backends\
    │   ├── base.py                  # 后端接口
    │   ├── mock.py                  # 纯 Python 弹幕模拟器 (可复现)
    │   └── memory.py                # 真实游戏: 内存读取 + 按键注入 + 截图
    ├── games\
    │   ├── registry.py              # 作品注册表 + 安装目录自动查找
    │   └── offsets\th06..th14.json  # 各作品的内存偏移表
    ├── inputs\
    │   ├── keyboard.py              # SendInput 键盘注入 + 窗口置前
    │   ├── actions.py               # 动作字符串 -> 按键/宏 解析
    │   └── controller.py            # 60Hz 帧同步控制器 + 宏动作
    ├── memory\
    │   ├── process.py               # 进程/窗口/模块 (纯 ctypes, 支持 64 位读 32 位)
    │   ├── reader.py                # 按偏移表把内存读成 GameState
    │   └── scanner.py               # 首扫/再扫, 候选持久化
    ├── capture\
    │   ├── screenshot.py            # PrintWindow/BitBlt 多级回退截图
    │   └── visualize.py             # 状态可视化 (PIL)
    ├── rl\
    │   ├── reward.py                # 奖励函数 (权重可在线调)
    │   └── stats.py                 # 单局/跨局统计
    └── training\                    # 本地智能体训练 (LLM 当"教练", 不逐帧亲自打)
        ├── features.py              # 状态 -> 特征向量 (以自机为中心的最近 K 弹)
        ├── policy.py                # 纯 numpy MLP 策略 (REINFORCE + 基线 + 价值头)
        ├── trainer.py               # 后台线程训练器 (启动/停止/在线调参)
        ├── gym_env.py               # 可选 Gymnasium 适配 (接 SB3 / CleanRL)
        ├── visualize.py             # 学习曲线 / 仪表盘 / 帧拼图
        └── watch.py                 # Tkinter 实时监视窗口
```

## 9. 已知限制与下一步

**限制**

- **实时性**：真实游戏是 60fps 实时运行，Agent 决策慢于 60Hz 时游戏不会等你。
  所以真实后端请优先用宏动作（`dodge` / `move_to` / `frames` 较大的 `step`），
  或先用 mock 把策略与奖励调好再上真机。
- **菜单导航是 best-effort**：`menu_sequence` 中的方向键次数是估计值
  （用"先按到顶再往下数"的锚定法，对版本差异有一定容错）。
  如果导航失败，用 `th_send_keys` 手动找正确的按键序列，再改
  `games/offsets/thXX.json` 的 `menu_sequence`。
- **th11 之外的偏移量未标定**，只能拿到截图/可视化（视觉 Agent 可用），
  数值状态需要按第 7 节标定。
- **截图**：D3D8 时代的老作品在全屏下可能抓到黑屏，务必窗口化。
- `mock` 是**简化**的弹幕模拟，不是东方物理的复刻，用于跑通管线与调参，
  不能替代真机。

**下一步（按价值排序）**

1. 用 `th_memory_scan` 把 th11 的 `player_x` `player_y` `graze` `boss_hp` 补齐 →
   解锁 `move_to` / `dodge` / 伤害奖励。
2. 把 `41 / 42` 里的 `frame` 偏移标定出来 → 严格的帧级同步（比墙钟同步稳得多）。
3. ✅ 已提供**时间缩放钩子**（`native/speedhack/`，钩 `timeGetTime`）：
   `th_set_game_speed(factor>1)` 加速可提高真机训练吞吐，`<1` 减速便于观察。
4. ✅ 已提供 `gymnasium` 适配（`training/gym_env.py`），可直接接 SB3 / CleanRL；
   训练循环也已通过 `th_train_*` 工具暴露给 Agent（后台线程跑，不占用 MCP 往返）。
5. ✅ 标定流程已脚本化：`th_memory_find_player` / `th_memory_find_frame` 自动定位，
   `th_offsets_write` 一键写回 offsets json（原子写 + `.json.bak` 备份）。
6. 用 replay（`E:\thManager` 里的 `threp`）做离线预训练与行为克隆。

---

## 10. mock 训练增强（2026-10-18）

针对"策略学会**被弹后靠无敌帧白嫖输出**、根本不学躲弹"这一根因做的改造：

**奖励（`touhou_rl_mcp/rl/reward.py` + `config.json`）**

- 新增 `invincible_output_scale`（默认 `0.0`）：**无敌帧期间"Δ分数 / Boss伤害 / 存活奖励"归零**，
  封堵"故意被弹 → 2.5 秒零风险输出"的漏洞；真机未标定 `invincible` 时，退化为
  "掉残后 `invincible_frames`(150) 帧内视为无敌"（`reward.py::_invincible_active`）。
- `miss_penalty -30 → -200`、`death_penalty -150 → -600`：让"防止被弹"成为第一优先。
- 新增 `spell_clear_bonus`（击破一张符卡）。
- **奖励配置统一**：`config.json` 的 `reward` 段是唯一来源，`tools/mock_full_clear.py`
  不再自带一套参数。

**Boss（`backends/mock.py`）** —— 参照正作：

- 每个 Boss **2~3 张符卡**，每张符卡有**独立血量 / 弹型 / 移动方式**；
- 弹型：`ring / aimed / spiral / fan / wall / burst`；移动：`static / sine_x / drift / figure8`；
- **刷新位置随机**（不固定中央）、**Boss 会移动**、血量随面数提升；
- 打空一张符卡 → 下一张；全部打空 → 过面（`raw["spell_cleared"] / ["stage_cleared"]` 事件）。

**课程学习**：`start_stage` 全链路可用（`config.json` / `TrainConfig` / `th_start_new_game` /
`examples/train.py --start-stage` / `tools/mock_full_clear.py --start-stage`），可从第 N 面开局逐面推进。

**动作空间**：`use_focus=True` 现在是 **9 快 + 9 慢(focus) 共 18 个动作**（原来只是给全部动作都加 focus）。
`focus` 能低速精躲，但在本 mock 里高速躲避通常更好——策略自己会权衡。

**算法**：纯 numpy 实现 **PPO（GAE + clipped surrogate）**，`algorithm="ppo"`（默认），
保留 `reinforce_baseline`；新增**向量化行为克隆** `MLPPolicy.imitate_batch`。

**规则老师（`training/rule_teacher.py`）**：与 `dodge` 宏同源（`planner="force"`），
在 mock 上可稳定躲弹，用于 BC / DAgger（`--teacher rule` / `rule_dagger`）。

**关键经验**

- **决策粒度**：mock 上 `step_frames=1`（60Hz 决策）时规则老师与 `dodge` 宏完全一致；
  `step_frames=4` 明显变差。学习躲避建议用 `--step-frames 1`。
- 纯 PPO 从零训练容易塌缩到单一动作；**先用规则老师 BC 热身，再 PPO 微调**更稳。

```powershell
# 1) 行为克隆: 规则老师 -> 策略 (会躲)
.\.venv\Scripts\python.exe tools\mock_full_clear.py --teacher rule --episodes 150 `
    --step-frames 1 --max-steps 12000 --batch 8 --lr 2e-3 --entropy 0 --min-entropy 0 `
    --k-enemies 8 --teacher-epochs 20 --ppo-minibatch 512 --no-focus `
    --out .runtime\mock_bc

# 2) PPO 微调 (从 BC 策略继续; 60Hz 要用大 gamma + KL 早停 + 奖励缩放)
.\.venv\Scripts\python.exe tools\mock_full_clear.py --algorithm ppo --load .runtime\mock_bc\policy_best `
    --episodes 3000 --step-frames 1 --max-steps 12000 --batch 16 --lr 1e-4 --gamma 0.999 `
    --ppo-minibatch 4096 --ppo-epochs 2 --target-kl 0.03 --shaping-scale 0.01 `
    --entropy 0.01 --k-enemies 8 --no-focus --out .runtime\mock_ppo --eval-every 200

# 3) 课程学习: 从第 3 面开局
.\.venv\Scripts\python.exe tools\mock_full_clear.py --start-stage 3 --episodes 3000 --step-frames 1 `
    --out .runtime\mock_s3
```

---

**实测结论（2026-10-18，mock，`step_frames=1`，6 个固定种子）**

| 策略 | 平均存活(ticks) | 最远面数 | 过第一面 |
|---|---|---|---|
| 规则老师 `RuleTeacher` | ~5400 | 3 | 6/6 |
| BC（仅给相对弹幕位置） | ~1800 | 2 | 2/6 |
| DAgger（同上） | ~1700 | 2 | 2/6 |
| BC **+ 危险梯度特征** | ~2550 | 2 | 4/6 |
| **DAgger + 危险梯度 + 危险加权** | **~4460** | **3** | **6/6** |
| PPO（从 BC 微调） | ~600 | 1 | 0/6 |

**关键结论**

- **观测够了**：只给老师"最近 16 颗弹"，它的表现与"看全部弹"完全一致（6/6）。
- **瓶颈是"反应式躲避"这个函数难被表示**：纯 BC/DAgger 只给相对位置时一致率上不去
  （离线 95%，on-policy 仅 66%），且长局"蝴蝶效应"会级联致死。
- **解法：危险梯度特征 + 危险加权 DAgger**。危险梯度 = 半径 30px 内弹幕的斥力合成
  （正是规则老师躲避的核心量）；把它作为特征后，策略**追平老师（6/6 过第一面、到第 3 面）**。
- **PPO 从 BC 微调会退化**（已修 gamma/KL/value 三个数值问题，但仍不稳），
  当前推荐**以 DAgger 为主**。
- **PPO 还会"刷存活"**：从 DAgger5 微调后，6 个种子里 5 个**存活到 20000 ticks 却从不通关**
  —— 它学会"永远躲、不打 Boss"来吃 `alive_bonus`（长局里存活奖励压过了通关奖励）。
  用 PPO 时需**压低 `alive_bonus`、加大 `time_penalty` 或缩短 `max_steps`** 来堵这个漏洞。

**制胜配方**

```powershell
# 1) 危险梯度特征 + 行为克隆
.\.venv\Scripts\python.exe tools\mock_full_clear.py --teacher rule --danger-grad `
    --episodes 250 --step-frames 1 --max-steps 12000 --batch 8 --lr 2e-3 --entropy 0 `
    --min-entropy 0 --k-enemies 8 --teacher-epochs 20 --ppo-minibatch 512 --no-focus `
    --out .runtime\mock_bc

# 2) DAgger 修漂移 (危险加权 + beta 调度)
.\.venv\Scripts\python.exe tools\mock_full_clear.py --teacher rule_dagger --danger-grad `
    --dagger-beta 0.4 --dagger-beta-min 0.1 --danger-weight 3 `
    --load .runtime\mock_bc\policy_best --episodes 250 --step-frames 1 --max-steps 12000 `
    --batch 8 --lr 1e-3 --entropy 0 --min-entropy 0 --k-enemies 8 --teacher-epochs 20 `
    --ppo-minibatch 512 --no-focus --out .runtime\mock_dagger
```

---

### 真机弹幕复刻到 mock（2026-10-18）

东方 `.rpy` 只存"随机种子 + 每帧输入"、**不存弹幕**（已实测：`t11r` 头 + 明文元数据
`Version/Name/Chara/Rank/Score` + 压缩的输入流）。所以**复刻真实弹幕的唯一办法是
"进游戏逐帧读弹幕"**。本项目提供完整链路：

1. **录制** `tools/record_danmaku.py`：用规则老师边躲边录，以 60Hz 逐帧检测
   "新弹生成"（按 `active_slots` 跨帧差分 + 用下一帧位移估计初速），存成脚本
   `{"frames": N, "difficulty": d, "character": c, "shot": s, "spawns": [[frame,x,y,vx,vy],...]}`。
   **支持多局 `--runs`，并按"难度 / 自机角色 / 机体"区分命名**（弹幕与它们强相关）；
   **训练时也可同时录制**：`examples/train.py --record-danmaku-dir <dir>`。
   对话/静默期会自动**连续点按 Z 快进**（见 `docs/COMBAT_TRAINING.md`）。
2. **回放** mock 的 `danmaku_script` 模式（`config.danmaku_script`）：按脚本逐帧生成
   **真实弹幕**，并**关掉内置弹型**；Boss/敌机仍作为攻击目标（伤害/过面奖励不变）；
3. **演示** `tools/mock_demo.py --danmaku-script <脚本>`；
4. **录完整弹幕（全 6 面）** `tools/record_replay.py`：让游戏**自带回放播放器**播一份
   通关录像，逐帧读弹幕 —— 这是拿到"第 2 面帕露西"等后期弹幕的最省事办法。
   菜单导航：标题 → **短按**下×3 到 Replay → 短按 Z（回放页）→ 短按 Z（选第一个回放）
   → 短按 Z（从第 1 面开始）。**注意：菜单必须短按（`hold=2`），长按会触发连发而跳过选项。**

```powershell
# 录一份完整通关录像的弹幕 (需 TH11 可启动)
.\.venv\Scripts\python.exe tools\record_replay.py --seconds 3600 `
    --out .runtime\danmaku\th11_replay_full.json
```

```powershell
# 1) 录制真机弹幕 (需 TH11 可被本项目启动; 窗口化)
.\.venv\Scripts\python.exe tools\record_danmaku.py --seconds 90 --restart `
    --out .runtime\danmaku\th11_stage1.json

# 2) 用真实弹幕跑 mock (规则老师 / 策略 / 演示 GIF)
.\.venv\Scripts\python.exe tools\mock_demo.py --teacher rule `
    --danmaku-script .runtime\danmaku\th11_stage1.json --step-frames 1 --out .runtime\demo_real_danmaku

# 3) 直接用真实弹幕训练
.\.venv\Scripts\python.exe examples\train.py --backend mock --episodes 500 --step-frames 1 `
    --danger-grad --k-enemies 8 --out .runtime\mock_real
```

> 注：`--danger-grad` 特征 + 危险加权 DAgger 的配方同样适用于真实弹幕。
> **坑**：TH11 的 `frame`（`0xA5734`）在**换面时会重置**，录制器已自动续成单调帧号
> （`DanmakuRecorder._monotonic_frame`），否则跨面的脚本帧号会错乱。

---

### 真机 Boss 位置 + 双线程训练（2026-10-18）

- **真机 Boss 位置**：TH11 的 Boss 对象指针是 `entities.combat.boss`（= 敌人管理器 `+0x1C`），
  位置在 `boss + 0x1070/+0x1074`（与敌人链表同布局，加 `(192,0)` 场地偏移）。
  已接入 `state.raw["boss_position"]` —— 于是**真机上规则老师与策略都能对准 Boss 输出**
  （修复前老师只会在中线站桩、打不到 Boss）。实测：自机 `|Δx|` 从 46 收敛到 1，
  Boss 血量持续下降，一局打到第 2 面。
- **双线程训练** `tools/dual_train.py`：**真机线程**（DAgger + 录弹幕）与 **mock 线程**
  （用录到的真机弹幕脚本）**共享同一个策略**，用一把锁保护 forward/backward，
  让策略同时从"真机分布"和"真机弹幕的 mock 回放"学习。

```powershell
# 真机 + mock 双线程共享策略训练 (真机 60fps, mock 高速刷量)
.\.venv\Scripts\python.exe tools\dual_train.py --seconds 900 `
    --danmaku-dir .runtime\danmaku_real --out .runtime\dual
```

> **坑（已解决）**：Python 的 GIL 会让 CPU 密集的 mock 线程**饿死**真机线程 —— 真机跟不上
> 60fps 就会被游戏"超车"而速死（实测每局只剩 ~1000 步，正常是 1.5 万步）。
> 修法：mock 线程**每步 `sleep`**（`--mock-step-sleep 0.003`，主动让出 GIL）+
> `sys.setswitchinterval(0.001)`。修复后真机每局恢复到 1.4 万+ 步、打到第 2 面。

**更稳的做法（双进程 + 合并）**：`tools/dual_proc_train.py` 把 mock 放到**独立进程**
（彻底绕开 GIL），真机进程照常 60fps；两边从**同一初始策略**出发，每隔 N 秒
**按权重平均**（model soup，`training/policy.py::blend_policies`）并互相载入，再继续真机训练。

```powershell
# 更高难度真机训练 + 独立进程 mock + 周期合并
.\.venv\Scripts\python.exe tools\dual_proc_train.py --real-seconds 900 --mock-seconds 900 `
    --merge-every 120 --difficulty 1 --danmaku-dir .runtime\danmaku_real --out .runtime\dualproc
```

**一键循环入口（推荐）**：`tools/auto_pipeline.py` + `run_pipeline.ps1` 把
"**真机 + mock 双进程 → 合并 → 继续 → 再合并 …**" **自动往返循环**：每轮结束把合并策略
作为下一轮的起点（`policy_current.npz`），每轮产物在 `roundNNN/` 下。

```powershell
# 一键循环 (默认 Normal, 每轮真机 10 分钟; Ctrl-C 停止并保存 policy_final)
powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -Rounds 0 -RealSeconds 600 -Difficulty 1

# 不占用真机, 只跑 mock 验证流程
powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -NoReal -Rounds 2 -RealSeconds 60

# 也可以直接调 python
.\.venv\Scripts\python.exe tools\auto_pipeline.py --rounds 0 --real-seconds 600 --difficulty 1 `
    --danmaku-dir .runtime\danmaku_real --out .runtime\auto_pipeline
```

---

## 11. 约定与致谢

- th11 的内存偏移来自你已有的 `D:\东方AI实时分析\th11.py`（pymem 版本），
  本项目中改为纯 ctypes 实现（同时支持 64 位 Python 读 32 位游戏）。
- 覆盖层 DLL / 共享内存方案（`E:\TouhouOverlay`、`E:\thManager\TouHouManager\overlay`）
  本 MCP 未使用——MCP 走 stdio 返回图片，不往游戏里画东西。
- 菜单导航、帧同步、宏动作的设计参考了社区里 Touhou RL / bot 项目的常用做法。
