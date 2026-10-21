# 工程评审（2026-10-20 快照）

> 本文是**一次静态评审的记录**，不是现状来源。能力边界与实验基线仍以
> [`STATUS.md`](STATUS.md) 为准；本文只回答"结构是否合理、哪里欠债"。
> 评审方式：通读代码 + 跑测试 + ruff + 依赖/引用扫描，未改动任何行为。

## 1. 规模与健康度

| 指标 | 实测 |
|---|---|
| 库 `touhou_rl_mcp/` | 58 个模块 / 14310 行（含 dashboard 前端 885 行 JS） |
| `tools/` | 31 个活跃 CLI / 9428 行（另有 `legacy/` 19 个归档） |
| `tests/` | 15 个文件 / 2848 行 / **172 用例（170 通过、2 跳过）**，约 18 s（评审时；修完 P0 后为 195） |
| `docs/` | 9 篇 / 1821 行 |
| ruff（E9+F） | 通过，0 告警 |
| `.runtime/` | 680 MB（`scanner` 202 MB、`th11_clear` 72 MB、`runs` 30 MB…） |

结论：**工程纪律已经在及格线以上**（CI = ruff + unittest + 自检；
配置单一来源；命名与文档密度都高于同类个人项目）。欠债集中在"覆盖盲区"和
"三个超长文件"上，而不是缺基础设施。

## 2. 分层结构

```
Agent 层    LLM Agent / OpenCode ──MCP──▶ server.py (30 个 th_* 工具)
调度层      Trainer 后台线程（60 Hz，进程内，绕开 MCP 往返）
核心层      TouhouEnv：reset/step/observe + RewardModel + StatsTracker + RLock
后端层      Backend(mock | memory)
支撑层      memory(进程/读取/扫描/实体) · inputs(按键/动作/控制器)
            capture(截图/可视化) · games(注册表/偏移表) · dashboard · training
```

这个分层是干净的：`env` 只依赖 `Backend` 抽象，奖励与统计与后端无关；
`training` 与 `server` 都只通过 `TouhouEnv` 打交道；看板只读 `.runtime` 的 JSONL。
**唯一的概念泄漏**是 `env.py` 里堆了真机专用逻辑（对话快进、紧急放弹、
bomb_guard / attack guard 之类），按职责本应下沉到 `inputs/` 或 memory 后端。

## 3. 功能盘点（能做到 / 做不到）

**已闭环**

- mock 后端：多弹型（ring/aimed/spiral/fan/wall/burst）+ 多阶段 Boss + 符卡，
  1400 step/s，可复现，支撑了 CI 与全部算法实验。
- memory 后端（仅 TH11 1.00a）：9 个标量 + 自机坐标 + 弹幕数组 + 敌人链表 +
  Boss 血量/位置，坐标统一为 384×448 场地系，带版本签名校验。
- 动作层：按键串 / 动词 / 宏动作（`move_to`、`dodge`），解决"LLM 1~10 Hz vs 游戏 60 fps"。
- 奖励层：20+ 可在线调权重 + `reward_parts` 分解 + 具名 profile，这是"agentic RL 调参"的接口。
- 训练层：纯 numpy PPO / REINFORCE / BC / DAgger + 规则老师 + 缺口观测。
- 观测层：最近 K 弹 / 危险图 / 未来占用栅格 TTA / 危险梯度，多套可切换编码。
- 工具层：偏移标定（首扫/再扫/撞墙法找自机/找 frame/一键写回）、速度钩子、看板、GIF。

**明确做不到**

- TH11 之外 7 部作品：偏移表 12 个字段**全为 `null`**，只能截图/可视化。
- TH11 缺 `invincible` / 激光 / `bombs`：无敌期只能靠"掉残后 150 帧"回退推断。
- 策略单独稳定过第一面：未达成（见 STATUS 0.2）。
- 真机异常自恢复：进程崩溃/失焦/弹窗后没有看门狗。

## 4. 需要完善的地方

### P0 · 正确性与一致性（✅ 四条已修，见第 6 节）

1. **`config.json` 的 `search_roots` 双重转义**：`"F:\\\\"` 实际解析成 `F:\\`
   （长度 4 的字符串），`DEFAULT_CONFIG` 里写的是 `"F:\\"`。目前靠 `Path` 归一化
   侥幸工作，但两处写法不一致，属配置脏数据。
2. **README 与代码数字脱节**：README §8 仍写"133 个用例"（实为 172）；
   目录树缺 `backends/patterns.py`、`capture/bullets.py`、`memory/{entities,danmaku,cheats,slowdown,bullettrack}.py`、
   `training/{gap_label,gap_teacher,rule_teacher,tactical}.py`、`dashboard/{recipes,launcher,registry}.py`。
   `docs/STATUS.md` 自称"唯一现状来源"，但 README 仍是大多数人的入口。
3. **`capture/bullets.py`（164 行）是死代码**：视觉弹幕检测（不依赖偏移，
   理论上能救 7 部未标定作品），全项目零引用、零测试、README 未提。
   要么接进 `memory` 后端作为"无偏移时的弹幕来源"，要么明确删除。
4. **`Config.patch` 只校验顶层键**：`th_configure({"reward": {"typo": 1}})` 能写进配置，
   直到下一帧 `env.configure` 才在 `RewardModel.update_weights` 抛错；而 `save=True`
   时坏键已经落盘。

### P1 · 测试盲区（最大的隐性风险）

5. **`memory/scanner.py` 1309 行，无任何专门测试** —— 它是全库最大文件，
   同时承担首扫/再扫/候选持久化/自动标定（写回 offsets）。自动标定写坏偏移是
   "静默错误"，恰好最需要回归测试。
6. **`tools/` 31 个活跃脚本 0 测试**（STATUS P7 已承认）。关键路径
   `record_danmaku / record_replay / gap_trainer / mock_full_clear` 一旦 argparse
   改名，只有下次跑实验时才发现。
7. **无覆盖率门槛**：`keyboard.py`、`process.py`、`screenshot.py` 只能靠真机/人工验证，
   建议至少给纯函数部分（帧同步、窗口查找、像素→坐标换算）加单测。

### P2 · 结构债

8. **三个超长文件**：`scanner.py`(1309) / `trainer.py`(1130) / `server.py`(977)。
   `server.py` 可按"环境类 / 标定类 / 训练类"拆成三个 router 模块；
   `trainer.py` 的训练循环与评估/日志逻辑可分离。
9. **`env.py` 职责过载**：建议把"对话快进 + 紧急放弹 + 输出缩放"收进
   `inputs/assists.py`，让 `env.step` 只剩计时、奖励、终局判定。
10. **`TouhouEnv` 是单例全局**（`server._ENV`）：单进程单客户端没问题，
    但多会话并发时"谁改了奖励/谁占了窗口"无法隔离；若要做多 Agent 必须引入 session 概念。

### P3 · 能力缺口（与 STATUS 路线图一致，按价值排序）

11. **训练目标 vs 评估口径**：贴角退化（86~92% 时间贴角）说明惩罚必须进
    PPO/DAgger 的损失，而不是只进 V 标签；验收改成"模式基准 5 seeds + 最差局 + 贴角比例"。
12. **观测**：弹幕只有 `(x, y)`，没有半径/类型/速度（真机有速度、mock 有速度但
    `bullet_radius` 仍在 planner 里留 TODO）。缺少 TTC、多帧堆叠。
13. **真机鲁棒性**：看门狗（进程存活/窗口焦点/黑屏检测）、异常自动复位、
    `invincible` 与激光标定。
14. **奖励"苟活"漏洞**：`alive_bonus=0.3` vs `time_penalty=-0.01`，量级差 30 倍，
    长局中存在"只躲不打"的局部最优。建议改成"过面进度 + 擦弹 − 碰撞"。
15. **多作品**：差的不是代码而是标定工作量；可考虑先用视觉方案（见第 3 条）
    给未标定作品提供最小可用观测。

### P4 · 工程收尾

16. `.runtime` 680 MB 未清理（`scanner` 候选、`th11_clear` 录像）；建议加
    `tools/cleanup.py --older-than 7d` 并写进 README。
17. 缺 `mypy`/`pyright` 类型检查（`pyproject` 已写 `requires-python>=3.11`，
    但类型标注只覆盖约一半函数）；可选地给 `types.py`/`config.py`/`reward.py`
    这几个核心数据结构先开严格模式。
18. `requirements.txt` 与 `pyproject` 的双份依赖需人工同步，建议 README 只留
    `pip install -e .[dev]`。

## 5. 一句话建议

先花半天做 **P0 + P1 里的 3 个洞**（配置转义、README 数字、死代码裁决、
scanner 回归测试），再回到 STATUS 的 P0（反贴角）—— 前者的收益是"以后不踩坑"，
后者的收益是"能力真的上一个台阶"，两者不冲突但顺序不能反。

## 6. 第一批修复记录（2026-10-20）

| 项 | 处理 | 验证 |
|---|---|---|
| 1. `search_roots` 双重转义 | 改成 `"F:\\"`；新增测试断言"恰好一个反斜杠" | `tests/test_config.py` |
| 2. README 滞后 | 用例数 133→195；目录树补齐约 20 个模块；补 `docs/VALUE_DISTILL.md` / `PROJECT_REVIEW.md`；新增视觉兜底说明 | 人工核对 |
| 3. `capture/bullets.py` 死代码 | **接入**而非删除：新增 `bullets_from_frame()`（裁剪场地 + 坐标归零），`MemoryBackend` 在 `vision.enabled=true` 且内存无可靠弹幕时调用；参数进 `config.vision`，**默认关闭** | `tests/test_capture_bullets.py`（9 例，含缩放窗口与失败降级） |
| 4. `Config.patch` 只校验顶层 | 改为递归校验：未知嵌套键报 `reward.typo` 这样的完整路径；段被写成非对象直接拒绝；`control.keymap` 保留自由键名 | `tests/test_config.py`（14 例） |

顺带修掉的隐患：

- `gap_margin_scale` 在 `WEIGHT_DOCS` 里存在、`config.json` / `DEFAULT_CONFIG` 里却缺失——
  递归校验会拒绝它，因此补进两处（默认 `0.0`，行为不变），并加测试断言
  `WEIGHT_DOCS ⊆ config.json["reward"]`，防止以后再漂移。
- `detect_bullets` 的 docstring 原来把"裁剪后输入 + 窗口系输出"写成了"游戏内部坐标"，
  容易误用；已说明两者的区别，并指向 `bullets_from_frame`。

基线：`195 用例 / 193 通过 / 2 跳过`，`ruff` 0 告警。
