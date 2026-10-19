# 项目现状与路线图

> 这是**唯一**的"当前状态"来源。历史会话记录见
> [`docs/history/`](history/)，具体实验数据见文末链接。
> 最后核验：2026-10-19。

## 1. 一句话

把东方正作（目前真正打通的是 **TH11 地灵殿 1.00a**）封装成标准 RL 环境，并以 MCP
服务器暴露给 LLM Agent：LLM 以 1~10 Hz 做高层决策与调参，本地策略在后台以 60 Hz
执行。核心链路（观测 → 动作 → 奖励 → 训练 → 策略上机）已跑通。

## 2. 实测基线（2026-10-19）

| 项 | 结果 |
|---|---|
| 单元/集成测试 | `python -m unittest discover -s tests` → **121 通过，2 跳过**，约 15 s |
| 自检 | `python -m touhou_rl_mcp selftest --dodge --steps 200` → 正常出图 |
| MCP 工具 | **30 个** `th_*`（server.py） |
| 源码规模 | 约 2.07 万行（库 + tools + tests + native + docs） |
| mock 吞吐 | 约 1400 step/s（真机约 9~20 step/s） |
| 偏移覆盖 | th11：9 个标量字段 + 实体（弹幕/敌机/Boss）；其余 7 作：仅菜单模板，0 字段 |

### TH11 已标定 / 未标定

- ✅ `score` `lives` `power` `stage` `spell_id` `char_main` `char_sub` `graze` `frame`
- ✅ 自机坐标（指针链）、普通弹幕数组、敌人链表、Boss 血量与位置、符卡
- ✅ 坐标统一为 `playfield_top_left`（384×448），实体读取带**版本签名校验**
- ⚠️ 未接入：**激光**、`invincible`、`bombs`（TH11 放弹消耗 power，无独立计数）

## 3. 目录职责

```
touhou_rl_mcp/     库本体（env / 后端 / 内存 / 输入 / 采集 / 奖励 / 训练 / 看板）
touhou_rl_mcp/dashboard/  训练可视化看板（标准库 http.server + 原生 canvas）
tools/             活跃 CLI（索引见 tools/README.md）
tools/research/    可复现实验（gap_rl / gap_benchmark）
tools/legacy/      已归档的一次性脚本（见其 README）
tests/             unittest 套件
examples/          train / rl_loop 示例入口
docs/              设计文档；docs/STATUS.md 即本文件
docs/history/      历史会话交接记录（只读，不保证与现状一致）
archive/           非维护文件（根目录残留 + 一次性诊断脚本）
native/speedhack/  32 位时间缩放钩子（可选，需自行构建）
```

`.runtime/` 只存运行产物（训练输出、日志、截图、弹幕脚本），已被 git 忽略。

## 4. 已知限制（诚实版）

1. **只有 TH11 能读数值状态**；其余作品只能截图/可视化。
2. **未稳定通关**：真机第一面有过成功记录（`docs/STAGE1_ATTEMPTS.md`，2026-10-17，
   靠规则辅助），但策略单独稳定通关尚未达成；mock 上贪心策略最远到第 2 面。
3. **mock ≠ 真机**：mock 是简化弹幕模拟，通关 mock 不代表能过真机。
4. **实时性**：真机 60 fps 不会等 Agent；低频决策必须用宏动作（`dodge` / `move_to`）。
5. **前台依赖**：按键注入要求游戏窗口在前台；进程崩溃/失焦没有自动恢复。
6. **性能**：每帧批量读约 4.6 MB（2000 个弹幕槽）。
7. **奖励仍有"苟活"空间**：`alive_bonus (0.3) >> |time_penalty (0.01)|`，
   长局里存活收益可能压过通关收益（见路线图 P2）。

## 5. 路线图

### P0 · 工程基线（2026-10-19 已完成）

- [x] `git init` + 基线提交（此前无版本控制）
- [x] 修复 `trainer.py` 中 56 处被替换成 `??` 的中文
- [x] 根目录残留与 47 个一次性探针脚本归档到 `archive/`
- [x] 19 个被取代的 tools 归档到 `tools/legacy/`
- [x] `PROGRESS.txt` / `NEXT_SESSION.txt` 移入 `docs/history/`，现状收敛到本文件
- [x] 补 `LICENSE`、CI 骨架、ruff 配置，清理依赖清单

### P1 · 配置与一致性（进行中）

- [x] 奖励默认值只保留在 `config.json`；实验覆盖改为具名 profile（`rl/reward.py`）
- [ ] 文档与代码数字对齐（工具数/用例数/标定状态）—— 持续检查
- [ ] 为 `tools/` 关键路径补冒烟测试（目前 41 → 22 个活跃脚本仍无测试）

### P2 · 模型与观测（下一阶段，先做诊断再调参）

1. 诊断"苟活"：统计无敌期输出占比、通关/存活奖励比、动作分布。
2. 重平衡奖励：下调 `alive_bonus` 或与危险度挂钩；确认 `clip` 不吞过面奖励。
3. 观测增强：弹幕半径/类型、TTC（预计最近接近距离）、多帧堆叠。
4. 算法：以 DAgger 为主，PPO 需解决"只躲不打"；实验一律记录配置与种子。

#### P2 可行性验证（2026-10-19，实验性）

针对"斥力避弹在环弹下被逼到墙角、不会找缝上穿"做了对照实验：

- 斥力老师 `force` 在 6 种弹型上 `min_y ≥ 358`（场地 448），**最长连续上移仅 13 帧**
  （约 0.2 秒）——行为上限是贴底横移，从不离开下半区。
- 短程规划老师 `horizon` 在 6 种弹型中 5 种于 587~1975 帧内死亡，`y` 被压到 400~440。
- 新增 `training/planner.py`（未来推演 + 宏动作 beam search）与
  `tools/research/gap_benchmark.py`（可复现基准）：
  - 静态单缺口环：规划器能穿出（最大位移 432px），但基准里仍掉 1 残，
    `min_clear=6.2`；参数稍变结果就不稳定。
  - 多层闭合环（新弹型 `closing_ring`）：noop 546 帧、force 582 帧、
    planner 630 帧，全部未过关。
  - 密集环：planner 1500 帧 0 死，但 `mean_y=431`（贴底龟缩）；
    force 在 6 帧承诺下 594 帧死亡。

**结论**：未来推演 + 宏动作是必要组件，但**手写搜索代价不能替代价值函数**。
继续调代价只会在"龟缩 / 贴边 / 冒进"之间换一种失败方式。下一步应直接学习
`V(s)`/策略（future occupancy 观测 + PPO，或用学习到的 V 给搜索做 bootstrap），
而不是继续堆搜索规则。`MacroPlanner` 目前仅作实验脚手架，不接入训练老师。

> ⚠️ 上述结论有一部分被后续发现的 bug 影响：`memory/bullettrack.py` 曾无条件覆盖
> `bullet_velocity`，在 mock 上没有槽位信息时把所有弹速写成 0，导致"时空搜索"实际
> 推演的是一群静止弹。该 bug 已修复（`tests/test_memory_reader.py` 有回归测试），
> `planner.py` 的基准需要在修复后重跑一次才算数。

#### P2 可行性验证 · 二（2026-10-19，缺口任务）

新增 `wall_gate`（真能封死的弹幕墙 + 单缺口）与 `training/occupancy.py`
（时间分层占用栅格 + 每列最早到达时间 TTA 剖面），在 `tools/research/gap_rl.py` 里做
"表示 / 密集信号 / 信任域"三层对照。**结论：几何缺口 + 占用栅格可行**，
但必须配套弱监督与 BC 锚，纯 RL 会坍缩成"不动"。完整数据见
[`docs/GAP_LEARNING.md`](GAP_LEARNING.md)。

### P3 · 真机与多作品

- 真机鲁棒性：进程/窗口/焦点看门狗，异常自动复位。
- 补齐 TH11 的 `invincible`、激光；再考虑第二部作品（建议从 th10/th13 中选一）。
- replay → BC 数据管线落地（设计见 `docs/REPLAY_CLONING.md`）。

## 6. 相关文档

- 偏移标定：`docs/OFFSET_CALIBRATION.md`
- TH11 实体布局：`docs/TH11_ENTITIES.md`
- 战斗训练 / 符卡与对话：`docs/COMBAT_TRAINING.md`
- 第一面尝试记录：`docs/STAGE1_ATTEMPTS.md`
- replay 克隆设计：`docs/REPLAY_CLONING.md`
- 缺口学习实验：`docs/GAP_LEARNING.md`
- 训练看板：`docs/DASHBOARD.md`
- 历史会话：`docs/history/2026-10-17_PROGRESS.txt`、`docs/history/2026-10-18_NEXT_SESSION.txt`
