# Replay → 行为克隆 / 离线 RL（设计文档）

目标：把东方正作的 **replay（`.rpy`）** 变成 `(真实状态, 人类动作)` 数据集，
用来做行为克隆或离线 RL，缩小 mock→真机的 sim-to-real 差距。

> 状态：**部分落地**。`tools/record_replay.py --bc-out` 已能"让游戏回放 → 逐帧读状态 →
> 反推真人动作 → 存 npz"；`tools/bc_from_replay.py` 已能直接做行为克隆。
> 尚未实现的是**直接解析 `.rpy` 文件**（当前绕开了它，改用游戏自带播放器）。

## 1. 关键事实（决定可行做法）

东方的 `.rpy` **只保存随机种子 + 每一帧的输入**，**不保存敌机/弹幕位置**。
回放时游戏用同一个种子 + 逐帧输入**确定性重演**。

因此：

- ✗ 不能"从 `.rpy` 直接解析出弹幕轨迹"；
- ✓ 可行做法：**让游戏回放 → 用我们已有的实体读取器逐帧读真实状态 → 与输入配对**。

## 2. 数据采集流程

```
.rpy 文件 ──(1)解析──▶ 逐帧输入表 (frame -> 按键集合)
                              │
                              ▼
游戏回放 (2)  ──▶ 每帧: (3)读状态 ──▶ (4)对齐 ──▶ (state, action) 数据集
```

1. **解析 `.rpy`**：按 TH11 格式解析（可参考 thtk 等社区工具；或用现成工具转成帧输入表）。
   需要：`frame -> {up,down,left,right,shoot,focus,bomb}`。
2. **让游戏回放**：优先用游戏自带 replay 播放器（帧锁定最稳）；若需要精确逐帧控制，
   用 `SendInput` 逐帧喂入。
3. **逐帧读状态**：复用 `memory/entities.py` + `memory/reader.py`，
   以 `frame`（th11 `0xA5734`，已标定）计数，**每帧读一次**。
   成本：每帧约 4.6 MB 批量读（1~3 ms），60Hz 可行。
4. **对齐**：把第 N 帧的输入与第 N 帧的状态配对，编码成与训练一致的观测向量
   （`training/features.py::encode_state`，参数需与目标策略一致：
   `k_bullets / use_velocity / k_enemies / use_invincible / use_boss`）。

产物：`obs (N, D)` + `action (N,)`（离散动作索引，需把人类按键映射到动作表
`training/trainer.py::build_action_table` 的 9/18 个动作上；无法映射的帧丢弃或并入最近动作）。

## 3. 训练

- **行为克隆**：`MLPPolicy.imitate_batch(obs, action)`（已向量化，见 `training/policy.py`）。
- **离线 RL（可选）**：CQL / IQL 等；或先用 BC 预训练，再在真机用 PPO/DAgger 在线微调
  （真机可用 `native/speedhack` 4× 提速）。

## 4. 代码现状

| 位置 | 状态 |
|---|---|
| `tools/record_replay.py --bc-out <npz>` | ✅ 播放回放、逐帧读状态、反推动作、落盘 |
| `tools/bc_from_replay.py --data <npz> --out <dir>` | ✅ BC 训练（含验证集切分） |
| `touhou_rl_mcp/replay/rpy.py`（解析 `.rpy`） | ⬜ 未实现；当前用游戏播放器绕开 |
| `touhou_rl_mcp/replay/collect.py` | ⬜ 未抽出为库模块（逻辑仍在 CLI 里） |

> 如果以后要做"不启动游戏的离线解析"，再从第 2 节第 1 步开始补 `.rpy` 解析。

## 5. 前提与风险

- **版本必须一致**：replay 与游戏版本要匹配（F 盘 1.00a ✓）。
- **人类动作的按键语义**：需要确认 replay 里的按键位与实际动作的对应。
- **观测对齐**：`frame` 是游戏帧计数；回放的帧与 `frame` 必须严格同步（用帧计数等待，而非墙钟）。
- **数据集规模**：一个 replay 可能几千~几万帧；建议累积多个 replay。
- **仍是 TH11 专用**：实体布局只支持 TH11 1.00a。

## 6. 与当前 mock 管线的衔接

mock 侧已证明："规则老师 → BC → 策略" 能学到躲避与过面。replay 侧把"老师"换成
**人类输入**，其余（特征、BC 更新、评估）完全复用：

```
mock:    RuleTeacher ─┐
                      ├─▶ (obs, action) ─▶ MLPPolicy.imitate_batch ─▶ 策略 ─▶ PPO/DAgger 微调
replay:  human input ─┘
```
