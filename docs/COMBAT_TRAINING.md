# Boss 奖励、对话辅助与六机体训练

## 奖励

TH11 1.00a：敌人管理器 `+0x1C` 是 Boss；Boss `+0x252C/+0x2530`
是当前/最大血量。UI 代码 `0x41B872/0x41B886` 用这两个值计算血条比例。
符卡管理器为 `read_u32(base+0xA8D6C)`；`+0x8DC/+0x8E0/+0x8E4`
依次为符卡 ID、标志、Bonus 金额。

结算代码 `0x40CDC2` 检查标志位 2，随后把 Bonus/10 加入原始分数。
活动位为 1。读取器验证相关代码签名，`raw.entities.combat` 输出诊断。

- `score_scale` 原本就包含实际 Bonus 带来的分数增长。
- 新 `spell_capture_bonus`（默认 20）：活动结束、仍有奖励资格、同符卡，
  且真实分数增长覆盖 Bonus 金额时，额外发一次 `spell_capture`。
- `boss_damage_scale`：同 Boss、同阶段的血量下降产生密集奖励。
  训练脚本设为 0.01；排除已检测的超时标志和阶段切换，防止超时清血刷奖励。
- 有些击破因使用炸弹/掉残不具备游戏原生符卡收取资格，这时不发额外捕获奖励，
  但实际伤害与得分仍可产生奖励。
- 未把“符卡编号变化”“敌人消失”当作击破证据。离散轮询可能漏过短暂结算，
  宁可漏奖也不把超时猜成成功。尚未添加无收取资格情况下的独立“血量击破奖”。

## 对话辅助（2026-10-18 改为"连续点按 Z"）

默认 `dialogue_assist=true`。东方是**边沿触发**：长按 Z 不会连跳，必须反复"按下-松开"，
所以旧版"按住射击"会卡在对话里、攻击长时间中断。

现在：当**局内 + 场上无弹幕 + 没有会攻击的敌人**（含 TH11 对话期间"不攻击的 Boss 演员"）
连续达到 `dialogue_threshold_frames`（默认 25）控制帧时，进入对话快进：
**连续点按 Z**（`dialogue_tap_hold` 按住帧 / `dialogue_tap_after` 间隔帧 /
`dialogue_max_taps` 单步上限，默认 2/6/12），并**保留本步的移动意图**（方向键保持按住）。
**每次点按后重新读一次状态**：一旦敌人/弹幕回来就立刻停手回到战斗。

这只是空场启发式，不是对话识别；正常波次空隙也可能触发（此时点按 Z 正好在输出，无害）。
不会据此判定通关，不在 Game Over 时点按。对话辅助步没有存活奖励，
且不会把被覆盖的移动动作计入轨迹。相关工具：`DialogueAssist.is_empty` /
`Controller.tap_repeat`；测试见 `tests/test_mock_combat.py::TestDialogueFastForward`。

## 机体与继续训练

TH11：character=0 灵梦、1 魔理沙；shot=0/1/2 对应 A/B/C。
发现并修复菜单错误：武器列表会循环（不能连按 4 次上归零），
角色切换使用左右键。重启后从初始选项按索引选择，并实测核对内存身份。
此前第一面成功试验请求 A，实际可能为 C；不能依请求值声称是灵梦 A 成功。

`TrainConfig` 增加 character、shot、rotate_loadouts；命令行也提供对应参数。
旧 74 维输入不包含机体身份，推荐六机体各自从原模型独立微调。
`tools/train_loadouts.py` 不启用规则移动覆盖，保证采样动作与训练标签一致。
模型持续射击；当前 9 动作模型不含炸弹动作，炸弹仍是辅助游玩规则，不能宣称模型学会放炸弹。

已实测六机体各 300 步，实际角色/武器与请求全部匹配；每种完成一次梯度更新，
权重确有变化。原 mock 检查点未覆盖。仅为短时探索，不是收敛或通关率比较。

输出：

- 灵梦三机体：`.runtime/loadout_training/20261017_225306/reimu_A|B|C/policy.npz`
- 魔理沙三机体：`.runtime/loadout_training/20261017_225725/marisa_A|B|C/policy.npz`

灵梦批次运行期间修复了统计字段：该批次 misses/truncated 显示可能不准，
应参考 reward_parts.miss 和 max_steps；魔理沙批次已使用新统计。

```powershell
# 六机体独立继续训练（每个从指定检查点开始，不覆盖源文件）
.\.venv\Scripts\python.exe -u tools\train_loadouts.py --steps 1500 --episodes 3
# 单独继续灵梦 C
.\.venv\Scripts\python.exe examples\train.py --backend memory --restart-process --character 0 --shot 2 --load .runtime\loadout_training\20261017_225306\reimu_C\policy.npz --out .runtime\reimu_C_next
```

## 真机 focus（按住 Shift 慢速）

**为什么必须当真**：真机实测自机速度 4.5 px/帧、focus 后 2.0 px/帧（`CONCLUSIONS.md` §1）。
判定半径只有 2 + 3 = 5px，全速一步 4.5px 相当于"每一步都在赌"，密弹里真人几乎全程
按住 Shift。所以真机训练的动作表必须是 **18 个动作**（9 快 + 9 慢），并且**推理时
同样用 18 个动作**——否则真机上模型只能全速跨步，看起来"笨"。

落地要点（2026-10-22）：

1. **输入链路**：`inputs/keyboard.py` 里 `"focus" → 0x10 (Shift)`；控制器按
   `action_table` 里的 `focus` 后缀同时按下 Shift + 射击。真机上 Shift 与对话快进的
   点按 Z 互不影响。
2. **训练开关**：`TrainConfig.use_focus` 默认 **True**（`--no-focus` 只用于对照组）。
   观测与动作维度会随之变化（9→18），旧模型不能用 `--load` 硬接，需 `expand`。
3. **老师在真机物理上建模**：`MPCPlanner.focus_speed = speed × FOCUS_SPEED_RATIO`
   （2/4.4 ≈ 0.45），并且 `step_frames=k` 时**跟着 × k** —— 之前 focus 速度写死 2.0，
   k=2 时老师以为半步走 2px、实际走 4px，算出来的"半步位"全错。
4. **训练前先自检**（关键）：Shift 有没有被游戏收到，只有真机能量出来。

```powershell
# 游戏需已进入关卡（不是标题/菜单）；比例落在 0.30~0.65 才算 PASS
.\.venv\Scripts\python.exe tools\check_focus.py --frames 10 --rounds 3
```

失败表现与处置：

| 现象 | 含义 | 处置 |
|---|---|---|
| 慢速/全速 ≈ 1.0 | Shift 没被游戏收到 | 查窗口置前、游戏内 Slow 键映射、`focus_window` |
| 慢速/全速 < 0.15 | Shift 一直是按下状态或绑错键 | 查 sticky key / 键盘映射 |
| 比例偏离 0.45 较多 | 老师建模的半步位不准 | 调 `mpc.FOCUS_SPEED_RATIO`，别用默认值硬跑 |

5. **评测也要带 focus**：`tools/pattern_bench.py` 默认 `focus=True`（`--no-focus` 关闭）；
   用 9 动作模型去跑 18 动作配置会维度不匹配，反过来也一样。

