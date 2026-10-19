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
