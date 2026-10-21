# 攻击意图：奖励配平 + 蒸馏加强（2026-10-21）

> 问题："模型不主动攻击敌人，只会找安全路径；躲避效果也不好。"
> 本文记录诊断、改动与实测。**现状仍以 [`STATUS.md`](STATUS.md) 为准。**

## 0. 结论

1. **不攻击首先是配平问题，不是能力问题。** 旧权重下 `score` 项一局只贡献 **+0.2**，
   而 `danger` 项扣 **−3306** —— 策略唯一能学到的信号是"离弹幕越远越好"，而密集
   弹幕里"远"只有角落。
2. **BC/DAgger 完全不看奖励。** 蒸馏阶段的攻击意图来自**老师**：`RuleTeacher` 会在
   安全时把落点拉向 Boss 的 x（`boss_home`），而缺口老师 `GapTeacher` 把 2D 压成
   1D 车道、根本不看 Boss —— 所以 gap 线训出来的策略"只躲不打"是结构性的。
3. **PPO 微调当时是第二个独立问题 —— 已修。** 根因是两个数值 bug（GAE 价值被
   重复缩放、`load()` 没同步 lr）加上 `kl_anchor` 冻结时机错误；修完后 PPO **首次
   超过 DAgger**：3905 步 / 6274 伤害 / 77% 对准（DAgger R2 是 3521 / 5476 / 80%）。
4. 新增攻击指标（Boss 伤害 / 对准率 / 净分），避免"活着但一枪不打"被存活帧数掩盖。
5. **"分数进监督"已实现且可验证，但第一版老师不是更好的老师**：`ScoreTeacher`
   自己跑 4000 步不死、伤害 5082，可蒸馏出来的策略只有 33~41% 对准（规则老师
   能蒸到 80%）。原因见 §6.4 —— 9 选 1 的 argmax 标签比连续力场更难被 143 维 MLP 拟合。

## 1. 诊断：奖励实际在教什么

规则老师跑一局 mock 第一面（3196 步）的 `reward_parts` 累计：

| 分量 | 累计 |
|---|---|
| `danger` | **−3306** |
| `stage_clear` | +3000 |
| `alive` | +825 |
| `death` + `miss` | −1000 |
| `boss_damage` | +460 |
| `score` | **+0.2** |

`score_scale = 1e-5`，而对准 Boss 时 mock 每帧只加 4 分：一局 2 万分 × 1e-5 = 0.2。
危险度惩罚是它的 **16000 倍**。

另一组对照说明"对准"才是分数的来源（同一局面、同一种子）：

| 行为 | 存活步 | 净分 | 对准率 |
|---|---|---|---|
| 站原地只按 shoot | 632 | 636 | 7% |
| 只追 Boss x（完全不躲） | 743 | 4477 | 59% |
| 规则老师（躲 + 朝 Boss 归位） | 3196 | 20222 | 59% |

## 2. 改动

### 2.1 奖励：`attack` 配平（已成为 `config.json` 默认值）

| 权重 | 旧 | 新 | 理由 |
|---|---|---|---|
| `score_scale` | 1e-5 | **3e-2** | 对准每步 0.12，与 `alive_bonus`(0.3) 同量级；一局 2 万分 ≈ +600 |
| `danger_scale` | 4.0 | **0.8** | 退回"微调"，不再压过输出收益 |
| `danger_radius` | 50 | **32** | 只对真正贴脸的弹幕计罚 |
| `miss_penalty` / `death_penalty` | −200 / −600 | 不变 | 一次被弹 ≈ 1000 步攻击收益，"用命换分"仍不划算 |
| `boss_damage_scale` | 0.1 | 不变 | 真机上它比 score 更可靠（score 还受道具影响） |

具名 profile 仍保留：`tools/*.py --reward-profile attack`；`gap_survive` 仍会整份覆盖。

### 2.2 指标：`EpisodeRecord` 新增攻击诊断

`boss_damage`（掉血累计，符卡回血不算）/ `aligned_steps`+`boss_steps` →
`align_ratio` / `start_score` → `score_gain`。训练日志与 `tools/eval_policy.py`
都会打印，`--reward-profile` 可用于"按同一份奖励评估"。

### 2.3 修掉一个让 `kl_anchor` 静默失效的 bug

参考策略原来**只在 `reconfigure` 把老师摘掉的那一刻**冻结。而"载入蒸馏策略 →
直接 PPO"这种用法从启动就没有老师，切换点不存在 → `kl_anchor` 全程为 0。
现在 `Trainer.start()` 也会冻结（`_capture_rl_reference()`），并有回归测试。

### 2.4 蒸馏加强

第二轮 DAgger：载入第一轮策略 → 100 局，`teacher_epochs 12`、`danger_weight 3`、
`beta 0.3 → 0.15`、`lr 5e-4`。

## 3. 实验结果

评估协议：贪心，12 局，`step_frames=1`、`difficulty=0`、单局上限 4000 步；
"存活步""伤害""净分""对准率"都来自同一评估环境。

| 策略 | 存活步 | Boss 伤害 | 对准率 | 净分 | 过第一面 |
|---|---|---|---|---|---|
| 规则老师（参考） | 3196 | 5010 | 59% | 20222 | — |
| 蒸馏 R1（120 局） | 3043 | 4462 | 78% | 19113 | 12/12 |
| **蒸馏 R2（+100 局）** | **3521** | **5476** | **80%** | **22505** | **12/12** |
| PPO + 旧奖励（60 局） | 856 | 35 | 3% | 1128 | 0/12 |
| PPO + attack（60 局） | 999 | 45 | 2% | 1182 | 0/12 |
| PPO + attack + KL 1.0 | 1760 | 1735 | 48% | 7851 | 7/12 |
| PPO + attack + KL 2.0 + lr 3e-5 | 1280 | 1156 | 48% | 5176 | 6/12 |

训练过程中的趋势（每 20 局滚动均值，同一份蒸馏起点）：

| PPO 配置 | ep20 | ep40 | ep60 | 对准率 |
|---|---|---|---|---|
| 旧奖励 | 2152 步 / 1802 伤害 | 1492 / 944 | 971 / 109 | 34% → 20% → 5% |
| attack | 3347 / 2508 | 2513 / 1395 | 1754 / 420 | 43% → 25% → 8% |
| attack + KL 1.0 | 2406 / 3187 | 2048 / 2475 | 1505 / 1499 | 66% → 60% → **51%** |

读法：

- **奖励确实影响攻击意图**：同一批步数下，旧奖励的 PPO 把对准率打到 5%，
  attack 配置还能守在 8%~51%；KL 锚让"打"的行为掉得最慢。
- **但 PPO 本身在退化**（存活、伤害、对准同步下降），和 STATUS 里"PPO 会打崩
  DAgger"的记录一致。这不是奖励能单独解决的。
- **蒸馏加轮次是当前唯一稳定增益**：R1 → R2，存活 +16%、伤害 +23%、净分 +18%，
  且伤害/步（1.56）远高于任何 PPO 版本（≤0.84）。

## 4. 复现命令

```powershell
# R1: 规则老师 DAgger (120 局)
python tools\mock_full_clear.py --teacher rule_dagger --danger-grad --no-focus `
  --step-frames 1 --max-steps 4000 --batch 8 --hidden 256,256 --k-enemies 8 --k-bullets 16 `
  --episodes 120 --teacher-epochs 12 --danger-weight 3 --dagger-beta 0.8 --dagger-beta-min 0.3 `
  --eval-every 99999 --out .runtime\attack\distill

# R2: 在同一策略上再蒸一轮 (100 局, 老师接管比例降到 0.3)
python tools\mock_full_clear.py --teacher rule_dagger --danger-grad --no-focus `
  --step-frames 1 --max-steps 4000 --batch 8 --hidden 256,256 --k-enemies 8 --k-bullets 16 `
  --episodes 100 --teacher-epochs 12 --danger-weight 3 --dagger-beta 0.3 --dagger-beta-min 0.15 `
  --lr 5e-4 --load .runtime\attack\distill\policy.npz --eval-every 99999 `
  --out .runtime\attack\distill2

# 评估 (必须带与训练一致的观测/动作开关)
python tools\eval_policy.py --policy .runtime\attack\distill2\policy.npz --episodes 12 `
  --step-frames 1 --max-steps 4000 --hidden 256,256 --difficulty 0 --k-enemies 8 `
  --bullet-velocity --danger-grad --no-focus --reward-profile attack
```

## 5. 下一步（按优先级）

1. **把"分数"做进蒸馏**：现在的老师是"安全优先 + 硬编码朝 Boss 归位"，
   并不会在多个安全动作之间按**得分**取舍。正确做法是"奖励加权 DAgger"——
   在老师给出的安全候选集里按 `reward`（分数/伤害）排序再打标签，这样
   `score_scale` 才真正进入监督信号。
2. **PPO 要么修好、要么绕开**：修方向是每 batch 评估早停 + 余弦 lr + 更小
   batch；绕开方向是用 DAgger/优势加权回归做离线改进，不做 on-policy 大步更新。
3. **模式基准加入 `align_ratio`**：只用存活帧数验收，等于允许"贴角不打"。

---

## 6. 第二轮（2026-10-21）：分数进监督 + 修好 PPO + GIF

### 6.1 `ScoreTeacher`：安全是硬约束，分数是排序目标

`training/score_teacher.py` 把决策拆成两层，对应"生存第一、其次最大化分数"：

1. **过滤**：9 个方向各做时间展开（`HORIZON = 0/3/6/10/15/22` 帧），取未来与
   弹幕（含速度预测）的最小间距 `clearance`；低于 `safety_margin`（默认 12px）出局；
2. **排序**：在安全方向里最大化
   `(score_scale + boss_damage_scale) × score_rate × 对准帧数 − danger_scale × 危险度`，
   其中"对准帧数"= 路径 x 与目标 x（Boss，否则最近敌机）在 `align_tol` 内的加权帧数，
   `score_rate = 4`（mock 对准时每帧的得分）。

`score_scale` 直接进公式，因此**标签会随权重变化**；`tests/test_score_teacher.py`
用"同一局面，`score_scale=0` 就该原地不动、`>0` 就该走过去"锁住了这个契约。

老师自身实测（seed 101/202，5000 步上限）：

| 老师 | 存活 | Boss 伤害 | 对准率 | 方向切换率 |
|---|---|---|---|---|
| `RuleTeacher` | 3196 / 3306 | 5010 / 5334 | 82~84% | 82.9% |
| `ScoreTeacher` | 4000（到上限） | 5082 / 5006 | 58~74% | **23.5%** |

### 6.2 蒸馏结果：机制成立，但第一版不是更好的老师

单种子评估（贪心 12 局，`step_frames=1`）：

| 策略 | 存活步 | Boss 伤害 | 对准率 | 净分 | 过第一面 |
|---|---|---|---|---|---|
| rule-DAgger R2（基线） | 3521 | 5476 | 80% | 22505 | 12/12 |
| score-DAgger R1（120 局） | 1339 | 758 | 33% | 2725 | 1/12 |
| score-DAgger R2（+100 局） | 2002 | 1393 | 41% | 6098 | 6/12 |
| score-DAgger R3（+120 局） | 2594 | 1704 | 37% | 7752 | 6/12 |

R3 比 R2 好，说明不是"训不动"，但离规则老师那条线仍差一大截。

### 6.3 修好 PPO（两个数值 bug + 一个时机 bug）

| bug | 现象 | 根因 | 修法 |
|---|---|---|---|
| GAE 价值被**重复缩放** | 第一轮更新 `KL=5.2`、`clip=46%`，策略一更新就崩 | `_gae` 把 `values` 也除以 `scale`，而价值头学的本来就是"缩放后的回报" → 基准≈0，优势退化成逐步即时奖励 | `rewards/scale`，`values`/`bootstrap` 不除 |
| `load()` 不同步 lr | `--lr 3e-5` 完全无效，实际用 checkpoint 的 `1e-3` | `Trainer.load` 只替换 policy，不回写 config 的超参 | 新增 `apply_hyperparams()`，load/start 都同步 |
| `kl_anchor` 冻结时机 | 从磁盘直接 PPO 时锚恒为 0 | 只在"摘掉老师"的 reconfigure 里冻结 | `start()` 也调 `_capture_rl_reference()` |

修完后的效果（同一份 rule-DAgger R2 策略起步，80 局 PPO，attack 奖励）：

| | 存活步 | Boss 伤害 | 对准率 | 净分 | 过第一面 |
|---|---|---|---|---|---|
| 修复前（60 局） | 856~1760 | 35~1735 | 3~48% | 1128~7851 | 0~7/12 |
| **修复后（80 局）** | **3905** | **6274** | **77%** | **24861** | **12/12** |

`KL` 从首轮 5.2 降到 0.08~0.39，`value_loss` 从 33 降到 0.001。**PPO 第一次真正
是"微调"而不是"摧毁"**。

### 6.4 为什么 score 监督反而更差

- **标签更难拟合**：`ScoreTeacher` 每帧做 9 选 1 的 argmax（价值函数里含
  clearance 的分段行为），而 `RuleTeacher` 输出的是连续力场量化后的方向。
  143 维 MLP 对后者的拟合误差小得多（策略 80% vs 老师 84%），对前者掉到 37~41%。
- **对准的定义更"窄"**：ScoreTeacher 只在 22 帧内可对齐时才追求对齐（`aligned`），
  超出后只剩很弱的 `approach` 项，所以老师自身对准率（58%）就低于规则老师（84%）。
- 这说明"把分数塞进标签"不能靠**替换**老师，而应该在**规则老师的安全先验上重排**。

### 6.5 产物

| 文件 | 内容 |
|---|---|
| `.runtime/policies/attack_ppo_fixed.npz` | **当前最强**：rule-DAgger R2 + 修复后 PPO |
| `.runtime/policies/score_dagger_r3.npz` | 分数监督（3 轮 DAgger） |
| `.runtime/gifs/ppo_fixed/demo_ep1.gif` | 第 3 面、剩 2 残、20222 分（种子 101，3000 步） |
| `.runtime/gifs/rule_distill/demo_ep1.gif` | 第 3 面、剩 1 残、19874 分 |
| `.runtime/gifs/score_distill/demo_ep1.gif` | 第 2 面阵亡、10073 分 |

### 6.6 下一步（修正后的方向）

1. **hybrid 老师**：保留 `RuleTeacher` 的力场作为安全先验，只对它的 top-k 候选
   按 reward 重排 —— 既保住可模仿性，又让 `score_scale` 进入标签。
2. **PPO**：现在能用了，可以开始调 `target_kl`（0.03 → 0.02）与 `kl_anchor`，
   并给 `ppo_minibatch` 做按批大小自适应。
3. **评估**：把 `align_ratio` 与"伤害/千步"纳入模式基准的固定报表。
