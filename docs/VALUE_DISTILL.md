# V(s) + 短视界规划 + 蒸馏：第一次实验（2026-10-20）

> 状态：**实验性，尚未达到"真正会躲"**。本文件记录设计、结果和失败诊断。
> 复现入口：`tools/research/value_distill.py`。

## 1. 设计

```
当前子弹 (位置 + 速度)
        │
        ├─ V(s): 108 维局部特征 → 128×128 MLP → "未来还能活多久" (0~1)
        │
        └─ ValuePlanner: 9 方向 × 深度 D 的 beam search
              每个叶子构造 GameState → V(s_leaf) + 安全间隙 + 边界惩罚
              → 选第一步
        │
        └─ 规划器示范 (state, action) → BC → MLP 策略
```

### V(s) 训练

- 数据：规则老师（`RuleTeacher(planner="horizon")`）+ 随机策略；
- 弹型：`aimed_single / aimed_spread / dense_ring / ring_gate / closing_ring /
  wall_gate / random_spray`；
- 随机化：自机起点、弹型速度/间隔/相位/漂移；
- 标签：`min(剩余帧, 600)/600 - edge_weight * 边界惩罚`；
- 网络：108 → 128 → 128 → 1，numpy Adam，MSE。

### 规划器

- `MacroPlanner` 的碰撞检查（3 帧/宏动作，9 方向）；
- 深度 `D=6~10`，beam 12~24；
- 叶子用 V(s) 评分，不用"离弹越远越好"的手写代价；
- 可选边界惩罚，避免学成贴角。

### 蒸馏

- 规划器在多种弹型上生成 `(policy_obs, action)`；
- `policy_obs = 108 维状态特征 + 上一动作 one-hot(9) + 同向连续步数(1)`；
- `MLPPolicy(118 → 128 → 128 → 9)`，`imitate_batch`；
- DAgger：在策略自己走到的状态上再问规划器，补标签，再训练。

## 2. 结果

### V(s)

| 数据 | val MSE | val MAE | val corr |
|---|---|---|---|
| 18 828 样本（无边界惩罚） | 0.0208 | 0.107 | **0.895** |
| 12 632 样本（edge=0.6） | 0.0265 | 0.120 | 0.890 |
| 12 632 样本（edge=2.0） | 0.0422 | 0.146 | 0.867 |

V 能排序"现在更危险/更安全"，但排序 ≠ 会规划。

### 规划器（确定性 seed 11，dt=3）

| 弹型 | 规划器 | 站桩 |
|---|---|---|
| `aimed_single` | **2400** | 468~? |
| `aimed_spread` | **2400** | — |
| `ring_gate` | **2400** | 783 |
| `closing_ring` | 1020（edge 0.6）/ 1374（edge 2.0） | 546 |
| `wall_gate` | 828（edge 0.6）/ 747（edge 2.0） | — |

规划器能找到上穿/横向对齐，但在 `wall_gate` 这种需要多步站位的场景仍然弱。

### 蒸馏策略（`policy_dagger.npz`）

模式评估（随机化 seeds 31/37）：

| 弹型 | 存活 | 贴角比例 |
|---|---|---|
| `aimed_single` | **2400 / 2400** | 92% |
| `ring_gate` | **2400 / 2400** | 75% |
| `random_spray` | **2400 / 2400** | 79% |
| `dense_ring` | 1143 / 1065 | 87% |
| `aimed_spread` | 840 / 1080 | 87% |
| `closing_ring` | 834 / 1017 | 0.4% |
| `wall_gate` | 687 / 729 | 67% |

完整 replay（`th11_replay_full.json`，7 个 test 片段，dt=3）：

| 片段 | 蒸馏策略 | 站桩 | 规划器 |
|---|---|---|---|
| 0.59 | 2400 | 1776 | 2400 |
| 0.65 | 2400 | 2400 | 2400 |
| 0.71 | 2400 | 1992 | 2400 |
| 0.77 | 2400 | 2055 | 2154 |
| 0.83 | 2400 | 2400 | 1530 |
| 0.89 | 1911 | 1383 | 1104 |
| 0.95 | 2400 | 2166 | 777 |
| **均值** | **2330** | **2025** | **1824** |

## 3. 关键失败：回放高分是"贴角"而不是"会躲"

轨迹检查发现 `policy_dagger` 在 replay 的 0.59~0.89 片段有 **86~92%** 的时间
位于 `x<24 / x>360 / y>416` 的角落，例如：

- 0.83：坐标 `(376, 440)`；
- 0.89：坐标 `(3, 426)`。

回放里很多弹幕覆盖不到底角，所以"去角落"能活满，但它不是通用躲避。
模式评估也印证了这一点：`aimed_spread / dense_ring / closing_ring / wall_gate`
都只有 500~1200 帧。

尝试加边界惩罚：

| 配置 | replay 均值 | 贴角比例 | 模式表现 |
|---|---|---|---|
| 无惩罚 | 2330 | 86~92% | 见上 |
| edge=0.6 | 约 2200 | 65~85% | 下降 |
| edge=2.0 | 约 1600 | 54~90% | 进一步下降 |

**结论：只在 V 标签/叶子评分里加边界惩罚不够**。BC 仍然会从示范/状态分布里学到
贴角；需要把惩罚放进**训练目标/PPO 奖励**，并把"贴角比例"作为硬性验收指标。

## 4. 当前结论

- **V(s) + 短视界规划 + 蒸馏这条链路是通的**：V 排序有效，规划器能在
  `ring_gate` 找到上穿（783 → 2400），蒸馏策略能在单弹/环缺口上 2400；
- **但第一版策略不是"真正会躲"**：它在完整 replay 上靠贴角拿到 2330，模式泛化
  不稳定；
- 单看 replay 存活会严重高估能力；必须同时看：
  - 模式基准的**最差局**；
  - **贴角比例**；
  - 上穿/逃逸成功率。

## 5. 下一步（按优先级）

1. **把贴角惩罚放进 PPO/DAgger 的训练目标**，不是只放进 V 标签；目标：
   模式基准上贴角比例 <10%，且存活不低于当前。
2. **用模式基准做主验收**：`aimed_single / aimed_spread / dense_ring / ring_gate /
   closing_ring / wall_gate`，固定 5 seeds，报最差局 + 贴角比例。
3. **改进蒸馏**：更多随机化种子、DAgger 2~3 轮、策略加 GRU/3 帧动作历史；
   当前 118 维 MLP 对"规划器的多步承诺"仍有歧义。
4. **规划器**：V 作为终端价值已经可用；下一步加长深度/更细网格，或学习 V 的
   不确定性；`wall_gate` 需要多步站位，单靠短视界仍然弱。
5. **回放评估**：先过滤掉贴角片段，或把回放和模式基准一起报；不要把 2330 当成
   "会躲"的证据。

## 6. 复现

```powershell
# 1) 收集价值数据 + 训练 V
.\.venv\Scripts\python.exe tools\research\value_distill.py --stage value `
    --seeds 11,17 --max-frames 2400 --value-epochs 30

# 2) 规划器示范 + BC
.\.venv\Scripts\python.exe tools\research\value_distill.py --stage distill `
    --seeds 11,17 --max-frames 2400 --depth 6 --beam 12 --epochs 100

# 3) 评估
.\.venv\Scripts\python.exe tools\research\value_distill.py --stage eval `
    --seeds 31,37 --max-frames 2400 --dt 3
```

> 当前最好 replay 策略是 `policy_rand.npz` 经过一轮 DAgger 后的
> `policy_dagger.npz`；DAgger 数据/训练脚本目前记录在本文件的实验流程中，
> 下一步会收敛成 CLI 的 `--stage dagger`。
