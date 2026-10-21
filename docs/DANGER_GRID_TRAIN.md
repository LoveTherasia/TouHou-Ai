# 危险图 + 墙特征训练（2026-10-21）

> 目标：**彻底不用 `k_bullets`/斥力**，用"危险图（2D 栅格）+ 12 维墙特征 + 规划器标签"
> 训练一个能同时处理普通弹幕和墙的策略。
> 代码：`mock_full_clear.py --grid ... --wall-feat`；验收：`pattern_bench.py --grid ... --wall-feat`。

## 1. 观测

- 危险图：以自机为中心，`12×12` 栅格，`±330px`，含速度通道（3 通道）；
- 墙特征：末尾追加 12 维（覆盖率 / 缺口中心 / 缺口宽 / ETA / 8 列占用）；
- 全局特征：自机/Boss/残机/无敌/危险梯度；
- 总维度：**459**；
- 完全不使用 `k_bullets`（`use_grid=True` 时该参数不参与编码）。

## 2. 训练

| 项 | 值 |
|---|---|
| 起点 | 从零训练（没有 143/155 维 checkpoint） |
| 老师 | `viability` 规划器（时间展开反向 DP），`planner_dt=4` |
| 弹型 | `wall_gate`×3 + `aimed_single / aimed_spread / dense_ring / ring_gate / closing_ring / random_spray` |
| 局数 | 48 局 |
| 单局 | 1200 步，`step_frames=1`，9 动作（无 focus） |
| 网络 | 256×256 MLP |

## 3. 结果

`pattern_bench`，2 seeds，上限 2000 步：

| 弹型 | 危险图+墙特征 (459) | 旧 k_bullets 基线 (143) | 规划器 |
|---|---|---|---|
| `wall_gate` | **2000（0/2 死）** | 814（3/3 死） | 2000 |
| `aimed_single` | 1708（1/2 死） | 2000 | 2000 |
| `aimed_spread` | **548（2/2 死）** | 2000 | 1770 |
| `dense_ring` | **513（2/2 死）** | 2000 | 2000 |
| `ring_gate` | **682（2/2 死）** | 2000 | 2000 |
| `closing_ring` | **758（2/2 死）** | 2000 | 1990 |
| 平均 | **1035（52%）** | 1852（93%） | 1970（98%） |

## 4. 结论

1. **墙确实学会了**：`wall_gate` 从 814 提到 2000，说明"危险图 + 墙特征"能看见
   384px 宽的墙和缺口，不再需要分层切换兜底。
2. **但普通弹幕严重欠拟合**：`dense_ring` 513、`aimed_spread` 548、`ring_gate` 682，
   与规划器 1970 差得很远。48 局 × 1200 步只有约 5.8 万样本，对 459 维输入 +
   6 种弹型远远不够。
3. **没有 k_bullets/斥力**也能学墙，这一点成立；但"单策略多任务"仍需要更多数据
   或更好的训练分解。
4. 当前最好方案仍是 **分层（通用策略 + 规划器/墙专家）**；危险图+墙特征可作为
   墙专家的观测，而不强求一个 459 维 MLP 同时学完所有弹型。

## 5. 下一步

1. **把数据量提上去**：每类弹型至少 50~100 局（总 300~500 局），或先分场景训练再合并。
2. **先训危险图通用策略**（不含墙），再用墙特征做墙专家 DAgger；最后用学习门控合并。
3. **训练目标**：当前是纯模仿规划器，规划器动作在墙段/普通段分布差异大；可尝试
   - 分阶段课程（普通 → 墙 → 混合）；
   - 分类别输出头 / 专家网络；
   - 只对墙段使用规划器标签，普通段用规则/hybrid 老师。
4. **验收**：固定 `pattern_bench`，报均值/最差/死亡/伤害/对准；不要只看 `wall_gate`。

## 6. 复现

```powershell
# 训练
.\.venv\Scripts\python.exe tools\mock_full_clear.py `
  --pattern-cycle wall_gate,wall_gate,wall_gate,aimed_single,aimed_spread,dense_ring,ring_gate,closing_ring,random_spray `
  --grid --grid-size 12 --grid-span 330 --wall-feat --no-focus `
  --teacher viability --planner-dt 4 `
  --episodes 48 --max-steps 1200 --step-frames 1 --batch 8 --hidden 256,256 `
  --lr 2e-4 --teacher-epochs 4 --eval-every 0 --eval-episodes 5 `
  --out .runtime/grid_wall_mixed

# 验收
.\.venv\Scripts\python.exe tools\pattern_bench.py `
  --policy .runtime/grid_wall_mixed/policy_best.npz `
  --patterns aimed_single,aimed_spread,dense_ring,ring_gate,closing_ring,wall_gate `
  --seeds 2 --max-steps 2000 --step-frames 1 --no-focus `
  --grid --grid-size 12 --grid-span 330 --wall-feat --bullet-velocity --danger-grad
```
