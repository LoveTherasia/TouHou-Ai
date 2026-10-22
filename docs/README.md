# 文档地图

> 18 篇文档容易失焦，这张表说明**每篇管什么、现在还算不算数**。
> 想快速了解项目结论 → 读 [`CONCLUSIONS.md`](CONCLUSIONS.md)；
> 想看当前状态与优先级 → 读 [`STATUS.md`](STATUS.md)。

## 1. 现行（结论仍然有效，优先读）

| 文档 | 管什么 | 说明 |
|---|---|---|
| [`CONCLUSIONS.md`](CONCLUSIONS.md) | **结论总表** | 什么被证实 / 什么已作废 / 什么没解决。所有专题文档的收敛结果 |
| [`STATUS.md`](STATUS.md) | 当前状态、能力边界、路线图 | 唯一的状态来源；历史实验流水已移到 `history/` |
| [`REAL_GAP.md`](REAL_GAP.md) | 真机差距归因、replay 数据修复、加难课程 | 最新（2026-10-21 晚），含所有 replay 相关数字 |
| [`PLANNER_V2.md`](PLANNER_V2.md) | **规划器 v2 设计**：Deep Sets + 多目标奖励 + 动力学 + MPC/CEM + 蒸馏 | 2026-10-22，含 M1~M6 实测与三个坑 |
| [`TH11_ENTITIES.md`](TH11_ENTITIES.md) | TH11 内存实体布局（弹/敌/Boss/道具） | 参考手册，地址与签名 |
| [`OFFSET_CALIBRATION.md`](OFFSET_CALIBRATION.md) | 偏移标定流程 | 换作品/换版本时照着做 |
| [`COMBAT_TRAINING.md`](COMBAT_TRAINING.md) | 战斗训练用法、符卡与对话快进 | 操作手册 |
| [`DASHBOARD.md`](DASHBOARD.md) | 训练看板 | 工具用法 |
| [`ITEMS.md`](ITEMS.md) | 道具（P 点）偏移的尝试与排除范围 | **未找到**，记录了不要重复试的路 |
| [`HIERARCHICAL.md`](HIERARCHICAL.md) | 分层切换（策略 + 规划器） | 含三次训练的负面结果 |
| [`VIABILITY_PLANNER.md`](VIABILITY_PLANNER.md) | 可行域规划器 | wall_gate 814→2000 的来源 |

## 2. 专题记录（结论已并入 CONCLUSIONS，过程留档）

| 文档 | 内容 | 注意 |
|---|---|---|
| [`ATTACK_INTENT.md`](ATTACK_INTENT.md) | 攻击意图诊断、奖励配平、PPO 三个 bug | PPO "必崩" 结论已作废 |
| [`HYBRID_TEACHER.md`](HYBRID_TEACHER.md) | hybrid 老师设计 + 弹型基准 + 负面结果 | "老师强 ≠ 学生强" |
| [`GAP_LEARNING.md`](GAP_LEARNING.md) | 缺口观测学习的完整实验线（最长的一篇） | 结论：gap 只是高层先验，不是通用躲避 |
| [`VALUE_DISTILL.md`](VALUE_DISTILL.md) | V(s) + 规划 + 蒸馏原型 | 蒸馏分支 86~92% 时间贴角 |
| [`WALL_FEAT_MOCK.md`](WALL_FEAT_MOCK.md) | 155 维墙特征 mock 验收 | 学会墙但遗忘通用弹型 |
| [`DANGER_GRID_TRAIN.md`](DANGER_GRID_TRAIN.md) | 危险图（无 k_bullets）+ 墙特征训练 | 48 局不够，墙 2000 但普通弹型均值 1035 |

## 3. 历史 / 快照（不要再当现状引用）

| 文档 | 为什么归档 |
|---|---|
| [`REPLAY_CLONING.md`](REPLAY_CLONING.md) | 设计稿；实际路线已改为"录制弹幕脚本 + mock 复刻"（见 REAL_GAP） |
| [`STAGE1_ATTEMPTS.md`](STAGE1_ATTEMPTS.md) | 2026-10-17 第一面尝试记录，已被真机通关第 1 面取代 |
| [`PROJECT_REVIEW.md`](PROJECT_REVIEW.md) | 2026-10-20 工程评审快照（模块数/测试数已变） |
| [`history/`](history/) | 历史会话记录与 2026-10-21 实验流水 |

## 4. 约定

1. **数字只能有一个来源**：现状数字进 `STATUS.md` 或 `CONCLUSIONS.md`，专题文档只写
   过程与判断，不重复维护同一张表。
2. **作废的结论要显式标注**，而不是删掉 —— `CONCLUSIONS.md` §3 是唯一清单。
3. 新增实验记录前先看 `CONCLUSIONS.md` §3，避免重跑已证伪的路。
