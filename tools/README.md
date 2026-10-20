# tools/ — 命令行入口索引

这里的脚本都是**独立 CLI**（`python tools/xxx.py --help`），不是 import 用的包。
库代码在 `touhou_rl_mcp/`；做实验请优先用 `tools/research/` 下的可复现脚本。

> 约定：脚本一律用 `sys.path.insert(0, <项目根>)` 起步，因此放在越深的目录，
> `parent` 的层数要同步改（见各文件开头）。

## research/ — 可复现实验

| 脚本 | 用途 |
|---|---|
| `research/gap_rl.py` | 缺口任务最小学习实验：观测对照 / 感知弱监督 / BC 锚，写标准 run 记录 |
| `research/gap_trainer.py` | 同一批实验走**正式训练链路**（Trainer）：`use_gap` 观测 + 新奖励 + 缺口老师，BC→DAgger→PPO 分阶段 |
| `research/gap_diagnose.py` | 把失败局拆开：死在**哪一堵墙**、赶路来不来得及、最后 30 帧有没有朝缺口走 |
| `research/gap_gif.py` | 训练好的策略放回 v3 场景跑，输出 GIF + 自机/缺口横向轨迹图 |
| `research/gap_report.py` | 存活饱和时改看"飞行方式"（擦弹、最近弹距、横移占比） |
| `research/aimed_probe.py` | 单颗斜向自机狙探针：检查策略是否把"自机所在车道"误判成安全 |
| `research/gap_benchmark.py` | 避弹基准：站着不动 vs 斥力老师 vs 时空规划器 |

```powershell
# 正式链路（v3 上目前最好的一组）
.\.venv\Scripts\python.exe tools\research\gap_trainer.py --bench v3 --label reach `
    --bc-episodes 60 --dagger-episodes 100 --rl-episodes 120 `
    --eval-every 20 --eval-phases 3 --rand --kl-anchor 1.0

# P0: dev/test 分离。训练区间之外、互不重叠的两组片段
# dev 只用于选 policy_best; test 只在最后报一次
.\.venv\Scripts\python.exe tools\research\gap_trainer.py `
    --script .runtime\danmaku\th11_replay_full.json --script-train-range 0 0.55 `
    --dev 0.56,0.62,0.68,0.74,0.80,0.86,0.92,0.98 `
    --test 0.59,0.65,0.71,0.77,0.83,0.89,0.95 `
    --label reach --bc-episodes 40 --dagger-episodes 60 --rl-episodes 40

# P1: 目标承诺 / 滞回 / 动作年龄
.\.venv\Scripts\python.exe tools\research\gap_trainer.py `
    --script .runtime\danmaku\th11_replay_full.json --script-train-range 0 0.55 `
    --dev 0.56,0.62,0.68,0.74,0.80,0.86,0.92,0.98 `
    --test 0.59,0.65,0.71,0.77,0.83,0.89,0.95 `
    --label reach_hold --commit-frames 18 --action-age `
    --bc-episodes 40 --dagger-episodes 60 --rl-episodes 40

# 历史 run 的诚实评估（不训练，读 policy_best 或 --last 读最后一版）
.\.venv\Scripts\python.exe tools\research\gap_trainer.py --eval-only `
    --run-dir .runtime\runs\script-best --last `
    --dev 0.56,0.62,0.68,0.74,0.80,0.86,0.92,0.98 `
    --test 0.59,0.65,0.71,0.77,0.83,0.89,0.95

# 真机录制的弹幕（脚本模式）：每局从脚本随机位置开始，评估用最后 25% 的未训练片段
.\.venv\Scripts\python.exe tools\research\gap_trainer.py `
    --script .runtime\danmaku\th11_replay_full.json --label reach `
    --bc-episodes 60 --dagger-episodes 100 --kl-anchor 1.0

# 失败诊断：每一局死在哪、来不来得及
.\.venv\Scripts\python.exe tools\research\gap_diagnose.py --run .runtime\runs\v3-anchor1
.\.venv\Scripts\python.exe tools\research\gap_diagnose.py --run .runtime\runs\v3-anchor1 --teacher

# 看它怎么飞（GIF + 轨迹；默认优先用按评估择优的 policy_best）
.\.venv\Scripts\python.exe tools\research\gap_gif.py --run .runtime\runs\v3-anchor1 --all
.\.venv\Scripts\python.exe tools\research\gap_gif.py --run .runtime\runs\script-best --all
# 指定脚本片段（默认优先读 run 的 test_fracs）；新策略会带上 action_age
.\.venv\Scripts\python.exe tools\research\gap_gif.py --run .runtime\runs\gif-hold18 --last `
    --fracs 0.71,0.89 --all

# 单颗斜向自机狙探针（检查"自机所在车道是否被误判成安全"）
.\.venv\Scripts\python.exe tools\research\aimed_probe.py `
    --run .runtime\runs\gif-aimfix --seeds 11,17,23
.\.venv\Scripts\python.exe tools\research\aimed_probe.py `
    --run .runtime\runs\gif-aimfix --seeds 17 --gif
```

## train — 训练入口

| 脚本 | 用途 |
|---|---|
| `train_patterns.py` | 按弹幕类型分别训练专属策略（每类一个）；`--teacher none` = 纯 PPO |
| `train_loadouts.py` | 在真机各机体组合上分别继续训练一个已有 checkpoint |
| `dual_train.py` | 双线程：真机线程 + mock 线程共享同一策略 |
| `dual_proc_train.py` | 双进程 + 周期合并（配 `mock_worker.py`） |
| `mock_full_clear.py` | 在 mock 上训练到打通全部 6 面 |
| `auto_pipeline.py` | 循环流水线：真机 + mock → 合并 → 继续 |
| `bc_from_replay.py` | 从回放数据做行为克隆 |

## eval / 观察 — 看它到底在干什么

| 脚本 | 用途 |
|---|---|
| `eval_policy.py` | 贪心评估已保存策略（不带探索） |
| `watch_training.py` | 训练过程中的 Tkinter 实时监视窗口 |
| `analyze_training_log.py` | 离线分析 `train.jsonl` / `steps.jsonl`，定位死因与动作偏好 |
| `grid_preview.py` | 画出"危险图"观测，看清策略眼里的弹幕地图 |
| `play_real.py` | 用 mock 训好的策略在真机上贪心跑 |
| `mock_demo.py` | mock 演示一局，输出 GIF + 关键帧拼图 |

## calibrate — 真机标定（Windows + 游戏本体）

| 脚本 | 用途 |
|---|---|
| `probe.py` | 真机探针：内存偏移标定命令行 |
| `calibrate_player.py` | 标定自机坐标 `player_x` / `player_y` |
| `find_invincible.py` | 定位无敌帧偏移 |
| `inspect_th11_entities.py` | 只读查看 TH11 实体布局 |
| `validate_real_observation.py` | 只读校验 TH11 观测是否正常 |
| `record_danmaku.py` | 录制真机弹幕，生成 mock 可复刻的脚本 |
| `record_replay.py` | 播放 replay 并逐帧录制弹幕（可 `--bc-out` 导出 BC 数据） |

## legacy/ — 已归档

被取代的一次性脚本，保留仅作参考。说明见 `legacy/README.md`。
不要再往这里加东西；新脚本按上面的分类放。
