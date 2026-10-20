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
| `research/gap_benchmark.py` | 避弹基准：站着不动 vs 斥力老师 vs 时空规划器 |

```powershell
.\.venv\Scripts\python.exe tools\research\gap_rl.py --arm geo --updates 25 `
    --bc-episodes 30 --bc-epochs 8 --bc-coef 2.0 --shaping 2.0

# 正式链路: 缺口观测 + 过面进度奖励 + slack 标签, 留出场景评估
.\.venv\Scripts\python.exe tools\research\gap_trainer.py --bench v2 --label slack `
    --reward gap_survive --bc-episodes 30 --dagger-episodes 60 --rl-episodes 200
```

## train — 训练入口

| 脚本 | 用途 |
|---|---|
| `train_patterns.py` | 按弹幕类型分别训练专属策略（每类一个） |
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
