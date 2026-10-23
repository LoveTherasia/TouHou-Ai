"""在 mock 上训练到【完整通关】(打通全部 6 面)。

相对 examples/train.py 的增强 (针对"被弹刷输出"漏洞与多符卡 Boss):
  * 奖励**统一来自 config.json 的 reward 段** (不再自带一套参数);
  * 观测: 敌机位置 + 弹幕速度 + 无敌帧;
  * 动作: 9 快 + 9 慢(focus) 共 18 个;
  * 算法: PPO (GAE + clipped surrogate), 可选 reinforce_baseline;
  * 熵退火 + 贪心评估 + 最佳快照 (只在变好时存 policy_best);
  * 课程学习: --start-stage 从第 N 面开局, 逐面推进。

    python tools/mock_full_clear.py --episodes 20000
    python tools/mock_full_clear.py --start-stage 3 --episodes 8000 --out .runtime/mock_s3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from touhou_rl_mcp.config import Config  # noqa: E402
from touhou_rl_mcp.env import TouhouEnv  # noqa: E402
from touhou_rl_mcp.games.registry import GameRegistry  # noqa: E402
from touhou_rl_mcp.rl.reward import REWARD_PROFILES, reward_profile  # noqa: E402
from touhou_rl_mcp.training import TrainConfig, Trainer  # noqa: E402
from touhou_rl_mcp.training.visualize import save_png  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20000)
    ap.add_argument("--max-steps", type=int, default=12000)
    ap.add_argument("--step-frames", type=int, default=4)
    ap.add_argument("--hidden", default="256,256")
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--gamma", type=float, default=0.99,
                    help="折扣因子; step-frames=1(60Hz) 时建议 0.997~0.999")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--entropy", type=float, default=0.06)
    ap.add_argument("--min-entropy", type=float, default=0.01,
                    help="熵退火的下限 (防止过早塌缩到单一动作)")
    ap.add_argument("--k-enemies", type=int, default=8)
    ap.add_argument("--k-bullets", type=int, default=16,
                    help="观测里最近的弹幕数 (密集弹幕可调大到 32)")
    ap.add_argument("--danmaku-script", default="",
                    help="训练用的真机弹幕脚本 (空=内置弹型)")
    ap.add_argument("--eval-danmaku-script", default="",
                    help="贪心评估用的弹幕脚本 (空=内置弹型; 用来区分策略强弱)")
    ap.add_argument("--script-offset-range", default="",
                    help="脚本模式的起始偏移范围 (占全长的比例, 如 0,1 = 整段随机); "
                         "固定 offset 会让每局都练同一段弹幕")
    ap.add_argument("--danmaku-offset", type=int, default=0,
                    help="弹幕脚本起始帧偏移 (用来直接练某一面)")
    ap.add_argument("--danmaku-pattern", default="",
                    help="弹幕类型库: dense_ring / expanding_homing_ring / aimed_trail / "
                         "static_swarm / spiral / fan / wall_gap / random_spray / flower / aimed_spread")
    ap.add_argument("--algorithm", default="ppo", choices=["ppo", "reinforce_baseline"])
    ap.add_argument("--teacher", default="",
                    choices=["", "guard", "dagger", "rule", "rule_dagger", "gap", "gap_dagger",
                             "score", "score_dagger", "hybrid", "hybrid_dagger",
                             "viability", "hier", "hier_dagger", "mpc", "mpc_dagger"],
                    help="模仿学习: rule=规则老师(与 dodge 宏同源)接管并克隆; rule_dagger=策略驱动+老师打标签; "
                         "score/score_dagger=按奖励权重(score_scale)在安全方向里排序打标签; "
                         "hybrid/hybrid_dagger=力场先验 + 分数重排; "
                         "hier/hier_dagger=分层: 只把'整幅横墙/多向收网'的步交给规划器打标签")
    ap.add_argument("--dagger-beta", type=float, default=0.2)
    ap.add_argument("--dagger-beta-min", type=float, default=0.15,
                    help="DAgger 老师接管比例的调度下限 (从 --dagger-beta 线性降到它)")
    ap.add_argument("--score-safety-margin", type=float, default=12.0,
                    help="score 老师: 未来 H 步最小间距低于它就出局 (px)")
    ap.add_argument("--score-align-tol", type=float, default=20.0,
                    help="score 老师: |自机 x − 目标 x| 小于它算对准 (px)")
    ap.add_argument("--score-commit-margin", type=float, default=0.0,
                    help="score 老师: 切换方向需要的价值优势 (滞回, 减少左右互搏)")
    ap.add_argument("--hybrid-prior-weight", type=float, default=2.5,
                    help="hybrid 老师: 力场先验权重 (0=退化成纯 score 老师)")
    ap.add_argument("--hybrid-barrier-weight", type=float, default=25.0,
                    help="hybrid 老师: 软安全屏障权重 (clearance 归零时的惩罚)")
    ap.add_argument("--hybrid-align-horizon", type=float, default=22.0,
                    help="hybrid 老师: 对准收益的估计窗口(帧); 拉长会牺牲存活")
    ap.add_argument("--mpc-horizon", type=int, default=24,
                    help="MPC 老师视界(帧): 24=反应式; 72+ 才会为墙让位")
    ap.add_argument("--mpc-dt", type=int, default=3, help="MPC 老师的控制步长(帧)")
    ap.add_argument("--mpc-samples", type=int, default=48, help="MPC 的 CEM 采样数")
    ap.add_argument("--mpc-iters", type=int, default=3, help="MPC 的 CEM 迭代轮数")
    ap.add_argument("--mpc-gap", type=float, default=0.0,
                    help="缺口锚点权重 (墙/闭合环需要 >0)")
    ap.add_argument("--mpc-column", type=float, default=0.0,
                    help="正上方弹幕柱权重 (与 --mpc-gap 二选一)")
    ap.add_argument("--mpc-adaptive", action="store_true",
                    help="按墙体检测切换长短视界")
    ap.add_argument("--soft-labels", action="store_true",
                    help="用老师的逐动作代价做**软标签**蒸馏 (代替 argmax; "
                         "MPC 的 argmax 自一致率只有 ~25%%)")
    ap.add_argument("--soft-temp", type=float, default=20.0,
                    help="软标签温度: 代价差多少算'同样好'")
    ap.add_argument("--mpc-plan-every", type=int, default=1,
                    help="MPC 规划一次执行几步 (2~3 更稳也更快)")
    ap.add_argument("--mpc-commit", type=int, default=0,
                    help="MPC 老师「并列动作承诺」: 选定方向后 N 帧不换向 "
                         "(把标签噪声从源头掐掉; 建议 2~4)")
    ap.add_argument("--mpc-switch", type=float, default=0.0,
                    help="承诺期内换向所需的代价优势 (px/代价单位)")
    ap.add_argument("--mpc-focus-margin", type=float, default=0.0,
                    help="MPC 老师: 慢速(focus)要比全速便宜这么多才按 shift")
    ap.add_argument("--no-mpc-deterministic", dest="mpc_deterministic",
                    action="store_false",
                    help="关闭「每帧用 seed+frame 重播种 CEM 噪声」的确定化")
    ap.add_argument("--mpc-smooth", type=float, default=0.0,
                    help="MPC 的「平滑安全余量」权重 (cost-to-go 蒸馏建议 10; "
                         "让代价随离弹距离连续变化, 打破'并列最优')")
    ap.add_argument("--mpc-smooth-sigma", type=float, default=20.0,
                    help="平滑余量的衰减尺度 (px)")
    ap.add_argument("--rank-ties", action="store_true",
                    help="用**并列感知的成对排序损失**蒸馏 cost-to-go (代价差 < margin 的动作对"
                         "视为并列、不产生梯度); 比 --cost-to-go 回归更抗并列")
    ap.add_argument("--rank-margin", type=float, default=20.0,
                    help="排序损失: 多大的代价差才算真的有差别 (代价单位)")
    ap.add_argument("--listmle", action="store_true",
                    help="用 **tie-aware ListMLE** 蒸馏 (按并列组连乘的 listwise 排序, "
                         "比 --rank-ties 多用了组外的完整次序; 建议配 --lr 3e-4~1e-3)")
    ap.add_argument("--rank-candidates", type=int, default=0,
                    help="ListMLE 每个状态只随机取 K 个候选动作 (必含 argmin); 0=全部")
    ap.add_argument("--position-reg", action="store_true",
                    help="蒸馏时加**位置正则**: 贴顶/贴底/贴边时压住'继续往外走'的动作 "
                         "(防止上穿之后一直贴顶)")
    ap.add_argument("--position-reg-weight", type=float, default=2.0,
                    help="位置正则的权重")
    ap.add_argument("--position-reg-zone", type=float, default=70.0,
                    help="进入'贴边区'的深度阈值 (px)")
    ap.add_argument("--position-reg-pull", type=float, default=0.8,
                    help="'回中场'吸引力: 0=只罚继续往外(实测贴顶 21%%, ring_gate 78%%); "
                         "0.8=同时奖励往回走, 才是'允许上穿但不常驻天花板'")
    ap.add_argument("--mpc-commit-frames", type=int, default=0,
                    help="MPC 承诺期 (游戏帧, 8~15); 0=不用。承诺期结束后重新规划")
    ap.add_argument("--mpc-top-w", type=float, default=0.0,
                    help="MPC 的「持续贴顶」惩罚权重 (>0 才生效; 建议 1~3)")
    ap.add_argument("--mpc-top-margin", type=float, default=42.0,
                    help="顶部安全区厚度 (px), 自机 y 小于它算贴顶")
    ap.add_argument("--mpc-bottom-w", type=float, default=0.0,
                    help="MPC 的「持续贴底」惩罚权重")
    ap.add_argument("--mpc-edge-w", type=float, default=0.0,
                    help="MPC 的「持续贴左右边」惩罚权重")
    ap.add_argument("--cost-to-go", action="store_true",
                    help="用 **cost-to-go 回归**蒸馏 (学生输出层回归 -相对代价/T, "
                         "自己 argmax); 与 --soft-labels 相比梯度不饱和")
    ap.add_argument("--cost-clip", type=float, default=8.0,
                    help="cost-to-go 回归时相对代价的截断上限 (除以温度后=logits 下限)")
    ap.add_argument("--planner-horizon", type=int, default=180,
                    help="viability 老师: 规划视野(帧)")
    ap.add_argument("--hier-replan-every", type=int, default=1,
                    help="hier 老师: 墙段内沿用上次规划方向的步数 (3~4 可把训练提速 3~4 倍)")
    ap.add_argument("--kl-base", type=float, default=0.0,
                    help="对无标签步的 KL 锚系数 (分层老师必开: 0 时通才会被打崩)")
    ap.add_argument("--pattern-cycle", default="",
                    help="每局轮换的弹型, 逗号分隔 (如 wall_gate,closing_ring); 空=用 --danmaku-pattern")
    ap.add_argument("--planner-dt", type=int, default=2,
                    help="viability 老师: 每层帧数 (2 => 4.4px 网格)")
    ap.add_argument("--hybrid-wall-horizon", type=float, default=0.0,
                    help="hybrid 老师: 直线弹幕墙的前瞻帧数 (0=关; 60 可让 wall_gate 翻倍,"
                         " 但会牺牲普通弹型的存活)")
    ap.add_argument("--danger-weight", type=float, default=0.0,
                    help="危险状态加权: 离最近弹幕越近, 模仿损失权重越大 (0=不加权; 建议 3)")
    ap.add_argument("--teacher-epochs", type=int, default=8,
                    help="模仿学习每个 batch 重复几次梯度 (越大收敛越快)")
    ap.add_argument("--ppo-minibatch", type=int, default=512)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--target-kl", type=float, default=0.05)
    ap.add_argument("--kl-anchor", type=float, default=0.0,
                    help="RL 阶段对'监督末值'的 KL 锚 (0=关闭)。PPO 微调常把 DAgger 的"
                         "躲避能力打崩, 1.0 能让它只在自己附近改进")
    ap.add_argument("--start-stage", type=int, default=1)
    ap.add_argument("--reward-profile", default="", choices=[""] + sorted(REWARD_PROFILES),
                    help="用具名奖励 profile 覆盖 config.json 的 reward 段 "
                         "(attack = 攻击意识: 分数有分量、危险度降为微调、存活惩罚不变)")
    ap.add_argument("--score-scale", type=float, default=None,
                    help="临时覆盖 reward.score_scale (不填则用 profile / config 的值)")
    ap.add_argument("--progress-scale", type=float, default=None,
                    help="临时覆盖 reward.progress_scale。**改奖励时一次只动这一个旋钮**: "
                         "multi 档里 progress(6, 整条血仅 6 分) 与 alive(0.3×600步≈180) 差 30 倍, "
                         "是 PPO 每轮都苟化的根因")
    ap.add_argument("--snapshot-every", type=int, default=0,
                    help="每 N 局存一份快照到 <out>/snapshots/policy_NNNN.npz (+meta.json)。"
                         "0=关。用它在训练后按 6 弹型全评测选最优, 而不是靠 mean_ret 撞运气")
    ap.add_argument("--no-focus", dest="focus", action="store_false",
                    help="不加低速(focus)动作 (默认加)")
    ap.add_argument("--no-invincible", dest="invincible", action="store_false",
                    help="观测不含无敌帧特征 (默认含)")
    ap.add_argument("--gap", dest="gap", action="store_true",
                    help="用「未来占用栅格 + 几何缺口」观测 (整场 32 车道, 替代最近 K 颗弹)")
    ap.add_argument("--gap-layers", type=int, default=1, help="占用栅格的 TTA 层数")
    ap.add_argument("--gap-label", default="reach", choices=["now", "reach", "future"],
                    help="缺口老师的标签口径")
    ap.add_argument("--wall-feat", action="store_true",
                    help="观测末尾追加 12 维墙特征 (需要 --expand-input 才能续训旧策略)")
    ap.add_argument("--expand-input", action="store_true",
                    help="载入 checkpoint 时对新增输入列做零初始化扩维")
    ap.add_argument("--danger-grad", action="store_true",
                    help="观测加入危险梯度(最近弹幕斥力合成), 让反应式躲避更易被表示")
    ap.add_argument("--grid", action="store_true",
                    help="观测改用危险图(以自机为中心的弹幕位置栅格)而不是最近K颗弹的一维向量")
    ap.add_argument("--grid-size", type=int, default=16, help="危险图边长(格数, 默认 16)")
    ap.add_argument("--grid-span", type=float, default=110.0,
                    help="危险图半径(像素, 默认 110): 覆盖自机周围 ±span")
    ap.add_argument("--grid-scales", default="",
                    help="多尺度危险图 'size:span,...' 如 '12:90,12:330' (近场细+远场粗); 非空时忽略 --grid-size/--grid-span")
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--shaping-scale", type=float, default=1.0,
                    help="奖励整体缩放 (不改相对权重; PPO 数值稳定性用, 如 0.01)")
    ap.add_argument("--eval-every", type=int, default=400)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--out", default=".runtime/mock_full")
    ap.add_argument("--load", default="")
    ap.set_defaults(focus=True, invincible=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.txt"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(msg + "\n")

    cfg = Config.load()
    cfg.patch({"backend": "mock", "game": "th11", "step_frames": args.step_frames,
               "start_stage": args.start_stage,
               "danmaku_script": args.danmaku_script,
               "danmaku_offset": args.danmaku_offset,
               "danmaku_pattern": args.danmaku_pattern})
    if args.shaping_scale != 1.0:
        cfg.patch({"reward": {"shaping_scale": args.shaping_scale}})
    if args.reward_profile:
        cfg.patch({"reward": reward_profile(args.reward_profile)})
        log(f"[full] reward profile = {args.reward_profile}")
    if args.score_scale is not None:
        cfg.patch({"reward": {"score_scale": args.score_scale}})
    if args.progress_scale is not None:
        cfg.patch({"reward": {"progress_scale": args.progress_scale}})
        log(f"[full] override progress_scale = {args.progress_scale}")
    reward = dict(cfg.section("reward"))
    env = TouhouEnv(cfg, registry=GameRegistry())
    env.start()

    def make_config(**extra) -> TrainConfig:
        base = dict(
            max_steps=args.max_steps, step_frames=args.step_frames,
            hidden=tuple(int(x) for x in args.hidden.split(",")),
            gamma=args.gamma,
            k_bullets=args.k_bullets, bullet_velocity=True, k_enemies=args.k_enemies,
            use_focus=args.focus, use_invincible=args.invincible,
            use_danger_grad=args.danger_grad,
            use_gap=args.gap, gap_layers=args.gap_layers, gap_label=args.gap_label,
            use_wall_feat=args.wall_feat,
            expand_input=args.expand_input,
            use_grid=args.grid, grid_size=args.grid_size, grid_span=args.grid_span,
            grid_scales=args.grid_scales,
            algorithm=args.algorithm, start_stage=args.start_stage,
            ppo_minibatch=args.ppo_minibatch, ppo_epochs=args.ppo_epochs,
            value_coef=args.value_coef, target_kl=args.target_kl,
            kl_anchor=args.kl_anchor,
            teacher=args.teacher, dagger_beta=args.dagger_beta,
            danger_weight=args.danger_weight,
            teacher_epochs=args.teacher_epochs,
            score_safety_margin=args.score_safety_margin,
            score_align_tol=args.score_align_tol,
            score_commit_margin=args.score_commit_margin,
            hybrid_prior_weight=args.hybrid_prior_weight,
            hybrid_barrier_weight=args.hybrid_barrier_weight,
            hybrid_align_horizon=args.hybrid_align_horizon,
            hybrid_wall_horizon=args.hybrid_wall_horizon,
            mpc_horizon=args.mpc_horizon, mpc_dt=args.mpc_dt,
            mpc_samples=args.mpc_samples, mpc_iters=args.mpc_iters,
            mpc_gap=args.mpc_gap, mpc_column=args.mpc_column,
            mpc_adaptive=args.mpc_adaptive, mpc_plan_every=args.mpc_plan_every,
            mpc_commit=args.mpc_commit, mpc_switch=args.mpc_switch,
            mpc_focus_margin=args.mpc_focus_margin,
            mpc_smooth=args.mpc_smooth, mpc_smooth_sigma=args.mpc_smooth_sigma,
            mpc_deterministic=args.mpc_deterministic,
            soft_labels=args.soft_labels, soft_temp=args.soft_temp,
            cost_to_go=args.cost_to_go, cost_clip=args.cost_clip,
            rank_ties=args.rank_ties, rank_margin=args.rank_margin,
            listmle=args.listmle, rank_candidates=args.rank_candidates,
            position_reg=args.position_reg, position_reg_weight=args.position_reg_weight,
            position_reg_zone=args.position_reg_zone,
            position_reg_pull=args.position_reg_pull,
            mpc_commit_frames=args.mpc_commit_frames,
            mpc_top_w=args.mpc_top_w, mpc_top_margin=args.mpc_top_margin,
            mpc_bottom_w=args.mpc_bottom_w, mpc_edge_w=args.mpc_edge_w,
            planner_horizon=args.planner_horizon,
            planner_dt=args.planner_dt,
            hier_replan_every=args.hier_replan_every,
            kl_base=args.kl_base,
            pattern_cycle=[x.strip() for x in args.pattern_cycle.split(",") if x.strip()],
            script_offset_range=[float(x) for x in args.script_offset_range.split(",") if x.strip()],
            difficulty=args.difficulty, seed=20261018,
        )
        base.update(extra)
        return TrainConfig(**base)

    tc = make_config(episodes=10 ** 9, batch_episodes=args.batch, lr=args.lr,
                     entropy_coef=args.entropy, capture_every=0, eval_every=0,
                     save_every=0, out_dir=str(out))
    trainer = Trainer(env, tc, env_lock=env.lock, out_dir=out)
    if args.load:
        log(f"[full] 载入 {trainer.load(args.load)}")

    eval_data = cfg.as_dict()
    eval_data["danmaku_script"] = args.eval_danmaku_script
    eval_cfg = Config(eval_data)
    eval_env = TouhouEnv(eval_cfg, registry=GameRegistry())
    eval_env.start()
    eval_tc = make_config(episodes=0, seed=999983)
    eval_trainer = Trainer(eval_env, eval_tc, env_lock=eval_env.lock)

    def greedy_eval(count: int):
        trainer.save(out / "policy_snapshot")
        eval_trainer.load(out / "policy_snapshot")
        clears = best_stage = 0
        rets = []
        steps = []
        damages = []
        aligns = []
        scores = []
        tops = []
        min_ys = []
        for k in range(count):
            rec, _ = eval_trainer._run_episode(k, greedy=True)
            rets.append(rec.return_)
            steps.append(rec.steps)
            damages.append(rec.boss_damage)
            aligns.append(rec.align_ratio or 0.0)
            scores.append(rec.score_gain or 0)
            # 行为诊断: 上穿是合法手段, 常驻天花板是退化 (见 PLANNER_V2 §13)
            tops.append(getattr(rec, "top_ratio", 0.0) or 0.0)
            if getattr(rec, "min_y", None) is not None:
                min_ys.append(float(rec.min_y))
            best_stage = max(best_stage, rec.max_stage or 0)
            if rec.cleared:
                clears += 1
        n = max(1, len(rets))
        return (clears, best_stage, sum(rets) / n, sum(steps) / n,
                sum(damages) / n, sum(aligns) / n, sum(scores) / n,
                sum(tops) / n, min(min_ys) if min_ys else -1.0)

    log(f"[full] reward={json.dumps(reward, ensure_ascii=False)}")
    log(f"[full] algo={tc.algorithm} start_stage={tc.start_stage} focus={tc.use_focus} "
        f"invincible={tc.use_invincible} hidden={tc.hidden} lr={tc.lr} batch={tc.batch_episodes} "
        f"k_enemies={tc.k_enemies} obs_dim={trainer.obs_dim} actions={len(trainer.action_table)}")
    trainer.start()
    best = (-1, -1, -1e9)
    last_eval = last_log = 0
    last_snap = 0
    first_clear = None
    while len(trainer.episodes) < args.episodes and trainer.running:
        time.sleep(0.5)
        n = len(trainer.episodes)
        if n > 0 and n % 50 == 0:
            frac = min(1.0, n / max(1, args.episodes))
            try:
                trainer.reconfigure({
                    "entropy_coef": max(args.min_entropy, args.entropy * (1 - frac) + args.min_entropy * frac),
                    "dagger_beta": max(args.dagger_beta_min,
                                       args.dagger_beta * (1 - frac) + args.dagger_beta_min * frac),
                })
            except Exception:
                pass
        if args.snapshot_every and n - last_snap >= args.snapshot_every and n > 0:
            last_snap = n
            try:
                snap_dir = out / "snapshots"
                trainer.save(snap_dir / f"policy_{n:04d}")
                recent = trainer.episodes[-50:]
                meta = {
                    "episode": n,
                    "steps_mean50": round(sum(e.steps for e in recent) / max(1, len(recent)), 1),
                    "damage_mean50": round(
                        sum(e.boss_damage for e in recent) / max(1, len(recent)), 1),
                    "misses_mean50": round(
                        sum(e.misses for e in recent) / max(1, len(recent)), 2),
                    "align_mean50": round(
                        sum((e.align_ratio or 0.0) for e in recent) / max(1, len(recent)), 3),
                    "top_mean50": round(
                        sum((getattr(e, "top_ratio", 0.0) or 0.0) for e in recent)
                        / max(1, len(recent)), 3),
                }
                (snap_dir / f"policy_{n:04d}.json").write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                log(f"[full] 快照 @ep{n}: 存活{meta['steps_mean50']:.0f} "
                    f"伤害{meta['damage_mean50']:.0f} 对准{meta['align_mean50']:.0%} "
                    f"-> {snap_dir / ('policy_%04d.npz' % n)}")
            except Exception as exc:  # 快照失败不该中断训练
                log(f"[full] !!! 快照失败 @ep{n}: {exc}")
        if n - last_eval >= args.eval_every and n > 0:
            last_eval = n
            (clears, stage, mean_ret, mean_steps, mean_dmg, mean_align, mean_score,
             mean_top, min_y) = greedy_eval(args.eval_episodes)
            attack = (f"均伤害 {mean_dmg:6.0f} 对准率 {mean_align:.0%} 均净分 {mean_score:6.0f}"
                      f" 贴顶 {mean_top:.0%} min_y {min_y:5.0f}")
            if args.danmaku_script:
                # 脚本模式 = 弹幕生存: 面数不可靠, 用"存活步数"当指标
                key = (round(mean_steps, 1), round(mean_ret, 1))
                label = f"存活步数 {mean_steps:7.0f} 均回报 {mean_ret:8.1f} {attack}"
            else:
                key = (clears, stage, round(mean_ret, 1))
                label = (f"通关 {clears}/{args.eval_episodes} 最远 {stage} 面 "
                         f"均回报 {mean_ret:.1f} {attack}")
            if key > best:
                best = key
                try:
                    trainer.save(out / "policy_best")
                except Exception:
                    pass
                log(f"[full] *** 新最佳(贪心): {label} @ep{n} (已存 policy_best)")
            else:
                log(f"[full] 评估: {label} @ep{n}")
            if not args.danmaku_script and first_clear is None and clears > 0:
                first_clear = n
                log(f"[full] === 首次完整通关 (贪心) @ep{n} ===")
        if n - last_log >= args.log_every and n > 0:
            last_log = n
            recent = trainer.episodes[-50:]
            m = max(1, len(recent))
            extra = ""
            stats = getattr(trainer, "_last_update", None) or {}
            if "listmle_loss" in stats:
                extra = (f" ListMLE[acc={stats.get('rank_acc', 0):.0%} "
                         f"组数={stats.get('groups', 0):.1f} "
                         f"loss={stats.get('listmle_loss', 0):.3f}]")
            elif "rank_acc" in stats:
                extra = (f" 排序[acc={stats.get('rank_acc', 0):.0%} "
                         f"good={stats.get('good_mean', 0)} "
                         f"loss={stats.get('rank_loss', 0):.3f}]")
            elif "agree" in stats:
                extra = (f" 蒸馏[cost_mae={stats.get('cost_mae', 0):.3f} "
                         f"argmin一致={stats.get('agree', 0):.0%}]")
            elif "imitation_loss" in stats:
                extra = f" 模仿[loss={stats.get('imitation_loss', 0):.3f}]"
            elif "policy_loss" in stats:
                # PPO 诊断: kl 涨得太快 / clip_frac 接近 1 = 更新过猛, 策略会崩
                extra = (f" PPO[kl={stats.get('kl', 0):.4f} "
                         f"clip={stats.get('clip_frac', 0):.0%} "
                         f"ent={stats.get('entropy', 0):.3f} "
                         f"vloss={stats.get('value_loss', 0):.2f}]")
            log(f"[full] ep={n:5d} 回报(近50均)={sum(e.return_ for e in recent)/m:8.1f} "
                f"存活步数(近50均)={sum(e.steps for e in recent)/m:7.0f} "
                f"掉残(均)={sum(e.misses for e in recent)/m:.1f} "
                f"伤害(均)={sum(e.boss_damage for e in recent)/m:6.0f} "
                f"对准率={sum((e.align_ratio or 0.0) for e in recent)/m:.0%} "
                f"贴顶={sum((getattr(e, 'top_ratio', 0.0) or 0.0) for e in recent)/m:.0%}" + extra)

    trainer.stop(save=True)
    save_png(trainer.progress_image(), out, name="progress.png")
    if trainer.error:
        # 训练线程里的异常默认只写进 status; 不打印出来就会看到"0 局"却不知道为什么。
        log(f"[full] !!! 训练异常: {trainer.error}")
        return 1
    log(f"[full] 结束: {len(trainer.episodes)} 局 | 最佳(贪心)={best} | 首次通关={first_clear}")
    log(f"[full] 最佳策略: {out / 'policy_best.npz'}")
    eval_env.close()
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
