"""可启动的实验清单 (launch recipes).

为什么不是"随便填命令行"
------------------------
看板上的启动入口如果能接受任意命令, 它就变成了一个本地 RCE 面板。这里只暴露
**白名单食谱**: 每个食谱固定脚本 + 固定的参数表 (类型/范围/枚举), 前端只能填这些
字段, 后端逐项校验后再拼 argv。想加新实验就往这里加一条食谱, 而不是开放命令行。

参数用 ``--key value`` 的短横线形式 (``bc_episodes`` -> ``--bc-episodes``), 与
argparse 的定义一一对应; ``kind == "bool"`` 的参数只在为真时追加 switch。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: 仓库根目录 (dashboard/recipes.py -> touhou_rl_mcp/dashboard -> touhou_rl_mcp -> 根)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Param:
    """一个可由前端填写的参数。"""

    key: str
    label: str
    kind: str = "float"          # int | float | bool | choice
    default: Any = 0
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()
    help: str = ""

    @property
    def flag(self) -> str:
        return "--" + self.key.replace("_", "-")

    def spec(self) -> dict[str, Any]:
        return {
            "key": self.key, "label": self.label, "kind": self.kind,
            "default": self.default, "minimum": self.minimum,
            "maximum": self.maximum, "choices": list(self.choices),
            "help": self.help,
        }

    def coerce(self, raw: Any) -> Any:
        """把前端传来的值转成合法值; 不合法就抛 ValueError (由接口转成 400)。"""
        if self.kind == "bool":
            if isinstance(raw, str):
                return raw.lower() in ("1", "true", "on", "yes")
            return bool(raw)
        if self.kind == "choice":
            value = str(raw)
            if value not in self.choices:
                raise ValueError(f"{self.key}: 只接受 {list(self.choices)}, 收到 {value!r}")
            return value
        if raw is None or raw == "":
            raw = self.default
        try:
            value = int(raw) if self.kind == "int" else float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{self.key}: 需要 {self.kind}, 收到 {raw!r}") from exc
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"{self.key}: 不能小于 {self.minimum}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"{self.key}: 不能大于 {self.maximum}")
        return value

    def argv(self, raw: Any) -> list[str]:
        value = self.coerce(raw)
        if self.kind == "bool":
            return [self.flag] if value else []
        if self.kind == "choice":
            return [self.flag, str(value)]
        # int 用整数写法, 避免 "--updates 25.0" 这种让 argparse 困惑的输入
        if self.kind == "int":
            return [self.flag, str(int(value))]
        return [self.flag, repr(float(value))]


@dataclass(frozen=True)
class Recipe:
    id: str
    name: str
    description: str
    script: str
    params: tuple[Param, ...] = ()
    #: 不暴露给前端、但每次都要带上的参数
    fixed: tuple[str, ...] = ()
    #: 这些参数的值会拼进 run 目录名 (一眼看出是哪组对照)
    name_params: tuple[str, ...] = ()

    def spec(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "description": self.description,
            "script": self.script, "params": [p.spec() for p in self.params],
            "fixed": list(self.fixed),
        }

    def argv(self, params: dict[str, Any], *, run_dir: Path) -> list[str]:
        known = {p.key for p in self.params}
        unknown = set(params) - known
        if unknown:
            raise ValueError(f"未知参数: {sorted(unknown)}")
        argv = list(self.fixed)
        for param in self.params:
            raw = params.get(param.key, param.default)
            argv.extend(param.argv(raw))
        # run 目录由看板指定, 才能立刻把日志与曲线关联起来
        argv.extend(["--run-dir", str(run_dir)])
        return argv


_GAP_COMMON: tuple[Param, ...] = (
    Param("arm", "观测", "choice", "geo", choices=("geo", "grid", "both"),
          help="geo=几何缺口剖面, grid=时间分层占用栅格, both=拼接"),
    Param("bench", "基准", "choice", "v2", choices=("v1", "v2", "v3"),
          help="v1=初版缺口墙; v2=更窄更快、漂移更大; "
               "v3=规则基线也打不满 (缺口跟随约 1800 帧)"),
    Param("label", "弱监督标签", "choice", "reach",
          choices=("now", "reach", "future", "slack"),
          help="now=当前最空的一列; reach=考虑能否及时赶到; future=沿方向推 4 步; "
               "slack=两堵墙的 maximin 余量"),
    Param("seed", "随机种子", "int", 7, 0, 999999),
    Param("max_frames", "单局上限 (帧)", "int", 2400, 300, 30000),
    Param("eval_every", "每 N 次更新评估", "int", 5, 1, 200),
    Param("baseline", "顺带跑 noop / 缺口跟随基准", "bool", True),
)

RECIPES: dict[str, Recipe] = {
    r.id: r for r in (
        Recipe(
            id="gap_bc_rl",
            name="缺口学习 · 弱监督 + RL",
            description=("先用几何缺口标签做感知预训练, 再用带 BC 锚的 REINFORCE 微调。"
                         "锚定系数为 0 时实测会把感知冲掉 (1136 -> 926 帧)。"),
            script="tools/research/gap_rl.py",
            name_params=("bench", "label", "arm"),
            params=_GAP_COMMON + (
                Param("updates", "RL 更新次数", "int", 25, 0, 500),
                Param("bc_episodes", "预训练采样局数", "int", 30, 0, 500),
                Param("bc_epochs", "预训练轮数", "int", 8, 1, 60),
                Param("bc_coef", "BC 锚强度", "float", 2.0, 0.0, 50.0),
                Param("shaping", "势函数塑形系数", "float", 2.0, 0.0, 50.0),
                Param("dagger", "DAgger 轮数", "int", 6, 0, 50,
                      help="0=纯离线 BC; >0 用策略自己的轨迹补标签再重训"),
            ),
        ),
        Recipe(
            id="gap_rl",
            name="缺口学习 · 纯 RL",
            description="不做任何监督, 直接从回报里学 —— 用来确认动作分布是否坍缩成原地。",
            script="tools/research/gap_rl.py",
            name_params=("bench", "label", "arm"),
            params=_GAP_COMMON + (
                Param("updates", "RL 更新次数", "int", 40, 1, 500),
                Param("epsilon", "ε-探索", "float", 0.2, 0.0, 1.0),
                Param("shaping", "势函数塑形系数", "float", 2.0, 0.0, 50.0),
            ),
        ),
        Recipe(
            id="gap_bc",
            name="缺口学习 · 仅弱监督",
            description="只做感知预训练, 不跑 RL —— 看清标签本身能带来多少存活时长。",
            script="tools/research/gap_rl.py",
            name_params=("bench", "label", "arm"),
            params=_GAP_COMMON + (
                Param("updates", "RL 更新次数", "int", 0, 0, 1),
                Param("bc_episodes", "预训练采样局数", "int", 30, 1, 500),
                Param("bc_epochs", "预训练轮数", "int", 8, 1, 60),
                Param("dagger", "DAgger 轮数", "int", 6, 0, 50,
                      help="0=纯离线 BC (一步准确率虚高、闭环会漂); >0 在策略自己的轨迹上补标签"),
            ),
        ),
        Recipe(
            id="gap_trainer",
            name="缺口学习 · 正式训练器 (BC→DAgger→PPO)",
            description=("走 Trainer 的训练链路: GapObservation 观测 + 缺口标签老师 + 可选新奖励。"
                         "分阶段跑 BC / DAgger / PPO, 每段在留出场景上评估并写 eval.jsonl。"),
            script="tools/research/gap_trainer.py",
            name_params=("bench", "label", "arm", "reward"),
            params=(
                Param("arm", "观测", "choice", "geo", choices=("geo", "grid", "both"),
                      help="geo=几何缺口剖面; grid=时间分层占用栅格; both=拼接"),
                Param("bench", "基准", "choice", "v2", choices=("v1", "v2", "v3")),
                Param("label", "弱监督标签", "choice", "slack",
                      choices=("now", "reach", "future", "slack"),
                      help="slack=两堵墙的 maximin 余量; reach=只看第一次威胁"),
                Param("reward", "奖励", "choice", "gap_survive",
                      choices=("gap_survive", "default"),
                      help="gap_survive=过面进度+擦弹-碰撞 (关斥力); default=config.json 原样"),
                Param("gap_margin", "缺口势函数系数", "float", 0.0, 0.0, 20.0,
                      help="0=关; 2~5 给'别站着不动'一个密集梯度"),
                Param("bc_episodes", "BC 局数", "int", 30, 0, 2000),
                Param("dagger_episodes", "DAgger 局数", "int", 60, 0, 2000),
                Param("rl_episodes", "PPO 局数", "int", 120, 0, 5000),
                Param("batch", "每 N 局更新一次", "int", 4, 1, 64),
                Param("eval_every", "每 N 局评估一次", "int", 20, 1, 500),
                Param("max_frames", "单局上限 (帧)", "int", 2400, 300, 30000),
                Param("hidden", "隐层宽度 (0=线性)", "int", 64, 0, 512),
                Param("lr", "学习率", "float", 2e-3, 1e-5, 1e-1),
                Param("seed", "随机种子", "int", 7, 0, 999999),
            ),
        ),
    )
}


def get(recipe_id: str) -> Recipe:
    recipe = RECIPES.get(recipe_id)
    if recipe is None:
        raise ValueError(f"未知实验: {recipe_id!r}")
    return recipe


def _safe_slug(text: str) -> str:
    keep = [ch if (ch.isalnum() or ch in "-_") else "-" for ch in text]
    slug = "".join(keep).strip("-")
    return slug or "run"


def default_run_dir(root: Path, recipe: Recipe, *, stamp: str,
                    params: dict[str, Any] | None = None) -> Path:
    """``<root>/runs/<recipe>[-<关键参数>]-<时间戳>``; 重名时追加序号, 绝不覆盖。"""
    base = Path(root) / "runs"
    bits = [_safe_slug(recipe.id)]
    for key in recipe.name_params:
        value = (params or {}).get(key)
        if isinstance(value, str) and value:
            bits.append(_safe_slug(value))
    name = "-".join(bits + [stamp])
    candidate = base / name
    index = 2
    while candidate.exists():
        candidate = base / f"{name}-{index}"
        index += 1
    return candidate
