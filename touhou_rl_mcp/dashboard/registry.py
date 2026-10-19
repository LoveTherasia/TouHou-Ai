"""run 发现与读取.

"run" 的定义只有一个: **目录下有 ``train.jsonl``**。这是 :class:`training.journal.Journal`
已经在写的文件, 所以已有的训练产物不需要任何迁移就能被平台看到。

指标不写死: :func:`numeric_metrics` 会从数据里扫出所有数值字段 (含
``reward_breakdown`` 这种嵌套字典), 前端据此动态生成曲线选项。新增指标时
只改训练脚本, 不用动平台。
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

#: 一个目录里出现它就认为是一次训练 run
RUN_MARKER = "train.jsonl"
#: 可选的自描述元数据 (训练脚本可以写, 平台会展示)
META_FILE = "meta.json"
#: 可选的结构化评估结果, 每行 ``{"step": int, "name": str, "score": float, ...}``
EVAL_FILE = "eval.jsonl"

#: 扫描时跳过的目录名
_SKIP = {"__pycache__", "node_modules", ".git", "legacy", "archive"}


@dataclass
class RunInfo:
    id: str                     # 相对 root 的 posix 路径, 作为唯一标识
    name: str
    path: str                   # 绝对路径
    mtime: float
    size: int
    episodes: int
    has_eval: bool = False
    #: 路径里有 ``_`` 开头的段 (_diag / _smoke ...) —— 属于内部临时产物, 默认折叠
    hidden: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "mtime": self.mtime,
            "age_s": max(0.0, time.time() - self.mtime),
            "size": self.size,
            "episodes": self.episodes,
            "has_eval": self.has_eval,
            "hidden": self.hidden,
            "meta": self.meta,
        }


# ----------------------------------------------------------------------
def _iter_run_dirs(root: Path, max_depth: int) -> Iterable[Path]:
    """在 root 下找含 train.jsonl 的目录 (限制深度, 避免扫进 .venv)。"""
    root = root.resolve()
    base_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        depth = len(current.parts) - base_depth
        if depth >= max_depth:
            dirnames[:] = []
        dirnames[:] = [d for d in dirnames if d not in _SKIP and not d.startswith(".")]
        if RUN_MARKER in filenames:
            yield current


def _count_episodes(path: Path) -> int:
    try:
        with open(path, "rb") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return 0


def _read_meta(run_dir: Path) -> dict[str, Any]:
    meta_path = run_dir / META_FILE
    if not meta_path.is_file():
        return {}
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def discover_runs(roots: list[str | Path], *, max_depth: int = 3) -> list[RunInfo]:
    """扫描若干根目录, 返回按修改时间倒序的 run 列表。"""
    runs: list[RunInfo] = []
    seen: set[str] = set()
    for root in roots:
        base = Path(root).expanduser()
        if not base.is_dir():
            continue
        for run_dir in _iter_run_dirs(base, max_depth):
            try:
                key = str(run_dir.resolve())
            except OSError:
                continue
            if key in seen:
                continue
            seen.add(key)
            train_path = run_dir / RUN_MARKER
            try:
                stat = train_path.stat()
            except OSError:
                continue
            try:
                relative = run_dir.resolve().relative_to(base.resolve())
            except ValueError:
                continue
            runs.append(RunInfo(
                id=str(relative).replace("\\", "/"),
                name=run_dir.name,
                path=str(run_dir),
                mtime=stat.st_mtime,
                size=stat.st_size,
                episodes=_count_episodes(train_path),
                has_eval=(run_dir / EVAL_FILE).is_file(),
                hidden=any(part.startswith("_") for part in relative.parts),
                meta=_read_meta(run_dir),
            ))
    runs.sort(key=lambda r: r.mtime, reverse=True)
    return runs


def _parse_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
                if limit is not None and len(rows) >= limit:
                    break
    except OSError:
        return rows
    return rows


def read_episodes(run_dir: str | Path, *, since: int = 0,
                  limit: int | None = None) -> list[dict[str, Any]]:
    """读 ``train.jsonl``, 跳过前 ``since`` 条 (增量轮询用)。"""
    rows = _parse_jsonl(Path(run_dir) / RUN_MARKER)
    tail = rows[max(0, since):]
    return tail[:limit] if limit is not None else tail


def read_eval(run_dir: str | Path) -> list[dict[str, Any]]:
    return _parse_jsonl(Path(run_dir) / EVAL_FILE)


# ----------------------------------------------------------------------
def _flatten(row: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """把一行拍平成 ``{"reward_breakdown.score": 1.2}`` 这样的数值表。"""
    out: dict[str, float] = {}
    for key, value in row.items():
        name = f"{prefix}{key}"
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                continue
            out[name] = float(value)
        elif isinstance(value, dict):
            out.update(_flatten(value, prefix=f"{name}."))
    return out


def numeric_metrics(episodes: list[dict[str, Any]], *, sample: int = 60) -> list[str]:
    """扫出可画的数值字段 (跳过 ``episode`` 这类恒等索引)。"""
    keys: set[str] = set()
    for row in episodes[:sample]:
        keys.update(_flatten(row).keys())
    # 动作直方图 (actions.*) 交给专门的面板, 不当曲线画; episode 是 x 轴本身
    noisy = {"episode"}
    return sorted(k for k in keys
                  if k not in noisy and not k.startswith("actions."))


def action_totals(episodes: list[dict[str, Any]]) -> dict[str, int]:
    """聚合多个 episode 的 ``actions`` 直方图。"""
    totals: dict[str, int] = {}
    for row in episodes:
        actions = row.get("actions")
        if not isinstance(actions, dict):
            continue
        for name, count in actions.items():
            if isinstance(count, (int, float)):
                totals[str(name)] = totals.get(str(name), 0) + int(count)
    return totals


def summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """给列表页/详情页用的粗粒度摘要。"""
    if not episodes:
        return {"count": 0}

    def _vals(key: str) -> list[float]:
        return [float(r[key]) for r in episodes if isinstance(r.get(key), (int, float))]

    out: dict[str, Any] = {"count": len(episodes)}
    for key in ("return", "steps", "frames", "max_stage", "misses", "cleared"):
        values = _vals(key)
        if values:
            out[f"{key}_last"] = round(values[-1], 3)
            out[f"{key}_best"] = round(max(values), 3)
    if out.get("cleared_best"):
        out["clear_rate"] = round(sum(1 for v in _vals("cleared") if v) / len(episodes), 3)
    return out


def _merge_eval(episodes: list[dict[str, Any]],
                eval_rows: list[dict[str, Any]]) -> None:
    """把 ``eval.jsonl`` 按 step 贴回对应 episode, 这样泛化分数也能画成曲线。"""
    episodes_ids = [row.get("episode") for row in episodes]
    numbers = [e for e in episodes_ids if isinstance(e, (int, float))]
    if len(set(numbers)) != len(numbers):
        # 同一 out_dir 被多次训练追加写入时 episode 会重复, 按编号合并会贴错分数
        return
    by_step = {int(row["step"]): row.get("score")
               for row in eval_rows
               if isinstance(row.get("step"), (int, float))}
    if not by_step:
        return
    for row in episodes:
        step = row.get("episode")
        if isinstance(step, (int, float)) and int(step) in by_step:
            score = by_step[int(step)]
            if isinstance(score, (int, float)):
                row["eval"] = float(score)


def load_run(run_dir: str | Path, *, since: int = 0,
             limit: int | None = None) -> dict[str, Any]:
    """一次取齐: 元数据 + 摘要 + 增量 episode + 指标名 + 动作直方图。"""
    run_path = Path(run_dir)
    episodes = read_episodes(run_path, since=since, limit=limit)
    all_rows = read_episodes(run_path) if since else episodes
    eval_rows = read_eval(run_path)
    _merge_eval(episodes, eval_rows)
    if all_rows is not episodes:
        _merge_eval(all_rows, eval_rows)
    return {
        "info": {
            "path": str(run_path),
            "name": run_path.name,
            "meta": _read_meta(run_path),
            "eval": eval_rows[-50:],
        },
        "total": len(all_rows),
        "episodes": episodes,
        "metrics": numeric_metrics(all_rows or episodes),
        "actions": action_totals(all_rows[-40:] or episodes),
        "summary": summarize(episodes),
    }
