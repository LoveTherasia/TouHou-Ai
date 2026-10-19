"""训练可视化平台.

设计约束 (重要):
  * **零新依赖** —— 只用标准库 ``http.server`` + 原生 canvas, 不引入 Flask/FastAPI/CDN。
    平台本身不应该成为项目的部署负担。
  * **schema 跟随已有产物** —— run = 任何含有 ``train.jsonl`` 的目录 (``training.Journal``
    已经在写这个文件)。不新造格式, 也不用改现有训练器就能看到曲线。
  * **指标自动发现** —— 前端从数据里推断可画的数值字段, 所以以后新增训练脚本/新指标
    不需要改前端。

用法::

    python -m touhou_rl_mcp.dashboard --root .runtime --port 8770
"""

from .registry import RunInfo, discover_runs, load_run, read_episodes

__all__ = ["RunInfo", "discover_runs", "load_run", "read_episodes"]
