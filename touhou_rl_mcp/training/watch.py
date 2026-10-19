"""实时监视窗口 —— 一边训练一边"看着它打".

用 tkinter 显示 :func:`~touhou_rl_mcp.training.visualize.draw_dashboard` 生成的
仪表盘 (左边当前画面, 右边学习曲线, 顶部实时统计), 每 ``refresh_ms`` 毫秒刷新一次。

用法::

    python tools/watch_training.py --backend mock --episodes 2000
    python tools/watch_training.py --backend memory --game th11 --episodes 50

快捷键::

    空格  暂停 / 继续训练
    S     立刻保存策略
    Esc   停止训练并关窗

.. warning::
   用 **真实游戏后端** 时, 游戏窗口必须保持在前台 —— 按键是通过 SendInput 发到
   前台窗口的。点一下这个监视窗口会把焦点抢走, 虽然训练器每步都会把游戏窗口
   重新置前, 但会造成闪烁。所以真机训练时请**别点监视窗口**, 用快捷键操作。
"""

from __future__ import annotations

import threading
from typing import Any

from .visualize import draw_dashboard


def run_watch_window(
    trainer: Any,
    *,
    refresh_ms: int = 700,
    title: str = "Touhou RL 训练监视",
    on_close: Any = None,
) -> None:
    """打开监视窗口 (阻塞, 直到用户关窗或按 Esc)."""
    import tkinter as tk

    from PIL import ImageTk

    root = tk.Tk()
    root.title(title)
    root.configure(bg="#101220")
    root.geometry("1480x580")

    image_label = tk.Label(root, bg="#101220", bd=0)
    image_label.pack(fill="both", expand=True)

    status_var = tk.StringVar(value="准备中...")
    status = tk.Label(
        root,
        textvariable=status_var,
        anchor="w",
        bg="#1a1d2c",
        fg="#c8d2e6",
        font=("Microsoft YaHei UI", 10),
        padx=8,
        pady=4,
    )
    status.pack(fill="x", side="bottom")

    holder: dict[str, Any] = {"photo": None, "closed": False}

    def tick() -> None:
        if holder["closed"]:
            return
        try:
            image = trainer.dashboard_image(title=title)
            holder["photo"] = ImageTk.PhotoImage(image)
            image_label.configure(image=holder["photo"])
            state = trainer.status(tail=0)
            flag = "运行中" if state["running"] else ("已暂停" if state["paused"] else "已停止")
            status_var.set(
                f"空格=暂停/继续   S=保存策略   Esc=停止并退出    │  {flag}"
                f"   局数 {state['episodes_done']}"
                f"   步/秒 {state['steps_per_sec']}"
                f"   近50局回报 {state['mean_return_last50']}"
                f"   最好 {state['best_return']}"
                f"   面数 {state['best_stage']}"
                f"   {('错误: ' + state['error'].splitlines()[0]) if state['error'] else ''}"
            )
        except Exception as exc:  # pragma: no cover
            status_var.set(f"刷新失败: {exc}")
        root.after(refresh_ms, tick)

    def toggle_pause(_event: Any = None) -> None:
        if trainer.paused:
            trainer.resume()
        else:
            trainer.pause()

    def save_now(_event: Any = None) -> None:
        try:
            trainer.save()
            status_var.set(f"策略已保存到 {trainer.policy_path()}.npz")
        except Exception as exc:
            status_var.set(f"保存失败: {exc}")

    def close(_event: Any = None) -> None:
        holder["closed"] = True
        if on_close is not None:
            try:
                on_close()
            except Exception:
                pass
        root.destroy()

    root.bind("<space>", toggle_pause)
    root.bind("s", save_now)
    root.bind("S", save_now)
    root.bind("<Escape>", close)
    root.protocol("WM_DELETE_WINDOW", close)
    root.after(300, tick)
    root.mainloop()


def run_watch_thread(trainer: Any, **kwargs: Any) -> threading.Thread:
    """在后台线程里打开监视窗口 (训练照常在另一个线程跑)."""
    thread = threading.Thread(target=run_watch_window, args=(trainer,), kwargs=kwargs, daemon=True)
    thread.start()
    return thread


__all__ = ["draw_dashboard", "run_watch_thread", "run_watch_window"]
