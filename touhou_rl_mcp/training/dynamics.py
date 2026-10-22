"""轻量动力学模型：预测弹幕未来 10~30 帧的位置.

设计要点
--------
* **残差形式**：``p(t+dt) = p + v·dt + f_θ(p, v, dt)``。匀速直线弹 (mock 的全部弹)
  对应 ``f_θ ≡ 0``，所以模型不会比"匀速外推"更差 —— 它只学"外推错在哪"。
* **极小**：2 层 32 单元，参数 < 2k，numpy 前向一次批量只需几十微秒。MPC 每一步要
  推演几十条候选轨迹、每条几十帧，靠这个量级才跑得动。
* **数据两种来源**：
  1. 弹幕脚本（``spawns`` 的 ``frame,x,y,vx,vy,life``）—— 重建轨迹对；
  2. 真机逐帧槽位轨迹（锁血采集）—— 唯一能学到曲线/加速弹的数据。
     mock 的弹是匀速直线，在这上面训练只能验证"接口与误差上限"。

为什么不用"直接回归绝对位置": 绝对位置的方差主要来自场地分布，模型会把容量花在记
"哪个区域弹多"，而 MPC 关心的是**相对误差**。残差形式天然把匀速部分吃掉。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

FIELD_W = 384.0
FIELD_H = 448.0
VEL_SCALE = 8.0
DT_SCALE = 30.0
#: 特征维度: (x/W, y/H, vx/S, vy/S, ax/A, ay/A, dt/30)
#:
#: **加速度必须进特征** —— 残差 ``0.5·a·dt²`` 的量级完全由加速度决定, 只给
#: ``(x,y,vx,vy)`` 时同一状态下残差可以有任意大 (实测: 加了加速度后 30 帧误差
#: 12.6px → 0.5px, 不加则模型学不动)。
FEATURE_DIM = 7
#: 加速度归一化尺度 (px/帧²)
ACC_SCALE = 0.1


class ConstantVelocity:
    """基线: 匀速直线外推。给动力学模型当"下限"参照。"""

    name = "constant"

    def predict(self, x: float, y: float, vx: float, vy: float, dt: float,
                ax: float = 0.0, ay: float = 0.0) -> tuple[float, float]:
        return x + vx * dt, y + vy * dt

    def predict_batch(self, facts: np.ndarray, dt: float = 0.0) -> np.ndarray:
        """``facts``: ``(N, 6)`` = ``(x, y, vx, vy, ax, ay)``；返回残差（恒 0）。"""
        f = np.asarray(facts, dtype=np.float64)
        return np.zeros((f.shape[0], 2), dtype=np.float64)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name}


class ResidualMLP:
    """残差动力学: ``f_θ(x, y, vx, vy, dt)`` 用两层 tanh MLP，输出像素残差。"""

    name = "residual"

    def __init__(self, hidden: int = 32, seed: int = 0, lr: float = 0.02) -> None:
        rng = np.random.default_rng(seed)
        self.hidden = int(hidden)
        self.lr = float(lr)
        self.W1 = rng.standard_normal((FEATURE_DIM, self.hidden)) * (1.0 / math.sqrt(FEATURE_DIM))
        self.b1 = np.zeros(self.hidden)
        self.W2 = rng.standard_normal((self.hidden, 2)) * (1.0 / math.sqrt(self.hidden))
        self.b2 = np.zeros(2)
        # 动量
        self._mom = {k: np.zeros_like(getattr(self, k)) for k in ("W1", "b1", "W2", "b2")}
        self.train_loss: list[float] = []

    # ------------------------------------------------------------------
    @staticmethod
    def features(x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray,
                 dt: np.ndarray, ax: np.ndarray | None = None,
                 ay: np.ndarray | None = None) -> np.ndarray:
        zero = np.zeros_like(np.asarray(dt, dtype=np.float64))
        ax = zero if ax is None else np.asarray(ax, dtype=np.float64)
        ay = zero if ay is None else np.asarray(ay, dtype=np.float64)
        return np.stack([x / FIELD_W, y / FIELD_H, vx / VEL_SCALE, vy / VEL_SCALE,
                         ax / ACC_SCALE, ay / ACC_SCALE, dt / DT_SCALE], axis=1)

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = np.tanh(X @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2
        return h, out

    def predict(self, x: float, y: float, vx: float, vy: float, dt: float,
                ax: float = 0.0, ay: float = 0.0) -> tuple[float, float]:
        X = self.features(np.array([x]), np.array([y]), np.array([vx]), np.array([vy]),
                          np.array([dt]), np.array([ax]), np.array([ay]))
        _, out = self._forward(X)
        return float(x + vx * dt + out[0, 0]), float(y + vy * dt + out[0, 1])

    def predict_batch(self, facts: np.ndarray, dt: float) -> np.ndarray:
        """``facts``: ``(N, 6)``；返回**残差** ``(N, 2)``（调用方自己加匀速项）。"""
        f = np.asarray(facts, dtype=np.float64)
        if f.shape[1] < 6:
            pad = np.zeros((f.shape[0], 6 - f.shape[1]))
            f = np.concatenate([f, pad], axis=1)
        X = self.features(f[:, 0], f[:, 1], f[:, 2], f[:, 3], np.full(f.shape[0], dt),
                          f[:, 4], f[:, 5])
        _, out = self._forward(X)
        return out

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, Y: np.ndarray, *, epochs: int = 300,
            batch: int = 256, weight_decay: float = 1e-5, lr: float | None = None,
            rng: np.random.Generator | None = None) -> "ResidualMLP":
        """``X``: ``(N,5)`` 原始特征; ``Y``: ``(N,2)`` **残差**目标（不是绝对位置）。"""
        Xr = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Xr.ndim != 2 or Xr.shape[0] == 0:
            raise ValueError("训练集为空")
        # 入口统一转成**归一化特征**: 调用方给的是原始量纲 (x,y,vx,vy,ax,ay,dt),
        # 与推理端的 features() 必须一致 —— 否则训练/推理两套输入, 模型看似收敛
        # 但实际预测完全错位 (踩过一次: MAE 比匀速基线还差)。
        X = self.features(Xr[:, 0], Xr[:, 1], Xr[:, 2], Xr[:, 3], Xr[:, -1],
                          Xr[:, 4] if Xr.shape[1] > 5 else None,
                          Xr[:, 5] if Xr.shape[1] > 6 else None)
        n = X.shape[0]
        rng = rng or np.random.default_rng(0)
        # Adam: 这个模型要拟合的是「0.5·a·dt²」这种乘积项, 纯 SGD 步长稍小就完全
        # 学不动 (实测 MSE 停在残差方差上); Adam 的逐参数步长让它几十轮就收敛。
        lr = float(self.lr if lr is None else lr)
        b1, b2, eps = 0.9, 0.999, 1e-8
        m = {k: np.zeros_like(getattr(self, k)) for k in ("W1", "b1", "W2", "b2")}
        v = {k: np.zeros_like(getattr(self, k)) for k in ("W1", "b1", "W2", "b2")}
        step = 0
        for _ in range(int(epochs)):
            idx = rng.permutation(n)
            for start in range(0, n, batch):
                sel = idx[start:start + batch]
                xb, yb = X[sel], Y[sel]
                h = np.tanh(xb @ self.W1 + self.b1)
                pred = h @ self.W2 + self.b2
                err = pred - yb
                g_out = err * (2.0 / xb.shape[0])
                grads = {"W2": h.T @ g_out + weight_decay * self.W2,
                         "b2": g_out.sum(axis=0)}
                g_h = (g_out @ self.W2.T) * (1.0 - h * h)
                grads["W1"] = xb.T @ g_h + weight_decay * self.W1
                grads["b1"] = g_h.sum(axis=0)
                step += 1
                for k, g in grads.items():
                    m[k] = b1 * m[k] + (1 - b1) * g
                    v[k] = b2 * v[k] + (1 - b2) * g * g
                    mh = m[k] / (1 - b1 ** step)
                    vh = v[k] / (1 - b2 ** step)
                    setattr(self, k, getattr(self, k) - lr * mh / (np.sqrt(vh) + eps))
            pred_all = np.tanh(X @ self.W1 + self.b1) @ self.W2 + self.b2
            self.train_loss.append(float(np.mean((pred_all - Y) ** 2)))
        return self

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "hidden": self.hidden, "params": int(
            self.W1.size + self.b1.size + self.W2.size + self.b2.size)}

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 hidden=np.array([self.hidden]))
        return out

    @classmethod
    def load(cls, path: str | Path) -> "ResidualMLP":
        data = np.load(path)
        model = cls(hidden=int(data["hidden"][0]))
        model.W1, model.b1 = data["W1"], data["b1"]
        model.W2, model.b2 = data["W2"], data["b2"]
        return model


# ----------------------------------------------------------------------
# 数据构造
# ----------------------------------------------------------------------
def tracks_from_script(
    path: str | Path,
) -> list[list[tuple[float, ...]]]:
    """从弹幕脚本重建每条弹的轨迹 ``[(frame, x, y, vx, vy), ...]``。

    脚本是"生成事件 + 匀速"的紧凑表示，这里按 ``life`` 展开成逐帧点
    （``life<=0`` 时按飞出场地估算）。注意：这样得到的轨迹**必然是匀速直线**，
    只能用来验证接口/误差下限，学不到曲线。
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("spawns") or []
    tracks: list[list[tuple[float, ...]]] = []
    for row in rows:
        f, x, y, vx, vy = (float(v) for v in row[:5])
        life = int(row[5]) if len(row) > 5 else 0
        if life <= 0:
            life = 240
        # 每 3 帧采一个点，控制内存
        pts = []
        for k in range(0, min(life, 240), 3):
            pts.append((int(f) + k, x + vx * k, y + vy * k, vx, vy))
        tracks.append(pts)
    return tracks


def pairs_from_tracks(
    tracks: Iterable[Sequence[tuple[float, ...]]],
    horizons: Sequence[int] = (10, 20, 30),
    *,
    stride: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把轨迹对展开成训练集 ``(X, Y_residual, dt)``。"""
    X: list[list[float]] = []
    Y: list[list[float]] = []
    D: list[float] = []
    for track in tracks:
        seq = list(track)
        if len(seq) < 2:
            continue
        for i in range(0, len(seq) - 1, max(1, stride)):
            point = seq[i]
            _, x, y, vx, vy = point[:5]
            ax = float(point[5]) if len(point) > 5 else 0.0
            ay = float(point[6]) if len(point) > 6 else 0.0
            for h in horizons:
                j = i + max(1, h // 3)          # 轨迹按 3 帧采样
                if j >= len(seq):
                    continue
                fx, fy = seq[j][1], seq[j][2]
                dt = float(seq[j][0] - seq[i][0])
                if dt <= 0:
                    continue
                X.append([x, y, vx, vy, ax, ay, dt])
                Y.append([fx - (x + vx * dt), fy - (y + vy * dt)])
                D.append(dt)
    return (np.asarray(X, dtype=np.float64).reshape(-1, FEATURE_DIM),
            np.asarray(Y, dtype=np.float64).reshape(-1, 2),
            np.asarray(D, dtype=np.float64))


def evaluate(model: Any, X: np.ndarray, Y: np.ndarray) -> dict[str, float]:
    """按 dt 分组报平均绝对误差（像素）。``Y`` 是残差目标。"""
    if X.shape[0] == 0:
        return {"n": 0.0, "mae": 0.0}
    dts = X[:, -1]           # 最后一列是 dt (前面还有 ax, ay)
    pred = np.zeros_like(Y)
    if isinstance(model, ResidualMLP):
        # predict_batch 的特征里带 dt，必须按 dt 分组调用
        for dt in np.unique(dts):
            mask = np.isclose(dts, dt)
            pred[mask] = model.predict_batch(X[mask, :6], float(dt))
    err = np.hypot(*(pred - Y).T)
    out = {"n": float(X.shape[0]), "mae": float(err.mean())}
    for dt in sorted(set(np.round(dts, 3))):
        mask = np.isclose(dts, dt)
        if mask.any():
            out[f"mae@{dt:g}"] = float(err[mask].mean())
    return out
