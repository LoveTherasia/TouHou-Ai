"""可选的 Gymnasium 适配 (想接 Stable-Baselines3 / CleanRL 时用).

gymnasium 不是本项目的依赖; 只有你显式使用这个模块时才需要 ``pip install gymnasium``。
不装也能训练 —— :class:`~touhou_rl_mcp.training.trainer.Trainer` 自带一个
纯 numpy 的 REINFORCE + 基线实现。

用法::

    from touhou_rl_mcp.training.gym_env import make_gym_env
    env = make_gym_env(config)          # gymnasium.Env
    from stable_baselines3 import PPO
    PPO("MlpPolicy", env).learn(200_000)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..env import TouhouEnv
from .features import encode_state, feature_dim
from .trainer import build_action_table

try:  # pragma: no cover
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "需要先安装 gymnasium: pip install gymnasium"
    ) from exc


class TouhouGymEnv(gym.Env):
    """把 :class:`TouhouEnv` 包成 Gymnasium 环境.

    * 动作: ``Discrete(9)`` 或 ``Discrete(18)`` (是否启用低速 focus), 均常按射击。
    * 观测: 定长浮点向量 (见 :func:`~touhou_rl_mcp.training.features.encode_state`)。
      想用画面做输入的话, 把 ``observation_mode="image"`` 打开 (返回 480x640x3 的
      uint8 截图), 再配 ``CNN`` 策略。
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Any = None,
        *,
        use_focus: bool = False,
        k_bullets: int = 16,
        observation_mode: str = "features",
        max_steps: int = 1500,
        step_frames: int = 4,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.touhou = TouhouEnv(config) if config is not None else TouhouEnv()
        self.touhou.start()
        self.action_table = build_action_table(use_focus=use_focus)
        self.observation_mode = observation_mode
        self.k_bullets = k_bullets
        self.max_steps = max_steps
        self.step_frames = step_frames
        self._step_index = 0
        self._rng = np.random.default_rng(seed or 0)

        self.action_space = spaces.Discrete(len(self.action_table))
        if observation_mode == "image":
            self.observation_space = spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(
                -4.0, 4.0, shape=(feature_dim(k_bullets),), dtype=np.float32
            )

    # ------------------------------------------------------------------
    def _obs(self, state, *, terminal: bool = False) -> np.ndarray:
        if self.observation_mode == "image":
            image, _method = self.touhou.screenshot()
            image = image.convert("RGB").resize((640, 480))
            return np.asarray(image, dtype=np.uint8)
        try:
            return encode_state(state, k_bullets=self.k_bullets, validate=not terminal).astype(np.float32)
        except Exception:
            controller = self.touhou.backend.controller
            if controller is not None:
                controller.release_all()
            raise

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        result = self.touhou.reset(**{**(options or {}), "seed": int(seed) if seed is not None else None})
        self._step_index = 0
        return self._obs(result.state), {"stage": result.state.stage}

    def step(self, action: int):
        self._step_index += 1
        result = self.touhou.step(self.action_table[int(action)], frames=self.step_frames)
        terminated = bool(result.done)
        truncated = bool(result.truncated) or self._step_index >= self.max_steps
        info = {
            "stage": result.state.stage,
            "lives": result.state.lives,
            "reward_parts": result.info.get("reward_parts"),
            "done_reason": result.info.get("done_reason"),
        }
        return self._obs(result.state, terminal=terminated), float(result.reward), terminated, truncated, info

    def close(self) -> None:
        self.touhou.close()


def make_gym_env(config: Any = None, **kwargs: Any) -> "TouhouGymEnv":
    return TouhouGymEnv(config, **kwargs)
