"""RL 训练: 特征编码 / 策略网络 / 训练器."""

from __future__ import annotations

from .features import BULLET_FEATURE_DIM, GLOBAL_FEATURE_DIM, encode_frames, encode_state, feature_dim
from .policy import MLPPolicy, blend_policies
from .trainer import DIRECTIONS, EpisodeRecord, TrainConfig, Trainer, build_action_table

__all__ = [
    "BULLET_FEATURE_DIM",
    "DIRECTIONS",
    "GLOBAL_FEATURE_DIM",
    "EpisodeRecord",
    "MLPPolicy",
    "blend_policies",
    "TrainConfig",
    "Trainer",
    "build_action_table",
    "encode_frames",
    "encode_state",
    "feature_dim",
]
