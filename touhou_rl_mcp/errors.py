"""项目内统一的异常类型."""

from __future__ import annotations


class TouhouError(Exception):
    """本项目的异常基类."""


class ConfigError(TouhouError):
    """配置不合法."""


class GameNotFoundError(TouhouError):
    """找不到游戏进程 / 可执行文件."""


class BackendError(TouhouError):
    """后端 (内存 / 模拟器) 出错."""


class ProcessMemoryError(BackendError):
    """读写目标进程内存失败."""


class NotInGameError(BackendError):
    """当前不在游戏局内 (例如还停在标题画面)."""


class OffsetUnavailableError(BackendError):
    """所需的偏移量尚未标定 (例如 player_x / player_y / bullets).

    这类错误会附带 ``hint``, 告诉使用者如何用 MCP 自带的内存扫描工具标定.
    """

    def __init__(self, field: str, *, game: str, hint: str = "") -> None:
        self.field = field
        self.game = game
        base = (
            f"游戏 {game} 的偏移 `{field}` 尚未标定, 该功能不可用。\n"
            f"标定方法: 用 `memory_scan` / `memory_filter` 工具扫描内存, "
            f"找到地址后写入 config/offsets 即可。"
        )
        super().__init__(base + (("\n" + hint) if hint else ""))


class NoActiveEpisodeError(BackendError):
    """尚未调用 start_new_game 就开始 step / 查询状态."""
