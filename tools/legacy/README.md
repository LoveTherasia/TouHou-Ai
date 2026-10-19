# tools/legacy —— 已归档工具

这里的脚本**不再维护**，保留是为了让历史实验可追溯。日常请使用 `tools/` 下的活跃脚本，
或 MCP 里对应的 `th_*` 工具。

| 归档脚本 | 被什么取代 |
|---|---|
| `mock_clear.py` `mock_train_clear.py` | `tools/mock_full_clear.py`、`examples/train.py`；这两个脚本还各自带了一份 `REWARD`，与 `config.json` 冲突，已废弃 |
| `demo_train.py` | `examples/train.py` + `tools/mock_demo.py` |
| `find_bullets.py` `find_bullets2.py` `find_bullet_list.py` `find_bullet_bp.py` | `touhou_rl_mcp/memory/entities.py`（带版本签名校验的固定布局，不再需要现场扫描） |
| `find_player.py` `find_player_iter.py` `find_player_dialogue.py` `find_pointer_chain.py` `verify_xy.py` `calib_player.py` | `tools/calibrate_player.py`、`tools/probe.py find-coord`、MCP 的 `th_memory_find_player` |
| `find_lives.py` `find_items.py` `probe_item_list.py` | 一次性标定流程；`lives` 已写入 `games/offsets/th11.json` |
| `ce_find_x.py` | 依赖 Cheat Engine；本项目改为纯 ctypes + `th_memory_*`，不再使用 CE |
| `_diag_respawn.py` `_sendkeys.py` | 临时调试脚本 |

注意：这些文件已移入子目录，`sys.path` 已同步修正（多了一层 `parent`），但脚本内部
硬编码的 `.runtime/...` 路径假设可能已过期。仅作参考，不建议直接运行。
