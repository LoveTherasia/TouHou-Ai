"""本地训练可视化服务 (标准库 http.server, 零新依赖)。

接口
----
GET /                     单页应用
GET /static/<file>        前端资源
GET /api/runs             所有 run 的列表 (按修改时间倒序)
GET /api/run?id=&since=   单个 run: 元数据 + 增量 episode + 指标名 + 动作直方图
GET /api/asset?id=&name=  读取 run 目录下的产物 (png / json / txt)
GET /api/health           存活探测

设计要点
--------
* ``since`` 让前端只取新增的 episode, 训练进行中也能每 2 秒刷新而不重传历史。
* 所有路径都校验在 run 目录内, 只放行白名单后缀, 避免被当成任意文件读取器。
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .registry import discover_runs, load_run

STATIC_DIR = Path(__file__).resolve().parent / "static"
ASSET_SUFFIXES = {".png", ".jpg", ".json", ".txt", ".jsonl", ".csv", ".gif"}


class DashboardState:
    """持有扫描根目录, 按需刷新 run 索引。"""

    def __init__(self, roots: list[str], *, max_depth: int = 3) -> None:
        self.roots = roots
        self.max_depth = max_depth
        self._runs: dict[str, Any] = {}
        self._stamp = 0.0

    def refresh(self) -> dict[str, Any]:
        runs = discover_runs(self.roots, max_depth=self.max_depth)
        self._runs = {run.id: run for run in runs}
        self._stamp = max((r.mtime for r in runs), default=0.0)
        return self._runs

    @property
    def stamp(self) -> float:
        return self._stamp

    def get(self, run_id: str):
        if not self._runs:
            self.refresh()
        return self._runs.get(run_id)


def _content_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def make_handler(state: DashboardState, *, verbose: bool = False):
    class Handler(BaseHTTPRequestHandler):
        server_version = "TouhouDashboard/1.0"
        protocol_version = "HTTP/1.1"

        # -- 输出 --------------------------------------------------------
        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(body, "application/json; charset=utf-8", status)

        def _error(self, status: int, message: str) -> None:
            self._json({"error": message}, status)

        # -- 路由 --------------------------------------------------------
        def do_HEAD(self) -> None:  # noqa: N802
            self.do_GET()

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)
            try:
                if path in ("/", "/index.html"):
                    return self._serve_static("index.html")
                if path.startswith("/static/"):
                    return self._serve_static(path[len("/static/"):])
                if path == "/api/health":
                    return self._json({"ok": True, "runs": len(state._runs)})
                if path == "/api/runs":
                    return self._api_runs()
                if path == "/api/run":
                    return self._api_run(query)
                if path == "/api/asset":
                    return self._api_asset(query)
                return self._error(404, "not found")
            except BrokenPipeError:      # 浏览器提前断开 (轮询时常见)
                return
            except Exception as exc:     # 兜底: 单次请求失败不该拖垮服务
                return self._error(500, f"{type(exc).__name__}: {exc}")

        # -- 具体接口 ----------------------------------------------------
        def _serve_static(self, relative: str) -> None:
            target = (STATIC_DIR / relative).resolve()
            if not str(target).startswith(str(STATIC_DIR)) or not target.is_file():
                return self._error(404, "static not found")
            self._send(target.read_bytes(), _content_type(target))

        def _api_runs(self) -> None:
            runs = state.refresh()
            payload = [run.to_dict() for run in runs.values()]
            payload.sort(key=lambda r: r["mtime"], reverse=True)
            self._json({"runs": payload})

        def _api_run(self, query: dict[str, list[str]]) -> None:
            run_id = (query.get("id") or [""])[0]
            since = int((query.get("since") or ["0"])[0] or 0)
            run = state.get(run_id)
            if run is None:
                state.refresh()
                run = state.get(run_id)
            if run is None:
                return self._error(404, f"unknown run: {run_id}")
            payload = load_run(run.path, since=max(0, since))
            payload["run"] = run.to_dict()
            payload["since"] = max(0, since)
            self._json(payload)

        def _api_asset(self, query: dict[str, list[str]]) -> None:
            run_id = (query.get("id") or [""])[0]
            name = (query.get("name") or [""])[0]
            run = state.get(run_id)
            if run is None or not name:
                return self._error(404, "unknown run or asset")
            target = (Path(run.path) / name).resolve()
            try:
                target.relative_to(Path(run.path).resolve())
            except ValueError:
                return self._error(403, "outside run directory")
            if target.suffix.lower() not in ASSET_SUFFIXES or not target.is_file():
                return self._error(404, "asset not found")
            self._send(target.read_bytes(), _content_type(target))

        # -- 日志 --------------------------------------------------------
        def log_message(self, fmt: str, *args: Any) -> None:
            if verbose:
                super().log_message(fmt, *args)

    return Handler


def serve(roots: list[str], *, host: str = "127.0.0.1", port: int = 8770,
          max_depth: int = 3, open_browser: bool = False,
          verbose: bool = False) -> None:
    state = DashboardState(roots, max_depth=max_depth)
    runs = state.refresh()
    handler = make_handler(state, verbose=verbose)
    httpd = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"
    print(f"训练看板: {url}")
    print(f"扫描根目录: {', '.join(str(Path(r).resolve()) for r in roots)}")
    print(f"发现 {len(runs)} 个 run")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="本地训练可视化看板")
    ap.add_argument("--root", action="append", default=None,
                    help="扫描根目录 (可重复); 默认 .runtime")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--open", action="store_true", help="启动后打开浏览器")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    roots = args.root or [".runtime"]
    serve(roots, host=args.host, port=args.port, max_depth=args.max_depth,
          open_browser=args.open, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
