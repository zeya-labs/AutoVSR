#!/usr/bin/env python3
"""Serve an EDA block-diagram report with disk-backed graph editing."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


class ReportHandler(SimpleHTTPRequestHandler):
    output_dir: Path

    def translate_path(self, path: str) -> str:
        clean = unquote(path.split("?", 1)[0].split("#", 1)[0]).lstrip("/")
        if not clean:
            clean = "index.html"
        output_target = (self.output_dir / clean).resolve()
        if _is_relative_to(output_target, self.output_dir) and output_target.exists():
            return str(output_target)
        project_target = (PROJECT_ROOT / clean).resolve()
        if _is_relative_to(project_target, PROJECT_ROOT) and project_target.exists():
            return str(project_target)
        return str(output_target if _is_relative_to(output_target, self.output_dir) else self.output_dir / "__forbidden__")

    def do_POST(self) -> None:
        if self.path != "/api/save_graph":
            self._json_response({"ok": False, "error": "unknown endpoint"}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = self._save_graph(payload)
        except Exception as exc:
            self._json_response({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=400)
            return
        self._json_response(result)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _save_graph(self, payload: dict[str, Any]) -> dict[str, Any]:
        case_id = str(payload.get("case_id") or "").strip()
        if not case_id or "/" in case_id or "\\" in case_id:
            raise ValueError("invalid case_id")
        layout = payload.get("layout")
        if not isinstance(layout, dict):
            raise ValueError("layout must be an object")
        svg = str(payload.get("svg") or "")
        if not svg.lstrip().startswith("<svg"):
            raise ValueError("svg must start with <svg")

        layout_path = (self.output_dir / f"{case_id}.graph.layout.json").resolve()
        svg_path = (self.output_dir / f"{case_id}.graph.edited.svg").resolve()
        if not _is_relative_to(layout_path, self.output_dir) or not _is_relative_to(svg_path, self.output_dir):
            raise ValueError("resolved output path escaped output_dir")

        layout_path.write_text(json.dumps(layout, indent=2, ensure_ascii=False), encoding="utf-8")
        svg_path.write_text(svg, encoding="utf-8")
        return {
            "ok": True,
            "layout_path": layout_path.name,
            "svg_path": svg_path.name,
        }

    def _json_response(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def guess_type(self, path: str) -> str:
        if path.endswith(".svg"):
            return "image/svg+xml"
        if path.endswith(".json"):
            return "application/json"
        return mimetypes.guess_type(path)[0] or "application/octet-stream"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True, help="Report output directory containing index.html.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = output_dir.resolve()
    if not (output_dir / "index.html").exists():
        raise SystemExit(f"index.html not found in {output_dir}")

    ReportHandler.output_dir = output_dir
    server = ThreadingHTTPServer((args.host, args.port), ReportHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Serving {output_dir}")
    print(f"Open {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
