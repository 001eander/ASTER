"""把答辩 deck 打成单文件导出页（issue #161）。

读 ``site/slides/index.html``，写 ``site/slides/export.html``：

- ``../assets/tokens.css`` 与 ``slides.css`` 按原顺序内联为 ``<style>``；
- ``slides-route.js`` 与 ``slides.js`` 合成一个内联 ``<script type="module">``，
  去掉 ``export`` 关键字与 ``import`` 语句，保证浏览器直接执行；
- favicon 与 ``slide-*.svg`` 等本地图片改写成 ``data:`` URI，CSS 里的 ``url()`` 同样内联；
- 删掉指向站点目录的 ``<link>``、``<script src>``、下载链接，导出页不再出现
  ``../``、``slides.css``、``slides.js``、``assets/``。

输出确定性：同样输入两次生成字节一致，UTF-8 无 BOM、LF。

用法::

    uv run python scripts/build_slides_export.py
    uv run python scripts/build_slides_export.py --source site/slides/index.html --output /tmp/x.html
"""
from __future__ import annotations

import argparse
import base64
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "site" / "slides" / "index.html"
DEFAULT_OUTPUT = ROOT / "site" / "slides" / "export.html"

ROUTE_JS = "slides-route.js"
SLIDES_JS = "slides.js"

MIME_TYPES = {
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
}

STYLESHEET_LINK = re.compile(r'[ \t]*<link rel="stylesheet" href="(?P<href>[^"]+)">\n')
ICON_LINK = re.compile(r'<link rel="icon" href="(?P<href>[^"]+)"')
MODULE_SCRIPT = re.compile(r'[ \t]*<script type="module" src="(?P<src>[^"]+)"></script>\n')
DOWNLOAD_LINK = re.compile(r'[ \t]*<a\b[^>]*\bdownload="[^"]*"[^>]*>.*?</a>\n?', re.S)
IMAGE_SRC = re.compile(r'(<img\b[^>]*?\bsrc=")(?P<src>[^"]*)(")')
CSS_URL = re.compile(r"""url\(\s*(?P<quote>["']?)(?P<path>[^"')]+)(?P=quote)\s*\)""")
IMPORT_BLOCK = re.compile(r'^import\s*\{.*?\}\s*from\s*"[^"]+";\n', re.S | re.M)
EXPORT_KEYWORD = re.compile(r"^export\s+", re.M)

FORBIDDEN = ("../", "slides.css", "slides.js", "assets/", "http://", "https://")


class BuildError(RuntimeError):
    """输入缺文件或产物里仍有站点目录引用。"""


def read_text(path: Path) -> str:
    """按 UTF-8 读入并统一成 LF，保证跨平台字节一致。"""
    return path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")


def data_uri(path: Path) -> str:
    """把本地图片编码成 data URI。"""
    mime = MIME_TYPES.get(path.suffix.lower())
    if mime is None:
        raise BuildError(f"不支持的图片类型：{path}")
    if not path.is_file():
        raise BuildError(f"图片不存在：{path}")
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def inline_css(css: str, base: Path) -> str:
    """把 CSS 里指向本地文件的 url() 换成 data URI。"""

    def replace(match: re.Match[str]) -> str:
        target = match.group("path").strip()
        if target.startswith(("data:", "http://", "https://", "#")):
            return match.group(0)
        return f'url("{data_uri((base / target).resolve())}")'

    return CSS_URL.sub(replace, css)


def inline_module(base: Path) -> str:
    """把纯路由模块与交互脚本合成一段可直接执行的模块代码。"""
    route = EXPORT_KEYWORD.sub("", read_text(base / ROUTE_JS))
    script = read_text(base / SLIDES_JS)

    stripped, count = IMPORT_BLOCK.subn("", script, count=1)
    if count != 1:
        raise BuildError(f"{SLIDES_JS} 的 import 语句格式变了，导出脚本需要同步修改")

    return f"{route.rstrip()}\n\n{stripped.lstrip()}"


def render(source: Path) -> str:
    """由源 HTML 生成导出的完整文本。"""
    base = source.resolve().parent
    text = read_text(source)

    def style_block(match: re.Match[str]) -> str:
        css_path = (base / match.group("href")).resolve()
        if not css_path.is_file():
            raise BuildError(f"样式表不存在：{css_path}")
        return f"  <style>\n{inline_css(read_text(css_path), css_path.parent)}\n  </style>\n"

    text = STYLESHEET_LINK.sub(style_block, text)

    def icon(match: re.Match[str]) -> str:
        return f'<link rel="icon" href="{data_uri((base / match.group("href")).resolve())}"'

    text = ICON_LINK.sub(icon, text)

    def image(match: re.Match[str]) -> str:
        return f'{match.group(1)}{data_uri((base / match.group("src")).resolve())}{match.group(3)}'

    text = IMAGE_SRC.sub(image, text)
    text = DOWNLOAD_LINK.sub("", text)

    module = inline_module(base)
    text, count = MODULE_SCRIPT.subn(lambda _: f'  <script type="module">\n{module}\n  </script>\n', text, count=1)
    if count != 1:
        raise BuildError("index.html 里没有找到 slides.js 的模块脚本标签")

    for needle in FORBIDDEN:
        if needle in text:
            raise BuildError(f"导出结果仍含 {needle}，说明有资源没有内联或链接没有改写")

    if not text.endswith("\n"):
        text += "\n"
    return text


def build_export(source: Path = DEFAULT_SOURCE, output: Path = DEFAULT_OUTPUT) -> Path:
    """生成导出页，返回写出的路径。"""
    text = render(Path(source))
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(text.encode("utf-8"))
    return target


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="生成答辩 deck 的单文件导出页")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="源 HTML，默认 site/slides/index.html")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出路径，默认 site/slides/export.html")
    args = parser.parse_args(argv)

    try:
        target = build_export(args.source, args.output)
    except BuildError as error:
        print(f"导出失败：{error}", file=sys.stderr)
        return 1

    print(f"已生成 {target}（{target.stat().st_size} 字节）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
