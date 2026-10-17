"""单文件导出脚本的错误分支与确定性。

全部用 ``tmp_path`` 里的最小 deck 夹具，不依赖仓库的 ``site/`` 产物。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.build_slides_export import (
    BuildError,
    build_export,
    data_uri,
    render,
)

SOURCE = """<!DOCTYPE html>
<html lang="zh-CN" class="no-js">
<head>
  <meta charset="utf-8">
  <link rel="icon" href="icon.svg" type="image/svg+xml">
  <link rel="stylesheet" href="tokens.css">
  <link rel="stylesheet" href="deck.css">
</head>
<body>
  <a href="export.html" download="deck.html">下载单文件 HTML</a>
  <img src="pic.svg" alt="示意图">
  <script type="module" src="slides.js"></script>
</body>
</html>
"""


def _write_fixture(tmp_path: Path, source: str = SOURCE) -> Path:
    (tmp_path / "tokens.css").write_text(":root { --color-accent: #7a5bb5; }\n", encoding="utf-8")
    (tmp_path / "deck.css").write_text("body { margin: 0; }\n", encoding="utf-8")
    (tmp_path / "slides-route.js").write_text(
        "export function clampSlide(n) {\n  return n;\n}\n", encoding="utf-8"
    )
    (tmp_path / "slides.js").write_text(
        'import { clampSlide } from "./slides-route.js";\n\nconsole.log(clampSlide(1));\n',
        encoding="utf-8",
    )
    (tmp_path / "icon.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1"></svg>\n', encoding="utf-8"
    )
    (tmp_path / "pic.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2 1"></svg>\n', encoding="utf-8"
    )
    path = tmp_path / "index.html"
    path.write_text(source, encoding="utf-8")
    return path


def test_unsupported_image_mime_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "pic.bmp"
    target.write_bytes(b"BM")

    with pytest.raises(BuildError, match="不支持的图片类型"):
        data_uri(target)


def test_missing_stylesheet_is_rejected(tmp_path: Path) -> None:
    source = _write_fixture(
        tmp_path,
        SOURCE.replace(
            '<link rel="stylesheet" href="deck.css">',
            '<link rel="stylesheet" href="absent.css">',
        ),
    )

    with pytest.raises(BuildError, match="样式表不存在"):
        render(source)


def test_missing_module_script_is_rejected(tmp_path: Path) -> None:
    source = _write_fixture(
        tmp_path, SOURCE.replace('  <script type="module" src="slides.js"></script>\n', "")
    )

    with pytest.raises(BuildError, match="slides.js"):
        render(source)


def test_residual_site_reference_is_rejected(tmp_path: Path) -> None:
    source = _write_fixture(
        tmp_path, SOURCE.replace("<body>", '<body>\n  <p>https://example.com</p>')
    )

    with pytest.raises(BuildError, match="仍含"):
        render(source)


def test_same_input_produces_identical_bytes(tmp_path: Path) -> None:
    source = _write_fixture(tmp_path)
    first = build_export(source=source, output=tmp_path / "first.html")
    second = build_export(source=source, output=tmp_path / "second.html")

    data = first.read_bytes()
    assert data == second.read_bytes(), "同样输入两次生成必须字节一致"
    assert not data.startswith(b"\xef\xbb\xbf"), "导出文件不能带 BOM"
    assert b"\r\n" not in data, "导出文件必须用 LF"

    text = data.decode("utf-8")
    for needle in ("../", "slides.css", "slides.js", "assets/", "http://", "https://"):
        assert needle not in text, f"导出结果仍引用 {needle}"
    assert "<style>" in text and "type=\"module\"" in text
    assert "下载单文件 HTML" not in text, "导出页应删掉下载链接"
