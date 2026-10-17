"""幻灯片 deck（#158 / #159）的静态验收。

覆盖 11 页骨架与页码、五部分顺序、封面与前三页内容、第 4 到第 7 页的
项目部分（问题/目标/三层、三层与因子库接口、Agent 六术语闭环、
26/13/10/3 与四层共线性防御、血统链）、默认与无 JS 状态、打印钩子、
本地资源边界，以及纯路由模块的存在与引用。
深链接与翻页边界的行为由 `src/slides-route.test.ts` 覆盖。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
SLIDES = SITE / "slides"
DECK = SLIDES / "index.html"
DECK_CSS = SLIDES / "slides.css"
DECK_JS = SLIDES / "slides.js"
ROUTE_JS = SLIDES / "slides-route.js"
TOKENS = SITE / "assets" / "tokens.css"
REGISTRY = ROOT / "factor_library" / "registry.json"

TITLE = "【拉普拉斯奖学金】陶唐-上海财经大学-本科-数据科学与大数据技术-2024级"
PAGE_COUNT = 11
PARTS = ("1", "2", "3") + ("4",) * 7 + ("5",)
CAPTURED_TAGS = {"h1", "h2", "h3", "figcaption"}
PROJECT_SLIDES = {4: "项目总览", 5: "系统架构", 6: "Agent 因子挖掘", 7: "因子库与门控"}
LOOP_TERMS = ("课题", "灵感", "轮", "trace", "沙盒", "门控")
DEFENSE_LAYERS = ("源头分流", "生成避让", "评分折扣", "入库与整库")

VOID = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "source",
    "track",
    "wbr",
}


def compact(text: str) -> str:
    """去掉全部空白，便于断言跨行、带空格的字段。"""
    return re.sub(r"\s+", "", text)


class Element:
    def __init__(self, tag: str, attrs: dict[str, str], scopes: tuple[str, ...]) -> None:
        self.tag = tag
        self.attrs = attrs
        self.scopes = scopes


class Page(HTMLParser):
    """收集标签、属性、脚本与按 id 归属的文本。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.elements: list[Element] = []
        self.scripts: list[str] = []
        self.captures: list[tuple[str, str, tuple[str, ...]]] = []
        self._open: list[tuple[str, str | None]] = []
        self._all: list[str] = []
        self._by_id: dict[str, list[str]] = {}
        self._title: list[str] = []
        self._in_script = False
        self._in_title = False
        self._capture: list[tuple[str, list[str], tuple[str, ...]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key: (value or "") for key, value in attrs}
        scopes = tuple(ident for _, ident in self._open if ident)
        self.elements.append(Element(tag, attr, scopes))
        if tag == "script":
            self._in_script = True
        if tag == "title":
            self._in_title = True
        if tag in CAPTURED_TAGS:
            self._capture.append((tag, [], scopes))
        if tag not in VOID:
            self._open.append((tag, attr.get("id")))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key: (value or "") for key, value in attrs}
        scopes = tuple(ident for _, ident in self._open if ident)
        self.elements.append(Element(tag, attr, scopes))

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._in_script = False
        if tag == "title":
            self._in_title = False
        if tag in CAPTURED_TAGS and self._capture and self._capture[-1][0] == tag:
            captured_tag, parts, scopes = self._capture.pop()
            self.captures.append((captured_tag, " ".join("".join(parts).split()), scopes))
        for index in range(len(self._open) - 1, -1, -1):
            if self._open[index][0] == tag:
                del self._open[index:]
                return

    def handle_data(self, data: str) -> None:
        self._all.append(data)
        if self._in_script:
            self.scripts.append(data)
        if self._in_title:
            self._title.append(data)
        if self._capture:
            self._capture[-1][1].append(data)
        for _, ident in self._open:
            if ident:
                self._by_id.setdefault(ident, []).append(data)

    @property
    def text(self) -> str:
        return " ".join("".join(self._all).split())

    @property
    def title(self) -> str:
        return " ".join("".join(self._title).split())

    def text_of(self, ident: str) -> str:
        return " ".join("".join(self._by_id.get(ident, [])).split())

    def has_id(self, ident: str) -> bool:
        return any(element.attrs.get("id") == ident for element in self.elements)

    def find(self, tag: str) -> list[Element]:
        return [element for element in self.elements if element.tag == tag]

    def slides(self) -> list[Element]:
        return [element for element in self.elements if "data-slide" in element.attrs]

    def slide(self, number: int) -> Element:
        for element in self.slides():
            if element.attrs.get("data-slide") == str(number):
                return element
        raise AssertionError(f"缺少第 {number} 页")

    def inside(self, tag: str, number: int) -> list[Element]:
        scope = f"slide-{number}"
        return [element for element in self.find(tag) if scope in element.scopes]

    def captured(self, tag: str, number: int) -> list[str]:
        scope = f"slide-{number}"
        return [text for name, text, scopes in self.captures if name == tag and scope in scopes]


@pytest.fixture(scope="module")
def source() -> str:
    assert DECK.exists(), "缺少 site/slides/index.html"
    return DECK.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def page(source: str) -> Page:
    parser = Page()
    parser.feed(source)
    parser.close()
    return parser


@pytest.fixture(scope="module")
def css() -> str:
    assert DECK_CSS.exists(), "缺少 site/slides/slides.css"
    return DECK_CSS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def script() -> str:
    assert DECK_JS.exists(), "缺少 site/slides/slides.js"
    return DECK_JS.read_text(encoding="utf-8")


def test_deck_is_static_html(source: str, page: Page) -> None:
    assert source.lstrip().lower().startswith("<!doctype html>")
    html = page.find("html")
    assert html and html[0].attrs.get("lang") == "zh-CN"
    assert page.title == TITLE


def test_eleven_slides_in_order(page: Page) -> None:
    slides = page.slides()
    assert len(slides) == PAGE_COUNT
    assert [element.attrs["data-slide"] for element in slides] == [
        str(number) for number in range(1, PAGE_COUNT + 1)
    ]
    ids = [element.attrs.get("id") for element in slides]
    assert all(ids), "每页都需要 id"
    assert len(set(ids)) == PAGE_COUNT, "幻灯片 id 必须唯一"
    assert len({element.attrs.get("id") for element in page.elements if element.attrs.get("id")}) == len(
        [element for element in page.elements if element.attrs.get("id")]
    ), "页面 id 必须唯一"

    for element in slides:
        assert element.tag == "section"
        assert element.attrs.get("role") == "group"
        assert element.attrs.get("aria-roledescription") == "幻灯片"
        label = element.attrs.get("aria-labelledby", "")
        assert page.has_id(label), f"第 {element.attrs['data-slide']} 页的标题引用失效"


def test_every_slide_carries_part_kicker_and_folio(page: Page) -> None:
    for number in range(1, PAGE_COUNT + 1):
        element = page.slide(number)
        text = compact(page.text_of(element.attrs["id"]))
        part = element.attrs.get("data-part")
        assert part, f"第 {number} 页缺少 data-part"
        assert f"第{part}部分" in text, f"第 {number} 页缺少部分 kicker"
        assert f"{number:02d}/11" in text, f"第 {number} 页缺少页码"


def test_five_parts_follow_template_order(page: Page) -> None:
    parts = tuple(element.attrs.get("data-part") for element in page.slides())
    assert parts == PARTS
    assert parts[:3] == ("1", "2", "3")
    assert parts[3:10] == ("4",) * 7
    assert parts[10] == "5"


def test_cover_completes_application_identity(page: Page) -> None:
    cover = compact(page.text_of("slide-1"))
    for item in (
        "ASTER",
        "星菀",
        "陶唐",
        "上海财经大学",
        "信息管理与工程学院",
        "数据科学与大数据技术",
        "2024级",
        "学号2024111155",
        "申请日期2026-10-17",
    ):
        assert item in cover, f"封面缺少 {item}"
    assert len(page.find("h1")) == 1, "整份 deck 只应有一个 h1"


def test_profile_slide_covers_academics(page: Page) -> None:
    profile = compact(page.text_of("slide-2"))
    for item in ("个人简介", "学业基础", "GPA3.81/4.0", "前5%", "主修课程", "治学理念"):
        assert item in profile, f"第 2 页缺少 {item}"


def test_ai_learning_slide_covers_preparation(page: Page) -> None:
    slide = compact(page.text_of("slide-3"))
    for item in ("AI学习与技术积累", "课程", "技术栈", "实习", "认知"):
        assert item in slide, f"第 3 页缺少 {item}"


def test_placeholder_slides_stay_skeleton(page: Page) -> None:
    for number in range(8, PAGE_COUNT + 1):
        element = page.slide(number)
        text = compact(page.text_of(element.attrs["id"]))
        assert "待补充" in text, f"第 {number} 页应保留待补充占位"
        for premature in ("已实现", "规划中", "AutoResearch", "26个因子工件"):
            assert premature not in text, f"第 {number} 页提前写入后续票据内容：{premature}"


def test_project_slides_replace_placeholders(page: Page) -> None:
    for number, title in PROJECT_SLIDES.items():
        element = page.slide(number)
        text = compact(page.text_of(element.attrs["id"]))
        assert "待补充" not in text, f"第 {number} 页仍是占位"
        assert element.attrs.get("data-part") == "4"
        assert element.attrs.get("aria-labelledby") == f"slide-{number}-title"
        titles = [compact(value) for value in page.captured("h2", number)]
        assert titles == [compact(title)], f"第 {number} 页 h2 应为 {title}"


def test_overview_slide_covers_problem_goal_and_layers(page: Page) -> None:
    text = compact(page.text_of("slide-4"))
    for item in ("问题", "目标", "三层架构", "数据层", "研究层", "生产层"):
        assert item in text, f"第 4 页缺少 {item}"
    assert text.index("数据层") < text.index("研究层") < text.index("生产层")
    for item in ("alpha", "衰减", "因子库", "T+1调仓单", "沙盒", "可复现"):
        assert item in text, f"第 4 页缺少 {item}"


def test_architecture_slide_explains_single_interface(page: Page) -> None:
    text = compact(page.text_of("slide-5"))
    assert "研究层与生产层解耦，因子库是唯一接口。" in text
    assert [compact(value) for value in page.captured("h3", 5)] == ["数据层", "研究层", "生产层"]
    for item in ("akshare", "Parquet", "沙盒", "每日跑批", "T+1调仓单", "因子工件"):
        assert item in text, f"第 5 页缺少 {item}"


def test_agent_loop_slide_orders_six_terms(page: Page) -> None:
    headings = [compact(value) for value in page.captured("h3", 6)]
    assert headings[: len(LOOP_TERMS)] == [compact(term) for term in LOOP_TERMS]
    text = compact(page.text_of("slide-6"))
    for item in ("3条灵感", "活跃课题上限3", "30轮熔断", "连续3轮无产出", "入库10个低相关因子", "休眠"):
        assert item in text, f"第 6 页缺少 {item}"


def test_factor_library_slide_counts_match_registry(page: Page) -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    factors = registry["factors"]
    ops = Counter(factor["lineage"]["op"] for factor in factors)
    assert (ops["seed"], ops["mutation"], ops["crossover"]) == (13, 10, 3)
    text = compact(page.text_of("slide-7"))
    expected = (
        f"{len(factors)}个因子工件="
        f"{ops['seed']}种子+{ops['mutation']}变异+{ops['crossover']}交叉"
    )
    assert expected in text, f"第 7 页缺少构成式 {expected}"


def test_factor_library_slide_shows_four_layer_defense(page: Page) -> None:
    text = compact(page.text_of("slide-7"))
    positions = [text.index(name) for name in DEFENSE_LAYERS]
    assert positions == sorted(positions), "四层防御顺序应为源头分流、生成避让、评分折扣、入库与整库"
    for item in ("max_corr", "0.7", "50%", "同簇留强"):
        assert item in text, f"第 7 页缺少 {item}"


def test_factor_library_slide_shows_lineage_chain(page: Page) -> None:
    text = compact(page.text_of("slide-7"))
    assert "lineage{op,parents,run_id,generation}" in text
    for item in ("seed", "mutation", "crossover", "flagship-mining"):
        assert item in text, f"第 7 页缺少 {item}"


def test_project_slides_carry_one_labelled_figure(page: Page) -> None:
    for number in (5, 6, 7):
        images = page.inside("img", number)
        assert len(images) == 1, f"第 {number} 页应有一张主图"
        image = images[0]
        src = image.attrs.get("src", "")
        alt = image.attrs.get("alt", "").strip()
        assert src.startswith("../assets/") and src.endswith(".svg"), f"第 {number} 页主图应为本地 SVG"
        assert "factor-corr" not in src, "不得引用并行票据尚未生成的图片"
        assert len(alt) >= 20, f"第 {number} 页主图 alt 太短"
        target = (SLIDES / src).resolve()
        assert target.exists(), f"{src} 不存在"
        assert "<svg" in target.read_text(encoding="utf-8")
        assert len(page.inside("figure", number)) == 1, f"第 {number} 页主图应包在 figure 里"
        captions = page.captured("figcaption", number)
        assert captions and len(compact(captions[0])) >= 12, f"第 {number} 页缺少图注"



def test_default_state_is_first_slide(page: Page) -> None:
    classes = [element.attrs.get("class", "") for element in page.slides()]
    assert "is-current" in classes[0].split()
    assert all("is-current" not in value.split() for value in classes[1:])
    assert "1 / 11" in page.text
    buttons = [element for element in page.find("button") if element.attrs.get("data-action")]
    prev = [element for element in buttons if element.attrs.get("data-action") == "prev"]
    nxt = [element for element in buttons if element.attrs.get("data-action") == "next"]
    assert prev and "disabled" in prev[0].attrs
    assert nxt and "disabled" not in nxt[0].attrs


def test_navigation_controls_are_labelled(page: Page) -> None:
    labels = {element.attrs.get("aria-label") for element in page.find("button")}
    assert {"上一页", "下一页"} <= labels
    navs = page.find("nav")
    assert navs and any(element.attrs.get("aria-label") == "幻灯片导航" for element in navs)
    progress = [element for element in page.elements if element.attrs.get("role") == "progressbar"]
    assert progress, "缺少进度条"
    attrs = progress[0].attrs
    assert attrs.get("aria-valuemin") == "1"
    assert attrs.get("aria-valuemax") == str(PAGE_COUNT)
    assert attrs.get("aria-valuenow") == "1"
    assert attrs.get("aria-label")


def test_no_js_fallback_keeps_every_slide(source: str, css: str) -> None:
    html = re.search(r"<html[^>]*>", source)
    assert html and "no-js" in html.group(0)
    inline = re.findall(r"<script>(.*?)</script>", source, re.S)
    assert any("documentElement" in script for script in inline), "缺少尽早切换 js class 的内联脚本"
    assert any("no-js" in script and "js" in script for script in inline), "内联脚本应移除 no-js"
    assert re.search(r"html\.js\s+\.slide:not\(\.is-current\)", css), "只在 .js 下隐藏非当前页"
    assert re.search(r"html\.no-js\s+\.slide", css), "无 JS 时应纵向排列全部幻灯片"


def test_print_rules_put_one_slide_per_page(css: str) -> None:
    assert "@media print" in css
    print_block = css[css.index("@media print") :]
    assert "@page" in print_block
    assert "1280px" in print_block and "720px" in print_block
    assert "break-after" in print_block
    for chrome in ("deck-bar", "deck-progress", "rotate-hint", "skip"):
        assert chrome in print_block, f"打印时未隐藏 {chrome}"
    assert "display: none" in print_block


def test_stylesheet_defines_scalable_deck(css: str) -> None:
    assert "aspect-ratio" in css
    assert "--slide-w: 1280px" in css
    assert "--slide-h: 720px" in css
    assert "var(--color-accent" in css
    assert "var(--font-serif" in css
    assert "prefers-reduced-motion" in css
    assert ":focus-visible" in css
    assert "@import" not in css
    assert "http://" not in css and "https://" not in css
    for width in ("1440", "1024", "768", "390"):
        assert width in css, f"缺少 {width} 断点"


def test_script_wires_keyboard_and_hash(script: str) -> None:
    for token in (
        "ArrowLeft",
        "ArrowRight",
        "Home",
        "End",
        "hashchange",
        "popstate",
        "replaceState",
        "location.hash",
        "SecurityError",
        "try",
        "catch",
    ):
        assert token in script, f"脚本缺少 {token}"
    assert "slides-route.js" in script, "脚本应引用纯路由模块"
    assert "data-slide" in script
    assert "aria-current" in script
    assert "prefers-reduced-motion" not in script, "动效偏好由 CSS 处理"


def test_route_module_is_pure_and_exports_contract() -> None:
    assert ROUTE_JS.exists(), "缺少 site/slides/slides-route.js"
    route = ROUTE_JS.read_text(encoding="utf-8")
    for name in ("parseSlideHash", "clampSlide", "prevSlide", "nextSlide", "slideHash"):
        assert re.search(rf"export function {name}\b", route), f"路由模块缺少 {name}"
    for name in ("FIRST_SLIDE", "LAST_SLIDE", "SLIDE_COUNT"):
        assert re.search(rf"export const {name}\b", route), f"路由模块缺少 {name}"
    assert "LAST_SLIDE = 11" in route
    assert "document" not in route and "window" not in route, "路由模块不能依赖 DOM"


def test_assets_are_local_and_resolve(page: Page) -> None:
    for image in page.find("img"):
        assert image.attrs.get("alt", "").strip(), "每张图片都必须有 alt"
    for element in page.elements:
        for attribute in ("href", "src"):
            value = element.attrs.get(attribute)
            if not value or value.startswith("#"):
                continue
            assert not re.match(r"^(?:[a-z]+:)?//", value), f"外部资源 {value}"
            assert not value.startswith("/"), f"站内路径不能用根绝对路径 {value}"
            target = (SLIDES / value.split("#")[0].split("?")[0]).resolve()
            assert target.exists(), f"{value} 指向不存在的文件"
    tokens = [element for element in page.find("link") if element.attrs.get("rel") == "stylesheet"]
    hrefs = [element.attrs.get("href") for element in tokens]
    assert "../assets/tokens.css" in hrefs
    assert "slides.css" in hrefs
    module_scripts = [element for element in page.find("script") if element.attrs.get("src")]
    assert module_scripts and module_scripts[0].attrs.get("type") == "module"
    assert module_scripts[0].attrs["src"] == "slides.js"


def test_shared_tokens_exist() -> None:
    assert TOKENS.exists(), "缺少共享 token 文件"
    tokens = TOKENS.read_text(encoding="utf-8")
    assert "--color-accent:" in tokens
    assert "--font-serif:" in tokens


def test_no_phone_analytics_or_cookie(source: str, page: Page) -> None:
    assert not re.search(r"(?<!\d)1[3-9]\d{9}(?!\d)", compact(page.text))
    for path in (DECK, DECK_CSS, DECK_JS, ROUTE_JS):
        lowered = path.read_text(encoding="utf-8").lower()
        for marker in ("gtag", "googletagmanager", "google-analytics", "plausible", "umami", "matomo"):
            assert marker not in lowered, f"{path.name} 含分析脚本 {marker}"
        assert "document.cookie" not in lowered, f"{path.name} 读写 Cookie"
    assert "tel:" not in source.lower()
