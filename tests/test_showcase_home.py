"""展示站主页的静态验收：标题、页首字段、导航、外链与资源边界，以及系统全景、Agent、因子库与 AutoML 各段。

测试只解析站点产物，不依赖 CSS 类名或 DOM 层级。末尾覆盖 #162 的幻灯片入口
（封面预览、11 项目录、进入按钮、无 iframe）、打印展开证据的钩子、
reduced motion 与打印规则，以及全站无外部资源。
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

import pytest

from scripts.build_site_assets import flatten_metrics, format_metric

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
HOME = SITE / "index.html"
TOKENS = SITE / "assets" / "tokens.css"
SITE_CSS = SITE / "assets" / "site.css"
METRICS = SITE / "assets" / "metrics.json"
REGISTRY = ROOT / "factor_library" / "registry.json"
WORKFLOW = ROOT / ".github" / "workflows" / "pages.yml"

TITLE = "【拉普拉斯奖学金】陶唐-上海财经大学-本科-数据科学与大数据技术-2024级"
REPO = "https://github.com/001eander/ASTER"
WIKI = "https://github.com/001eander/ASTER/wiki"
PROFILE = "https://github.com/001eander"
MAIL = "mailto:2024111155@stu.sufe.edu.cn"
ALLOWED_OFFSITE = {REPO, WIKI, PROFILE, MAIL}

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

RESERVED_SECTION_IDS = (
    "system",
    "agent",
    "factor-library",
    "automl",
    "portfolio",
    "production",
    "evidence",
    "slides",
)

LAYER_TERMS = ("数据层", "研究层", "生产层")
AGENT_TERMS = ("课题", "灵感", "轮", "trace", "沙盒", "门控")
DEFENSE_TERMS = ("源头分流", "生成避让", "评分折扣", "入库门控")
AUTOML_CONCEPTS = ("数据集工件", "AutoGluon", "walk-forward", "模型注册表", "信号工件")
PORTFOLIO_POINTS = ("约束族", "放松阶梯", "冻结持仓", "整手")
PRODUCTION_FLOW = ("数据增量", "因子重算", "信号合成", "组合优化", "T+1调仓单")


def compact(text: str) -> str:
    """去掉全部空白，便于断言跨行、带空格的字段。"""
    return re.sub(r"\s+", "", text)


class Element:
    def __init__(self, tag: str, attrs: dict[str, str], scopes: tuple[str, ...]) -> None:
        self.tag = tag
        self.attrs = attrs
        self.scopes = scopes


class Page(HTMLParser):
    """收集标签、属性、按 id 归属的文本与图注文本。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.elements: list[Element] = []
        self.scripts: list[str] = []
        self.captions: list[tuple[tuple[str, ...], str]] = []
        self._open: list[tuple[str, str | None]] = []
        self._all: list[str] = []
        self._by_id: dict[str, list[str]] = {}
        self._title: list[str] = []
        self._caption: list[str] | None = None
        self._caption_scopes: tuple[str, ...] = ()
        self._in_script = False
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key: (value or "") for key, value in attrs}
        scopes = tuple(ident for _, ident in self._open if ident)
        self.elements.append(Element(tag, attr, scopes))
        if tag == "script":
            self._in_script = True
        if tag == "title":
            self._in_title = True
        if tag == "figcaption":
            self._caption = []
            self._caption_scopes = scopes
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
        if tag == "figcaption" and self._caption is not None:
            caption = " ".join("".join(self._caption).split())
            self.captions.append((self._caption_scopes, caption))
            self._caption = None
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
        if self._caption is not None:
            self._caption.append(data)
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


@pytest.fixture(scope="module")
def page() -> Page:
    assert HOME.exists(), "缺少 site/index.html"
    parser = Page()
    parser.feed(HOME.read_text(encoding="utf-8"))
    parser.close()
    return parser


def test_home_is_static_html(tmp_path: Path) -> None:
    assert HOME.exists()
    source = HOME.read_text(encoding="utf-8")
    assert source.lstrip().lower().startswith("<!doctype html>")


def test_title_matches_template(page: Page) -> None:
    assert len(page.find("title")) == 1
    assert page.title == TITLE


def test_document_language_and_landmarks(page: Page) -> None:
    html = page.find("html")
    assert html and html[0].attrs.get("lang") == "zh-CN"
    for tag in ("header", "nav", "main", "footer"):
        assert page.find(tag), f"缺少 {tag} landmark"


def test_skip_link_targets_main(page: Page) -> None:
    anchors = [element for element in page.elements if element.tag == "a"]
    skip = [anchor for anchor in anchors if anchor.attrs.get("href", "").startswith("#")]
    assert skip, "缺少跳转链接"
    assert skip[0].attrs["href"] == "#main"
    assert "跳到主要内容" in compact(page.text)
    main = [element for element in page.find("main") if element.attrs.get("id") == "main"]
    assert main and main[0].attrs.get("tabindex") == "-1"


def test_hero_carries_application_identity(page: Page) -> None:
    hero = compact(page.text_of("top"))
    assert hero, "缺少 #top 页首"
    required = (
        "ASTER",
        "陶唐",
        "上海财经大学",
        "信息管理与工程学院",
        "数据科学与大数据技术",
        "2024级",
        "学号2024111155",
        "申请日期2026-10-17",
    )
    for item in required:
        assert item in hero, f"页首缺少 {item}"
    for word in ("Agent", "因子", "AutoML"):
        assert word in hero, f"页首定位未提及 {word}"


def test_navigation_reaches_author_and_footer(page: Page) -> None:
    assert page.find("nav"), "缺少导航"
    targets = {element.attrs.get("href") for element in page.elements if element.tag == "a"}
    assert "#author" in targets
    assert "#footer" in targets
    assert page.has_id("author")
    assert page.has_id("footer")


def test_navigation_carries_repository_and_wiki(page: Page) -> None:
    hrefs = [element.attrs.get("href") for element in page.find("a")]
    assert REPO in hrefs
    assert WIKI in hrefs


def test_navigation_links_to_system_and_agent(page: Page) -> None:
    assert page.find("nav"), "缺少导航"
    hrefs = {element.attrs.get("href") for element in page.find("a") if "top" in element.scopes}
    for target in ("#system", "#agent"):
        assert target in hrefs, f"导航缺少 {target}"
    assert page.has_id("system") and page.has_id("agent")


def test_home_orders_sections_from_hero_to_author(page: Page) -> None:
    ids = [element.attrs.get("id") for element in page.elements]
    for ident in ("top", "system", "agent", "author"):
        assert ident in ids, f"缺少 #{ident}"
    assert ids.index("top") < ids.index("system") < ids.index("agent") < ids.index("author")


def test_system_section_reads_in_layer_order(page: Page) -> None:
    text = compact(page.text_of("system"))
    assert len(text) >= 200, "#system 内容过少"
    positions = [text.find(term) for term in LAYER_TERMS]
    for term, position in zip(LAYER_TERMS, positions):
        assert position >= 0, f"系统全景缺少 {term}"
    assert positions == sorted(positions), "三层顺序应为数据层、研究层、生产层"


def test_factor_library_is_the_only_interface(page: Page) -> None:
    text = compact(page.text_of("system"))
    assert "研究层与生产层解耦" in text
    assert "因子库是唯一接口" in text
    assert "pool" in text, "接口说明应指出生产层只读 pool 条目"


def test_agent_section_covers_six_terms_in_order(page: Page) -> None:
    text = compact(page.text_of("agent"))
    assert len(text) >= 200, "#agent 内容过少"
    positions = [text.find(term) for term in AGENT_TERMS]
    for term, position in zip(AGENT_TERMS, positions):
        assert position >= 0, f"Agent 段缺少 {term}"
    assert positions == sorted(positions), "六个概念应按课题、灵感、轮、trace、沙盒、门控的顺序出现"


def test_agent_section_states_loop_budget(page: Page) -> None:
    text = compact(page.text_of("agent"))
    for item in ("3条灵感", "上限3", "30轮", "连续3轮无产出", "10个低相关因子"):
        assert item in text, f"Agent 段缺少关键数字 {item}"


def test_agent_loop_figure_has_alt_and_caption(page: Page) -> None:
    images = [
        element
        for element in page.find("img")
        if "agent" in element.scopes and element.attrs.get("src", "").endswith("agent-loop.svg")
    ]
    assert len(images) == 1, "Agent 段需要一张本地 agent-loop.svg 闭环图"
    assert images[0].attrs.get("alt", "").strip(), "闭环图缺少 alt"

    captions = [text for scopes, text in page.captions if "agent" in scopes]
    assert len(captions) == 1, "闭环图需要一条图注"
    assert "闭环" in compact(captions[0]), "图注应说明闭环"


def test_agent_loop_asset_is_self_contained_svg() -> None:
    asset = SITE / "assets" / "agent-loop.svg"
    assert asset.exists(), "缺少 site/assets/agent-loop.svg"
    source = asset.read_text(encoding="utf-8")
    root = ElementTree.fromstring(source)
    assert root.tag.endswith("svg"), "agent-loop.svg 根节点不是 svg"
    assert "href" not in source, "闭环图不应引用外部资源"
    assert "<image" not in source
    for term in AGENT_TERMS:
        assert term in source, f"闭环图缺少 {term}"



def test_home_orders_library_and_automl_between_agent_and_author(page: Page) -> None:
    ids = [element.attrs.get("id") for element in page.elements]
    for ident in ("agent", "factor-library", "automl", "author"):
        assert ident in ids, f"缺少 #{ident}"
    assert (
        ids.index("agent")
        < ids.index("factor-library")
        < ids.index("automl")
        < ids.index("author")
    ), "栏目顺序应为 Agent、因子库、AutoML、作者"


def test_factor_library_counts_add_up(page: Page) -> None:
    text = compact(page.text_of("factor-library"))
    assert len(text) >= 300, "#factor-library 内容过少"
    for item in ("26个因子工件", "13个种子", "10个变异", "3个交叉"):
        assert item in text, f"因子库段缺少 {item}"
    assert "清空" not in text, "不应写与当前 registry 冲突的清空说法"


def test_factor_library_states_generation_and_source_run(page: Page) -> None:
    text = compact(page.text_of("factor-library"))
    for item in ("第0代", "第1代", "flagship-mining", "pool"):
        assert item in text, f"因子库段缺少 {item}"


def test_factor_library_lists_four_defenses_in_order(page: Page) -> None:
    text = compact(page.text_of("factor-library"))
    positions = [text.find(term) for term in DEFENSE_TERMS]
    for term, position in zip(DEFENSE_TERMS, positions):
        assert position >= 0, f"四层防御缺少 {term}"
    assert positions == sorted(positions), "四层防御应按源头、生成、评分、入库的顺序出现"
    assert "方向分布地图" in text, "源头分流应说明方向分布地图"
    assert "同方向" in text, "生成避让应说明同方向已入库摘要"
    assert "1−|max_corr|/0.7" in text, "评分折扣应写出折扣公式"
    assert "0.7" in text and "一半" in text, "入库门控应写出阈值与 pool 容量规则"
    assert "同一相关簇" in text or "同簇" in text, "整库应说明同簇留强"


def test_factor_library_explains_lineage_and_trace(page: Page) -> None:
    text = compact(page.text_of("factor-library"))
    for item in (
        "lineage{op,parents,run_id,generation}",
        "seed",
        "mutation",
        "crossover",
        "血统",
        "trace",
    ):
        assert item in text, f"血统链说明缺少 {item}"
    assert "过程记录" in text and "结构化记录" in text, "应区分 trace 与血统的记录性质"


def test_factor_library_heatmap_is_local_with_alt_and_caption(page: Page) -> None:
    asset = SITE / "assets" / "factor-corr.png"
    images = [
        element
        for element in page.find("img")
        if "factor-library" in element.scopes
    ]
    if not asset.exists():
        assert not images, "缺少热图文件时页面不应引用图片"
        return
    assert len(images) == 1, "因子库段需要一张本地的因子相关性热图"
    assert images[0].attrs.get("src") == "assets/factor-corr.png"
    assert images[0].attrs.get("alt", "").strip(), "热图缺少 alt"
    captions = [text for scopes, text in page.captions if "factor-library" in scopes]
    assert len(captions) == 1, "热图需要一条图注"
    assert "相关" in compact(captions[0]), "图注应说明相关性"


def test_automl_and_autoresearch_are_parallel_columns(page: Page) -> None:
    columns = [element for element in page.find("section") if "automl" in element.scopes]
    assert len(columns) == 2, "AutoML 段应为左右两栏"
    ids = [element.attrs.get("id") for element in page.elements]
    assert "automl-implemented" in ids and "automl-planned" in ids
    assert ids.index("automl-implemented") < ids.index("automl-planned"), "左栏应为 AutoML"


def test_automl_column_marks_implemented_and_covers_five_concepts(page: Page) -> None:
    text = compact(page.text_of("automl-implemented"))
    assert "AutoML" in text
    assert "已实现" in text
    for term in AUTOML_CONCEPTS:
        assert term in text, f"AutoML 栏缺少 {term}"


def test_autoresearch_column_is_planned_roadmap(page: Page) -> None:
    text = compact(page.text_of("automl-planned"))
    assert "AutoResearch" in text
    assert "规划中" in text
    assert "M5" in text
    for item in ("Agent调参", "网络结构演化", "walk-forward", "样本外IC"):
        assert item in text, f"AutoResearch 栏缺少 {item}"
    assert "已实现" not in text, "AutoResearch 不能写成已实现"
    assert "已完成" not in text, "AutoResearch 不能写成已完成"


def test_navigation_links_to_library_and_automl(page: Page) -> None:
    hrefs = [
        element.attrs.get("href")
        for element in page.find("a")
        if "top" in element.scopes
    ]
    expected = ("#system", "#agent", "#factor-library", "#automl", "#author", "#footer")
    for target in expected:
        assert target in hrefs, f"导航缺少 {target}"
    positions = [hrefs.index(target) for target in expected]
    assert positions == sorted(positions), "导航顺序应保持不变，新链接插在 Agent 与作者之间"
    assert page.has_id("factor-library") and page.has_id("automl")


def test_author_section_completes_application_record(page: Page) -> None:
    author = compact(page.text_of("author"))
    assert author, "缺少 #author 作者区"
    for item in (
        "上海财经大学",
        "信息管理与工程学院",
        "数据科学与大数据技术",
        "2024级",
        "2024111155",
        "2026-10-17",
        "2024111155@stu.sufe.edu.cn",
        "001eander",
        "3.81/4.0",
        "前5%",
        "国家奖学金",
        "2025",
        "华数杯",
        "国家一等奖",
        "2026",
    ):
        assert item in author, f"作者区缺少 {item}"
    assert author.count("实习") >= 3

    mailto = [
        element.attrs["href"]
        for element in page.find("a")
        if element.attrs.get("href", "").startswith("mailto:")
    ]
    assert mailto == [MAIL]


def test_footer_states_maintenance_and_disclaimer(page: Page) -> None:
    footer = page.text_of("footer")
    assert "2026" in footer
    assert "陶唐" in footer
    assert "历史回测，非实盘，不构成投资建议" in footer


def test_no_external_resource_requests(page: Page) -> None:
    for element in page.elements:
        src = element.attrs.get("src")
        if src is not None:
            assert not re.match(r"^(?:[a-z]+:)?//", src), f"外部资源 {src}"
        href = element.attrs.get("href")
        if href is None:
            continue
        if re.match(r"^(?:https?:)?//", href):
            assert href in ALLOWED_OFFSITE, f"未登记的外链 {href}"
            assert set(element.scopes) & {"top", "author"}, f"外链位置越界 {href}"
    for tag in ("link", "script", "img", "iframe", "source", "video", "audio"):
        for element in page.find(tag):
            target = element.attrs.get("href") or element.attrs.get("src")
            if target:
                assert not re.match(r"^(?:[a-z]+:)?//", target), f"{tag} 引用外部资源 {target}"


def test_local_assets_resolve(page: Page) -> None:
    for element in page.elements:
        for attribute in ("href", "src"):
            value = element.attrs.get(attribute)
            if not value or value.startswith(("#", "mailto:", "http", "//")):
                continue
            path = SITE / value.split("#")[0].split("?")[0]
            assert path.exists(), f"{value} 指向不存在的站内文件"


def test_no_phone_analytics_or_cookies(page: Page) -> None:
    source = HOME.read_text(encoding="utf-8")
    assert not re.search(r"(?<!\d)1[3-9]\d{9}(?!\d)", compact(page.text)), "页面含手机号"
    assert "tel:" not in source.lower()
    for asset in SITE.rglob("*"):
        if asset.suffix not in {".html", ".css", ".js"}:
            continue
        lowered = asset.read_text(encoding="utf-8").lower()
        for marker in ("gtag", "googletagmanager", "google-analytics", "plausible", "umami", "matomo"):
            assert marker not in lowered, f"{asset.name} 含分析脚本 {marker}"
        assert "document.cookie" not in lowered, f"{asset.name} 读写 Cookie"


def test_every_image_has_alt(page: Page) -> None:
    for image in page.find("img"):
        assert "alt" in image.attrs, f"图片缺少 alt: {image.attrs.get('src')}"


def test_reserved_sections_are_not_empty_placeholders(page: Page) -> None:
    for ident in RESERVED_SECTION_IDS:
        if page.has_id(ident):
            assert len(compact(page.text_of(ident))) >= 40, f"#{ident} 是空占位"


def test_stylesheet_defines_visual_system() -> None:
    tokens = TOKENS.read_text(encoding="utf-8")
    site_css = SITE_CSS.read_text(encoding="utf-8")
    combined = tokens + site_css
    assert re.search(r"--color-accent:\s*#[0-9a-fA-F]{6}", tokens)
    assert re.search(r"--font-serif:", tokens)
    assert re.search(r"--font-mono:", tokens)
    assert "system-ui" in tokens
    assert "prefers-reduced-motion" in combined
    assert ":focus-visible" in combined
    for width in ("1440", "1024", "768", "390"):
        assert width in combined, f"缺少 {width} 断点"
    assert "@import" not in combined
    assert "http://" not in combined
    assert "https://" not in combined
    assert "url(" not in combined.lower(), "样式不应请求外部或本地图片"


def test_body_text_meets_wcag_aa_contrast() -> None:
    tokens = TOKENS.read_text(encoding="utf-8")

    def color(name: str) -> str:
        match = re.search(rf"--{name}:\s*(#[0-9a-fA-F]{{6}})", tokens)
        assert match, f"缺少 --{name}"
        return match.group(1)

    def luminance(value: str) -> float:
        channels = [int(value[index : index + 2], 16) / 255 for index in (1, 3, 5)]
        linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    def ratio(foreground: str, background: str) -> float:
        light, dark = sorted((luminance(foreground), luminance(background)), reverse=True)
        return (light + 0.05) / (dark + 0.05)

    pairs = (
        ("color-ink", "color-paper"),
        ("color-ink-muted", "color-paper"),
        ("color-accent", "color-paper"),
        ("color-ink", "color-paper-raised"),
        ("color-ink-muted", "color-paper-raised"),
        ("color-accent", "color-paper-raised"),
        ("color-hero-ink", "color-hero-bg"),
        ("color-hero-muted", "color-hero-bg"),
        ("color-accent-soft", "color-hero-bg"),
    )
    for foreground, background in pairs:
        assert ratio(color(foreground), color(background)) >= 4.5, f"{foreground} 对比度不足"


def test_pages_workflow_publishes_site() -> None:
    assert WORKFLOW.exists(), "缺少 Pages 部署工作流"
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "push" in workflow and "main" in workflow
    assert "workflow_dispatch" in workflow
    assert "actions/configure-pages" in workflow
    assert "actions/upload-pages-artifact" in workflow
    assert "actions/deploy-pages" in workflow
    assert re.search(r"path:\s*site", workflow)
    assert "pages: write" in workflow
    assert "id-token: write" in workflow
    assert "concurrency" in workflow


def _css_block(source: str, start: int) -> str:
    """从 start 处的左花括号开始，返回配平后的规则体。"""
    assert source[start] == "{", "CSS 规则起始位置错误"
    depth = 0
    for index in range(start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start + 1 : index]
    raise AssertionError("CSS 花括号未配平")


def _media_blocks(source: str) -> list[tuple[str, str]]:
    """返回全部 @media 块的条件与规则体。"""
    blocks: list[tuple[str, str]] = []
    for match in re.finditer(r"@media[^{]*", source):
        brace = source.find("{", match.end())
        assert brace != -1, "@media 缺少规则体"
        blocks.append((source[match.start() : brace], _css_block(source, brace)))
    return blocks


def test_home_orders_portfolio_and_production_between_automl_and_author(page: Page) -> None:
    ids = [element.attrs.get("id") for element in page.elements]
    for ident in ("automl", "portfolio", "production", "author"):
        assert ident in ids, f"缺少 #{ident}"
    assert (
        ids.index("automl")
        < ids.index("portfolio")
        < ids.index("production")
        < ids.index("author")
    ), "栏目顺序应为 AutoML、组合优化、生产闭环、作者"


def test_portfolio_section_covers_four_points_in_order(page: Page) -> None:
    text = compact(page.text_of("portfolio"))
    assert len(text) >= 400, "#portfolio 内容过少"
    positions = [text.find(term) for term in PORTFOLIO_POINTS]
    for term, position in zip(PORTFOLIO_POINTS, positions):
        assert position >= 0, f"组合优化段缺少 {term}"
    assert positions == sorted(positions), "四个要点应按约束族、放松阶梯、冻结持仓、整手的顺序出现"


def test_portfolio_constraints_are_parameterised_against_benchmark(page: Page) -> None:
    text = compact(page.text_of("portfolio"))
    for item in (
        "满仓",
        "禁卖空",
        "bench_w±0.005",
        "非成分≤0.005",
        "行业偏离±0.5%",
        "市值偏离±0.5%倍std",
        "六风格",
        "成分覆盖度≥0.5",
        "Σ|w−w_prev|≤0.2",
        "min−αᵀw+λwᵀΣw",
        "Ledoit-Wolf",
    ):
        assert item in text, f"约束族缺少 {item}"


def test_portfolio_relaxation_ladder_is_ordered_and_bounded(page: Page) -> None:
    text = compact(page.text_of("portfolio-relax"))
    assert text, "缺少放松阶梯条目"
    for item in ("Minimize(0)", "每3轮", "0.001", "0.01", "最多60轮", "2.0", "保持持仓"):
        assert item in text, f"放松阶梯缺少 {item}"
    positions = [text.find(term) for term in ("风格", "行业/市值", "换手")]
    for term, position in zip(("风格", "行业/市值", "换手"), positions):
        assert position >= 0, f"放松阶梯缺少 {term}"
    assert positions == sorted(positions), "放松顺序应为风格、行业与市值、换手"


def test_portfolio_frozen_positions_explain_execution_delay(page: Page) -> None:
    text = compact(page.text_of("portfolio"))
    for item in ("停牌", "涨跌停", "递延", "可成交日", "既有仓位维持", "损益照常计入"):
        assert item in text, f"冻结持仓说明缺少 {item}"
    assert "ST不做任何特殊处理" in text, "应写明 ST 不特殊处理"


def test_portfolio_lot_rules_cover_every_board(page: Page) -> None:
    text = compact(page.text_of("portfolio"))
    for item in ("主板", "创业板", "100股", "科创板", "200股", "北交所", "1股递增", "零股"):
        assert item in text, f"整手说明缺少 {item}"


def test_production_flow_lists_five_steps_in_order(page: Page) -> None:
    assert page.has_id("production-flow"), "缺少 #production-flow 流程条"
    text = compact(page.text_of("production-flow"))
    positions = [text.find(step) for step in PRODUCTION_FLOW]
    for step, position in zip(PRODUCTION_FLOW, positions):
        assert position >= 0, f"流程条缺少 {step}"
    assert positions == sorted(positions), "流程应为数据增量、因子重算、信号合成、组合优化、T+1 调仓单"


def test_production_flow_runs_through_manual_following(page: Page) -> None:
    text = compact(page.text_of("production"))
    assert len(text) >= 300, "#production 内容过少"
    for item in ("人工跟单", "回录成交", "简报", "T+1"):
        assert item in text, f"生产闭环缺少 {item}"
    assert "runs/orders" in text and "runs/reports" in text, "应写明产物落盘位置"


def test_production_flow_declares_layout_hook(page: Page) -> None:
    flows = [
        element
        for element in page.find("ol")
        if element.attrs.get("id") == "production-flow"
    ]
    assert len(flows) == 1, "流程条应为单一一组有序步骤"
    assert flows[0].attrs.get("data-flow") == "production", "流程条缺少可测试的布局钩子"
    steps = [
        element
        for element in page.find("li")
        if "production-flow" in element.scopes
    ]
    assert len(steps) == 5, "流程条应恰好有五步"


def test_production_flow_stacks_on_phone_without_overflow() -> None:
    css = SITE_CSS.read_text(encoding="utf-8")
    selector = '[data-flow="production"]'
    assert selector in css, "样式缺少流程条布局钩子"

    bodies: list[str] = []
    for match in re.finditer(re.escape(selector), css):
        brace = css.find("{", match.end())
        if brace != -1:
            bodies.append(_css_block(css, brace))
    assert bodies, "流程条样式没有规则体"
    assert any("minmax(0, 1fr)" in body or "flex-wrap: wrap" in body for body in bodies), (
        "流程条应可换行或自动分列"
    )
    assert not any("nowrap" in body for body in bodies), "流程条不应禁止换行"

    def max_width(condition: str) -> int:
        match = re.search(r"max-width:\s*(\d+)px", condition)
        return int(match.group(1)) if match else 0

    stacked = [
        body
        for condition, body in _media_blocks(css)
        if max_width(condition) >= 390
        and selector in body
        and (
            "grid-template-columns: minmax(0, 1fr)" in body
            or "grid-template-columns:minmax(0,1fr)" in body
            or "flex-direction: column" in body
        )
    ]
    assert stacked, "390 宽度下流程条应纵向排列或换行"
    for body in stacked:
        assert "width:" not in body or "min-width: 0" in body, "窄屏规则不应写死宽度"


# ---------------------------------------------------------------------------
# 证据区与指标资产（#157）
# ---------------------------------------------------------------------------

FLAGSHIP_DISPLAY = {
    "flagship.start": "2025-01-02",
    "flagship.end": "2026-09-28",
    "flagship.benchmark": "000300",
    "flagship.rebalance": "周频",
    "flagship.initial_cash": "10 万",
    "flagship.total_return": "13.66%",
    "flagship.excess_return": "0.04%",
    "flagship.max_drawdown": "-23.58%",
    "flagship.sharpe": "0.41",
    "flagship.trades": "950",
    "flagship.rejects": "1",
    "flagship.cost_ratio": "10.0%",
}

ZZ1000_DISPLAY = {
    "zz1000.start": "2025-01-02",
    "zz1000.end": "2026-09-28",
    "zz1000.benchmark": "000852",
    "zz1000.rebalance": "月频",
    "zz1000.initial_cash": "100 万",
    "zz1000.total_return": "33.98%",
    "zz1000.excess_return": "6.48%",
    "zz1000.max_drawdown": "-12.34%",
    "zz1000.sharpe": "1.11",
}

DISCLAIMER = "历史回测，非实盘，不构成投资建议"
ZZ1000_NOTE = "多次实验中的一个配置，不是旗舰策略"


def load_metrics() -> dict:
    assert METRICS.exists(), "缺少 site/assets/metrics.json"
    return json.loads(METRICS.read_text(encoding="utf-8"))


def metric_values(source: str) -> dict[str, str]:
    """按 data-metric 取元素自身的文本。"""
    pairs = re.findall(
        r'<([a-z0-9]+)([^>]*\sdata-metric="([^"]+)"[^>]*)>([^<]*)</\1>',
        source,
        flags=re.IGNORECASE,
    )
    return {key: " ".join(value.split()) for _, _, key, value in pairs}


def test_evidence_sits_between_production_and_author(page: Page) -> None:
    ids = [element.attrs.get("id") for element in page.elements]
    for ident in ("production", "evidence", "author"):
        assert ident in ids, f"缺少 #{ident}"
    assert ids.index("production") < ids.index("evidence") < ids.index("author")


def test_evidence_is_collapsed_by_default(page: Page) -> None:
    details = [element for element in page.find("details") if "evidence" in element.scopes]
    assert len(details) == 1, "证据区应使用一个原生 details"
    assert "open" not in details[0].attrs, "证据区默认应收起"
    assert details[0].attrs.get("data-evidence") == "backtest", "details 缺少测试钩子"

    summaries = [
        element for element in page.find("summary") if "evidence" in element.scopes
    ]
    assert len(summaries) == 1, "details 需要一个 summary"
    summary_text = compact(page.text)
    assert "查看回测证据" in summary_text, "summary 文案应说明展开内容"


def test_evidence_has_two_sample_cards(page: Page) -> None:
    cards = [
        element
        for element in page.find("article")
        if "evidence" in element.scopes and element.attrs.get("id")
    ]
    ids = [element.attrs.get("id") for element in cards]
    assert ids == ["evidence-flagship", "evidence-zz1000"], "应有两张样本卡片"


def test_flagship_card_shows_config_and_metrics(page: Page) -> None:
    text = compact(page.text_of("evidence-flagship"))
    assert "旗舰10万样本" in text
    for key, expected in FLAGSHIP_DISPLAY.items():
        assert compact(expected) in text, f"旗舰卡片缺少 {key} 的展示值 {expected}"
    for field in ("区间", "基准", "调仓", "初始资金", "总收益", "超额", "最大回撤", "夏普"):
        assert field in text, f"旗舰卡片缺少字段 {field}"
    assert "950笔" in text and "1笔" in text, "旗舰卡片应展示成交与拒单笔数"
    assert "占初始资金" in text, "成本应注明口径"


def test_zz1000_card_marks_experiment_not_flagship(page: Page) -> None:
    text = compact(page.text_of("evidence-zz1000"))
    assert "中证1000指数增强月度样本" in text
    for key, expected in ZZ1000_DISPLAY.items():
        assert compact(expected) in text, f"指增卡片缺少 {key} 的展示值 {expected}"
    assert ZZ1000_NOTE in text, "指增样本必须标注多次实验之一"
    assert "旗舰策略" in text, "指增标注应说明不是旗舰策略"


def test_both_cards_carry_the_disclaimer(page: Page) -> None:
    for ident in ("evidence-flagship", "evidence-zz1000"):
        assert DISCLAIMER in compact(page.text_of(ident)), f"#{ident} 缺少免责声明"


def test_every_data_metric_matches_metrics_json(page: Page) -> None:
    metrics = load_metrics()
    flat = flatten_metrics(metrics)
    values = metric_values(HOME.read_text(encoding="utf-8"))

    assert values, "页面没有 data-metric 标注"
    missing = sorted(set(values) - set(flat))
    assert not missing, f"页面标注了 metrics 未提供的键：{missing}"
    for key, text in values.items():
        assert text == format_metric(key, flat[key]), f"{key} 的页面文本与 metrics 不一致"


def test_evidence_metric_text_carries_locked_values(page: Page) -> None:
    values = metric_values(HOME.read_text(encoding="utf-8"))

    expected = {**FLAGSHIP_DISPLAY, **ZZ1000_DISPLAY}
    for key, display in expected.items():
        assert values.get(key) == display, f"{key} 的展示值应为 {display}"


def test_factor_counts_are_metric_driven(page: Page) -> None:
    metrics = load_metrics()
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    factors = registry["factors"]
    counts = {
        "total": len(factors),
        "seed": sum(1 for factor in factors if factor["lineage"]["op"] == "seed"),
        "mutation": sum(1 for factor in factors if factor["lineage"]["op"] == "mutation"),
        "crossover": sum(1 for factor in factors if factor["lineage"]["op"] == "crossover"),
    }

    assert metrics["factors"] == counts, "metrics 因子计数与 registry 不一致"
    values = metric_values(HOME.read_text(encoding="utf-8"))
    for key, expected in counts.items():
        assert values.get(f"factors.{key}") == str(expected)


def test_metrics_samples_keep_backtest_values() -> None:
    metrics = load_metrics()
    flagship = metrics["samples"]["flagship"]
    zz1000 = metrics["samples"]["zz1000"]

    assert flagship["total_return"] == pytest.approx(0.136633, abs=1e-6)
    assert flagship["excess_return"] == pytest.approx(0.000376, abs=1e-6)
    assert flagship["max_drawdown"] == pytest.approx(-0.235810, abs=1e-6)
    assert flagship["sharpe"] == pytest.approx(0.4138, abs=1e-4)
    assert flagship["cost_ratio"] == pytest.approx(0.100159, abs=1e-6)
    assert (flagship["trades"], flagship["rejects"]) == (950, 1)
    assert zz1000["total_return"] == pytest.approx(0.339833, abs=1e-6)
    assert zz1000["excess_return"] == pytest.approx(0.064841, abs=1e-6)
    assert zz1000["max_drawdown"] == pytest.approx(-0.123360, abs=1e-6)
    assert zz1000["sharpe"] == pytest.approx(1.1116, abs=1e-4)


def test_evidence_uses_local_nav_svgs(page: Page) -> None:
    images = [
        element for element in page.find("img") if "evidence" in element.scopes
    ]
    assert len(images) == 2, "两张卡片各需要一张净值图"
    assert [image.attrs.get("src") for image in images] == [
        "assets/nav-flagship.svg",
        "assets/nav-zz1000.svg",
    ]
    for image in images:
        assert image.attrs.get("alt", "").strip(), "净值图缺少 alt"


def test_evidence_cards_stack_on_narrow_screens() -> None:
    css = SITE_CSS.read_text(encoding="utf-8")
    selector = ".evidence__grid"
    assert selector in css, "样式缺少证据卡片布局"

    bodies: list[str] = []
    for match in re.finditer(re.escape(selector), css):
        brace = css.find("{", match.end())
        if brace != -1:
            bodies.append(_css_block(css, brace))
    assert any("repeat(2, minmax(0, 1fr))" in body for body in bodies), "桌面应两栏"

    stacked = [
        body
        for condition, body in _media_blocks(css)
        if re.search(r"max-width:\s*768px", condition)
        and selector in body
        and "minmax(0, 1fr)" in body
    ]
    assert stacked, "768 以下证据卡片应单列"


# ---------------------------------------------------------------------------
# 幻灯片入口与跨端收口（#162）
# ---------------------------------------------------------------------------

SITE_JS = SITE / "assets" / "site.js"
DECK = SITE / "slides" / "index.html"
DECK_CSS = SITE / "slides" / "slides.css"

SLIDE_TITLES = (
    "封面",
    "个人简介",
    "AI 学习与技术积累",
    "项目总览",
    "系统架构",
    "Agent 因子挖掘",
    "因子库与门控",
    "AutoML 与 AutoResearch",
    "组合优化与生产闭环",
    "创新点与边界",
    "未来规划",
)


def slides_section_source() -> str:
    """截取主页 #slides 到 #author 之间的标记，避免命中其他栏目的同名文案。"""
    source = HOME.read_text(encoding="utf-8")
    start = source.index('id="slides"')
    end = source.index('id="author"')
    assert start < end, "幻灯片入口应位于作者区之前"
    return source[start:end]


def strip_tags(markup: str) -> str:
    return compact(re.sub(r"<[^>]+>", "", markup))


def test_slides_section_sits_between_evidence_and_author(page: Page) -> None:
    ids = [element.attrs.get("id") for element in page.elements]
    for ident in ("evidence", "slides", "author"):
        assert ident in ids, f"缺少 #{ident}"
    assert ids.index("evidence") < ids.index("slides") < ids.index("author"), (
        "栏目顺序应为证据、幻灯片入口、作者"
    )


def test_slides_catalog_lists_eleven_pages_in_order() -> None:
    section = slides_section_source()
    entries = re.findall(
        r'<li[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        section,
        re.S,
    )
    assert len(entries) == 11, f"目录应有 11 项，实际 {len(entries)}"
    assert [href for href, _ in entries] == [f"slides/#{number}" for number in range(1, 12)]
    for (href, inner), title in zip(entries, SLIDE_TITLES):
        assert compact(title) in strip_tags(inner), f"{href} 的标题应为 {title}"


def test_slides_section_offers_primary_button() -> None:
    section = slides_section_source()
    buttons = [
        strip_tags(inner)
        for inner in re.findall(r'<a[^>]*href="slides/"[^>]*>(.*?)</a>', section, re.S)
    ]
    assert "打开答辩幻灯片" in buttons, "缺少进入 deck 的主按钮"


def test_slides_cover_preview_replicates_first_slide() -> None:
    section = slides_section_source()
    previews = re.findall(r'<a[^>]*href="slides/"[^>]*>(.*?)</a>', section, re.S)
    preview = next((inner for inner in previews if "ASTER" in inner), None)
    assert preview is not None, "封面预览应整体链到 slides/"
    text = strip_tags(preview)
    for item in (
        "ASTER",
        "陶唐",
        "上海财经大学",
        "信息管理与工程学院",
        "数据科学与大数据技术",
        "2024级",
    ):
        assert item in text, f"封面预览缺少 {item}"
    assert compact(TITLE) in text, "封面预览应含模板题名"


def test_home_embeds_no_iframe(page: Page) -> None:
    assert not page.find("iframe"), "主页不得嵌 iframe"
    assert "iframe" not in slides_section_source().lower()


def test_navigation_links_to_slides(page: Page) -> None:
    hrefs = [element.attrs.get("href") for element in page.find("a") if "top" in element.scopes]
    assert "#slides" in hrefs, "导航缺少 #slides 入口"
    assert page.has_id("slides")


def test_home_print_expands_evidence_and_restores_it() -> None:
    script = SITE_JS.read_text(encoding="utf-8")
    assert re.search(r'addEventListener\(\s*"beforeprint"', script), "缺少 beforeprint 钩子"
    assert re.search(r'addEventListener\(\s*"afterprint"', script), "缺少 afterprint 钩子"
    assert "data-evidence" in script, "打印钩子应定位证据 details"
    assert re.search(r"\.open\s*=", script), "打印钩子应设置 details.open"
    assert "open = true" in script, "打印时应展开证据区"


def test_home_print_rules_hide_chrome_without_overflow() -> None:
    css = SITE_CSS.read_text(encoding="utf-8")
    assert "@media print" in css
    print_block = css[css.index("@media print") :]
    assert "display: none" in print_block
    assert "button" in print_block, "打印时应隐藏按钮"
    assert "nav" in print_block, "打印时应隐藏导航"
    assert "slide" in print_block, "打印时应重排幻灯片入口"
    assert "minmax(0, 1fr)" in print_block, "打印时入口应单列，避免横向溢出"


def test_reduced_motion_switches_off_animation_and_smooth_scroll() -> None:
    for path in (SITE_CSS, DECK_CSS):
        css = path.read_text(encoding="utf-8")
        blocks = [
            body
            for condition, body in _media_blocks(css)
            if "prefers-reduced-motion" in condition
        ]
        assert blocks, f"{path.name} 缺少 reduced motion 规则"
        body = " ".join(blocks)
        assert "scroll-behavior: auto" in body, f"{path.name} 未关闭平滑滚动"
        assert re.search(r"animation:\s*none", body), f"{path.name} 未关闭动画"
        assert re.search(r"transition:\s*none", body), f"{path.name} 未关闭过渡"
        assert "smooth" not in body, f"{path.name} reduced motion 下仍有平滑滚动"


def test_slides_entry_stacks_on_phone() -> None:
    css = SITE_CSS.read_text(encoding="utf-8")

    def max_width(condition: str) -> int:
        match = re.search(r"max-width:\s*(\d+)px", condition)
        return int(match.group(1)) if match else 0

    stacked = [
        body
        for condition, body in _media_blocks(css)
        if max_width(condition) >= 768
        and "slide" in body
        and "grid-template-columns: minmax(0, 1fr)" in body
    ]
    assert stacked, "768 以下幻灯片入口应单列，避免横向溢出"


def test_deck_keeps_sixteen_by_nine_and_hints_rotation() -> None:
    css = DECK_CSS.read_text(encoding="utf-8")
    assert re.search(r"aspect-ratio:\s*16\s*/\s*9", css), "deck 应保持 16:9"
    narrow = [
        body
        for condition, body in _media_blocks(css)
        if re.search(r"max-width:\s*768px", condition)
    ]
    assert any("rotate-hint" in body and "display: block" in body for body in narrow), (
        "窄屏应提示横屏或全屏"
    )


def test_every_site_page_is_self_contained() -> None:
    pages = sorted(SITE.rglob("*.html"))
    assert pages, "站点没有任何 HTML"
    for path in pages:
        parser = Page()
        parser.feed(path.read_text(encoding="utf-8"))
        parser.close()
        assert not parser.find("iframe"), f"{path.name} 含 iframe"
        for tag in ("link", "script", "img", "source", "video", "audio"):
            for element in parser.find(tag):
                target = element.attrs.get("href") or element.attrs.get("src")
                if not target:
                    continue
                assert not re.match(r"^(?:[a-z]+:)?//", target), (
                    f"{path.name} 的 {tag} 请求外部资源 {target}"
                )
