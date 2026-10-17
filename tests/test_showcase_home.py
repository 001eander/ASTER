"""展示站主页的静态验收：标题、页首字段、导航、外链与资源边界，以及系统全景、Agent、因子库与 AutoML 各段。

测试只解析站点产物，不依赖 CSS 类名或 DOM 层级。
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
HOME = SITE / "index.html"
TOKENS = SITE / "assets" / "tokens.css"
SITE_CSS = SITE / "assets" / "site.css"
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
        ("color-hero-ink", "color-hero-bg"),
        ("color-hero-muted", "color-hero-bg"),
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
