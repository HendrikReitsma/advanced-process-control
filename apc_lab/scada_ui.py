"""Small reusable presentation helpers for the retro SCADA dashboard."""

from dataclasses import dataclass
from html import escape

import streamlit as st


SCADA_CSS = """
<style>
:root {
    --scada-grey: #c0c0c0;
    --scada-light: #ffffff;
    --scada-mid: #808080;
    --scada-dark: #404040;
    --scada-blue: #000080;
    --scada-screen: #07110f;
    --scada-green: #35d05b;
    --scada-amber: #ffbf2f;
    --scada-red: #ef3f3f;
}

html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: Tahoma, "MS Sans Serif", Arial, sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: var(--scada-grey);
    color: #000000;
}

[data-testid="stMainBlockContainer"] {
    max-width: 1550px;
    padding: 0.7rem 1rem 1.2rem;
}

[data-testid="stHeader"] {
    background: transparent;
    height: 2.7rem;
    min-height: 2.7rem;
    pointer-events: none;
}

[data-testid="stToolbar"] {
    background: transparent;
    pointer-events: none;
    visibility: visible;
}

[data-testid="stExpandSidebarButton"] {
    background: var(--scada-grey) !important;
    border: 2px solid !important;
    border-color: var(--scada-light) var(--scada-dark) var(--scada-dark) var(--scada-light) !important;
    border-radius: 0 !important;
    box-shadow: 1px 1px 0 #202020;
    color: #000000 !important;
    margin-left: 0.45rem;
    min-height: 1.95rem;
    min-width: 2.15rem;
    pointer-events: auto;
    visibility: visible !important;
}

[data-testid="stExpandSidebarButton"]:active {
    border-color: var(--scada-dark) var(--scada-light) var(--scada-light) var(--scada-dark) !important;
    box-shadow: none;
    transform: translate(1px, 1px);
}

[data-testid="stAppViewContainer"]:has([data-testid="stExpandSidebarButton"])
[data-testid="stMainBlockContainer"] {
    padding-top: 3.25rem;
}

[data-testid="stSidebar"] {
    background: var(--scada-grey);
    border-right: 3px solid;
    border-color: var(--scada-light) var(--scada-dark) var(--scada-dark) var(--scada-light);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #000000 !important;
}

[data-testid="stSidebarUserContent"] {
    padding-top: 0.7rem;
}

[data-testid="stSidebar"] [data-baseweb="select"] *,
[data-testid="stSidebar"] [data-baseweb="select"] svg {
    color: #000000 !important;
    fill: #000000 !important;
}

#MainMenu, footer, [data-testid="stDecoration"] {
    visibility: hidden;
}

.scada-titlebar {
    background: var(--scada-blue);
    color: #ffffff;
    border: 2px solid;
    border-color: #5f74bd #00003f #00003f #5f74bd;
    padding: 0.42rem 0.65rem;
    margin-bottom: 0.45rem;
    box-shadow: 2px 2px 0 #404040;
}

.scada-titlebar h1 {
    color: #ffffff;
    font-family: "Courier New", monospace;
    font-size: 1.42rem;
    letter-spacing: 0.04em;
    line-height: 1.1;
    margin: 0;
}

.scada-titlebar p {
    color: #e4e9ff;
    font-size: 0.78rem;
    margin: 0.16rem 0 0;
}

.scada-sidebar-title, .scada-section-title {
    background: var(--scada-blue);
    color: #ffffff;
    border: 2px solid;
    border-color: #5f74bd #00003f #00003f #5f74bd;
    font-family: "Courier New", monospace;
    font-weight: 700;
    letter-spacing: 0.035em;
    padding: 0.26rem 0.45rem;
    margin: 0.28rem 0 0.38rem;
}

.scada-section-title small {
    float: right;
    color: #dbe1ff;
    font-weight: 400;
}

.scada-status-strip {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.35rem;
    margin: 0.4rem 0 0.55rem;
}

.scada-status-item, .scada-value-card, .scada-message {
    background: var(--scada-grey);
    border: 2px solid;
    border-color: var(--scada-light) var(--scada-dark) var(--scada-dark) var(--scada-light);
    box-shadow: 1px 1px 0 #202020;
}

.scada-status-item {
    padding: 0.28rem 0.42rem;
    font: 700 0.76rem "Courier New", monospace;
}

.scada-lamp {
    display: inline-block;
    width: 0.64rem;
    height: 0.64rem;
    margin-right: 0.35rem;
    border: 1px solid #202020;
    border-radius: 50%;
    box-shadow: inset 1px 1px 0 rgba(255,255,255,0.55);
    vertical-align: -0.05rem;
}

.state-normal .scada-lamp { background: var(--scada-green); }
.state-warning .scada-lamp { background: var(--scada-amber); }
.state-alarm .scada-lamp { background: var(--scada-red); }
.state-neutral .scada-lamp { background: #5f85d9; }

.scada-value-grid {
    display: grid;
    gap: 0.42rem;
    margin-bottom: 0.58rem;
}

.scada-value-card {
    min-width: 0;
    padding: 0.38rem 0.48rem;
}

.scada-value-card.state-warning { background: #e8c96e; }
.scada-value-card.state-alarm { background: #e48282; }

.scada-tag {
    color: #303030;
    font: 700 0.68rem "Courier New", monospace;
    letter-spacing: 0.05em;
}

.scada-label {
    font-size: 0.75rem;
    font-weight: 700;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.scada-reading {
    background: #101813;
    border: 2px solid;
    border-color: #404040 #f2f2f2 #f2f2f2 #404040;
    color: #7dff8e;
    font: 700 1.18rem "Courier New", monospace;
    margin-top: 0.24rem;
    padding: 0.22rem 0.32rem;
    white-space: nowrap;
}

.scada-unit {
    color: #d5e3d8;
    font-size: 0.68rem;
    font-weight: 400;
}

.scada-message {
    font: 0.73rem "Courier New", monospace;
    margin: 0.35rem 0 0.5rem;
    padding: 0.38rem 0.48rem;
}

.scada-message strong { color: var(--scada-blue); }

.scada-event-banner {
    background: #e8c96e;
    border: 2px solid;
    border-color: #fff1b7 #705400 #705400 #fff1b7;
    box-shadow: 1px 1px 0 #202020;
    font: 700 0.75rem "Courier New", monospace;
    margin: 0.35rem 0 0.5rem;
    padding: 0.38rem 0.48rem;
}

.scada-event-banner.is-idle {
    visibility: hidden;
}

.scada-showcase {
    background: #e8c96e;
    border: 3px solid;
    border-color: #fff1b7 #705400 #705400 #fff1b7;
    box-shadow: 2px 2px 0 #404040;
    font-family: "Courier New", monospace;
    margin: 0.42rem 0 0.55rem;
    padding: 0.42rem 0.55rem;
}

.scada-showcase-head {
    display: flex;
    justify-content: space-between;
    gap: 0.6rem;
    font-size: 0.82rem;
    font-weight: 700;
}

.scada-showcase-detail {
    font-size: 0.73rem;
    margin-top: 0.2rem;
}

.scada-showcase-progress {
    background: #404040;
    border: 2px solid;
    border-color: #303030 #ffffff #ffffff #303030;
    height: 0.7rem;
    margin-top: 0.32rem;
    padding: 1px;
}

.scada-showcase-progress span {
    background: var(--scada-blue);
    display: block;
    height: 100%;
}

.stButton > button, .stDownloadButton > button {
    background: var(--scada-grey);
    border: 2px solid;
    border-color: var(--scada-light) var(--scada-dark) var(--scada-dark) var(--scada-light);
    border-radius: 0;
    box-shadow: 1px 1px 0 #202020;
    color: #000000;
    font-family: Tahoma, "MS Sans Serif", sans-serif;
    font-weight: 700;
    min-height: 1.95rem;
    padding: 0.2rem 0.7rem;
}

.stButton > button:active, .stDownloadButton > button:active {
    border-color: var(--scada-dark) var(--scada-light) var(--scada-light) var(--scada-dark);
    box-shadow: none;
    transform: translate(1px, 1px);
}

[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #e8c96e;
    border-color: #fff1b7 #705400 #705400 #fff1b7;
}

[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background: #ffffff;
    border-radius: 0 !important;
    border-color: #404040 #ffffff #ffffff #404040 !important;
}

[data-testid="stExpander"] details {
    background: #c8c8c8;
    border: 2px solid;
    border-color: var(--scada-light) var(--scada-dark) var(--scada-dark) var(--scada-light);
    border-radius: 0;
}

[data-testid="stExpander"] summary {
    font-family: "Courier New", monospace;
    font-weight: 700;
}

[data-testid="stPlotlyChart"] {
    background: var(--scada-screen);
    border: 3px solid;
    border-color: #404040 #f0f0f0 #f0f0f0 #404040;
    padding: 0.1rem;
}

@media (max-width: 900px) {
    .scada-status-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
</style>
"""


@dataclass(frozen=True)
class ScadaValue:
    tag: str
    label: str
    value: str
    unit: str
    state: str = "neutral"


def apply_scada_theme() -> None:
    st.markdown(SCADA_CSS, unsafe_allow_html=True)


def constraint_state(
    value: float,
    lower: float,
    upper: float,
    warning_fraction: float = 0.10,
) -> str:
    """Return normal, warning, or alarm for a bounded process value."""

    if value < lower or value > upper:
        return "alarm"
    warning_margin = max(upper - lower, 0.0) * warning_fraction
    if min(value - lower, upper - value) <= warning_margin:
        return "warning"
    return "normal"


def render_title_bar(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="scada-titlebar"><h1>{escape(title)}</h1>'
        f'<p>{escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def render_sidebar_title(title: str) -> None:
    st.markdown(
        f'<div class="scada-sidebar-title">{escape(title)}</div>',
        unsafe_allow_html=True,
    )


def render_section_title(title: str, detail: str = "") -> None:
    detail_html = f"<small>{escape(detail)}</small>" if detail else ""
    st.markdown(
        f'<div class="scada-section-title">{escape(title)}{detail_html}</div>',
        unsafe_allow_html=True,
    )


def render_status_strip(items: list[tuple[str, str, str]]) -> None:
    cells = "".join(
        f'<div class="scada-status-item state-{escape(state)}">'
        f'<span class="scada-lamp"></span>{escape(label)}: {escape(value)}</div>'
        for label, value, state in items
    )
    st.markdown(
        f'<div class="scada-status-strip">{cells}</div>', unsafe_allow_html=True
    )


def render_value_grid(values: list[ScadaValue], columns: int) -> None:
    cards = "".join(
        f'<div class="scada-value-card state-{escape(item.state)}">'
        f'<div class="scada-tag">{escape(item.tag)}</div>'
        f'<div class="scada-label">{escape(item.label)}</div>'
        f'<div class="scada-reading">{escape(item.value)} '
        f'<span class="scada-unit">{escape(item.unit)}</span></div></div>'
        for item in values
    )
    st.markdown(
        f'<div class="scada-value-grid" style="grid-template-columns:'
        f' repeat({columns}, minmax(0, 1fr));">{cards}</div>',
        unsafe_allow_html=True,
    )


def render_message(title: str, text: str) -> None:
    st.markdown(
        f'<div class="scada-message"><strong>{escape(title)}:</strong> '
        f'{escape(text)}</div>',
        unsafe_allow_html=True,
    )


def render_event_banner(text: str, active: bool = True) -> None:
    """Render an amber informational process-event banner."""

    state_class = "" if active else " is-idle"
    st.markdown(
        f'<div class="scada-event-banner{state_class}">EVENT: {escape(text)}</div>',
        unsafe_allow_html=True,
    )


def render_showcase_banner(
    phase: str,
    status: str,
    minute: int,
    description: str,
    next_action: str,
    progress: float,
) -> None:
    """Render the compact guided-scenario status above Process Trends."""

    progress_percent = min(max(float(progress) * 100.0, 0.0), 100.0)
    st.markdown(
        '<div class="scada-showcase">'
        '<div class="scada-showcase-head">'
        f'<span>{escape(phase)} | {escape(status)}</span>'
        f'<span>T+{int(minute):03d} MIN</span>'
        '</div>'
        f'<div class="scada-showcase-detail">{escape(description)} | '
        f'{escape(next_action)}</div>'
        '<div class="scada-showcase-progress">'
        f'<span style="width:{progress_percent:.1f}%"></span></div></div>',
        unsafe_allow_html=True,
    )
