"""Small reusable presentation helpers for the compact SCADA dashboard."""

from dataclasses import dataclass
from html import escape

import streamlit as st


SCADA_CSS = """
<style>
:root { --scada-grey: #c8c8c8; --scada-blue: #17365d; --scada-screen: #07110f; --scada-green: #39864a; --scada-amber: #d99a16; --scada-red: #c83c3c; }
html, body, [class*="css"], [data-testid="stAppViewContainer"] { font-family: Tahoma, "Segoe UI", Arial, sans-serif; }
[data-testid="stAppViewContainer"], [data-testid="stSidebar"] { background: var(--scada-grey); color: #111; }
[data-testid="stMainBlockContainer"] { margin-left: auto; margin-right: auto; max-width: 1200px; padding: 0.55rem 0.8rem 1rem; width: 100%; }
[data-testid="stHeader"] { background: transparent; height: 2.7rem; min-height: 2.7rem; pointer-events: none; }
[data-testid="stToolbar"] { background: transparent; pointer-events: none; visibility: visible; }
[data-testid="stExpandSidebarButton"] { background: #e8e8e8 !important; border: 1px solid #737373 !important; border-radius: 0 !important; color: #111 !important; margin-left: 0.45rem; min-height: 1.95rem; min-width: 2.15rem; pointer-events: auto; visibility: visible !important; }
[data-testid="stAppViewContainer"]:has([data-testid="stExpandSidebarButton"]) [data-testid="stMainBlockContainer"] { padding-top: 3.25rem; }
[data-testid="stSidebar"] { border-right: 1px solid #737373; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span, [data-testid="stSidebar"] [data-baseweb="select"] * { color: #111 !important; }
[data-testid="stSidebar"] [data-baseweb="select"] svg { fill: #111 !important; }
[data-testid="stSidebarUserContent"] { padding-top: 0.7rem; }
#MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden; }

.scada-titlebar, .scada-sidebar-title, .scada-section-title { background: var(--scada-blue); border: 1px solid #0c243f; color: #fff; font-weight: 700; }
.scada-titlebar { margin-bottom: 0.3rem; padding: 0.36rem 0.55rem; }
.scada-titlebar h1 { color: #fff; font-size: 1.25rem; letter-spacing: 0.01em; line-height: 1.1; margin: 0; }
.scada-titlebar p { color: #e8edf5; font-size: 0.78rem; margin: 0.14rem 0 0; }
.scada-sidebar-title, .scada-section-title { letter-spacing: 0.015em; margin: 0.25rem 0; padding: 0.22rem 0.42rem; }
.scada-section-title small { color: #e8edf5; float: right; font-weight: 400; }

.scada-status-strip { display: grid; gap: 0; grid-template-columns: repeat(5, minmax(0, 1fr)); margin: 0.25rem 0 0.35rem; }
.scada-status-item, .scada-value-card, .scada-message { background: #f5f5f5; border: 1px solid #8b8b8b; }
.scada-status-item { font-size: 0.74rem; font-weight: 700; padding: 0.22rem 0.38rem; }
.scada-lamp { border: 1px solid #333; border-radius: 50%; display: inline-block; height: 0.6rem; margin-right: 0.3rem; vertical-align: -0.05rem; width: 0.6rem; }
.state-normal .scada-lamp { background: var(--scada-green); }.state-warning .scada-lamp { background: var(--scada-amber); }.state-alarm .scada-lamp { background: var(--scada-red); }.state-neutral .scada-lamp { background: #6c7a89; }

.scada-value-grid { display: grid; gap: 0.32rem; margin-bottom: 0.4rem; }.scada-value-card { min-width: 0; padding: 0.3rem 0.4rem; }.scada-value-card.state-warning { background: #fff0cc; }.scada-value-card.state-alarm { background: #ffd8d8; }
.scada-tag, .scada-table-tag { color: #555; font: 700 0.68rem "Courier New", monospace; letter-spacing: 0.04em; }.scada-label { font-size: 0.75rem; font-weight: 700; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.scada-reading { background: #fff; border: 1px solid #9a9a9a; color: #111; font-size: 1rem; font-weight: 700; margin-top: 0.18rem; padding: 0.18rem 0.28rem; white-space: nowrap; }.scada-unit { color: #444; font-size: 0.68rem; font-weight: 400; }
.scada-message { font-size: 0.76rem; margin: 0.25rem 0 0.35rem; padding: 0.28rem 0.4rem; }.scada-message strong { color: var(--scada-blue); }
.scada-event-banner, .scada-showcase { background: #fff3d4; border: 1px solid #b98012; margin: 0.25rem 0 0.35rem; padding: 0.28rem 0.4rem; }.scada-event-banner { font-size: 0.76rem; font-weight: 700; }.scada-event-banner.is-idle { display: none; }.scada-showcase-head { display: flex; font-size: 0.82rem; font-weight: 700; gap: 0.6rem; justify-content: space-between; }.scada-showcase-detail { font-size: 0.73rem; margin-top: 0.2rem; }.scada-showcase-progress { background: #e5e5e5; border: 1px solid #8b8b8b; height: 0.6rem; margin-top: 0.25rem; }.scada-showcase-progress span { background: var(--scada-blue); display: block; height: 100%; }

.stButton > button, .stDownloadButton > button { background: #e8e8e8; border: 1px solid #737373; border-radius: 0; box-shadow: none; color: #111; font-family: Tahoma, "Segoe UI", Arial, sans-serif; font-weight: 700; min-height: 1.85rem; padding: 0.18rem 0.65rem; }.stButton > button:active, .stDownloadButton > button:active { transform: translate(1px, 1px); }[data-testid="stSidebar"] .stButton > button[kind="primary"] { background: #fff3d4; border-color: #b98012; }
[data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input { background: #fff; border-radius: 0 !important; border-color: #8b8b8b !important; }
[data-testid="stExpander"] details { background: #eee; border: 1px solid #8b8b8b; border-radius: 0; }[data-testid="stExpander"] summary { font-family: Tahoma, "Segoe UI", Arial, sans-serif; font-weight: 700; }
[data-testid="stPlotlyChart"] { background: var(--scada-screen); border: 1px solid #555; padding: 0.1rem; }

.scada-parameter-table { background: #fff; border: 1px solid #8b8b8b; border-collapse: collapse; font-size: 0.78rem; margin: 0 0 0.3rem; table-layout: fixed; width: 100%; }.scada-parameter-table th { background: #e5e5e5; border: 1px solid #a6a6a6; color: #111; font-weight: 700; padding: 0.18rem 0.3rem; text-align: left; white-space: nowrap; }.scada-parameter-table td { background: #fff; border: 1px solid #c4c4c4; color: #111; padding: 0.19rem 0.3rem; vertical-align: middle; }.scada-table-controlled th:nth-child(1) { width: 30%; }.scada-table-controlled th:nth-child(2) { width: 13%; }.scada-table-controlled th:nth-child(3), .scada-table-controlled th:nth-child(4), .scada-table-controlled th:nth-child(5) { width: 12%; }.scada-table-controlled th:nth-child(6) { width: 21%; }.scada-table-manipulated th:nth-child(1) { width: 34%; }.scada-table-manipulated th:nth-child(2) { width: 17%; }.scada-table-manipulated th:nth-child(3), .scada-table-manipulated th:nth-child(4) { width: 15%; }.scada-table-manipulated th:nth-child(5) { width: 19%; }.scada-table-disturbances th:nth-child(1) { width: 30%; }.scada-table-disturbances th:nth-child(2) { width: 20%; }.scada-table-disturbances th:nth-child(3) { width: 32%; }.scada-table-disturbances th:nth-child(4) { width: 18%; }.scada-parameter-table .scada-table-current { font-weight: 700; }.scada-parameter-table .scada-table-current.state-warning { background: #fff0cc; }.scada-parameter-table .scada-table-current.state-alarm { background: #ffd8d8; }.scada-table-tag { margin-right: 0.32rem; }
@media (max-width: 900px) { .scada-status-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }.scada-parameter-table { font-size: 0.72rem; } }
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


def render_parameter_table(
    headers: list[str],
    rows: list[list[str]],
    current_states: list[str] | None = None,
    current_column: int = 1,
) -> None:
    """Render a compact SCADA parameter table with optional current-value states."""

    header_cells = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body_rows = []
    for row_index, row in enumerate(rows):
        state = (current_states or ["neutral"] * len(rows))[row_index]
        cells = []
        for column_index, value in enumerate(row):
            classes = ""
            if column_index == current_column:
                classes = f' class="scada-table-current state-{escape(state)}"'
            cells.append(f"<td{classes}>{escape(value)}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    table_kind = {6: "controlled", 5: "manipulated", 4: "disturbances"}.get(
        len(headers), ""
    )
    st.markdown(
        f'<table class="scada-parameter-table scada-table-{table_kind}"><thead><tr>'
        f"{header_cells}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>",
        unsafe_allow_html=True,
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
