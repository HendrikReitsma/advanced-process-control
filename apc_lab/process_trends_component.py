"""Persistent client-side Process Trends component and payload helpers."""

from importlib.resources import files
from typing import Any

from streamlit.components.v2 import component


_PROCESS_TRENDS_HTML = """
<div class="apc-process-trends">
  <div id="apc-input-trends" class="apc-trend-chart"></div>
  <div id="apc-output-trends" class="apc-trend-chart"></div>
</div>
"""

_MAX_TREND_ROWS = 5
_TREND_ROW_HEIGHT = 88
_TREND_VERTICAL_CHROME = 80
_PROCESS_TRENDS_HEIGHT = (
    _MAX_TREND_ROWS * _TREND_ROW_HEIGHT + _TREND_VERTICAL_CHROME
)

_PROCESS_TRENDS_CSS = """
.apc-process-trends {
    --apc-chart-height: __PROCESS_TRENDS_HEIGHT__px;
    display: grid;
    gap: 1.5rem;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    width: 100%;
    height: var(--apc-chart-height);
    min-height: var(--apc-chart-height);
    align-items: stretch;
}
.apc-trend-chart {
    min-width: 0;
    width: 100%;
    height: var(--apc-chart-height);
    min-height: var(--apc-chart-height);
    background: #07110f;
}
"""
_PROCESS_TRENDS_CSS = _PROCESS_TRENDS_CSS.replace(
    "__PROCESS_TRENDS_HEIGHT__", str(_PROCESS_TRENDS_HEIGHT)
)

_ASSET_ROOT = files("apc_lab").joinpath("components", "process_trends")
_PROCESS_TRENDS_JS = (
    _ASSET_ROOT.joinpath("plotly-3.5.0.min.js").read_text(encoding="utf-8")
    + "\n"
    + _ASSET_ROOT.joinpath("component.js").read_text(encoding="utf-8")
)

def _register_process_trends():
    return component(
        "apc_process_trends",
        html=_PROCESS_TRENDS_HTML,
        css=_PROCESS_TRENDS_CSS,
        js=_PROCESS_TRENDS_JS,
        isolate_styles=False,
    )


_process_trends = _register_process_trends()


def prepare_trend_payload(
    *,
    run_id: int,
    sample_id: int,
    sample: dict[str, Any] | None,
    events: list[dict[str, Any]],
    last_sample_id: int,
    last_event_id: int,
    snapshot: list[dict[str, Any]] | None,
    config: dict[str, Any],
    predictions: dict[str, Any],
) -> tuple[dict[str, Any], int, int]:
    """Filter already-sent items and assemble one component update."""

    new_sample = (
        sample
        if sample is not None and int(sample["sample_id"]) > last_sample_id
        else None
    )
    new_events = [
        event for event in events if int(event["event_id"]) > last_event_id
    ]
    next_sample_id = (
        int(new_sample["sample_id"]) if new_sample is not None else last_sample_id
    )
    next_event_id = max(
        (int(event["event_id"]) for event in new_events),
        default=last_event_id,
    )
    payload = {
        "run_id": run_id,
        "sample_id": sample_id,
        "sample": new_sample,
        "events": new_events,
        "snapshot": snapshot,
        "config": config,
        "predictions": predictions,
    }
    return payload, next_sample_id, next_event_id


def render_process_trends(payload: dict[str, Any]) -> None:
    """Mount or update the persistent client-side Process Trends component."""

    global _process_trends

    mount_args = {
        "key": "live_process_trends_v3",
        "data": payload,
        "width": "stretch",
        "height": _PROCESS_TRENDS_HEIGHT,
    }
    try:
        _process_trends(**mount_args)
    except ValueError as exc:
        # AppTest can replace Streamlit's runtime registry while retaining this
        # imported module. Register against the new runtime and mount once more.
        if "is not registered" not in str(exc):
            raise
        _process_trends = _register_process_trends()
        _process_trends(**mount_args)
