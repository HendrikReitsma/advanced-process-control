"""Persistent client-side Mollier / stickiness operating-map component."""

from importlib.resources import files
from typing import Any

from streamlit.components.v2 import component


_OPERATING_MAP_HEIGHT = 480
_OPERATING_MAP_HTML = """
<div class="apc-operating-map">
  <div id="apc-operating-map-chart" class="apc-operating-map-chart"></div>
</div>
"""
_OPERATING_MAP_CSS = f"""
.apc-operating-map {{
    --apc-map-height: {_OPERATING_MAP_HEIGHT}px;
    width: 100%;
    height: var(--apc-map-height);
    min-height: var(--apc-map-height);
}}
.apc-operating-map-chart {{
    box-sizing: border-box;
    width: 100%;
    height: var(--apc-map-height);
    min-height: var(--apc-map-height);
    background: #07110f;
    border: 3px solid;
    border-color: #404040 #f0f0f0 #f0f0f0 #404040;
}}
"""

_ASSET_ROOT = files("apc_lab").joinpath("components")
_OPERATING_MAP_JS = (
    _ASSET_ROOT.joinpath("process_trends", "plotly-3.5.0.min.js").read_text(
        encoding="utf-8"
    )
    + "\n"
    + _ASSET_ROOT.joinpath("operating_map", "component.js").read_text(
        encoding="utf-8"
    )
)


def _register_operating_map():
    return component(
        "apc_operating_map",
        html=_OPERATING_MAP_HTML,
        css=_OPERATING_MAP_CSS,
        js=_OPERATING_MAP_JS,
        isolate_styles=False,
    )


# Register when Streamlit renders the map rather than while Python imports this
# module. This keeps Cloud's hot-reload import path free of component side effects.
_operating_map = None


def prepare_operating_map_payload(
    *,
    run_id: int,
    sample: dict[str, Any] | None,
    current: dict[str, Any],
    snapshot: list[dict[str, Any]],
    last_sample_id: int,
    background: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """Filter duplicate trail samples and assemble one map update."""

    new_sample = (
        sample
        if sample is not None and int(sample["sample_id"]) > last_sample_id
        else None
    )
    next_sample_id = (
        int(new_sample["sample_id"])
        if new_sample is not None
        else last_sample_id
    )
    return (
        {
            "run_id": run_id,
            "sample": new_sample,
            "current": current,
            "snapshot": snapshot,
            "background": background,
        },
        next_sample_id,
    )


def render_operating_map(payload: dict[str, Any]) -> None:
    """Mount or incrementally update the operating map."""

    global _operating_map

    mount_args = {
        "key": "live_operating_map_v1",
        "data": payload,
        "width": "stretch",
        "height": _OPERATING_MAP_HEIGHT,
    }
    try:
        if _operating_map is None:
            _operating_map = _register_operating_map()
        _operating_map(**mount_args)
    except ValueError as exc:
        if "is not registered" not in str(exc):
            raise
        _operating_map = _register_operating_map()
        _operating_map(**mount_args)
