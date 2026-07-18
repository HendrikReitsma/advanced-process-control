# Plotly components

The buildless Process Trends and Mollier / Stickiness Map Streamlit v2
components use the committed Plotly.js 3.5.0 production bundle. Install the
project normally with:

```powershell
python -m pip install -e ".[dev]"
```

No Node.js build is required. The Python wrappers load the component scripts
and pinned `plotly-3.5.0.min.js` bundle from package data, then register them
lazily with Streamlit's v2 component API. Process Trends appends live samples
with `Plotly.extendTraces`; RESET or a new run replaces the bounded snapshot
while preserving one stable chart mount during normal scans. The operating map
uses the same packaged bundle and updates its operating point and trail in
place.

The bundle is the one distributed with Plotly.py 6.7.0. Its license and
attribution are retained in `THIRD_PARTY_NOTICES.md` at the repository root.
