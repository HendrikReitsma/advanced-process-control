const Plotly = globalThis.Plotly;
const MAX_TRAIL_POINTS = 60;
const FONT_FAMILY = "Courier New, Consolas, monospace";
const COLORS = {
  background: "#07110f",
  grid: "#294137",
  text: "#dce8df",
  safe: "#35d05b",
  approaching: "#ffbf2f",
  risk: "#ef4444",
  rh: "#3f8f89",
  enthalpy: "#756f99",
  saturation: "#54d9ff",
  trail: "#54d9ff",
};

function chartHeight(root) {
  return Number.parseFloat(
    getComputedStyle(root).getPropertyValue("--apc-map-height")
  );
}

function pointColor(status) {
  if (status === "STICKY RISK") return COLORS.risk;
  if (status === "APPROACHING") return COLORS.approaching;
  return COLORS.safe;
}

function referenceTrace(curve, color, dash) {
  return {
    x: curve.humidity,
    y: curve.temperature,
    mode: "lines",
    name: curve.label,
    line: { color, width: 1, dash },
    showlegend: false,
    hovertemplate: `${curve.label}<br>w=%{x:.4f}<br>T=%{y:.1f} C<extra></extra>`,
  };
}

function regionTrace(name, x, upper, lower, color) {
  return {
    x: [...x, ...[...x].reverse()],
    y: [...upper, ...[...lower].reverse()],
    mode: "lines",
    name,
    line: { width: 0 },
    fill: "toself",
    fillcolor: color,
    hoverinfo: "skip",
    showlegend: true,
  };
}

function staticTraces(background, snapshot, current) {
  const boundary = background.boundary;
  const safeBoundary = background.safe_boundary;
  const constraint = background.constraint;
  const x = boundary.humidity;
  const y = boundary.temperature;
  const safe = safeBoundary.temperature;
  const yMin = background.temperature_range[0];
  const yMax = background.temperature_range[1];
  const traces = [
    regionTrace(
      "SAFE",
      x,
      safe,
      x.map(() => yMin),
      "rgba(53,208,91,0.08)"
    ),
    regionTrace(
      "SAFETY OFFSET",
      x,
      y,
      safe,
      "rgba(255,191,47,0.16)"
    ),
    regionTrace(
      "STICKY RISK",
      x,
      x.map(() => yMax),
      y,
      "rgba(239,68,68,0.10)"
    ),
  ];

  background.relative_humidity_curves.forEach((curve) => {
    traces.push(referenceTrace(curve, COLORS.rh, "dot"));
  });
  background.enthalpy_curves.forEach((curve) => {
    traces.push(referenceTrace(curve, COLORS.enthalpy, "dash"));
  });
  traces.push(referenceTrace(background.saturation, COLORS.saturation, "solid"));
  traces.push({
    x,
    y,
    mode: "lines",
    name: boundary.label,
    line: { color: COLORS.approaching, width: 2 },
    hovertemplate: `${boundary.label}<br>w=%{x:.4f}<br>T=%{y:.1f} C<extra></extra>`,
  });
  traces.push({
    x: safeBoundary.humidity,
    y: safe,
    mode: "lines",
    name: safeBoundary.label,
    line: { color: COLORS.safe, width: 2, dash: "dash" },
    hovertemplate: `${safeBoundary.label}<br>w=%{x:.4f}<br>T=%{y:.1f} C<extra></extra>`,
  });
  traces.push({
    x: background.humidity_range,
    y: [constraint.maximum_exhaust_temperature, constraint.maximum_exhaust_temperature],
    mode: "lines",
    name: "Maximum exhaust temperature",
    line: { color: COLORS.risk, width: 1.5, dash: "dot" },
    hovertemplate: `Maximum exhaust temperature: ${constraint.maximum_exhaust_temperature.toFixed(1)} C<extra></extra>`,
  });
  traces.push({
    x: [constraint.maximum_humidity, constraint.maximum_humidity],
    y: background.temperature_range,
    mode: "lines",
    name: "Derived humidity maximum",
    line: { color: COLORS.risk, width: 1.5, dash: "dash" },
    hovertemplate: `Derived humidity maximum: ${constraint.maximum_humidity.toFixed(4)} kg/kg dry air<extra></extra>`,
  });
  traces.push({
    x: [constraint.maximum_humidity],
    y: [constraint.maximum_exhaust_temperature],
    mode: "markers",
    name: "Safe-limit intersection",
    marker: { color: COLORS.risk, size: 9, symbol: "diamond" },
    hovertemplate: `Safe-limit intersection<br>w=${constraint.maximum_humidity.toFixed(4)} kg/kg dry air<br>T=${constraint.maximum_exhaust_temperature.toFixed(1)} C<extra></extra>`,
  });
  traces.push({
    x: [null],
    y: [null],
    mode: "lines",
    name: "RH: 10 / 20 / 30 / 50%",
    line: { color: COLORS.rh, width: 1, dash: "dot" },
    hoverinfo: "skip",
  });
  traces.push({
    x: [null],
    y: [null],
    mode: "lines",
    name: "h: 250 / 350 / 450 / 550 kJ/kg",
    line: { color: COLORS.enthalpy, width: 1, dash: "dash" },
    hoverinfo: "skip",
  });

  const trailIndex = traces.length;
  traces.push({
    x: snapshot.map((sample) => sample.humidity),
    y: snapshot.map((sample) => sample.temperature),
    mode: "lines+markers",
    name: "Recent operating trail",
    line: { color: COLORS.trail, width: 1.5 },
    marker: { color: COLORS.trail, size: 4 },
    hoverinfo: "skip",
  });
  const pointIndex = traces.length;
  traces.push({
    x: [current.humidity],
    y: [current.temperature],
    mode: "markers",
    name: "Current exhaust condition",
    marker: {
      color: pointColor(current.status),
      size: 13,
      line: { color: "#ffffff", width: 1.5 },
    },
    text: [current.tooltip],
    hovertemplate: "%{text}<extra></extra>",
  });
  return { traces, trailIndex, pointIndex };
}

function makeLayout(background, runId, width, height) {
  return {
    autosize: false,
    width,
    height,
    margin: { l: 92, r: 24, t: 32, b: 68 },
    paper_bgcolor: COLORS.background,
    plot_bgcolor: COLORS.background,
    font: { family: FONT_FAMILY, color: COLORS.text, size: 12 },
    legend: {
      orientation: "h",
      x: 0,
      y: 1.08,
      font: { family: FONT_FAMILY, size: 10 },
      bgcolor: "rgba(0,0,0,0)",
    },
    annotations: [
      {
        x: background.constraint.maximum_humidity,
        y: background.constraint.maximum_exhaust_temperature,
        text: `w max ${background.constraint.maximum_humidity.toFixed(4)}`,
        showarrow: true,
        arrowhead: 2,
        ax: 48,
        ay: -28,
        font: { family: FONT_FAMILY, size: 10, color: COLORS.risk },
        bgcolor: "rgba(7,17,15,0.85)",
        bordercolor: COLORS.risk,
      },
    ],
    xaxis: {
      range: background.humidity_range,
      fixedrange: false,
      automargin: true,
      title: {
        text: "Exhaust-air humidity ratio (kg water/kg dry air)",
        standoff: 14,
      },
      tickformat: ".3f",
      gridcolor: COLORS.grid,
      zeroline: false,
    },
    yaxis: {
      range: background.temperature_range,
      fixedrange: false,
      automargin: true,
      title: {
        text: "Exhaust dry-bulb temperature (C)",
        standoff: 14,
      },
      gridcolor: COLORS.grid,
      zeroline: false,
    },
    uirevision: `apc-operating-map-${runId}`,
    hovermode: "closest",
  };
}

async function resetMap(state, payload) {
  const snapshot = payload.snapshot || [];
  const setup = staticTraces(payload.background, snapshot, payload.current);
  state.runId = payload.run_id;
  state.backgroundSignature = JSON.stringify(payload.background.constraint);
  state.processedSampleId = snapshot.reduce(
    (latest, sample) => Math.max(latest, sample.sample_id),
    -1
  );
  state.trailIndex = setup.trailIndex;
  state.pointIndex = setup.pointIndex;
  state.background = payload.background;
  const width = Math.max(state.root.getBoundingClientRect().width, 720);
  const height = chartHeight(state.root);
  const layout = makeLayout(payload.background, payload.run_id, width, height);
  const config = { displayModeBar: false, responsive: false, scrollZoom: false };

  if (state.mounted) {
    await Plotly.react(state.chart, setup.traces, layout, config);
  } else {
    await Plotly.newPlot(state.chart, setup.traces, layout, config);
    state.mounted = true;
  }
}

async function updatePoint(state, current) {
  await Plotly.restyle(
    state.chart,
    {
      x: [[current.humidity]],
      y: [[current.temperature]],
      text: [[current.tooltip]],
      "marker.color": [pointColor(current.status)],
    },
    [state.pointIndex]
  );
}

async function appendPayload(state, payload) {
  const sample = payload.sample;
  if (sample && sample.sample_id > state.processedSampleId) {
    await Plotly.extendTraces(
      state.chart,
      { x: [[sample.humidity]], y: [[sample.temperature]] },
      [state.trailIndex],
      MAX_TRAIL_POINTS
    );
    state.processedSampleId = sample.sample_id;
  }
  await updatePoint(state, payload.current);
}

export default function operatingMap(component) {
  const { data, parentElement } = component;
  const root = parentElement.querySelector(".apc-operating-map");
  let state = root.__apcOperatingMap;

  if (!state) {
    state = {
      root,
      chart: parentElement.querySelector("#apc-operating-map-chart"),
      runId: null,
      backgroundSignature: null,
      processedSampleId: -1,
      trailIndex: -1,
      pointIndex: -1,
      mounted: false,
      queue: Promise.resolve(),
      lastWidth: root.getBoundingClientRect().width,
      resizeObserver: null,
    };
    state.resizeObserver = new ResizeObserver(([entry]) => {
      const width = entry.contentRect.width;
      if (!state.mounted || width < 1 || Math.abs(width - state.lastWidth) < 1) {
        return;
      }
      state.lastWidth = width;
      Plotly.relayout(state.chart, { width, height: chartHeight(state.root) });
    });
    state.resizeObserver.observe(root);
    root.__apcOperatingMap = state;
  }

  const backgroundSignature = JSON.stringify(data.background.constraint);
  state.queue = state.queue.then(() =>
    state.runId !== data.run_id || state.backgroundSignature !== backgroundSignature
      ? resetMap(state, data)
      : appendPayload(state, data)
  );

  return () => {
    state.resizeObserver?.disconnect();
    if (state.mounted) Plotly.purge(state.chart);
    delete root.__apcOperatingMap;
  };
}
