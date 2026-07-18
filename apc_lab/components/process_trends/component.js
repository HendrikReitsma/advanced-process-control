const Plotly = globalThis.Plotly;
const MAX_POINTS = 120;
const FONT_FAMILY = "Courier New, Consolas, monospace";
const FONT_SIZE = {
  base: 11,
  tick: 11,
  title: 12,
  subplot: 12,
  legend: 11,
  annotation: 11,
};
const COLORS = {
  background: "#07110f",
  grid: "#294137",
  text: "#dce8df",
  measuredInput: "#54d9ff",
  measuredOutput: "#35d05b",
  prediction: "#ffbf2f",
  limit: "#ef4444",
  target: "#54d9ff",
  event: "#ffbf2f",
};

function axisName(prefix, index) {
  return index === 0 ? prefix : `${prefix}${index + 1}`;
}

function axisLayoutName(prefix, index) {
  return index === 0 ? `${prefix}axis` : `${prefix}axis${index + 1}`;
}

function domains(count) {
  const gap = count > 3 ? 0.035 : 0.055;
  const height = (1 - gap * (count - 1)) / count;
  return Array.from({ length: count }, (_, index) => {
    const upper = 1 - index * (height + gap);
    return [upper - height, upper];
  });
}

function chartHeight(root) {
  return Number.parseFloat(
    getComputedStyle(root).getPropertyValue("--apc-chart-height")
  );
}

function sizeChartContainers(state) {
  const height = chartHeight(state.root);
  state.inputElement.style.height = `${height}px`;
  state.outputElement.style.height = `${height}px`;
  return height;
}

function measuredPoints(snapshot, key, index) {
  return {
    x: snapshot.map((sample) => sample.time),
    y: snapshot.map((sample) => sample[key][index]),
  };
}

function makeTraces(snapshot, names, dataKey, measuredColor, predictions) {
  const futureTimes = predictions.times || [];
  const futureValues = predictions[dataKey] || [];
  return names.flatMap((name, index) => {
    const measured = measuredPoints(snapshot, dataKey, index);
    return [
      {
        x: measured.x,
        y: measured.y,
        mode: "lines+markers",
        name,
        line: { color: measuredColor[index] || measuredColor[0], width: 1.6 },
        marker: { size: 4 },
        xaxis: axisName("x", index),
        yaxis: axisName("y", index),
        hovertemplate: "%{x}<br>%{y:.4g}<extra></extra>",
      },
      {
        x: futureTimes,
        y: futureValues.map((row) => row[index]),
        mode: "lines",
        name: `${name} prediction`,
        line: { color: COLORS.prediction, width: 1.4, dash: "dot" },
        xaxis: axisName("x", index),
        yaxis: axisName("y", index),
        hoverinfo: "skip",
      },
    ];
  });
}

function shapeAxis(index) {
  return axisName("y", index);
}

function horizontalShape(value, index, color, dash) {
  return {
    type: "line",
    xref: "paper",
    x0: 0,
    x1: 1,
    yref: shapeAxis(index),
    y0: value,
    y1: value,
    line: { color, width: 1, dash },
  };
}

function chartShapes(kind, config, events) {
  const limits = kind === "inputs" ? config.input_limits : config.output_limits;
  const shapes = [];

  limits.lower.forEach((value, index) => {
    shapes.push(horizontalShape(value, index, COLORS.limit, "dash"));
    shapes.push(horizontalShape(limits.upper[index], index, COLORS.limit, "dash"));
  });
  if (config.target && config.target.group === kind.slice(0, -1)) {
    shapes.push(
      horizontalShape(config.target.value, config.target.index, COLORS.target, "dot")
    );
  }
  events.forEach((event) => {
    shapes.push({
      type: "line",
      xref: "x",
      x0: event.time,
      x1: event.time,
      yref: "paper",
      y0: 0,
      y1: 1,
      line: { color: COLORS.event, width: 1.5, dash: "dash" },
    });
  });
  return shapes;
}

function makeLayout(kind, config, events, runId, width, height) {
  const names = kind === "inputs" ? config.input_names : config.output_names;
  const units = kind === "inputs" ? config.input_units : config.output_units;
  const rows = domains(names.length);
  const layout = {
    autosize: false,
    width,
    height,
    margin: { l: 102, r: 12, t: 28, b: 42 },
    paper_bgcolor: COLORS.background,
    plot_bgcolor: COLORS.background,
    font: { family: FONT_FAMILY, color: COLORS.text, size: FONT_SIZE.base },
    legend: { font: { family: FONT_FAMILY, size: FONT_SIZE.legend } },
    showlegend: false,
    uirevision: `apc-trends-${runId}`,
    shapes: chartShapes(kind, config, events),
    annotations: [],
  };

  names.forEach((name, index) => {
    const xName = axisLayoutName("x", index);
    const yName = axisLayoutName("y", index);
    layout[xName] = {
      domain: [0, 1],
      anchor: axisName("y", index),
      matches: index === 0 ? undefined : "x",
      showticklabels: index === names.length - 1,
      tickfont: { family: FONT_FAMILY, size: FONT_SIZE.tick },
      gridcolor: COLORS.grid,
      zeroline: false,
      title: index === names.length - 1
        ? { text: "Process minute", font: { family: FONT_FAMILY, size: FONT_SIZE.title } }
        : undefined,
    };
    layout[yName] = {
      domain: rows[index],
      anchor: axisName("x", index),
      automargin: true,
      tickfont: { family: FONT_FAMILY, size: FONT_SIZE.tick },
      gridcolor: COLORS.grid,
      zeroline: false,
      title: {
        text: units[index],
        standoff: 12,
        font: { family: FONT_FAMILY, size: FONT_SIZE.title },
      },
    };
    layout.annotations.push({
      text: name,
      x: 0.5,
      xref: "paper",
      y: rows[index][1] - 0.005,
      yref: "paper",
      yanchor: "top",
      showarrow: false,
      font: {
        family: FONT_FAMILY,
        color: COLORS.text,
        size: FONT_SIZE.subplot,
      },
    });
  });
  return layout;
}

function predictionUpdate(predictions, key, count) {
  const values = predictions[key] || [];
  return {
    x: Array.from({ length: count }, () => predictions.times || []),
    y: Array.from({ length: count }, (_, index) =>
      values.map((row) => row[index])
    ),
  };
}

function snapshotForRun(payload) {
  return payload.snapshot || (payload.sample ? [payload.sample] : []);
}

async function resetCharts(state, payload) {
  state.runId = payload.run_id;
  state.processedSampleId = -1;
  state.processedEventIds = new Set();
  state.events = [];
  state.userZoomed.inputs = false;
  state.userZoomed.outputs = false;
  payload.events.forEach((event) => {
    state.processedEventIds.add(event.event_id);
    state.events.push(event);
  });
  const snapshot = snapshotForRun(payload);
  snapshot.forEach((sample) => {
    state.processedSampleId = Math.max(state.processedSampleId, sample.sample_id);
  });

  const processInputCount = payload.config.process_input_count || 3;
  const inputColors = payload.config.input_names.map((_, index) =>
    index >= processInputCount
      ? COLORS.prediction
      : COLORS.measuredInput
  );
  const processOutputCount = payload.config.process_output_count || 4;
  const outputColors = payload.config.output_names.map((_, index) =>
    index >= processOutputCount
      ? COLORS.prediction
      : COLORS.measuredOutput
  );
  const inputTraces = makeTraces(
    snapshot,
    payload.config.input_names,
    "inputs",
    inputColors,
    payload.predictions
  );
  const outputTraces = makeTraces(
    snapshot,
    payload.config.output_names,
    "outputs",
    outputColors,
    payload.predictions
  );
  const plotConfig = { displayModeBar: false, responsive: false, scrollZoom: false };
  const height = sizeChartContainers(state);
  const rootWidth = state.root.getBoundingClientRect().width;
  const chartWidth = rootWidth <= 900 ? rootWidth : (rootWidth - 24) / 2;

  if (state.mounted) {
    await Promise.all([
      Plotly.react(
        state.inputElement,
        inputTraces,
        makeLayout("inputs", payload.config, state.events, payload.run_id, chartWidth, height),
        plotConfig
      ),
      Plotly.react(
        state.outputElement,
        outputTraces,
        makeLayout("outputs", payload.config, state.events, payload.run_id, chartWidth, height),
        plotConfig
      ),
    ]);
  } else {
    await Promise.all([
      Plotly.newPlot(
        state.inputElement,
        inputTraces,
        makeLayout("inputs", payload.config, state.events, payload.run_id, chartWidth, height),
        plotConfig
      ),
      Plotly.newPlot(
        state.outputElement,
        outputTraces,
        makeLayout("outputs", payload.config, state.events, payload.run_id, chartWidth, height),
        plotConfig
      ),
    ]);
    state.mounted = true;
    state.inputElement.on("plotly_relayout", (changes) => {
      if (state.rangeChangeInProgress) return;
      if (changes["xaxis.autorange"]) state.userZoomed.inputs = false;
      if (changes["xaxis.range[0]"] !== undefined) state.userZoomed.inputs = true;
    });
    state.outputElement.on("plotly_relayout", (changes) => {
      if (state.rangeChangeInProgress) return;
      if (changes["xaxis.autorange"]) state.userZoomed.outputs = false;
      if (changes["xaxis.range[0]"] !== undefined) state.userZoomed.outputs = true;
    });
  }
}

async function updateLiveRanges(state, time) {
  const upper = Math.max(10, time + 1);
  const lower = Math.max(0, upper - MAX_POINTS);
  const updates = [];
  state.rangeChangeInProgress = true;
  if (!state.userZoomed.inputs) {
    updates.push(Plotly.relayout(state.inputElement, { "xaxis.range": [lower, upper] }));
  }
  if (!state.userZoomed.outputs) {
    updates.push(Plotly.relayout(state.outputElement, { "xaxis.range": [lower, upper] }));
  }
  try {
    await Promise.all(updates);
  } finally {
    state.rangeChangeInProgress = false;
  }
}

async function appendPayload(state, payload) {
  const sample = payload.sample;
  if (sample && sample.sample_id > state.processedSampleId) {
    const inputIndices = payload.config.input_names.map((_, index) => index * 2);
    const outputIndices = payload.config.output_names.map((_, index) => index * 2);
    await Promise.all([
      Plotly.extendTraces(
        state.inputElement,
        {
          x: inputIndices.map(() => [sample.time]),
          y: sample.inputs.map((value) => [value]),
        },
        inputIndices,
        MAX_POINTS
      ),
      Plotly.extendTraces(
        state.outputElement,
        {
          x: outputIndices.map(() => [sample.time]),
          y: sample.outputs.map((value) => [value]),
        },
        outputIndices,
        MAX_POINTS
      ),
    ]);
    state.processedSampleId = sample.sample_id;
    await updateLiveRanges(state, sample.time);
  }

  payload.events.forEach((event) => {
    if (!state.processedEventIds.has(event.event_id)) {
      state.processedEventIds.add(event.event_id);
      state.events.push(event);
    }
  });

  const inputPrediction = predictionUpdate(
    payload.predictions,
    "inputs",
    payload.config.input_names.length
  );
  const outputPrediction = predictionUpdate(
    payload.predictions,
    "outputs",
    payload.config.output_names.length
  );
  const inputPredictionIndices = payload.config.input_names.map((_, index) => index * 2 + 1);
  const outputPredictionIndices = payload.config.output_names.map((_, index) => index * 2 + 1);
  await Promise.all([
    Plotly.restyle(state.inputElement, inputPrediction, inputPredictionIndices),
    Plotly.restyle(state.outputElement, outputPrediction, outputPredictionIndices),
    Plotly.relayout(state.inputElement, {
      shapes: chartShapes("inputs", payload.config, state.events),
    }),
    Plotly.relayout(state.outputElement, {
      shapes: chartShapes("outputs", payload.config, state.events),
    }),
  ]);
}

export default function processTrends(component) {
  const { data, parentElement } = component;
  const root = parentElement.querySelector(".apc-process-trends");
  let state = root.__apcProcessTrends;

  if (!state) {
    state = {
      root,
      runId: null,
      processedSampleId: -1,
      processedEventIds: new Set(),
      events: [],
      mounted: false,
      inputElement: parentElement.querySelector("#apc-input-trends"),
      outputElement: parentElement.querySelector("#apc-output-trends"),
      queue: Promise.resolve(),
      rangeChangeInProgress: false,
      userZoomed: { inputs: false, outputs: false },
      lastWidth: root.getBoundingClientRect().width,
      resizeObserver: null,
    };
    state.resizeObserver = new ResizeObserver(([entry]) => {
      const width = entry.contentRect.width;
      if (!state.mounted || Math.abs(width - state.lastWidth) < 1) return;
      state.lastWidth = width;
      const chartWidth = width <= 900 ? width : (width - 24) / 2;
      const height = sizeChartContainers(state);
      Plotly.relayout(state.inputElement, { width: chartWidth, height });
      Plotly.relayout(state.outputElement, { width: chartWidth, height });
    });
    state.resizeObserver.observe(root);
    root.__apcProcessTrends = state;
  }

  state.queue = state.queue.then(() =>
    state.runId !== data.run_id ? resetCharts(state, data) : appendPayload(state, data)
  );

  return () => {
    state.resizeObserver?.disconnect();
    if (state.mounted) {
      Plotly.purge(state.inputElement);
      Plotly.purge(state.outputElement);
    }
    delete root.__apcProcessTrends;
  };
}
