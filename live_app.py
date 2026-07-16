"""Live, interactive spray-dryer APC simulation."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from apc_lab.live_dryer import (
    GAIN_MATRIX,
    INPUT_MAX,
    INPUT_MIN,
    INPUT_NAMES,
    INPUT_UNITS,
    MAX_MOVE,
    NOMINAL_INPUTS,
    NOMINAL_OUTPUTS,
    OUTPUT_TAU,
    OUTPUT_NAMES,
    OUTPUT_UNITS,
    ConstrainedDryerMPC,
    LiveSprayDryer,
    steady_outputs,
)
from apc_lab.model_fitting import arrays_from_dataframe, fit_dynamic_model

st.set_page_config(page_title="Live Spray Dryer APC", layout="wide")

DEFAULT_OUTPUT_MIN = np.array([75.0, 3.0, 3.0, 0.090])
DEFAULT_OUTPUT_MAX = np.array([105.0, 5.5, 5.2, 0.145])
CONTROLLER_VERSION = 4


def initialize() -> None:
    if "dryer" not in st.session_state:
        st.session_state.dryer = LiveSprayDryer()
        st.session_state.inputs = NOMINAL_INPUTS.copy()
        st.session_state.outputs = NOMINAL_OUTPUTS.copy()
        st.session_state.minute = 0
        st.session_state.running = False
        st.session_state.history = {
            "minute": [],
            **{name: [] for name in INPUT_NAMES + OUTPUT_NAMES},
        }
    if st.session_state.get("controller_version") != CONTROLLER_VERSION:
        st.session_state.controller = ConstrainedDryerMPC()
        st.session_state.controller_version = CONTROLLER_VERSION
        st.session_state.pop("fitted_model", None)


def reset() -> None:
    for key in ("dryer", "controller", "controller_version", "inputs", "outputs", "minute", "history"):
        st.session_state.pop(key, None)
    initialize()


def trend_figure(
    names: tuple[str, ...],
    planned: np.ndarray | None = None,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    target_index: int | None = None,
    target: float | None = None,
) -> go.Figure:
    fig = make_subplots(rows=len(names), cols=1, shared_xaxes=True, vertical_spacing=0.04)
    minutes = st.session_state.history["minute"]
    for index, name in enumerate(names):
        fig.add_trace(
            go.Scatter(x=minutes, y=st.session_state.history[name], name=name),
            row=index + 1,
            col=1,
        )
        if planned is not None:
            future = np.arange(st.session_state.minute + 1, st.session_state.minute + 1 + len(planned))
            fig.add_trace(
                go.Scatter(
                    x=future,
                    y=planned[:, index],
                    name=f"{name} plan",
                    line={"dash": "dot"},
                ),
                row=index + 1,
                col=1,
            )
        if lower is not None and upper is not None:
            fig.add_hline(y=lower[index], line_dash="dash", line_color="red", row=index + 1, col=1)
            fig.add_hline(y=upper[index], line_dash="dash", line_color="red", row=index + 1, col=1)
        if target_index == index and target is not None:
            fig.add_hline(y=target, line_dash="dot", line_color="green", row=index + 1, col=1)
    fig.update_layout(
        height=155 * len(names),
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        showlegend=False,
        uirevision="keep",
        transition={"duration": 250},
    )
    return fig


initialize()
st.title("Live Spray Dryer APC")
st.caption(
    "Manual mode lets you operate the dryer. APC mode optimizes your selected "
    "parameter while respecting predicted output constraints."
)

with st.sidebar:
    mode = st.radio("Control mode", ("Manual", "APC"), key="mode")
    st.subheader("Optimization objective")
    objective_parameter = st.selectbox(
        "Parameter",
        INPUT_NAMES + OUTPUT_NAMES,
        index=6,
    )
    objective_mode = st.radio(
        "Goal", ("Target", "Maximize", "Minimize"), horizontal=True
    ).lower()
    if objective_parameter in INPUT_NAMES:
        objective_group = "input"
        objective_index = INPUT_NAMES.index(objective_parameter)
        objective_bounds = (INPUT_MIN[objective_index], INPUT_MAX[objective_index])
        objective_default = NOMINAL_INPUTS[objective_index]
        objective_step = (objective_bounds[1] - objective_bounds[0]) / 100
    else:
        objective_group = "output"
        objective_index = OUTPUT_NAMES.index(objective_parameter)
        output_ranges = [(60.0, 120.0, 1.0), (2.0, 7.0, 0.1), (2.0, 8.0, 0.1), (0.070, 0.170, 0.001)]
        objective_bounds = output_ranges[objective_index][:2]
        objective_default = NOMINAL_OUTPUTS[objective_index]
        objective_step = output_ranges[objective_index][2]
    objective_target = float(objective_default)
    if objective_mode == "target":
        objective_target = st.slider(
            f"{objective_parameter} target",
            float(objective_bounds[0]),
            float(objective_bounds[1]),
            float(objective_default),
            float(objective_step),
            key=f"objective_target_{objective_group}_{objective_index}",
        )
    st.subheader("Manipulated inputs")
    manual_inputs = np.array(
        [
            st.slider(INPUT_NAMES[i], float(INPUT_MIN[i]), float(INPUT_MAX[i]), float(NOMINAL_INPUTS[i]), key=f"input_{i}")
            for i in range(3)
        ]
    )
    st.subheader("APC input constraints")
    input_enabled = np.empty(3, dtype=bool)
    input_min = np.empty(3)
    input_max = np.empty(3)
    max_move = np.empty(3)
    for i, name in enumerate(INPUT_NAMES):
        input_enabled[i] = st.checkbox(
            f"Allow APC to change {name}", value=True, key=f"enabled_{i}"
        )
        selected = st.slider(
            f"{name} operating range",
            float(INPUT_MIN[i]),
            float(INPUT_MAX[i]),
            (float(INPUT_MIN[i]), float(INPUT_MAX[i])),
            key=f"input_constraint_{i}",
        )
        input_min[i], input_max[i] = selected
        max_move[i] = st.slider(
            f"{name} maximum move per minute",
            0.0,
            float(MAX_MOVE[i] * 3),
            float(MAX_MOVE[i]),
            key=f"max_move_{i}",
        )
    st.subheader("Output constraints")
    output_min = np.empty(4)
    output_max = np.empty(4)
    ranges = [(60.0, 120.0, 1.0), (2.0, 7.0, 0.1), (2.0, 8.0, 0.1), (0.070, 0.170, 0.001)]
    for i, name in enumerate(OUTPUT_NAMES):
        selected = st.slider(
            name,
            ranges[i][0],
            ranges[i][1],
            (float(DEFAULT_OUTPUT_MIN[i]), float(DEFAULT_OUTPUT_MAX[i])),
            ranges[i][2],
            key=f"constraint_{i}",
        )
        output_min[i], output_max[i] = selected
    left, right = st.columns(2)
    if left.button("Run" if not st.session_state.running else "Pause", width="stretch"):
        st.session_state.running = not st.session_state.running
    if right.button("Reset", width="stretch"):
        reset()
        st.rerun()

with st.expander("Model equations and parameter effects"):
    st.markdown(
        r"""
The model first calculates delayed steady-state outputs:

$$\mathbf{y}_{ss}=\mathbf{y}_0+\mathbf{K}(\mathbf{u}_{delayed}-\mathbf{u}_0)$$

Each measured output then approaches its steady-state value:

$$y_i(k+1)=y_i(k)+\frac{\Delta t}{\tau_i}\left(y_{ss,i}-y_i(k)\right)+noise$$

The input delay is **3 simulated minutes**. A gain is the final output change
caused by one unit of input change, with other inputs held constant.

**How to read the model**

- A positive gain means increasing that input eventually increases the output.
- A negative gain means increasing that input eventually decreases the output.
- Gain describes the final effect; the time constant describes how quickly it
  appears. After one time constant, about 63% of the final change is visible.
  After three time constants, about 95% is visible.
- The plant waits for the dead time before responding. During this wait an
  optimizer can make several moves before seeing the result of its first move.
- Effects add together in this linear model. Real dryers also contain nonlinear
  interactions, changing gains, and unmeasured disturbances.

With deviations from nominal written as $\Delta$, the steady-state formulas are:

```text
Exhaust temperature = 90 - 0.18*DeltaFeed + 1.20*DeltaAir + 0.55*DeltaInletTemp
Feed pressure       =  4 + 0.055*DeltaFeed - 0.05*DeltaAir
Powder moisture     = 4.5 + 0.050*DeltaFeed - 0.10*DeltaAir - 0.055*DeltaInletTemp
Exhaust humidity    = .12 + .0022*DeltaFeed - .0040*DeltaAir - .0018*DeltaInletTemp
```

The optimizer uses the same gain, lag, and dead-time structure to predict
future outputs. It minimizes $J=w_oJ_o+w_m\sum(\Delta u^2)$, where $J_o$ is
$\sum(value-target)^2$ for **Target**, $-\sum(value)$ for **Maximize**, or
$+\sum(value)$ for **Minimize**.

subject to all selected input and output limits. Model mismatch still occurs
when fitted coefficients differ from the simulated plant.
"""
    )
    gain_rows = []
    for output_index, output_name in enumerate(OUTPUT_NAMES):
        gain_rows.append(
            {
                "Output": output_name,
                "Time constant (min)": OUTPUT_TAU[output_index],
                "95% response after delay (min)": 3 * OUTPUT_TAU[output_index] + 3,
                **{
                    f"Effect per +1 {INPUT_UNITS[input_index]} {input_name}": GAIN_MATRIX[output_index, input_index]
                    for input_index, input_name in enumerate(INPUT_NAMES)
                },
            }
        )
    st.dataframe(gain_rows, width="stretch", hide_index=True)
    effect_input = st.selectbox("Explore input effect", INPUT_NAMES)
    effect_index = INPUT_NAMES.index(effect_input)
    effect_change = st.slider(
        f"Change in {effect_input} ({INPUT_UNITS[effect_index]})",
        float(-0.25 * (INPUT_MAX[effect_index] - INPUT_MIN[effect_index])),
        float(0.25 * (INPUT_MAX[effect_index] - INPUT_MIN[effect_index])),
        float(0.1 * (INPUT_MAX[effect_index] - INPUT_MIN[effect_index])),
        key="effect_change",
    )
    changed_inputs = NOMINAL_INPUTS.copy()
    changed_inputs[effect_index] += effect_change
    changed_outputs = steady_outputs(changed_inputs)
    effect_rows = [
        {
            "Output": name,
            "Nominal": NOMINAL_OUTPUTS[i],
            "New steady state": changed_outputs[i],
            "Change": changed_outputs[i] - NOMINAL_OUTPUTS[i],
            "Unit": OUTPUT_UNITS[i],
        }
        for i, name in enumerate(OUTPUT_NAMES)
    ]
    st.dataframe(effect_rows, width="stretch", hide_index=True)
    st.markdown("**Full-range effects and practical interpretation**")
    interpretation_rows = []
    for output_index, output_name in enumerate(OUTPUT_NAMES):
        for input_index, input_name in enumerate(INPUT_NAMES):
            full_effect = GAIN_MATRIX[output_index, input_index] * (
                INPUT_MAX[input_index] - INPUT_MIN[input_index]
            )
            direction = "increases" if full_effect > 0 else "decreases" if full_effect < 0 else "does not affect"
            interpretation_rows.append(
                {
                    "When this input increases": input_name,
                    "This output": output_name,
                    "Direction": direction,
                    "Effect across full input range": full_effect,
                    "Output unit": OUTPUT_UNITS[output_index],
                }
            )
    st.dataframe(interpretation_rows, width="stretch", hide_index=True)

with st.expander("Fit the controller model to a dataset"):
    st.markdown(
        """
Upload timestamp-ordered CSV process data. The fitter estimates integer dead
time, four output time constants, and the complete 4 x 3 gain matrix.

Inputs must move independently enough to identify their separate effects. A
useful dataset covers the relevant operating range and avoids long periods
where all inputs move together. The fitted model updates the **controller
predictor**; the simulated plant remains unchanged so model mismatch is visible.
"""
    )
    example_rng = np.random.default_rng(42)
    example_inputs = np.tile(NOMINAL_INPUTS, (240, 1))
    for k in range(0, 240, 20):
        example_inputs[k:] += example_rng.uniform(-MAX_MOVE * 4, MAX_MOVE * 4)
        example_inputs[k:] = np.clip(example_inputs[k:], INPUT_MIN, INPUT_MAX)
    example_dryer = LiveSprayDryer(seed=42)
    example_outputs = np.array([example_dryer.step(row) for row in example_inputs])
    example_data = pd.DataFrame(
        np.column_stack([example_inputs, example_outputs]),
        columns=INPUT_NAMES + OUTPUT_NAMES,
    )
    st.download_button(
        "Download example identification CSV",
        example_data.to_csv(index=False),
        "spray_dryer_identification_example.csv",
        "text/csv",
    )
    uploaded = st.file_uploader("Upload process CSV", type="csv")
    max_delay = st.slider("Maximum dead time to search (samples)", 0, 30, 10)
    if uploaded is not None:
        try:
            dataset = pd.read_csv(uploaded)
            st.dataframe(dataset.head(20), width="stretch")
            fit_inputs, fit_outputs = arrays_from_dataframe(dataset)
            if st.button("Fit and use model"):
                fitted = fit_dynamic_model(fit_inputs, fit_outputs, max_delay=max_delay)
                st.session_state.controller.gain_matrix = fitted.gain_matrix.copy()
                st.session_state.controller.output_tau = fitted.output_tau.copy()
                st.session_state.controller.delay_steps = fitted.delay_steps
                st.session_state.controller.nominal_inputs = fitted.nominal_inputs.copy()
                st.session_state.controller.nominal_outputs = fitted.nominal_outputs.copy()
                st.session_state.fitted_model = fitted
                st.success(f"Fitted model applied using {fitted.samples} samples.")
        except Exception as error:
            st.error(str(error))
    if "fitted_model" in st.session_state:
        fitted = st.session_state.fitted_model
        st.write(f"Estimated dead time: **{fitted.delay_steps} samples**")
        st.write(
            "Dataset baseline inputs: "
            + ", ".join(
                f"{name}={value:.3f}" for name, value in zip(INPUT_NAMES, fitted.nominal_inputs)
            )
        )
        fitted_rows = []
        for output_index, output_name in enumerate(OUTPUT_NAMES):
            fitted_rows.append(
                {
                    "Output": output_name,
                    "Estimated tau": fitted.output_tau[output_index],
                    "Derivative RMSE": fitted.rmse[output_index],
                    **{
                        f"Gain from {name}": fitted.gain_matrix[output_index, i]
                        for i, name in enumerate(INPUT_NAMES)
                    },
                }
            )
        st.dataframe(fitted_rows, width="stretch", hide_index=True)
        if st.button("Restore built-in controller model"):
            st.session_state.controller.gain_matrix = GAIN_MATRIX.copy()
            st.session_state.controller.output_tau = OUTPUT_TAU.copy()
            st.session_state.controller.delay_steps = 3
            st.session_state.controller.nominal_inputs = NOMINAL_INPUTS.copy()
            st.session_state.controller.nominal_outputs = NOMINAL_OUTPUTS.copy()
            st.session_state.pop("fitted_model", None)
            st.rerun()


@st.fragment(run_every=1.0)
def live_panel() -> None:
    if st.session_state.running:
        if st.session_state.mode == "Manual":
            st.session_state.inputs = manual_inputs.copy()
        else:
            st.session_state.inputs = st.session_state.controller.move(
                st.session_state.outputs,
                st.session_state.inputs,
                output_min,
                output_max,
                objective_group,
                objective_index,
                objective_mode,
                objective_target,
                input_min,
                input_max,
                max_move,
                input_enabled,
            )
        st.session_state.outputs = st.session_state.dryer.step(st.session_state.inputs)
        st.session_state.minute += 1
        st.session_state.history["minute"].append(st.session_state.minute)
        for name, value in zip(INPUT_NAMES, st.session_state.inputs):
            st.session_state.history[name].append(float(value))
        for name, value in zip(OUTPUT_NAMES, st.session_state.outputs):
            st.session_state.history[name].append(float(value))
        for name in INPUT_NAMES + OUTPUT_NAMES:
            st.session_state.history[name] = st.session_state.history[name][-120:]
        st.session_state.history["minute"] = st.session_state.history["minute"][-120:]

    st.subheader(f"Process minute {st.session_state.minute}")
    input_cols = st.columns(3)
    for col, name, unit, value in zip(input_cols, INPUT_NAMES, INPUT_UNITS, st.session_state.inputs):
        col.metric(name, f"{value:.2f} {unit}")

    output_cols = st.columns(4)
    for i, (col, name, unit, value) in enumerate(zip(output_cols, OUTPUT_NAMES, OUTPUT_UNITS, st.session_state.outputs)):
        safe = output_min[i] <= value <= output_max[i]
        col.metric(name, f"{value:.3f} {unit}", "inside limits" if safe else "CONSTRAINT VIOLATION", delta_color="normal" if safe else "inverse")

    plan_inputs = st.session_state.controller.last_input_plan if st.session_state.mode == "APC" else None
    plan_outputs = st.session_state.controller.last_output_plan if st.session_state.mode == "APC" else None
    input_target_index = objective_index if objective_group == "input" and objective_mode == "target" else None
    output_target_index = objective_index if objective_group == "output" and objective_mode == "target" else None
    left, right = st.columns(2)
    left.plotly_chart(
        trend_figure(INPUT_NAMES, plan_inputs, input_min, input_max, input_target_index, objective_target),
        width="stretch",
        key="live_input_trends",
        config={"displayModeBar": False},
    )
    right.plotly_chart(
        trend_figure(OUTPUT_NAMES, plan_outputs, output_min, output_max, output_target_index, objective_target),
        width="stretch",
        key="live_output_trends",
        config={"displayModeBar": False},
    )

    if st.session_state.mode == "APC":
        status = "Optimization solved" if st.session_state.controller.last_success else "Optimizer could not find a feasible move"
        next_moves = ", ".join(
            f"{name}: {move:+.2f} {unit}/min"
            for name, unit, move in zip(
                INPUT_NAMES, INPUT_UNITS, st.session_state.controller.last_move
            )
        )
        st.info(
            f"{status}. Objective: {objective_mode} {objective_parameter}. "
            f"Limiting constraint: {st.session_state.controller.last_limiting_constraint}. "
            "Dotted lines show the current MPC prediction."
        )
        if objective_group == "input" and not input_enabled[objective_index]:
            st.warning(f"{objective_parameter} is frozen, so APC cannot optimize it directly.")
        st.code(f"Next optimized moves: {next_moves}\nSolver: {st.session_state.controller.last_message}")
    elif st.session_state.running:
        st.info("Manual operation: move the input sliders and watch the delayed process response.")

live_panel()
