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
    MEASUREMENT_NOISE_STD,
    NOISE_MULTIPLIERS,
    NOMINAL_INPUTS,
    NOMINAL_OUTPUTS,
    OUTPUT_NAMES,
    OUTPUT_TAU,
    OUTPUT_UNITS,
    ConstrainedDryerMPC,
    LiveSprayDryer,
    steady_outputs,
)
from apc_lab.model_fitting import arrays_from_dataframe, fit_dynamic_model
from apc_lab.scada_ui import (
    ScadaValue,
    apply_scada_theme,
    constraint_state,
    render_message,
    render_section_title,
    render_sidebar_title,
    render_status_strip,
    render_title_bar,
    render_value_grid,
)

st.set_page_config(page_title="Spray Dryer APC Station", layout="wide")
apply_scada_theme()

DEFAULT_OUTPUT_MIN = np.array([75.0, 3.0, 3.0, 0.090])
DEFAULT_OUTPUT_MAX = np.array([105.0, 5.5, 5.2, 0.145])
CONTROLLER_VERSION = 4
SIMULATION_STATE_VERSION = 2


def initialize() -> None:
    """Create or migrate the live plant and controller session state."""

    if st.session_state.get("simulation_state_version") != SIMULATION_STATE_VERSION:
        for key in (
            "dryer",
            "inputs",
            "outputs",
            "true_outputs",
            "measurements",
            "minute",
            "running",
            "history",
        ):
            st.session_state.pop(key, None)
        st.session_state.simulation_state_version = SIMULATION_STATE_VERSION

    if "dryer" not in st.session_state:
        st.session_state.dryer = LiveSprayDryer()
        st.session_state.inputs = NOMINAL_INPUTS.copy()
        st.session_state.true_outputs = NOMINAL_OUTPUTS.copy()
        st.session_state.measurements = NOMINAL_OUTPUTS.copy()
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
    for key in (
        "dryer",
        "controller",
        "controller_version",
        "simulation_state_version",
        "inputs",
        "outputs",
        "true_outputs",
        "measurements",
        "minute",
        "history",
    ):
        st.session_state.pop(key, None)
    initialize()


def trend_figure(
    names: tuple[str, ...],
    units: tuple[str, ...],
    planned: np.ndarray | None = None,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    target_index: int | None = None,
    target: float | None = None,
    measured_color: str = "#35d05b",
) -> go.Figure:
    """Build a compact dark SCADA trend panel."""

    fig = make_subplots(
        rows=len(names),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.065,
        subplot_titles=names,
    )
    minutes = st.session_state.history["minute"]
    for index, name in enumerate(names):
        fig.add_trace(
            go.Scatter(
                x=minutes,
                y=st.session_state.history[name],
                name=name,
                line={"color": measured_color, "width": 1.6},
            ),
            row=index + 1,
            col=1,
        )
        if planned is not None:
            future = np.arange(
                st.session_state.minute + 1,
                st.session_state.minute + 1 + len(planned),
            )
            fig.add_trace(
                go.Scatter(
                    x=future,
                    y=planned[:, index],
                    name=f"{name} prediction",
                    line={"color": "#ffbf2f", "dash": "dot", "width": 1.4},
                ),
                row=index + 1,
                col=1,
            )
        if lower is not None and upper is not None:
            fig.add_hline(
                y=lower[index],
                line_dash="dash",
                line_color="#ef3f3f",
                line_width=1,
                row=index + 1,
                col=1,
            )
            fig.add_hline(
                y=upper[index],
                line_dash="dash",
                line_color="#ef3f3f",
                line_width=1,
                row=index + 1,
                col=1,
            )
        if target_index == index and target is not None:
            fig.add_hline(
                y=target,
                line_dash="dot",
                line_color="#54d9ff",
                line_width=1.5,
                row=index + 1,
                col=1,
            )
        fig.update_yaxes(
            title_text=units[index],
            title_font={"size": 9, "color": "#b6c5bd"},
            tickfont={"size": 9, "color": "#b6c5bd"},
            gridcolor="#26352e",
            zeroline=False,
            row=index + 1,
            col=1,
        )
    fig.update_xaxes(
        tickfont={"size": 9, "color": "#b6c5bd"},
        gridcolor="#26352e",
    )
    fig.update_xaxes(
        title_text="Process minute",
        title_font={"size": 9, "color": "#b6c5bd"},
        row=len(names),
        col=1,
    )
    fig.update_annotations(font={"family": "Courier New", "size": 10, "color": "#dce8df"})
    fig.update_layout(
        height=132 * len(names),
        margin={"l": 52, "r": 10, "t": 28, "b": 32},
        paper_bgcolor="#07110f",
        plot_bgcolor="#07110f",
        font={"family": "Courier New", "color": "#dce8df"},
        showlegend=False,
        uirevision="keep",
    )
    return fig


def render_model_explanation() -> None:
    with st.expander("Model equations and parameter effects"):
        st.markdown(
            r"""
The model first calculates delayed steady-state outputs:

$$\mathbf{y}_{ss}=\mathbf{y}_0+\mathbf{K}(\mathbf{u}_{delayed}-\mathbf{u}_0)$$

Each **true plant output** approaches its steady-state value without sensor
noise changing the plant state:

$$y_{true,i}(k+1)=y_{true,i}(k)+\frac{\Delta t}{\tau_i}\left(y_{ss,i}-y_{true,i}(k)\right)$$

The dashboard and controller receive a separate sensor measurement:

$$y_{measured,i}(k)=y_{true,i}(k)+\epsilon_i(k),\qquad
\epsilon_i\sim\mathcal{N}(0,\sigma_i^2)$$

The input delay is **3 simulated minutes**. A gain is the final output change
caused by one unit of input change, with other inputs held constant. Sensor
noise can be disabled or scaled in the operator station.

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
$+\sum(value)$ for **Minimize**, subject to all selected input and output
limits. Model mismatch still occurs when fitted coefficients differ from the
simulated plant.
"""
        )
        noise_rows = [
            {
                "Measured output": name,
                "Normal noise standard deviation": MEASUREMENT_NOISE_STD[i],
                "Unit": OUTPUT_UNITS[i],
            }
            for i, name in enumerate(OUTPUT_NAMES)
        ]
        st.markdown("**Sensor model**")
        st.dataframe(noise_rows, width="stretch", hide_index=True)

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
                direction = (
                    "increases"
                    if full_effect > 0
                    else "decreases"
                    if full_effect < 0
                    else "does not affect"
                )
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


def render_model_fitting() -> None:
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
                    fitted = fit_dynamic_model(
                        fit_inputs, fit_outputs, max_delay=max_delay
                    )
                    controller = st.session_state.controller
                    controller.gain_matrix = fitted.gain_matrix.copy()
                    controller.output_tau = fitted.output_tau.copy()
                    controller.delay_steps = fitted.delay_steps
                    controller.nominal_inputs = fitted.nominal_inputs.copy()
                    controller.nominal_outputs = fitted.nominal_outputs.copy()
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
                    f"{name}={value:.3f}"
                    for name, value in zip(INPUT_NAMES, fitted.nominal_inputs)
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
                controller = st.session_state.controller
                controller.gain_matrix = GAIN_MATRIX.copy()
                controller.output_tau = OUTPUT_TAU.copy()
                controller.delay_steps = 3
                controller.nominal_inputs = NOMINAL_INPUTS.copy()
                controller.nominal_outputs = NOMINAL_OUTPUTS.copy()
                st.session_state.pop("fitted_model", None)
                st.rerun()


initialize()

with st.sidebar:
    render_sidebar_title("OPERATOR STATION")
    mode = st.radio("Control mode", ("Manual", "APC"), key="mode", horizontal=True)
    left, right = st.columns(2)
    if left.button(
        "RUN" if not st.session_state.running else "HOLD", width="stretch"
    ):
        st.session_state.running = not st.session_state.running
        st.rerun()
    if right.button("RESET", width="stretch"):
        reset()
        st.rerun()

    render_sidebar_title("SENSOR INPUT")
    noise_enabled = st.checkbox("Measurement noise enabled", value=True)
    noise_profile = st.selectbox(
        "Noise profile", tuple(NOISE_MULTIPLIERS), index=1, disabled=not noise_enabled
    )

    render_sidebar_title("OPTIMIZATION")
    objective_parameter = st.selectbox(
        "Parameter", INPUT_NAMES + OUTPUT_NAMES, index=6
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
        output_ranges = [
            (60.0, 120.0, 1.0),
            (2.0, 7.0, 0.1),
            (2.0, 8.0, 0.1),
            (0.070, 0.170, 0.001),
        ]
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

    with st.expander("Manual MV commands", expanded=mode == "Manual"):
        manual_inputs = np.array(
            [
                st.slider(
                    INPUT_NAMES[i],
                    float(INPUT_MIN[i]),
                    float(INPUT_MAX[i]),
                    float(NOMINAL_INPUTS[i]),
                    key=f"input_{i}",
                )
                for i in range(3)
            ]
        )

    with st.expander("MV constraints"):
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

    with st.expander("CV constraints"):
        output_min = np.empty(4)
        output_max = np.empty(4)
        ranges = [
            (60.0, 120.0, 1.0),
            (2.0, 7.0, 0.1),
            (2.0, 8.0, 0.1),
            (0.070, 0.170, 0.001),
        ]
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


render_title_bar(
    "SPRAY DRYER APC TRAINER",
    "SIM-01  |  Educational multivariable control station  |  1 minute scan",
)


@st.fragment(run_every=1.0 if st.session_state.running else None)
def live_panel() -> None:
    if st.session_state.running:
        if st.session_state.mode == "Manual":
            st.session_state.inputs = manual_inputs.copy()
        else:
            st.session_state.inputs = st.session_state.controller.move(
                st.session_state.measurements,
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
        st.session_state.true_outputs = st.session_state.dryer.advance(
            st.session_state.inputs
        )
        st.session_state.measurements = st.session_state.dryer.measure(
            enabled=noise_enabled,
            multiplier=NOISE_MULTIPLIERS[noise_profile],
        )
        st.session_state.minute += 1
        st.session_state.history["minute"].append(st.session_state.minute)
        for name, value in zip(INPUT_NAMES, st.session_state.inputs):
            st.session_state.history[name].append(float(value))
        for name, value in zip(OUTPUT_NAMES, st.session_state.measurements):
            st.session_state.history[name].append(float(value))
        for name in INPUT_NAMES + OUTPUT_NAMES:
            st.session_state.history[name] = st.session_state.history[name][-120:]
        st.session_state.history["minute"] = st.session_state.history["minute"][-120:]

    controller = st.session_state.controller
    process_state = "normal" if st.session_state.running else "warning"
    process_text = "RUN" if st.session_state.running else "HOLD"
    mpc_active = st.session_state.mode == "APC"
    solver_state = (
        "normal"
        if mpc_active and controller.last_success
        else "alarm"
        if mpc_active
        else "neutral"
    )
    solver_text = (
        "OK"
        if mpc_active and controller.last_success
        else "FAULT"
        if mpc_active
        else "STANDBY"
    )
    sensor_text = noise_profile.upper() if noise_enabled else "OFF"
    render_status_strip(
        [
            ("PROCESS", process_text, process_state),
            ("CONTROL", "MPC ACTIVE" if mpc_active else "MANUAL", "normal" if mpc_active else "neutral"),
            ("SENSORS", sensor_text, "normal" if noise_enabled else "neutral"),
            ("SOLVER", solver_text, solver_state),
        ]
    )

    render_section_title(
        "MEASURED PROCESS VALUES",
        f"SCAN {st.session_state.minute:05d} | amber near limit, red outside",
    )
    output_formats = (".2f", ".3f", ".3f", ".4f")
    output_values = [
        ScadaValue(
            tag=f"PV-{101 + i}",
            label=name,
            value=format(float(value), output_formats[i]),
            unit=unit,
            state=constraint_state(value, output_min[i], output_max[i]),
        )
        for i, (name, unit, value) in enumerate(
            zip(OUTPUT_NAMES, OUTPUT_UNITS, st.session_state.measurements)
        )
    ]
    render_value_grid(output_values, columns=4)

    render_section_title("MANIPULATED VARIABLE COMMANDS", "controller outputs")
    input_values = [
        ScadaValue(
            tag=f"MV-{101 + i}",
            label=name,
            value=f"{float(value):.2f}",
            unit=unit,
        )
        for i, (name, unit, value) in enumerate(
            zip(INPUT_NAMES, INPUT_UNITS, st.session_state.inputs)
        )
    ]
    render_value_grid(input_values, columns=3)

    objective_text = f"{objective_mode.upper()} {objective_parameter}"
    if objective_mode == "target":
        objective_unit = (
            INPUT_UNITS[objective_index]
            if objective_group == "input"
            else OUTPUT_UNITS[objective_index]
        )
        objective_text += f" = {objective_target:g} {objective_unit}"
    render_message("CONTROL OBJECTIVE", objective_text)

    plan_inputs = controller.last_input_plan if mpc_active else None
    plan_outputs = controller.last_output_plan if mpc_active else None
    input_target_index = (
        objective_index
        if objective_group == "input" and objective_mode == "target"
        else None
    )
    output_target_index = (
        objective_index
        if objective_group == "output" and objective_mode == "target"
        else None
    )
    render_section_title(
        "PROCESS TRENDS", "green/cyan actual | amber prediction | red limits | cyan target"
    )
    left, right = st.columns(2)
    left.plotly_chart(
        trend_figure(
            INPUT_NAMES,
            INPUT_UNITS,
            plan_inputs,
            input_min,
            input_max,
            input_target_index,
            objective_target,
            measured_color="#54d9ff",
        ),
        width="stretch",
        key="live_input_trends",
        config={"displayModeBar": False, "scrollZoom": False},
    )
    right.plotly_chart(
        trend_figure(
            OUTPUT_NAMES,
            OUTPUT_UNITS,
            plan_outputs,
            output_min,
            output_max,
            output_target_index,
            objective_target,
        ),
        width="stretch",
        key="live_output_trends",
        config={"displayModeBar": False, "scrollZoom": False},
    )

    if mpc_active:
        next_moves = ", ".join(
            f"{name} {move:+.2f} {unit}/min"
            for name, unit, move in zip(INPUT_NAMES, INPUT_UNITS, controller.last_move)
        )
        render_message(
            "MPC STATUS",
            f"{solver_text} | limiting: {controller.last_limiting_constraint} | next: {next_moves}",
        )
        if objective_group == "input" and not input_enabled[objective_index]:
            st.warning(
                f"{objective_parameter} is frozen, so APC cannot optimize it directly."
            )
        with st.expander("Solver detail"):
            st.code(controller.last_message or "Waiting for first optimization scan.")
    elif st.session_state.running:
        render_message(
            "OPERATOR MESSAGE",
            "Manual operation active. MV slider changes appear after the configured process delay.",
        )


live_panel()
render_model_explanation()
render_model_fitting()
