"""Live, interactive spray-dryer APC simulation."""

import numpy as np
import pandas as pd
import streamlit as st

from apc_lab.equations import (
    FEED_LINE_EQUATION,
    MEASUREMENT_EQUATION,
    MPC_OBJECTIVE_EQUATION,
    OUTPUT_EFFECT_EQUATIONS,
    PLANT_DYNAMICS_EQUATION,
    STEADY_STATE_EQUATION,
)
from apc_lab.live_dryer import (
    GAIN_MATRIX,
    INPUT_MAX,
    INPUT_MIN,
    INPUT_NAMES,
    INPUT_UNITS,
    FEED_DRY_MATTER_GAIN,
    FEED_DRY_MATTER_TRANSITION_TAU,
    FEED_DRY_MATTER_UNIT,
    FEED_TANKS,
    HUMID_WEATHER_INCREASE,
    INLET_HUMIDITY_GAIN,
    INLET_HUMIDITY_UNIT,
    MAX_MOVE,
    MEASUREMENT_NOISE_STD,
    NOISE_MULTIPLIERS,
    NOMINAL_INPUTS,
    NOMINAL_INLET_HUMIDITY,
    NOMINAL_OUTPUTS,
    OUTPUT_NAMES,
    OUTPUT_TAU,
    OUTPUT_UNITS,
    ConstrainedDryerMPC,
    FeedTankEvent,
    FeedTankManager,
    InletWeatherManager,
    LiveSprayDryer,
    WEATHER_MODES,
    WeatherEvent,
    steady_outputs,
)
from apc_lab.model_fitting import arrays_from_dataframe, fit_dynamic_model
from apc_lab.operating_map_component import (
    prepare_operating_map_payload,
    render_operating_map,
)
from apc_lab.process_trends_component import (
    prepare_trend_payload,
    render_process_trends,
)
from apc_lab.psychrometrics import (
    MAP_HUMIDITY_RANGE,
    MAP_TEMPERATURE_RANGE,
    STANDARD_PRESSURE_KPA,
    assess_stickiness,
    moist_air_enthalpy,
    psychrometric_background,
)
from apc_lab.scada_ui import (
    ScadaValue,
    apply_scada_theme,
    constraint_state,
    render_event_banner,
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
CONTROLLER_VERSION = 5
SIMULATION_STATE_VERSION = 7
FEED_DRY_MATTER_HISTORY_NAME = "Feed dry matter"
FEED_TANK_HISTORY_NAME = "Feed tank"
INLET_HUMIDITY_HISTORY_NAME = "Inlet air humidity"
WEATHER_STATE_HISTORY_NAME = "Weather state"
TRUE_EXHAUST_TEMPERATURE_HISTORY_NAME = "True exhaust air temperature"
TRUE_EXHAUST_HUMIDITY_HISTORY_NAME = "True exhaust air humidity"
OPERATING_MAP_BACKGROUND = psychrometric_background()


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
            "feed_tank_manager",
            "tank_events",
            "last_tank_event",
            "weather_manager",
            "weather_events",
            "last_weather_event",
            "chart_run_id",
            "chart_last_sample_id",
            "chart_last_event_id",
            "chart_event_sequence",
            "chart_event_ids",
            "weather_event_ids",
            "chart_needs_snapshot",
            "operating_map_last_sample_id",
            "operating_map_needs_snapshot",
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
        st.session_state.feed_tank_manager = FeedTankManager()
        st.session_state.tank_events = []
        st.session_state.last_tank_event = None
        st.session_state.weather_manager = InletWeatherManager()
        st.session_state.weather_events = []
        st.session_state.last_weather_event = None
        st.session_state.history = {
            "minute": [],
            **{
                name: []
                for name in INPUT_NAMES
                + OUTPUT_NAMES
                + (
                    FEED_DRY_MATTER_HISTORY_NAME,
                    FEED_TANK_HISTORY_NAME,
                    INLET_HUMIDITY_HISTORY_NAME,
                    WEATHER_STATE_HISTORY_NAME,
                    TRUE_EXHAUST_TEMPERATURE_HISTORY_NAME,
                    TRUE_EXHAUST_HUMIDITY_HISTORY_NAME,
                )
            },
        }
    if "chart_run_id" not in st.session_state:
        st.session_state.chart_run_sequence = (
            st.session_state.get("chart_run_sequence", 0) + 1
        )
        st.session_state.chart_run_id = st.session_state.chart_run_sequence
        st.session_state.chart_last_sample_id = -1
        st.session_state.chart_last_event_id = 0
        st.session_state.chart_event_sequence = 0
        st.session_state.chart_event_ids = []
        st.session_state.weather_event_ids = []
        st.session_state.chart_needs_snapshot = True
        st.session_state.operating_map_last_sample_id = -1
        st.session_state.operating_map_needs_snapshot = True
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
        "feed_tank_manager",
        "tank_events",
        "last_tank_event",
        "weather_manager",
        "weather_events",
        "last_weather_event",
        "chart_run_id",
        "chart_last_sample_id",
        "chart_last_event_id",
        "chart_event_sequence",
        "chart_event_ids",
        "weather_event_ids",
        "chart_needs_snapshot",
        "operating_map_last_sample_id",
        "operating_map_needs_snapshot",
        "weather_mode",
    ):
        st.session_state.pop(key, None)
    initialize()


def record_tank_event(event: FeedTankEvent | None) -> None:
    """Apply a tank selection to the plant and retain one SCADA event record."""

    if event is None:
        return
    st.session_state.dryer.set_feed_dry_matter(event.new_dry_matter)
    st.session_state.last_tank_event = event
    st.session_state.chart_event_sequence += 1
    st.session_state.tank_events.append(event)
    st.session_state.chart_event_ids.append(st.session_state.chart_event_sequence)
    st.session_state.tank_events = st.session_state.tank_events[-12:]
    st.session_state.chart_event_ids = st.session_state.chart_event_ids[-12:]


def record_weather_event(event: WeatherEvent | None) -> None:
    """Retain one inlet-humidity event in the shared SCADA event sequence."""

    if event is None:
        return
    st.session_state.last_weather_event = event
    st.session_state.chart_event_sequence += 1
    st.session_state.weather_events.append(event)
    st.session_state.weather_event_ids.append(st.session_state.chart_event_sequence)
    st.session_state.weather_events = st.session_state.weather_events[-12:]
    st.session_state.weather_event_ids = st.session_state.weather_event_ids[-12:]


def toggle_running() -> None:
    st.session_state.running = not st.session_state.running


def change_selected_tank() -> None:
    tank_manager: FeedTankManager = st.session_state.feed_tank_manager
    selected_tank = st.session_state.manual_tank
    record_tank_event(
        tank_manager.change_to(
            selected_tank,
            st.session_state.minute,
            "Manual tank change",
        )
    )


def trigger_humid_weather() -> None:
    weather_manager: InletWeatherManager = st.session_state.weather_manager
    record_weather_event(weather_manager.trigger(st.session_state.minute))


@st.fragment
def render_feed_tank_command() -> None:
    """Render the manual tank command without causing a full application rerun."""

    tank_manager: FeedTankManager = st.session_state.feed_tank_manager
    tank_names = tuple(FEED_TANKS)
    next_tank_index = (tank_names.index(tank_manager.current_tank) + 1) % len(
        tank_names
    )
    selected_tank = st.selectbox(
        "Next feed tank",
        tank_names,
        index=next_tank_index,
        key="manual_tank",
    )
    st.button(
        "CHANGE FEED TANK",
        width="stretch",
        disabled=selected_tank == tank_manager.current_tank,
        type="primary",
        on_click=change_selected_tank,
    )
    st.caption(
        f"Active: {tank_manager.current_tank} | "
        f"{tank_manager.current_dry_matter:.1f}% dry matter"
    )


def chart_sample(index: int) -> dict[str, object]:
    """Return one bounded-history sample in the component's fixed array order."""

    history = st.session_state.history
    return {
        "sample_id": int(history["minute"][index]),
        "time": float(history["minute"][index]),
        "inputs": [
            *[float(history[name][index]) for name in INPUT_NAMES],
            float(history[FEED_DRY_MATTER_HISTORY_NAME][index]),
            float(history[INLET_HUMIDITY_HISTORY_NAME][index]),
        ],
        "outputs": [float(history[name][index]) for name in OUTPUT_NAMES],
        "inlet_humidity": float(history[INLET_HUMIDITY_HISTORY_NAME][index]),
        "weather_state": str(history[WEATHER_STATE_HISTORY_NAME][index]),
    }


def operating_point(
    sample_id: int, temperature: float, humidity: float
) -> dict[str, object]:
    """Build one true-plant operating point for the diagnostic map."""

    assessment = assess_stickiness(temperature, humidity)
    enthalpy = moist_air_enthalpy(temperature, humidity)
    return {
        "sample_id": int(sample_id),
        "temperature": float(temperature),
        "humidity": float(humidity),
        "enthalpy": enthalpy,
        "margin": assessment.margin,
        "status": assessment.status,
        "tooltip": (
            f"T={temperature:.2f} C<br>"
            f"w={humidity:.4f} kg water/kg dry air<br>"
            f"h={enthalpy:.1f} kJ/kg dry air<br>"
            f"stickiness margin={assessment.margin:.1f} C"
        ),
    }


def operating_map_history_sample(index: int) -> dict[str, object]:
    """Return one noise-free exhaust point from bounded simulation history."""

    history = st.session_state.history
    return operating_point(
        int(history["minute"][index]),
        float(history[TRUE_EXHAUST_TEMPERATURE_HISTORY_NAME][index]),
        float(history[TRUE_EXHAUST_HUMIDITY_HISTORY_NAME][index]),
    )


def chart_event_payloads() -> list[dict[str, object]]:
    """Serialize retained disturbance events with stable per-run identifiers."""

    events = [
        {
            "event_id": event_id,
            "time": event.minute,
            "type": event.event_type,
            "label": f"{event.old_tank} to {event.new_tank}",
        }
        for event_id, event in zip(
            st.session_state.chart_event_ids,
            st.session_state.tank_events,
        )
    ]
    events.extend(
        {
            "event_id": event_id,
            "time": event.minute,
            "type": event.event_type,
            "label": event.event_type,
        }
        for event_id, event in zip(
            st.session_state.weather_event_ids,
            st.session_state.weather_events,
        )
    )
    return sorted(events, key=lambda event: int(event["event_id"]))


def render_model_explanation() -> None:
    with st.expander("Model equations and parameter effects"):
        st.markdown(
            """
The model first calculates delayed steady-state outputs. Incoming feed dry
matter and inlet-air humidity are plant-only disturbances, so they are present
in the simulated dryer but absent from the MPC predictor. The controller
rejects both through measured-output feedback.
"""
        )
        st.latex(STEADY_STATE_EQUATION)

        st.markdown(
            "Each **true plant output** approaches its steady-state value "
            "without sensor noise changing the plant state."
        )
        st.latex(PLANT_DYNAMICS_EQUATION)

        st.markdown(
            "A selected tank changes the feed-line target, which transitions "
            "with a first-order mixing lag."
        )
        st.latex(FEED_LINE_EQUATION)

        st.markdown(
            "The dashboard and controller receive a separate sensor measurement."
        )
        st.latex(MEASUREMENT_EQUATION)

        st.markdown(
            """
The input delay is **3 simulated minutes**. A tank change sets a new dry-matter
target and the feed line transitions to it with a **2 minute** first-order lag.
Inlet humidity is represented as kg water per kg dry air. Sensible heating does
not change this humidity ratio; humid weather therefore adds water load at the
dryer inlet.
A gain is the final output change caused by one unit of input change, with
other inputs held constant. Sensor noise can be disabled or scaled in the
operator station.

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

With deviations from nominal written as `Delta`, the four steady-state output
equations are:
"""
        )
        st.latex(OUTPUT_EFFECT_EQUATIONS)

        st.markdown(
            "The optimizer uses the same gain, lag, and dead-time structure "
            "to predict future outputs. Its objective is:"
        )
        st.latex(MPC_OBJECTIVE_EQUATION)
        st.markdown(
            "Here, `J_o` is squared target error for **Target**, negative value "
            "for **Maximize**, or positive value for **Minimize**. All selected "
            "input and output limits remain active. Model mismatch is visible "
            "when fitted coefficients differ from the simulated plant."
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

        st.markdown("**Feed dry-matter disturbance model**")
        dry_matter_rows = [
            {
                "Output": name,
                "Effect per +1 % dry matter": FEED_DRY_MATTER_GAIN[i],
                "Output unit": OUTPUT_UNITS[i],
            }
            for i, name in enumerate(OUTPUT_NAMES)
        ]
        st.dataframe(dry_matter_rows, width="stretch", hide_index=True)

        st.markdown("**Inlet-air humidity disturbance model**")
        inlet_humidity_rows = [
            {
                "Output": name,
                f"Effect per +1.0 {INLET_HUMIDITY_UNIT}": INLET_HUMIDITY_GAIN[i],
                "Effect of +0.003 humid weather": (
                    INLET_HUMIDITY_GAIN[i] * HUMID_WEATHER_INCREASE
                ),
                "Output unit": OUTPUT_UNITS[i],
            }
            for i, name in enumerate(OUTPUT_NAMES)
        ]
        st.dataframe(inlet_humidity_rows, width="stretch", hide_index=True)

        st.markdown(
            """
**Mollier / stickiness diagnostic**

The operating map uses the true simulated exhaust temperature and humidity
ratio. Moist-air enthalpy is calculated as
`h = 1.006 T + w (2501 + 1.86 T)` in kJ/kg dry air. Relative-humidity
references use the Antoine saturation-pressure correlation at 101.325 kPa.
The displayed stickiness curve is a configured, product-independent model
boundary. It is not an operating limit for a real product or dryer.
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


initialize()
tank_manager: FeedTankManager = st.session_state.feed_tank_manager
weather_manager: InletWeatherManager = st.session_state.weather_manager

with st.sidebar:
    render_sidebar_title("OPERATOR STATION")
    mode = st.radio("Control mode", ("Manual", "APC"), key="mode", horizontal=True)
    left, right = st.columns(2)
    left.button(
        "RUN" if not st.session_state.running else "HOLD",
        width="stretch",
        on_click=toggle_running,
    )
    right.button("RESET", width="stretch", on_click=reset)

    render_sidebar_title("SENSOR INPUT")
    noise_enabled = st.checkbox("Measurement noise enabled", value=True)
    noise_profile = st.selectbox(
        "Noise profile", tuple(NOISE_MULTIPLIERS), index=1, disabled=not noise_enabled
    )

    render_sidebar_title("SIMULATION RATE")
    simulation_minutes_per_tick = st.selectbox(
        "Simulated minutes per scan",
        (1, 2, 5),
        format_func=lambda minutes: f"{minutes} minute{'s' if minutes != 1 else ''}",
    )
    st.session_state.dryer.configure_time_step(simulation_minutes_per_tick)
    st.session_state.controller.configure_time_step(simulation_minutes_per_tick)

    render_sidebar_title("INLET AIR WEATHER")
    weather_mode = st.selectbox(
        "Inlet humidity mode",
        WEATHER_MODES,
        key="weather_mode",
    )
    weather_manager.configure_mode(weather_mode, st.session_state.minute)
    st.session_state.dryer.set_inlet_humidity(weather_manager.inlet_humidity)
    st.button(
        "TRIGGER HUMID WEATHER",
        width="stretch",
        type="primary",
        on_click=trigger_humid_weather,
    )
    st.caption(
        f"Manual event: +{HUMID_WEATHER_INCREASE:.4f} {INLET_HUMIDITY_UNIT}"
    )

    render_sidebar_title("FEED SUPPLY")
    automatic_tank_changes = st.checkbox("Automatic tank changes", value=False)
    automatic_tank_interval = st.slider(
        "Automatic change interval (sim min)",
        30,
        120,
        60,
        5,
        disabled=not automatic_tank_changes,
    )
    tank_manager.configure_automatic(
        automatic_tank_changes,
        automatic_tank_interval,
        st.session_state.minute,
    )
    render_feed_tank_command()

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
    "SIM-01  |  Multivariable control learning station  |  "
    f"{simulation_minutes_per_tick} minute"
    f"{'s' if simulation_minutes_per_tick != 1 else ''} per scan",
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
                max_move * simulation_minutes_per_tick,
                input_enabled,
            )
        event_minute = st.session_state.minute + simulation_minutes_per_tick
        record_tank_event(tank_manager.maybe_automatic_change(event_minute))
        record_weather_event(weather_manager.advance(event_minute))
        st.session_state.dryer.set_inlet_humidity(weather_manager.inlet_humidity)
        st.session_state.true_outputs = st.session_state.dryer.advance(
            st.session_state.inputs
        )
        st.session_state.measurements = st.session_state.dryer.measure(
            enabled=noise_enabled,
            multiplier=NOISE_MULTIPLIERS[noise_profile],
        )
        st.session_state.minute = event_minute
        st.session_state.history["minute"].append(st.session_state.minute)
        for name, value in zip(INPUT_NAMES, st.session_state.inputs):
            st.session_state.history[name].append(float(value))
        for name, value in zip(OUTPUT_NAMES, st.session_state.measurements):
            st.session_state.history[name].append(float(value))
        st.session_state.history[FEED_DRY_MATTER_HISTORY_NAME].append(
            float(st.session_state.dryer.feed_dry_matter)
        )
        st.session_state.history[FEED_TANK_HISTORY_NAME].append(
            tank_manager.current_tank
        )
        st.session_state.history[INLET_HUMIDITY_HISTORY_NAME].append(
            float(weather_manager.inlet_humidity)
        )
        st.session_state.history[WEATHER_STATE_HISTORY_NAME].append(
            weather_manager.state
        )
        st.session_state.history[TRUE_EXHAUST_TEMPERATURE_HISTORY_NAME].append(
            float(st.session_state.true_outputs[0])
        )
        st.session_state.history[TRUE_EXHAUST_HUMIDITY_HISTORY_NAME].append(
            float(st.session_state.true_outputs[3])
        )
        for name in (
            INPUT_NAMES
            + OUTPUT_NAMES
            + (
                FEED_DRY_MATTER_HISTORY_NAME,
                FEED_TANK_HISTORY_NAME,
                INLET_HUMIDITY_HISTORY_NAME,
                WEATHER_STATE_HISTORY_NAME,
                TRUE_EXHAUST_TEMPERATURE_HISTORY_NAME,
                TRUE_EXHAUST_HUMIDITY_HISTORY_NAME,
            )
        ):
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
    last_tank_event: FeedTankEvent | None = st.session_state.last_tank_event
    last_weather_event: WeatherEvent | None = st.session_state.last_weather_event
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

    recent_tank_event = (
        last_tank_event is not None
        and st.session_state.minute - last_tank_event.minute <= 5
    )
    render_section_title(
        "FEED SUPPLY DISTURBANCE",
        "unmeasured by MPC | amber marks recent tank changes",
    )
    last_change_values = (
        f"{last_tank_event.old_dry_matter:.1f}>{last_tank_event.new_dry_matter:.1f}"
        if last_tank_event is not None
        else "--"
    )
    last_change_minute = (
        f"T+{last_tank_event.minute:05d}" if last_tank_event is not None else "--"
    )
    feed_values = [
        ScadaValue(
            tag="DV-201",
            label="Active feed tank",
            value=tank_manager.current_tank,
            unit="selected",
            state="warning" if recent_tank_event else "normal",
        ),
        ScadaValue(
            tag="DV-202",
            label="Incoming feed dry matter",
            value=f"{st.session_state.dryer.feed_dry_matter:.2f}",
            unit=FEED_DRY_MATTER_UNIT,
            state="warning" if recent_tank_event else "neutral",
        ),
        ScadaValue(
            tag="EV-201",
            label="Last dry-matter change",
            value=last_change_values,
            unit=FEED_DRY_MATTER_UNIT,
            state="warning" if recent_tank_event else "neutral",
        ),
        ScadaValue(
            tag="EV-202",
            label="Last change time",
            value=last_change_minute,
            unit="sim min",
            state="warning" if recent_tank_event else "neutral",
        ),
    ]
    render_value_grid(feed_values, columns=4)
    event_banner_text = (
        (
            f"{last_tank_event.event_type}: {last_tank_event.old_tank} "
            f"({last_tank_event.old_dry_matter:.1f}% DM) to "
            f"{last_tank_event.new_tank} ({last_tank_event.new_dry_matter:.1f}% DM) "
            f"at T+{last_tank_event.minute:05d}"
        )
        if recent_tank_event
        else "No recent feed-supply event"
    )
    render_event_banner(event_banner_text, active=recent_tank_event)

    weather_active = weather_manager.state != "NORMAL"
    render_section_title(
        "INLET AIR MOISTURE",
        "true plant disturbance | unmeasured by MPC",
    )
    weather_values = [
        ScadaValue(
            tag="DV-301",
            label="Inlet-air humidity",
            value=f"{weather_manager.inlet_humidity:.4f}",
            unit=INLET_HUMIDITY_UNIT,
            state="warning" if weather_active else "normal",
        ),
        ScadaValue(
            tag="HS-301",
            label="Dynamic mode",
            value=weather_manager.mode,
            unit="selected",
            state="neutral",
        ),
        ScadaValue(
            tag="WS-301",
            label="Weather state",
            value=weather_manager.state,
            unit="status",
            state="warning" if weather_active else "normal",
        ),
        ScadaValue(
            tag="EV-301",
            label="Last humid-weather event",
            value=(
                f"T+{last_weather_event.minute:05d}"
                if last_weather_event is not None
                else "--"
            ),
            unit="sim min",
            state="warning" if weather_active else "neutral",
        ),
    ]
    render_value_grid(weather_values, columns=4)

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
    future_length = len(plan_inputs) if plan_inputs is not None else 0
    predictions = {
        "times": (
            st.session_state.minute
            + simulation_minutes_per_tick * np.arange(1, future_length + 1)
        ).tolist(),
        "inputs": (
            [row.tolist() + [None, None] for row in plan_inputs]
            if plan_inputs is not None
            else []
        ),
        "outputs": (
            plan_outputs.tolist()
            if plan_outputs is not None
            else []
        ),
    }
    target = (
        {
            "group": objective_group,
            "index": objective_index,
            "value": objective_target,
        }
        if objective_mode == "target"
        else None
    )
    history_size = len(st.session_state.history["minute"])
    snapshot = (
        [chart_sample(index) for index in range(history_size)]
        if st.session_state.chart_needs_snapshot
        else None
    )
    latest_sample = chart_sample(-1) if history_size else None
    payload, next_sample_id, next_event_id = prepare_trend_payload(
        run_id=st.session_state.chart_run_id,
        sample_id=st.session_state.minute,
        sample=latest_sample,
        events=chart_event_payloads(),
        last_sample_id=st.session_state.chart_last_sample_id,
        last_event_id=st.session_state.chart_last_event_id,
        snapshot=snapshot,
        config={
            "input_names": list(INPUT_NAMES)
            + [FEED_DRY_MATTER_HISTORY_NAME, INLET_HUMIDITY_HISTORY_NAME],
            "input_units": list(INPUT_UNITS) + ["% DM", "kg/kg dry air"],
            "output_names": list(OUTPUT_NAMES),
            "output_units": list(OUTPUT_UNITS[:3]) + ["kg/kg dry air"],
            "process_input_count": len(INPUT_NAMES),
            "input_limits": {
                "lower": input_min.tolist(),
                "upper": input_max.tolist(),
            },
            "output_limits": {
                "lower": output_min.tolist(),
                "upper": output_max.tolist(),
            },
            "target": target,
        },
        predictions=predictions,
    )
    render_section_title(
        "PROCESS TRENDS",
        "left: inputs/disturbances | right: controlled outputs",
    )
    render_process_trends(payload)
    st.session_state.chart_last_sample_id = next_sample_id
    st.session_state.chart_last_event_id = next_event_id
    st.session_state.chart_needs_snapshot = False

    current_map_point = operating_point(
        st.session_state.minute,
        float(st.session_state.true_outputs[0]),
        float(st.session_state.true_outputs[3]),
    )
    map_snapshot = (
        [
            operating_map_history_sample(index)
            for index in range(max(0, history_size - 60), history_size)
        ]
        if st.session_state.operating_map_needs_snapshot
        else []
    )
    latest_map_sample = (
        operating_map_history_sample(-1) if history_size else None
    )
    map_payload, next_map_sample_id = prepare_operating_map_payload(
        run_id=st.session_state.chart_run_id,
        sample=latest_map_sample,
        current=current_map_point,
        snapshot=map_snapshot,
        last_sample_id=st.session_state.operating_map_last_sample_id,
        background=OPERATING_MAP_BACKGROUND,
    )
    map_state = {
        "SAFE": "normal",
        "APPROACHING": "warning",
        "STICKY RISK": "alarm",
    }[str(current_map_point["status"])]
    with st.expander("MOLLIER / STICKINESS MAP", expanded=False):
        render_value_grid(
            [
                ScadaValue(
                    tag="MAP-101",
                    label="Exhaust temperature",
                    value=f"{current_map_point['temperature']:.2f}",
                    unit="C true plant",
                    state=map_state,
                ),
                ScadaValue(
                    tag="MAP-102",
                    label="Exhaust humidity ratio",
                    value=f"{current_map_point['humidity']:.4f}",
                    unit="kg water/kg dry air",
                    state=map_state,
                ),
                ScadaValue(
                    tag="MAP-103",
                    label="Moist-air enthalpy",
                    value=f"{current_map_point['enthalpy']:.1f}",
                    unit="kJ/kg dry air",
                    state="neutral",
                ),
                ScadaValue(
                    tag="MAP-104",
                    label="Stickiness margin",
                    value=f"{current_map_point['margin']:+.1f}",
                    unit=f"C | {current_map_point['status']}",
                    state=map_state,
                ),
            ],
            columns=4,
        )
        render_message(
            "OPERATING MAP",
            f"Configured boundary | {STANDARD_PRESSURE_KPA:.3f} kPa | "
            f"range w={MAP_HUMIDITY_RANGE[0]:.2f}-{MAP_HUMIDITY_RANGE[1]:.2f}, "
            f"T={MAP_TEMPERATURE_RANGE[0]:.0f}-{MAP_TEMPERATURE_RANGE[1]:.0f} C",
        )
        render_operating_map(map_payload)
    st.session_state.operating_map_last_sample_id = next_map_sample_id
    st.session_state.operating_map_needs_snapshot = False

    with st.expander("PROCESS DISTURBANCE EVENT LOG"):
        event_rows = [
            {
                "Sim minute": event.minute,
                "Event": event.event_type,
                "From": f"{event.old_tank} ({event.old_dry_matter:.1f}% DM)",
                "To": f"{event.new_tank} ({event.new_dry_matter:.1f}% DM)",
            }
            for event in reversed(st.session_state.tank_events)
        ]
        event_rows.extend(
            {
                "Sim minute": event.minute,
                "Event": event.event_type,
                "From": "NORMAL",
                "To": f"+{event.humidity_increase:.4f} {INLET_HUMIDITY_UNIT}",
            }
            for event in st.session_state.weather_events
        )
        event_rows.sort(key=lambda row: int(row["Sim minute"]), reverse=True)
        st.dataframe(
            pd.DataFrame(
                event_rows,
                columns=("Sim minute", "Event", "From", "To"),
            ),
            width="stretch",
            hide_index=True,
            height=210,
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
