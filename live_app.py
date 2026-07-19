"""Live, interactive spray-dryer APC simulation."""

import numpy as np
import pandas as pd
import streamlit as st

from apc_lab.commissioning import (
    PERIOD_ESTIMATION,
    PERIOD_VALIDATION,
    TuningPreset,
    apply_model,
    build_guided_plan,
    compare_tunings,
    diagnose_excitation,
    model_from_controller,
    rate_limited_inputs,
    restore_builtin_model,
    sample_record,
    samples_to_dataframe,
    split_estimation_validation,
    validate_candidate,
)
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
    maximum_safe_humidity_ratio,
    moist_air_enthalpy,
    psychrometric_background,
)
from apc_lab.scada_ui import (
    ScadaValue,
    apply_scada_theme,
    constraint_state,
    render_message,
    render_parameter_table,
    render_section_title,
    render_showcase_banner,
    render_sidebar_title,
    render_status_strip,
    render_title_bar,
    render_value_grid,
)
from apc_lab.showcase import (
    ACTION_APC_CHALLENGE,
    ACTION_APC_ENABLE,
    ACTION_COMPLETE,
    ACTION_TANK_CHANGE,
    SHOWCASE_SCAN_MINUTES,
    ShowcaseEvent,
    ShowcasePhase,
    ShowcaseState,
    apply_showcase_controller_tuning,
    calculate_showcase_metrics,
)

st.set_page_config(page_title="Spray Dryer APC Station", layout="wide")
apply_scada_theme()

DEFAULT_MAX_EXHAUST_TEMPERATURE = 100.0
DEFAULT_OUTPUT_MIN = np.array([75.0, 75.0, 3.0, 0.090])
DEFAULT_OUTPUT_MAX = np.array(
    [
        DEFAULT_MAX_EXHAUST_TEMPERATURE,
        137.5,
        5.2,
        maximum_safe_humidity_ratio(DEFAULT_MAX_EXHAUST_TEMPERATURE),
    ]
)
CONTROLLER_VERSION = 6
SIMULATION_STATE_VERSION = 8
FEED_DRY_MATTER_HISTORY_NAME = "Feed dry matter"
FEED_TANK_HISTORY_NAME = "Feed tank"
INLET_HUMIDITY_HISTORY_NAME = "Inlet air humidity"
WEATHER_STATE_HISTORY_NAME = "Weather state"
TRUE_EXHAUST_TEMPERATURE_HISTORY_NAME = "True exhaust air temperature"
TRUE_EXHAUST_HUMIDITY_HISTORY_NAME = "True exhaust air humidity"


def empty_live_history() -> dict[str, list[object]]:
    """Create bounded station history without coupling commissioning data to it."""

    return {
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
            "showcase",
            "showcase_events",
            "showcase_event_ids",
            "showcase_handoff_notice",
            "showcase_handoff_pending",
            "control_mode",
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
        st.session_state.showcase = ShowcaseState()
        st.session_state.showcase_events = []
        st.session_state.showcase_handoff_notice = False
        st.session_state.showcase_handoff_pending = False
        st.session_state.history = empty_live_history()
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
        st.session_state.showcase_event_ids = []
        st.session_state.chart_needs_snapshot = True
        st.session_state.operating_map_last_sample_id = -1
        st.session_state.operating_map_needs_snapshot = True
    if "control_mode" not in st.session_state:
        st.session_state.control_mode = st.session_state.get("mode", "Manual")
    if st.session_state.get("controller_version") != CONTROLLER_VERSION:
        st.session_state.controller = ConstrainedDryerMPC()
        st.session_state.controller_version = CONTROLLER_VERSION
        for key in (
            "fitted_model",
            "commissioning_candidate",
            "commissioning_candidate_revision",
            "commissioning_validation",
            "commissioning_fit_data",
            "commissioning_comparison",
        ):
            st.session_state.pop(key, None)
        st.session_state.active_model_source = "Built-in model"
        st.session_state.active_model_revision = "BUILT-IN"
    commissioning_defaults = {
        "workspace": "APC Station",
        "commissioning_samples": [],
        "commissioning_plan": [],
        "commissioning_plan_index": 0,
        "commissioning_collecting": False,
        "commissioning_prepared": False,
        "commissioning_candidate": None,
        "commissioning_candidate_revision": None,
        "commissioning_validation": None,
        "commissioning_fit_data": None,
        "commissioning_diagnostics": None,
        "commissioning_revision_sequence": 0,
        "commissioning_tuning_a": None,
        "commissioning_tuning_b": None,
        "commissioning_comparison": None,
        "active_model_source": "Built-in model",
        "active_model_revision": "BUILT-IN",
    }
    for key, value in commissioning_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


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
        "showcase",
        "showcase_events",
        "showcase_event_ids",
        "showcase_handoff_notice",
        "showcase_handoff_pending",
        "control_mode",
        "workspace",
        "commissioning_samples",
        "commissioning_plan",
        "commissioning_plan_index",
        "commissioning_collecting",
        "commissioning_prepared",
        "commissioning_candidate",
        "commissioning_candidate_revision",
        "commissioning_validation",
        "commissioning_fit_data",
        "commissioning_diagnostics",
        "commissioning_revision_sequence",
        "commissioning_tuning_a",
        "commissioning_tuning_b",
        "commissioning_comparison",
        "active_model_source",
        "active_model_revision",
        "commissioning_target_output",
        "commissioning_target_value",
        "commissioning_objective_weight",
        "commissioning_move_weight",
        "commissioning_prediction_horizon",
        "commissioning_target_value_0",
        "commissioning_target_value_1",
        "commissioning_target_value_2",
        "commissioning_target_value_3",
        "commissioning_dataset_source",
        "commissioning_upload",
        "commissioning_upload_sample_minutes",
        "commissioning_estimation_fraction",
        "commissioning_max_delay_minutes",
        "fitted_model",
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


def record_showcase_event(event: ShowcaseEvent) -> None:
    """Retain one guided-scenario marker in the shared SCADA event sequence."""

    st.session_state.chart_event_sequence += 1
    st.session_state.showcase_events.append(event)
    st.session_state.showcase_event_ids.append(st.session_state.chart_event_sequence)
    st.session_state.showcase_events = st.session_state.showcase_events[-12:]
    st.session_state.showcase_event_ids = st.session_state.showcase_event_ids[-12:]


def toggle_running() -> None:
    if (
        not st.session_state.running
        and st.session_state.get("workspace") == "Commissioning Lab"
        and st.session_state.get("commissioning_prepared", False)
        and st.session_state.commissioning_plan_index
        < len(st.session_state.commissioning_plan)
    ):
        st.session_state.commissioning_collecting = True
    st.session_state.running = not st.session_state.running


def prepare_commissioning_experiment() -> None:
    """Prepare a deterministic manual experiment from a reset plant state."""

    sample_minutes = float(st.session_state.simulation_minutes_per_tick)
    st.session_state.dryer = LiveSprayDryer()
    st.session_state.dryer.configure_time_step(sample_minutes)
    st.session_state.inputs = NOMINAL_INPUTS.copy()
    st.session_state.true_outputs = NOMINAL_OUTPUTS.copy()
    st.session_state.measurements = NOMINAL_OUTPUTS.copy()
    st.session_state.minute = 0
    st.session_state.history = empty_live_history()
    st.session_state.feed_tank_manager = FeedTankManager()
    st.session_state.tank_events = []
    st.session_state.last_tank_event = None
    st.session_state.weather_manager = InletWeatherManager()
    st.session_state.weather_events = []
    st.session_state.last_weather_event = None
    st.session_state.chart_run_sequence = st.session_state.get(
        "chart_run_sequence", 0
    ) + 1
    st.session_state.chart_run_id = st.session_state.chart_run_sequence
    st.session_state.chart_last_sample_id = -1
    st.session_state.chart_last_event_id = 0
    st.session_state.chart_event_sequence = 0
    st.session_state.chart_event_ids = []
    st.session_state.weather_event_ids = []
    st.session_state.chart_needs_snapshot = True
    st.session_state.operating_map_last_sample_id = -1
    st.session_state.operating_map_needs_snapshot = True
    st.session_state.commissioning_plan = build_guided_plan(sample_minutes)
    st.session_state.commissioning_plan_index = 0
    st.session_state.commissioning_samples = []
    st.session_state.commissioning_collecting = False
    st.session_state.commissioning_prepared = True
    st.session_state.commissioning_candidate = None
    st.session_state.commissioning_candidate_revision = None
    st.session_state.commissioning_validation = None
    st.session_state.commissioning_fit_data = None
    st.session_state.commissioning_diagnostics = None
    st.session_state.commissioning_comparison = None
    st.session_state.mode = "Manual"
    st.session_state.control_mode = "Manual"
    st.session_state.weather_mode = "Constant"
    st.session_state.automatic_tank_changes = False
    st.session_state.running = False


def start_commissioning_collection() -> None:
    """Start or resume the prepared experiment through the normal RUN state."""

    if not st.session_state.commissioning_prepared:
        return
    if st.session_state.commissioning_plan_index >= len(
        st.session_state.commissioning_plan
    ):
        return
    st.session_state.commissioning_collecting = True
    st.session_state.running = True
    st.session_state.mode = "Manual"
    st.session_state.control_mode = "Manual"


def current_commissioning_tuning() -> TuningPreset:
    output_name = st.session_state.commissioning_target_output
    output_index = OUTPUT_NAMES.index(output_name)
    return TuningPreset(
        target_output_index=output_index,
        target=float(st.session_state[f"commissioning_target_value_{output_index}"]),
        objective_weight=float(st.session_state.commissioning_objective_weight),
        move_weight=float(st.session_state.commissioning_move_weight),
        prediction_horizon=int(st.session_state.commissioning_prediction_horizon),
    )


def apply_commissioning_tuning() -> None:
    """Apply the current scalar tuning through the shared live MPC."""

    tuning = current_commissioning_tuning()
    controller = st.session_state.controller
    controller.objective_weight = tuning.objective_weight
    controller.move_weight = tuning.move_weight
    controller.prediction_horizon = tuning.prediction_horizon
    output_name = OUTPUT_NAMES[tuning.target_output_index]
    st.session_state.objective_parameter = output_name
    st.session_state.objective_mode = "Target"
    st.session_state[f"objective_target_output_{tuning.target_output_index}"] = (
        tuning.target
    )


def save_commissioning_tuning(slot: str) -> None:
    st.session_state[f"commissioning_tuning_{slot.lower()}"] = (
        current_commissioning_tuning()
    )


def widget_default(key: str, value: object) -> object | None:
    """Supply a widget default only before Session State owns its key."""

    return value if key not in st.session_state else None


def _set_showcase_widget_defaults() -> None:
    """Apply the known operator configuration used by the guided run."""

    st.session_state.workspace = "APC Station"
    st.session_state.mode = "Manual"
    st.session_state.control_mode = "Manual"
    st.session_state.noise_enabled = True
    st.session_state.noise_profile = "Normal"
    st.session_state.simulation_minutes_per_tick = SHOWCASE_SCAN_MINUTES
    st.session_state.weather_mode = "Constant"
    st.session_state.automatic_tank_changes = False
    st.session_state.automatic_tank_interval = 60
    st.session_state.objective_parameter = "Powder moisture"
    st.session_state.objective_mode = "Target"
    st.session_state.objective_target_output_2 = float(NOMINAL_OUTPUTS[2])
    showcase_max_moves = (2.5, 0.05, 1.0)
    for index, value in enumerate(NOMINAL_INPUTS):
        st.session_state[f"input_{index}"] = float(value)
        st.session_state[f"enabled_{index}"] = True
        st.session_state[f"input_constraint_{index}"] = (
            float(INPUT_MIN[index]),
            float(INPUT_MAX[index]),
        )
        st.session_state[f"max_move_{index}"] = showcase_max_moves[index]
    for index in range(len(OUTPUT_NAMES) - 1):
        st.session_state[f"constraint_{index}"] = (
            float(DEFAULT_OUTPUT_MIN[index]),
            float(DEFAULT_OUTPUT_MAX[index]),
        )
    st.session_state.constraint_3_lower = float(DEFAULT_OUTPUT_MIN[3])


def start_showcase() -> None:
    """Begin automation over the current reset operator session."""

    _set_showcase_widget_defaults()
    st.session_state.weather_manager.configure_mode("Constant", st.session_state.minute)
    apply_showcase_controller_tuning(st.session_state.controller)
    st.session_state.showcase.start()
    st.session_state.showcase_handoff_notice = False
    record_showcase_event(
        ShowcaseEvent(0, "SHOWCASE START", "Manual baseline at nominal inputs")
    )
    if not st.session_state.running:
        toggle_running()


def release_showcase() -> None:
    """Release scheduled automation without changing the running process."""

    st.session_state.showcase.stop()
    st.session_state.showcase_handoff_notice = True
    st.session_state.showcase_handoff_pending = True


def stop_showcase() -> None:
    """Cancel automation while leaving the current operator session intact."""

    st.session_state.showcase.stop()


def apply_showcase_actions(actions: list[str], minute: int) -> None:
    """Apply due guided actions through existing disturbance/control paths."""

    tank_manager: FeedTankManager = st.session_state.feed_tank_manager
    for action in actions:
        if action == ACTION_TANK_CHANGE:
            st.session_state.manual_tank = "Tank C"
            change_selected_tank("Showcase tank change")
        elif action == ACTION_APC_ENABLE:
            st.session_state.mode = "APC"
            st.session_state.control_mode = "APC"
            record_showcase_event(
                ShowcaseEvent(minute, "APC ENABLED", "Controller takeover")
            )
        elif action == ACTION_APC_CHALLENGE:
            st.session_state.manual_tank = "Tank A"
            change_selected_tank("APC challenge tank change")
        elif action == ACTION_COMPLETE:
            record_showcase_event(
                ShowcaseEvent(
                    minute,
                    "SHOWCASE COMPLETE",
                    "APC active; operator control released",
                )
            )
            release_showcase()


def change_selected_tank(event_type: str = "Manual tank change") -> None:
    tank_manager: FeedTankManager = st.session_state.feed_tank_manager
    selected_tank = st.session_state.manual_tank
    record_tank_event(
        tank_manager.change_to(
            selected_tank,
            st.session_state.minute,
            event_type,
        )
    )


def trigger_humid_weather() -> None:
    weather_manager: InletWeatherManager = st.session_state.weather_manager
    record_weather_event(weather_manager.trigger(st.session_state.minute))


@st.fragment
def render_feed_tank_command(disabled: bool = False) -> None:
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
        disabled=disabled,
    )
    st.button(
        "CHANGE FEED TANK",
        width="stretch",
        disabled=disabled or selected_tank == tank_manager.current_tank,
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
    events.extend(
        {
            "event_id": event_id,
            "time": event.minute,
            "type": event.event_type,
            "label": event.event_type,
        }
        for event_id, event in zip(
            st.session_state.showcase_event_ids,
            st.session_state.showcase_events,
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
Inlet-air humidity is represented as kg water per kg dry air. Sensible heating
does not change this humidity ratio; humid weather therefore adds water load at
the dryer inlet.
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
                f"Effect of +{HUMID_WEATHER_INCREASE:.3f} humid weather": (
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
The displayed stickiness curve is a configured example boundary. It is not a
validated correlation or operating limit for a real product or dryer.
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


def render_commissioning_lab(
    input_min: np.ndarray,
    input_max: np.ndarray,
    max_move: np.ndarray,
    output_min: np.ndarray,
    output_max: np.ndarray,
    sample_minutes: float,
    noise_enabled: bool,
    noise_profile: str,
) -> None:
    """Render the focused data-to-controller commissioning workflow."""

    prepared = bool(st.session_state.commissioning_prepared)
    plan = st.session_state.commissioning_plan
    plan_index = int(st.session_state.commissioning_plan_index)
    collection_complete = prepared and plan_index >= len(plan)
    candidate = st.session_state.commissioning_candidate
    validation = st.session_state.commissioning_validation
    tuning_a = st.session_state.commissioning_tuning_a
    tuning_b = st.session_state.commissioning_tuning_b

    render_status_strip(
        [
            (
                "PROCESS",
                "RUN" if st.session_state.running else "HOLD",
                "normal" if st.session_state.running else "warning",
            ),
            ("CONTROL", "MANUAL", "neutral"),
            (
                "MODEL",
                st.session_state.active_model_revision,
                "normal" if st.session_state.active_model_revision != "BUILT-IN" else "neutral",
            ),
            ("SAMPLES", str(len(st.session_state.commissioning_samples)), "neutral"),
            ("SCAN", f"{sample_minutes:g} MIN", "neutral"),
        ]
    )
    render_message(
        "COMMISSIONING WORKFLOW",
        "1 Collect data  |  2 Fit candidate  |  3 Validate  |  "
        "4 Apply predictor  |  5 Tune and compare",
    )
    if st.session_state.running:
        next_action = (
            "Collection is running. Watch the phase and sample counter below, "
            "or press HOLD in the sidebar to pause."
        )
    elif not prepared:
        next_action = (
            "Start from RESET, choose the scan duration and sensor-noise setting "
            "in the sidebar, then select PREPARE GUIDED EXPERIMENT."
        )
    elif not collection_complete:
        next_action = "Select START / RESUME COLLECTION. The input steps are automatic."
    elif candidate is None:
        next_action = "The dataset is complete. Select FIT CANDIDATE MODEL in step 2."
    elif validation is None:
        next_action = "The candidate is fitted but inactive. Select VALIDATE CANDIDATE."
    elif (
        st.session_state.active_model_revision
        != st.session_state.commissioning_candidate_revision
    ):
        next_action = "Validation is complete. Select APPLY TO MPC when ready."
    elif tuning_a is None or tuning_b is None:
        next_action = (
            "The fitted predictor is active. Adjust tuning, save two alternatives "
            "as A and B, then compare them."
        )
    else:
        next_action = "Tuning A and B are ready. Select RUN FAIR A/B COMPARISON."
    render_message("NEXT ACTION", next_action)

    with st.expander("HOW TO USE THE COMMISSIONING LAB", expanded=True):
        st.markdown(
            """
1. **RESET and configure:** choose a 1, 2, or 5 minute scan and the desired
   sensor-noise level in the sidebar. The scan duration is also the dataset
   sample time.
2. **Prepare and collect:** preparation creates a clean nominal experiment and
   clears earlier lab samples. Start runs a fixed sequence of positive and
   negative input steps in Manual mode. **HOLD** pauses; **RUN** resumes.
3. **Fit:** the estimation samples create a candidate gain, time-constant, and
   dead-time model. Fitting does not change the active MPC.
4. **Validate and apply:** validation predicts data that was kept out of the
   fit. Apply deliberately copies a validated candidate into the MPC predictor;
   the simulated plant is never changed.
5. **Tune and compare:** save two controller settings, then run the same tank
   disturbance offline for both. The offline comparison does not move the live
   process.

You can use an uploaded CSV instead of collecting guided data. Uploaded data
must contain the seven MV/CV columns shown in the README and an explicit sample
duration.
"""
        )

    render_section_title("1. COLLECT COMMISSIONING DATA")
    st.caption(
        "PREPARE builds the experiment but does not start it. START / RESUME "
        "runs the normal simulation and records measured CVs while one MV at a "
        "time is moved automatically. Tank and weather events remain disabled."
    )
    reset_ready = st.session_state.minute == 0 and not st.session_state.history["minute"]
    prepare_col, run_col = st.columns([1, 1])
    prepare_col.button(
        "PREPARE GUIDED EXPERIMENT",
        width="stretch",
        on_click=prepare_commissioning_experiment,
        disabled=not reset_ready or st.session_state.running,
        help=(
            "Create the deterministic estimation/validation step sequence and "
            "clear previous Commissioning Lab samples. This does not start RUN."
        ),
    )
    run_col.button(
        "START / RESUME COLLECTION",
        width="stretch",
        type="primary",
        on_click=start_commissioning_collection,
        disabled=not prepared or collection_complete or st.session_state.running,
        help=(
            "Start or resume automatic Manual-mode MV excitation and record one "
            "measurement row per simulation scan."
        ),
    )
    if not reset_ready and not prepared:
        st.info("Use RESET before preparing a new guided experiment.")
    if prepared:
        progress = 1.0 if not plan else min(1.0, plan_index / len(plan))
        st.progress(progress)
        if collection_complete:
            st.success("Estimation and validation collection complete. Process is in HOLD.")
        else:
            current_step = plan[plan_index]
            st.caption(
                f"{current_step.period} | {current_step.phase} | "
                f"sample {plan_index + 1} of {len(plan)}"
            )
    collected = samples_to_dataframe(st.session_state.commissioning_samples)
    if not collected.empty:
        period_counts = collected["Period"].value_counts()
        st.write(
            f"Estimation samples: **{int(period_counts.get(PERIOD_ESTIMATION, 0))}** | "
            f"Validation samples: **{int(period_counts.get(PERIOD_VALIDATION, 0))}**"
        )
        st.download_button(
            "DOWNLOAD COMMISSIONING CSV",
            collected.to_csv(index=False),
            "spray_dryer_commissioning.csv",
            "text/csv",
        )

    render_section_title("2. FIT CANDIDATE MODEL")
    st.caption(
        "Guided data uses its labelled Estimation and Validation periods. For an "
        "uploaded CSV, the chronological estimation fraction keeps the later "
        "samples separate for validation. FIT creates a candidate only."
    )
    source = st.radio(
        "Dataset source",
        ("Guided commissioning buffer", "Uploaded CSV"),
        horizontal=True,
        key="commissioning_dataset_source",
    )
    dataset: pd.DataFrame | None = collected if source.startswith("Guided") else None
    fit_sample_minutes = float(sample_minutes)
    estimation_fraction = 0.7
    if source == "Uploaded CSV":
        uploaded = st.file_uploader(
            "Upload timestamp-ordered CSV",
            type="csv",
            key="commissioning_upload",
        )
        fit_sample_minutes = float(
            st.number_input(
                "Uploaded-data sample duration (minutes)",
                min_value=0.1,
                max_value=60.0,
                value=1.0,
                step=0.1,
                key="commissioning_upload_sample_minutes",
            )
        )
        estimation_fraction = st.slider(
            "Chronological estimation fraction",
            0.5,
            0.9,
            0.7,
            0.05,
            key="commissioning_estimation_fraction",
        )
        if uploaded is not None:
            try:
                dataset = pd.read_csv(uploaded)
            except Exception as error:
                st.error(f"Could not read CSV: {error}")
    elif not collected.empty:
        sample_values = collected["Sample duration (min)"].astype(float).unique()
        if len(sample_values) == 1:
            fit_sample_minutes = float(sample_values[0])
    max_delay_minutes = st.slider(
        "Maximum dead time to search (minutes)",
        0,
        30,
        10,
        key="commissioning_max_delay_minutes",
    )
    st.caption(
        "Maximum dead time is the longest input-to-output delay the fitter is "
        "allowed to test. Ten minutes is a sensible starting value for this lab."
    )
    if dataset is not None and not dataset.empty:
        st.dataframe(dataset.head(12), width="stretch", hide_index=True)
    guided_dataset_incomplete = (
        source == "Guided commissioning buffer" and not collection_complete
    )
    fit_requested = st.button(
        "FIT CANDIDATE MODEL",
        type="primary",
        disabled=(
            st.session_state.running
            or dataset is None
            or dataset.empty
            or guided_dataset_incomplete
        ),
        help=(
            "Estimate the 4 x 3 gains, one time constant per CV, and one shared "
            "dead time from Estimation data. The active MPC remains unchanged."
        ),
    )
    if fit_requested and dataset is not None:
        try:
            estimation_data, validation_data = split_estimation_validation(
                dataset, estimation_fraction
            )
            estimation_inputs, estimation_outputs = arrays_from_dataframe(
                estimation_data
            )
            validation_inputs, validation_outputs = arrays_from_dataframe(
                validation_data
            )
            diagnostics = diagnose_excitation(estimation_inputs, estimation_outputs)
            st.session_state.commissioning_diagnostics = diagnostics
            if diagnostics.blocking_errors:
                st.session_state.commissioning_candidate = None
                st.session_state.commissioning_validation = None
            else:
                maximum_delay_steps = int(
                    np.ceil(max_delay_minutes / fit_sample_minutes)
                )
                candidate = fit_dynamic_model(
                    estimation_inputs,
                    estimation_outputs,
                    max_delay=maximum_delay_steps,
                    dt=fit_sample_minutes,
                )
                st.session_state.commissioning_revision_sequence += 1
                revision = (
                    f"FIT-{st.session_state.commissioning_revision_sequence:03d}"
                )
                st.session_state.commissioning_candidate = candidate
                st.session_state.commissioning_candidate_revision = revision
                st.session_state.commissioning_validation = None
                st.session_state.commissioning_fit_data = {
                    "estimation_data": estimation_data,
                    "validation_data": validation_data,
                    "estimation_inputs": estimation_inputs,
                    "estimation_outputs": estimation_outputs,
                    "validation_inputs": validation_inputs,
                    "validation_outputs": validation_outputs,
                    "sample_minutes": fit_sample_minutes,
                }
                st.success(f"Candidate {revision} fitted. The active MPC model is unchanged.")
        except Exception as error:
            st.error(str(error))

    diagnostics = st.session_state.commissioning_diagnostics
    if diagnostics is not None:
        for message in diagnostics.blocking_errors:
            st.error(message)
        for message in diagnostics.warnings:
            st.warning(message)
        st.caption(
            f"MV matrix rank {diagnostics.input_rank}/3 | "
            f"condition number {diagnostics.condition_number:.1f}"
        )

    candidate = st.session_state.commissioning_candidate
    validation = st.session_state.commissioning_validation
    if candidate is not None:
        model_rows = []
        for output_index, output_name in enumerate(OUTPUT_NAMES):
            model_rows.append(
                {
                    "Output": output_name,
                    "Time constant (min)": candidate.output_tau[output_index],
                    "Derivative-fit RMSE": candidate.rmse[output_index],
                    **{
                        f"Gain from {name}": candidate.gain_matrix[output_index, index]
                        for index, name in enumerate(INPUT_NAMES)
                    },
                }
            )
        st.dataframe(model_rows, width="stretch", hide_index=True)
        st.caption(
            f"Candidate {st.session_state.commissioning_candidate_revision} | "
            f"dead time {candidate.delay_minutes:.2f} min "
            f"({candidate.delay_steps} fitted samples) | derivative-fit RMSE is "
            "the regression error, not free-run output error."
        )

    render_section_title("3. VALIDATE AND APPLY")
    st.markdown(
        "**VALIDATE** tests free-run predictions on separate data. "
        "**APPLY TO MPC** activates the validated predictor. "
        "**RESTORE BUILT-IN** removes the fitted predictor from the controller. "
        "All three actions require process HOLD."
    )
    validate_col, apply_col, restore_col = st.columns(3)
    validate_requested = validate_col.button(
        "VALIDATE CANDIDATE",
        width="stretch",
        disabled=st.session_state.running or candidate is None,
        help=(
            "Generate measured-versus-predicted responses and separate "
            "estimation/validation error metrics. This does not activate the model."
        ),
    )
    if validate_requested:
        fit_data = st.session_state.commissioning_fit_data
        try:
            st.session_state.commissioning_validation = validate_candidate(
                candidate,
                fit_data["estimation_inputs"],
                fit_data["estimation_outputs"],
                fit_data["validation_inputs"],
                fit_data["validation_outputs"],
                fit_data["sample_minutes"],
            )
            validation = st.session_state.commissioning_validation
        except Exception as error:
            st.error(str(error))
    apply_requested = apply_col.button(
        "APPLY TO MPC",
        width="stretch",
        type="primary",
        disabled=st.session_state.running or candidate is None or validation is None,
        help=(
            "Copy the validated candidate into the shared MPC predictor. The "
            "synthetic plant and collected data are unchanged."
        ),
    )
    if apply_requested:
        apply_model(st.session_state.controller, candidate)
        st.session_state.active_model_source = "Validated fitted model"
        st.session_state.active_model_revision = (
            st.session_state.commissioning_candidate_revision
        )
        st.success("Validated candidate applied to the MPC predictor only.")
    restore_requested = restore_col.button(
        "RESTORE BUILT-IN",
        width="stretch",
        disabled=(
            st.session_state.running
            or st.session_state.active_model_revision == "BUILT-IN"
        ),
        help="Restore every built-in MPC predictor parameter.",
    )
    if restore_requested:
        restore_builtin_model(st.session_state.controller)
        st.session_state.active_model_source = "Built-in model"
        st.session_state.active_model_revision = "BUILT-IN"
        st.success("Built-in MPC predictor restored. The candidate remains available.")
    render_message(
        "MODEL LIFECYCLE",
        f"Active: {st.session_state.active_model_source} "
        f"[{st.session_state.active_model_revision}] | "
        f"Candidate: {st.session_state.commissioning_candidate_revision or '--'} | "
        f"Validation: {'complete' if validation is not None else 'required'}",
    )

    if validation is not None:
        metric_rows = []
        for output_index, output_name in enumerate(OUTPUT_NAMES):
            metric_rows.append(
                {
                    "Output": output_name,
                    "Estimation RMSE": validation.estimation.rmse[output_index],
                    "Validation RMSE": validation.validation.rmse[output_index],
                    "Estimation MAE": validation.estimation.mae[output_index],
                    "Validation MAE": validation.validation.mae[output_index],
                    "Estimation fit (%)": validation.estimation.fit_percent[output_index],
                    "Validation fit (%)": validation.validation.fit_percent[output_index],
                    "Unit": OUTPUT_UNITS[output_index],
                }
            )
        st.dataframe(metric_rows, width="stretch", hide_index=True)
        fit_data = st.session_state.commissioning_fit_data
        estimation_outputs = fit_data["estimation_outputs"]
        validation_outputs = fit_data["validation_outputs"]
        split = len(estimation_outputs)
        for output_index, output_name in enumerate(OUTPUT_NAMES):
            overlay = pd.DataFrame(
                {
                    "Measured": np.r_[
                        estimation_outputs[:, output_index],
                        validation_outputs[:, output_index],
                    ],
                    "Predicted": np.r_[
                        validation.estimation.predictions[:, output_index],
                        validation.validation.predictions[:, output_index],
                    ],
                    "Period": [PERIOD_ESTIMATION] * split
                    + [PERIOD_VALIDATION] * len(validation_outputs),
                }
            )
            st.caption(f"{output_name} measured versus free-run predicted")
            st.line_chart(overlay[["Measured", "Predicted"]], height=180)
        st.caption(
            f"The chronological boundary is after sample {split}. Metrics above "
            "report estimation and validation periods separately."
        )

    render_section_title("4. TUNE THE CURRENT MPC")
    st.caption(
        "Choose the CV that should follow a target. A higher CV tracking weight "
        "pushes harder toward that target. A higher MV move penalty gives smoother, "
        "slower commands. A longer prediction horizon looks further ahead but costs "
        "more calculation. The control horizon remains fixed."
    )
    controller = st.session_state.controller
    target_output = st.selectbox(
        "Target-controlled CV",
        OUTPUT_NAMES,
        index=widget_default("commissioning_target_output", 2),
        key="commissioning_target_output",
    )
    target_index = OUTPUT_NAMES.index(target_output)
    target_key = f"commissioning_target_value_{target_index}"
    target_value = st.number_input(
        f"{target_output} target ({OUTPUT_UNITS[target_index]})",
        value=widget_default(target_key, float(NOMINAL_OUTPUTS[target_index])),
        key=target_key,
    )
    st.session_state.commissioning_target_value = float(target_value)
    tuning_columns = st.columns(3)
    tuning_columns[0].number_input(
        "CV tracking weight",
        min_value=1.0,
        max_value=5000.0,
        value=widget_default(
            "commissioning_objective_weight", float(controller.objective_weight)
        ),
        step=10.0,
        key="commissioning_objective_weight",
    )
    tuning_columns[1].number_input(
        "MV move penalty",
        min_value=0.001,
        max_value=10.0,
        value=widget_default(
            "commissioning_move_weight", float(controller.move_weight)
        ),
        step=0.01,
        format="%.3f",
        key="commissioning_move_weight",
    )
    tuning_columns[2].number_input(
        "Prediction horizon (scans)",
        min_value=max(5, controller.control_horizon),
        max_value=40,
        value=widget_default(
            "commissioning_prediction_horizon", int(controller.prediction_horizon)
        ),
        step=1,
        key="commissioning_prediction_horizon",
    )
    apply_tuning_col, save_a_col, save_b_col = st.columns(3)
    apply_tuning_col.button(
        "APPLY TUNING TO MPC",
        width="stretch",
        on_click=apply_commissioning_tuning,
        disabled=st.session_state.running,
        help=(
            "Use these values in the shared live MPC. The process must be in HOLD."
        ),
    )
    save_a_col.button(
        "SAVE AS TUNING A",
        width="stretch",
        on_click=save_commissioning_tuning,
        args=("a",),
        help="Store the displayed settings as comparison preset A; do not apply them.",
    )
    save_b_col.button(
        "SAVE AS TUNING B",
        width="stretch",
        on_click=save_commissioning_tuning,
        args=("b",),
        help="Store the displayed settings as comparison preset B; do not apply them.",
    )
    tuning_a = st.session_state.commissioning_tuning_a
    tuning_b = st.session_state.commissioning_tuning_b
    st.caption(
        f"Tuning A: {tuning_a or '--'} | Tuning B: {tuning_b or '--'}"
    )

    render_section_title("5. COMPARE TUNINGS")
    st.caption(
        "Both presets start from fresh identical plant/controller state and use "
        "the same active predictor, seed, noise, Tank C disturbance, constraints, "
        "and no weather event. Lower recovery time and error are better; zero "
        "constraint violations is preferred. MV movement shows the cost of a more "
        "aggressive response. The shared live session is not changed."
    )
    compare_requested = st.button(
        "RUN FAIR A/B COMPARISON",
        type="primary",
        disabled=st.session_state.running or tuning_a is None or tuning_b is None,
        help=(
            "Run both saved presets offline from identical initial conditions. "
            "This does not advance or modify the live process."
        ),
    )
    if compare_requested:
        try:
            st.session_state.commissioning_comparison = compare_tunings(
                tuning_a,
                tuning_b,
                model_from_controller(st.session_state.controller),
                output_min,
                output_max,
                input_min,
                input_max,
                max_move,
                sample_minutes,
                NOISE_MULTIPLIERS[noise_profile] if noise_enabled else 0.0,
            )
        except Exception as error:
            st.error(str(error))
    comparison = st.session_state.commissioning_comparison
    if comparison is not None:
        rows = []
        for label, run in (("Tuning A", comparison.tuning_a), ("Tuning B", comparison.tuning_b)):
            metrics = run.metrics
            rows.append(
                {
                    "Preset": label,
                    "Recovery (min)": metrics.recovery_minutes,
                    "Target CV IAE": metrics.integrated_absolute_error,
                    "Normalized CV error": metrics.normalized_cv_error,
                    "Max normalized constraint violation": metrics.maximum_constraint_violation,
                    "Constraint violation count": metrics.constraint_violation_count,
                    "Normalized MV movement": metrics.normalized_mv_movement,
                }
            )
        st.dataframe(rows, width="stretch", hide_index=True)
        output_chart = pd.DataFrame({"Minute": comparison.tuning_a.minutes})
        input_chart = pd.DataFrame({"Minute": comparison.tuning_a.minutes})
        for label, run in (("A", comparison.tuning_a), ("B", comparison.tuning_b)):
            for index, name in enumerate(OUTPUT_NAMES):
                output_chart[f"{label} {name}"] = run.outputs[:, index]
            for index, name in enumerate(INPUT_NAMES):
                input_chart[f"{label} {name}"] = run.inputs[:, index]
        st.caption("Controlled-variable responses")
        st.line_chart(output_chart.set_index("Minute"), height=260)
        st.caption("Manipulated-variable movement")
        st.line_chart(input_chart.set_index("Minute"), height=240)
    render_message(
        "NEXT STEP",
        "Return to APC Station to operate the active model and tuning, then use "
        "the existing APC Showcase after a normal RESET.",
    )


initialize()
tank_manager: FeedTankManager = st.session_state.feed_tank_manager
weather_manager: InletWeatherManager = st.session_state.weather_manager
showcase: ShowcaseState = st.session_state.showcase

with st.sidebar:
    render_sidebar_title("OPERATOR STATION")
    workspace = st.radio(
        "Workspace",
        ("APC Station", "Commissioning Lab"),
        index=widget_default("workspace", 0),
        horizontal=True,
        key="workspace",
        disabled=showcase.engaged,
    )
    commissioning_workspace = workspace == "Commissioning Lab"
    if commissioning_workspace:
        st.session_state.mode = "Manual"
        st.session_state.control_mode = "Manual"
        st.session_state.weather_mode = "Constant"
        st.session_state.automatic_tank_changes = False
    render_sidebar_title("APC SHOWCASE")
    if not showcase.engaged:
        st.button(
            "RUN APC SHOWCASE",
            width="stretch",
            type="primary",
            on_click=start_showcase,
            disabled=(
                commissioning_workspace
                or
                st.session_state.minute != 0
                or bool(st.session_state.history["minute"])
                or st.session_state.get("showcase_handoff_notice", False)
            ),
        )
        if commissioning_workspace:
            st.caption("Showcase is available from APC Station after RESET.")
    else:
        st.button("STOP SHOWCASE", width="stretch", on_click=stop_showcase)
        st.caption("Guided sequence active | HOLD remains available")
    if st.session_state.get("showcase_handoff_notice", False):
        st.caption("SHOWCASE COMPLETE - APC ACTIVE - OPERATOR CONTROL")

    if showcase.engaged:
        mode = "APC" if showcase.apc_enabled else "Manual"
        st.radio(
            "Control mode",
            ("Manual", "APC"),
            index=1 if mode == "APC" else 0,
            key=f"showcase_mode_{showcase.phase.value}",
            horizontal=True,
            disabled=True,
        )
    elif commissioning_workspace:
        mode = "Manual"
        st.radio(
            "Control mode",
            ("Manual", "APC"),
            index=0,
            key="commissioning_control_mode_display",
            horizontal=True,
            disabled=True,
        )
    else:
        mode = st.radio(
            "Control mode",
            ("Manual", "APC"),
            index=widget_default("mode", 0),
            key="mode",
            horizontal=True,
        )
        st.session_state.control_mode = mode
    left, right = st.columns(2)
    left.button(
        "RUN" if not st.session_state.running else "HOLD",
        width="stretch",
        on_click=toggle_running,
        disabled=(
            showcase.complete
            or (
                commissioning_workspace
                and (
                    not st.session_state.commissioning_prepared
                    or st.session_state.commissioning_plan_index
                    >= len(st.session_state.commissioning_plan)
                )
            )
        ),
    )
    right.button(
        "RESET", width="stretch", on_click=reset, disabled=showcase.engaged
    )

    render_sidebar_title("SENSOR SETTINGS")
    noise_checkbox_args: dict[str, object] = {
        "key": "noise_enabled",
        "disabled": showcase.engaged,
    }
    noise_default = widget_default("noise_enabled", True)
    if noise_default is not None:
        noise_checkbox_args["value"] = bool(noise_default)
    noise_enabled = st.checkbox(
        "Measurement noise enabled",
        **noise_checkbox_args,
    )
    noise_profile = st.selectbox(
        "Noise profile",
        tuple(NOISE_MULTIPLIERS),
        index=widget_default("noise_profile", 1),
        key="noise_profile",
        disabled=showcase.engaged or not noise_enabled,
    )

    render_sidebar_title("SIMULATION RATE")
    simulation_minutes_per_tick = st.selectbox(
        "Simulated minutes per scan",
        (1, 2, 5),
        index=widget_default("simulation_minutes_per_tick", 0),
        format_func=lambda minutes: f"{minutes} minute{'s' if minutes != 1 else ''}",
        key="simulation_minutes_per_tick",
        disabled=showcase.engaged or st.session_state.commissioning_prepared,
    )
    st.session_state.dryer.configure_time_step(simulation_minutes_per_tick)
    st.session_state.controller.configure_time_step(simulation_minutes_per_tick)

    render_sidebar_title("INLET-AIR WEATHER")
    weather_mode = st.selectbox(
        "Inlet-air humidity mode",
        WEATHER_MODES,
        index=widget_default("weather_mode", 0),
        key="weather_mode",
        disabled=showcase.engaged or commissioning_workspace,
    )
    weather_manager.configure_mode(weather_mode, st.session_state.minute)
    st.session_state.dryer.set_inlet_humidity(weather_manager.inlet_humidity)
    st.button(
        "TRIGGER HUMID WEATHER",
        width="stretch",
        type="primary",
        on_click=trigger_humid_weather,
        disabled=showcase.engaged or commissioning_workspace,
    )
    st.caption(
        f"Humidity increase: +{HUMID_WEATHER_INCREASE:.4f} {INLET_HUMIDITY_UNIT}"
    )

    render_sidebar_title("FEED SUPPLY")
    automatic_tank_checkbox_args: dict[str, object] = {
        "key": "automatic_tank_changes",
        "disabled": showcase.engaged or commissioning_workspace,
    }
    automatic_tank_default = widget_default("automatic_tank_changes", False)
    if automatic_tank_default is not None:
        automatic_tank_checkbox_args["value"] = bool(automatic_tank_default)
    automatic_tank_changes = st.checkbox(
        "Automatic tank changes",
        **automatic_tank_checkbox_args,
    )
    automatic_tank_interval = st.slider(
        "Automatic change interval (simulation min)",
        30,
        120,
        widget_default("automatic_tank_interval", 60),
        5,
        key="automatic_tank_interval",
        disabled=(
            showcase.engaged or commissioning_workspace or not automatic_tank_changes
        ),
    )
    tank_manager.configure_automatic(
        automatic_tank_changes,
        automatic_tank_interval,
        st.session_state.minute,
    )
    render_feed_tank_command(disabled=showcase.engaged or commissioning_workspace)

    render_sidebar_title("OPTIMIZATION")
    objective_parameter = st.selectbox(
        "Parameter",
        INPUT_NAMES + OUTPUT_NAMES,
        index=widget_default("objective_parameter", 6),
        key="objective_parameter",
        disabled=showcase.engaged or commissioning_workspace,
    )
    objective_mode = st.radio(
        "Goal",
        ("Target", "Maximize", "Minimize"),
        index=widget_default("objective_mode", 0),
        key="objective_mode",
        horizontal=True,
        disabled=showcase.engaged or commissioning_workspace,
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
            (50.0, 175.0, 1.0),
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
            widget_default(
                f"objective_target_{objective_group}_{objective_index}",
                float(objective_default),
            ),
            float(objective_step),
            key=f"objective_target_{objective_group}_{objective_index}",
            disabled=showcase.engaged or commissioning_workspace,
        )

    with st.expander("Manual MV commands", expanded=mode == "Manual"):
        manual_inputs = np.array(
            [
                st.slider(
                    INPUT_NAMES[i],
                    float(INPUT_MIN[i]),
                    float(INPUT_MAX[i]),
                    widget_default(f"input_{i}", float(NOMINAL_INPUTS[i])),
                    key=f"input_{i}",
                    disabled=showcase.engaged or st.session_state.commissioning_collecting,
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
            input_enabled_checkbox_args: dict[str, object] = {
                "key": f"enabled_{i}",
                "disabled": showcase.engaged or commissioning_workspace,
            }
            input_enabled_default = widget_default(f"enabled_{i}", True)
            if input_enabled_default is not None:
                input_enabled_checkbox_args["value"] = bool(input_enabled_default)
            input_enabled[i] = st.checkbox(
                f"Allow APC to change {name}",
                **input_enabled_checkbox_args,
            )
            selected = st.slider(
                f"{name} operating range",
                float(INPUT_MIN[i]),
                float(INPUT_MAX[i]),
                widget_default(
                    f"input_constraint_{i}",
                    (float(INPUT_MIN[i]), float(INPUT_MAX[i])),
                ),
                key=f"input_constraint_{i}",
                disabled=showcase.engaged or commissioning_workspace,
            )
            input_min[i], input_max[i] = selected
            max_move[i] = st.slider(
                f"{name} maximum move per minute",
                0.0,
                float(MAX_MOVE[i] * 3),
                widget_default(f"max_move_{i}", float(MAX_MOVE[i])),
                key=f"max_move_{i}",
                disabled=showcase.engaged or commissioning_workspace,
            )

    with st.expander("CV constraints"):
        output_min = np.empty(4)
        output_max = np.empty(4)
        ranges = [
            (60.0, 120.0, 1.0),
            (50.0, 175.0, 1.0),
            (2.0, 8.0, 0.1),
            (0.070, 0.170, 0.001),
        ]
        for i, name in enumerate(OUTPUT_NAMES[:3]):
            selected = st.slider(
                name,
                ranges[i][0],
                ranges[i][1],
                widget_default(
                    f"constraint_{i}",
                    (float(DEFAULT_OUTPUT_MIN[i]), float(DEFAULT_OUTPUT_MAX[i])),
                ),
                ranges[i][2],
                key=f"constraint_{i}",
                disabled=showcase.engaged or commissioning_workspace,
            )
            output_min[i], output_max[i] = selected
        derived_humidity_max = maximum_safe_humidity_ratio(output_max[0])
        output_min[3] = st.slider(
            "Exhaust air humidity lower process/efficiency limit",
            ranges[3][0],
            derived_humidity_max - ranges[3][2],
            widget_default("constraint_3_lower", float(DEFAULT_OUTPUT_MIN[3])),
            ranges[3][2],
            key="constraint_3_lower",
            disabled=showcase.engaged or commissioning_workspace,
        )
        output_max[3] = derived_humidity_max
        st.caption(
            "Exhaust-air humidity upper limit is derived from the configured "
            "safe stickiness boundary and maximum exhaust temperature: "
            f"{derived_humidity_max:.4f} kg water/kg dry air."
        )


render_title_bar(
    "SPRAY DRYER COMMISSIONING LAB"
    if commissioning_workspace
    else "SPRAY DRYER APC TRAINER",
    (
        "DATA-TO-MODEL WORKFLOW"
        if commissioning_workspace
        else "SIM-01  |  Multivariable control learning station"
    )
    + "  |  "
    f"{simulation_minutes_per_tick} minute"
    f"{'s' if simulation_minutes_per_tick != 1 else ''} per scan",
)


@st.fragment(run_every=1.0 if st.session_state.running else None)
def live_panel() -> None:
    tank_manager: FeedTankManager = st.session_state.feed_tank_manager
    weather_manager: InletWeatherManager = st.session_state.weather_manager
    showcase: ShowcaseState = st.session_state.showcase
    if st.session_state.running:
        event_minute = st.session_state.minute + simulation_minutes_per_tick
        guided_sample = None
        if (
            commissioning_workspace
            and st.session_state.commissioning_collecting
            and st.session_state.commissioning_plan_index
            < len(st.session_state.commissioning_plan)
        ):
            guided_sample = st.session_state.commissioning_plan[
                st.session_state.commissioning_plan_index
            ]
        if showcase.engaged:
            actions = showcase.advance(event_minute, running=True)
            apply_showcase_actions(actions, event_minute)
        effective_mode = st.session_state.control_mode
        if guided_sample is not None:
            st.session_state.inputs = rate_limited_inputs(
                st.session_state.inputs,
                guided_sample.target_inputs,
                simulation_minutes_per_tick,
                input_min,
                input_max,
                max_move,
            )
        elif effective_mode == "Manual":
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
        if not showcase.engaged and not commissioning_workspace:
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
        if guided_sample is not None:
            st.session_state.commissioning_samples.append(
                sample_record(
                    event_minute,
                    simulation_minutes_per_tick,
                    guided_sample,
                    st.session_state.inputs,
                    st.session_state.measurements,
                )
            )
            st.session_state.commissioning_plan_index += 1
            if st.session_state.commissioning_plan_index >= len(
                st.session_state.commissioning_plan
            ):
                st.session_state.commissioning_collecting = False
                st.session_state.running = False
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
        if st.session_state.pop("showcase_handoff_pending", False):
            st.rerun(scope="app")

    if commissioning_workspace:
        render_commissioning_lab(
            input_min,
            input_max,
            max_move,
            output_min,
            output_max,
            simulation_minutes_per_tick,
            noise_enabled,
            noise_profile,
        )
        return

    controller = st.session_state.controller
    process_state = "normal" if st.session_state.running else "warning"
    process_text = "RUN" if st.session_state.running else "HOLD"
    mpc_active = st.session_state.control_mode == "APC"
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
            ("MPC", "ACTIVE" if mpc_active else "MANUAL", "normal" if mpc_active else "neutral"),
            ("SENSORS", sensor_text, "normal" if noise_enabled else "neutral"),
            ("SOLVER", solver_text, solver_state),
            ("SCAN", f"T+{st.session_state.minute:05d}", "neutral"),
        ]
    )

    render_section_title("CONTROLLED VARIABLES")
    output_formats = (".2f", ".3f", ".3f", ".4f")
    output_states = [
        constraint_state(value, output_min[i], output_max[i])
        for i, value in enumerate(st.session_state.measurements)
    ]
    output_rows = [
        [
            name,
            format(float(value), output_formats[i]),
            (
                format(float(objective_target), output_formats[i])
                if objective_group == "output"
                and objective_mode == "target"
                and objective_index == i
                else "--"
            ),
            format(float(output_min[i]), output_formats[i]),
            format(float(output_max[i]), output_formats[i]),
            unit,
        ]
        for i, (name, unit, value) in enumerate(
            zip(OUTPUT_NAMES, OUTPUT_UNITS, st.session_state.measurements)
        )
    ]
    render_parameter_table(
        ["Parameter", "Current", "Target", "Lower", "Upper", "Unit"],
        output_rows,
        output_states,
    )

    render_section_title("MANIPULATED VARIABLES")
    input_rows = [
        [
            name,
            f"{float(value):.2f}",
            f"{float(input_min[i]):.2f}",
            f"{float(input_max[i]):.2f}",
            unit,
        ]
        for i, (name, unit, value) in enumerate(
            zip(INPUT_NAMES, INPUT_UNITS, st.session_state.inputs)
        )
    ]
    render_parameter_table(
        ["Parameter", "Current command", "Lower", "Upper", "Unit"],
        input_rows,
        ["normal"] * len(input_rows),
    )

    recent_tank_event = (
        last_tank_event is not None
        and st.session_state.minute - last_tank_event.minute <= 5
    )
    last_change_values = (
        f"{last_tank_event.old_dry_matter:.1f}>{last_tank_event.new_dry_matter:.1f}"
        if last_tank_event is not None
        else "--"
    )
    last_change_minute = (
        f"T+{last_tank_event.minute:05d}" if last_tank_event is not None else "--"
    )
    weather_active = weather_manager.state != "NORMAL"
    render_section_title("DISTURBANCES")
    disturbance_rows = [
        [
            "Active feed tank",
            tank_manager.current_tank,
            f"{last_change_values} at {last_change_minute}",
            "--",
        ],
        [
            "Incoming feed dry matter",
            f"{st.session_state.dryer.feed_dry_matter:.2f}",
            "recent change" if recent_tank_event else "stable",
            FEED_DRY_MATTER_UNIT,
        ],
        [
            "Inlet-air humidity",
            f"{weather_manager.inlet_humidity:.4f}",
            weather_manager.state,
            INLET_HUMIDITY_UNIT,
        ],
        [
            "Inlet-humidity mode",
            weather_manager.mode,
            (
                f"last event T+{last_weather_event.minute:05d}"
                if last_weather_event is not None
                else "no weather event"
            ),
            "--",
        ],
    ]
    render_parameter_table(
        ["Parameter", "Current value", "State / last change", "Unit"],
        disturbance_rows,
        [
            "warning" if recent_tank_event else "normal",
            "warning" if recent_tank_event else "neutral",
            "warning" if weather_active else "normal",
            "warning" if weather_active else "neutral",
        ],
    )

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
    if showcase.engaged:
        phase_info = showcase.info
        next_action = showcase.next_action()
        next_action_text = (
            f"Next: {next_action[0]} in {next_action[1]} min"
            if next_action is not None
            else "Sequence complete | operator handoff"
        )
        render_showcase_banner(
            phase=f"SHOWCASE {phase_info.number}/5 - {phase_info.title}",
            status="APC ON" if showcase.apc_enabled else "APC OFF",
            minute=showcase.scenario_minute,
            description=phase_info.description,
            next_action=next_action_text,
            progress=showcase.progress,
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
            "input_units": list(INPUT_UNITS)
            + ["% dry matter", "kg water/kg dry air"],
            "output_names": list(OUTPUT_NAMES),
            "output_units": list(OUTPUT_UNITS),
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
        "inputs | controlled outputs",
    )
    render_process_trends(payload)
    st.session_state.chart_last_sample_id = next_sample_id
    st.session_state.chart_last_event_id = next_event_id
    st.session_state.chart_needs_snapshot = False

    if showcase.complete:
        history = st.session_state.history
        metrics = calculate_showcase_metrics(
            history["minute"],
            np.column_stack([history[name] for name in INPUT_NAMES]),
            np.column_stack([history[name] for name in OUTPUT_NAMES]),
            output_min,
            output_max,
        )
        period_rows = metrics["periods"]
        total_violations = sum(
            int(row["Output constraint violations"]) for row in period_rows
        )
        maximum_moisture_deviation = max(
            float(row["Maximum moisture deviation"]) for row in period_rows
        )
        recovery_time = metrics["recovery_time"]
        render_section_title(
            "SHOWCASE PERFORMANCE SUMMARY",
            "guided sequence summary | not a controlled A/B benchmark",
        )
        render_value_grid(
            [
                ScadaValue(
                    "KPI-101",
                    "Recovery after APC enable",
                    f"{recovery_time:.0f}" if recovery_time is not None else "--",
                    "sim min",
                ),
                ScadaValue(
                    "KPI-102",
                    "Maximum moisture deviation",
                    f"{maximum_moisture_deviation:.3f}",
                    "% points",
                ),
                ScadaValue(
                    "KPI-103",
                    "Output constraint violations",
                    str(total_violations),
                    "signal-samples",
                ),
                ScadaValue(
                    "KPI-104",
                    "Normalized MV movement",
                    f"{float(metrics['total_mv_movement']):.3f}",
                    "range fractions",
                ),
            ],
            columns=4,
        )
        st.dataframe(
            pd.DataFrame(period_rows), width="stretch", hide_index=True
        )
        render_message(
            "GUIDED RUN INTERPRETATION",
            "During manual operation, the disturbances moved the process away "
            "from target while input commands remained fixed. After APC "
            "activation, the controller adjusted the available inputs and "
            "moved the controlled outputs toward their targets while applying "
            "the configured constraints.",
        )
        replay, leave = st.columns(2)
        replay_requested = replay.button(
            "RUN SHOWCASE AGAIN",
            key="showcase_summary_replay",
            width="stretch",
        )
        exit_requested = leave.button(
            "EXIT SHOWCASE",
            key="showcase_summary_exit",
            width="stretch",
        )
        if replay_requested:
            start_showcase()
            st.rerun(scope="app")
        if exit_requested:
            stop_showcase()
            st.rerun(scope="app")

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
        background=psychrometric_background(float(output_max[0])),
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
                    label="Exhaust air humidity ratio",
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
            f"Configured boundary and safe offset | {STANDARD_PRESSURE_KPA:.3f} kPa | "
            f"range w={MAP_HUMIDITY_RANGE[0]:.2f}-{MAP_HUMIDITY_RANGE[1]:.2f}, "
            f"T={MAP_TEMPERATURE_RANGE[0]:.0f}-{MAP_TEMPERATURE_RANGE[1]:.0f} C",
        )
        derived_humidity_max = float(output_max[3])
        render_message(
            "DERIVED HUMIDITY LIMIT",
            f"w <= {derived_humidity_max:.4f} kg water/kg dry air at "
            f"maximum exhaust temperature {output_max[0]:.1f} C. The lower "
            f"limit {output_min[3]:.4f} is a process/efficiency limit.",
        )
        if (
            objective_group == "output"
            and objective_index == 3
            and objective_mode == "target"
            and objective_target > derived_humidity_max
        ):
            render_message(
                "CONFIGURATION CONFLICT",
                f"Exhaust air humidity target {objective_target:.4f} exceeds "
                "the derived "
                f"safe upper limit {derived_humidity_max:.4f} kg water/kg dry air.",
            )
        render_operating_map(map_payload)
    st.session_state.operating_map_last_sample_id = next_map_sample_id
    st.session_state.operating_map_needs_snapshot = False

    with st.expander("PROCESS EVENT LOG"):
        event_rows = [
            {
                "Simulation minute": event.minute,
                "Event": event.event_type,
                "From": f"{event.old_tank} ({event.old_dry_matter:.1f}% DM)",
                "To": f"{event.new_tank} ({event.new_dry_matter:.1f}% DM)",
            }
            for event in reversed(st.session_state.tank_events)
        ]
        event_rows.extend(
            {
                "Simulation minute": event.minute,
                "Event": event.event_type,
                "From": "NORMAL",
                "To": f"+{event.humidity_increase:.4f} {INLET_HUMIDITY_UNIT}",
            }
            for event in st.session_state.weather_events
        )
        event_rows.extend(
            {
                "Simulation minute": event.minute,
                "Event": event.event_type,
                "From": "APC SHOWCASE",
                "To": event.detail,
            }
            for event in st.session_state.showcase_events
        )
        event_rows.sort(key=lambda row: int(row["Simulation minute"]), reverse=True)
        st.dataframe(
            pd.DataFrame(
                event_rows,
                columns=("Simulation minute", "Event", "From", "To"),
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
            f"{solver_text} | model: {st.session_state.active_model_revision} | "
            f"limiting: {controller.last_limiting_constraint} | next: {next_moves}",
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
if not commissioning_workspace:
    render_model_explanation()
