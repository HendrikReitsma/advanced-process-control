import numpy as np
from streamlit.testing.v1 import AppTest

from apc_lab.live_dryer import (
    HUMID_WEATHER_INCREASE,
    NOMINAL_INLET_HUMIDITY,
    NOMINAL_INPUTS,
    InletWeatherManager,
    steady_outputs,
)


def _element_by_label(elements, label):
    return next(element for element in elements if element.label == label)


def test_constant_inlet_humidity_mode_remains_constant():
    manager = InletWeatherManager()

    values = []
    for minute in range(0, 361, 5):
        assert manager.advance(minute) is None
        values.append(manager.inlet_humidity)

    assert np.allclose(values, NOMINAL_INLET_HUMIDITY)
    assert manager.state == "NORMAL"


def test_daily_inlet_humidity_is_smooth_and_deterministic():
    first = InletWeatherManager()
    second = InletWeatherManager()
    first.configure_mode("Daily variation", 0)
    second.configure_mode("Daily variation", 0)

    first_values = []
    second_values = []
    for minute in range(0, 361, 5):
        first.advance(minute)
        second.advance(minute)
        first_values.append(first.inlet_humidity)
        second_values.append(second.inlet_humidity)

    assert np.array_equal(first_values, second_values)
    assert np.ptp(first_values) > 0.003
    assert np.min(first_values) < NOMINAL_INLET_HUMIDITY < np.max(first_values)
    assert np.max(np.abs(np.diff(first_values))) < 0.0003


def test_manual_humid_weather_rises_holds_recovers_and_moves_true_plant_target():
    manager = InletWeatherManager()
    event = manager.trigger(0)

    assert event is not None
    assert manager.trigger(0) is None

    states = {}
    values = {}
    for minute in (0, 15, 30, 75, 100, 120):
        manager.advance(minute)
        states[minute] = manager.state
        values[minute] = manager.inlet_humidity

    assert values[0] == NOMINAL_INLET_HUMIDITY
    assert NOMINAL_INLET_HUMIDITY < values[15] < values[30]
    assert values[30] == NOMINAL_INLET_HUMIDITY + HUMID_WEATHER_INCREASE
    assert values[75] == values[30]
    assert NOMINAL_INLET_HUMIDITY < values[100] < values[75]
    assert values[120] == NOMINAL_INLET_HUMIDITY
    assert states == {
        0: "APPROACHING",
        15: "APPROACHING",
        30: "HUMID/STORM",
        75: "HUMID/STORM",
        100: "RECOVERING",
        120: "NORMAL",
    }

    baseline = steady_outputs(NOMINAL_INPUTS)
    humid = steady_outputs(
        NOMINAL_INPUTS,
        inlet_humidity=NOMINAL_INLET_HUMIDITY + HUMID_WEATHER_INCREASE,
    )
    dry = steady_outputs(
        NOMINAL_INPUTS,
        inlet_humidity=NOMINAL_INLET_HUMIDITY - 0.002,
    )
    assert humid[0] < baseline[0]
    assert humid[1] == baseline[1]
    assert humid[2] > baseline[2]
    assert humid[3] > baseline[3]
    assert dry[0] > baseline[0]
    assert dry[2] < baseline[2]
    assert dry[3] < baseline[3]


def test_reset_restores_initial_weather_state_and_clears_events():
    app = AppTest.from_file("live_app.py").run(timeout=30)
    initial_run_id = app.session_state.chart_run_id

    _element_by_label(app.selectbox, "Inlet-air humidity mode").set_value(
        "Daily variation + weather events"
    )
    _element_by_label(app.button, "TRIGGER HUMID WEATHER").click().run(timeout=30)

    assert len(app.session_state.weather_events) == 1
    assert app.session_state.weather_manager.active_event_minute == 0

    _element_by_label(app.button, "RESET").click().run(timeout=30)

    assert app.session_state.weather_manager.mode == "Constant"
    assert app.session_state.weather_manager.state == "NORMAL"
    assert app.session_state.weather_manager.inlet_humidity == NOMINAL_INLET_HUMIDITY
    assert app.session_state.weather_manager.active_event_minute is None
    assert app.session_state.weather_events == []
    assert app.session_state.weather_event_ids == []
    assert app.session_state.chart_run_id != initial_run_id
    assert app.session_state.history["Inlet air humidity"] == []
