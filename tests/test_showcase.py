import numpy as np
from streamlit.testing.v1 import AppTest

from apc_lab.live_dryer import NOMINAL_INPUTS
from apc_lab.showcase import (
    ACTION_APC_CHALLENGE,
    ACTION_APC_ENABLE,
    ACTION_COMPLETE,
    ACTION_HUMID_WEATHER,
    ACTION_TANK_CHANGE,
    ShowcasePhase,
    ShowcaseState,
)


def _button_by_label(app: AppTest, label: str):
    return next(button for button in app.button if button.label == label)


def test_showcase_phases_follow_simulation_time_in_order():
    showcase = ShowcaseState()
    showcase.start()

    observed = [showcase.phase]
    for minute in (15, 30, 40, 55, 75, 100):
        showcase.advance(minute)
        observed.append(showcase.phase)

    assert observed == [
        ShowcasePhase.BASELINE,
        ShowcasePhase.HUMID_WEATHER,
        ShowcasePhase.TANK_CHANGE,
        ShowcasePhase.MANUAL_DRIFT,
        ShowcasePhase.APC_TAKEOVER,
        ShowcasePhase.APC_CHALLENGE,
        ShowcasePhase.COMPLETE,
    ]


def test_showcase_actions_execute_once_when_updates_repeat():
    showcase = ShowcaseState()
    showcase.start()

    assert showcase.advance(100) == [
        ACTION_HUMID_WEATHER,
        ACTION_TANK_CHANGE,
        ACTION_APC_ENABLE,
        ACTION_APC_CHALLENGE,
        ACTION_COMPLETE,
    ]
    assert showcase.advance(100) == []


def test_showcase_apc_is_off_until_takeover():
    showcase = ShowcaseState()
    showcase.start()

    showcase.advance(54)
    assert showcase.apc_enabled is False
    showcase.advance(55)
    assert showcase.apc_enabled is True


def test_showcase_hold_pauses_time_phase_and_actions():
    showcase = ShowcaseState()
    showcase.start()
    showcase.advance(10)

    assert showcase.advance(30, running=False) == []
    assert showcase.scenario_minute == 10
    assert showcase.phase == ShowcasePhase.BASELINE
    assert showcase.executed_actions == set()


def test_showcase_start_and_stop_return_app_to_clean_normal_state():
    app = AppTest.from_file("live_app.py").run(timeout=30)

    _button_by_label(app, "RUN APC SHOWCASE").click().run(timeout=30)
    assert app.session_state.showcase.engaged is True
    assert app.session_state.running is True
    assert app.session_state.control_mode == "Manual"
    np.testing.assert_allclose(app.session_state.inputs, NOMINAL_INPUTS)
    assert len(app.session_state.showcase_events) == 1
    assert _button_by_label(app, "RESET").disabled is True

    _button_by_label(app, "STOP SHOWCASE").click().run(timeout=30)
    assert app.session_state.showcase.phase == ShowcasePhase.IDLE
    assert app.session_state.running is False
    assert app.session_state.minute == 0
    assert app.session_state.history["minute"] == []
    assert app.session_state.showcase_events == []
    assert app.session_state.tank_events == []
    assert app.session_state.weather_events == []
    assert _button_by_label(app, "RESET").disabled is False
