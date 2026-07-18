from streamlit.testing.v1 import AppTest


def _element_by_label(elements, label):
    return next(element for element in elements if element.label == label)


def test_manual_tank_change_is_recorded_once_and_reset_restores_initial_state():
    app = AppTest.from_file("live_app.py").run(timeout=30)
    initial_chart_run_id = app.session_state.chart_run_id

    assert app.session_state.feed_tank_manager.current_tank == "Tank A"
    _element_by_label(app.selectbox, "Next feed tank").set_value("Tank B")
    _element_by_label(app.button, "CHANGE FEED TANK").click().run(timeout=30)

    assert len(app.session_state.tank_events) == 1
    assert app.session_state.feed_tank_manager.current_tank == "Tank B"
    app.run(timeout=30)
    assert len(app.session_state.tank_events) == 1

    _element_by_label(app.button, "RESET").click().run(timeout=30)
    assert app.session_state.feed_tank_manager.current_tank == "Tank A"
    assert app.session_state.tank_events == []
    assert app.session_state.dryer.feed_dry_matter == 50.0
    assert app.session_state.chart_run_id != initial_chart_run_id
    assert app.session_state.chart_last_sample_id == -1
    assert app.session_state.chart_last_event_id == 0


def test_selected_scan_rate_advances_simulation_time_by_selected_minutes():
    app = AppTest.from_file("live_app.py").run(timeout=30)

    _element_by_label(app.selectbox, "Simulated minutes per scan").set_value(5)
    _element_by_label(app.button, "RUN").click().run(timeout=30)

    assert app.session_state.dryer.dt == 5.0
    assert app.session_state.controller.dt == 5.0
    assert app.session_state.minute == 5


def test_hold_prevents_process_advancement():
    app = AppTest.from_file("live_app.py").run(timeout=30)

    assert app.session_state.running is False
    assert app.session_state.minute == 0
    app.run(timeout=30)
    assert app.session_state.minute == 0

    _element_by_label(app.button, "RUN").click().run(timeout=30)
    assert app.session_state.running is True
    running_minute = app.session_state.minute
    assert running_minute == 1

    _element_by_label(app.button, "HOLD").click().run(timeout=30)
    assert app.session_state.running is False
    assert app.session_state.minute == running_minute
    app.run(timeout=30)
    assert app.session_state.minute == running_minute


def test_automatic_tank_change_runs_once_at_scheduled_tick():
    app = AppTest.from_file("live_app.py").run(timeout=30)

    _element_by_label(app.checkbox, "Automatic tank changes").set_value(True)
    app.run(timeout=30)
    app.session_state.feed_tank_manager.next_auto_change_minute = 1

    _element_by_label(app.button, "RUN").click().run(timeout=30)

    assert app.session_state.minute == 1
    assert len(app.session_state.tank_events) == 1
    assert app.session_state.tank_events[0].event_type == "Automatic tank change"
    assert app.session_state.feed_tank_manager.current_tank != "Tank A"

    _element_by_label(app.button, "HOLD").click().run(timeout=30)
    app.run(timeout=30)
    assert app.session_state.minute == 1
    assert len(app.session_state.tank_events) == 1
