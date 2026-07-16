from apc_lab.scada_ui import constraint_state


def test_constraint_state_marks_normal_warning_and_alarm_ranges():
    assert constraint_state(50.0, 0.0, 100.0) == "normal"
    assert constraint_state(5.0, 0.0, 100.0) == "warning"
    assert constraint_state(95.0, 0.0, 100.0) == "warning"
    assert constraint_state(-0.1, 0.0, 100.0) == "alarm"
    assert constraint_state(100.1, 0.0, 100.0) == "alarm"
