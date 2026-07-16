import numpy as np

from apc_lab.live_dryer import (
    INPUT_MAX,
    INPUT_MIN,
    MAX_MOVE,
    NOMINAL_INPUTS,
    NOMINAL_OUTPUTS,
    ConstrainedDryerMPC,
    LiveSprayDryer,
    steady_outputs,
)


def test_process_directions_are_plausible():
    high_feed = steady_outputs(NOMINAL_INPUTS + np.array([10.0, 0.0, 0.0]))
    hot_air = steady_outputs(NOMINAL_INPUTS + np.array([0.0, 0.0, 10.0]))
    assert high_feed[1] > NOMINAL_OUTPUTS[1]
    assert high_feed[2] > NOMINAL_OUTPUTS[2]
    assert high_feed[3] > NOMINAL_OUTPUTS[3]
    assert hot_air[0] > NOMINAL_OUTPUTS[0]
    assert hot_air[2] < NOMINAL_OUTPUTS[2]
    assert hot_air[3] < NOMINAL_OUTPUTS[3]


def test_process_has_input_delay():
    dryer = LiveSprayDryer()
    changed = NOMINAL_INPUTS + np.array([10.0, 0.0, 0.0])
    first = dryer.step(changed, noisy=False)
    assert np.allclose(first, NOMINAL_OUTPUTS)
    for _ in range(dryer.delay_steps + 1):
        later = dryer.step(changed, noisy=False)
    assert later[1] > NOMINAL_OUTPUTS[1]


def test_multivariable_mpc_respects_move_and_input_limits():
    controller = ConstrainedDryerMPC()
    lower = np.array([70.0, 2.5, 2.5, 0.080])
    upper = np.array([115.0, 6.5, 6.5, 0.160])
    next_inputs = controller.move(
        NOMINAL_OUTPUTS.copy(),
        NOMINAL_INPUTS.copy(),
        lower,
        upper,
        objective_group="output",
        objective_index=3,
        objective_mode="target",
        objective_target=0.105,
    )
    assert np.all(next_inputs >= INPUT_MIN)
    assert np.all(next_inputs <= INPUT_MAX)
    assert np.all(np.abs(next_inputs - NOMINAL_INPUTS) <= MAX_MOVE + 1e-6)
    assert controller.last_output_plan is not None


def test_each_optimization_goal_produces_a_useful_move():
    lower = np.array([75.0, 3.0, 3.0, 0.090])
    upper = np.array([105.0, 5.5, 5.2, 0.145])

    maximize_air = ConstrainedDryerMPC()
    air_move = maximize_air.move(
        NOMINAL_OUTPUTS.copy(),
        NOMINAL_INPUTS.copy(),
        lower,
        upper,
        "input",
        1,
        "maximize",
        NOMINAL_INPUTS[1],
    )
    assert air_move[1] > NOMINAL_INPUTS[1]

    minimize_moisture = ConstrainedDryerMPC()
    moisture_move = minimize_moisture.move(
        NOMINAL_OUTPUTS.copy(),
        NOMINAL_INPUTS.copy(),
        lower,
        upper,
        "output",
        2,
        "minimize",
        NOMINAL_OUTPUTS[2],
    )
    assert moisture_move[0] < NOMINAL_INPUTS[0]
    assert moisture_move[1] > NOMINAL_INPUTS[1]
    assert moisture_move[2] > NOMINAL_INPUTS[2]


def test_frozen_feed_flow_is_not_changed():
    controller = ConstrainedDryerMPC()
    lower = np.array([75.0, 3.0, 3.0, 0.090])
    upper = np.array([105.0, 5.5, 5.2, 0.145])
    next_inputs = controller.move(
        NOMINAL_OUTPUTS.copy(),
        NOMINAL_INPUTS.copy(),
        lower,
        upper,
        "output",
        2,
        "minimize",
        NOMINAL_OUTPUTS[2],
        INPUT_MIN,
        INPUT_MAX,
        MAX_MOVE,
        np.array([False, True, True]),
    )
    assert next_inputs[0] == NOMINAL_INPUTS[0]
    assert next_inputs[1] > NOMINAL_INPUTS[1]
    assert next_inputs[2] > NOMINAL_INPUTS[2]
