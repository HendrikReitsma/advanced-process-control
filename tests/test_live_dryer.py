import numpy as np

from apc_lab.live_dryer import (
    INPUT_MAX,
    INPUT_MIN,
    MAX_MOVE,
    MEASUREMENT_NOISE_STD,
    NOMINAL_FEED_DRY_MATTER,
    NOMINAL_INPUTS,
    NOMINAL_OUTPUTS,
    ConstrainedDryerMPC,
    FeedTankManager,
    LiveSprayDryer,
    steady_outputs,
)


def test_process_directions_are_plausible():
    assert NOMINAL_OUTPUTS[1] == 100.0
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


def test_measurement_noise_is_deterministic_and_does_not_change_plant_state():
    first_dryer = LiveSprayDryer(seed=27)
    second_dryer = LiveSprayDryer(seed=27)
    moved_inputs = NOMINAL_INPUTS + np.array([5.0, 1.0, 5.0])

    first_true = first_dryer.advance(moved_inputs)
    second_true = second_dryer.advance(moved_inputs)
    first_measurement = first_dryer.measure()
    second_measurement = second_dryer.measure()

    assert np.array_equal(first_true, second_true)
    assert np.array_equal(first_measurement, second_measurement)
    assert np.array_equal(first_dryer.true_outputs, first_true)
    assert not np.array_equal(first_measurement, first_true)


def test_disabled_noise_returns_true_state_without_consuming_random_sample():
    first_dryer = LiveSprayDryer(seed=33)
    second_dryer = LiveSprayDryer(seed=33)
    first_dryer.advance(NOMINAL_INPUTS)
    second_dryer.advance(NOMINAL_INPUTS)

    disabled_measurement = first_dryer.measure(enabled=False)
    measurement_after_disabled_sample = first_dryer.measure()
    first_enabled_measurement = second_dryer.measure()

    assert np.array_equal(disabled_measurement, first_dryer.true_outputs)
    assert np.array_equal(measurement_after_disabled_sample, first_enabled_measurement)


def test_noise_multiplier_scales_the_same_seeded_sensor_sample():
    normal_dryer = LiveSprayDryer(seed=45)
    high_dryer = LiveSprayDryer(seed=45)
    normal_true = normal_dryer.advance(NOMINAL_INPUTS)
    high_true = high_dryer.advance(NOMINAL_INPUTS)

    normal_offset = normal_dryer.measure(multiplier=1.0) - normal_true
    high_offset = high_dryer.measure(multiplier=2.0) - high_true

    assert np.allclose(high_offset, 2.0 * normal_offset)
    assert np.all(MEASUREMENT_NOISE_STD > 0)


def test_tank_change_updates_feed_dry_matter_before_outputs_respond():
    dryer = LiveSprayDryer()
    dryer.set_feed_dry_matter(52.0)

    dryer.advance(NOMINAL_INPUTS)

    assert NOMINAL_FEED_DRY_MATTER < dryer.feed_dry_matter < 52.0
    assert dryer.feed_dry_matter_target == 52.0


def test_higher_feed_dry_matter_changes_true_outputs_in_expected_directions():
    baseline = LiveSprayDryer()
    tank_b_feed = LiveSprayDryer()
    tank_b_feed.set_feed_dry_matter(52.0)

    for _ in range(80):
        baseline.advance(NOMINAL_INPUTS)
        tank_b_feed.advance(NOMINAL_INPUTS)

    assert tank_b_feed.true_outputs[0] > baseline.true_outputs[0]
    assert tank_b_feed.true_outputs[1] > baseline.true_outputs[1]
    assert tank_b_feed.true_outputs[2] < baseline.true_outputs[2]
    assert tank_b_feed.true_outputs[3] < baseline.true_outputs[3]
    assert baseline.true_outputs[2] - tank_b_feed.true_outputs[2] > 0.5
    assert baseline.true_outputs[3] - tank_b_feed.true_outputs[3] > 0.004


def test_larger_simulation_step_uses_stable_first_order_response():
    dryer = LiveSprayDryer()
    dryer.configure_time_step(5)
    dryer.set_feed_dry_matter(52.0)

    dryer.advance(NOMINAL_INPUTS)

    assert dryer.dt == 5.0
    assert dryer.delay_steps == 1
    assert NOMINAL_FEED_DRY_MATTER < dryer.feed_dry_matter < 52.0
    assert np.all(np.isfinite(dryer.true_outputs))


def test_controller_delay_remains_three_physical_minutes_across_scan_rates():
    controller = ConstrainedDryerMPC()

    controller.configure_time_step(1)
    assert controller.delay_steps == 3
    controller.configure_time_step(2)
    assert controller.delay_steps == 2
    controller.configure_time_step(5)
    assert controller.delay_steps == 1
    assert controller.delay_minutes == 3.0


def test_measurement_noise_remains_separate_from_tank_disturbance():
    dryer = LiveSprayDryer(seed=55)
    dryer.set_feed_dry_matter(48.5)
    true_outputs = dryer.advance(NOMINAL_INPUTS)
    measured_outputs = dryer.measure()

    assert dryer.feed_dry_matter < NOMINAL_FEED_DRY_MATTER
    assert np.array_equal(dryer.measure(enabled=False), true_outputs)
    assert not np.array_equal(measured_outputs, true_outputs)
    assert dryer.feed_dry_matter_target == 48.5


def test_tank_events_are_deterministic_and_not_repeated_on_rerun():
    first_manager = FeedTankManager(seed=71)
    second_manager = FeedTankManager(seed=71)
    first_manager.configure_automatic(True, 60, minute=0)
    second_manager.configure_automatic(True, 60, minute=0)

    first_event = first_manager.maybe_automatic_change(60)
    second_event = second_manager.maybe_automatic_change(60)

    assert first_event == second_event
    assert first_event is not None
    assert first_manager.maybe_automatic_change(60) is None
    assert first_manager.next_auto_change_minute == 120


def test_manual_and_automatic_tank_changes_select_new_tanks():
    manager = FeedTankManager(seed=17)

    manual_event = manager.change_to("Tank B", minute=4)
    manager.configure_automatic(True, 30, minute=4)
    automatic_event = manager.maybe_automatic_change(34)

    assert manual_event is not None
    assert manual_event.new_tank == "Tank B"
    assert automatic_event is not None
    assert automatic_event.event_type == "Automatic tank change"
    assert automatic_event.new_tank != "Tank B"


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
    lower = np.array([75.0, 75.0, 3.0, 0.090])
    upper = np.array([105.0, 137.5, 5.2, 0.145])

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
    lower = np.array([75.0, 75.0, 3.0, 0.090])
    upper = np.array([105.0, 137.5, 5.2, 0.145])
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
