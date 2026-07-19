import numpy as np
from streamlit.testing.v1 import AppTest

from apc_lab.commissioning import (
    PERIOD_ESTIMATION,
    PERIOD_VALIDATION,
    TuningPreset,
    apply_model,
    compare_tunings,
    diagnose_excitation,
    model_from_controller,
    restore_builtin_model,
    simulate_guided_dataset,
    split_estimation_validation,
    validate_candidate,
)
from apc_lab.live_dryer import (
    GAIN_MATRIX,
    INPUT_MAX,
    INPUT_MIN,
    INPUT_NAMES,
    MAX_MOVE,
    NOMINAL_INPUTS,
    NOMINAL_OUTPUTS,
    OUTPUT_NAMES,
    OUTPUT_TAU,
    ConstrainedDryerMPC,
    LiveSprayDryer,
)
from apc_lab.model_fitting import arrays_from_dataframe, fit_dynamic_model


def _button(app: AppTest, label: str):
    return next(button for button in app.button if button.label == label)


def test_guided_commissioning_dataset_is_deterministic_and_separated():
    first = simulate_guided_dataset(sample_minutes=5, seed=19)
    second = simulate_guided_dataset(sample_minutes=5, seed=19)

    assert first.equals(second)
    assert tuple(first.columns[-7:]) == INPUT_NAMES + OUTPUT_NAMES
    assert set(first["Period"]) == {PERIOD_ESTIMATION, PERIOD_VALIDATION}
    estimation, validation = split_estimation_validation(first)
    assert estimation["Simulation minute"].max() < validation["Simulation minute"].min()
    assert np.all(estimation["Sample duration (min)"] == 5)


def test_chronological_split_and_excitation_diagnostics_are_explicit():
    dataset = simulate_guided_dataset(sample_minutes=5, noise_multiplier=0.0)
    legacy = dataset.drop(columns=["Period", "Experiment phase"])
    estimation, validation = split_estimation_validation(legacy, 0.7)
    inputs, outputs = arrays_from_dataframe(estimation)
    diagnostics = diagnose_excitation(inputs, outputs)

    assert len(estimation) + len(validation) == len(legacy)
    assert diagnostics.input_rank == 3
    assert diagnostics.blocking_errors == ()

    unexcited = np.tile(NOMINAL_INPUTS, (30, 1))
    blocked = diagnose_excitation(unexcited, np.tile(NOMINAL_OUTPUTS, (30, 1)))
    assert blocked.blocking_errors


def test_candidate_validation_reports_free_run_output_metrics():
    dataset = simulate_guided_dataset(sample_minutes=5, noise_multiplier=0.0)
    estimation, validation = split_estimation_validation(dataset)
    estimation_inputs, estimation_outputs = arrays_from_dataframe(estimation)
    validation_inputs, validation_outputs = arrays_from_dataframe(validation)
    candidate = fit_dynamic_model(
        estimation_inputs,
        estimation_outputs,
        max_delay=3,
        dt=5,
    )

    result = validate_candidate(
        candidate,
        estimation_inputs,
        estimation_outputs,
        validation_inputs,
        validation_outputs,
        sample_minutes=5,
    )

    assert result.estimation.predictions.shape == estimation_outputs.shape
    assert result.validation.predictions.shape == validation_outputs.shape
    assert np.all(np.isfinite(result.validation.rmse))
    assert np.all(np.isfinite(result.validation.mae))
    assert candidate.sample_minutes == 5
    assert candidate.delay_minutes == candidate.delay_steps * 5


def test_candidate_application_changes_only_predictor_and_restore_is_complete():
    plant = LiveSprayDryer()
    controller = ConstrainedDryerMPC()
    plant_state = plant.true_outputs.copy()
    candidate = model_from_controller(controller)
    altered = type(candidate)(
        candidate.gain_matrix * 0.8,
        candidate.output_tau * 1.2,
        2,
        candidate.rmse,
        100,
        candidate.nominal_inputs + 1.0,
        candidate.nominal_outputs + 0.1,
        2.0,
        4.0,
    )

    apply_model(controller, altered)

    np.testing.assert_allclose(controller.gain_matrix, altered.gain_matrix)
    assert controller.delay_minutes == 4.0
    assert controller.delay_steps == 4
    np.testing.assert_array_equal(plant.true_outputs, plant_state)

    restore_builtin_model(controller)
    np.testing.assert_allclose(controller.gain_matrix, GAIN_MATRIX)
    np.testing.assert_allclose(controller.output_tau, OUTPUT_TAU)
    np.testing.assert_allclose(controller.nominal_inputs, NOMINAL_INPUTS)
    np.testing.assert_allclose(controller.nominal_outputs, NOMINAL_OUTPUTS)
    assert controller.delay_minutes == 3.0


def test_tuning_comparison_is_repeatable_and_preserves_mv_limits():
    controller = ConstrainedDryerMPC(prediction_horizon=5, control_horizon=5)
    model = model_from_controller(controller)
    slow = TuningPreset(2, NOMINAL_OUTPUTS[2], 40.0, 1.0, 5)
    aggressive = TuningPreset(2, NOMINAL_OUTPUTS[2], 300.0, 0.03, 5)
    output_min = np.array([75.0, 75.0, 3.0, 0.09])
    output_max = np.array([100.0, 137.5, 5.2, 0.1286])

    first = compare_tunings(
        slow,
        aggressive,
        model,
        output_min,
        output_max,
        sample_minutes=5,
        noise_multiplier=0.0,
        seed=33,
    )
    repeated = compare_tunings(
        slow,
        aggressive,
        model,
        output_min,
        output_max,
        sample_minutes=5,
        noise_multiplier=0.0,
        seed=33,
    )

    np.testing.assert_allclose(first.tuning_a.outputs, repeated.tuning_a.outputs)
    np.testing.assert_allclose(first.tuning_b.inputs, repeated.tuning_b.inputs)
    assert not np.allclose(first.tuning_a.inputs, first.tuning_b.inputs)
    for run in (first.tuning_a, first.tuning_b):
        assert np.all(run.inputs >= INPUT_MIN - 1e-9)
        assert np.all(run.inputs <= INPUT_MAX + 1e-9)
        moves = np.diff(np.vstack([NOMINAL_INPUTS, run.inputs]), axis=0)
        assert np.all(np.abs(moves) <= MAX_MOVE * 5 + 1e-6)
        assert run.metrics.integrated_absolute_error >= 0


def test_commissioning_workspace_is_separate_and_reset_restores_builtin_state():
    app = AppTest.from_file("live_app.py").run(timeout=30)
    workspace = next(radio for radio in app.radio if radio.label == "Workspace")

    workspace.set_value("Commissioning Lab").run(timeout=30)

    assert not app.exception
    assert app.session_state.control_mode == "Manual"
    assert _button(app, "RUN APC SHOWCASE").disabled is True
    assert _button(app, "PREPARE GUIDED EXPERIMENT").disabled is False

    app.session_state.active_model_revision = "FIT-999"
    _button(app, "RESET").click().run(timeout=30)

    assert app.session_state.workspace == "APC Station"
    assert app.session_state.active_model_revision == "BUILT-IN"
    assert app.session_state.commissioning_candidate is None
