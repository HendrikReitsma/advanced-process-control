"""Pure commissioning, validation, and tuning-comparison helpers."""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .live_dryer import (
    GAIN_MATRIX,
    INPUT_MAX,
    INPUT_MIN,
    INPUT_NAMES,
    INPUT_SCALE,
    MAX_MOVE,
    MEASUREMENT_NOISE_STD,
    NOMINAL_INPUTS,
    NOMINAL_OUTPUTS,
    OUTPUT_NAMES,
    OUTPUT_SCALE,
    OUTPUT_TAU,
    ConstrainedDryerMPC,
    LiveSprayDryer,
)
from .model_fitting import FittedDryerModel

PERIOD_ESTIMATION = "Estimation"
PERIOD_VALIDATION = "Validation"
COMMISSIONING_METADATA_COLUMNS = (
    "Simulation minute",
    "Sample duration (min)",
    "Period",
    "Experiment phase",
)
COMMISSIONING_COLUMNS = COMMISSIONING_METADATA_COLUMNS + INPUT_NAMES + OUTPUT_NAMES


@dataclass(frozen=True)
class ExperimentSample:
    """One requested operating point in the guided commissioning plan."""

    period: str
    phase: str
    target_inputs: np.ndarray


@dataclass(frozen=True)
class ExcitationDiagnostics:
    """Identification-readiness checks for one estimation dataset."""

    input_ranges: np.ndarray
    output_ranges: np.ndarray
    input_rank: int
    condition_number: float
    warnings: tuple[str, ...]
    blocking_errors: tuple[str, ...]


@dataclass(frozen=True)
class ResponseMetrics:
    """Free-run output metrics and predictions for one data period."""

    predictions: np.ndarray
    rmse: np.ndarray
    mae: np.ndarray
    fit_percent: np.ndarray
    samples: int


@dataclass(frozen=True)
class ModelValidation:
    """Separate estimation and validation response results."""

    estimation: ResponseMetrics
    validation: ResponseMetrics


@dataclass(frozen=True)
class TuningPreset:
    """The three learner-facing tuning concepts supported by the current MPC."""

    target_output_index: int
    target: float
    objective_weight: float
    move_weight: float
    prediction_horizon: int


@dataclass(frozen=True)
class ComparisonMetrics:
    recovery_minutes: float | None
    integrated_absolute_error: float
    normalized_cv_error: float
    maximum_constraint_violation: float
    constraint_violation_count: int
    normalized_mv_movement: float


@dataclass(frozen=True)
class ComparisonRun:
    minutes: np.ndarray
    inputs: np.ndarray
    outputs: np.ndarray
    metrics: ComparisonMetrics


@dataclass(frozen=True)
class TuningComparison:
    tuning_a: ComparisonRun
    tuning_b: ComparisonRun
    tank_change_minute: float


def _simulate_dynamic_model(
    model: FittedDryerModel,
    inputs: np.ndarray,
    initial_outputs: np.ndarray,
    sample_minutes: float,
) -> np.ndarray:
    """Free-run a fitted model without requiring a cross-module helper import."""

    inputs = np.asarray(inputs, dtype=float)
    initial_outputs = np.asarray(initial_outputs, dtype=float)
    if inputs.ndim != 2 or inputs.shape[1] != 3:
        raise ValueError(f"inputs must have 3 columns: {INPUT_NAMES}")
    if initial_outputs.shape != (4,):
        raise ValueError(f"initial_outputs must have 4 values: {OUTPUT_NAMES}")
    if sample_minutes <= 0:
        raise ValueError("sample time must be positive")

    delay_minutes = float(
        getattr(
            model,
            "delay_minutes",
            model.delay_steps * getattr(model, "sample_minutes", sample_minutes),
        )
    )
    delay_steps = max(0, int(np.ceil(delay_minutes / sample_minutes)))
    predictions = np.empty((len(inputs), 4), dtype=float)
    if len(inputs) == 0:
        return predictions
    predictions[0] = initial_outputs
    response_fraction = 1.0 - np.exp(-sample_minutes / model.output_tau)
    for sample_index in range(1, len(inputs)):
        delayed_index = sample_index - delay_steps
        delayed_inputs = (
            model.nominal_inputs if delayed_index < 0 else inputs[delayed_index]
        )
        steady_prediction = model.nominal_outputs + model.gain_matrix @ (
            delayed_inputs - model.nominal_inputs
        )
        predictions[sample_index] = predictions[sample_index - 1] + response_fraction * (
            steady_prediction - predictions[sample_index - 1]
        )
    return predictions


def build_guided_plan(sample_minutes: float) -> list[ExperimentSample]:
    """Build deterministic, one-MV-at-a-time estimation and validation plans."""

    if sample_minutes <= 0:
        raise ValueError("sample time must be positive")
    plan: list[ExperimentSample] = []
    amplitudes = {
        PERIOD_ESTIMATION: np.array([8.0, 1.6, 8.0]),
        PERIOD_VALIDATION: np.array([12.0, 2.4, 12.0]),
    }

    def add_stage(period: str, phase: str, target: np.ndarray, minutes: float) -> None:
        scans = max(1, int(np.ceil(minutes / sample_minutes)))
        plan.extend(
            ExperimentSample(period, phase, np.asarray(target, dtype=float).copy())
            for _ in range(scans)
        )

    for period in (PERIOD_ESTIMATION, PERIOD_VALIDATION):
        add_stage(period, "Stable baseline", NOMINAL_INPUTS, 15.0)
        for input_index, input_name in enumerate(INPUT_NAMES):
            positive = NOMINAL_INPUTS.copy()
            positive[input_index] += amplitudes[period][input_index]
            negative = NOMINAL_INPUTS.copy()
            negative[input_index] -= amplitudes[period][input_index]
            add_stage(period, f"{input_name} positive step", positive, 30.0)
            add_stage(period, f"{input_name} recovery", NOMINAL_INPUTS, 10.0)
            add_stage(period, f"{input_name} negative step", negative, 30.0)
            add_stage(period, f"{input_name} recovery", NOMINAL_INPUTS, 10.0)
    return plan


def rate_limited_inputs(
    current_inputs: np.ndarray,
    target_inputs: np.ndarray,
    sample_minutes: float,
    input_min: np.ndarray = INPUT_MIN,
    input_max: np.ndarray = INPUT_MAX,
    maximum_move: np.ndarray = MAX_MOVE,
) -> np.ndarray:
    """Move toward an experiment target using the live MV bounds and rates."""

    current_inputs = np.asarray(current_inputs, dtype=float)
    target_inputs = np.asarray(target_inputs, dtype=float)
    maximum_change = np.asarray(maximum_move, dtype=float) * float(sample_minutes)
    moved = current_inputs + np.clip(
        target_inputs - current_inputs, -maximum_change, maximum_change
    )
    return np.clip(moved, input_min, input_max)


def sample_record(
    minute: float,
    sample_minutes: float,
    experiment: ExperimentSample,
    inputs: np.ndarray,
    measured_outputs: np.ndarray,
) -> dict[str, object]:
    """Create one ordered, CSV-ready commissioning sample."""

    record: dict[str, object] = {
        "Simulation minute": float(minute),
        "Sample duration (min)": float(sample_minutes),
        "Period": experiment.period,
        "Experiment phase": experiment.phase,
    }
    record.update(zip(INPUT_NAMES, np.asarray(inputs, dtype=float).tolist()))
    record.update(zip(OUTPUT_NAMES, np.asarray(measured_outputs, dtype=float).tolist()))
    return record


def samples_to_dataframe(samples: Iterable[dict[str, object]]) -> pd.DataFrame:
    """Return commissioning records in stable metadata and signal order."""

    return pd.DataFrame(list(samples), columns=COMMISSIONING_COLUMNS)


def split_estimation_validation(
    dataframe: pd.DataFrame,
    estimation_fraction: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use explicit periods when present, otherwise make a chronological split."""

    if "Period" in dataframe.columns:
        estimation = dataframe[dataframe["Period"] == PERIOD_ESTIMATION].copy()
        validation = dataframe[dataframe["Period"] == PERIOD_VALIDATION].copy()
        if not estimation.empty and not validation.empty:
            return estimation.reset_index(drop=True), validation.reset_index(drop=True)
        if not estimation.empty or not validation.empty:
            raise ValueError(
                "dataset must contain distinct Estimation and Validation periods"
            )
    if not 0.5 <= estimation_fraction <= 0.9:
        raise ValueError("estimation fraction must be between 0.5 and 0.9")
    split_index = int(np.floor(len(dataframe) * estimation_fraction))
    if split_index < 2 or len(dataframe) - split_index < 2:
        raise ValueError("dataset is too short for separate estimation and validation periods")
    return (
        dataframe.iloc[:split_index].copy().reset_index(drop=True),
        dataframe.iloc[split_index:].copy().reset_index(drop=True),
    )


def simulate_guided_dataset(
    sample_minutes: float = 2.0,
    noise_multiplier: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a deterministic example by executing the same guided plan."""

    plant = LiveSprayDryer(seed=seed)
    plant.configure_time_step(sample_minutes)
    current_inputs = NOMINAL_INPUTS.copy()
    minute = 0.0
    records: list[dict[str, object]] = []
    for experiment in build_guided_plan(sample_minutes):
        current_inputs = rate_limited_inputs(
            current_inputs, experiment.target_inputs, sample_minutes
        )
        minute += sample_minutes
        plant.advance(current_inputs)
        measured_outputs = plant.measure(
            enabled=noise_multiplier > 0.0,
            multiplier=noise_multiplier,
        )
        records.append(
            sample_record(
                minute,
                sample_minutes,
                experiment,
                current_inputs,
                measured_outputs,
            )
        )
    return samples_to_dataframe(records)


def diagnose_excitation(inputs: np.ndarray, outputs: np.ndarray) -> ExcitationDiagnostics:
    """Report simple, transparent diagnostics before dynamic model fitting."""

    inputs = np.asarray(inputs, dtype=float)
    outputs = np.asarray(outputs, dtype=float)
    if inputs.ndim != 2 or inputs.shape[1] != 3:
        raise ValueError(f"inputs must have 3 columns: {INPUT_NAMES}")
    if outputs.ndim != 2 or outputs.shape[1] != 4 or len(outputs) != len(inputs):
        raise ValueError(f"outputs must have 4 columns: {OUTPUT_NAMES}")
    input_ranges = np.ptp(inputs, axis=0)
    output_ranges = np.ptp(outputs, axis=0)
    centered = inputs - np.mean(inputs, axis=0)
    input_rank = int(np.linalg.matrix_rank(centered))
    singular_values = np.linalg.svd(centered, compute_uv=False)
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if len(singular_values) and singular_values[-1] > 1e-12
        else float("inf")
    )
    warnings: list[str] = []
    blocking_errors: list[str] = []
    for index, name in enumerate(INPUT_NAMES):
        if input_ranges[index] < 0.05 * INPUT_SCALE[index]:
            blocking_errors.append(
                f"Insufficient {name} movement: span is {input_ranges[index]:.3g}."
            )
    if input_rank < len(INPUT_NAMES):
        blocking_errors.append(
            "MV excitation is rank deficient; independent input effects cannot be separated."
        )
    elif condition_number > 30.0:
        warnings.append(
            f"MV excitation is strongly correlated (condition number {condition_number:.1f})."
        )
    weak_threshold = np.maximum(4.0 * MEASUREMENT_NOISE_STD, 0.02 * OUTPUT_SCALE)
    for index, name in enumerate(OUTPUT_NAMES):
        if output_ranges[index] < weak_threshold[index]:
            warnings.append(f"Weak observed response for {name}.")
    return ExcitationDiagnostics(
        input_ranges,
        output_ranges,
        input_rank,
        condition_number,
        tuple(warnings),
        tuple(blocking_errors),
    )


def response_metrics(
    model: FittedDryerModel,
    inputs: np.ndarray,
    outputs: np.ndarray,
    sample_minutes: float,
) -> ResponseMetrics:
    """Calculate free-run output-response metrics in engineering units."""

    outputs = np.asarray(outputs, dtype=float)
    predictions = _simulate_dynamic_model(
        model, inputs, outputs[0], sample_minutes
    )
    residual = outputs - predictions
    rmse = np.sqrt(np.mean(residual**2, axis=0))
    mae = np.mean(np.abs(residual), axis=0)
    centered_norm = np.linalg.norm(outputs - np.mean(outputs, axis=0), axis=0)
    fit_percent = 100.0 * (
        1.0 - np.linalg.norm(residual, axis=0) / np.maximum(centered_norm, 1e-12)
    )
    return ResponseMetrics(predictions, rmse, mae, fit_percent, len(outputs))


def validate_candidate(
    model: FittedDryerModel,
    estimation_inputs: np.ndarray,
    estimation_outputs: np.ndarray,
    validation_inputs: np.ndarray,
    validation_outputs: np.ndarray,
    sample_minutes: float,
) -> ModelValidation:
    """Validate one candidate on distinct estimation and validation periods."""

    return ModelValidation(
        response_metrics(model, estimation_inputs, estimation_outputs, sample_minutes),
        response_metrics(model, validation_inputs, validation_outputs, sample_minutes),
    )


def model_from_controller(controller: ConstrainedDryerMPC) -> FittedDryerModel:
    """Snapshot the controller predictor without referencing the plant."""

    return FittedDryerModel(
        controller.gain_matrix.copy(),
        controller.output_tau.copy(),
        int(controller.delay_steps),
        np.full(4, np.nan),
        0,
        controller.nominal_inputs.copy(),
        controller.nominal_outputs.copy(),
        float(controller.dt),
        float(controller.delay_minutes),
    )


def apply_model(controller: ConstrainedDryerMPC, model: FittedDryerModel) -> None:
    """Apply a validated model only to the MPC predictor."""

    controller.gain_matrix = model.gain_matrix.copy()
    controller.output_tau = model.output_tau.copy()
    controller.delay_minutes = model.delay_minutes
    controller.nominal_inputs = model.nominal_inputs.copy()
    controller.nominal_outputs = model.nominal_outputs.copy()
    controller.configure_time_step(controller.dt)


def restore_builtin_model(controller: ConstrainedDryerMPC) -> None:
    """Restore all built-in MPC predictor parameters."""

    controller.gain_matrix = GAIN_MATRIX.copy()
    controller.output_tau = OUTPUT_TAU.copy()
    controller.delay_minutes = 3.0
    controller.nominal_inputs = NOMINAL_INPUTS.copy()
    controller.nominal_outputs = NOMINAL_OUTPUTS.copy()
    controller.configure_time_step(controller.dt)


def _comparison_run(
    tuning: TuningPreset,
    model: FittedDryerModel,
    output_min: np.ndarray,
    output_max: np.ndarray,
    input_min: np.ndarray,
    input_max: np.ndarray,
    maximum_move: np.ndarray,
    sample_minutes: float,
    noise_multiplier: float,
    seed: int,
    tank_change_minute: float,
    end_minute: float,
) -> ComparisonRun:
    plant = LiveSprayDryer(seed=seed)
    plant.configure_time_step(sample_minutes)
    controller = ConstrainedDryerMPC(
        prediction_horizon=tuning.prediction_horizon,
        objective_weight=tuning.objective_weight,
        move_weight=tuning.move_weight,
    )
    controller.configure_time_step(sample_minutes)
    apply_model(controller, model)
    current_inputs = NOMINAL_INPUTS.copy()
    measurements = NOMINAL_OUTPUTS.copy()
    minutes: list[float] = []
    inputs: list[np.ndarray] = []
    outputs: list[np.ndarray] = []
    tank_changed = False
    minute = 0.0
    while minute < end_minute:
        minute += sample_minutes
        current_inputs = controller.move(
            measurements,
            current_inputs,
            output_min,
            output_max,
            objective_group="output",
            objective_index=tuning.target_output_index,
            objective_mode="target",
            objective_target=tuning.target,
            input_min=input_min,
            input_max=input_max,
            max_move=maximum_move * sample_minutes,
        )
        if not tank_changed and minute >= tank_change_minute:
            plant.set_feed_dry_matter(48.5)
            tank_changed = True
        plant.advance(current_inputs)
        measurements = plant.measure(
            enabled=noise_multiplier > 0.0,
            multiplier=noise_multiplier,
        )
        minutes.append(minute)
        inputs.append(current_inputs.copy())
        outputs.append(measurements.copy())

    minute_array = np.asarray(minutes)
    input_array = np.asarray(inputs)
    output_array = np.asarray(outputs)
    event_mask = minute_array >= tank_change_minute
    event_outputs = output_array[event_mask]
    event_inputs = input_array[event_mask]
    target_error = np.abs(
        event_outputs[:, tuning.target_output_index] - tuning.target
    )
    integrated_error = float(np.sum(target_error) * sample_minutes)
    normalized_cv_error = float(
        np.sum(np.abs(event_outputs - NOMINAL_OUTPUTS) / OUTPUT_SCALE)
        * sample_minutes
    )
    lower_violation = np.maximum(output_min - event_outputs, 0.0)
    upper_violation = np.maximum(event_outputs - output_max, 0.0)
    violations = np.maximum(lower_violation, upper_violation)
    violation_count = int(np.count_nonzero(violations > 0.0))
    maximum_violation = float(np.max(violations / OUTPUT_SCALE))
    movement = np.diff(
        np.vstack([NOMINAL_INPUTS, event_inputs]), axis=0
    )
    normalized_movement = float(np.sum(np.abs(movement) / INPUT_SCALE))
    tolerance = 0.05 * OUTPUT_SCALE[tuning.target_output_index]
    required_scans = max(2, int(np.ceil(5.0 / sample_minutes)))
    recovery_minutes: float | None = None
    search_start = int(np.searchsorted(minute_array[event_mask], tank_change_minute + 5.0))
    within = target_error <= tolerance
    for index in range(search_start, len(within) - required_scans + 1):
        if np.all(within[index : index + required_scans]):
            recovery_minutes = float(
                minute_array[event_mask][index] - tank_change_minute
            )
            break
    return ComparisonRun(
        minute_array,
        input_array,
        output_array,
        ComparisonMetrics(
            recovery_minutes,
            integrated_error,
            normalized_cv_error,
            maximum_violation,
            violation_count,
            normalized_movement,
        ),
    )


def compare_tunings(
    tuning_a: TuningPreset,
    tuning_b: TuningPreset,
    model: FittedDryerModel,
    output_min: np.ndarray,
    output_max: np.ndarray,
    input_min: np.ndarray = INPUT_MIN,
    input_max: np.ndarray = INPUT_MAX,
    maximum_move: np.ndarray = MAX_MOVE,
    sample_minutes: float = 2.0,
    noise_multiplier: float = 1.0,
    seed: int = 71,
) -> TuningComparison:
    """Run a fair, deterministic tank-disturbance comparison from fresh state."""

    if tuning_a.target_output_index != tuning_b.target_output_index:
        raise ValueError("Tuning A and B must control the same output for comparison")
    if not np.isclose(tuning_a.target, tuning_b.target):
        raise ValueError("Tuning A and B must use the same target for comparison")
    tank_change_minute = 15.0
    end_minute = 75.0
    common = (
        model,
        np.asarray(output_min, dtype=float),
        np.asarray(output_max, dtype=float),
        np.asarray(input_min, dtype=float),
        np.asarray(input_max, dtype=float),
        np.asarray(maximum_move, dtype=float),
        float(sample_minutes),
        float(noise_multiplier),
        int(seed),
        tank_change_minute,
        end_minute,
    )
    return TuningComparison(
        _comparison_run(tuning_a, *common),
        _comparison_run(tuning_b, *common),
        tank_change_minute,
    )
