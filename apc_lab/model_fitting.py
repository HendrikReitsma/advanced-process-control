from dataclasses import dataclass

import numpy as np

from .live_dryer import INPUT_NAMES, OUTPUT_NAMES


@dataclass(frozen=True)
class FittedDryerModel:
    gain_matrix: np.ndarray
    output_tau: np.ndarray
    delay_steps: int
    rmse: np.ndarray
    samples: int
    nominal_inputs: np.ndarray
    nominal_outputs: np.ndarray
    sample_minutes: float = 1.0
    dead_time_minutes: float | None = None

    @property
    def delay_minutes(self) -> float:
        """Return the fitted dead time in physical simulation minutes."""

        if self.dead_time_minutes is not None:
            return float(self.dead_time_minutes)
        return float(self.delay_steps * self.sample_minutes)


def fit_dynamic_model(
    inputs: np.ndarray,
    outputs: np.ndarray,
    max_delay: int = 10,
    dt: float = 1.0,
) -> FittedDryerModel:
    """Fit a multivariable first-order model by searching integer input delay.

    For each output, linear least squares fits:
    dy/dt = alpha * (nominal_y - y) + beta @ (delayed_u - nominal_u)
    Then tau = 1/alpha and process gains = beta/alpha.
    """

    inputs = np.asarray(inputs, dtype=float)
    outputs = np.asarray(outputs, dtype=float)
    if inputs.ndim != 2 or inputs.shape[1] != 3:
        raise ValueError(f"inputs must have 3 columns: {INPUT_NAMES}")
    if outputs.ndim != 2 or outputs.shape[1] != 4:
        raise ValueError(f"outputs must have 4 columns: {OUTPUT_NAMES}")
    if len(inputs) != len(outputs) or len(inputs) < max_delay + 10:
        raise ValueError("input and output data must have equal length and enough samples")
    if not np.all(np.isfinite(inputs)) or not np.all(np.isfinite(outputs)):
        raise ValueError("dataset contains missing or non-numeric values")

    nominal_inputs = np.mean(inputs, axis=0)
    nominal_outputs = np.mean(outputs, axis=0)
    best: FittedDryerModel | None = None
    best_error = float("inf")
    for delay in range(max_delay + 1):
        start = max(1, delay)
        current_y = outputs[start:-1]
        derivative = (outputs[start + 1 :] - current_y) / dt
        delayed_u = inputs[start - delay : len(inputs) - 1 - delay]
        gains = np.empty((4, 3))
        tau = np.empty(4)
        predictions = np.empty_like(derivative)
        for output_index in range(4):
            design = np.column_stack(
                [
                    nominal_outputs[output_index] - current_y[:, output_index],
                    delayed_u - nominal_inputs,
                ]
            )
            coefficients, *_ = np.linalg.lstsq(
                design, derivative[:, output_index], rcond=None
            )
            alpha = float(np.clip(coefficients[0], 1 / 300.0, 1.0))
            tau[output_index] = 1.0 / alpha
            gains[output_index] = coefficients[1:] / alpha
            predictions[:, output_index] = design @ coefficients
        residual = derivative - predictions
        rmse = np.sqrt(np.mean(residual**2, axis=0))
        score = float(np.sum((rmse / np.maximum(np.std(derivative, axis=0), 1e-9)) ** 2))
        if score < best_error:
            best_error = score
            best = FittedDryerModel(
                gains,
                tau,
                delay,
                rmse,
                len(inputs),
                nominal_inputs,
                nominal_outputs,
                float(dt),
            )
    assert best is not None
    return best


def simulate_dynamic_model(
    model: FittedDryerModel,
    inputs: np.ndarray,
    initial_outputs: np.ndarray,
    dt: float | None = None,
) -> np.ndarray:
    """Free-run a fitted first-order model over an input sequence.

    Predicted outputs are propagated from prior predictions, rather than being
    corrected with measured outputs at every sample. This makes the result
    suitable for independent validation overlays and output-response metrics.
    """

    inputs = np.asarray(inputs, dtype=float)
    initial_outputs = np.asarray(initial_outputs, dtype=float)
    if inputs.ndim != 2 or inputs.shape[1] != 3:
        raise ValueError(f"inputs must have 3 columns: {INPUT_NAMES}")
    if initial_outputs.shape != (4,):
        raise ValueError(f"initial_outputs must have 4 values: {OUTPUT_NAMES}")
    if not np.all(np.isfinite(inputs)) or not np.all(np.isfinite(initial_outputs)):
        raise ValueError("simulation inputs contain missing or non-numeric values")
    sample_minutes = model.sample_minutes if dt is None else float(dt)
    if sample_minutes <= 0:
        raise ValueError("sample time must be positive")

    delay_steps = max(0, int(np.ceil(model.delay_minutes / sample_minutes)))
    predictions = np.empty((len(inputs), 4), dtype=float)
    if len(inputs) == 0:
        return predictions
    predictions[0] = initial_outputs
    response_fraction = 1.0 - np.exp(-sample_minutes / model.output_tau)
    for sample_index in range(1, len(inputs)):
        delayed_index = sample_index - delay_steps
        delayed_inputs = (
            model.nominal_inputs
            if delayed_index < 0
            else inputs[delayed_index]
        )
        steady_prediction = model.nominal_outputs + model.gain_matrix @ (
            delayed_inputs - model.nominal_inputs
        )
        predictions[sample_index] = predictions[sample_index - 1] + response_fraction * (
            steady_prediction - predictions[sample_index - 1]
        )
    return predictions


def arrays_from_dataframe(dataframe) -> tuple[np.ndarray, np.ndarray]:
    required = list(INPUT_NAMES + OUTPUT_NAMES)
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    return (
        dataframe[list(INPUT_NAMES)].to_numpy(dtype=float),
        dataframe[list(OUTPUT_NAMES)].to_numpy(dtype=float),
    )
