from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

INPUT_NAMES = ("Feed flow", "Inlet air flow", "Inlet air temperature")
OUTPUT_NAMES = ("Exhaust air temperature", "Feed pressure", "Powder moisture", "Exhaust air humidity")
INPUT_UNITS = ("kg/min", "m3/s", "C")
OUTPUT_UNITS = ("C", "bar", "%", "kg water/kg dry air")

NOMINAL_INPUTS = np.array([100.0, 20.0, 180.0])
NOMINAL_OUTPUTS = np.array([90.0, 4.0, 4.5, 0.120])
INPUT_MIN = np.array([70.0, 14.0, 150.0])
INPUT_MAX = np.array([140.0, 28.0, 220.0])
MAX_MOVE = np.array([2.0, 0.4, 2.0])

# Rows are outputs; columns are feed flow, air flow, and inlet temperature.
GAIN_MATRIX = np.array(
    [
        [-0.18, 1.20, 0.55],
        [0.055, -0.05, 0.00],
        [0.050, -0.10, -0.055],
        [0.0022, -0.0040, -0.0018],
    ]
)
OUTPUT_TAU = np.array([5.0, 3.0, 15.0, 8.0])
INPUT_SCALE = INPUT_MAX - INPUT_MIN
OUTPUT_SCALE = np.array([30.0, 2.5, 3.0, 0.050])


def steady_outputs(
    inputs: np.ndarray,
    gain_matrix: np.ndarray = GAIN_MATRIX,
    nominal_inputs: np.ndarray = NOMINAL_INPUTS,
    nominal_outputs: np.ndarray = NOMINAL_OUTPUTS,
) -> np.ndarray:
    return nominal_outputs + gain_matrix @ (np.asarray(inputs) - nominal_inputs)


@dataclass
class LiveSprayDryer:
    """Educational 3-input, 4-output spray-dryer process."""

    dt: float = 1.0
    delay_steps: int = 3
    outputs: np.ndarray = field(default_factory=lambda: NOMINAL_OUTPUTS.copy())
    seed: int = 11
    _input_history: deque = field(init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._input_history = deque(
            [NOMINAL_INPUTS.copy() for _ in range(self.delay_steps + 1)],
            maxlen=self.delay_steps + 1,
        )
        self._rng = np.random.default_rng(self.seed)

    def step(self, inputs: np.ndarray, noisy: bool = True) -> np.ndarray:
        inputs = np.clip(np.asarray(inputs, dtype=float), INPUT_MIN, INPUT_MAX)
        self._input_history.append(inputs.copy())
        target = steady_outputs(self._input_history[0])
        self.outputs += self.dt / OUTPUT_TAU * (target - self.outputs)
        if noisy:
            noise_scale = np.array([0.12, 0.008, 0.025, 0.00035])
            return self.outputs + self._rng.normal(0.0, noise_scale)
        return self.outputs.copy()


@dataclass
class ConstrainedDryerMPC:
    """Compact multivariable MPC for visualizing plans and active constraints."""

    prediction_horizon: int = 20
    control_horizon: int = 5
    objective_weight: float = 100.0
    move_weight: float = 0.12
    gain_matrix: np.ndarray = field(default_factory=lambda: GAIN_MATRIX.copy())
    output_tau: np.ndarray = field(default_factory=lambda: OUTPUT_TAU.copy())
    delay_steps: int = 3
    nominal_inputs: np.ndarray = field(default_factory=lambda: NOMINAL_INPUTS.copy())
    nominal_outputs: np.ndarray = field(default_factory=lambda: NOMINAL_OUTPUTS.copy())
    last_input_plan: np.ndarray | None = None
    last_output_plan: np.ndarray | None = None
    last_success: bool = True
    last_message: str = ""
    last_objective_before: float = 0.0
    last_objective_after: float = 0.0
    last_move: np.ndarray = field(default_factory=lambda: np.zeros(3))
    last_limiting_constraint: str = "none"

    def predict(self, outputs: np.ndarray, current_inputs: np.ndarray, moves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        moves = moves.reshape(self.control_horizon, 3)
        padded = np.vstack(
            [moves, np.zeros((self.prediction_horizon - self.control_horizon, 3))]
        )
        input_plan = current_inputs + np.cumsum(padded, axis=0)
        y = outputs.copy()
        output_plan = np.empty((self.prediction_horizon, 4))
        for k in range(self.prediction_horizon):
            delayed_u = current_inputs if k < self.delay_steps else input_plan[k - self.delay_steps]
            y = y + (
                steady_outputs(
                    delayed_u,
                    self.gain_matrix,
                    self.nominal_inputs,
                    self.nominal_outputs,
                )
                - y
            ) / self.output_tau
            output_plan[k] = y
        return input_plan, output_plan

    def move(
        self,
        outputs: np.ndarray,
        current_inputs: np.ndarray,
        output_min: np.ndarray,
        output_max: np.ndarray,
        objective_group: str = "output",
        objective_index: int = 3,
        objective_mode: str = "target",
        objective_target: float = 0.120,
        input_min: np.ndarray = INPUT_MIN,
        input_max: np.ndarray = INPUT_MAX,
        max_move: np.ndarray = MAX_MOVE,
        input_enabled: np.ndarray | None = None,
    ) -> np.ndarray:
        input_enabled = (
            np.ones(3, dtype=bool)
            if input_enabled is None
            else np.asarray(input_enabled, dtype=bool)
        )
        input_min = np.asarray(input_min, dtype=float)
        input_max = np.asarray(input_max, dtype=float)
        max_move = np.asarray(max_move, dtype=float)
        effective_move = np.where(input_enabled, max_move, 0.0)
        constrained_min = np.where(input_enabled, input_min, current_inputs)
        constrained_max = np.where(input_enabled, input_max, current_inputs)
        input_scale = np.maximum(constrained_max - constrained_min, 1e-6)

        def objective_cost(flat_moves: np.ndarray) -> float:
            input_plan, output_plan = self.predict(outputs, current_inputs, flat_moves)
            move_scale = np.where(effective_move > 0, effective_move, 1.0)
            moves = flat_moves.reshape(self.control_horizon, 3) / move_scale
            plan = input_plan if objective_group == "input" else output_plan
            scale = input_scale if objective_group == "input" else OUTPUT_SCALE
            values = plan[:, objective_index]
            normalized = values / scale[objective_index]
            if objective_mode == "target":
                objective_cost = np.sum(
                    ((values - objective_target) / scale[objective_index]) ** 2
                )
            elif objective_mode == "maximize":
                objective_cost = -np.sum(normalized)
            else:
                objective_cost = np.sum(normalized)
            return float(
                self.objective_weight * objective_cost
                + self.move_weight * np.sum(moves * moves)
            )

        def output_constraints(flat_moves: np.ndarray) -> np.ndarray:
            _, output_plan = self.predict(outputs, current_inputs, flat_moves)
            # If noise or a manual move starts outside a limit, require each
            # prediction to be no worse and let the optimizer recover.
            initial_lower_violation = np.maximum(output_min - outputs, 0.0)
            initial_upper_violation = np.maximum(outputs - output_max, 0.0)
            recovery = np.linspace(1.0, 0.0, self.prediction_horizon)[:, None]
            lower_margin = output_plan - output_min + recovery * initial_lower_violation
            upper_margin = output_max - output_plan + recovery * initial_upper_violation
            return np.r_[lower_margin.ravel(), upper_margin.ravel()]

        def input_constraints(flat_moves: np.ndarray) -> np.ndarray:
            input_plan, _ = self.predict(outputs, current_inputs, flat_moves)
            initial_lower_violation = np.maximum(constrained_min - current_inputs, 0.0)
            initial_upper_violation = np.maximum(current_inputs - constrained_max, 0.0)
            recovery = np.linspace(1.0, 0.0, self.prediction_horizon)[:, None]
            return np.r_[
                ((input_plan - constrained_min + recovery * initial_lower_violation) / input_scale).ravel(),
                ((constrained_max - input_plan + recovery * initial_upper_violation) / input_scale).ravel(),
            ]

        bounds = [
            (-effective_move[j], effective_move[j])
            for _ in range(self.control_horizon)
            for j in range(3)
        ]
        zero_moves = np.zeros(self.control_horizon * 3)
        self.last_objective_before = objective_cost(zero_moves)
        result = minimize(
            objective_cost,
            zero_moves,
            method="SLSQP",
            bounds=bounds,
            constraints=[
                {
                    "type": "ineq",
                    "fun": lambda moves: output_constraints(moves)
                    / np.tile(OUTPUT_SCALE, self.prediction_horizon * 2),
                },
                {"type": "ineq", "fun": input_constraints},
            ],
            options={"maxiter": 150, "ftol": 1e-7},
        )
        self.last_success = bool(result.success)
        self.last_message = str(result.message)
        # A useful feasible result is better than freezing solely because SLSQP
        # stopped at its iteration limit.
        feasible = (
            np.min(output_constraints(result.x)) >= -1e-5
            and np.min(input_constraints(result.x)) >= -1e-5
        )
        moves = result.x if result.success or feasible else zero_moves
        self.last_objective_after = objective_cost(moves)
        self.last_input_plan, self.last_output_plan = self.predict(
            outputs, current_inputs, moves
        )
        self.last_move = self.last_input_plan[0] - current_inputs
        margins: list[tuple[float, str]] = []
        for i, name in enumerate(INPUT_NAMES):
            margins.append(
                (
                    float(np.min((self.last_input_plan[:, i] - constrained_min[i]) / input_scale[i])),
                    f"{name} minimum",
                )
            )
            margins.append(
                (
                    float(np.min((constrained_max[i] - self.last_input_plan[:, i]) / input_scale[i])),
                    f"{name} maximum",
                )
            )
        for i, name in enumerate(OUTPUT_NAMES):
            margins.append(
                (
                    float(np.min((self.last_output_plan[:, i] - output_min[i]) / OUTPUT_SCALE[i])),
                    f"{name} minimum",
                )
            )
            margins.append(
                (
                    float(np.min((output_max[i] - self.last_output_plan[:, i]) / OUTPUT_SCALE[i])),
                    f"{name} maximum",
                )
            )
        self.last_limiting_constraint = min(margins, key=lambda item: item[0])[1]
        return self.last_input_plan[0].copy()
