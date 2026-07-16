from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .identification import FOPDTModel


@dataclass
class MPC:
    """Small SISO MPC that optimizes future actuator changes."""

    model: FOPDTModel
    prediction_horizon: int = 30
    control_horizon: int = 8
    move_penalty: float = 0.15
    u_min: float = 150.0
    u_max: float = 210.0
    max_change: float = 2.0
    _past_u: list[float] = field(default_factory=list, repr=False)

    def reset(self, initial_u: float) -> None:
        self._past_u = [initial_u] * (self.model.delay_steps + 1)

    def move(self, measurement: float, setpoint: float, current_u: float) -> float:
        if not self._past_u:
            self.reset(current_u)

        def predict(du: np.ndarray) -> np.ndarray:
            moves = np.pad(
                du, (0, self.prediction_horizon - len(du)), mode="constant"
            )
            future_u = np.clip(current_u + np.cumsum(moves), self.u_min, self.u_max)
            history = self._past_u + future_u.tolist()
            y = measurement
            predicted = np.empty(self.prediction_horizon)
            delay = self.model.delay_steps
            for k in range(self.prediction_horizon):
                delayed_u = history[len(self._past_u) + k - delay]
                target = measurement + self.model.gain * (delayed_u - current_u)
                y += self.model.dt / self.model.tau * (target - y)
                predicted[k] = y
            return predicted

        def objective(du: np.ndarray) -> float:
            error = predict(du) - setpoint
            return float(error @ error + self.move_penalty * (du @ du))

        constraints = [
            {
                "type": "ineq",
                "fun": lambda du: self.u_max
                - (current_u + np.cumsum(du)),
            },
            {
                "type": "ineq",
                "fun": lambda du: (current_u + np.cumsum(du)) - self.u_min,
            },
        ]
        result = minimize(
            objective,
            np.zeros(self.control_horizon),
            method="SLSQP",
            bounds=[(-self.max_change, self.max_change)] * self.control_horizon,
            constraints=constraints,
            options={"maxiter": 100, "ftol": 1e-6},
        )
        change = float(result.x[0]) if result.success else 0.0
        new_u = float(np.clip(current_u + change, self.u_min, self.u_max))
        self._past_u.append(new_u)
        self._past_u = self._past_u[-(self.model.delay_steps + 1) :]
        return new_u
