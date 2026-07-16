from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class FOPDTModel:
    gain: float
    tau: float
    dead_time: float
    dt: float = 1.0

    @property
    def delay_steps(self) -> int:
        return max(0, round(self.dead_time / self.dt))

    def simulate(self, u: np.ndarray, y0: float, u0: float) -> np.ndarray:
        y = np.empty(len(u), dtype=float)
        y[0] = y0
        delay = self.delay_steps
        for k in range(1, len(u)):
            delayed_u = u[max(0, k - delay)]
            target = y0 + self.gain * (delayed_u - u0)
            y[k] = y[k - 1] + self.dt / self.tau * (target - y[k - 1])
        return y


def fit_fopdt(u: np.ndarray, y: np.ndarray, dt: float = 1.0) -> FOPDTModel:
    """Fit gain, time constant, and dead time to step-response data."""

    u = np.asarray(u, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(u) != len(y) or len(u) < 5:
        raise ValueError("u and y must have equal lengths of at least 5")

    y0 = float(np.mean(y[: min(5, len(y))]))
    u0 = float(u[0])
    du = u[-1] - u0
    gain_guess = (y[-1] - y0) / du if abs(du) > 1e-9 else -0.05

    def residual(parameters: np.ndarray) -> np.ndarray:
        model = FOPDTModel(parameters[0], parameters[1], parameters[2], dt)
        return model.simulate(u, y0, u0) - y

    result = least_squares(
        residual,
        x0=np.array([gain_guess, 15.0, 4.0]),
        bounds=([-10.0, dt, 0.0], [10.0, 200.0, 30.0]),
    )
    return FOPDTModel(*result.x, dt=dt)
