from dataclasses import dataclass


@dataclass
class PID:
    """PID with derivative-on-measurement, anti-windup, and output rate limit."""

    kp: float
    ki: float
    kd: float = 0.0
    dt: float = 1.0
    output_min: float = float("-inf")
    output_max: float = float("inf")
    max_change: float = float("inf")
    integral: float = 0.0
    previous_measurement: float | None = None
    output: float = 0.0
    _bias: float | None = None

    def update(self, setpoint: float, measurement: float) -> float:
        if self._bias is None:
            self._bias = self.output
        error = setpoint - measurement
        derivative = 0.0
        if self.previous_measurement is not None:
            derivative = -(measurement - self.previous_measurement) / self.dt

        def constrain(value: float) -> float:
            amplitude_limited = min(self.output_max, max(self.output_min, value))
            return min(
                self.output + self.max_change,
                max(self.output - self.max_change, amplitude_limited),
            )

        candidate_integral = self.integral + error * self.dt
        desired = (
            self._bias + self.kp * error + self.ki * candidate_integral + self.kd * derivative
        )
        constrained = constrain(desired)

        # Reject integration when it pushes farther into amplitude or rate limits.
        pushes_high = desired > constrained and error * self.ki > 0
        pushes_low = desired < constrained and error * self.ki < 0
        if not (pushes_high or pushes_low):
            self.integral = candidate_integral
        else:
            desired = (
                self._bias + self.kp * error + self.ki * self.integral + self.kd * derivative
            )
            constrained = constrain(desired)

        self.output = constrained
        self.previous_measurement = measurement
        return self.output
