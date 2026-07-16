from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SprayDryer:
    """Simple spray-dryer moisture model.

    Higher inlet-air temperature dries the powder. Higher inlet humidity makes
    the powder wetter. The temperature effect has transport delay and lag.
    """

    dt: float = 1.0
    tau: float = 18.0
    dead_time: float = 5.0
    moisture: float = 5.0
    nominal_temperature: float = 180.0
    nominal_humidity: float = 0.50
    temperature_gain: float = -0.08
    humidity_gain: float = 4.0
    noise_std: float = 0.04
    seed: int = 7
    _temperature_history: deque = field(init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        delay_steps = max(0, round(self.dead_time / self.dt))
        self._temperature_history = deque(
            [self.nominal_temperature] * (delay_steps + 1),
            maxlen=delay_steps + 1,
        )
        self._rng = np.random.default_rng(self.seed)

    def step(
        self, inlet_temperature: float, inlet_humidity: float, noisy: bool = True
    ) -> float:
        self._temperature_history.append(float(inlet_temperature))
        delayed_temperature = self._temperature_history[0]
        steady_moisture = (
            5.0
            + self.temperature_gain
            * (delayed_temperature - self.nominal_temperature)
            + self.humidity_gain * (inlet_humidity - self.nominal_humidity)
        )
        self.moisture += self.dt / self.tau * (steady_moisture - self.moisture)
        noise = self._rng.normal(0.0, self.noise_std) if noisy else 0.0
        return self.moisture + noise
