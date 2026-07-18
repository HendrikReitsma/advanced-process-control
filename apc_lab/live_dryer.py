from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

INPUT_NAMES = ("Feed flow", "Inlet air flow", "Inlet air temperature")
OUTPUT_NAMES = ("Exhaust air temperature", "Feed pressure", "Powder moisture", "Exhaust air humidity")
INPUT_UNITS = ("kg/min", "m3/s", "C")
OUTPUT_UNITS = ("C", "bar", "%", "kg water/kg dry air")

NOMINAL_INPUTS = np.array([100.0, 20.0, 180.0])
NOMINAL_OUTPUTS = np.array([90.0, 100.0, 4.5, 0.120])
INPUT_MIN = np.array([70.0, 14.0, 150.0])
INPUT_MAX = np.array([140.0, 28.0, 220.0])
MAX_MOVE = np.array([2.0, 0.4, 2.0])

# Rows are outputs; columns are feed flow, air flow, and inlet temperature.
GAIN_MATRIX = np.array(
    [
        [-0.18, 1.20, 0.55],
        [1.375, -1.25, 0.00],
        [0.050, -0.10, -0.055],
        [0.0022, -0.0040, -0.0018],
    ]
)
OUTPUT_TAU = np.array([5.0, 3.0, 15.0, 8.0])
INPUT_SCALE = INPUT_MAX - INPUT_MIN
OUTPUT_SCALE = np.array([30.0, 62.5, 3.0, 0.050])

# Incoming feed dry matter is an unmeasured plant disturbance. Higher dry
# matter leaves less water to evaporate, reduces outlet humidity and powder
# moisture, and increases temperature and feed pressure.
NOMINAL_FEED_DRY_MATTER = 50.0
FEED_DRY_MATTER_UNIT = "% dry matter"
# Per one percentage-point change in incoming feed dry matter, in OUTPUT_NAMES
# order. A Tank A to B switch (50% to 52% dry matter) changes steady-state
# powder moisture by -0.56 percentage points and exhaust humidity by -0.005.
FEED_DRY_MATTER_GAIN = np.array([0.28, 0.625, -0.28, -0.0025])
FEED_DRY_MATTER_TRANSITION_TAU = 2.0
FEED_TANKS = {"Tank A": 50.0, "Tank B": 52.0, "Tank C": 48.5}

# Inlet-air humidity ratio is a second unmeasured plant disturbance. Higher
# inlet humidity reduces drying potential: exhaust temperature falls while
# powder moisture and exhaust humidity rise. Feed pressure has no direct gain.
NOMINAL_INLET_HUMIDITY = 0.008
INLET_HUMIDITY_UNIT = "kg water/kg dry air"
INLET_HUMIDITY_GAIN = np.array([-450.0, 0.0, 225.0, 1.8])
INLET_HUMIDITY_DAILY_AMPLITUDE = 0.002
INLET_HUMIDITY_DAILY_PERIOD = 4 * 60
HUMID_WEATHER_INCREASE = 0.004
HUMID_WEATHER_APPROACH_MINUTES = 30
HUMID_WEATHER_PLATEAU_MINUTES = 50
HUMID_WEATHER_RECOVERY_MINUTES = 40
WEATHER_MODES = (
    "Constant",
    "Daily variation",
    "Daily variation + weather events",
)

# One-sigma sensor noise at the Normal setting, in OUTPUT_NAMES order:
# 0.20 C exhaust temperature, 0.375 bar pressure, 0.040 moisture percentage
# points, and 0.0005 kg water/kg dry air exhaust humidity.
MEASUREMENT_NOISE_STD = np.array([0.20, 0.375, 0.040, 0.0005])
NOISE_MULTIPLIERS = {"Low": 0.5, "Normal": 1.0, "High": 2.0}


def steady_outputs(
    inputs: np.ndarray,
    gain_matrix: np.ndarray = GAIN_MATRIX,
    nominal_inputs: np.ndarray = NOMINAL_INPUTS,
    nominal_outputs: np.ndarray = NOMINAL_OUTPUTS,
    feed_dry_matter: float = NOMINAL_FEED_DRY_MATTER,
    inlet_humidity: float = NOMINAL_INLET_HUMIDITY,
) -> np.ndarray:
    return (
        nominal_outputs
        + gain_matrix @ (np.asarray(inputs) - nominal_inputs)
        + FEED_DRY_MATTER_GAIN * (feed_dry_matter - NOMINAL_FEED_DRY_MATTER)
        + INLET_HUMIDITY_GAIN * (inlet_humidity - NOMINAL_INLET_HUMIDITY)
    )


@dataclass(frozen=True)
class FeedTankEvent:
    """A reproducible feed-supply switch recorded by simulation minute."""

    minute: int
    event_type: str
    old_tank: str
    new_tank: str
    old_dry_matter: float
    new_dry_matter: float


@dataclass
class FeedTankManager:
    """Manage tank selection and deterministic automatic tank changes."""

    tanks: dict[str, float] = field(default_factory=lambda: FEED_TANKS.copy())
    current_tank: str = "Tank A"
    seed: int = 23
    auto_interval: int = 60
    automatic_enabled: bool = False
    next_auto_change_minute: int = field(init=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.current_tank not in self.tanks:
            raise ValueError(f"Unknown initial tank: {self.current_tank}")
        self._rng = np.random.default_rng(self.seed)
        self.next_auto_change_minute = self.auto_interval

    @property
    def current_dry_matter(self) -> float:
        return self.tanks[self.current_tank]

    def configure_automatic(self, enabled: bool, interval: int, minute: int) -> None:
        """Apply an operator scheduling choice without backfilling events."""

        interval = int(interval)
        if enabled and (not self.automatic_enabled or interval != self.auto_interval):
            self.next_auto_change_minute = minute + interval
        self.automatic_enabled = enabled
        self.auto_interval = interval

    def change_to(
        self, tank: str, minute: int, event_type: str = "Manual tank change"
    ) -> FeedTankEvent | None:
        """Switch to a selected tank, returning no event for a no-op request."""

        if tank not in self.tanks:
            raise ValueError(f"Unknown feed tank: {tank}")
        if tank == self.current_tank:
            return None
        old_tank = self.current_tank
        old_dry_matter = self.current_dry_matter
        self.current_tank = tank
        return FeedTankEvent(
            minute=minute,
            event_type=event_type,
            old_tank=old_tank,
            new_tank=tank,
            old_dry_matter=old_dry_matter,
            new_dry_matter=self.current_dry_matter,
        )

    def maybe_automatic_change(self, minute: int) -> FeedTankEvent | None:
        """Return at most one deterministic event for a given scheduled minute."""

        if not self.automatic_enabled or minute < self.next_auto_change_minute:
            return None
        candidates = [
            tank
            for tank, dry_matter in self.tanks.items()
            if tank != self.current_tank
            and abs(dry_matter - self.current_dry_matter) >= 0.5
        ]
        new_tank = str(self._rng.choice(candidates))
        event = self.change_to(new_tank, minute, "Automatic tank change")
        self.next_auto_change_minute = minute + self.auto_interval
        return event


@dataclass(frozen=True)
class WeatherEvent:
    """A reproducible inlet-air humidity event recorded by simulation minute."""

    minute: int
    event_type: str = "HUMID WEATHER"
    humidity_increase: float = HUMID_WEATHER_INCREASE


def _smoothstep(fraction: float) -> float:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return fraction * fraction * (3.0 - 2.0 * fraction)


@dataclass
class InletWeatherManager:
    """Generate deterministic daily inlet humidity and smooth weather events."""

    mode: str = "Constant"
    seed: int = 37
    inlet_humidity: float = NOMINAL_INLET_HUMIDITY
    state: str = "NORMAL"
    active_event_minute: int | None = None
    next_auto_event_minute: int = field(init=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in WEATHER_MODES:
            raise ValueError(f"Unknown inlet weather mode: {self.mode}")
        self._rng = np.random.default_rng(self.seed)
        self.next_auto_event_minute = self._next_auto_minute(0)

    @property
    def event_duration(self) -> int:
        return (
            HUMID_WEATHER_APPROACH_MINUTES
            + HUMID_WEATHER_PLATEAU_MINUTES
            + HUMID_WEATHER_RECOVERY_MINUTES
        )

    def _next_auto_minute(self, minute: int) -> int:
        return int(minute + self._rng.integers(180, 301))

    def _base_humidity(self, minute: float) -> float:
        if self.mode == "Constant":
            return NOMINAL_INLET_HUMIDITY
        phase = 2.0 * np.pi * float(minute) / INLET_HUMIDITY_DAILY_PERIOD
        return float(
            NOMINAL_INLET_HUMIDITY
            + INLET_HUMIDITY_DAILY_AMPLITUDE * np.sin(phase)
        )

    def _event_offset(self, minute: float) -> float:
        if self.active_event_minute is None:
            self.state = "NORMAL"
            return 0.0
        elapsed = float(minute - self.active_event_minute)
        if elapsed < HUMID_WEATHER_APPROACH_MINUTES:
            self.state = "APPROACHING"
            return HUMID_WEATHER_INCREASE * _smoothstep(
                elapsed / HUMID_WEATHER_APPROACH_MINUTES
            )
        plateau_end = (
            HUMID_WEATHER_APPROACH_MINUTES + HUMID_WEATHER_PLATEAU_MINUTES
        )
        if elapsed < plateau_end:
            self.state = "HUMID/STORM"
            return HUMID_WEATHER_INCREASE
        if elapsed < self.event_duration:
            self.state = "RECOVERING"
            return HUMID_WEATHER_INCREASE * (
                1.0
                - _smoothstep(
                    (elapsed - plateau_end) / HUMID_WEATHER_RECOVERY_MINUTES
                )
            )
        self.active_event_minute = None
        self.state = "NORMAL"
        return 0.0

    def configure_mode(self, mode: str, minute: int) -> None:
        """Apply an operator mode choice without generating rerun events."""

        if mode not in WEATHER_MODES:
            raise ValueError(f"Unknown inlet weather mode: {mode}")
        if mode != self.mode and mode == WEATHER_MODES[2]:
            self.next_auto_event_minute = self._next_auto_minute(minute)
        self.mode = mode
        self.inlet_humidity = self._base_humidity(minute) + self._event_offset(minute)

    def trigger(self, minute: int) -> WeatherEvent | None:
        """Start one manual or automatic event, ignoring duplicate triggers."""

        if self.active_event_minute is not None:
            return None
        self.active_event_minute = int(minute)
        self.state = "APPROACHING"
        self.next_auto_event_minute = self._next_auto_minute(
            minute + self.event_duration
        )
        return WeatherEvent(minute=int(minute))

    def advance(self, minute: int) -> WeatherEvent | None:
        """Update humidity from simulation time and return at most one event."""

        event = None
        if (
            self.mode == WEATHER_MODES[2]
            and self.active_event_minute is None
            and minute >= self.next_auto_event_minute
        ):
            event = self.trigger(minute)
        self.inlet_humidity = self._base_humidity(minute) + self._event_offset(minute)
        return event


@dataclass
class LiveSprayDryer:
    """Educational 3-input, 4-output spray-dryer process."""

    dt: float = 1.0
    delay_steps: int = 3
    delay_minutes: float = 3.0
    true_outputs: np.ndarray = field(default_factory=lambda: NOMINAL_OUTPUTS.copy())
    seed: int = 11
    feed_dry_matter: float = NOMINAL_FEED_DRY_MATTER
    feed_dry_matter_target: float = NOMINAL_FEED_DRY_MATTER
    feed_dry_matter_transition_tau: float = FEED_DRY_MATTER_TRANSITION_TAU
    inlet_humidity: float = NOMINAL_INLET_HUMIDITY
    _input_history: deque = field(init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._input_history = deque(
            [NOMINAL_INPUTS.copy() for _ in range(self.delay_steps + 1)],
            maxlen=self.delay_steps + 1,
        )
        self._rng = np.random.default_rng(self.seed)

    def set_feed_dry_matter(self, value: float) -> None:
        """Set the incoming feed dry-matter target after a tank change."""

        self.feed_dry_matter_target = float(value)

    def set_inlet_humidity(self, value: float) -> None:
        """Set the true inlet-air humidity ratio used by the synthetic plant."""

        self.inlet_humidity = float(value)

    def configure_time_step(self, minutes: float) -> None:
        """Set the simulated minutes represented by one live dashboard tick."""

        if minutes <= 0:
            raise ValueError("Simulation time step must be positive")
        self.dt = float(minutes)
        delay_steps = max(1, int(np.ceil(self.delay_minutes / self.dt)))
        if delay_steps != self.delay_steps:
            latest_input = self._input_history[-1].copy()
            self.delay_steps = delay_steps
            self._input_history = deque(
                [latest_input.copy() for _ in range(delay_steps + 1)],
                maxlen=delay_steps + 1,
            )

    def advance(self, inputs: np.ndarray) -> np.ndarray:
        """Advance the noise-free plant state by one simulation interval."""

        inputs = np.clip(np.asarray(inputs, dtype=float), INPUT_MIN, INPUT_MAX)
        self._input_history.append(inputs.copy())
        feed_line_fraction = 1.0 - np.exp(
            -self.dt / self.feed_dry_matter_transition_tau
        )
        self.feed_dry_matter += feed_line_fraction * (
            self.feed_dry_matter_target - self.feed_dry_matter
        )
        target = steady_outputs(
            self._input_history[0],
            feed_dry_matter=self.feed_dry_matter,
            inlet_humidity=self.inlet_humidity,
        )
        output_fraction = 1.0 - np.exp(-self.dt / OUTPUT_TAU)
        self.true_outputs += output_fraction * (target - self.true_outputs)
        return self.true_outputs.copy()

    def measure(self, enabled: bool = True, multiplier: float = 1.0) -> np.ndarray:
        """Sample sensors without modifying the underlying plant state."""

        if not enabled or multiplier <= 0:
            return self.true_outputs.copy()
        noise = self._rng.normal(0.0, MEASUREMENT_NOISE_STD * multiplier)
        return self.true_outputs + noise

    def step(
        self,
        inputs: np.ndarray,
        noisy: bool = True,
        noise_multiplier: float = 1.0,
    ) -> np.ndarray:
        """Compatibility helper that advances the plant and samples sensors."""

        self.advance(inputs)
        return self.measure(enabled=noisy, multiplier=noise_multiplier)


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
    dt: float = 1.0
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

    def configure_time_step(self, minutes: float) -> None:
        """Keep the predictor sample time aligned with the plant scan."""

        if minutes <= 0:
            raise ValueError("Simulation time step must be positive")
        self.dt = float(minutes)
        self.delay_steps = max(1, int(np.ceil(3.0 / self.dt)))

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
            response_fraction = 1.0 - np.exp(-self.dt / self.output_tau)
            y = y + response_fraction * (
                steady_outputs(
                    delayed_u,
                    self.gain_matrix,
                    self.nominal_inputs,
                    self.nominal_outputs,
                )
                - y
            )
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
