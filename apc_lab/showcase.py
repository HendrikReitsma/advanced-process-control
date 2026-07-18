"""Deterministic orchestration and summary metrics for the APC showcase."""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from apc_lab.live_dryer import INPUT_SCALE, NOMINAL_OUTPUTS, OUTPUT_SCALE


SHOWCASE_SEED = 2026
SHOWCASE_SCAN_MINUTES = 2
SHOWCASE_TANK_CHANGE_MINUTE = 15
SHOWCASE_APC_MINUTE = 35
SHOWCASE_CHALLENGE_MINUTE = 65
SHOWCASE_END_MINUTE = 100
SHOWCASE_TARGET_TOLERANCE = 0.10 * OUTPUT_SCALE


@dataclass(frozen=True)
class ShowcaseControllerTuning:
    """More responsive MPC settings used only by the guided scenario."""

    prediction_horizon: int = 20
    control_horizon: int = 5
    objective_weight: float = 1000.0
    move_weight: float = 0.005


SHOWCASE_CONTROLLER_TUNING = ShowcaseControllerTuning()


def apply_showcase_controller_tuning(controller: object) -> None:
    """Apply the guided-run MPC preset without changing normal mode defaults."""

    for name, value in SHOWCASE_CONTROLLER_TUNING.__dict__.items():
        setattr(controller, name, value)


class ShowcasePhase(str, Enum):
    IDLE = "IDLE"
    BASELINE = "BASELINE"
    MANUAL_DRIFT = "MANUAL_DRIFT"
    APC_TAKEOVER = "APC_TAKEOVER"
    APC_CHALLENGE = "APC_CHALLENGE"
    COMPLETE = "COMPLETE"


ACTION_TANK_CHANGE = "tank_change"
ACTION_APC_ENABLE = "apc_enable"
ACTION_APC_CHALLENGE = "apc_challenge"
ACTION_COMPLETE = "complete"

ACTION_SCHEDULE = (
    (SHOWCASE_TANK_CHANGE_MINUTE, ACTION_TANK_CHANGE, "tank change"),
    (SHOWCASE_APC_MINUTE, ACTION_APC_ENABLE, "APC takeover"),
    (SHOWCASE_CHALLENGE_MINUTE, ACTION_APC_CHALLENGE, "APC challenge"),
    (SHOWCASE_END_MINUTE, ACTION_COMPLETE, "showcase complete"),
)


@dataclass(frozen=True)
class PhaseInfo:
    number: int
    title: str
    description: str


PHASE_INFO = {
    ShowcasePhase.BASELINE: PhaseInfo(
        1, "MANUAL BASELINE", "Stable operation at nominal manual inputs"
    ),
    ShowcasePhase.MANUAL_DRIFT: PhaseInfo(
        2, "MANUAL TANK RESPONSE", "Tank change is visible while APC is held off"
    ),
    ShowcasePhase.APC_TAKEOVER: PhaseInfo(
        3, "APC RECOVERY", "Controller is recovering outputs toward target"
    ),
    ShowcasePhase.APC_CHALLENGE: PhaseInfo(
        4, "APC TANK RESPONSE", "APC is rejecting the second tank change"
    ),
    ShowcasePhase.COMPLETE: PhaseInfo(
        5, "SEQUENCE COMPLETE", "Final trends and guided-run metrics are retained"
    ),
}


def phase_for_minute(minute: int) -> ShowcasePhase:
    """Return the guided phase for one scenario simulation minute."""

    if minute >= SHOWCASE_END_MINUTE:
        return ShowcasePhase.COMPLETE
    if minute >= SHOWCASE_CHALLENGE_MINUTE:
        return ShowcasePhase.APC_CHALLENGE
    if minute >= SHOWCASE_APC_MINUTE:
        return ShowcasePhase.APC_TAKEOVER
    if minute >= SHOWCASE_TANK_CHANGE_MINUTE:
        return ShowcasePhase.MANUAL_DRIFT
    return ShowcasePhase.BASELINE


@dataclass
class ShowcaseState:
    """Per-session, idempotent state for the single guided scenario."""

    phase: ShowcasePhase = ShowcasePhase.IDLE
    scenario_minute: int = 0
    executed_actions: set[str] = field(default_factory=set)
    seed: int = SHOWCASE_SEED

    @property
    def engaged(self) -> bool:
        return self.phase != ShowcasePhase.IDLE

    @property
    def complete(self) -> bool:
        return self.phase == ShowcasePhase.COMPLETE

    @property
    def apc_enabled(self) -> bool:
        return ACTION_APC_ENABLE in self.executed_actions

    @property
    def progress(self) -> float:
        return min(max(self.scenario_minute / SHOWCASE_END_MINUTE, 0.0), 1.0)

    @property
    def info(self) -> PhaseInfo:
        if not self.engaged:
            raise ValueError("The idle showcase has no active phase information")
        return PHASE_INFO[self.phase]

    def start(self) -> None:
        self.phase = ShowcasePhase.BASELINE
        self.scenario_minute = 0
        self.executed_actions.clear()

    def stop(self) -> None:
        self.phase = ShowcasePhase.IDLE
        self.scenario_minute = 0
        self.executed_actions.clear()

    def advance(self, simulation_minute: int, running: bool = True) -> list[str]:
        """Advance from simulation time and return each due action once."""

        if not self.engaged or self.complete or not running:
            return []
        self.scenario_minute = min(int(simulation_minute), SHOWCASE_END_MINUTE)
        due_actions = []
        for minute, action, _ in ACTION_SCHEDULE:
            if self.scenario_minute >= minute and action not in self.executed_actions:
                self.executed_actions.add(action)
                due_actions.append(action)
        self.phase = phase_for_minute(self.scenario_minute)
        return due_actions

    def next_action(self) -> tuple[str, int] | None:
        for minute, action, label in ACTION_SCHEDULE:
            if action not in self.executed_actions:
                return label, max(0, minute - self.scenario_minute)
        return None


@dataclass(frozen=True)
class ShowcaseEvent:
    minute: int
    event_type: str
    detail: str


def calculate_showcase_metrics(
    minutes: list[float],
    inputs: np.ndarray,
    outputs: np.ndarray,
    output_min: np.ndarray,
    output_max: np.ndarray,
) -> dict[str, object]:
    """Summarize one guided run using normalized, unit-independent errors."""

    time = np.asarray(minutes, dtype=float)
    mv = np.asarray(inputs, dtype=float)
    cv = np.asarray(outputs, dtype=float)
    if time.size == 0:
        return {"periods": [], "recovery_time": None, "total_mv_movement": 0.0}

    normalized_error = np.abs(cv - NOMINAL_OUTPUTS) / OUTPUT_SCALE
    within_target = np.all(
        np.abs(cv - NOMINAL_OUTPUTS) <= SHOWCASE_TARGET_TOLERANCE,
        axis=1,
    )
    sample_minutes = np.diff(time, prepend=time[0])
    if time.size > 1:
        sample_minutes[0] = float(np.median(np.diff(time)))
    else:
        sample_minutes[0] = 1.0

    periods = []
    for label, start, end in (
        ("Manual disturbance period", SHOWCASE_TANK_CHANGE_MINUTE, SHOWCASE_APC_MINUTE),
        ("APC recovery/challenge period", SHOWCASE_APC_MINUTE, SHOWCASE_END_MINUTE + 1),
    ):
        mask = (time >= start) & (time < end)
        period_error = normalized_error[mask]
        period_minutes = sample_minutes[mask]
        periods.append(
            {
                "Period": label,
                "Time within target bands": (
                    f"{100.0 * np.mean(within_target[mask]):.1f}%"
                    if np.any(mask)
                    else "--"
                ),
                "Normalized cumulative CV error": (
                    float(np.sum(np.mean(period_error, axis=1) * period_minutes))
                    if np.any(mask)
                    else 0.0
                ),
                "Maximum moisture deviation": (
                    float(np.max(np.abs(cv[mask, 2] - NOMINAL_OUTPUTS[2])))
                    if np.any(mask)
                    else 0.0
                ),
                "Output constraint violations": (
                    int(np.sum((cv[mask] < output_min) | (cv[mask] > output_max)))
                    if np.any(mask)
                    else 0
                ),
            }
        )

    recovery_time = None
    apc_indices = np.flatnonzero(time >= SHOWCASE_APC_MINUTE)
    for index in apc_indices:
        if index + 5 <= len(time) and np.all(within_target[index : index + 5]):
            recovery_time = float(time[index] - SHOWCASE_APC_MINUTE)
            break

    total_mv_movement = 0.0
    if len(mv) > 1:
        total_mv_movement = float(np.sum(np.abs(np.diff(mv, axis=0)) / INPUT_SCALE))

    return {
        "periods": periods,
        "recovery_time": recovery_time,
        "total_mv_movement": total_mv_movement,
    }
