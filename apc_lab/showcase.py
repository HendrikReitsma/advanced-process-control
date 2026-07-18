"""Deterministic orchestration and summary metrics for the APC showcase."""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from apc_lab.live_dryer import INPUT_SCALE, NOMINAL_OUTPUTS, OUTPUT_SCALE


SHOWCASE_SEED = 2026
SHOWCASE_END_MINUTE = 100
SHOWCASE_APC_MINUTE = 55
SHOWCASE_TARGET_TOLERANCE = 0.10 * OUTPUT_SCALE


class ShowcasePhase(str, Enum):
    IDLE = "IDLE"
    BASELINE = "BASELINE"
    HUMID_WEATHER = "HUMID_WEATHER"
    TANK_CHANGE = "TANK_CHANGE"
    MANUAL_DRIFT = "MANUAL_DRIFT"
    APC_TAKEOVER = "APC_TAKEOVER"
    APC_CHALLENGE = "APC_CHALLENGE"
    COMPLETE = "COMPLETE"


ACTION_HUMID_WEATHER = "humid_weather"
ACTION_TANK_CHANGE = "tank_change"
ACTION_APC_ENABLE = "apc_enable"
ACTION_APC_CHALLENGE = "apc_challenge"
ACTION_COMPLETE = "complete"

ACTION_SCHEDULE = (
    (15, ACTION_HUMID_WEATHER, "humid weather"),
    (30, ACTION_TANK_CHANGE, "tank change"),
    (55, ACTION_APC_ENABLE, "APC takeover"),
    (75, ACTION_APC_CHALLENGE, "APC challenge"),
    (100, ACTION_COMPLETE, "showcase complete"),
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
    ShowcasePhase.HUMID_WEATHER: PhaseInfo(
        2, "HUMID WEATHER", "Inlet-air moisture is disturbing the process"
    ),
    ShowcasePhase.TANK_CHANGE: PhaseInfo(
        3, "TANK CHANGE", "Feed composition has changed; manual inputs stay fixed"
    ),
    ShowcasePhase.MANUAL_DRIFT: PhaseInfo(
        4, "MANUAL OPERATION UNDER DISTURBANCE", "Outputs drift with APC held off"
    ),
    ShowcasePhase.APC_TAKEOVER: PhaseInfo(
        5, "APC TAKEOVER", "Controller is recovering outputs toward target"
    ),
    ShowcasePhase.APC_CHALLENGE: PhaseInfo(
        6, "APC DISTURBANCE REJECTION", "APC is responding to a later tank change"
    ),
    ShowcasePhase.COMPLETE: PhaseInfo(
        7, "SEQUENCE COMPLETE", "Final trends and guided-run metrics are retained"
    ),
}


def phase_for_minute(minute: int) -> ShowcasePhase:
    """Return the guided phase for one scenario simulation minute."""

    if minute >= 100:
        return ShowcasePhase.COMPLETE
    if minute >= 75:
        return ShowcasePhase.APC_CHALLENGE
    if minute >= 55:
        return ShowcasePhase.APC_TAKEOVER
    if minute >= 40:
        return ShowcasePhase.MANUAL_DRIFT
    if minute >= 30:
        return ShowcasePhase.TANK_CHANGE
    if minute >= 15:
        return ShowcasePhase.HUMID_WEATHER
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
        ("Manual disturbance period", 15, SHOWCASE_APC_MINUTE),
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
