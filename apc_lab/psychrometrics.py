"""Small SI psychrometric helpers for the operating map."""

from dataclasses import dataclass

import numpy as np


STANDARD_PRESSURE_KPA = 101.325
MAP_HUMIDITY_RANGE = (0.04, 0.22)
MAP_TEMPERATURE_RANGE = (55.0, 135.0)
RELATIVE_HUMIDITY_LEVELS = (10, 20, 30, 50)
ENTHALPY_LEVELS = (250, 350, 450, 550)

# This configured curve is not product- or dryer-specific.
# Temperature is in degC and humidity ratio is kg water/kg dry air.
STICKINESS_BOUNDARY = {
    "reference_humidity": 0.120,
    "reference_temperature": 108.0,
    "linear_coefficient": -340.0,
    "quadratic_coefficient": -900.0,
    "safety_margin": 5.0,
}


@dataclass(frozen=True)
class StickinessAssessment:
    boundary_temperature: float
    margin: float
    status: str


def moist_air_enthalpy(temperature_c: float, humidity_ratio: float) -> float:
    """Return moist-air enthalpy in kJ/kg dry air."""

    return float(
        1.006 * temperature_c
        + humidity_ratio * (2501.0 + 1.86 * temperature_c)
    )


def saturation_vapor_pressure_kpa(temperature_c: float | np.ndarray) -> np.ndarray:
    """Return water saturation pressure from the two-range Antoine equation.

    Antoine coefficients use temperature in degC and pressure in mmHg. The
    lower set covers 1-100 degC and the upper set covers 99-374 degC; pressure
    is converted to kPa. The map only evaluates the 60-125 degC overlap.
    """

    temperature = np.asarray(temperature_c, dtype=float)
    low = temperature <= 99.0
    coefficient_a = np.where(low, 8.07131, 8.14019)
    coefficient_b = np.where(low, 1730.63, 1810.94)
    coefficient_c = np.where(low, 233.426, 244.485)
    pressure_mmhg = 10.0 ** (
        coefficient_a - coefficient_b / (coefficient_c + temperature)
    )
    return pressure_mmhg * 0.133322368


def humidity_ratio_at_relative_humidity(
    temperature_c: float | np.ndarray,
    relative_humidity: float,
    pressure_kpa: float = STANDARD_PRESSURE_KPA,
) -> np.ndarray:
    """Return humidity ratio for relative humidity expressed from 0 to 1."""

    if not 0.0 < relative_humidity <= 1.0:
        raise ValueError("Relative humidity must be in the interval (0, 1]")
    vapor_pressure = relative_humidity * saturation_vapor_pressure_kpa(
        temperature_c
    )
    denominator = pressure_kpa - vapor_pressure
    return np.where(
        denominator > 0.0,
        0.621945 * vapor_pressure / denominator,
        np.nan,
    )


def stickiness_boundary_temperature(humidity_ratio: float | np.ndarray) -> np.ndarray:
    """Return the configured stickiness-boundary temperature in degC."""

    humidity = np.asarray(humidity_ratio, dtype=float)
    delta = humidity - STICKINESS_BOUNDARY["reference_humidity"]
    return (
        STICKINESS_BOUNDARY["reference_temperature"]
        + STICKINESS_BOUNDARY["linear_coefficient"] * delta
        + STICKINESS_BOUNDARY["quadratic_coefficient"] * delta**2
    )


def safe_stickiness_temperature(humidity_ratio: float | np.ndarray) -> np.ndarray:
    """Return the configured stickiness boundary offset by its safety margin."""

    return stickiness_boundary_temperature(humidity_ratio) - STICKINESS_BOUNDARY[
        "safety_margin"
    ]


def maximum_safe_humidity_ratio(maximum_temperature: float) -> float:
    """Solve the safe boundary for the relevant humidity ratio on the map."""

    coefficients = (
        STICKINESS_BOUNDARY["quadratic_coefficient"],
        STICKINESS_BOUNDARY["linear_coefficient"],
        STICKINESS_BOUNDARY["reference_temperature"]
        - STICKINESS_BOUNDARY["safety_margin"]
        - float(maximum_temperature),
    )
    roots = np.roots(coefficients) + STICKINESS_BOUNDARY["reference_humidity"]
    humidity_min, humidity_max = MAP_HUMIDITY_RANGE
    valid_roots = [
        float(root.real)
        for root in roots
        if np.isreal(root) and humidity_min <= root.real <= humidity_max
    ]
    if len(valid_roots) != 1:
        raise ValueError(
            "The safe stickiness boundary must intersect the map exactly once."
        )
    return valid_roots[0]


def assess_stickiness(
    exhaust_temperature: float, exhaust_humidity: float
) -> StickinessAssessment:
    """Classify the simulated point relative to the configured boundary."""

    boundary = float(stickiness_boundary_temperature(exhaust_humidity))
    margin = boundary - float(exhaust_temperature)
    if margin < 0.0:
        status = "STICKY RISK"
    elif margin <= STICKINESS_BOUNDARY["safety_margin"]:
        status = "APPROACHING"
    else:
        status = "SAFE"
    return StickinessAssessment(boundary, margin, status)


def psychrometric_background(maximum_exhaust_temperature: float) -> dict[str, object]:
    """Build bounded, serializable reference curves for the Plotly component."""

    temperatures = np.linspace(*MAP_TEMPERATURE_RANGE, 261)
    humidity_min, humidity_max = MAP_HUMIDITY_RANGE

    relative_humidity_curves = []
    for percent in RELATIVE_HUMIDITY_LEVELS:
        humidity = humidity_ratio_at_relative_humidity(
            temperatures, percent / 100.0
        )
        mask = np.isfinite(humidity) & (humidity >= humidity_min) & (
            humidity <= humidity_max
        )
        relative_humidity_curves.append(
            {
                "label": f"{percent}% RH",
                "humidity": humidity[mask].tolist(),
                "temperature": temperatures[mask].tolist(),
            }
        )

    enthalpy_curves = []
    for enthalpy in ENTHALPY_LEVELS:
        humidity = (enthalpy - 1.006 * temperatures) / (
            2501.0 + 1.86 * temperatures
        )
        mask = (humidity >= humidity_min) & (humidity <= humidity_max)
        enthalpy_curves.append(
            {
                "label": f"h={enthalpy} kJ/kg",
                "humidity": humidity[mask].tolist(),
                "temperature": temperatures[mask].tolist(),
            }
        )

    saturation = humidity_ratio_at_relative_humidity(temperatures, 1.0)
    saturation_mask = np.isfinite(saturation) & (saturation >= humidity_min) & (
        saturation <= humidity_max
    )
    boundary_humidity = np.linspace(*MAP_HUMIDITY_RANGE, 180)
    maximum_humidity = maximum_safe_humidity_ratio(maximum_exhaust_temperature)

    return {
        "pressure_kpa": STANDARD_PRESSURE_KPA,
        "humidity_range": list(MAP_HUMIDITY_RANGE),
        "temperature_range": list(MAP_TEMPERATURE_RANGE),
        "relative_humidity_curves": relative_humidity_curves,
        "enthalpy_curves": enthalpy_curves,
        "saturation": {
            "label": "Saturation",
            "humidity": saturation[saturation_mask].tolist(),
            "temperature": temperatures[saturation_mask].tolist(),
        },
        "boundary": {
            "label": "Stickiness boundary",
            "humidity": boundary_humidity.tolist(),
            "temperature": stickiness_boundary_temperature(
                boundary_humidity
            ).tolist(),
        },
        "safe_boundary": {
            "label": f"Safe boundary (-{STICKINESS_BOUNDARY['safety_margin']:.0f} C)",
            "humidity": boundary_humidity.tolist(),
            "temperature": safe_stickiness_temperature(boundary_humidity).tolist(),
        },
        "constraint": {
            "maximum_exhaust_temperature": float(maximum_exhaust_temperature),
            "maximum_humidity": maximum_humidity,
            "safety_margin": STICKINESS_BOUNDARY["safety_margin"],
        },
    }
