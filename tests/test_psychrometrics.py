import numpy as np
import pytest

from apc_lab.operating_map_component import prepare_operating_map_payload
from apc_lab.psychrometrics import (
    assess_stickiness,
    humidity_ratio_at_relative_humidity,
    moist_air_enthalpy,
    stickiness_boundary_temperature,
)


def test_moist_air_enthalpy_at_known_exhaust_condition():
    assert moist_air_enthalpy(90.0, 0.120) == pytest.approx(410.748)


def test_relative_humidity_curve_is_positive_and_increases_with_temperature():
    humidity = humidity_ratio_at_relative_humidity(
        np.array([60.0, 80.0, 100.0]), 0.20
    )

    assert np.all(np.isfinite(humidity))
    assert np.all(humidity > 0.0)
    assert np.all(np.diff(humidity) > 0.0)


def test_stickiness_classification_on_both_sides_of_boundary():
    humidity = 0.120
    boundary = float(stickiness_boundary_temperature(humidity))

    assert assess_stickiness(boundary - 10.0, humidity).status == "SAFE"
    assert assess_stickiness(boundary - 2.0, humidity).status == "APPROACHING"
    assert assess_stickiness(boundary + 1.0, humidity).status == "STICKY RISK"


def test_reset_payload_starts_new_run_with_empty_trail_snapshot():
    current = {"sample_id": 0, "temperature": 90.0, "humidity": 0.120}
    payload, cursor = prepare_operating_map_payload(
        run_id=2,
        sample=None,
        current=current,
        snapshot=[],
        last_sample_id=-1,
        background={},
    )

    assert payload["run_id"] == 2
    assert payload["snapshot"] == []
    assert payload["sample"] is None
    assert cursor == -1
