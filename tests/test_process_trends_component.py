from apc_lab.process_trends_component import prepare_trend_payload


def _payload_inputs():
    return {
        "run_id": 1,
        "sample_id": 4,
        "sample": {
            "sample_id": 4,
            "time": 4.0,
            "inputs": [100.0, 20.0, 180.0, 50.0, 0.008],
            "outputs": [90.0, 4.0, 4.5, 0.12],
        },
        "events": [],
        "last_sample_id": 3,
        "last_event_id": 0,
        "snapshot": [],
        "config": {"input_names": [], "output_names": []},
        "predictions": {"times": [], "inputs": [], "outputs": []},
    }


def test_payload_contains_expected_fields_and_filters_duplicate_sample():
    arguments = _payload_inputs()
    payload, sample_cursor, _ = prepare_trend_payload(**arguments)

    assert set(payload) == {
        "run_id",
        "sample_id",
        "sample",
        "events",
        "snapshot",
        "config",
        "predictions",
    }
    assert payload["sample"]["sample_id"] == 4
    assert sample_cursor == 4

    arguments["last_sample_id"] = sample_cursor
    duplicate_payload, duplicate_cursor, _ = prepare_trend_payload(**arguments)
    assert duplicate_payload["sample"] is None
    assert duplicate_cursor == 4


def test_payload_includes_each_tank_event_once():
    arguments = _payload_inputs()
    arguments["events"] = [
        {"event_id": 1, "time": 3, "type": "Manual tank change"}
    ]

    payload, _, event_cursor = prepare_trend_payload(**arguments)
    assert [event["event_id"] for event in payload["events"]] == [1]
    assert event_cursor == 1

    arguments["last_event_id"] = event_cursor
    duplicate_payload, _, duplicate_cursor = prepare_trend_payload(**arguments)
    assert duplicate_payload["events"] == []
    assert duplicate_cursor == 1
