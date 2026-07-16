import numpy as np

from apc_lab import MPC, PID, SprayDryer, fit_fopdt


def test_hotter_air_reduces_moisture_after_delay():
    dryer = SprayDryer(noise_std=0.0)
    values = [dryer.step(195.0, 0.5, noisy=False) for _ in range(50)]
    assert values[2] == 5.0
    assert values[-1] < 4.2


def test_pid_respects_output_and_rate_limits():
    pid = PID(
        kp=-100,
        ki=-10,
        output_min=150,
        output_max=210,
        max_change=2,
        output=180,
    )
    outputs = [pid.update(4.0, 6.0) for _ in range(30)]
    assert max(outputs) <= 210
    assert np.max(np.abs(np.diff([180.0] + outputs))) <= 2.0


def test_fopdt_fit_recovers_dryer_parameters():
    dryer = SprayDryer(noise_std=0.0)
    u = np.r_[np.full(15, 180.0), np.full(100, 190.0)]
    y = np.array([dryer.step(value, 0.5, noisy=False) for value in u])
    model = fit_fopdt(u, y)
    assert abs(model.gain - dryer.temperature_gain) < 0.02
    assert abs(model.tau - dryer.tau) < 5
    assert abs(model.dead_time - dryer.dead_time) < 3


def test_mpc_respects_temperature_and_move_constraints():
    model = fit_fopdt(
        np.r_[np.full(10, 180.0), np.full(80, 190.0)],
        _step_data(),
    )
    controller = MPC(model=model)
    u = 180.0
    moves = []
    for _ in range(20):
        next_u = controller.move(5.0, 4.0, u)
        moves.append(next_u - u)
        u = next_u
    assert 150 <= u <= 210
    assert np.max(np.abs(moves)) <= 2.0 + 1e-6


def _step_data():
    dryer = SprayDryer(noise_std=0.0)
    u = np.r_[np.full(10, 180.0), np.full(80, 190.0)]
    return np.array([dryer.step(value, 0.5, noisy=False) for value in u])
