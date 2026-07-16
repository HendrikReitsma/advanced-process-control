import numpy as np

from apc_lab.live_dryer import GAIN_MATRIX, INPUT_MAX, INPUT_MIN, NOMINAL_INPUTS, OUTPUT_TAU, LiveSprayDryer
from apc_lab.model_fitting import fit_dynamic_model


def test_dynamic_model_fit_recovers_simulated_model():
    rng = np.random.default_rng(9)
    inputs = np.tile(NOMINAL_INPUTS, (500, 1))
    current = NOMINAL_INPUTS.copy()
    for k in range(0, len(inputs), 12):
        current = np.clip(
            NOMINAL_INPUTS + rng.uniform([-20, -4, -20], [20, 4, 20]),
            INPUT_MIN,
            INPUT_MAX,
        )
        inputs[k : k + 12] = current
    dryer = LiveSprayDryer(seed=9)
    outputs = np.array([dryer.step(row, noisy=False) for row in inputs])

    fitted = fit_dynamic_model(inputs, outputs, max_delay=8)

    assert abs(fitted.delay_steps - dryer.delay_steps) <= 1
    assert np.allclose(fitted.output_tau, OUTPUT_TAU, rtol=0.25, atol=1.0)
    assert np.allclose(fitted.gain_matrix, GAIN_MATRIX, rtol=0.25, atol=0.01)
