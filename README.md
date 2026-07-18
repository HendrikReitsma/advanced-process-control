# APC Spray Dryer Learning Lab

A Python process-control project that progresses from single-input control
foundations to a live constrained multivariable model predictive control (MPC)
spray-dryer simulation.

> This project is a process-control simulation and has not been validated for
> operational control.

![Live APC spray-dryer dashboard](docs/images/dashboard.png)

## Who This Is For

The project is intended for process and control engineers learning Python and
for technical reviewers who want an inspectable APC example. Equations,
controller behavior, constraints, and disturbances remain visible rather than
being hidden behind a control-system interface.

## Learning Progression

### 1. SISO Foundations

`run_lab.py` introduces powder moisture as the controlled variable, inlet-air
temperature as the manipulated variable, and inlet humidity as a disturbance.
It covers:

1. a first-order process simulation with noise and dead time;
2. a hand-written PID with anti-windup and actuator limits;
3. first-order-plus-dead-time fitting from a step test;
4. a compact SISO MPC and PID-versus-MPC comparison.

The script writes reproducible plots to the ignored `artifacts/` directory.

### 2. Live Multivariable APC

`live_app.py` is the capstone lab. Its compact industrial SCADA-style screen
simulates three manipulated variables:

- feed flow;
- inlet-air flow;
- inlet-air temperature.

The MPC predicts four constrained outputs:

- exhaust-air temperature;
- feed pressure;
- powder moisture;
- exhaust-air humidity.

Manual mode accepts operator input commands. APC mode uses the MPC to predict
future behavior, solve the selected Target, Maximize, or Minimize objective,
apply the first move, and solve again. Each input has an operating range, move
limit, and enable/freeze switch.

### Guided APC Showcase

Press **RUN APC SHOWCASE** in the operator station for one deterministic,
100-simulation-minute operator sequence with two minutes represented by each
scan. It starts the normal simulation in manual control, changes from Tank A to
Tank C around minute 15, enables APC around minute 35, and changes back to Tank
A around minute 65. At minute 100 the automation releases control without
resetting or pausing: the process keeps running, MPC remains active, and the
normal controls become available with all state and trends retained. **HOLD**
can pause the sequence. A new Showcase requires a normal **RESET**.

### Measurements And Disturbances

The simulated plant retains noise-free internal states. The dashboard and
controller receive measured outputs with configurable, seeded sensor noise:
Off, Low, Normal, or High.

Tank A, Tank B, and Tank C have incoming feed dry matter of 50.0%, 52.0%, and
48.5%. Manual or scheduled tank changes alter the feed water load through a
two-minute feed-line mixing lag. Feed pressure is shown on a synthetic
100-bar nominal scale. Inlet-air humidity can remain constant, follow a smooth
daily cycle above and below nominal, or include reproducible humid-weather
events.

Feed dry matter and inlet humidity are not supplied to the controller as
measured-disturbance feedforward variables. The MPC rejects their effects
through measured-output feedback after the plant responds.

### Live Process Trends

The persistent client-side Plotly.js component appends samples in place without
rebuilding the charts each scan. The left column contains the three input
commands plus feed dry matter and inlet-air humidity. The right column contains
the four controlled outputs, predictions, targets, and constraints. Both
columns share simulation time and tank/weather event markers while retaining
zoom, auto-follow, HOLD, and RESET behavior.

### Mollier / Stickiness Map

The collapsed operating-map section combines true simulated exhaust-air
temperature and humidity ratio. It includes selected relative-humidity curves,
constant-enthalpy lines, a saturation segment, a 60-sample operating trail,
a configured stickiness boundary, and a five-degree temperature safety offset.
The intersection of that safe boundary with the configured maximum exhaust-air
temperature determines the MPC upper constraint for exhaust air humidity. The
same limit appears in Process Trends, the dashboard, and the map. The lower
humidity limit remains a process/efficiency constraint. With the default
100 C maximum exhaust-air temperature, the derived upper humidity limit is
approximately 0.1286 kg water/kg dry air. The displayed margin is:

```text
stickiness margin = boundary temperature at current humidity
                  - current exhaust temperature
```

The boundary is configured simulation logic, not a universally valid product or
industrial correlation. Tank, weather, and control changes move the point only
through the simulated process response.

## Architecture

| Path | Purpose |
| --- | --- |
| `live_app.py` | Streamlit UI and live simulation orchestration. |
| `run_lab.py` | Runnable SISO learning sequence and static figures. |
| `apc_lab/live_dryer.py` | Multivariable process simulation and constrained MPC. |
| `apc_lab/equations.py` | Display-ready model and controller equations. |
| `apc_lab/model_fitting.py` | Multivariable gain, time-constant, and delay fitting. |
| `apc_lab/psychrometrics.py` | Moist-air references and configured stickiness assessment. |
| `apc_lab/process_trends_component.py` | Persistent Process Trends wrapper and payload logic. |
| `apc_lab/operating_map_component.py` | Persistent operating-map wrapper and payload logic. |
| `apc_lab/scada_ui.py` | Reusable compact SCADA styling and status helpers. |
| `apc_lab/spray_dryer.py` | Introductory SISO process simulation. |
| `apc_lab/pid.py` | PID implementation with anti-windup and limits. |
| `apc_lab/identification.py` | SISO FOPDT model and step-response fitting. |
| `apc_lab/mpc.py` | Introductory SISO MPC. |
| `tests/` | Process, controller, component, disturbance, and fitting tests. |

The live steady-state model has the form:

```text
y_ss = y_nominal + K @ (u_delayed - u_nominal)
     + k_DM * (DM - DM_nominal) + k_H * (H_in - H_in_nominal)
y_true[k+1] = y_true[k] + response_fraction * (y_ss[k] - y_true[k])
y_measured[k] = y_true[k] + sensor_noise[k]
```

The true plant, noisy measurements, and MPC predictor are separate. Fitting a
dataset updates the controller predictor and does not replace the simulated
plant.

## Installation

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On macOS or Linux, activate the environment with
`source .venv/bin/activate`.

## Run

Start the live dashboard:

```powershell
streamlit run live_app.py
```

Run the SISO learning sequence:

```powershell
python run_lab.py
```

## Streamlit Community Cloud

Deploy the repository with branch `main` and entry point `live_app.py`. The
root `requirements.txt` contains `.`, so Community Cloud installs the project,
its runtime dependencies, and the packaged Plotly.js component assets from
`pyproject.toml`. No secrets or external services are required. The CI and
hosted deployment use Python 3.12; local installations support Python 3.10 or
newer. Streamlit 1.57 or newer is required for the v2 custom-component API.

## Test

```powershell
python -m pytest -q
python -c "from streamlit.testing.v1 import AppTest; app=AppTest.from_file('live_app.py'); app.run(timeout=30); assert not app.exception"
python -m pip check
```

The tests use fixed seeds and generated process data for reproducibility.

## Model Fitting Data

The dashboard accepts evenly sampled, time-ordered CSV data with these exact
columns:

```text
Feed flow
Inlet air flow
Inlet air temperature
Exhaust air temperature
Feed pressure
Powder moisture
Exhaust air humidity
```

An example identification dataset can be downloaded from the app. Uploaded
data remains in the active Streamlit session and is not written to the
repository.

## Limitations

- The process is a linear gain, delay, and first-order lag approximation rather
  than a full mass-and-energy balance.
- Sensor noise is independent Gaussian noise without bias, drift, filtering,
  or correlated disturbances.
- Feed composition uses a simple feed-line lag, and inlet-air humidity uses a
  deterministic daily profile and generic weather event.
- The operating map uses approximate moist-air relationships and one configured
  example boundary that is not validated for a product or production dryer.
- The MPC supports one primary objective at a time.
- Dataset fitting assumes clean, numeric, evenly sampled data and does not
  calculate statistical confidence.

## License

The project is released under the [MIT License](LICENSE). The vendored
Plotly.js license is retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
