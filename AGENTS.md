# Agent Guide

## Project Goal

Maintain an educational Python project that progresses from SISO process-
control foundations to a live constrained multivariable MPC spray-dryer lab.
The intended audience is process/control engineers learning Python and
technical reviewers.

All model values and data are synthetic. Never describe the model as validated
for operational control, and do not add employer-specific names, tags, paths,
screenshots, or production data.

## Scope

The primary application is `live_app.py`. The introductory SISO track is
`run_lab.py` plus `spray_dryer.py`, `pid.py`, `identification.py`, and `mpc.py`.
Preserve both tracks and explain their learning relationship in the README.

Prefer small maintenance, documentation, and test changes. Do not add major
APC, MPC, modelling, identification, or UI features unless explicitly asked.

## Important Invariants

The live model uses fixed array order throughout the model, UI, fitting, and
tests:

```text
Inputs:  Feed flow, Inlet air flow, Inlet air temperature
Outputs: Exhaust air temperature, Feed pressure, Powder moisture,
         Exhaust air humidity
```

Do not reorder these values without updating every dependent array, UI index,
CSV column rule, and test.

Keep the simulated plant separate from the controller model. Applying a fitted
dataset updates the MPC predictor, not the synthetic plant.

When changing persisted `ConstrainedDryerMPC` fields or defaults, increment
`CONTROLLER_VERSION` in `live_app.py` so open Streamlit sessions do not retain
stale controller objects.

## Verification

From an activated environment installed with `python -m pip install -e
".[dev]"`, run:

```powershell
python -m pytest -q
python -c "from streamlit.testing.v1 import AppTest; app=AppTest.from_file('live_app.py'); app.run(timeout=30); assert not app.exception"
python -m pip check
```

Also run `python run_lab.py` when changing the SISO track. Its generated images
belong in the ignored `artifacts/` directory.

## Repository Hygiene

- Keep `docs/images/dashboard.png` generic and free of private browser data.
- Do not commit caches, local environments, secrets, logs, or generated lab
  output.
- Use pandas for CSV parsing and NumPy arrays for model calculations.
- Add focused tests for changed process directions, constraints, objectives,
  frozen inputs, or fitting behavior.
