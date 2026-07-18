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

## Repository Hygiene

- Keep `docs/images/dashboard.png` generic and free of private browser data.
- Do not commit caches, local environments, secrets, logs, or generated lab
  output.
- Use pandas for CSV parsing and NumPy arrays for model calculations.
- Add focused tests for changed process directions, constraints, objectives,
  frozen inputs, or fitting behavior.

## Implementation Style

Prefer the smallest change that directly satisfies the request.

- Reuse the existing architecture, helpers, state and configuration.
- Prefer parameter, configuration or CSS changes over new logic when they can solve the problem cleanly.
- Do not broaden the task into refactoring, redesign or general hardening.
- Do not add abstractions, fallback systems, dependencies or test infrastructure for hypothetical future needs.
- Fix problems that are actually observed. Do not investigate speculative edge cases unless they present an obvious safety or correctness risk.
- Do not create multiple competing implementations.
- Preserve working behaviour outside the requested scope.
- Once the requested change works and the relevant verification passes, stop.

Keep task updates concise. Do not narrate every file inspection or routine command.

## Proportional Verification

Match verification to the size and risk of the change.

- Documentation, wording, `.gitignore`, CSS and simple parameter changes: inspect the diff and run one directly relevant check.
- Live dashboard or UI changes: run Streamlit AppTest. Perform one visual inspection only when appearance cannot be verified otherwise.
- Model, controller, simulation-state or event changes: add or update focused tests and run `python -m pytest -q` once.
- SISO-track changes: also run `python run_lab.py`.
- Dependency or packaging changes: run `python -m pip check` and the relevant package-build check.
- Before committing and pushing a substantial change: run the full applicable verification once.

Do not repeatedly rerun successful checks. If a check is blocked by the environment, report it instead of searching for elaborate workarounds.

Always run:

```powershell
git diff --check
