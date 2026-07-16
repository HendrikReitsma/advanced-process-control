"""Run the spray-dryer APC learning lab and save plots in ./artifacts."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from apc_lab import FOPDTModel, MPC, PID, SprayDryer, fit_fopdt

ARTIFACTS = Path("artifacts")


def step_test() -> FOPDTModel:
    dryer = SprayDryer(seed=1)
    minutes = np.arange(90)
    temperature = np.where(minutes < 15, 180.0, 190.0)
    moisture = np.array([dryer.step(u, 0.50) for u in temperature])
    model = fit_fopdt(temperature, moisture)
    fitted = model.simulate(temperature, moisture[:5].mean(), temperature[0])

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    axes[0].step(minutes, temperature, where="post")
    axes[0].set_ylabel("Inlet temp (C)")
    axes[1].plot(minutes, moisture, ".", alpha=0.6, label="noisy process")
    axes[1].plot(minutes, fitted, label="fitted FOPDT")
    axes[1].set(ylabel="Moisture (%)", xlabel="Time (min)")
    axes[1].legend()
    fig.suptitle(
        f"Step test: K={model.gain:.3f}, tau={model.tau:.1f} min, "
        f"delay={model.dead_time:.1f} min"
    )
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "01_step_test.png", dpi=150)
    plt.close(fig)
    return model


def closed_loop(model: FOPDTModel) -> None:
    n = 150
    minutes = np.arange(n)
    setpoint = np.where(minutes < 20, 5.0, 4.3)
    humidity = np.where(minutes < 80, 0.50, 0.58)

    pid_dryer = SprayDryer(seed=2)
    pid = PID(
        kp=-12.0,
        ki=-0.45,
        kd=-5.0,
        output_min=150.0,
        output_max=210.0,
        max_change=2.0,
        output=180.0,
    )
    pid_u = np.empty(n)
    pid_y = np.empty(n)
    measurement = 5.0
    for k in range(n):
        pid_u[k] = pid.update(setpoint[k], measurement)
        measurement = pid_dryer.step(pid_u[k], humidity[k])
        pid_y[k] = measurement

    mpc_dryer = SprayDryer(seed=2)
    mpc = MPC(model=model, move_penalty=0.3)
    mpc_u = np.empty(n)
    mpc_y = np.empty(n)
    measurement = 5.0
    u = 180.0
    for k in range(n):
        u = mpc.move(measurement, setpoint[k], u)
        mpc_u[k] = u
        measurement = mpc_dryer.step(u, humidity[k])
        mpc_y[k] = measurement

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    axes[0].plot(minutes, setpoint, "k--", label="setpoint")
    axes[0].plot(minutes, pid_y, label="PID")
    axes[0].plot(minutes, mpc_y, label="MPC")
    axes[0].set_ylabel("Moisture (%)")
    axes[0].legend()
    axes[1].plot(minutes, pid_u, label="PID")
    axes[1].plot(minutes, mpc_u, label="MPC")
    axes[1].axhline(210, color="k", linestyle=":", label="max temperature")
    axes[1].set_ylabel("Inlet temp (C)")
    axes[1].legend()
    axes[2].plot(minutes, humidity, color="tab:green")
    axes[2].set(ylabel="Inlet humidity", xlabel="Time (min)")
    fig.suptitle("Spray dryer: setpoint change at 20 min, disturbance at 80 min")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "02_pid_vs_mpc.png", dpi=150)
    plt.close(fig)

    pid_iae = np.abs(pid_y - setpoint).sum()
    mpc_iae = np.abs(mpc_y - setpoint).sum()
    print(f"PID integral absolute error: {pid_iae:.2f}")
    print(f"MPC integral absolute error: {mpc_iae:.2f}")
    print(f"Maximum PID move: {np.max(np.abs(np.diff(pid_u))):.2f} C/min")
    print(f"Maximum MPC move: {np.max(np.abs(np.diff(mpc_u))):.2f} C/min")


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    model = step_test()
    print(
        f"Identified model: gain={model.gain:.3f}, tau={model.tau:.1f} min, "
        f"dead time={model.dead_time:.1f} min"
    )
    closed_loop(model)
    print(f"Plots written to {ARTIFACTS.resolve()}")


if __name__ == "__main__":
    main()
