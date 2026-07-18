"""Display-ready LaTeX source strings used by the live model explanation."""

STEADY_STATE_EQUATION = (
    r"\mathbf{y}_{ss}=\mathbf{y}_0+\mathbf{K}"
    r"(\mathbf{u}_{delayed}-\mathbf{u}_0)+\mathbf{k}_{DM}(DM-DM_0)"
    r"+\mathbf{k}_{H}(H_{in}-H_{in,0})"
)

PLANT_DYNAMICS_EQUATION = (
    r"y_{true,i}(k+1)=y_{true,i}(k)+\frac{\Delta t}{\tau_i}"
    r"\left(y_{ss,i}-y_{true,i}(k)\right)"
)

FEED_LINE_EQUATION = (
    r"DM(k+1)=DM(k)+\frac{\Delta t}{\tau_{DM}}"
    r"\left(DM_{tank}-DM(k)\right)"
)

MEASUREMENT_EQUATION = (
    r"y_{measured,i}(k)=y_{true,i}(k)+\epsilon_i(k),\qquad "
    r"\epsilon_i\sim\mathcal{N}(0,\sigma_i^2)"
)

OUTPUT_EFFECT_EQUATIONS = (
    r"\begin{aligned}"
    r"T_{exh,ss} &= 90-0.18\Delta F+1.20\Delta A+0.55\Delta T_{in}"
    r"+0.28\Delta DM-450\Delta H_{in} \\"
    r"P_{feed,ss} &= 100+1.375\Delta F-1.25\Delta A+0.625\Delta DM \\"
    r"M_{powder,ss} &= 4.5+0.050\Delta F-0.10\Delta A-0.055\Delta T_{in}"
    r"-0.28\Delta DM+225\Delta H_{in} \\"
    r"H_{exh,ss} &= 0.12+0.0022\Delta F-0.0040\Delta A-0.0018\Delta T_{in}"
    r"-0.0025\Delta DM+1.8\Delta H_{in}"
    r"\end{aligned}"
)

MPC_OBJECTIVE_EQUATION = (
    r"J=w_oJ_o+w_m\sum_{k=0}^{N_c-1}"
    r"\left\lVert\Delta\mathbf{u}_k\right\rVert_2^2"
)

ALL_EQUATIONS = (
    STEADY_STATE_EQUATION,
    PLANT_DYNAMICS_EQUATION,
    FEED_LINE_EQUATION,
    MEASUREMENT_EQUATION,
    OUTPUT_EFFECT_EQUATIONS,
    MPC_OBJECTIVE_EQUATION,
)
