from streamlit.testing.v1 import AppTest

from apc_lab.equations import (
    ALL_EQUATIONS,
    FEED_LINE_EQUATION,
    MEASUREMENT_EQUATION,
    OUTPUT_EFFECT_EQUATIONS,
    STEADY_STATE_EQUATION,
)


def test_latex_sources_are_balanced_and_not_wrapped_in_markdown_delimiters():
    for equation in ALL_EQUATIONS:
        assert equation.count("{") == equation.count("}")
        assert "$$" not in equation
        assert "\\" in equation


def test_latex_sources_contain_the_intended_model_notation():
    assert r"\mathbf{k}_{DM}(DM-DM_0)" in STEADY_STATE_EQUATION
    assert r"\tau_{DM}" in FEED_LINE_EQUATION
    assert r"\epsilon_i\sim\mathcal{N}(0,\sigma_i^2)" in MEASUREMENT_EQUATION
    assert r"M_{powder,ss}" in OUTPUT_EFFECT_EQUATIONS
    assert r"H_{exh,ss}" in OUTPUT_EFFECT_EQUATIONS


def test_live_app_renders_each_model_equation_as_latex():
    app = AppTest.from_file("live_app.py").run(timeout=30)

    assert not app.exception
    assert len(app.latex) == len(ALL_EQUATIONS)
    for source, rendered in zip(ALL_EQUATIONS, app.latex):
        assert source in rendered.value
