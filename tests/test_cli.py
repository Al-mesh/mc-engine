import re

import pytest

from mc_engine.cli import main


def run(capsys, *argv) -> str:
    assert main(list(argv)) == 0
    return capsys.readouterr().out


def gap_in_se(out: str) -> float:
    return float(re.search(r"reference = ([+-][\d.]+) SE", out).group(1))


@pytest.mark.parametrize(
    "argv",
    [
        ["--model", "gbm", "--payoff", "european"],
        ["--model", "gbm", "--payoff", "european", "--put", "--antithetic", "--paths", "50001"],
        ["--model", "gbm", "--payoff", "digital"],
        ["--model", "gbm", "--payoff", "geometric-asian", "--steps", "12"],
        ["--model", "gbm", "--payoff", "european", "--qmc"],
        ["--model", "heston", "--payoff", "european", "--paths", "50000"],
        ["--model", "heston", "--payoff", "european", "--paths", "20000", "--qmc"],
        ["--model", "merton", "--payoff", "european"],
    ],
)
def test_priced_with_reference_agree(capsys, argv):
    out = run(capsys, "price", *argv)
    assert "reference" in out
    assert abs(gap_in_se(out)) < 4


@pytest.mark.parametrize("payoff", ["asian", "barrier", "lookback-floating", "lookback-fixed"])
def test_exotics_run(capsys, payoff):
    extra = ["--barrier", "130"] if payoff == "barrier" else []
    out = run(capsys, "price", "--model", "heston", "--payoff", payoff, "--paths", "4000", *extra)
    assert "95% CI" in out and "reference" not in out


def test_american_put_against_tree(capsys):
    out = run(
        capsys,
        "price",
        "--payoff",
        "american",
        "--put",
        "--S0",
        "36",
        "--K",
        "40",
        "--r",
        "0.06",
        "--paths",
        "50000",
        "--steps",
        "50",
    )
    assert "CRR Bermudan" in out and abs(gap_in_se(out)) < 4


def test_greeks_command(capsys):
    out = run(capsys, "greeks", "--paths", "20000")
    for g in ("delta", "gamma", "vega", "rho"):
        assert g in out
    assert "pathwise" in out and "lr" in out


def test_barrier_argument_errors():
    with pytest.raises(SystemExit):
        main(["price", "--payoff", "barrier"])
    with pytest.raises(SystemExit):
        main(["price", "--payoff", "barrier", "--barrier", "130", "--put"])
