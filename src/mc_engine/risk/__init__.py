"""Portfolio risk: correlated scenarios, VaR/ES, and VaR backtesting."""

from mc_engine.risk.backtest import (
    backtest_summary,
    christoffersen_independence,
    kupiec_pof,
    rolling_var_backtest,
)
from mc_engine.risk.portfolio import OptionPosition, Portfolio, risk_report
from mc_engine.risk.scenarios import cholesky_factor, correlated_innovations, simulate_log_returns
from mc_engine.risk.var import gaussian_var_es, var_es

__all__ = [
    "OptionPosition",
    "Portfolio",
    "backtest_summary",
    "cholesky_factor",
    "christoffersen_independence",
    "correlated_innovations",
    "gaussian_var_es",
    "kupiec_pof",
    "risk_report",
    "rolling_var_backtest",
    "simulate_log_returns",
    "var_es",
]
