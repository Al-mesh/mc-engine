"""Historical prices with an on-disk cache, so tests and CI never touch the network.

The first call downloads adjusted closes with yfinance (an optional dependency:
``pip install mc-engine[risk]``) and writes a CSV to ``cache_dir``. Every later call reads
the CSV. The cached file is committed to the repository, which makes the backtest
results reproducible byte for byte even if Yahoo later revises its history.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TICKERS = ("AAPL", "MSFT", "JPM", "XOM", "JNJ")
DEFAULT_START = "2010-01-01"
DEFAULT_END = "2025-01-01"


def cache_path(tickers: Sequence[str], start: str, end: str, cache_dir: Path) -> Path:
    return Path(cache_dir) / f"prices_{'-'.join(tickers)}_{start}_{end}.csv"


def load_prices(
    tickers: Sequence[str] = DEFAULT_TICKERS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_dir: Path | str = "data",
    refresh: bool = False,
    session: object | None = None,
) -> pd.DataFrame:
    """Daily adjusted closes, one column per ticker, rows with any missing value dropped.

    ``session`` is passed to yfinance (a ``curl_cffi.requests.Session``), e.g. to supply a
    custom CA bundle on machines where antivirus software intercepts TLS.
    """
    path = cache_path(tickers, start, end, Path(cache_dir))
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("downloading prices needs yfinance: pip install mc-engine[risk]") from exc
    extra = {} if session is None else {"session": session}
    raw = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
        **extra,
    )
    prices = raw["Close"][list(tickers)].dropna()
    if prices.empty:
        raise RuntimeError("yfinance returned no data; check the network and tickers")
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path, float_format="%.6f")
    return prices


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff().dropna()
