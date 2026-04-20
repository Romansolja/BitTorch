import logging

import pandas as pd
import yfinance as yf

from app.config import YFINANCE_HTTP_TIMEOUT

logger = logging.getLogger(__name__)


def flatten_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def download(symbol: str, **kwargs) -> pd.DataFrame:
    """yfinance.download wrapped with HTTP timeout and MultiIndex flattening.

    yfinance >=0.2 supports the `timeout` kwarg directly and manages its own
    (curl_cffi) session — do not pass `session=`.

    Raises yfinance- or urllib-level exceptions on network failure. Callers at
    the API boundary should translate those into HTTPException(503).
    """
    kwargs.setdefault("progress", False)
    kwargs.setdefault("auto_adjust", True)
    kwargs.setdefault("timeout", YFINANCE_HTTP_TIMEOUT)
    df = yf.download(symbol, **kwargs)
    return flatten_multiindex(df)
