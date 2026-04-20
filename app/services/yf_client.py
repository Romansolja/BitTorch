import logging

import pandas as pd
import yfinance as yf

from app.config import YFINANCE_HTTP_TIMEOUT

logger = logging.getLogger(__name__)


class UpstreamDataError(Exception):
    """Raised when a yfinance-backed download fails for any reason.

    Normalizes the zoo of possible causes (requests errors, yfinance
    exceptions, curl_cffi errors, urllib connection errors, JSON decode
    errors, etc.) into a single class callers can map to HTTP 503.
    """


def flatten_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def download(symbol: str, **kwargs) -> pd.DataFrame:
    """yfinance.download wrapped with HTTP timeout and MultiIndex flattening.

    yfinance >=0.2 supports the `timeout` kwarg directly and manages its own
    (curl_cffi) session — do not pass `session=`.

    Raises UpstreamDataError on any underlying failure so API routes can
    treat all data-provider outages uniformly (503).
    """
    if "session" in kwargs:
        logger.warning(
            "yf_client.download() ignores caller-provided session kwarg; "
            "yfinance manages its own curl_cffi session."
        )
        kwargs.pop("session")

    kwargs.setdefault("progress", False)
    kwargs.setdefault("auto_adjust", True)
    kwargs.setdefault("timeout", YFINANCE_HTTP_TIMEOUT)

    try:
        df = yf.download(symbol, **kwargs)
    except Exception as e:
        logger.warning("yfinance download failed for %s: %s", symbol, e)
        raise UpstreamDataError(f"yfinance download failed for {symbol}: {e}") from e

    return flatten_multiindex(df)
