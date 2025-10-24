from .alpha_vantage_common import _make_api_request, format_datetime_for_api
from datetime import datetime, timedelta
import json
from typing import Any, Dict, List, Optional


def _call_news_sentiment(params: Dict[str, Any]) -> Dict[str, Any]:
    """Call Alpha Vantage NEWS_SENTIMENT and return parsed JSON."""
    raw_response = _make_api_request("NEWS_SENTIMENT", params)
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unexpected response format from Alpha Vantage NEWS_SENTIMENT: {exc}") from exc
    return parsed


def _serialize_news_payload(feed: List[Dict[str, Any]], metadata: Dict[str, Any]) -> str:
    """Return a JSON string that mirrors Alpha Vantage's structure but only for requested articles."""
    payload = {
        "feed": feed,
        "total_items": len(feed),
        "metadata": metadata,
    }
    return json.dumps(payload, default=str)


def _parse_time_published(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        # Normalize Z suffix for fromisoformat on Python 3.10
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except ValueError:
        try:
            return datetime.strptime(value, "%Y%m%dT%H%M%S")
        except ValueError:
            return None


def get_news(ticker, start_date, end_date) -> dict[str, str] | str:
    """Returns live and historical market news & sentiment data from premier news outlets worldwide.

    Covers stocks, cryptocurrencies, forex, and topics like fiscal policy, mergers & acquisitions, IPOs.

    Args:
        ticker: Stock symbol for news articles.
        start_date: Start date for news search.
        end_date: End date for news search.

    Returns:
        Dictionary containing news sentiment data or JSON string.
    """

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    if start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt

    total_limit = 10
    metadata: Dict[str, Any] = {"symbol": ticker, "requested_range": [start_date, end_date]}

    params = {
        "tickers": ticker,
        "sort": "LATEST",
        "limit": str(total_limit),
    }

    # Supply server-side lower bound only; filter upper bound locally to reduce response latency
    params["time_from"] = format_datetime_for_api(start_dt.strftime("%Y-%m-%d"))

    response_json = _call_news_sentiment(params)
    feed_items = response_json.get("feed", []) or []

    start_bound = start_dt.date()
    end_bound = end_dt.date()
    filtered_feed: List[Dict[str, Any]] = []

    for item in feed_items:
        published = _parse_time_published(item.get("time_published"))
        if published is None:
            continue
        published_date = published.date()
        if start_bound <= published_date <= end_bound:
            filtered_feed.append(item)

    # Sort by publication time descending to keep most recent
    filtered_feed.sort(key=lambda entry: _parse_time_published(entry.get("time_published")) or datetime.min, reverse=True)

    return _serialize_news_payload(filtered_feed[:total_limit], metadata)


def get_insider_transactions(symbol: str) -> dict[str, str] | str:
    """Returns latest and historical insider transactions by key stakeholders.

    Covers transactions by founders, executives, board members, etc.

    Args:
        symbol: Ticker symbol. Example: "IBM".

    Returns:
        Dictionary containing insider transaction data or JSON string.
    """

    params = {
        "symbol": symbol,
    }

    return _make_api_request("INSIDER_TRANSACTIONS", params)


def get_global_news(curr_date: str, look_back_days: int = 7, limit: int = 5) -> dict[str, str] | str:
    """Returns global/macroeconomic news and sentiment data from Alpha Vantage.

    Uses Alpha Vantage NEWS_SENTIMENT API with topics instead of specific tickers
    to fetch global market and economic news relevant for trading decisions.

    Args:
        curr_date: Current date in yyyy-mm-dd format.
        look_back_days: Number of days to look back for news (default: 7).
        limit: Maximum number of news articles to return (default: 5).

    Returns:
        Dictionary containing global news sentiment data or JSON string.
    """
    # Calculate start date based on look_back_days
    end_date = datetime.strptime(curr_date, '%Y-%m-%d')
    start_date = end_date - timedelta(days=look_back_days)

    # Use topics parameter for global/macro news instead of specific tickers
    # Available topics: blockchain, earnings, ipo, mergers_and_acquisitions,
    # financial_markets, economy, finance, life_sciences, manufacturing,
    # real_estate, retail_wholesale, technology
    params = {
        "topics": "economy,financial_markets,technology",
        "time_from": format_datetime_for_api(start_date.strftime('%Y-%m-%d')),
        "sort": "LATEST",
        "limit": str(max(1, min(limit, 10))),
    }

    response_json = _call_news_sentiment(params)
    feed_items = response_json.get("feed", []) or []

    start_bound = start_date.date()
    end_bound = end_date.date()

    filtered_feed: List[Dict[str, Any]] = []
    for item in feed_items:
        published = _parse_time_published(item.get("time_published"))
        if published is None:
            continue
        published_date = published.date()
        if start_bound <= published_date <= end_bound:
            filtered_feed.append(item)

    filtered_feed.sort(key=lambda entry: _parse_time_published(entry.get("time_published")) or datetime.min, reverse=True)

    metadata = {
        "topics": params["topics"],
        "requested_range": [start_date.strftime('%Y-%m-%d'), curr_date],
        "look_back_days": look_back_days,
    }
    return _serialize_news_payload(filtered_feed[:limit], metadata)
