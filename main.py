from dotenv import load_dotenv
import sys  # noqa: E402
import os  # noqa: E402
from datetime import datetime

load_dotenv()
sys.path.append(os.path.join(os.path.dirname(__file__), 'TradingAgents'))

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG


def validate_and_format_date(date_input):
    """
    Validates and formats date input to YYYY-MM-DD format.

    Accepts multiple formats:
    - YYYY/MM/DD (e.g., 2025/10/14)
    - YYYY-MM-DD (e.g., 2025-10-14)
    - YYYYMMDD (e.g., 20251014)

    Args:
        date_input (str): Date string in any supported format

    Returns:
        str: Date in YYYY-MM-DD format

    Raises:
        ValueError: If date format is invalid
    """
    # Try multiple common date formats
    formats = [
        '%Y/%m/%d',   # 2025/10/14
        '%Y-%m-%d',   # 2025-10-14
        '%Y%m%d',     # 20251014
        '%m/%d/%Y',   # 10/14/2025
        '%m-%d-%Y',   # 10-14-2025
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(date_input.strip(), fmt)
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            continue

    # If no format matched, raise an error
    raise ValueError(
        f"Invalid date format: '{date_input}'\n"
        f"Please use one of these formats:\n"
        f"  - YYYY-MM-DD (e.g., 2025-10-14)\n"
        f"  - YYYY/MM/DD (e.g., 2025/10/14)\n"
        f"  - YYYYMMDD (e.g., 20251014)"
    )

# Create a custom config
config = DEFAULT_CONFIG.copy()
config["deep_think_llm"] = "gpt-5-nano"  # Use a different model
config["quick_think_llm"] = "gpt-5-nano"  # Use a different model
config["max_debate_rounds"] = 1  # Increase debate rounds

# Configure data vendors (default uses yfinance and Alpha Vantage)
config["data_vendors"] = {
    "core_stock_apis": "yfinance",           # Options: yfinance, alpha_vantage, local
    "technical_indicators": "yfinance",      # Options: yfinance, alpha_vantage, local
    "fundamental_data": "alpha_vantage",     # Options: openai, alpha_vantage, local
    "news_data": "alpha_vantage",            # Options: openai, alpha_vantage, google, local
}
config["strict_vendor_routing"] = True

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# forward propagate
print("What ticker do you want to trade?")
ticker = input().strip().upper()
print(f"You want to trade {ticker}")

print("What date do you want to trade? (YYYY-MM-DD, YYYY/MM/DD, or YYYYMMDD)")
date_input = input().strip()

try:
    date = validate_and_format_date(date_input)
    print(f"Trading {ticker} on {date}")
except ValueError as e:
    print(f"Error: {e}")
    sys.exit(1)

print("Running the trading agent...")
# forward propagate
_, decision = ta.propagate(ticker, date)
print("The trading agent has finished running")
print(f"The decision is {decision}")

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
