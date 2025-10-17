from dotenv import load_dotenv
import sys  # noqa: E402
import os  # noqa: E402
load_dotenv()
sys.path.append(os.path.join(os.path.dirname(__file__), 'TradingAgents'))

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

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

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# forward propagate
print("What ticker do you want to trade?")
ticker = input()
print(f"You want to trade {ticker}")
print("What date do you want to trade?")
date = input()
print(f"You want to trade {ticker} on {date}")
print("Running the trading agent...")
# forward propagate
_, decision = ta.propagate(ticker, date)
print("The trading agent has finished running")
print(f"The decision is {decision}")

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns