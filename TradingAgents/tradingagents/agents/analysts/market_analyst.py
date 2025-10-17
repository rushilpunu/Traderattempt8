from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_stock_data, get_indicators
from tradingagents.dataflows.config import get_config


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        tools = [
            get_stock_data,
            get_indicators,
        ]

        system_message = (
            """You are a trading assistant analyzing financial markets. Select up to 8 relevant indicators from this list:

Moving Averages: close_50_sma (50-day SMA), close_200_sma (200-day SMA), close_10_ema (10-day EMA)
MACD: macd, macds (signal), macdh (histogram)
Momentum: rsi (RSI)
Volatility: boll (middle), boll_ub (upper), boll_lb (lower), atr (ATR)
Volume: vwma (volume-weighted MA)

Instructions:
1. Call get_stock_data first, then get_indicators with exact indicator names
2. Select diverse, complementary indicators (avoid redundancy like rsi + stochrsi)
3. Write detailed trend analysis with specific insights for trading decisions
4. End with a Markdown table summarizing key points

Provide nuanced analysis - avoid generic "mixed trends" statements."""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Collaborative AI assistant. Use tools to analyze. If final decision reached, prefix with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**.\n"
                    "Tools: {tool_names}\n{system_message}\nDate: {current_date}, Company: {ticker}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content
       
        return {
            "messages": [result],
            "market_report": report,
        }

    return market_analyst_node
