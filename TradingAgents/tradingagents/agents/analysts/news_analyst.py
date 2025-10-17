from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_news, get_global_news
from tradingagents.dataflows.config import get_config


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        tools = [
            get_news,
            get_global_news,
        ]

        system_message = (
            "Analyze recent news using get_news (company-specific) and get_global_news (macroeconomic). "
            "Write detailed analysis with trading insights. End with Markdown table summary."
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
            "news_report": report,
        }

    return news_analyst_node
