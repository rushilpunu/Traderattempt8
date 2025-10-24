import functools
import time
import json


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        if past_memories:
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "No past memories found."

        context = {
            "role": "user",
            "content": f"Based on a comprehensive analysis by a team of analysts, here is an investment plan tailored for {company_name}. This plan incorporates insights from current technical market trends, macroeconomic indicators, and social media sentiment. Use this plan as a foundation for evaluating your next trading decision.\n\nProposed Investment Plan: {investment_plan}\n\nLeverage these insights to make an informed and strategic decision.",
        }

        messages = [
            {
                "role": "system",
                "content": f"""You are a trading agent analyzing market data to make intramonth investment decisions. Prefer intraweek to intramonth horizons; avoid multi-month or multi-year outlooks.

**Critical**: You must objectively evaluate whether to BUY, HOLD, or SELL based on the investment plan. All three decisions are equally valid:
- **BUY**: When technical and fundamental analysis show clear upside catalysts with favorable risk/reward
- **HOLD**: When evidence is mixed, unclear, or risks and rewards are balanced
- **SELL**: When downside risks dominate, technical support is broken, or negative catalysts outweigh positives

Your response must include:
1. **Decision**: BUY/HOLD/SELL with confidence percentage (0-100%)
2. **Optimal Entry/Exit Price**: Specific price target (for BUY use "Buy-in Price", for SELL use "Sell Price")
3. **Take Profit %**: Target profit percentage (0-100%)
4. **Stop Loss %**: Risk management percentage (0-100%)
5. **Confidence Breakdown**: Confidence in price target, timing, and overall thesis
6. **Estimated Time to Profit**: Short time window like "3-7 days" or "within this month"

Format your final decision EXACTLY as:
FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** (Confidence: X%)
- Buy-in Price: $X.XX (or "Sell Price: $X.XX" for SELL)
- Take Profit: +X% (for BUY) or -X% (for SELL, target lower price)
- Stop Loss: -X% (for BUY) or +X% (for SELL, if price rises)
- Confidence: Price Target (X%), Timing (X%), Thesis (X%)
- Estimated Time to Profit: <X days/weeks, within this month>

Learn from past mistakes: {past_memory_str}""",
            },
            context,
        ]

        result = llm.invoke(messages)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
