import time
import json


def create_research_manager(llm, memory):
    def research_manager_node(state) -> dict:
        history = state["investment_debate_state"].get("history", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        investment_debate_state = state["investment_debate_state"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""As the portfolio manager and debate facilitator, your role is to critically evaluate this round of debate and make a definitive decision with confidence metrics and trading parameters. Favor intraweek to intramonth horizons; avoid multi-month or multi-year outlooks.

Your decision must include:
1. **Recommendation**: BUY/HOLD/SELL with confidence percentage (0-100%)
2. **Optimal Entry Price**: Specific price target for entry (if BUY)
3. **Take Profit %**: Target profit percentage (0-100%)
4. **Stop Loss %**: Risk management percentage (0-100%)
5. **Confidence Breakdown**: Confidence in price target, timing, and overall thesis
6. **Estimated Time to Profit**: e.g., "3-7 days" or "within this month"

Format your final recommendation as:
FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** (Confidence: X%)
- Entry Price: $X.XX
- Take Profit: +X%
- Stop Loss: -X%
- Confidence: Price Target (X%), Timing (X%), Thesis (X%)
 - Estimated Time to Profit: <X days/weeks, within this month>

Summarize key arguments from both sides, provide rationale, and develop strategic actions. Learn from past mistakes: 

Here are your past reflections on mistakes:
\"{past_memory_str}\"

Here is the debate:
Debate History:
{history}"""
        response = llm.invoke(prompt)

        new_investment_debate_state = {
            "judge_decision": response.content,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": response.content,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": response.content,
        }

    return research_manager_node
