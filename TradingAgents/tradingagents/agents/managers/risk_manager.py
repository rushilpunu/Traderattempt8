import time
import json


def create_risk_manager(llm, memory):
    def risk_manager_node(state) -> dict:

        company_name = state["company_of_interest"]

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["news_report"]
        sentiment_report = state["sentiment_report"]
        trader_plan = state["investment_plan"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""As the Risk Management Judge and Debate Facilitator, your goal is to evaluate the debate between three risk analysts and make the final trading decision with comprehensive confidence metrics and risk parameters.

Your FINAL DECISION must include:
1. **Final Recommendation**: BUY/HOLD/SELL with overall confidence percentage (0-100%)
2. **Optimal Entry Price**: Specific price target for entry (if BUY)
3. **Take Profit %**: Target profit percentage (0-100%)
4. **Stop Loss %**: Risk management percentage (0-100%)
5. **Confidence Breakdown**: Confidence in price target, timing, and overall thesis
6. **Risk Assessment**: Specific risk factors and mitigation strategies

Format your final decision as:
FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** (Confidence: X%)
- Entry Price: $X.XX
- Take Profit: +X%
- Stop Loss: -X%
- Confidence: Price Target (X%), Timing (X%), Thesis (X%)
- Risk Level: Low/Medium/High

Guidelines:
1. **Summarize Key Arguments**: Extract strongest points from each analyst
2. **Refine the Trader's Plan**: Start with **{trader_plan}** and adjust based on risk insights
3. **Learn from Past Mistakes**: Use **{past_memory_str}** to avoid previous errors
4. **Be Decisive**: Choose Hold only if strongly justified, not as a fallback

---

**Analysts Debate History:**  
{history}

---

Focus on actionable insights and continuous improvement. Build on past lessons, critically evaluate all perspectives, and ensure each decision advances better outcomes."""

        response = llm.invoke(prompt)

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "risky_history": risk_debate_state["risky_history"],
            "safe_history": risk_debate_state["safe_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_risky_response": risk_debate_state["current_risky_response"],
            "current_safe_response": risk_debate_state["current_safe_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
        }

    return risk_manager_node
