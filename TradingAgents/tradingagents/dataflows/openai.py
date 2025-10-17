from openai import OpenAI
from .config import get_config


def get_stock_news_openai(query, start_date, end_date):
    config = get_config()
    client = OpenAI(base_url=config["backend_url"])

    response = client.responses.create(
        model=config["quick_think_llm"],
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Can you search Social Media for {query} from {start_date} to {end_date}? Make sure you only get the data posted during that period.",
                    }
                ],
            }
        ],
        text={"format": {"type": "text"}},
        reasoning={},
        tools=[
            {
                "type": "web_search_preview",
                "user_location": {"type": "approximate"},
                "search_context_size": "low",
            }
        ],
        temperature=1,
        max_output_tokens=4096,
        top_p=1,
        store=True,
    )

    return response.output[1].content[0].text


def get_global_news_openai(curr_date, look_back_days=7, limit=5):
    import time
    
    config = get_config()
    client = OpenAI(base_url=config["backend_url"])

    max_retries = 2
    base_delay = 10  # shorter base to avoid long stalls
    
    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=config["quick_think_llm"],
                input=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "input_text",
                                "text": f"Can you search global or macroeconomics news from {look_back_days} days before {curr_date} to {curr_date} that would be informative for trading purposes? Make sure you only get the data posted during that period. Limit the results to {limit} articles.",
                            }
                        ],
                    }
                ],
                text={"format": {"type": "text"}},
                reasoning={},
                tools=[
                    {
                        "type": "web_search_preview",
                        "user_location": {"type": "approximate"},
                        "search_context_size": "low",
                    }
                ],
                temperature=1,
                max_output_tokens=4096,
                top_p=1,
                store=True,
            )

            # Prefer the convenience property when available
            try:
                if hasattr(response, "output_text") and response.output_text:
                    return response.output_text
            except Exception:
                pass

            # Fallback: attempt to collect any text-like payloads
            try:
                texts = []
                for item in getattr(response, "output", []) or []:
                    # SDK objects sometimes expose .content as a list of parts
                    content = getattr(item, "content", None)
                    if isinstance(content, list):
                        for part in content:
                            # dict-style
                            if isinstance(part, dict) and (part.get("type") in {"output_text", "text"}):
                                if "text" in part and part["text"]:
                                    texts.append(part["text"])
                            else:
                                # object-style
                                txt = getattr(part, "text", None)
                                if txt:
                                    texts.append(txt)
                if texts:
                    return "\n".join(texts)
            except Exception:
                pass

            # Last resort: stringify the response object
            return str(response)
            
        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (attempt + 1)
                print(f"OpenAI API error: {e}. Waiting {delay} seconds before retry {attempt + 1}/{max_retries}")
                time.sleep(delay)
                continue
            else:
                raise


def get_fundamentals_openai(ticker, curr_date):
    config = get_config()
    client = OpenAI(base_url=config["backend_url"])

    response = client.responses.create(
        model=config["quick_think_llm"],
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Can you search Fundamental for discussions on {ticker} during of the month before {curr_date} to the month of {curr_date}. Make sure you only get the data posted during that period. List as a table, with PE/PS/Cash flow/ etc",
                    }
                ],
            }
        ],
        text={"format": {"type": "text"}},
        reasoning={},
        tools=[
            {
                "type": "web_search_preview",
                "user_location": {"type": "approximate"},
                "search_context_size": "low",
            }
        ],
        temperature=1,
        max_output_tokens=4096,
        top_p=1,
        store=True,
    )

    return response.output[1].content[0].text