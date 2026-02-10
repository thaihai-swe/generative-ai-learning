from openai import OpenAI
from dotenv import load_dotenv
import os
import json
load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY", "lm-studio")
OPEN_AI_API_BASE_URL = os.getenv("OPEN_AI_API_BASE_URL", "http://127.0.0.1:1234/v1")
OPEN_AI_MODEL = os.getenv("OPEN_AI_MODEL", "meta-llama-3.1-8b-instruct")
client = OpenAI(base_url=OPEN_AI_API_BASE_URL, api_key=OPEN_AI_API_KEY)


def get_current_weather(location: str,unit: str = "celsius") -> dict:
    """Get the current weather for a given location."""
    # Dummy implementation for illustration
    return {
        "location": location,
        "temperature": "22",
        "unit": unit,
        "condition": "Sunny"
    }

def get_stock_price(ticker: str) -> dict:
    """Get the current stock price for a given ticker symbol."""
    # Dummy implementation for illustration
    return {
        "ticker": ticker,
        "price": "150.00",
        "currency": "USD"
    }

def view_website(url: str) -> dict:
    """Fetch and summarize the content of a website."""
    # Dummy implementation for illustration
    return {
        "url": url,
        "summary": "This is a summary of the website content."
    }

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g., San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a given ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol, e.g., AAPL for Apple Inc."
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "view_website",
            "description": "Fetch and summarize the content of a website.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the website to view."
                    }
                },
                "required": ["url"]
            }
        }
    }
]


messages = [
    {"role": "system", "content": "You are a helpful assistant.Answer use funny tone."},
    {"role": "user", "content": "What's the weather like in New York City?"},
]

response = client.chat.completions.create(
    model=OPEN_AI_MODEL,  # Update to your new model
    messages=messages,
    tools=tools
)

# Extract tool call
tool_calls = getattr(response.choices[0].message, "tool_calls", None)
if tool_calls and len(tool_calls) > 0:
    tool_call = tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)

    if tool_call.function.name == "get_current_weather":
        tool_response = get_current_weather(
            location=arguments.get("location"),
            unit=arguments.get("unit", "celsius")
        )
    elif tool_call.function.name == "get_stock_price":
        tool_response = get_stock_price(
            ticker=arguments.get("ticker")
        )
    elif tool_call.function.name == "view_website":
        tool_response = view_website(
            url=arguments.get("url")
        )
    else:
        tool_response = {"error": "Unknown tool"}


    messages.append({"role": "tool", "name": tool_call.function.name, "content": json.dumps(tool_response)})

    final_response = client.chat.completions.create(
        model=OPEN_AI_MODEL,
        messages=messages
    )

    print(tool_response)

    print(final_response.choices[0].message.content)
else:
    print({"error": "No tool call returned by model"})
