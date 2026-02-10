from openai import OpenAI
from dotenv import load_dotenv
import os
import json
load_dotenv()

TOKEN = os.getenv("BOT_TOKEN")
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
OPEN_AI_MODEL = os.getenv("OPEN_AI_MODEL", "meta-llama-3.1-8b-instruct")
OPEN_AI_API_BASE_URL = os.getenv("OPEN_AI_API_BASE_URL", "http://127.0.0.1:1234/v1")
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

tools =[
    {
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
    },
    {
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
    },
    {
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
]


messages = [
    {"role": "system", "content": "You are a helpful assistant.Answer use funny tone."},
    {"role": "user", "content": "What's the weather like in New York City?"},
]

response = client.chat.completions.create(
    model=OPEN_AI_MODEL,
    messages=messages,
    tools=tools,
    tool_selection="auto"
)

tool_call = response.choices[0].message.tool_call[0]
arguments = json.loads(tool_call.function.arguments)

messages.append({"role": "assistant", "content": None, "tool_call": tool_call})

if tool_call.name == "get_current_weather":
    tool_response = get_current_weather(
        location=arguments.get("location"),
        unit=arguments.get("unit", "celsius")
    )
    messages.append({"role": "function", "name": tool_call.name, "tool_call_id": tool_call.id, "content": str(tool_response)})
elif tool_call.name == "get_stock_price":
    tool_response = get_stock_price(
        ticker=arguments.get("ticker")
    )
elif tool_call.name == "view_website":
    tool_response = view_website(
        url=arguments.get("url")
    )
else:
    tool_response = {"error": "Unknown tool"}
