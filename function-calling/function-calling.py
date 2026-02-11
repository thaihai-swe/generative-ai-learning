import os
import json
import logging
from typing import Any, Callable, Dict, List, Optional
from pprint import pprint
from datetime import datetime, timedelta

from openai import OpenAI
from dotenv import load_dotenv
import yfinance as yf
import requests
import inspect
from pydantic import TypeAdapter
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY", "lm-studio")
OPEN_AI_API_BASE_URL = os.getenv("OPEN_AI_API_BASE_URL", "http://127.0.0.1:1234/v1")
OPEN_AI_MODEL = os.getenv("OPEN_AI_MODEL", "meta-llama-3.1-8b-instruct")
MAX_TOOL_CALLS = 10  # Prevent infinite loops

client = OpenAI(base_url=OPEN_AI_API_BASE_URL, api_key=OPEN_AI_API_KEY)





def get_stock_symbol(company_name: str, country: str) -> Dict[str, Any]:
    """
    Get the stock ticker symbol for a given company name and country.

    Args:
        company_name: The name of the company to search for.
        country: The country where the company is located.

    Returns:
        The stock ticker symbol as a string if found, otherwise an error dict.
    """
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        params = {"q": company_name, "country": country}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()
        if "quotes" in data and len(data["quotes"]) > 0:
            symbol = data["quotes"][0]["symbol"]
            logger.info(f"Found symbol for {company_name}: {symbol}")
            return symbol

        logger.warning(f"No quotes found for {company_name}")
        return {"error": f"No stock symbol found for {company_name}"}

    except requests.RequestException as e:
        logger.error(f"Error fetching stock symbol: {e}")
        return {"error": f"Failed to fetch stock symbol: {str(e)}"}

def get_stock_price(ticker: str) -> Dict[str, Any]:
    """
    Get the current stock price for a given ticker symbol.

    Args:
        ticker: The stock ticker symbol (e.g., AAPL, MSFT).

    Returns:
        Dictionary with stock price data including close, open, high, low, and currency.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")

        if hist.empty:
            logger.warning(f"No historical data found for {ticker}")
            return {"error": f"No data available for ticker {ticker}"}

        latest_price = hist.iloc[-1]
        logger.info(f"Retrieved price for {ticker}: ${latest_price['Close']:.2f}")

        return {
            "ticker": ticker,
            "time": latest_price.name.strftime("%Y-%m-%d %H:%M:%S"),
            "close": float(latest_price["Close"]),
            "open": float(latest_price["Open"]),
            "high": float(latest_price["High"]),
            "low": float(latest_price["Low"]),
            "currency": stock.info.get("currency", "USD"),
        }

    except Exception as e:
        logger.error(f"Error fetching stock price for {ticker}: {e}")
        return {"error": f"Failed to fetch stock price: {str(e)}"}


def get_stock_trends(ticker: str, period: str = "3mo") -> Dict[str, Any]:
    """
    Analyze historical price trends and technical indicators for a stock.

    Calculates moving averages, trend direction, volatility, and support/resistance levels
    to provide comprehensive trend analysis over a specified period.

    Args:
        ticker: The stock ticker symbol (e.g., AAPL, MSFT).
        period: Analysis period - '1mo', '3mo', '6mo', '1y', '2y' (default: '3mo').

    Returns:
        Dictionary containing trend analysis with moving averages, trend direction,
        volatility, price momentum, and recommendations.
    """
    try:
        validate_period = {"1mo", "3mo", "6mo", "1y", "2y"}
        if period not in validate_period:
            return {"error": f"Invalid period. Use: {', '.join(validate_period)}"}

        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)

        if hist.empty or len(hist) < 20:
            logger.warning(f"Insufficient data for {ticker}")
            return {"error": f"Insufficient historical data for ticker {ticker}"}

        closes = hist["Close"].values
        dates = hist.index

        # Calculate moving averages
        ma_20 = float(np.mean(closes[-20:]))
        ma_50 = float(np.mean(closes[-50:])) if len(closes) >= 50 else float(np.mean(closes))
        ma_200 = float(np.mean(closes[-200:])) if len(closes) >= 200 else float(np.mean(closes))

        current_price = float(closes[-1])
        previous_price = float(closes[-2]) if len(closes) > 1 else current_price
        price_change = current_price - previous_price
        price_change_pct = (price_change / previous_price * 100) if previous_price != 0 else 0

        # Calculate volatility (standard deviation of returns)
        returns = np.diff(closes) / closes[:-1]
        volatility = float(np.std(returns) * 100)  # As percentage

        # Determine trend direction
        if current_price > ma_50 > ma_200:
            trend = "UPTREND"
            trend_strength = "Strong"
        elif current_price > ma_50:
            trend = "UPTREND"
            trend_strength = "Moderate"
        elif current_price < ma_50 < ma_200:
            trend = "DOWNTREND"
            trend_strength = "Strong"
        elif current_price < ma_50:
            trend = "DOWNTREND"
            trend_strength = "Moderate"
        else:
            trend = "SIDEWAYS"
            trend_strength = "Neutral"

        # Calculate support and resistance
        period_high = float(np.max(closes[-50:]))
        period_low = float(np.min(closes[-50:]))
        period_range = period_high - period_low

        # Momentum (ROC - Rate of Change)
        momentum = float((closes[-1] - closes[0]) / closes[0] * 100) if len(closes) > 0 else 0

        # Price position in range (0-100)
        price_position = float(((current_price - period_low) / period_range * 100) if period_range != 0 else 50)

        # Generate recommendation
        if trend == "UPTREND" and price_position < 70:
            recommendation = "BUY - Good entry point in uptrend"
        elif trend == "UPTREND" and price_position >= 70:
            recommendation = "HOLD/TAKE PROFIT - Near resistance"
        elif trend == "DOWNTREND" and price_position > 30:
            recommendation = "SELL/AVOID - Downtrend continues"
        elif trend == "DOWNTREND" and price_position <= 30:
            recommendation = "POTENTIAL BUY - Near support"
        else:
            recommendation = "HOLD - Sideways movement, await breakout"

        logger.info(f"Trend analysis complete for {ticker}")

        return {
            "ticker": ticker,
            "period": period,
            "timestamp": datetime.now().isoformat(),
            "current_price": current_price,
            "price_change": round(price_change, 2),
            "price_change_pct": round(price_change_pct, 2),
            "moving_averages": {
                "ma_20": round(ma_20, 2),
                "ma_50": round(ma_50, 2),
                "ma_200": round(ma_200, 2),
            },
            "trend": {
                "direction": trend,
                "strength": trend_strength,
                "momentum_pct": round(momentum, 2),
            },
            "volatility_pct": round(volatility, 2),
            "support_resistance": {
                "resistance": round(period_high, 2),
                "support": round(period_low, 2),
                "price_position_in_range": round(price_position, 2),
            },
            "recommendation": recommendation,
            "data_points": len(closes),
        }

    except Exception as e:
        logger.error(f"Error analyzing trends for {ticker}: {e}")
        return {"error": f"Failed to analyze trends: {str(e)}"}


def build_tool_schema(func: Callable) -> Dict[str, Any]:
    """
    Build a tool schema from a function's type hints and docstring.

    Args:
        func: The function to generate schema for.

    Returns:
        Tool definition compatible with OpenAI API.
    """
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": inspect.getdoc(func),
            "parameters": TypeAdapter(func).json_schema(),
        },
    }


# Tool definitions
TOOLS = [
    build_tool_schema(get_stock_symbol),
    build_tool_schema(get_stock_price),
    build_tool_schema(get_stock_trends),
]

FUNCTION_MAP: Dict[str, Callable] = {
    "get_stock_symbol": get_stock_symbol,
    "get_stock_price": get_stock_price,
    "get_stock_trends": get_stock_trends,
}

def get_completion(messages: List[Dict[str, Any]], tools: List[Dict]) -> Any:
    """
    Get a completion from the LLM with tool support.

    Args:
        messages: Conversation history.
        tools: Available tools/functions.

    Returns:
        API response object.
    """
    return client.chat.completions.create(
        model=OPEN_AI_MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )


def execute_tool_call(tool_call: Any, tool_arguments: Dict[str, Any]) -> str:
    """
    Execute a tool call and return the result as a JSON string.

    Args:
        tool_call: The tool call object from the API.
        tool_arguments: Parsed arguments for the tool.

    Returns:
        JSON string of the tool result.
    """
    tool_name = tool_call.function.name
    tool_function = FUNCTION_MAP.get(tool_name)

    if not tool_function:
        logger.error(f"Unknown tool: {tool_name}")
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    try:
        result = tool_function(**tool_arguments)
        logger.info(f"Successfully executed {tool_name}")
        return json.dumps(result)
    except TypeError as e:
        logger.error(f"Invalid arguments for {tool_name}: {e}")
        return json.dumps({"error": f"Invalid arguments: {str(e)}"})
    except Exception as e:
        logger.error(f"Error executing {tool_name}: {e}")
        return json.dumps({"error": f"Execution error: {str(e)}"})


def process_tool_calls(
    messages: List[Dict[str, Any]],
    tools: List[Dict],
) -> str:
    """
    Process tool calls in a loop until the model stops requesting tools.

    Args:
        messages: Conversation history (will be modified in place).
        tools: Available tools.

    Returns:
        Final response text from the model.
    """
    tool_call_count = 0

    while tool_call_count < MAX_TOOL_CALLS:
        response = get_completion(messages, tools)
        choice = response.choices[0]
        finish_reason = choice.finish_reason

        logger.info(f"Finish reason: {finish_reason}")

        # If model is done, return the response
        if finish_reason == "stop":
            return choice.message.content

        # If tool calls needed, process them
        if finish_reason == "tool_calls":
            tool_call_count += 1
            tool_calls = choice.message.tool_calls

            if not tool_calls:
                logger.warning("No tool calls found despite tool_calls finish reason")
                return choice.message.content

            # Add assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": choice.message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            })

            # Execute each tool call
            for tool_call in tool_calls:
                tool_arguments = json.loads(tool_call.function.arguments)
                tool_result = execute_tool_call(tool_call, tool_arguments)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": tool_result,
                })

                logger.info(f"Tool result: {tool_result}")
        else:
            logger.warning(f"Unexpected finish reason: {finish_reason}")
            return choice.message.content

    logger.error(f"Maximum tool calls ({MAX_TOOL_CALLS}) reached")
    return "Error: Maximum tool calls exceeded"


def main():
    """
    Main function to demonstrate function calling with stock analysis.
    Interactive mode - user enters company name, country, and analysis period.
    """
    print("\n" + "="*80)
    print("📈 AI-POWERED STOCK ANALYSIS SYSTEM")
    print("="*80)

    # Prompt user for company name
    company_name = input("\n🔍 Enter company name (e.g., Microsoft, Apple, Tesla): ").strip()
    if not company_name:
        print("❌ Company name cannot be empty")
        return None

    # Prompt user for country
    country = input("🌍 Enter country (default: United States): ").strip() or "United States"

    # Prompt user for analysis period
    print("\n📊 Select analysis period:")
    print("  1mo  - 1 Month (short-term trends)")
    print("  3mo  - 3 Months (medium-term) ⭐ Default")
    print("  6mo  - 6 Months")
    print("  1y   - 1 Year")
    print("  2y   - 2 Years (long-term)")

    period = input("\nEnter period (default: 3mo): ").strip() or "3mo"

    # Validate period
    valid_periods = {"1mo", "3mo", "6mo", "1y", "2y"}
    if period not in valid_periods:
        print(f"❌ Invalid period. Use one of: {', '.join(valid_periods)}")
        return None

    # Build the analysis query
    analysis_query = f"""Analyze {company_name} stock in {country}.
I want to understand:
1. Current stock price
2. Price trends over the last {period}
3. Technical indicators (moving averages, volatility, momentum)
4. Support and resistance levels
5. An investment recommendation based on technical analysis"""

    messages = [
        {
            "role": "system",
            "content": """You are an expert financial advisor AI assistant specializing in stock analysis and investment recommendations.
When analyzing stocks:
1. First, find the stock symbol using get_stock_symbol
2. Get the current price using get_stock_price
3. Analyze trends for the requested period using get_stock_trends
4. Provide comprehensive analysis with trend direction, momentum, support/resistance levels
5. Give actionable investment recommendations based on technical analysis
6. Format your response clearly with sections for price, trend, and recommendation""",
        },
        {"role": "user", "content": analysis_query},
    ]

    logger.info(f"Starting comprehensive stock analysis for {company_name} in {country}...")
    bot_response = process_tool_calls(messages, TOOLS)

    print("\n" + "="*80)
    print(f"📊 STOCK ANALYSIS REPORT: {company_name.upper()}")
    print("="*80)
    print(bot_response)
    print("="*80 + "\n")

    return bot_response


if __name__ == "__main__":
    main()
