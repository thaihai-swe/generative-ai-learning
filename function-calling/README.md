# Function Calling: AI-Powered Stock Analysis System

This project demonstrates advanced function calling with OpenAI models to perform real-time stock analysis, including symbol lookup, price retrieval, and technical trend analysis. It integrates with Yahoo Finance and leverages tools like yfinance, requests, and pydantic for robust data handling.

## Features
- **Stock Symbol Lookup:** Find ticker symbols for companies in specific countries.
- **Current Price Retrieval:** Get the latest stock price and key price data.
- **Trend Analysis:** Analyze historical price trends, moving averages, volatility, momentum, and support/resistance levels.
- **Investment Recommendations:** Receive actionable recommendations based on technical analysis.
- **Interactive Console:** User-friendly CLI for entering company, country, and analysis period.
- **OpenAI Function Calling:** Uses OpenAI's function calling to chain tool calls and generate comprehensive reports.

## Requirements
- Python 3.8+
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [yfinance](https://github.com/ranaroussi/yfinance)
- [python-dotenv](https://github.com/theskumar/python-dotenv)
- [pydantic](https://docs.pydantic.dev/)
- requests
- numpy

Install dependencies:
```bash
pip install -r requirements.txt
```

## Environment Variables
Create a `.env` file in the project directory with the following (or use defaults):
```
OPEN_AI_API_KEY=your_openai_api_key
OPEN_AI_API_BASE_URL=https://api.openai.com/v1
OPEN_AI_MODEL=gpt-4-1106-preview
```

## Usage
Run the main script:
```bash
python function-calling.py
```

Follow the prompts to enter the company name, country, and analysis period. The system will:
1. Find the stock symbol
2. Retrieve the current price
3. Analyze trends and technical indicators
4. Provide a detailed report and investment recommendation

## Example
```
📈 AI-POWERED STOCK ANALYSIS SYSTEM

🔍 Enter company name (e.g., Microsoft, Apple, Tesla): Apple
🌍 Enter country (default: United States):
📊 Select analysis period:
  1mo  - 1 Month (short-term trends)
  3mo  - 3 Months (medium-term) ⭐ Default
  6mo  - 6 Months
  1y   - 1 Year
  2y   - 2 Years (long-term)

Enter period (default: 3mo):

📊 STOCK ANALYSIS REPORT: APPLE
... (detailed analysis output) ...
```

## File Overview
- `function-calling.py`: Main script with all logic for function calling, tool definitions, and CLI.
- `requirements.txt`: Python dependencies.

## Notes
- The script defaults to using a local LLM endpoint if no OpenAI API key is provided.
- For best results, use a model that supports function calling (e.g., GPT-4 Turbo, Llama-3 with function calling support).
- Handles errors and edge cases gracefully with informative logging.

## License
MIT License
