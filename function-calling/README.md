# Function Calling Stock Analysis

This project demonstrates advanced function calling with OpenAI's API for stock analysis and investment recommendations. It integrates multiple tools to fetch stock symbols, prices, and analyze trends using Yahoo Finance and yfinance.

## Features
- Find stock ticker symbols for companies in specific countries
- Retrieve current stock prices
- Analyze historical price trends, moving averages, volatility, momentum, and support/resistance
- Provide actionable investment recommendations based on technical analysis
- Modular tool schema for easy extension

## How It Works
1. **Find Stock Symbol:** Uses Yahoo Finance search API to get the ticker symbol for a company and country.
2. **Get Stock Price:** Fetches current price and related data using yfinance.
3. **Analyze Trends:** Calculates moving averages, volatility, momentum, and generates recommendations.
4. **Function Calling Loop:** The LLM can call these tools in sequence, returning a comprehensive analysis.

## Usage
Run the script to analyze a stock (example: Microsoft in the United States):

```bash
python function-calling/function-calling.py
```

## Requirements
- Python 3.8+
- [openai](https://pypi.org/project/openai/)
- [yfinance](https://pypi.org/project/yfinance/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- [pydantic](https://pypi.org/project/pydantic/)
- [numpy](https://pypi.org/project/numpy/)
- [requests](https://pypi.org/project/requests/)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables
Create a `.env` file in the project root with the following variables:

```
OPEN_AI_API_KEY=your_api_key
OPEN_AI_API_BASE_URL=https://api.openai.com/v1
OPEN_AI_MODEL=gpt-4
```

## Example Output
The script prints a comprehensive stock analysis report including:
- Stock symbol
- Current price
- Trend direction and strength
- Volatility and momentum
- Support/resistance levels
- Investment recommendation

## License
MIT License

## Author
Your Name
