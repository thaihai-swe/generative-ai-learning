# Telegram Bot Chat

A simple Telegram chatbot using OpenAI's API and python-telegram-bot.

## Features
- Responds to user messages using an LLM (OpenAI-compatible API)
- Maintains chat history for context
- Easy to configure with environment variables

## Setup

1. **Clone the repository** and navigate to the `telegram-bot-chat` folder:
   ```bash
   git clone <repo-url>
   cd telegram-bot-chat
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file** in this folder with the following variables:
   ```env
   BOT_TOKEN=your_telegram_bot_token
   OPEN_AI_API_KEY=your_openai_api_key
   OPEN_AI_MODEL=meta-llama-3.1-8b-instruct  # or your preferred model
   OPEN_AI_API_BASE_URL=http://127.0.0.1:1234/v1  # or your OpenAI-compatible endpoint
   ```

5. **Run the bot:**
   ```bash
   python bot.py
   ```

## Usage
- Start a chat with your bot on Telegram.
- Send messages and receive AI-generated responses.

## Notes
- Requires a running OpenAI-compatible API endpoint.
- Make sure your bot token and API keys are kept secret.

## License
MIT
