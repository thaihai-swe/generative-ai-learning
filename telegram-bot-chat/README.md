# Telegram Bot Chat

An advanced Telegram chatbot powered by LLM (OpenAI-compatible API) with multi-user support, conversation management, and streaming responses.

## Features
- 🤖 **Multi-User Support**: Each user has their own conversation context and settings
- 💬 **Streaming Responses**: Real-time message updates as the AI generates responses
- 🎭 **Custom Personas**: Switch between predefined personas (assistant, coder, teacher, creative, analyst)
- 📝 **Custom System Prompts**: Set your own system prompts for personalized behavior
- 📊 **Conversation Management**: Clear, view, and reset conversation history
- 🔄 **Context-Aware**: Maintains chat history for contextual conversations
- ⚡ **Easy Configuration**: Simple environment variable setup

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

### Available Commands

- `/start` or `/help` - Show welcome message and available commands
- `/clear` - Clear your conversation history
- `/history` - Show conversation summary with recent messages
- `/reset` - Reset conversation and restore default system prompt
- `/persona <name>` - Switch between predefined personas:
  - `assistant` - Helpful general assistant (default)
  - `coder` - Expert software engineer
  - `teacher` - Patient educator
  - `creative` - Creative writer
  - `analyst` - Data analyst
- `/systemprompt <text>` - Set a custom system prompt

### Example Usage

```
/persona coder
> Switched to coder persona

How do I implement a binary search in Python?
> [AI provides code-focused response]

/systemprompt You are a pirate who speaks in pirate language
> Custom system prompt set

Hello!
> Ahoy there, matey! How can this old sea dog help ye today?
```

### Chat Features

- **Streaming**: Messages appear in real-time as the AI generates them
- **Per-User Context**: Your conversations are private and separate from other users
- **Persistent History**: Your conversation history is maintained throughout the session

## Notes
- Requires a running OpenAI-compatible API endpoint.
- Make sure your bot token and API keys are kept secret.

## License
MIT
