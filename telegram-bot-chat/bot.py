import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from openai import OpenAI

load_dotenv()

TOKEN = os.getenv("BOT_TOKEN")
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
OPEN_AI_MODEL = os.getenv("OPEN_AI_MODEL", "meta-llama-3.1-8b-instruct")
OPEN_AI_API_BASE_URL = os.getenv("OPEN_AI_API_BASE_URL", "http://127.0.0.1:1234/v1")
client = OpenAI(base_url=OPEN_AI_API_BASE_URL, api_key=OPEN_AI_API_KEY)

# Multi-user support: Store chat history and settings per user
user_sessions = {}

# Predefined personas
PERSONAS = {
    "assistant": "You are a helpful assistant.",
    "coder": "You are an expert software engineer who provides clear, concise code solutions with explanations.",
    "teacher": "You are a patient teacher who explains concepts in simple terms with examples.",
    "creative": "You are a creative writer who crafts engaging and imaginative content.",
    "analyst": "You are a data analyst who provides insightful analysis and clear explanations of complex topics."
}

def get_user_session(user_id):
    """Get or create user session"""
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "chat_history": [],
            "system_prompt": PERSONAS["assistant"],
            "persona": "assistant"
        }
    return user_sessions[user_id]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command"""
    user_id = update.effective_user.id
    get_user_session(user_id)  # Initialize session

    welcome_message = (
        "👋 Hello! I'm your AI assistant powered by LLaMA 3.\n\n"
        "📋 Available commands:\n"
        "/start - Start the bot\n"
        "/clear - Clear conversation history\n"
        "/history - Show conversation summary\n"
        "/reset - Start a new conversation\n"
        "/persona <name> - Switch persona (assistant, coder, teacher, creative, analyst)\n"
        "/systemprompt <text> - Set custom system prompt\n"
        "/help - Show this help message\n\n"
        "Just send me a message to chat!"
    )
    await context.bot.send_message(chat_id=update.effective_chat.id, text=welcome_message)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command"""
    await start(update, context)


async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /clear command - clear conversation history"""
    user_id = update.effective_user.id
    session = get_user_session(user_id)
    session["chat_history"] = []

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="✅ Conversation history cleared! Starting fresh."
    )


async def show_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /history command - show conversation summary"""
    user_id = update.effective_user.id
    session = get_user_session(user_id)
    chat_history = session["chat_history"]

    if not chat_history:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="📭 No conversation history yet. Start chatting!"
        )
        return

    summary = f"📊 Conversation Summary\n\n"
    summary += f"Total messages: {len(chat_history)}\n"
    summary += f"Current persona: {session['persona']}\n\n"
    summary += "Recent messages:\n"

    # Show last 5 exchanges
    recent = chat_history[-5:]
    for i, (user_msg, assistant_msg) in enumerate(recent, 1):
        summary += f"\n{i}. You: {user_msg[:50]}{'...' if len(user_msg) > 50 else ''}\n"
        summary += f"   Bot: {assistant_msg[:50]}{'...' if len(assistant_msg) > 50 else ''}\n"

    await context.bot.send_message(chat_id=update.effective_chat.id, text=summary)


async def reset_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /reset command - reset entire conversation"""
    user_id = update.effective_user.id
    session = get_user_session(user_id)
    session["chat_history"] = []
    session["system_prompt"] = PERSONAS["assistant"]
    session["persona"] = "assistant"

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="🔄 Conversation reset! System prompt restored to default assistant."
    )


async def set_persona(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /persona command - switch between predefined personas"""
    user_id = update.effective_user.id
    session = get_user_session(user_id)

    if not context.args:
        personas_list = ", ".join(PERSONAS.keys())
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"🎭 Available personas: {personas_list}\n\nUsage: /persona <name>"
        )
        return

    persona_name = context.args[0].lower()

    if persona_name not in PERSONAS:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"❌ Unknown persona. Available: {', '.join(PERSONAS.keys())}"
        )
        return

    session["persona"] = persona_name
    session["system_prompt"] = PERSONAS[persona_name]

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"✅ Persona switched to: {persona_name}\n\n{PERSONAS[persona_name]}"
    )


async def set_system_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /systemprompt command - set custom system prompt"""
    user_id = update.effective_user.id
    session = get_user_session(user_id)

    if not context.args:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="📝 Usage: /systemprompt <your custom prompt>\n\nExample: /systemprompt You are a pirate who speaks in pirate language"
        )
        return

    custom_prompt = " ".join(context.args)
    session["system_prompt"] = custom_prompt
    session["persona"] = "custom"

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"✅ Custom system prompt set:\n\n{custom_prompt}"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle user messages with streaming response"""
    user_id = update.effective_user.id
    session = get_user_session(user_id)

    # Build messages array with system prompt and history
    messages = [{"role": "system", "content": session["system_prompt"]}]

    for user_msg, assistant_msg in session["chat_history"]:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    user_message = update.message.text
    messages.append({"role": "user", "content": user_message})

    try:
        # Send typing indicator
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        # Stream the response
        stream = client.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=messages,
            stream=True
        )

        # Collect streamed response
        full_response = ""
        message_obj = None
        chunk_count = 0

        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                chunk_count += 1

                # Update message every 10 chunks or when we have substantial content
                if chunk_count % 10 == 0 or len(full_response) > 50:
                    if message_obj is None:
                        # Send initial message
                        message_obj = await context.bot.send_message(
                            chat_id=update.effective_chat.id,
                            text=full_response + "..."
                        )
                    else:
                        # Edit existing message
                        try:
                            await message_obj.edit_text(full_response + "...")
                        except Exception:
                            # Ignore errors if content hasn't changed
                            pass

        # Send or update final message
        if message_obj is None:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=full_response
            )
        else:
            await message_obj.edit_text(full_response)

        # Save to history
        session["chat_history"].append((user_message, full_response))

    except Exception as e:
        error_message = f"❌ Error: {str(e)}\n\nPlease try again or contact the administrator."
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=error_message
        )
        print(f"Error processing message for user {user_id}: {e}")


def main():
    """Start the bot"""
    application = ApplicationBuilder().token(TOKEN).build()

    # Command handlers
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(CommandHandler('clear', clear_history))
    application.add_handler(CommandHandler('history', show_history))
    application.add_handler(CommandHandler('reset', reset_conversation))
    application.add_handler(CommandHandler('persona', set_persona))
    application.add_handler(CommandHandler('systemprompt', set_system_prompt))

    # Message handler
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 Bot is running...")
    application.run_polling()


if __name__ == "__main__":
    main()
