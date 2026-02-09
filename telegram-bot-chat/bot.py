import os

from httpx import stream
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, Filters,ContextTypes
from openai import OpenAI

load_dotenv()

TOEKEN = os.getenv("BOT_TOKEN")
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "meta-llama-3.1-8b-instruct")

client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key=OPEN_AI_API_KEY)

chat_hitory = []

application = ApplicationBuilder().token(TOEKEN).build()



async def start(update: Update, context: ContextTypes) -> None:
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Hello! I'm your bot.")



async def echo(update: Update, context: ContextTypes) -> None:
    global chat_hitory
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        }
    ]
    for item in chat_hitory:
        messages.append({"role": 'user', "content": item[0]})
        messages.append({"role": 'assistant', "content": item[1]})

    user_message = update.message.text

    messages.append({"role": 'user', "content": user_message})

    response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages
        )
    assistant_message = response.choices[0].message.content
    chat_hitory.append((user_message, assistant_message))
    await context.bot.send_message(chat_id=update.effective_chat.id, text=assistant_message)


start_handler = CommandHandler('start', start)
application.add_handler(start_handler)

echo_handler = MessageHandler(Filters.text, echo)
application.add_handler(echo_handler)

application.run_polling()
