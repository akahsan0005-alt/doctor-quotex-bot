import os
from telegram import Bot
from telegram.ext import Updater, CommandHandler

TOKEN = os.environ.get("TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

def start(update, context):
    update.message.reply_text("✅ Bot is running!")

def status(update, context):
    update.message.reply_text("📊 Status: Active")

def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("status", status))

    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    print("Bot started...")
    main()
