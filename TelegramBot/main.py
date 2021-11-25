import authorsModel
import constants as keys
from telegram.ext import *
import responses as R
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackContext

print('Bot started...')


def start_command(update, context):
    update.message.reply_text('Select game type!')


def help_command(update, context):
    update.message.reply_text('google it!')


def handle_message(update, context):
    text = str(update.message.text).lower()
    response = R.sample_responses(text)

    update.message.reply_text(response)


def styles_command(update, context):
    update.message.reply_text('Game started! mode: styles')


def error(update, context):
    print(f"Update {update} caused error {context.error}")


def main():
    updater = Updater(keys.API_KEY, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start_command))
    dp.add_handler(CommandHandler("help", help_command))
    dp.add_handler(CommandHandler("styles", styles_command))
    dp.add_handler(CommandHandler("authors", authorsModel.authors_game))
    dp.add_handler(CallbackQueryHandler(authorsModel.button))

    dp.add_handler(MessageHandler(Filters.text, handle_message))

    dp.add_error_handler(error)

    updater.start_polling()
    updater.idle()


main()
