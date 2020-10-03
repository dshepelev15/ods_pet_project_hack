import os

from telegram.ext import Updater, MessageHandler, Filters, Dispatcher

from handlers import handle_file_upload

BOT_TOKEN = os.getenv('BOT_TOKEN', 'your_tokeen')


def main():
    # bot = Bot(token=BOT_TOKEN)
    updater = Updater(BOT_TOKEN, use_context=True)

    dp: Dispatcher = updater.dispatcher
    dp.add_handler(MessageHandler(Filters.photo, handle_file_upload))

    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
