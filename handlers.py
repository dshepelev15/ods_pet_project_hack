from uuid import uuid4

from PIL import Image
from telegram import Bot


def handle_file_upload(update, context):
    chat_id = update.message.chat.id
    bot: Bot = context.bot

    file_path = f"pictures/{uuid4()}.jpeg"
    with open(file_path, "wb") as fd:
        context.bot.get_file(update.message.photo[-1]).download(out=fd)

    im = Image.open(file_path).convert("RGB")
    webp_file_path = file_path.replace(".jpeg", ".webp")
    im.save(webp_file_path, "webp")

    with open(webp_file_path, "rb") as fd:
        bot.send_sticker(chat_id, fd)
