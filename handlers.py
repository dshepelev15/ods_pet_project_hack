import os

from uuid import uuid4

from PIL import Image
from telegram import Bot

from utils.landmarks_detector import LandmarksDetector
from utils.face_alignment import image_align
from UGATIT import get_initialized_model


def handle_file_upload(update, context):
    chat_id = update.message.chat.id
    bot: Bot = context.bot

    file_path = f"pictures/{uuid4()}.jpeg"
    with open(file_path, "wb") as fd:
        context.bot.get_file(update.message.photo[-1]).download(out=fd)

    aligned_image_files_path = process_input_image(file_path)

    gan_model = get_initialized_model()
    for aligned_image_file in aligned_image_files_path:
        output_image_path = gan_model.eval(aligned_image_file)

        im = Image.open(output_image_path).convert("RGB")
        webp_file_path = file_path.replace(".jpeg", ".webp")
        im.save(webp_file_path, "webp")

        with open(webp_file_path, "rb") as fd:
            bot.send_sticker(chat_id, fd)

        os.remove(webp_file_path) # clear


def process_input_image(input_file_path):
    landmarks_detector = LandmarksDetector()
    input_image = Image.open(input_file_path)

    output_files = []
    for face_landmarks in landmarks_detector.get_landmarks(input_file_path):
        output_image = image_align(input_image, face_landmarks, 256)
        output_file_path = f'pictures/{uuid4()}.jpeg'
        output_image.save(output_file_path, 'jpeg')
        output_files.append(output_file_path)

    return output_files
