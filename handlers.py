import os
import cv2
import numpy as np

from uuid import uuid4

from PIL import Image
from telegram import Bot

from utils.landmarks_detector import LandmarksDetector
from utils.face_alignment import image_align

from pix2pix import Pix2pixModels


def handle_file_upload(update, context):
    chat_id = update.message.chat.id
    bot: Bot = context.bot

    file_path = f"pictures/{uuid4()}.jpeg"
    with open(file_path, "wb") as fd:
        context.bot.get_file(update.message.photo[-1]).download(out=fd)

    aligned_image_files_path = process_input_image(file_path)

    for aligned_image_file in aligned_image_files_path:

        for output_image in Pix2pixModels.inference(aligned_image_file):
            opencv_output_image = cv2.cvtColor(np.array(output_image), cv2.COLOR_RGB2BGR)
            resized_image = cv2.resize(opencv_output_image, (512, 512))

            output_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            output_image = Image.fromarray(output_image)
            
            webp_file_path = f"pictures/{uuid4()}.webp"
            output_image.save(webp_file_path, "webp")

            with open(webp_file_path, "rb") as fd:
                bot.send_sticker(chat_id, fd)

            os.remove(webp_file_path) # clear
    
        os.remove(aligned_image_file)


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
