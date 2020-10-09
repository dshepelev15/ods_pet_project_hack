import bz2
import os
import sys
import logging

sys.path.insert(0, 'pix2pixHD')

import torch
from telegram.ext import Updater, MessageHandler, Filters, Dispatcher

from handlers import handle_file_upload
from utils.landmarks_detector import LandmarksDetector

from pix2pix import Pix2pixModels

from options.test_options import TestOptions


BOT_TOKEN = os.getenv('BOT_TOKEN', 'your_tokeen')


def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path


def main():

    parser = TestOptions()
    parser.initialize()
    opt = parser.parser.parse_args("")
    model_files = [
        '/home/popov/10_net_G.pth',
        # '/home/popov/10_net_G.pth',
        # '/home/popov/10_net_G.pth',
        # '/home/popov/10_net_G.pth',
        # '/home/popov/10_net_G.pth',
    ]

    Pix2pixModels.initialize(opt, model_files)

    detection_model_file_path = '/home/med1a/shape_predictor_68_face_landmarks.dat.bz2'
    detector_path = unpack_bz2(detection_model_file_path)
    detector = LandmarksDetector(detector_path)
    updater = Updater(BOT_TOKEN, use_context=True)

    dp: Dispatcher = updater.dispatcher
    dp.add_handler(MessageHandler(Filters.photo, handle_file_upload))

    print('Ready for image processing')
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
