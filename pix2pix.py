import numpy as np
import torch

from PIL import Image
from models.networks import GlobalGenerator, get_norm_layer


class Pix2pixModels:
    models = []

    @classmethod
    def initialize(cls, opt, files):
        for file in files:
            model = GlobalGenerator(
                opt.input_nc,
                opt.output_nc,
                opt.ngf,
                opt.n_downsample_global,
                opt.n_blocks_global,
                get_norm_layer()
            )
            state = torch.load(file)
            model.load_state_dict(state)
            cls.models.append(model)

    @classmethod
    def inference(cls, input_image):
        image = Image.open(input_image).convert("RGB")
        image = np.array(image) / 255. * 2 - 1
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2,0,1)[None]

        for model in cls.models:
            with torch.no_grad():
                output = model(image_tensor)

                output_image = Image.fromarray(
                    np.uint8((output[0].permute(1,2,0).detach().data.numpy() + 1) / 2 * 255)
                )
                yield output_image
