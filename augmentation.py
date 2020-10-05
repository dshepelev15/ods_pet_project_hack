import albumentations as A
import cv2
import os


transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.RGBShift(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.Blur(p=0.5),
    A.ElasticTransform(p=0.5)
])

infolder = '../pepe pix'
outfolder = '../pepe transformed'

for (dirpath, dirnames, filenames) in os.walk(infolder):
    filenames = [name for name in filenames if name.split('.')[1] == 'png']

for filename in filenames:
    image = cv2.imread(f'{infolder}/{filename}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i in range(3):
        transformed = transform(image=image)
        transformed_image = transformed["image"]
        cv2.imwrite(f'../{outfolder}/{filename.split(".")[0]}_{i}.png',
                    cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))
    print(filename)
