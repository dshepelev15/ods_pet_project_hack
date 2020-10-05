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

folder = '../pepe pix'

for (dirpath, dirnames, filenames) in os.walk(folder):
    filenames = [name for name in filenames if name.split('.')[1] == 'png']
    print(filenames)
    print(len(filenames))

for filename in filenames:
    image = cv2.imread(f'{folder}/{filename}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i in range(3):
        transformed = transform(image=image)
        transformed_image = transformed["image"]
        cv2.imwrite(f'../pepe transformed/{filename.split(".")[0]}_{i}.png',
                    cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))

    # cv2.imshow("Original", image)
    # cv2.imshow("Duplicate", transformed_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()                         