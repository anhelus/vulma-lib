from pathlib import Path
from PIL import Image
from glob import glob

DATA_PATH = Path('./dataset')

images = glob(f'{str(DATA_PATH)}/**/*.jpg')

for image in images:
    name = image.split('.')[:-1][0]
    img = Image.open(image)
    rgb_img = img.convert('RGB')
    rgb_img.save(f'{name}.jpeg')
