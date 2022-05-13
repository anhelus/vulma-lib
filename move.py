from glob import glob
from shutil import move
from pathlib import Path
import os


img_folder = Path('./images')
annot_folder = Path('./annots')

if not img_folder.exists():
    img_folder.mkdir(parents=True, exist_ok=True)

if not annot_folder.exists():
    annot_folder.mkdir(parents=True, exist_ok=True)

images = glob('./bridges/**/*.jpg', recursive=True)

counter = 1
for image in images:
    path = '\\'.join(image.split('\\')[:-1])
    img_file = image.split('\\')[-1]
    img_name = img_file.split('.')[0]
    annot_path = os.path.join(path, f'{img_name}.xml')
    # skip images without any annotations
    if not os.path.exists(annot_path):
        continue
    annot_file = annot_path.split('\\')[-1]
    # new_img_name = f'{counter}.jpg'
    # new_annot_name = f'{counter}.xml'
    move(image, Path(img_folder, f'{counter}.jpg'))
    move(annot_path, Path(annot_folder, f'{counter}.xml'))
    counter += 1

