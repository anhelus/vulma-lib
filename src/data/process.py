import cv2
import os
import numpy as np
import glob

BASE_PATH = 'C:\\Users\\acard\\Desktop\\bridge-vulma v3\\Bridge-VULMA-2'
DATASET_PATH = 'C:\\Users\\acard\\Desktop\\bridge-vulma v3\\Bridge-VULMA-2\\test\\labels'
IMG_PATH = 'C:\\Users\\acard\\Desktop\\bridge-vulma v3\\Bridge-VULMA-2\\test\\images'
DST_PATH = 'C:\\Users\\acard\\Desktop\\bridge-vulma v3\\Bridge-VULMA-2\\test\\newdata'

class_names = []
with open(os.path.join(BASE_PATH, 'obj.names')) as f:
    class_names = [x.strip('\n') for x in f.readlines()]
    for c in class_names:
        new_dir = os.path.join(DST_PATH, c)
        if not os.path.isdir(new_dir):
            os.makedirs(new_dir)


files = glob.glob(f"{DATASET_PATH}/*.txt")

idx = 0
for fn in files:
    with open(fn) as f:
        lines = f.readlines()
        if(lines):
            base_name = '.'.join(os.path.split(fn)[-1].split('.')[:-1])
            img_fn = os.path.join(IMG_PATH, f"{base_name}.jpg")
            img = cv2.imread(img_fn)
            if img is None:
                continue
            for line in lines:
                # Parsing della riga
                tmp = line.split(' ')
                tmp[-1].strip('\n')
                rect = [float(x) for x in tmp[1:]]
                DST_DIR = class_names[int(tmp[0])]
                height = img.shape[0]
                width = img.shape[1]
                x = int(np.round(rect[0] * width))
                y = int(np.round(rect[1] * height))
                w = int(np.round(rect[2] * width))
                h = int(np.round(rect[3] * height))
                
                ROI = img[y:y+h, x:x+w]
                roi_fn = os.path.join(DST_PATH, DST_DIR, f"{idx}.png")
                cv2.imwrite(roi_fn, ROI)
                idx += 1

