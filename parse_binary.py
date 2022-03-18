from glob import glob
from pathlib import Path
import xml.etree.ElementTree as ET

files = glob('./annots/**.xml')

annots_new = Path('./annots_n_b')

if not annots_new.exists():
    annots_new.mkdir(parents=True, exist_ok=True)

for file in files:
    tree = ET.parse(file)
    root = tree.getroot()

    for object in root.findall('object'):
        name = object.find('name')

        name.text = 'crack'
        

    if len(root.findall('object')) > 0:
        fname = file.split('\\')[-1].split('.')[0]
        tree.write(f'./annots_n_b/{fname}.xml')