from glob import glob
from pathlib import Path
import xml.etree.ElementTree as ET

files = glob('./annots/**.xml')

annots_new = Path('./annots_n')

allowed = [
    'staffe_scoperte_ossidate',
    'armatura_corrosa_ossidata',
    'vespai',
    'macchie_umid_attiva',
    'tracce_scolo',
    'distacco_copriferro',
    'macchie_umid_passiva',
    'cls_ammalorato',
    'fessure'
]

if not annots_new.exists():
    annots_new.mkdir(parents=True, exist_ok=True)

for file in files:
    tree = ET.parse(file)
    root = tree.getroot()

    for object in root.findall('object'):
        name = object.find('name')
        
        if 'alta' in name.text or 'media' in name.text or 'bassa' in name.text:
            new_text = name.text.split('_')
            new_text = ('_').join(new_text[:-1])
            name.text = new_text
        

        if name.text == 'armatura_corrorsa_ossidata_alta':
            name.text = 'armatura_corrosa_ossidata'
        if name.text == 'armatura_corrosa_ossidata_alta':
            name.text = 'armatura_corrosa_ossidata'
        if name.text == 'cls_ammalorato_alta':
            name.text = 'cls_ammalorato'
        if name.text == 'distacco_copriferro_alta':
            name.text = 'distacco_copriferro'
        if 'fessure' in name.text:
            name.text = 'fessure'
        if name.text == 'lesioni_ragnatela_bass':
            name.text = 'lesioni_ragnatela'
        if (name.text == 'macche_umid_attiva' or \
            name.text == 'macchie_umid_attiva' or \
            name.text == 'macchie_umid_attiva_alta'):
            name.text = 'macchie_umid_attiva'
        if (name.text == 'macchie_umid_passiva' or \
            name.text == 'macchie_umid_passiva_alta'):
            name.text = 'macchie_umid_passiva'
        if (name.text == 'staffe_ossidate_scoperte' or \
            name.text == 'staffe_scoperte_ossidate_alta' or \
            name.text == 'staffe_scoperte_ossidate_bassa' or \
            name.text == 'staffe_scoperte_ossidate_media'):
            name.text = 'staffe_scoperte_ossidate'
        if name.text == 'tracce_scolo_alta':
            name.text = 'tracce_scolo'
        if (name.text == 'vespai_alta' or \
            name.text == 'vespai_bassa' or \
            name.text == 'vespai_media'):
            name.text = 'vespai'
        if name.text == 'distacco_tamponi_testate':
            name.text = 'distacco_tampone'
        
        if not name.text in allowed:
            root.remove(object)

    if len(root.findall('object')) > 0:
        fname = file.split('\\')[-1].split('.')[0]
        tree.write(f'./annots_n/{fname}.xml')
