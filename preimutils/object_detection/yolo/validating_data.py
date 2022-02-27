import os

import tqdm


def check_dataset(annotations_path: str, images_path: str):
    txt_files = os.listdir(annotations_path)
    images_files = os.listdir(images_path)
    image_format = images_files.split('.')[-1]
    for txt_name in tqdm.tqdm(txt_files):
        name = txt_name.replace('txt', image_format)
        if not name in images_files:
            print(f'{name} not exist in directory')
