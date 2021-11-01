from random import shuffle
import glob
import os
from tqdm import tqdm
import shutil
# Shuffle your images and xmls name this work is require before your images to train
# If you dont shuffle your datasets then your Neural Network train mostly on one class object


def shuffle_img_xml(xmls_dir, images_dir, dst_dir):
    """
    Shuffling images and save in the destination path
    Args:
        xmls_dir: all xmls files directory.
        images_dir: your images directory.
        dst_dir:destination directory to save after shuffling

    Returns:
        No return
    """
    count = 0
    dst_images_path = os.path.join(dst_dir, 'images')
    dst_xmls_path = os.path.join(dst_dir, 'annotations')

    # check xmls path
    if not os.path.exists(dst_xmls_path):
        os.makedirs(dst_xmls_path)
    
    # check images path
    if not os.path.exists(dst_images_path):
        os.makedirs(dst_images_path)
    for source_xml_path in tqdm(glob.glob(os.path.join(xmls_dir , '*.xml'))):
        file_base_name = os.path.basename(source_xml_path)[:-4]
        source_image_path = glob.glob(os.path.join(
            images_dir , '{}.jpg'.format(file_base_name)))
        source_image_path = source_image_path[0]
        dst_image_path = os.path.join(
            dst_images_path, 'image{}.jpg'.format(count))
        dst_xml_path = os.path.join(dst_xmls_path, 'image{}.xml'.format(count))
        count += 1
        shutil.copy2(source_image_path, dst_image_path)
        shutil.copy2(source_xml_path, dst_xml_path)


