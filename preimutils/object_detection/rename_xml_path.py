import os
import glob
import xml.etree.ElementTree as ET
import argparse
from tqdm import tqdm
from .label_json import LabelHandler


def xml_address_changer(xmls_dir, images_dir):
    """change the wrong path that exist in xmls files
        you should do it after every move on your dataset
    Args:
        images_dir: images of your dataset path.
        xmls_dir: xmls of your dataset path

    Returns:
        bboxs point from object
    """
    for xml_file in tqdm(glob.glob(os.path.join(xmls_dir , '*.xml'))):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        xml_base_name = os.path.basename(xml_file).split('.')[0]
        root.find('filename').text = xml_base_name + '.jpg'

        root.find('path').text = images_dir + '/' + xml_base_name + '.jpg'
        tree.write(xml_file)
