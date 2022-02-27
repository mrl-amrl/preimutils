import os
import glob
import xml.etree.ElementTree as ET
from tqdm import tqdm
from .label_json import LabelHandler
import json


def label_checker(xmls_dir, classes_array):
    """If you have wrong label in your annotations file detect and show you

    Args:
        xmls_dir: source xmls directory.
        classes_array -> list: classed of object that you have 
    Returns:
        no return
    """
    labels = {}
    wrong_labels = set()
    for key in classes_array:
        labels[key] = 0
    for xml_file in tqdm(glob.glob(os.path.join(xmls_dir + '*.xml'))):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            try:
                labels[member.find('name').text] += 1
            except KeyError:
                wrong_labels.add(member.find('name').text)
            if not member.find('name').text in classes_array:
                print('label {} in file {} is wrong'.format(
                    member.find('name').text, xml_file))

    print('statics of labels', labels)
    with open('statics.json', 'w') as json_file:
        json.dump(labels, json_file)
    print('wrong labels are ', wrong_labels)


def replace_label(xmls_dir, from_label, dst_label):
    """replace the wrong label that you detect with label_checker the currect one

    Args:
        xmls_dir: source xmls directory.
        from_label -> str: wrong label
        dst_label -> str : currect label
    Returns:
        no return
    """
    for xml_file in tqdm(glob.glob(xmls_dir + '/*.xml')):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            if member.find('name').text == from_label:
                member.find('name').text = dst_label
        tree.write(xml_file)
