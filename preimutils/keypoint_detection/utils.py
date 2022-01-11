import os
import cv2
import numpy as np
import tqdm
import json
import xmltodict

def load_images_from_folder(dir: str, image_list:list) -> np.array:
    """

    :param dir: Images Directory
    :param image_list: List name of images
    :return: Numpuy array of Images
    """
    images = []
    for image in tqdm.tqdm(image_list):
        # Read image
        img = cv2.imread(os.path.join(dir, image), cv2.IMREAD_COLOR)
        images.append(img)

    return images


def xml_to_dict(xml_dir: str) -> dict:
    """
    :param xml_dir: Directory of XML file in string format
    :return: Dictionary of XML file
    """
    file_xml = open(xml_dir, 'r')
    data_dict = xmltodict.parse(file_xml.read())
    file_xml.close()
    data_json_str = json.dumps(data_dict)
    return json.loads(data_json_str)

def augmented_images_write(img:np.array, name:str) -> None:
    cv2.imwrite(img, name)