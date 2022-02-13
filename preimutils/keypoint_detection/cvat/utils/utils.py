import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import xmltodict
from tqdm import tqdm


def load_images_from_folder(dir: str, image_list: list) -> np.array:
    """
    Read Images on input directory depend on Images name list
    Args:
        dir: Images Directory
        image_list: List name of images

    Returns:
        Numpy array of Images
    """

    images = []
    for image in tqdm(image_list):
        # Read image
        img = cv2.imread(os.path.join(dir, image), cv2.IMREAD_COLOR)
        images.append(img)

    return images


def xml_to_dict(xml_dir: str) -> dict:
    """

    Args:
        xml_dir: Directory of XML file in string format

    Returns:
        Dictionary of XML file
    """

    file_xml = open(xml_dir, 'r')
    data_dict = xmltodict.parse(file_xml.read())
    file_xml.close()
    data_json_str = json.dumps(data_dict)
    return json.loads(data_json_str)


def augmented_images_write(img: list, key_points: list, images_name: list, save_dir: str) -> None:
    """
    Save Augmented Images
    Args:
        img: (list) Images
        key_points: (list) Images key_points
        images_name: (list) Images Names to save
        save_dir: (str) Directory to save Images

    """
    for i in range(len(img)):
        cv2.imwrite(os.path.join(save_dir, images_name[i]), img[i])

    data_dict = {}

    for i in range(len(images_name)):
        data_dict[images_name[i]] = key_points[i]

    df = pd.DataFrame.from_dict(data_dict, orient='index', )
    df.to_csv(os.path.join(save_dir, 'augmented.csv'))


def visualize_keypoints(images: np.array, keypoints: list, save: bool = False, name: str = None,
                        size: int = 50) -> None:
    """
    Function for visualizing key_points on image
    Args:
        images: Numpy array image
        keypoints: list of image keypoints
        save: (bool) if True image saved in directory
        name: (str) image file name to save (use when save=True)
        size: (int) size of keypoints on image (default = 50)

    """
    image = images.copy()
    current_keypoint = keypoints
    plt.imshow(image)

    for kp in current_keypoint:
        try:
            kp.pop(2)
        except:
            continue
    # print(current_keypoint)
    current_keypoint = np.array(current_keypoint)
    current_keypoint = current_keypoint[:, :2]
    for idx, (x, y) in enumerate(current_keypoint):
        plt.scatter([x], [y], marker=".", s=size, linewidths=5)
    if save:
        plt.savefig(f'{name}.png')
    plt.figure(figsize=(30, 30))
    plt.show()


def xml_to_csv(xml_dir: str) -> pd.DataFrame:
    """
    Convert XML file to Pandas DataFrame (CSV format)
    Args:
        xml_dir: Directory of xml file

    Returns:
        Pandas DataFrame created from xml file
    """

    ann_dict = xml_to_dict(xml_dir)
    images_name = [ann_dict['annotations']['image'][i]['@name'] for i in range(len(ann_dict['annotations']['image']))]

    keypoints = []
    for i in range(len(ann_dict['annotations']['image'])):
        kp = []
        for p in range(len(ann_dict['annotations']['image'][i]['points'])):
            xy = ann_dict['annotations']['image'][i]['points'][p]['@points']
            kplabel = ann_dict['annotations']['image'][i]['points'][p]['@label']
            xy = xy.split(',')
            x = float(xy[0])
            y = float(xy[1])
            kp.append((x, y))
        keypoints.append(kp)

    data_dict = {}

    for i in range(len(images_name)):
        data_dict[images_name[i]] = keypoints[i]
    print(data_dict)
    # df = pd.DataFrame(data_dict)
    df = pd.DataFrame.from_dict(data_dict, orient='index', )
    return df


def xml_csv_save(xml_dir: str, save_dir: str) -> None:
    """
    Convert XML file to CSV format and Save it
    Args:
        xml_dir: Directory of xml file
        save_dir: Directory to save csv file

    """
    df = xml_to_csv(xml_dir)
    try:
        df.to_csv(save_dir)
        print(f'Saved: {save_dir}')
    except Exception as e:
        print(e)


def read_images_and_annotations(image_dir: str, annotation_csv_dir: str):
    """
    Function to read Images and Annotations from Input Directory
    Args:
        image_dir: Image directory
        annotation_csv_dir: annotation csv file saved in augmention

    Returns:
            Images , List of each image key points
    """
    df = pd.read_csv(annotation_csv_dir)

    images = []
    keypoints = []
    for k in tqdm(df.iterrows()):
        img = cv2.imread(os.path.join(image_dir, k[1][0]), cv2.IMREAD_COLOR)
        images.append(img)

        dfsprow = df.loc[df['Unnamed: 0'] == k[1][0]]
        list_row_string = list(dfsprow.iloc[0])[1:]
        list_row_tuple = []
        for item in list_row_string:
            if not ('nan' in str(item)):
                list_row_tuple.append(eval(item))

        keypoints.append(list_row_tuple)

    return images, keypoints
