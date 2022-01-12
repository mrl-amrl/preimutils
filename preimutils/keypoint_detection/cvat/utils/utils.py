import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import xmltodict


def load_images_from_folder(dir: str, image_list: list) -> np.array:
    """

    :param dir: Images Directory
    :param image_list: List name of images
    :return: Numpy array of Images
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


def augmented_images_write(img: list, key_points: list, images_name: list, save_dir: str) -> None:
    for i in range(len(img)):
        cv2.imwrite(os.path.join(save_dir, images_name[i]), img[i])

    data_dict = {}

    for i in range(len(images_name)):
        data_dict[images_name[i]] = key_points[i]

    df = pd.DataFrame.from_dict(data_dict, orient='index', )
    df.to_csv(os.path.join(save_dir, 'augmented.csv'))


def visualize_keypoints(images, keypoints, save: bool = False, name: str = None) -> None:
    """

    Args:
        images: Numpy array image
        keypoints: list of image keypoints
        save: (bool) if True image saved in directory
        name: (str) image file name to save (use when save=True)

    Returns:

    """
    image = images.copy()
    current_keypoint = keypoints
    plt.imshow(image)
    current_keypoint = np.array(current_keypoint)
    current_keypoint = current_keypoint[:, :2]
    for idx, (x, y) in enumerate(current_keypoint):
        plt.scatter([x], [y], marker=".", s=50, linewidths=5)
    if save:
        plt.savefig(f'{name}.png')
    plt.figure(figsize=(30, 30))
    plt.show()


def xml_to_csv(xml_dir: str) -> pd.DataFrame:
    """
    :param xml_dir: Directory of xml file
    :return: Pandas DataFrame created from xml file
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
    :param xml_dir: Directory of xml file
    :param save_dir: Directory to save csv file
    """
    df = xml_to_csv(xml_dir)
    try:
        df.to_csv(save_dir)
        print(f'Saved: {save_dir}')
    except Exception as e:
        print(e)


def read_images_and_annotations(image_dir: str, annotation_csv_dir: str):
    """

    Args:
        image_dir: Image directory
        annotation_csv_dir: annotation csv file saved in augmention

    Returns:
            Images , List of each image key points
    """
    df = pd.read_csv(annotation_csv_dir)

    images = []
    keypoints = []
    for k in df.iterrows():
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
