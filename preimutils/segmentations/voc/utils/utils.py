# from dataset_info import Dataset
import glob
import os
import shutil
from glob import glob

import cv2
import imutils
import numpy as np
from imutils import contours
from matplotlib import pyplot as plt
from tqdm import tqdm
import mkautodoc
from ..dataset import LabelMap


def custom_to_voc(masks_dir, images_dir, target_dir):
    """Convert your custom dataset to normal voc format

    Args:

        masks_dir (str): your masks path.
        images_dir (str): your images path.

    Returns:

        (list) : unique colors from your masks
    """
    os.makedirs(target_dir, exist_ok=True)
    seg_cls = os.path.join(target_dir, 'SegmentationClass')
    jpegimages = os.path.join(target_dir, 'JPEGImages')
    seg_object = os.path.join(target_dir, 'SegmentationObject')
    seg_txt = os.path.join(target_dir, 'ImageSets', 'Segmentation')

    os.makedirs(seg_cls, exist_ok=True)
    os.makedirs(jpegimages, exist_ok=True)
    os.makedirs(seg_object, exist_ok=True)
    os.makedirs(seg_txt, exist_ok=True)

    unique_colors = unique_label_from_masks(masks_dir)
    label_map = ['label:color_rgb:parts:actions']
    label_map.append('background:0,0,0::')
    for i, color in enumerate(unique_colors):
        label_map.append('object{}:{},{},{}::'.format(
            i+1, color[0], color[1], color[2]))
    with open(os.path.join(target_dir, 'labelmap.txt'), 'w') as f:
        f.write('\n'.join(label_map))
    for mask in glob(os.path.join(masks_dir, '*.png')):
        shutil.copy2(mask, seg_cls)
    for image in glob(os.path.join(images_dir, '*.*')):
        shutil.copy2(image, jpegimages)


def export_path_count_for_each_label(color_label, images_dir, masks_dir, extention='jpg'):
    """Get statistics of dataset with their labels with their mask and images files path

    Args:

        masks_dir: your mask images directory.
        images_dir: your images directory.
        color_label:[(r,g,b):object1,(r,g,b):'object2',...,(r,g,b):'objectN']

    Return:

        dict{   label1: {
            count:
            masks_paths:[]
            images_paths:[]
                    },
                    ...,
        labelN: {
            count:
            masks_paths:[]
            images_paths:[]
                    }
        }
    """
    label_statistic = {value: {'count': 0, 'masks_path': [],
                               'images_path': []} for value in color_label.values()}
    for mask_path in tqdm(glob(os.path.join(masks_dir, '*.png'))):
        image_path = find_image_from_mask(
            mask_path, images_dir, extention=extention)
        image = cv2.imread(mask_path)
        orig = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = imutils.auto_canny(gray)
        ret, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for i in range(len(cnts)):
            mask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.drawContours(mask, cnts, i, 255, thickness=cv2.FILLED)
            mean = cv2.mean(image, mask=mask)[:3]
            mean_rgb = (round(mean[2]), round(mean[1]), round(mean[0]))
            try:
                color_name = color_label[mean_rgb]
            except KeyError:
                print('file : {} has wrong label please check it with rgb color of {}'.format(
                    mask_path, mean_rgb))
            label_statistic[color_name]['count'] += 1
            label_statistic[color_name]['masks_path'].append(mask_path)
            label_statistic[color_name]['images_path'].append(image_path[0])

    return label_statistic


def encode_segmap(mask, class_color):
    """Encode segmentation label images as pascal classes

    Args:

        mask (np.ndarray): raw segmentation label image of dimension
            (M, N, 3), in which the Pascal classes are encoded as colours.
        class_color(list): class colors 

    Returns:

        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    class_color = np.asarray(class_color)
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(class_color):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def decode_segmap(label_mask, class_color, plot=False):
    """Decode segmentation class labels into a color image

    Args:

        label_mask (np.ndarray): an (M,N) array of integer values denoting
            the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
            in a figure.

    Returns:

        (np.ndarray, optional): the resulting decoded color image.
    """
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, len(class_color)):
        r[label_mask == ll] = class_color[ll][0]
        g[label_mask == ll] = class_color[ll][1]
        b[label_mask == ll] = class_color[ll][2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    # rgb[:, :, 0] = r / 255.0
    # rgb[:, :, 1] = g / 255.0
    # rgb[:, :, 2] = b / 255.0

    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    rgb = rgb.astype(np.uint8)

    if plot:
        plt.imshow(rgb)
        plt.show()

    else:
        return rgb


def find_image_from_mask(mask, images_dir, extention='jpg'):
    """Export image path from the mask file 
    if your image and mask names are same

    Args:

        mask: single mask.png file path.
        images_dir: your images path.

    Returns:

        image path : path of the input mask  -> string.
    """
    mask = os.path.basename(mask)
    image_path = glob(os.path.join(images_dir, mask[:-4] + '.' + extention))
    if not len(image_path):
        raise ValueError("Image {} not found".format(
            images_dir + '/' + mask[:-4] + '.' + extention))
    return image_path


def find_maxmin_size_images(images_dir):
    """Export the maximum and minimum size of the dataset images 

    Args:

        images_dir: your images path.

    Returns:

        image path : path of the input mask  -> string.
    """
    min_height = 10000
    min_width = 10000
    max_height = 0
    max_width = 0
    for image in tqdm(glob(os.path.join(images_dir, '*.*'))):
        img = cv2.imread(image)
        try:
            height, width, _ = img.shape
        except AttributeError:
            print('{} is not a image file ', image)
        min_height = min(min_height, height)
        min_width = min(min_width, width)
        max_height = max(height, max_height)
        max_width = max(width, max_width)

    return {
        'min_height': min_height,
        'min_width': min_width,
        'max_height': max_height,
        'max_width': max_width
    }


def unique_label_from_masks(masks_dir):
    """get the unique colors(classes) from your masks

    Args:

        masks_dir:(str) your masks path.

    Returns:

        unique_colors : unique colors from your masks  -> list.
    """
    unique_colors = set()

    for mask_path in tqdm(glob(os.path.join(masks_dir, '*.png'))):
        mask = cv2.imread(mask_path)
        orig = mask.copy()
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        edged = imutils.auto_canny(gray)
        ret, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for i in range(len(cnts)):
            mask_zero = np.zeros(mask.shape[:2], dtype="uint8")
            cv2.drawContours(mask_zero, cnts, i, 255, thickness=cv2.FILLED)
            mean = cv2.mean(mask, mask=mask_zero)[:3]
            mean_rgb = (round(mean[2]), round(mean[1]), round(mean[0]))
            unique_colors.add(mean_rgb)

    unique_colors = list(unique_colors)
    # Add black (background) to unique colors
    unique_colors.insert(0, (0, 0, 0))
    return unique_colors
