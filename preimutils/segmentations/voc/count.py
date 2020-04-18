import cv2
import imutils
from imutils import contours
import numpy as np
from dataset_info import LabelMap
import glob
import os
from tqdm import tqdm
import utils

def export_path_count_for_each_label(color_label,images_dir, masks_dir):
    """Get statistics of dataset with their labels with their mask and images files path

    Args:
        xmls_dir: all xmls file directory.
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
    print(color_label)
    label_statistic = {value : {'count' : 0 , 'masks_path' : [],'images_path' : []}  for value in color_label.values()}
    for mask_path in tqdm(glob.glob(os.path.join(masks_dir, '*.png'))):
        image_path = utils.find_image_from_mask(mask_path,images_dir)
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
            label_statistic[color_name]['images_path'].append(image_path)

    return label_statistic
