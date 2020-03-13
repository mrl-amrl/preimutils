import cv2
import imutils
from imutils import contours
import numpy as np
from dataset_info import LabelMap
import glob
import os
from scipy.spatial import distance as dist
from tqdm import tqdm


def dataset_statistics(color_label, segmentation_class_dir):
    label_statistic = {value: 0 for value in color_label.values()}

    for image_path in tqdm(glob.glob(os.path.join(segmentation_class_dir, '*.png'))):
        image = cv2.imread(image_path)
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
                    image_path, mean_rgb))
            label_statistic[color_name] += 1
    return label_statistic


if __name__ == "__main__":
    labels_handler = LabelMap('/home/amir/segmentation/pascal_voc_seg/VOCdevkit/VOC2012/labelmap.txt')
    labels = labels_handler.color_label()
    static = dataset_statistics(
        labels, '/home/amir/segmentation/pascal_voc_seg/VOCdevkit/VOC2012/SegmentationClass')
