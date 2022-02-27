# import albumentations as A
# from scripts.annotations_xml import AnnotationsXML
import argparse
from .annotations_xml import AnnotationsXML
import albumentations as A
# from scripts.label_json import LabelHandler
from .label_json import LabelHandler
import cv2
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
import sys
import os
import glob
from tqdm import tnrange, tqdm
from .separate_with_label import export_path_count_for_each_label
sys.path.insert(0, os.path.dirname(__file__))



BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)


class AMRLImageAug:
    """A wrapper class on albumentations package to work on pascal voc format easily

    Longer class information....
    Longer class information....

    Attributes:
        xmls_dir: annotations files paths.
        images_dir: images files paths.
    """

    file_counter = 0

    def __init__(self, json_label_path, xmls_dir, images_dir):
        """

        Args:
            xmls_dir (str): annotations files paths.
            json_label_path (:obj:`str`): you should have a json file like this
                    {
                    "1": "object1",
                    "2": "object2",
                    "3": "object3",
                    "4": "object4",
                    "5": "object5",
                    }
            images_dir (str) :images files paths.
        """

        assert os.path.exists(json_label_path), 'Json file not exist'
        label_handeler = LabelHandler(json_label_path)

        self._categori_label_id = label_handeler.json_label_id_dic()
        self._categori_id_label = label_handeler.json_id_label_dic()

        assert os.path.exists(
            xmls_dir), 'XML path not exist please check the path'
        self.xmls_dir = xmls_dir

        assert os.path.exists(
            images_dir), 'Image path not exist please check the image path'
        self.images_dir = images_dir
        self._annotations_handler = AnnotationsXML(
            self.images_dir, self.xmls_dir)

    def load_image(self, path):

        image = cv2.imread(path, cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def visualize_bbox(self, img, bbox, class_id, class_idx_to_name, color=BOX_COLOR, thickness=2):
        x_min, y_min, x_max, y_max = bbox
        x_min, x_max, y_min, y_max = int(x_min), int(
            x_max), int(y_min), int(y_max)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max),
                      color=BOX_COLOR, thickness=thickness)
        class_name = class_idx_to_name[class_id]
        ((text_width, text_height), _) = cv2.getTextSize(
            class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)),
                      (x_min + text_width, y_min), BOX_COLOR, -1)
        cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_COLOR, lineType=cv2.LINE_AA)
        return img

    def visualize(self, annotations, category_id_to_name):
        img = annotations['image'].copy()
        for idx, bbox in enumerate(annotations['bboxes']):
            img = self.visualize_bbox(
                img, bbox, annotations['category_id'][idx], category_id_to_name)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_aug(self, aug, min_area=0., min_visibility=0.):
        return A.Compose(aug, A.BboxParams(format='pascal_voc', min_area=min_area,
                                           min_visibility=min_visibility, label_fields=['category_id']))

    def augment_image(self, xml_file_path, quantity, resize=False, width=0, height=0):
        """augmentation for one picture depend on quantity that you get for it
        if your image and xml names are same
        save your aug image in your dataset path with the following pattern aug_{counter}.jpg

        Args:
            xml_file_path: single xml file path.
            quantity: quantity for your image to augment
            resize:(bool : optional)-> defult False ... resize your augmented images
            width:(int : optional) width for resized ... if resize True you should use this arg
            height:(int : optional) height for resized... if resize True you should use this arg
        Returns:
            No return
        """
        filters_of_aug = [

            A.RandomSizedBBoxSafeCrop(
                width=448, height=336, erosion_rate=0.1, p=0.1),
            A.RandomRotate90(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomShadow(p=0.1),
            A.RandomSnow(snow_point_lower=0.1,
                         snow_point_upper=0.15, p=0.1),
            A.RGBShift(p=0.2),
            A.CLAHE(p=0.2),

            A.HueSaturationValue(
                p=0.1, hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10),

            A.MotionBlur(p=0.1),
            A.MedianBlur(p=0.2),
            A.ISONoise(p=0.2),
            A.Posterize(p=0.2),
            A.IAAPerspective(p=0.1),
            A.IAAPiecewiseAffine(p=0.1, scale=(0.01, 0.02)),
            A.IAAEmboss(p=0.2),
        ]
        if resize:
            filters_of_aug.append(A.Resize(width, height, always_apply=True))
        aug = self.get_aug(filters_of_aug)
        for i in tqdm(range(quantity), desc='Singel image'):
            annotations = self._annotations_handler.parse_pascal_voc(
                xml_file_path, self._categori_label_id)

            augmented = aug(**annotations)
            AMRLImageAug.file_counter += 1
            file_name = 'aug_image{}'.format(AMRLImageAug.file_counter)
            self._annotations_handler.pascal_image_voc_writer(
                augmented, file_name, self._categori_id_label)

    def auto_augmentation(self, count_of_each, resize=False, width=0, height=0):
        """auto augmentation for each picture depend on statistic of the object exist in your dataset
        if your image and xml names are same
        save your aug image in your dataset path with the following pattern aug_{counter}.jpg

        Args:
            count_of_each(int): How much of each label you want to have !
            resize:(bool : optional)-> defult False ... resize your augmented images
            width:(int : optional) width for resized ... if resize True you should use this arg
            height:(int : optional) height for resized... if resize True you should use this arg
        Returns:
            No return
        """
        labels_statistics = export_path_count_for_each_label(
            self.xmls_dir, self.images_dir, self._categori_label_id.keys())
        for label in tqdm(self._categori_label_id):
            count = labels_statistics[label]['count']
            xmls_paths = labels_statistics[label]['xmls_paths']
            if not count:
                continue
            coefficient = count_of_each // count
            print(coefficient)
            for xml in xmls_paths:
                try:
                    self.augment_image(xml, coefficient, resize, width, height)
                except ValueError as e:
                    print('File {} except because {}'.format(xml, e))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images_dir", required=True,
                    help="path to images file")
    ap.add_argument("-x", "--xmls_dir", required=True,
                    help="path to xmls file")
    ap.add_argument("-j", "--label_json_path", required=True,
                    help="path to label.json file")
    ap.add_argument("-q", "--quantity", required=True,
                    help="the amount that you want to each object")
    args = vars(ap.parse_args())
    images_dir = args['images_dir']
    xmls_dir = args["xmls_dir"]
    json_path = args['label_json_path']
    quantity = args["quantity"]
    img_aug = AMRLImageAug(json_path, xmls_dir, images_dir)
    img_aug.auto_augmentation(quantity)
