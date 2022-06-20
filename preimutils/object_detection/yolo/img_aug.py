import os

import albumentations as A
import cv2
from tqdm import tqdm

from .reader_writer import read_image_and_annotation, write_image_and_annotations
from .utils import mkdir_p
from .validating_data import check_dataset

BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)


class AMRLImageAug:
    """A wrapper class on albumentations package to work on pascal voc format easily

    Longer class information....
    Longer class information....

    Attributes:
        annotations_dir: annotations files paths.
        images_dir: images files paths.
    """

    file_counter = 0

    def __init__(self, annotations_dir, images_dir, output_dir):
        """

        Args:
            annotations_dir (:obj:`str`): annotations files paths.
            images_dir (str) :images files paths.
        """

        assert os.path.isdir(annotations_dir), 'annotations dir file not exist'
        assert os.path.isdir(images_dir), 'images dir not exist'
        mkdir_p(output_dir)
        mkdir_p(os.path.join(output_dir, 'annotations'))
        mkdir_p(os.path.join(output_dir, 'images'))

        self.annotations_dir = annotations_dir
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.output_labels_dir = os.path.join(output_dir, 'annotations')
        self.output_image_dir = os.path.join(output_dir, 'images')

    def check_data(self, annotations_dir, images_dir):
        check_dataset(annotations_dir, images_dir)

    def load_image(self, path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        return image

    def visualize_bbox(self, img, bbox, class_name, color=BOX_COLOR, thickness=2):
        cx, cy, w, h = bbox
        imgSize = img.shape

        xmin = max(float(cx) - float(w) / 2, 0)
        xmax = min(float(cx) + float(w) / 2, 1)
        ymin = max(float(cy) - float(h) / 2, 0)
        ymax = min(float(cy) + float(h) / 2, 1)

        x_min = int(imgSize[1] * xmin)
        x_max = int(imgSize[1] * xmax)
        y_min = int(imgSize[0] * ymin)
        y_max = int(imgSize[0] * ymax)

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max),
                      color=color, thickness=thickness)

        ((text_width, text_height), _) = cv2.getTextSize(
            str(class_name), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)

        cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)),
                      (x_min + text_width, y_min), color, -1)
        cv2.putText(img, str(class_name), (x_min, y_min - int(0.3 * text_height)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_COLOR, lineType=cv2.LINE_AA)
        return img

    def visualize(self, annotations):
        img = annotations['image'].copy()
        for idx, bbox in enumerate(annotations['bboxes']):
            img = self.visualize_bbox(
                img, bbox, annotations['class_name'][idx])
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_aug(self, aug, min_area=0., min_visibility=0.):
        return A.Compose(aug, A.BboxParams(format='yolo', min_area=min_area,
                                           min_visibility=min_visibility, label_fields=['bbox_classes']))

    def augment_image(self, txt_file_path, image_path, quantity, resize=False, width=0, height=0):
        """augmentation for one picture depend on quantity that you get for it
        if your image and xml names are same
        save your aug image in your dataset path with the following pattern aug_{counter}.jpg

        Args:
            txt_file_path: single txt file path.
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
            A.Perspective(p=0.1),
            A.PiecewiseAffine(p=0.1, scale=(0.01, 0.02)),
            A.Emboss(p=0.2),
        ]
        if resize:
            filters_of_aug.append(A.Resize(width, height, always_apply=True))
        aug = self.get_aug(filters_of_aug)
        for i in tqdm(range(quantity), desc=f'Singel image : {os.path.basename(image_path)}'):
            image, boxes_label, boxes = read_image_and_annotation(image_path, txt_file_path)

            augmented = aug(
                image=image,
                bboxes=boxes,
                bbox_classes=boxes_label
            )
            AMRLImageAug.file_counter += 1
            file_name = 'aug_image{}'.format(AMRLImageAug.file_counter)
            img_pth = os.path.join(self.output_image_dir, file_name + '.jpg')
            txt_pth = os.path.join(self.output_labels_dir, file_name + '.txt')
            write_image_and_annotations(
                image_path=img_pth,
                txt_path=txt_pth,
                image=augmented['image'],
                bboxes=augmented['bboxes'],
                bbox_classes=augmented['bbox_classes']
            )

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
        images_list = os.listdir(self.images_dir)
        for image in tqdm(images_list):
            if os.path.splitext(image)[0] + '.txt' in os.listdir(self.annotations_dir):
                img_name = os.path.join(self.images_dir, image)
                txt_name = os.path.join(self.annotations_dir, os.path.splitext(image)[0] + '.txt')
                self.augment_image(txt_name, img_name, count_of_each, resize, width, height)
            else:
                print('{} not found in {}'.format(os.path.splitext(image)[0] + '.txt', self.annotations_dir))
                continue
