import os
from glob import glob

import albumentations as A
import cv2
import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm




class ClassificationAug:

    """A wrapper class on albumentations package to work on voc segmentation format easily
    Attributes:
        mask_dir: masks files paths.
        images_dir: images files paths.
    Args:
        images_dir (str) :images files paths.
    """

    _file_counter = 0

    def __init__(self, images_dir, images_extention='jpg'):
        self.images_extention = images_extention
        assert os.path.exists(images_dir), 'label_map.txt file not exist'
        self._images_dir = images_dir
        self.filters_of_aug = [
            # flips
            A.HorizontalFlip(p=0.25),
            A.VerticalFlip(p=0.25),

            # color augmentations
            A.OneOf([
                A.HueSaturationValue(p=1.),
                A.RandomBrightnessContrast(p=1.),
                A.RGBShift(p=1.)
            ], p=0.25),

            # image quality
            A.OneOf([
                A.GaussNoise(p=1.),
                A.MultiplicativeNoise(p=1.),
                A.JpegCompression(p=1.),
                A.Downscale(scale_min=0.5,scale_max=0.99, p=1),
            ], p=0.5),

            A.ToGray(p=0.05),

            A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0,
                               shift_limit=0.1, p=1, border_mode=0),

            A.RandomShadow(p=0.1),
            A.RandomSnow(snow_point_lower=0.1,
                         snow_point_upper=0.15, p=0.1),
            A.CLAHE(p=0.2),

            A.MotionBlur(p=0.1),
            A.MedianBlur(p=0.2),
            A.ISONoise(p=0.2),
            A.Posterize(p=0.2),
            A.augmentations.geometric.transforms.Perspective(p=0.1),
            A.augmentations.geometric.transforms.PiecewiseAffine(
                p=0.1, scale=(0.01, 0.02)),
            A.augmentations.transforms.Emboss(p=0.2),
        ]

    def _get_aug(self, aug):
        return A.Compose(aug)

    def add_filter(self, filter):
        self.filters_of_aug.append(filter)

    def augment_image(self, image_path, quantity, resize=False, width=0, height=0):
        """
        Args:
            image_path str: image path.
            quantity int: number of augmentations.
            resize: if set true resize image to width and height.
            width: resize w
            height: resize h


        """
        aug = self.filters_of_aug.copy()
        image_base = os.path.basename(image_path)
        image_name = os.path.splitext(image_base)[0]
        if resize:
            aug.append(A.Resize(width, height, always_apply=True))
        aug = self._get_aug(aug)
        image = cv2.imread(image_path)
        for i in tqdm(range(quantity), desc='Singel image'):
            augmented = aug(image=np.array(image))
            nimage = augmented['image']
            ClassificationAug._file_counter += 1
            file_name = '{}_aug{}.{}'.format(image_name, ClassificationAug._file_counter, self.images_extention)
            path_to_save = os.path.join(self._images_dir, file_name)
            if not cv2.imwrite(path_to_save, nimage):
                print('Error in saving image', path_to_save)


    def auto_augmentation(self, count_of_each, resize=False, width=0, height=0):
        for image_path in tqdm(glob(os.path.join(self._images_dir, '*.{}'.format(self.images_extention))), desc='All images'):
            self.augment_image(image_path, count_of_each, resize, width, height)

    def full_augment(self, dataset_path, each_class_count, images_extention='jpg'):
        '''
        Augment all classes in dataset_path so that you will have #each_class_count images for each class.

        Classes must be like:
        dataset_path/
            class1/
                image1.jpg
                image2.jpg
                .
                .
                .
            class2/
                image1.jpg
                image2.jpg
                .
                .
                .
            class3/
                image1.jpg
                image2.jpg
                .
                .
                .
        '''
        classes = os.listdir(dataset_path)

        for c in classes:
            images_path = os.path.join(dataset_path, c)
            num_images = len(os.listdir(images_path))
            each_image = each_class_count // num_images

            self.auto_augmentation(each_image)

if __name__ == '__main__':
    sample_image_path = 'path/to/your/images_dir'
    ClassificationAug(sample_image_path).auto_augmentation(10, resize=True, width=512, height=512)


