import os

import albumentations as A
import numpy as np
from sklearn import utils
from tqdm import tqdm

from .utils import augmented_images_write



class KPImageAug():

    file_counter = 0

    def __init__(self, xml_dir: str = None, images_dir: str = None):
        """
        A wrapper class on albumentations package to work on cvat segmentation format easily

        Args:
            xml_dir: XML cvat path of Images and Annotations
            images_dir: Images Directory
        """
        self.xml_dir = xml_dir
        self.images_dir = images_dir

        assert os.path.exists(xml_dir), 'XML file not exist'
        assert os.path.exists(images_dir), 'Images path not exist'

    def augment_image(self, img: np.array, kpoints: list, quantity: int = 1, width: int = 0, height: int = 0,
                      save: bool = False, save_dir: str = None, return_filenames: bool = False):
        """
        augmentation for one picture depend on quantity that you get for it
        if your image and mask names are same
        (Optional) : save your aug image in save_path with the following pattern aug_image{counter}.jpg

        Args:
            img: (numpy array) Input Image
            kpoints: (list) List of Image Key points
            quantity: quantity for your image to augment
            width: (int) new width for augmented images (if 0 : use input image shape)
            height: (int) new height for augmented images (if 0 : use input image shape)
            save: (bool: optional) if True Augmented Images save in directory
            save_dir: (str) Directory to Save Augmented Images (only used when save=True)
            return_filenames: (bool) If True Augmented images name return

        Returns:
            (list): List of Augmented Images in np.array format
            (list): List of Augmented Key Points of each Image
            (list - if return_filenames=True) : List of Augmented images name ([aug_image1.jpg, aug_image2.jpg, ...])
        """

        if (width == 0) or (height == 0):
            width = img.shape(0)
            height = img.shape(1)

        transform = A.Compose(
            [A.Resize(width, height, always_apply=True),
             A.VerticalFlip(p=0.1),
             A.RandomRotate90(p=0.1),
             A.MotionBlur(p=0.1),
             A.MedianBlur(p=0.1),
             A.ISONoise(p=0.2),
             A.RandomBrightnessContrast(p=0.1),
             A.RandomShadow(p=0.1),
             A.RandomSnow(snow_point_lower=0.1,
                          snow_point_upper=0.15, p=0.1),
             A.RGBShift(p=0.2),
             A.CLAHE(p=0.2),

             A.HueSaturationValue(
                 p=0.1, hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10),
             ],
            keypoint_params=A.KeypointParams(format='xy')
        )

        augimages = []
        augkpoints = []
        augfilename = []

        for _ in tqdm(range(quantity)):
            augmented = transform(image=img, keypoints=kpoints)
            KPImageAug.file_counter += 1
            file_name = 'aug_image{}.jpg'.format(KPImageAug.file_counter)
            augimages.append(augmented['image'])
            augkpoints.append(augmented['keypoints'])
            augfilename.append(file_name)

        if save:
            assert os.path.exists(save_dir), 'save directory not exist'
            augmented_images_write(augimages, augkpoints, augfilename, save_dir)

        if return_filenames:
            return augimages, augkpoints, augfilename

        return augimages, augkpoints

    def auto_augmentation(self, images: list, kpoints: list, quantity_of_ech_image: int = 1, width: int = 0,
                      height: int = 0, save: bool = False, save_dir: str = None):
        """
        multi augmentation for each picture depend on quantity that you get for it
        if your image and mask names are same
        (Optional) : save your aug image in save_path with the following pattern aug_image{counter}.jpg

        Args:
            images: (list) List of Images
            kpoints: (list) List of Images Key points
            quantity_of_ech_image: quantity for your image to augment
            width: (int) new width for augmented images (if 0 : use input image shape)
            height: (int) new height for augmented images (if 0 : use input image shape)
            save: (bool: optional) if True Augmented Images save in directory
            save_dir: (str) Directory to Save Augmented Images (only used when save=True)

        Returns:
            (list): List of Augmented Images in np.array format
            (list): List of Augmented Key Points of each Image
        """
        m_aug_images = []
        m_aug_kpoints = []
        m_aug_filenames = []
        for t in range(len(images)):
            img = images[t]
            kps = kpoints[t]
            augimages, augkpoints, augfilenames = self.augment_image(img, kps, quantity_of_ech_image, width, height,
                                                                     False, '', True)
            m_aug_images = m_aug_images + augimages
            m_aug_kpoints = m_aug_kpoints + augkpoints
            m_aug_filenames = m_aug_filenames + augfilenames

        if save:
            assert os.path.exists(save_dir), 'save directory not exist'
            augmented_images_write(m_aug_images, m_aug_kpoints, m_aug_filenames, save_dir)

        return m_aug_images, m_aug_kpoints
