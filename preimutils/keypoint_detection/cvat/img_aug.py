import os

import albumentations as A
import numpy as np
from tqdm import tqdm

from utils import augmented_images_write


class KPImageAug():
    file_counter = 0

    def __init__(self, xml_dir: str = None, images_dir: str = None):
        self.xml_dir = xml_dir
        self.images_dir = images_dir

        # assert os.path.exists(xml_dir), 'XML file not exist'
        # assert os.path.exists(images_dir), 'Images path not exist'

    def augment_image(self, img: np.array, kpoints: list, quantity: int = 1, width: int = 0, height: int = 0,
                      save: bool = False, save_dir: str = None, return_filenames: bool = False):

        if (width == 0) or (height == 0):
            width = img.shape(0)
            height = img.shape(1)

        transform = A.Compose(
            [A.Resize(width, height, always_apply=True),
             A.VerticalFlip(p=1),
             A.RandomRotate90(p=0.2),
             A.MotionBlur(p=0.1),
             A.MedianBlur(p=0.2),
             A.ISONoise(p=0.2),
             A.IAAPerspective(p=0.1),
             A.IAAPiecewiseAffine(p=0.1, scale=(0.01, 0.02)),
             A.IAAEmboss(p=0.2),
             A.RandomBrightnessContrast(p=0.2),
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

    def multi_augment(self, images: list, kpoints: list, quantity_of_ech_image: int = 1, width: int = 0,
                      height: int = 0, save: bool = False, save_dir: str = None):
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
