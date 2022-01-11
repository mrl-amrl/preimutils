import albumentations as A
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

class KPImageAug():

    file_counter=0
    def __init__(self, xml_dir, images_dir):
        self.xml_dir = xml_dir
        self.images_dir = images_dir

        assert os.path.exists(xml_dir), 'XML file not exist'
        assert os.path.exists(images_dir), 'Images path not exist'


    def augment_image(self,img:np.array, kpoints,  quantity:int, width:int=0, height:int=0):
        transform = A.Compose(
            [A.Resize(width, height, always_apply=True),
             A.VerticalFlip(p=1),
             A.RandomRotate90(p=0.5),
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

        for i in tqdm(range(quantity), desc='Singel image'):
            augmented = transform(image=img, keypoints='')

            augmented = transform()
            KPImageAug.file_counter += 1
            file_name = 'aug_image{}'.format(KPImageAug.file_counter)

    def auto_augment(self, csv_dir:str) -> None:
        df = pd.read_csv(csv_dir)
        for image_name in tqdm(df.index):




