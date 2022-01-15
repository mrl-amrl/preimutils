import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import read_images_and_annotations
from .utils import xml_to_dict, load_images_from_folder


class DataSet(Dataset):
    def __init__(self, Images_dir: str, XML_dir: str = None, CSV_dir: str = None, data_size: int = None):
        """

        Args:
            Images_dir:
            XML_dir:
            CSV_dir:
            data_size:
        """
        self.images_dir = Images_dir
        self.xml_dir = XML_dir
        self.csv_dir = CSV_dir
        self.data_size = data_size

        assert ((os.path.exists(XML_dir)) or (os.path.exists(CSV_dir))), 'XML path or CSV path not exist!'

        if XML_dir:
            images, self.annotations, _ = self.load_raw_data()
        else:
            images, self.annotations = read_images_and_annotations(self.images_dir, self.csv_dir)

        images = torch.FloatTensor(images)
        self.images = images  # (images - images.mean()) / images.std()

    def load_raw_data(self):
        ann_dict = xml_to_dict(self.xml_dir)

        keypoint_names = [ann_dict['annotations']['meta']['task']['labels']['label'][i]['name'] for i in
                          range(len(ann_dict['annotations']['meta']['task']['labels']['label']))]

        images_name = [ann_dict['annotations']['image'][i]['@name'] for i in
                       range(len(ann_dict['annotations']['image']))]

        if self.data_size:
            data_length = self.data_size
            images_name = images_name[:self.data_size]
        else:
            data_length = len(ann_dict['annotations']['image'])

        images = load_images_from_folder(self.images_dir, images_name)

        keypoints = []
        for i in range(data_length):
            kp = []
            for p in range(len(ann_dict['annotations']['image'][i]['points'])):
                xy = ann_dict['annotations']['image'][i]['points'][p]['@points']
                xy = xy.split(',')
                x = float(xy[0])
                y = float(xy[1])
                kp.append((x, y))
            keypoints.append(kp)

        images = np.array(images)
        return images, keypoints, keypoint_names

    def __len__(self):
        return int(len(self.images))

    def __getitem__(self, item):
        img = self.images[item] / 255.0
        img = np.transpose(img, (2, 0, 1))
        # img_tensor = torch.tensor(img, requires_grad=True)
        kps = self.annotations[item]
        kps = [list(kp) for kp in kps]
        for k in kps:
            k.append(1000.0)
        kp = torch.tensor(kps, dtype=torch.float, requires_grad=True)

        img.type(torch.FloatTensor)
        kp.type(torch.FloatTensor)
        return img, kp

# ['waistband_bottom_right', 'waistband_bottom_left', 'waistband_center', 'j_top_left', 'j_bottom', 'j_center_left', 'j_center_right', 'coinpkt_top_right', 'coinpkt_top_left', 'yoke_right', 'yoke_left', 'right_frontpkt_top_right', 'right_frontpkt_bottom_right', 'right_frontpkt_top_left', 'left_frontpkt_top_left', 'left_frontpkt_bottom_left', 'left_frontpkt_top_right', 'right_leghemming_right', 'right_leghemming_left', 'left_leghemming_left', 'left_leghemming_right', 'crotch', 'j_top_right', 'waistband_top_left', 'waistband_top_right', 'right_beltloop', 'left_beltloop', 'right_bartack', 'left_bartack', 'button_right', 'button_left', 'button_center', 'waistband_top_center']
# 'C:\\Users\\Seyed\\Desktop\\preimutils\\preimutils\\keypoint_detection\\cvat\\data\\images'
# 'C:\\Users\\Seyed\\Desktop\\preimutils\\preimutils\\keypoint_detection\\data\\annotations.xml'
