import os

import numpy as np

from .utils import read_images_and_annotations
from .utils import xml_to_dict, load_images_from_folder


class Dataset:
    def __init__(self, Images_dir: str, XML_dir: str = None, CSV_dir: str = None, data_size: int = None):
        """
        Args:
            Images_dir: Images directory paths.
            XML_dir: XML cvat path of Images and Annotations
            CSV_dir: CSV path of Images and Annotations
            data_size: Size Dataset to read (if None, read all data)
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
            images, self.annotations = images[:data_size], self.annotations[:data_size]

        self.images = images

    def load_raw_data(self):
        """
        Returns:
            List of Images in numpy array format.
            List of key points for each image.
            List Label of Key points
        """
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
        """
        Returns:
                Length of Data
        """
        return len(self.images)

    def __getitem__(self, item):
        img = self.images[item]
        kps = self.annotations[item]
        kps = [list(kp) for kp in kps]
        return img, kps
