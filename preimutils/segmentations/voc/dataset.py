import os
import random
from glob import glob

from tqdm import tqdm

from . import utils


class LabelMap:
    """A class on to work and handel labelmap.txt classes

    Attributes:

        None

    Args:

        label_map_path (:obj:`str`): you should have a txt file like this

            object1:0,0,0::
            object2:128,0,0::
            object3:0,128,0::
            objectN:128,128,0::

        masks_dir (str): annotations files paths.
        images_dir (str) :images files paths.
    """

    def __init__(self, label_map_path):

        with open(label_map_path) as f:
            lines = f.readlines()
        self.lines = lines[1:]

    def label_color(self):
        """Export label_color dict 

        Args:

            None

        Returns:

            a dict of label:color
            {label1:(r,g,b),labelN:(r,g,b)}

        """
        label_map = {}
        for line in self.lines:
            line = line.split(':')
            color = tuple(line[1].split(','))
            color = (int(color[0]), int(color[1]), int(color[2]))
            label_map[line[0]] = color
        return label_map

    def color_label(self):
        """Export color_label dict 

        Args:

            None

        Returns:

            a dict of label:color 
            {(r,g,b):label1,(r,g,b):labelN}
        """
        label_map = {}
        for line in self.lines:
            line = line.split(':')
            color = tuple(line[1].split(','))
            color = (int(color[0]), int(color[1]), int(color[2]))
            label_map[line[0]] = color
            new_dic = {y: x for x, y in label_map.items()}
        return new_dic

    def color_list(self):
        """Export color_list in their saving order 

        Args:

            None

        Returns:

            list of colors
            {(r,g,b),(rN,gN,bN)}
        """
        colors = []
        for line in self.lines:
            line = line.split(':')
            color = tuple(line[1].split(','))
            color = (int(color[0]), int(color[1]), int(color[2]))
            colors.append(color)
        return colors


class Dataset:
    """Get dataset voc paths directly and some methods for work with it.

        ├── ImageSets
        │   └── Segmentation
        ├── JPEGImages
        ├── SegmentationClass
        ├── SegmentationClassRaw
        └── SegmentationObject 

    Attributes:

        masks_dir: SegmentationClass directory path 
        images_dir: JPEGImages directory paths.
        segmentations_object_dir : SegmentationObject directory path
        label_map_path = labelmap.txt path

    """

    def __init__(self, dataset_dir, images_extention='jpg'):
        self.__dataset_dir_model = """
├── ImageSets
│   └── Segmentation
├── JPEGImages
├── SegmentationClass
├── SegmentationClassRaw
└── SegmentationObject
                                """
        assert os.path.exists(
            dataset_dir), 'dataset directory not exist in {}'.format(dataset_dir)
        assert os.path.exists(os.path.join(dataset_dir, 'SegmentationClass')
                              ), 'dataset SegmentationClass directory not exist dataset should be in this tree {}'.format(self.__dataset_dir_model)
        assert os.path.exists(os.path.join(dataset_dir, 'JPEGImages')
                              ), 'dataset JPEGImages directory not exist in {}'.format(dataset_dir)
        assert os.path.exists(os.path.join(
            dataset_dir, 'SegmentationObject')), 'dataset SegmentationObject directory not exist dataset should be in this tree {}'.format(self.__dataset_dir_model)
        assert os.path.exists(os.path.join(dataset_dir, 'labelmap.txt')
                              ), 'dataset labelmap.txt not exist dataset should be in this tree {}'.format(self.__dataset_dir_model)
        # dataset successfully loaded
        self._dataset_dir = dataset_dir
        self.masks_dir = os.path.join(dataset_dir, 'SegmentationClass')
        self.images_dir = os.path.join(dataset_dir, 'JPEGImages')
        self.segmentations_object_dir = os.path.join(
            dataset_dir, 'SegmentationObject')
        self.label_map_path = os.path.join(dataset_dir, 'labelmap.txt')
        self.images_extention = images_extention

    def check_valid_dataset(self):
        """Check for all masks images if there isn't related 
            mask image print the work image path
            If image not exist raise ValueError

            Args:

                None

            ValueError : 

                if image that you want not exists

            Returns:

                None
        """
        for mask in glob(os.path.join(self.masks_dir, '*.png')):
            utils.find_image_from_mask(
                mask, self.images_dir, extention=self.images_extention)

    def seprate_dataset(self, shuffle=False, valid_persent=0.25, test_persent=None, save=True):
        """Seprate dataset to train.txt,trainval.txt,val.txt 

            Args:

                valid_persentage:(float), should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the validation split.
                 If None,it will be set to 0.25.
                test_persentage: (float), should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
                 If None, that means you don't have test dataset just have train validation.
                save :(bool), If True dataset save train.txt, trainval.txt, val.txt , if test_present exist test.txt

            Returns:

                None
        """
        total_masks_path = glob(os.path.join(self.masks_dir, '*.png'))
        total_masks_path = map(
            lambda x: os.path.basename(x[:-4]), total_masks_path)
        total_masks_path = list(total_masks_path)

        if shuffle:
            random.shuffle(total_masks_path)

        if test_persent:
            valid_len = round(len(total_masks_path) * valid_persent)
            test_len = round(len(total_masks_path) * test_persent)
            train_len = len(total_masks_path) - test_len - valid_len
            train_ds = total_masks_path[:train_len]
            remain_mask = total_masks_path[train_len:]
            test_ds = remain_mask[:test_len]
            valid_ds = remain_mask[test_len:]

        else:
            valid_len = round(len(total_masks_path) * valid_persent)
            valid_ds = total_masks_path[:valid_len]
            train_ds = total_masks_path[valid_len:]
            test_ds = None

        if save:
            with open(os.path.join(self._dataset_dir, 'ImageSets', 'Segmentation', 'trainval.txt'), 'w') as f:
                f.write('\n'.join(total_masks_path))
            with open(os.path.join(self._dataset_dir, 'ImageSets', 'Segmentation', 'train.txt'), 'w') as f:
                f.write('\n'.join(train_ds))
            with open(os.path.join(self._dataset_dir, 'ImageSets', 'Segmentation', 'val.txt'), 'w') as f:
                f.write('\n'.join(valid_ds))
            if test_persent:
                with open(os.path.join(self._dataset_dir, 'ImageSets', 'Segmentation', 'test.txt'), 'w') as f:
                    f.write('\n'.join(test_ds))

        return train_ds, valid_ds, test_ds
